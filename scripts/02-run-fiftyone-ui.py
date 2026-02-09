import numpy as np
import os
import sqlite3

import fiftyone as fo
import fiftyone.brain as fob
import sqlite_vec

from sklearn.decomposition import MiniBatchSparsePCA

DINO_EMBEDDINGS_DIM = 384
DINO_EMBEDDINGS_PCA_DIM = 16

if DINO_EMBEDDINGS_PCA_DIM > 50:
    print("ALERT: the number of dimensions in the PCA embedding vector is pretty high")


db = sqlite3.connect("dino_features.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

select_query = """\
SELECT id, embeddings FROM "dino_embeddings"
"""

select_query_pca = f"""\
SELECT id, embeddings FROM "dino_embeddings_sparsepca_{DINO_EMBEDDINGS_PCA_DIM}"
"""

create_table_query = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS dino_embeddings_sparsepca_{DINO_EMBEDDINGS_PCA_DIM} USING vec0(
    id TEXT PRIMARY KEY,
    embeddings float[{DINO_EMBEDDINGS_PCA_DIM}]
)
"""

insert_embeddings_query = f"""
INSERT INTO dino_embeddings_sparsepca_{DINO_EMBEDDINGS_PCA_DIM} VALUES(:id, :embeddings)
"""

get_number_of_diff_query = f"""
WITH all_ids AS (
    SELECT
        id
    FROM dino_embeddings
), all_pca_ids AS (
    SELECT
        id
    FROM dino_embeddings_sparsepca_{DINO_EMBEDDINGS_PCA_DIM}
)
SELECT
    COUNT(*)
FROM
    (SELECT * FROM all_ids EXCEPT SELECT * FROM all_pca_ids)
"""

db.execute(create_table_query)

num_diffs = db.execute(get_number_of_diff_query).fetchone()[0]

image_path = "facemask_dataset/images"




if num_diffs > 0:
    print("Found diffs, recalculating PCA")
    id_and_embeddings = db.execute(select_query).fetchall()

    all_images, all_embeddings = zip(*id_and_embeddings)

    all_images_path = [
        os.path.join(image_path, f"{img_id}.png") for img_id in all_images
    ]
    all_embeddings = np.stack([
        np.frombuffer(buffer, dtype=np.float32) for buffer in all_embeddings
    ])

    ipca = MiniBatchSparsePCA(DINO_EMBEDDINGS_PCA_DIM, batch_size=4)

    all_embeddings_reduced = ipca.fit_transform(all_embeddings)


    for imageid, embedding_reduced in zip(all_images, all_embeddings_reduced):
        db.execute(insert_embeddings_query, {"id": imageid, "embeddings": embedding_reduced.copy()})
    db.commit()
else:
    print("Using precalculated PCA...")
    id_and_embeddings = db.execute(select_query_pca).fetchall()
    all_images, all_embeddings = zip(*id_and_embeddings)

    all_images_path = [
        os.path.join(image_path, f"{img_id}.png") for img_id in all_images
    ]
    all_embeddings_reduced = np.stack([
        np.frombuffer(buffer, dtype=np.float32) for buffer in all_embeddings
    ])

dataset = fo.Dataset(f"facemask-images-{DINO_EMBEDDINGS_PCA_DIM}-pca", overwrite=True)
dataset.add_images(all_images_path)
if num_diffs > 0:
    viz = fob.compute_visualization(
        dataset,
        embeddings=all_embeddings,
        method="umap",
        create_index=True,
        brain_key="dinov3_umap"
    )
    dataset.set_values("dinov3_umap_points", viz.current_points.tolist())
    dataset.set_values("dinov3_embeddings_raw", all_embeddings.copy())
else:
    id_and_embeddings = db.execute(select_query).fetchall()
    _, all_embeddings = zip(*id_and_embeddings)
    all_embeddings = np.stack([
        np.frombuffer(buffer, dtype=np.float32) for buffer in all_embeddings
    ])
    viz = fob.compute_visualization(
        dataset,
        embeddings=all_embeddings,
        method="umap",
        create_index=True,
        brain_key="dinov3_umap"
    )
    dataset.set_values("dinov3_umap_points", viz.current_points.tolist())
    dataset.set_values("dinov3_embeddings_raw", all_embeddings.copy())


for i in range(DINO_EMBEDDINGS_PCA_DIM):
    points = np.zeros((len(all_embeddings_reduced), 2), dtype=all_embeddings_reduced.dtype)
    dim_embeddings = all_embeddings_reduced[:, i]
    dim_embeddings = dim_embeddings - dim_embeddings.mean()
    dim_embeddings /= np.std(dim_embeddings)
    points[:, 0] = dim_embeddings
    
    viz = fob.compute_visualization(
        dataset,
        embeddings=all_embeddings_reduced,
        method="manual",
        points=points,
        create_index=True,
        brain_key=f"dinov3_embeddings_{i}"
    )

    dataset.set_values(f"dinov3_pca_{i}", viz.current_points)

session = fo.launch_app(dataset)

session.wait()
