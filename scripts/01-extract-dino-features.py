import httpx
import aiosqlite
import sqlite_vec
import numpy as np
import cv2
import asyncio
from dataclasses import dataclass, field
from typing import Any
import os
from glob import glob
from tqdm import tqdm
from time import perf_counter
from aiofiles import open as aopen

URLS_TO_USE = [
    "http://127.0.0.1:8000/predict",
    "http://127.0.0.1:8081/predict",
    "http://127.0.0.1:8082/predict",
    "http://127.0.0.1:8083/predict"
]
N_WORKERS = len(URLS_TO_USE)

create_table_query = """
CREATE VIRTUAL TABLE IF NOT EXISTS dino_embeddings USING vec0(
    id TEXT PRIMARY KEY,
    cluster_id INTEGER,
    embeddings float[384]
)
"""

find_by_id_query = """
SELECT cluster_id FROM dino_embeddings WHERE id = :imageid 
"""

insert_embeddings_query = """
INSERT INTO dino_embeddings VALUES(:id, :cluster_id, :embeddings)
"""

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


async def readerworker(worker_id: int, filename_queue: asyncio.Queue, timing_queue: asyncio.Queue, dbwrite_queue: asyncio.Queue):
    url_to_use = URLS_TO_USE[worker_id%4]
    print(f"Worker {worker_id} Initialized.")
    async with httpx.AsyncClient() as client:
        while True:
            data = await filename_queue.get()
            data = data.item
            if data is None:
                filename_queue.task_done()
                if timing_queue is not None:
                    timing_queue.task_done()
                print(f"Worker ID {worker_id} done. Exitting..")
                return
            print(f"Worker ID {worker_id} processing data {data[0]}")
            t1 = perf_counter()
            async with aopen(data[1], "rb") as f:
                binary_content = await f.read()
            try:
                response = await client.post(url_to_use, content=binary_content, headers={"content-type": 'image/png'})
            except Exception as e:
                filename_queue.task_done()
                print(f"Worker ID {worker_id} got exception during processing data {data[0]}: {type(e)} {e} with url {url_to_use}")
                await filename_queue.put(PrioritizedItem(1, data))
                continue
            t2 = perf_counter()
            print(f"Worker ID {worker_id} finish processing data {data[0]} in {t2-t1}")
            imageid = os.path.basename(data[1])[:-4]
            embeddings = response.json()["embeddings"]
            await dbwrite_queue.put((imageid, embeddings))
            if timing_queue is not None:
                await timing_queue.put((t2-t1))
            filename_queue.task_done() 

async def dbwriter(connection: aiosqlite.Connection, dbwrite_queue: asyncio.Queue):
    print("DBWriter initializing..")
    cursor = await connection.cursor()
    print("DBWriter is listening....")
    while True:
        to_write = await dbwrite_queue.get()
        if to_write is None:
            print("Done with Everything. DBWrite exitting..")
            dbwrite_queue.task_done()
            await cursor.close()
            break
        image_id, vector_data = to_write
        print(f"Received embedding of {image_id}")
        vector_data = np.array(vector_data, dtype=np.float32).reshape(384)
        await cursor.execute(insert_embeddings_query, {"id": image_id, "cluster_id": -1, "embeddings": vector_data})
        await connection.commit()
        print(f"Embedding for {image_id} written to DB.")
        dbwrite_queue.task_done()
        

async def main(file_paths: list[str], db_write_path: str):
    print("Init DB and stuff")
    db = await aiosqlite.connect(db_write_path)
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)
    cursor = await db.cursor()
    await cursor.execute(create_table_query)
    processed_ids = await (await cursor.execute("SELECT id FROM dino_embeddings")).fetchall()
    print("Preparing data")
    processed_ids = set([ai[0] for ai in processed_ids])
    paths_to_process = []
    for fp in file_paths:
        fpid = os.path.basename(fp)[:-4]
        if fpid in processed_ids:
            continue
        paths_to_process.append(fp)
    if not len(paths_to_process):
        print("No more paths need to be processed. Exitting..")
        await cursor.close()
        await db.close()
        return
    # init queues
    unprocessed_queue = asyncio.PriorityQueue()
    timings_queue = asyncio.Queue()
    db_write_queue = asyncio.Queue()
    
    for i, path in enumerate(paths_to_process):
        await unprocessed_queue.put(PrioritizedItem(3, (i, path)))
    for _ in range(N_WORKERS):
        await unprocessed_queue.put(PrioritizedItem(5, None))
    print("Running worker processes")
    # init workers
    t1 = perf_counter()
    db_writers = [asyncio.create_task(dbwriter(db, db_write_queue))]
    processors = [asyncio.create_task(readerworker(i, unprocessed_queue, timings_queue, db_write_queue)) for i in range(N_WORKERS)]
    await unprocessed_queue.join()
    await asyncio.gather(*processors)
    await db_write_queue.put(None)
    await db_write_queue.join()
    await asyncio.gather(*db_writers)
    await db.commit()
    t2 = perf_counter()
    print("Done. Calculating stats..")
    total_duration = t2-t1
    throughput = len(paths_to_process)/total_duration
    timings = []
    for _ in range(len(paths_to_process)):
        timing = await timings_queue.get()
        timings.append(timing)
    timings = np.array(timings)
    print(f"Total duration: {total_duration}")
    print(f"Throughput: {throughput}")
    print(f"AVG: {np.mean(timings)}")
    print(f"Median: {np.median(timings)}")
    print(f"P95: {np.percentile(timings, 95)}")
    print(f"Range: {np.min(timings)} - {np.max(timings)}")
    await cursor.close()
    await db.close()

if __name__ == "__main__":
    all_data_path = glob("facemask_dataset/images/*.png")
    db_write_path = "dino_features.db"
    asyncio.run(main(all_data_path, db_write_path))
