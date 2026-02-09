# Technical Test for Verihubs

This repo contains my submission for the technical test.

I consider this unfinished and messy. I cannot make time to tidy things up and separate training into its own script, sorry.

Inside notebooks, you can find 2 notebooks:
1. EDA_and_train.ipynb. I write all my EDA analysis and also training code there.
2. Export_and_analysis.ipynb. I write all my analysis after training here.

Training took ~30 mins on kaggle notebook.

Inside scripts, you can find some scripts related to data viz / exploration, and an inference script. 

Everything should run locally, as long as you follow the setup guide.

Keep in mind the extract dino features script requires a separate service to serve the dinov3 model. I also included the code (it is a very simple expressjs server) just for FYI. I have done exploration using fiftyone, but i didnt write about it. You may find the markdown describing my exploration and findings to be submitted after deadline, i will just include it for completeness. Really sorry, bad timing on my part.

There is no deeper reason behind the separation besides my GPU can only run Vulkan, and the inference runtime im using only supports vulkan on their js library through webgpu. It is a hacky way to at least make use of my hardware for accelerated inference.

## Setup guide

Ensure you have installed UV. There is no reason to use just pip in 2025. 

Then you just need to run 
```sh
uv sync
```

And it will install everything required. Probably not needed if run on colab.

For the data, just download the data and put the `images` and `annotations` folder under `facemask_dataset`. You may or may not need to adjust the path to each folders in the code.

## Analysis details and reasoning

I write everything on the notebook, so you can see the explanation side by side with the code.

## Artifacts

Artifacts like the trained model, train-val-test splits, some training results, and onnx-converted models can be accessed in [this google drive link](https://drive.google.com/drive/folders/1mnncJ8cVTahDV3JYRX75JvM7XZd-Euul). I have made sure the permission is open, please contact me if it's not.

## Note on the fiftyone scripts

Under `scripts` you may find 2 numbered scripts besides the inference script. It is used to visually inspect the dataset using fiftyone. I used it to see how visually distributed the images are. The first script calls a bunch of dinov3 servers to extract all image features, deployed using expressjs because the only way my computer can run models in GPU is by using onnxruntime-nodejs' webgpu api. The second script essentially did some PCA on the image features to see if i can find a good way to segment the images visually, to complement the object count based splitting.

As you can tell, that didnt go too far. I did detail the exploration on [this report](./reports/visual-exploration.md).

## Known weakness and next steps

One thing is obvious. The lack of direction makes any kind of curation meaningless. What we need to test against is 100% decided by what kind of scenario do we see this model is going to be deployed for. Splitting based on statistics is basically cheating. We already know the distribution beforehand, and having train, val, and test set being split based on this is hacky at best and cheating at worst.

After knowing what I will need to aim for and iterate against, I will definitely try to address the class imbalance problem. Easiest to try to do is to use class weighting on the loss function. Combined with clever augmentations, it's highly possible to alleviate the imbalance a bit, especially now that we can make synthetic data using generative models. 

We can also change the approach altogether, by treating every kind of face as one class, then classifying which category does it belong to afterwards, ideally using something like metric learning.

All of these possible solutions still needs to be compared, which brought me back to my original point which is the entire point of developing the model. What is the exact use case in mind? 