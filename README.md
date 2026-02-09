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