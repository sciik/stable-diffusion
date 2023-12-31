# Stable Diffusion
## This is for understanding the Stable Diffusion. I built the stable diffusion following this [video](https://www.youtube.com/watch?v=ZBKpAp_6TGI). The stable diffusion is structured into 4 parts, which are Auto-Encoder, CLIP, latent space, and scheduler. All of these parts are implemented for PyTorch and are located in the 'sd' directory.

## You can execute the code for 'inference.py' in the 'sd' directory and test the model in two ways: image-to-image and text-to-image. If you want to test image-to-image, simply write the image path in the 'input_image' variable for the desired condition. If you want to test text-to-image, leave it as it is.

## Unfortunately, there is no checkpoint file due to a lack of memory in my local environment. Therefore, before you execute 'inference.py', you should download the ckpt file [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main). Any version of ckpt is possible.

