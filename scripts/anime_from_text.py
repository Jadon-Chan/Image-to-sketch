import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
)
import os

home = os.environ["HOME"]
model_path = f"{home}/img2img/models"
vae_path = f"{model_path}/sdxl-vae-fp16-fix"
model_path = f"{model_path}/animagine-xl-2.0"
lora_path = f"{model_path}/sketch-style-xl-lora"
lora_name = "sketch-style-xl.safetensors"

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    vae_path, 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path, 
    vae=vae,
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

pipe.load_lora_weights(lora_path, weight_name=lora_name)
pipe.fuse_lora(lora_scale=0.6)

# Define prompts and generate image
# prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
# negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

prompt = "face focus, cute, masterpiece, best quality, 1girl, sketch, monochrome, greyscale, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=1024,
    height=1024,
    guidance_scale=12,
    num_inference_steps=50
).images[0]

image.save(f"{home}/img2img/anime_sketch.png")