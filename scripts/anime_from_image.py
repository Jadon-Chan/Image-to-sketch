import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    EulerAncestralDiscreteScheduler,
    AutoencoderKL,
    AutoPipelineForImage2Image
)
from diffusers.utils import load_image
import os
import argparse

def parse_args():
    home = os.environ.get("HOME", "")

    parser = argparse.ArgumentParser(description="Image-to-Image processing script with adjustable parameters.")

    parser.add_argument('--model_path', type=str, default=f"{home}/img2img/models", help='Path to the model directory')
    parser.add_argument('--vae_path', type=str, default=f"{home}/img2img/models/sdxl-vae-fp16-fix", help='Path to the VAE model')
    parser.add_argument('--main_path', type=str, default=f"{home}/img2img/models/animagine-xl-2.0", help='Path to the main model')
    parser.add_argument('--lora_path', type=str, default=f"{home}/img2img/models/sketch-style-xl-lora", help='Path to the LoRA model')
    parser.add_argument('--lora_name', type=str, default="sketch-style-xl.safetensors", help='Name of the LoRA model file')
    parser.add_argument('--results_path', type=str, default=f"{home}/img2img/results", help='Path to save the results')
    parser.add_argument('--init_image_path', type=str, default=f"{home}/img2img/inits", help='Path to the initial image directory')
    parser.add_argument('--init_image_name', type=str, default="river_town.jpg", help='Name of the initial image file')
    parser.add_argument('--strength', type=float, default=0.5, help='Strength parameter for image processing')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load VAE component
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, 
        torch_dtype=torch.float16
    )

    # Configure the pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.main_path, 
        vae=vae,
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')

    pipe.load_lora_weights(args.lora_path, weight_name=args.lora_name)
    pipe.fuse_lora(lora_scale=0.6)

    # Define prompts and generate image
    # prompt = "face focus, cute, masterpiece, best quality, 1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
    # negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

    # prompt = "face focus, cute, masterpiece, best quality, 1girl, sketch, monochrome, greyscale, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck"
    # negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

    init_image = load_image(f"{args.init_image_path}/{args.init_image_name}")
 
    prompt = "masterpiece, best quality, sketch, monochrome, greyscale"
    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

    image = pipe(
        prompt, 
        image=init_image,
        negative_prompt=negative_prompt, 
        width=1024,
        height=1024,
        guidance_scale=12,
        num_inference_steps=50,
        strength=args.strength
    ).images[0]

    image.save(f"{args.results_path}/{os.path.splitext(args.init_image_name)[0]}.png")
