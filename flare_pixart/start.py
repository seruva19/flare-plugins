import os
from pathlib import Path
import uuid
from diffusers.utils import load_image
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

import torch
import cv2
from PIL import Image


def register():
    pixart = FlarePixArt()
    return {"flare-pixart": {"text-to-image": pixart.text_to_image}}


class FlarePixArt:
    transformer: Transformer2DModel = None
    pipe_t2i: PixArtSigmaPipeline = None

    def __init__(self):
        pass

    def load_pixart(self):
        if self.transformer is None:
            self.transformer = Transformer2DModel.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                cache_dir="weights",
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )

        if self.pipe_t2i is None:
            self.pipe_t2i = PixArtSigmaPipeline.from_pretrained(
                "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
                cache_dir="weights",
                transformer=self.transformer,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            )
            self.pipe_t2i.to("cuda")

    def text_to_image(self, params, response):
        self.load_pixart()

        flare_task = params["task"]

        if flare_task.size is None:
            width, height = map(int, flare_task.size.split(","))
        elif params.get("input_image") != "":
            source_image_path = os.path.join("build/images", params["input_image"])
            source_image = Image.open(source_image_path)
            width, height = source_image.size
        else:
            width, height = 1024, 1024

        prompt = response["outPrompt"]
        negative_prompt = ""
        num_inference_steps = 20
        guidance_scale = 4.5

        image = self.pipe_t2i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            clean_caption=False,
            num_images_per_prompt=1,
        ).images[0]

        save_image = params["tools"].save_image
        image_name = save_image(image)
        response["image"] = image_name
        self.offload_if_required(params)

    def offload_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading pixart")

            params["tools"].unload_model(self.transformer)
            self.transformer = None
            params["tools"].unload_model(self.pipe_t2i)
            self.pipe_t2i = None
