import os
from pathlib import Path
import uuid
from diffusers.utils import load_image
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

import torch
import cv2
from PIL import Image

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"


def register():
    sd3 = FlareSD3()
    return {"flare-sd3": {"text-to-image": sd3.text_to_image}}


class FlareSD3:
    t5_encoder: T5EncoderModel = None
    pipeline_t2i: StableDiffusion3Pipeline = None

    def __init__(self):
        pass

    def load_sd3(self):
        token = Path("hftoken").read_text()

        if self.t5_encoder is None:
            self.t5_encoder = T5EncoderModel.from_pretrained(
                model_id,
                cache_dir="weights",
                resume_download=True,
                subfolder="text_encoder_3",
                quantization_config=quantization_config,
                token=token,
            )

        if self.pipeline_t2i is None:
            self.pipeline_t2i = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                text_encoder_3=self.t5_encoder,
                device_map="balanced",
                cache_dir="weights",
                resume_download=True,
                torch_dtype=torch.float16,
                token=token,
            )

    def text_to_image(self, params, response):
        self.load_sd3()

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
        num_inference_steps = 28
        guidance_scale = 7.0

        image = self.pipeline_t2i(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        ).images[0]

        save_image = params["tools"].save_image
        image_name = save_image(image)
        response["image"] = image_name
        self.offload_if_required(params)

    def offload_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading sd3")

            params["tools"].unload_model(self.pipeline_t2i)
            self.pipeline_t2i = None
