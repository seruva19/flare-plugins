import os
from pathlib import Path
import uuid
from diffusers.utils import load_image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForInpainting,
)
import torch
import cv2
from PIL import Image


def register():
    sdxl = FlareSDXL()
    return {
        "flare-sdxl": {
            "text-to-image": sdxl.text_to_image,
            "inpaint": sdxl.inpaint,
            "remove-object": sdxl.remove_object,
        }
    }


default_checkpoint_url = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"


class FlareSDXL:
    pipeline_t2i: AutoPipelineForText2Image = None
    pipeline_inpaint: AutoPipelineForInpainting = None

    def __init__(self):
        pass

    def load_sdxl(self):
        if self.pipeline_t2i is None:
            if self.pipeline_inpaint is None:
                self.pipeline_t2i = StableDiffusionXLPipeline.from_single_file(
                    default_checkpoint_url,
                    torch_dtype=torch.float16,
                    cache_dir="weights",
                    resume_download=True,
                ).to("cuda")
            else:
                self.pipeline_t2i = AutoPipelineForText2Image.from_pipe(
                    self.pipeline_inpaint
                ).to("cuda")

    def load_sdxl_inpaint(self):
        if self.pipeline_inpaint is None:
            if self.pipeline_t2i is None:
                self.load_sdxl()
            self.pipeline_inpaint = AutoPipelineForInpainting.from_pipe(
                self.pipeline_t2i
            ).to("cuda")

    def text_to_image(self, params, response):
        self.load_sdxl()

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
        num_inference_steps = 50
        guidance_scale = 5.0

        image = self.pipeline_t2i(
            prompt=prompt,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        save_image = params["tools"].save_image
        image_name = save_image(image)
        response["image"] = image_name
        self.offload_if_required(params)

    def inpaint(self, params, response):
        flare_task = params["task"]
        params["inpaint_prompt"] = flare_task.target
        params["negative_prompt"] = ""
        return self._inpaint(params, response)

    def remove_object(self, params, response):
        flare_task = params["task"]
        params["inpaint_prompt"] = ""
        params["negative_prompt"] = flare_task.source
        return self._inpaint(params, response)

    def _inpaint(self, params, response):
        self.load_sdxl_inpaint()

        prompt = params["inpaint_prompt"]
        negative_prompt = params["negative_prompt"]

        init_image = params["input_image"]
        mask_image = params["mask_image"]

        image_path = os.path.join("build/images", init_image)
        input_image = Image.open(image_path)

        mask_path = os.path.join("build/images", mask_image)
        input_mask = Image.open(mask_path)

        width, height = input_image.size
        strength = 0.85
        num_inference_steps = 50
        guidance_scale = 12.5

        image = self.pipeline_inpaint(
            prompt=prompt,
            width=width,
            height=height,
            image=input_image,
            mask_image=input_mask,
            strength=strength,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        save_image = params["tools"].save_image
        image_name = save_image(image)
        response["image"] = image_name
        self.offload_if_required(params)

    def offload_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading sdxl")
            params["tools"].unload_model(self.pipeline_t2i)
            params["tools"].unload_model(self.pipeline_inpaint)
            self.pipeline_t2i = None
            self.pipeline_inpaint = None
