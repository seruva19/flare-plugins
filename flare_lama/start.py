import os
from simple_lama_inpainting import SimpleLama
from PIL import Image
from pathlib import Path

root = Path(__file__).parents[2]
weights_dir = os.path.join(root, "weights")


def register():
    flare_lama = FlareLama()
    return {"flare-lama": {"remove-object": flare_lama.remove_object}}


class FlareLama:
    lama: SimpleLama = None

    def __init__(self):
        os.environ["TORCH_HOME"] = weights_dir

    def remove_object(self, params, response):
        if self.lama is None:
            self.lama = SimpleLama()

        init_image = params["input_image"]
        mask_image = params["mask_image"]

        image_path = os.path.join("build/images", init_image)
        input_image = Image.open(image_path)

        mask_path = os.path.join("build/images", mask_image)
        input_mask = Image.open(mask_path).convert("L")

        result = self.lama(input_image, input_mask)

        save_image = params["tools"].save_image
        image_name = save_image(result)

        response["image"] = image_name

        if params["settings"]["offload_models"]:
            print("flare: offloading lama")
            params["tools"].unload_model(self.lama.model)
            self.lama = None
