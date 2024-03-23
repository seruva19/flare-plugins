import os
import uuid
import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from urllib.request import urlretrieve
from pathlib import Path
from supervision import Color


def register():
    sam_dino = FlareSAM_DINO()
    return {
        "flare-sam_dino": {
            "extract-bbox-mask": sam_dino.retrieve_bbox_mask_by_prompt,
            "extract-mask": sam_dino.retrieve_mask_by_prompt,
        }
    }


root = Path(__file__).parents[2]
weights_dir = os.path.join(root, "weights")

dino_weights_file = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
dino_checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
dino_config_file = os.path.join(os.path.dirname(__file__), "dino_config.py")

sam_weights_file = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
sam_checkpoint_url = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
)
sam_encoder_version = "vit_h"


class FlareSAM_DINO:
    grounding_dino = None
    segment_anything = None

    def __init__(self):
        pass

    def load_checkpoints(self):
        os.makedirs(weights_dir, exist_ok=True)

        if not os.path.exists(dino_weights_file):
            print("flare: downloading Grounding Dino weights...")
            urlretrieve(dino_checkpoint_url, dino_weights_file)

        if not os.path.exists(sam_weights_file):
            print("flare: downloading Segment Anything weights...")
            urlretrieve(sam_checkpoint_url, sam_weights_file)

    def load_dino(self, device):
        if self.grounding_dino is None:
            self.grounding_dino = Model(
                model_config_path=dino_config_file,
                model_checkpoint_path=dino_weights_file,
                device=device,
            )

    def load_sam(self, device):
        if self.segment_anything is None:
            self.segment_anything = sam_model_registry[sam_encoder_version](
                checkpoint=sam_weights_file
            )
            self.segment_anything.to(device=device)

    def retrieve_bbox_mask_by_prompt(self, params, response):
        self.load_checkpoints()

        flare_task = params["task"]
        prompt = flare_task.source

        source_image = params["input_image"]
        source_image_path = os.path.join("build/images", source_image)
        input_image = cv2.imread(source_image_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_dino(device)

        classes = [prompt]
        box_threshold = 0.25
        text_threshold = 0.25

        detections = self.grounding_dino.predict_with_classes(
            image=input_image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        height, width, _ = input_image.shape
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        x1, y1, x2, y2 = detections.xyxy[0].astype(int)

        mask[y1:y2, x1:x2] = 255

        save_image = params["tools"].save_image
        mask_name = save_image(mask, nd_arr=True)

        params["mask_image"] = mask_name
        self.offload_dino_if_required(params)

    def retrieve_mask_by_prompt(self, params, response):
        self.load_checkpoints()

        flare_task = params["task"]
        prompt = flare_task.source

        source_image = params["input_image"]
        source_image_path = os.path.join("build/images", source_image)
        input_image = cv2.imread(source_image_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_dino(device)

        classes = [prompt]
        box_threshold = 0.25
        text_threshold = 0.25
        nms_threshold = 0.8

        detections = self.grounding_dino.predict_with_classes(
            image=input_image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        self.offload_dino_if_required(params)

        self.load_sam(device)
        sam_predictor = SamPredictor(self.segment_anything)

        detections.mask = self.segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        height, width, _ = input_image.shape
        scene = np.zeros((height, width, 3), dtype=np.uint8)
        detections_mask = detections.mask[0]
        colored_mask = np.zeros_like(scene, dtype=np.uint8)

        colored_mask[:] = Color.white().as_bgr()
        opacity: float = 1
        scene = np.where(
            np.expand_dims(detections_mask, axis=-1),
            np.uint8(opacity * colored_mask + (1 - opacity) * scene),
            scene,
        )

        save_image = params["tools"].save_image
        mask_name = save_image(scene, nd_arr=True)

        params["mask_image"] = mask_name
        self.offload_sam_if_required(params)

    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray):
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def offload_dino_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading DINO")
            params["tools"].unload_model(self.grounding_dino.model)
            self.grounding_dino = None

    def offload_sam_if_required(self, params):
        if params["settings"]["offload_models"]:
            print("flare: offloading SAM")
            params["tools"].unload_model(self.segment_anything)
            self.segment_anything = None
