import sys

import requests
import torch
from PIL import Image

sys.path.append("/home/ubuntu/models_implem")

import os

from e2e_implem.debug_utils import debug_dynamic_shapes_pipeline
from e2e_implem.e2e_utils import E2EModel
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)

os.environ["TORCH_LOGS"] = "+dynamic"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
model_id = "PekingU/rtdetr_v2_r101vd"
# model_id = "PekingU/rtdetr_r50vd_coco_o365"
# model_id = "ustc-community/dfine-large-obj365"
# model_id = "hustvl/yolos-tiny"
# model_id = "facebook/detr-resnet-50"
image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
image_big = image.resize((1024, 1024))

e2e_rtdetr = E2EModel(model_id, AutoModelForObjectDetection, AutoImageProcessor, "post_process_object_detection").to(
    "cpu"
)
images = e2e_rtdetr.get_tensors_inputs(image, device="cpu")
target_sizes = torch.tensor([image.size[::-1]])
inputs = {"images": images, "post_process_kwargs": {"target_sizes": target_sizes, "threshold": 0.5}}

debug_dynamic_shapes_pipeline(e2e_rtdetr, inputs, model_name=model_id)
