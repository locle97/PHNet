import torchvision
import io
import numpy as np
import torch.onnx
import onnx
from tools import Inference, Matting, log, extract_matte, harmonize, css, execute_onnx_model
from omegaconf import OmegaConf
import os
import sys
import torch
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image
import cv2 as cv
from onnxruntime import InferenceSession

args = OmegaConf.load(os.path.join(f"./config/demo.yaml"))

log("Model loading")
phnet = Inference(**args)
stylematte = Matting(**args)
log("Model loaded")
model = stylematte.model

x = torch.randn((1, 3, 720, 1280))
mask = torch.ones((1, 1, 512, 512))
path = 'checkpoints/stylematte-test.onnx'

# Export
torch.onnx.export(model, x, path, opset_version=16)

# Validation
onnx_model = onnx.load(path)
onnx.checker.check_model(onnx_model)
# execute_onnx_model(x, onnx_model)
