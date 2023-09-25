import torch
from .model import PHNet
import torchvision.transforms.functional as tf
from .util import inference_img, log
from .stylematte import StyleMatte
import numpy as np
import onnx
from .engine import execute_onnx_model
import cv2
from torchvision import transforms
import time


class Inference:
    def __init__(self, **kwargs):
        self.rank = 0
        self.__dict__.update(kwargs)
        self.model = PHNet(enc_sizes=self.enc_sizes,
                           skips=self.skips,
                           grid_count=self.grid_counts,
                           init_weights=self.init_weights,
                           init_value=self.init_value)
        state = torch.load(self.checkpoint.harmonizer,
                           map_location=self.device)

        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def harmonize(self, composite, mask):
        if len(composite.shape) < 4:
            composite = composite.unsqueeze(0)
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)
        composite = tf.resize(composite, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])

        log(composite.shape, mask.shape)
        with torch.no_grad():
            harmonized = self.model(composite, mask)  # ['harmonized']

        result = harmonized * mask + composite * (1-mask)

        return result


class Matting:
    def __init__(self, **kwargs):
        self.rank = 0
        self.__dict__.update(kwargs)
        if self.onnx:
            self.model = onnx.load(self.checkpoint.matting_onnx)
        else:
            self.model = StyleMatte().to(self.device)
            state = torch.load(self.checkpoint.matting,
                               map_location=self.device)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()

    def extract(self, inp):
        mask = inference_img(self.model, inp, self.device, self.onnx)
        inp_np = np.array(inp)
        fg = mask[:, :, None]*inp_np

        return [mask, fg]


def inference_img(model, img, device='cpu', onnx=True):
    beg = time.time()
    h, w, _ = img.shape
    # print(img.shape)
    if h % 8 != 0 or w % 8 != 0:
        img = cv2.copyMakeBorder(img, 8-h % 8, 0, 8-w %
                                 8, 0, cv2.BORDER_REFLECT)
    # print(img.shape)

    tensor_img = torch.from_numpy(img).permute(2, 0, 1).to(device)
    input_t = tensor_img/255.0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    input_t = normalize(input_t)
    input_t = input_t.unsqueeze(0).float()
    end_p = time.time()

    if onnx:
        out = execute_onnx_model(input_t, model)
    else:
        with torch.no_grad():
            out = model(input_t).cpu().numpy()
    end = time.time()
    log(f"Inference time: {end-beg}, processing time: {end_p-beg}")
    # print("out",out.shape)
    result = out[0][:, -h:, -w:]
    # print(result.shape)

    return result[0]
