from torch import Tensor
from tools import Inference
from omegaconf import OmegaConf
import os
import PIL
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms.functional as tf
from pathlib import Path
import sys


def predict(comp, mask) -> np.uint8:
    comp = comp.convert("RGB")
    mask = mask.convert("1")
    in_shape = comp.size[::-1]
    image_size = args.input.image_size
    comp: Tensor = tf.resize(comp, [image_size, image_size])
    mask: Tensor = tf.resize(mask, [image_size, image_size])

    compt: Tensor = tf.to_tensor(comp)
    maskt: Tensor = tf.to_tensor(mask)
    res = infer.harmonize(compt, maskt)
    res: Tensor = tf.resize(res, in_shape)

    return np.uint8((res * 255)[0].permute(1, 2, 0).numpy())


if __name__ == "__main__":
    config_path = sys.argv[-1]
    dir_name = os.path.dirname(os.path.abspath(__file__))
    args = OmegaConf.load(os.path.join(dir_name, config_path))
    Path(args.output.path).mkdir(parents=True, exist_ok=True)

    infer = Inference(**args)
    path = os.path.split(args.input.composite_path)[-1]
    comp = PIL.Image.open(args.input.composite_path)
    mask = PIL.Image.open(args.input.mask_path)

    harmonized: np.uint8 = predict(comp, mask)
    path_harm: str = os.path.join(args.output.path, f"harm_{path}")
    path_comp: str = os.path.join(args.output.path, f"comp_{path}")
    path_mask: str = os.path.join(args.output.path, f"mask_{path[:-3]}.png")

    plt.imsave(path_harm, harmonized)
    comp.save(path_comp)
    mask.save(path_mask)
