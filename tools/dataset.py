import os.path
import torch
import torchvision.transforms.functional as tf
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import imutils
import cv2
import glob
import random


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[: min(max_dataset_size, len(images))]


class IhdDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, opt, stage="train", apply_augs=False, subset="IhdDataset"):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super(IhdDataset, self).__init__()
        self.opt = opt
        self.apply_augs = apply_augs
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.image_size = opt.crop_size

        if subset == "IhdDataset":
            self.datasets = [
                "HCOCO",
                "HFlickr",
                "Hday2night",
                "HAdobe5k",
            ]
        else:
            self.datasets = [subset]

        dataset_root = self.opt.iharmony
        self.paths = [os.path.join(dataset_root, dataset) for dataset in self.datasets]

        for dataset, path in zip(self.datasets, self.paths):
            file = os.path.join(path, f"{dataset}_{stage}.txt")
            with open(file, "r") as f:
                for line in f.readlines():
                    line = "composite_images/" + line.rstrip()
                    name_parts = line.split("_")
                    mask_path = line.replace("composite_images", "masks")
                    mask_path = mask_path.replace(("_" + name_parts[-1]), ".png")
                    gt_path = line.replace("composite_images", "real_images")
                    gt_path = gt_path.replace(
                        "_" + name_parts[-2] + "_" + name_parts[-1], ".jpg"
                    )
                    self.image_paths.append(os.path.join(dataset_root, path, line))
                    self.mask_paths.append(os.path.join(dataset_root, path, mask_path))
                    self.gt_paths.append(os.path.join(dataset_root, path, gt_path))

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
        ]
        self.transforms = transforms.Compose(transform_list)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """

        comp = Image.open(self.image_paths[index]).convert("RGB")
        real = Image.open(self.gt_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("1")

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        real = tf.resize(real, [self.image_size, self.image_size])

        comp = self.transforms(comp)
        mask = tf.to_tensor(mask)
        mask = torch.where(mask <= 0.5, 0, 1)
        real = self.transforms(real)

        to_flip = random.random() < 0.3
        if self.apply_augs and to_flip:
            real = transforms.functional.hflip(real)
            comp = transforms.functional.hflip(comp)
            mask = transforms.functional.hflip(mask)
        inputs = torch.cat([comp, mask], 0)

        return {
            "inputs": inputs,
            "comp": comp,
            "real": real,
            "img_path": self.image_paths[index],
            "mask": mask,
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


class FFHQH(data.Dataset):
    def __init__(self, opt, stage="train", apply_augs=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            stage -- stores train/test/val value for split
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super(FFHQH, self).__init__()
        self.opt = opt
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.image_size = opt.crop_size
        self.apply_augs = apply_augs

        input_list, alpha_list = [], []

        file = os.path.join(self.opt.ffhqh, f"{stage}_split.txt")
        with open(file, "r") as f:
            for line in f.readlines():
                line = line.rstrip()

                input_path = os.path.join(self.opt.ffhqh, "comp", line)
                mask_path = os.path.join(self.opt.ffhqh, "alpha", line)
                gt_path = os.path.join(self.opt.ffhq, line)

                self.image_paths.append(input_path)
                self.mask_paths.append(mask_path)
                self.gt_paths.append(gt_path)

        self.image_paths = self.image_paths
        self.mask_paths = self.mask_paths
        self.gt_paths = self.gt_paths

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        comp = Image.open(self.image_paths[index]).convert("RGB")
        real = Image.open(self.gt_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index])

        comp = tf.resize(comp, [self.image_size, self.image_size])
        mask = tf.resize(mask, [self.image_size, self.image_size])
        real = tf.resize(real, [self.image_size, self.image_size])

        comp = tf.to_tensor(comp)
        mask = tf.to_tensor(mask)
        mask = torch.where(mask <= 0.5, 0, 1)
        real = tf.to_tensor(real)

        to_flip = random.random() < 0.3
        if self.apply_augs and to_flip:
            real = transforms.functional.hflip(real)
            comp = transforms.functional.hflip(comp)
            mask = transforms.functional.hflip(mask)

        inputs = torch.cat([comp, mask], 0)

        return {
            "inputs": inputs,
            "comp": comp,
            "real": real,
            "mask": mask,
            "img_path": self.image_paths[index],
        }

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


class EasyPortraitH(data.Dataset):
    def __init__(self, opt, is_train=True, subset="FFHQH"):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        super(EasyPortraitH, self).__init__()
        self.opt = opt
        self.image_paths, self.mask_paths, self.gt_paths = [], [], []
        self.image_size = self.opt.crop_size

        if is_train:
            stage = "train"
        else:
            stage = "test"
        self.save_ratio = opt.save_ratio
        self.image_paths = list(
            sorted(glob.glob(os.path.join(self.opt.dataset_root, "images", stage, "*")))
        )
        self.mask_paths = list(
            sorted(
                glob.glob(
                    os.path.join(self.opt.dataset_root, "annotations", stage, "*")
                )
            )
        )
        self.gt_paths = list(
            sorted(glob.glob(os.path.join(self.opt.dataset_root, "real", stage, "*")))
        )

    def _resize_with_pad(self, img, pad_val=0):
        """Return a padded image with 0 values. Image is resized in according to its max
        dimension size, the short dimension is resized with aspect ration and padded to
        square image

        Parameters:
            img -- cv2.imread result, array with 3 channels.
            pad_val -- int | list[int], value of padding.

        Returns:
            resized image (3 channel for image, 1 channel for mask).
        """
        h, w = img.shape[:2]
        if w > h:
            resized = imutils.resize(
                img,
                width=self.image_size,
            )
        else:
            resized = imutils.resize(
                img,
                height=self.image_size,
            )
        in_shape = resized.shape
        padding = (
            int((self.image_size - in_shape[1]) / 2),
            int((self.image_size - in_shape[0]) / 2),
        )
        padded = cv2.copyMakeBorder(
            resized,
            padding[1],
            padding[1],
            padding[0],
            padding[0],
            cv2.BORDER_CONSTANT,
            value=pad_val,
        )
        result = cv2.resize(padded, (self.image_size, self.image_size))

        return result

    def _resize_without_pad(self, img):
        return tf.resize(img, [self.image_size, self.image_size])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        if self.save_ratio:
            comp = cv2.cvtColor(cv2.imread(self.image_paths[index]), cv2.COLOR_BGR2RGB)
            real = cv2.cvtColor(cv2.imread(self.gt_paths[index]), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.mask_paths[index], cv2.COLOR_BGR2GRAY)
            mean = cv2.mean(cv2.bitwise_and(comp, comp, mask=255 - mask))
            comp = self._resize_with_pad(comp, pad_val=mean)
            real = self._resize_with_pad(real, pad_val=mean)
            mask = self._resize_with_pad(mask, pad_val=0)
        else:
            comp = Image.open(self.image_paths[index]).convert("RGB")
            real = Image.open(self.gt_paths[index]).convert("RGB")
            mask = Image.open(self.mask_paths[index])
