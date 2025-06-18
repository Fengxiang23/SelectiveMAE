# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
import PIL
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image,ImageFile
from torch.utils import data
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms as transforms
import numpy as np
import torch

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class AllDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None, split=None, tag=None):

        train_files = []
        train_targets = []
        for root, dirs, files in os.walk(root):
            for filename in dirs:
                file_dir = os.path.join(root, filename)
                for dirpath, dirnames, filenames in os.walk(file_dir):
                    for filename in filenames:
                        image_path = os.path.join(file_dir,  filename)
                        train_files.append(image_path)
                        train_targets.append(int(1))

        self.files = train_files
        self.targets = train_targets
        self.transform = transform
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        print('Creating All dataset with {} examples'.format(len(self.targets)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img_path = self.files[i]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform != None:
                img = self.transform(img)
                if img.shape[0] == 1:
                    img = img.float()
                    img = np.stack((img,) * 3, axis=0)
                    img = torch.tensor(img).squeeze()
                elif img.shape[0] != 1 and img.shape[0] != 3:
                    return self.__getitem__(i+1)
        except:
            return self.__getitem__(i+1)

        return img,self.targets[i]
