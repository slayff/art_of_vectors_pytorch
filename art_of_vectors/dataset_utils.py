import json
import os

import numpy as np

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms


class ImageData(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms

        self.filenames = [n for n in os.listdir(self.path)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.path, self.filenames[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return dict(name=self.filenames[idx], image=image)


def fix_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.determenistic = True


def get_images_dataloader(path, batch_size, transforms=None, num_workers=5):
    dataset = ImageData(path, transforms)
    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        drop_last=True
    )


def get_images_transforms(perturbation=None):
    tr = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if perturbation is not None:
        tr.insert(2, transforms.Lambda(lambda x: x + perturbation))
        tr.insert(3, transforms.Lambda(lambda x: torch.clamp(x, 0, 1)))

    return transforms.Compose(tr)


def get_idx2label_map(labels_json_path):
    with open(labels_json_path) as f:
        j = json.load(f)

    idx2label = {}
    for k, v in j.items():
        idx2label[int(k)] = v[1]
    return idx2label


def normalize_image(img):
    img = (img - img.min()) / (img - img.max())
    return img.permute(1, 2, 0)
