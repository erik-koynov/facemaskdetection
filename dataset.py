from torch.utils.data import Dataset
import pandas as pd
from utils import resize_img_and_boxes, GaussianNoise
import numpy as np
from typing import List
import imgaug.augmenters as iaa
import torch
import cv2
from torchvision.transforms import RandomGrayscale, GaussianBlur, ColorJitter, Compose, ToTensor

class FaceMaskVOCDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame,
                 out_width: int,
                 out_height:int,
                 training = True,
                 cache= False):
        self.dataset = dataset.reset_index(drop=True)
        self.out_width = out_width
        self.out_height = out_height
        self.transform = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.1, 0.1)},scale=(0.7, 1),),
            iaa.PerspectiveTransform(scale=(0, 0.1)),

        ])
        #self.resize = iaa.Resize({"height":self.out_height, "width": self.out_width}, interpolation='linear'),

        self.color_augment = Compose([
            ToTensor(),
            RandomGrayscale(p=0.05),
            GaussianBlur(kernel_size=3,sigma=(0.001, 0.5)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            GaussianNoise(std=0.5, p=0.05)
        ])
        self.cache = cache
        self.training = training
        if self.cache:
            self.cache_memory = {}


    def __getitem__(self, idx: int):
        image_path, bboxes, labels= self.dataset.iloc[idx].values
        #print(image_path)
        if self.cache:
            image = self._load_from_cache(image_path)
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, bboxes = resize_img_and_boxes(image, bboxes, (self.out_height, self.out_width))

        if self.training:
            image, bboxes = self._apply_transforms(image, bboxes)
            image_tensor = self.color_augment(image)

        else:
            image_tensor = ToTensor()(image)

        targets = self._format_targets_as_dict(bboxes=bboxes, labels=labels, idx=idx)

        return image_tensor, targets


    def _apply_transforms(self, image: np.ndarray, bboxes: List[List[int]]):
        img_aug, bbox = self.transform (image=image,
            bounding_boxes=np.array(bboxes)[None, ...])
        return img_aug, bbox[0]

    def _format_targets_as_dict(self, bboxes: List[List[int]], labels: List[int], idx: int):
        boxes = torch.FloatTensor(bboxes)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        #print(labels)
        labels = torch.LongTensor(labels)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = image_id

        return target

    def _load_from_cache(self, image_path) -> np.ndarray:
        if self.cache_memory.get(image_path, None) is not None:
            image = self.cache_memory[image_path].copy()
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.cache_memory[image_path] = image.copy()
        return image

    def __len__(self):
        return len(self.dataset)