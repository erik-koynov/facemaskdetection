import numpy as np
from typing import List, Tuple, Dict
import cv2
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchvision.ops import nms

class GaussianNoise:
    def __init__(self, mean=0., std=1., p=0.):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        if np.random.rand(1) > 1-self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor


def resize_img_and_boxes(img: np.ndarray, bbox: List[List[int]], new_size: Tuple[int]):
    """
    Albumentations function does not transform the bboxes...
    :param img: the image (h w c)
    :param bbox: [[bbox1],[bbox2]]
    :param new_size: h, w
    :return:
    """
    h, w = img.shape[:-1]
    new_h, new_w = new_size

    new_bboxes = []
    for box in bbox:
        xmin, ymin, xmax, ymax = box

        xmin = (xmin / w) * new_w
        ymin = (ymin / h) * new_h

        xmax = (xmax / w) * new_w
        ymax = (ymax / h) * new_h

        new_bboxes.append([xmin, ymin, xmax, ymax])

    return cv2.resize(img, new_size), new_bboxes

def draw_basic_bbox(image: np.array, bboxes):
    image = image.copy()
    for bbox in bboxes:
        #print(bbox)
        cv2.rectangle(image, np.array(bbox[:2]).round().astype(int), np.array(bbox[2:]).round().astype(int),
                      color=(0, 255, 0), thickness=2)
    return image

def linear_learning_rate_warmup(optimizer, num_warmup_steps, num_training_steps, min_factor = 0.0):
    # a learning rate that decreases linearly from the initial lr set in the optimizer to lr*min_factor, after
    # a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    # num_training_steps = int(n_batches / grad_acc_steps=1) * n_epochs

    optimizer = optimizer

    return LambdaLR(optimizer, linear_warmup_factor_wrapper(num_warmup_steps, num_training_steps, min_factor), -1)

def linear_warmup_factor_wrapper(num_warmup_steps, num_training_steps, min_factor=0.0):
    num_warmup_steps = num_warmup_steps
    num_training_steps = num_training_steps
    def lr_lambda(current_step: int):
        #  computes a multiplicative factor given an integer parameter for the current (epoch)
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_factor, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return lr_lambda

def non_max_suppression(preds: dict, iou_threshold: float = 0.25, obj_score_thresh = 0.0)->Tuple[np.ndarray]:
    bboxes = preds['boxes']
    labels = preds['labels']
    scores = preds['scores']
    objectness_mask = scores>obj_score_thresh
    keep_indices = nms(bboxes[objectness_mask], scores[objectness_mask], iou_threshold)

    return bboxes[keep_indices].detach().numpy(), labels[keep_indices].detach().numpy(), scores[keep_indices].detach().numpy()

def create_img_for_plotting(img: np.array,
                            inv_encoding: Dict[int, str],
                            bboxes: np.ndarray,
                            labels: np.ndarray,
                            scores: np.ndarray):
    img = img.copy()
    for box, lbl, score in zip(bboxes.astype(int), labels, scores):
        label = inv_encoding[lbl-1]
        img = cv2.rectangle(img, box[:2], box[2:], (0,255,0), 2)

        (w, h), _ = cv2.getTextSize(f"{label}: {score:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        img = cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + w, box[1]), (0,255,0), -1)
        img = cv2.putText(img,f"{label}: {score:.2f}", (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return img