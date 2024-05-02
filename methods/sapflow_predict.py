import torch.nn as nn
import torch 
from pathlib import Path
from typing import Callable
from dataset.dataset import CropInfoDataset
from torch.utils.data import Dataset
def rgb_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->tuple[torch.Tensor, torch.Tensor]:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    return outputs, labels

def rgb_and_tm_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->torch.Tensor:
    images, temp_moistures, labels = data
    images = images.to(device)
    temp_moistures = temp_moistures.to(device)
    labels = labels.to(device)
    outputs = model(images, temp_moistures)
    return outputs, labels

def tm_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->tuple[torch.Tensor, torch.Tensor]:
    _, temp_moistures, labels = data
    temp_moistures = temp_moistures.to(device)
    labels = labels.to(device)
    outputs = model(temp_moistures)
    return outputs, labels