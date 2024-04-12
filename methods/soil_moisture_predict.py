import torch.nn as nn
import torch 
from pathlib import Path
from typing import Callable
from dataset.dataset import CropInfoDataset
from torch.utils.data import Dataset
def rgb_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->tuple[torch.Tensor, torch.Tensor]:
    images, labels, _ = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    return outputs, labels

def rgb_and_TM_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->torch.Tensor:
    pass


def rgb_dataset(
    rgb_images_dir: Path,
    labels_file_path: Path,
    transform: Callable =None
)->Dataset:
    dataset = CropInfoDataset(
            rgb_images_directory=rgb_images_dir, 
            # infrared_images_directory=infrared_image_dir, 
            # T_moisture_data_file_path=T_moisture_data_file_path,
            # sap_flow_data_file_path=sapflow_data_file_path,
            labels_file_path=labels_file_path,
            transform=transform
        )
    return dataset