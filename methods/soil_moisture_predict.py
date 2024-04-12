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
    images, T_moistures, labels, _ = data
    images = images.to(device)
    T_moistures = T_moistures.to(device)
    labels = labels.to(device)
    outputs = model(images, T_moistures)
    return outputs, labels


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

def rgb_TM_dataset(
    rgb_images_dir: Path,
    T_moisture_data_file_path: Path,
    labels_file_path: Path,
    transform: Callable =None
)->Dataset:
    dataset = CropInfoDataset(
            rgb_images_directory=rgb_images_dir, 
            T_moisture_data_file_path=T_moisture_data_file_path,
            # sap_flow_data_file_path=sapflow_data_file_path,
            labels_file_path=labels_file_path,
            transform=transform
        )
    return dataset

def rgb_TM_collate_fn(batch):
    # print(type(batch))
    # print(len(batch))
    # print(type(batch[0]))
    # print(len(batch[0]))
    # print(type(batch[0][0]))
    # for data in batch:
    #     print(type(data[1]))
    #     print(data[1].size())
    # TODO: hardcode
    max_length = max([data[1].size(0) for data in batch])
    batch = [
        [
            data[0],
            torch.cat([torch.zeros(max_length-len(data[1]), 3), data[1]], dim=0),
            data[2],
            data[3]
        ] for data in batch
    ]
    data0s = torch.stack([data[0] for data in batch], dim=0)
    data1s = torch.stack([data[1] for data in batch], dim=0)
    data2s = torch.stack([data[2] for data in batch], dim=0)
    data3s = [data[3] for data in batch]


    return [data0s, data1s, data2s, data3s]

    # print(type(batch))
    # print(len(batch))
    # print(type(batch[0]))
    # print(len(batch[0]))
    # print(type(batch[0][0]))
    # for data in batch:
    #     print(type(data[1]))
    #     print(data[1].size())
    return batch