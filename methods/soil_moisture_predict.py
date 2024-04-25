import torch.nn as nn
import torch 
from pathlib import Path
from typing import Callable

def rgb_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->tuple[torch.Tensor, torch.Tensor]:
    images, labels, _ = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    return outputs, labels

def rgb_and_tm_output(model: nn.Module, device: torch.device, data: list[torch.Tensor])->torch.Tensor:
    images, T_moistures, labels, _ = data
    images = images.to(device)
    T_moistures = T_moistures.to(device)
    labels = labels.to(device)
    outputs = model(images, T_moistures)
    return outputs, labels



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