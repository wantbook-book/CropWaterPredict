from pathlib import Path
from typing import Callable
from dataset.dataset import CropInfoDataset
from torch.utils.data import Dataset
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

def rgb_tm_dataset(
    rgb_images_dir: Path,
    temp_moisture_filepath: Path,
    soil_moisture_filepath: Path,
    transform: Callable =None
)->Dataset:
    dataset = CropInfoDataset(
            rgb_images_directory=rgb_images_dir, 
            T_moisture_data_file_path=temp_moisture_filepath,
            # sap_flow_data_file_path=sapflow_data_file_path,
            labels_file_path=soil_moisture_filepath,
            transform=transform
        )
    return dataset