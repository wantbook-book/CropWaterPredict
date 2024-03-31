import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from dataset import CropInfoDataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

src_dir = Path('./dataset_test.py').resolve().parents[1]
rgb_image_dir = src_dir / 'data' / 'rgb_images'
infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
T_moisture_data_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
sapflow_data_file_path = src_dir / 'data' / 'series_data' / 'sapflow_data.CSV'
labels_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'

dataset = CropInfoDataset(
        rgb_images_directory=rgb_image_dir, 
        infrared_images_directory=infrared_image_dir, 
        T_moisture_data_file_path=T_moisture_data_file_path,
        sap_flow_data_file_path=sapflow_data_file_path,
        labels_file_path=labels_file_path,
        transform=ToTensor()
    )

print(dataset.rgb_images)
print(dataset.labels)