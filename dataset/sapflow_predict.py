from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
from PIL import Image
import json
from torchvision.transforms import ToTensor
import torch
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import torch

class RgbDataset(Dataset):
    def __init__(self, rgb_images_dir: Path, sapflow_dir: Path, transform=None):
        self.transform = transform
        self.rgb_images = []
        self.sapflows = []
        for rgb_images_sub_dir in rgb_images_dir.iterdir():
            if rgb_images_sub_dir.is_dir():
                sapflow_file_path = sapflow_dir / (rgb_images_sub_dir.name+'.CSV')
                sapflow_file = pd.read_csv(sapflow_file_path, dtype={'date':str, 'time':str, 'sapflow': str})
                sapflow_file['datetime'] = pd.to_datetime(sapflow_file['date'] + ' ' + sapflow_file['time'])
                sapflow_file = sapflow_file.drop(columns=['date', 'time'])
                sapflow_file = sapflow_file.set_index('datetime')

                for rgb_image in rgb_images_sub_dir.iterdir():
                    if rgb_image.is_file():
                        date_time = datetime.strptime(rgb_image.stem, '%Y%m%d%H%M')
                        self.rgb_images.append(rgb_image)
                        # python标准float是float64
                        self.sapflows.append([float(sapflow_file.loc[date_time]['sapflow'])])
                        


    def __getitem__(self, idx: int):
        rgb_image = Image.open(self.rgb_images[idx])
        sapflow = torch.tensor(self.sapflows[idx], dtype=torch.float32)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        return rgb_image, sapflow

    def __len__(self):
        return len(self.rgb_images)

class RgbTmDataset(Dataset):
    T_MOISTURE_DELTA_DAY = 1
    def __init__(self, rgb_images_dir: Path, sapflow_dir: Path, tm_file_path: Path, transform=None, temp_moisture_num_per_day:int=48):
        self.t_mositure_num_per_day = temp_moisture_num_per_day
        self.transform = transform
        self.rgb_images = []
        self.sapflows = []
        self.temp_moistures = []
        self.tm_file = pd.read_csv(tm_file_path, dtype={'temp':float, 'moisture': float})
        self.tm_file['datetime'] = pd.to_datetime(self.tm_file['time'], format='%Y/%m/%d %H:%M')
        for rgb_images_sub_dir in rgb_images_dir.iterdir():
            if rgb_images_sub_dir.is_dir():
                sapflow_file_path = sapflow_dir / (rgb_images_sub_dir.name+'.CSV')
                sapflow_file = pd.read_csv(sapflow_file_path, dtype={'date':str, 'time':str, 'sapflow': str})
                sapflow_file['datetime'] = pd.to_datetime(sapflow_file['date'] + ' ' + sapflow_file['time'])
                sapflow_file = sapflow_file.drop(columns=['date', 'time'])
                sapflow_file = sapflow_file.set_index('datetime')

                for rgb_image in rgb_images_sub_dir.iterdir():
                    if rgb_image.is_file():
                        date_time = datetime.strptime(rgb_image.stem, '%Y%m%d%H%M')
                        self.rgb_images.append(rgb_image)
                        # python标准float是float64
                        self.sapflows.append([float(sapflow_file.loc[date_time]['sapflow'])])
                        self.temp_moistures.append(self.get_t_moisture_by_time(date_time))
    

    def get_t_moisture_by_time(self, time: datetime)->list[list[float]]:
        temp_moisture = []
        delta_days_range = [i for i in range(-self.T_MOISTURE_DELTA_DAY, self.T_MOISTURE_DELTA_DAY+1)]
        day_count = 0
        for index, row in self.tm_file.iterrows():
            delta_days = (row['datetime'] - time).days

            if day_count+1 < len(delta_days_range) and delta_days == delta_days_range[day_count+1]:
                if len(temp_moisture) < (day_count+1)*self.t_mositure_num_per_day:
                    temp_moisture = [[0,0]]*((day_count+1)*self.t_mositure_num_per_day-len(temp_moisture))+temp_moisture
                day_count += 1
 
            if abs(delta_days) <= self.T_MOISTURE_DELTA_DAY:
                temp_moisture.append([row['temp'], row['moisture']])

        if len(temp_moisture) < (2*self.T_MOISTURE_DELTA_DAY+1)*self.t_mositure_num_per_day:
            temp_moisture = temp_moisture + [[0,0]]*((2*self.T_MOISTURE_DELTA_DAY+1)*self.t_mositure_num_per_day-len(temp_moisture))
        return temp_moisture


    def __getitem__(self, idx: int):
        rgb_image = Image.open(self.rgb_images[idx])
        sapflow = torch.tensor(self.sapflows[idx], dtype=torch.float32)
        t_mositure = torch.tensor(self.temp_moistures[idx], dtype=torch.float32)
        if self.transform:
            rgb_image = self.transform(rgb_image)
        return rgb_image, t_mositure, sapflow

    def __len__(self):
        return len(self.rgb_images)



if __name__ == '__main__':
    rgb_dataset = RgbDataset(
        rgb_images_dir=Path('../data/sapflow_predict_data/rgb_images'),
        sapflow_dir=Path('../data/sapflow_predict_data/sapflow')
    )
    rgb_tm_dataset = RgbTmDataset(
        rgb_images_dir=Path('../data/sapflow_predict_data/rgb_images'),
        sapflow_dir=Path('../data/sapflow_predict_data/sapflow'),
        tm_file_path=Path('../data/series_data/T_moisture_data.csv')
    )
    breakpoint()
