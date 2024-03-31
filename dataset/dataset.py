from torch.utils.data import Dataset, DataLoader
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

class CropInfoDataset(Dataset):
    T_MOISTURE_DELTA_DAY = 1
    SAP_FLOW_DELTA_DAY = 1
    DAY_SECONDS = 24*3600
    RGB_IMAGE = 0
    INFRARED_IMAGE = 1
    T_MOISTURE_DATA = 2
    SAP_FLOW_DATA = 3
    LABEL_DATA = 4
    def __init__(
            self, 
            labels_file_path:Path,
            rgb_images_directory:Path=None, 
            infrared_images_directory:Path=None, 
            T_moisture_data_file_path:Path=None,
            sap_flow_data_file_path:Path=None,
            transform=None
        ):
        self.need_data = [
            rgb_images_directory is not None,
            infrared_images_directory is not None,
            T_moisture_data_file_path is not None,
            sap_flow_data_file_path is not None,
            labels_file_path is not None
        ]
        self.transform = transform
        self.rgb_images = []
        self.infrared_images = []
        self.T_moistures = []
        self.sap_flows = []
        self.labels = []
        self.have_or_nots = []
        self.labels_csv = pd.read_csv(labels_file_path, dtype={'day':int})
        self.labels_csv['date'] = pd.to_datetime(self.labels_csv['date'], format='%Y/%m/%d')
        columns = self.labels_csv.columns.tolist()
        day_index = columns.index('day')
        pots_column_names = columns[day_index+1:]
        for pot in pots_column_names:
            self.labels_csv[pot] = self.labels_csv[pot].astype(float)
        
        if self.need_data[self.T_MOISTURE_DATA]:
            self.T_moistures_csv = pd.read_csv(
                                            T_moisture_data_file_path, 
                                            dtype={'id':int, 'temp':float, 'moisture':float}
                                        )
            self.T_moistures_csv['time'] = pd.to_datetime(self.T_moistures_csv['time'], format='%Y/%m/%d %H:%M')

        if self.need_data[self.SAP_FLOW_DATA]:
            self.sap_flows_csv = pd.read_csv(sap_flow_data_file_path)
            self.sap_flows_csv['date'] = pd.to_datetime(self.sap_flows_csv['date'], format='%Y/%m/%d %H:%M')
            self.sap_flows_csv_colums = self.sap_flows_csv.columns.tolist()
            day_index = self.sap_flows_csv_colums.index('day')
            for pot in self.sap_flows_csv_colums[day_index+1:]:
                self.sap_flows_csv[pot] = self.sap_flows_csv[pot].astype(float)
        
        base_time = None
        for index, row in self.labels_csv.iterrows():
            date, day = row['date'], row['day']
            if base_time is None:
                base_time = date - timedelta(days=day)
            for pot in pots_column_names:
                have_or_not = [False for _ in range(5)]
                if self.need_data[self.RGB_IMAGE]:
                    rgb_image_path = rgb_images_directory / f'{date.month}-{date.day}' / f'{pot}.JPG'
                    if rgb_image_path.exists():
                        have_or_not[self.RGB_IMAGE] = True
                        self.rgb_images.append(rgb_image_path)
                if self.need_data[self.INFRARED_IMAGE]:
                    infrared_image_path = infrared_images_directory / f'{date.month}-{date.day}' /'infrared'/ f'{pot}.jpg'
                    if infrared_image_path.exists():
                        have_or_not[self.INFRARED_IMAGE] = True
                        self.infrared_images.append(infrared_image_path)
                if self.need_data[self.T_MOISTURE_DATA]:
                    T_moisture = self.get_T_moisture_by_day_and_pot(base_time, day)
                    if T_moisture is not None:
                        have_or_not[self.T_MOISTURE_DATA] = True
                        self.T_moistures.append(T_moisture)
                if self.need_data[self.SAP_FLOW_DATA]:
                    sap_flow = self.get_sap_flow_by_day_and_pot(base_time, day, pot)
                    if sap_flow is not None:
                        have_or_not[self.SAP_FLOW_DATA] = True
                        self.sap_flows.append(sap_flow)
                if (self.need_data[self.RGB_IMAGE] and have_or_not[self.RGB_IMAGE] or not self.need_data[self.RGB_IMAGE]) and \
                    (self.need_data[self.INFRARED_IMAGE] and have_or_not[self.INFRARED_IMAGE] or not self.need_data[self.INFRARED_IMAGE]) and \
                    (self.need_data[self.T_MOISTURE_DATA] and have_or_not[self.T_MOISTURE_DATA] or not self.need_data[self.T_MOISTURE_DATA]) and \
                    (self.need_data[self.SAP_FLOW_DATA] and have_or_not[self.SAP_FLOW_DATA] or not self.need_data[self.SAP_FLOW_DATA]):
                    self.labels.append([row[pot]])
                    have_or_not[self.LABEL_DATA] = True
                    self.have_or_nots.append(have_or_not)
                else:
                    if self.need_data[self.RGB_IMAGE] and have_or_not[self.RGB_IMAGE]:
                        self.rgb_images.pop()
                    if self.need_data[self.INFRARED_IMAGE] and have_or_not[self.INFRARED_IMAGE]:
                        self.infrared_images.pop()
                    if self.need_data[self.T_MOISTURE_DATA] and have_or_not[self.T_MOISTURE_DATA]:
                        self.T_moistures.pop()
                    if self.need_data[self.SAP_FLOW_DATA] and have_or_not[self.SAP_FLOW_DATA]:
                        self.sap_flows.pop()
        

    def time_to_seconds(self, dt:datetime)->int:
        return dt.hour*3600 + dt.minute*60 + dt.second

    def get_T_moisture_by_day_and_pot(self, base_time:datetime, day:int)->list[list[float]]:
        T_mositure = []
        for index, row in self.T_moistures_csv.iterrows():
            T_mositure_day = (row['time'] - base_time).days
            if abs(T_mositure_day-day) <= self.T_MOISTURE_DELTA_DAY:
                T_mositure.append([T_mositure_day*self.DAY_SECONDS+self.time_to_seconds(row['time']), float(row['temp']), row['moisture']])
        if len(T_mositure) == 0:
            return None
        return T_mositure

    def get_sap_flow_by_day_and_pot(self, base_time:datetime, day:int, pot:str)->list[list[float]]:
        if pot not in self.sap_flows_csv_colums:
            return None
        sap_flow = []
        start = False
        for index, row in self.sap_flows_csv.iterrows():
            sap_flow_day = (row['date'] - base_time).days
            if abs(sap_flow_day-day) <= self.SAP_FLOW_DELTA_DAY:
                if pd.isna(row[pot]):
                    if not start:
                        continue
                    else:
                        break
                start = True
                sap_flow.append([sap_flow_day*self.DAY_SECONDS+self.time_to_seconds(row['date']), row[pot]])
        if len(sap_flow) == 0:
            return None
        return sap_flow

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        have_or_not = self.have_or_nots[idx]
        rgb_image, infrared_image = None, None
        
        if self.need_data[self.RGB_IMAGE] and have_or_not[self.RGB_IMAGE]:
            rgb_image = Image.open(self.rgb_images[idx])
        if self.need_data[self.INFRARED_IMAGE] and have_or_not[self.INFRARED_IMAGE]:
            infrared_image = Image.open(self.infrared_images[idx])
        if self.transform:
            if self.need_data[self.RGB_IMAGE] and have_or_not[self.RGB_IMAGE]:
                rgb_image = self.transform(rgb_image)
            if self.need_data[self.INFRARED_IMAGE] and have_or_not[self.INFRARED_IMAGE]:
                infrared_image = self.transform(infrared_image)
        T_moisture, sapflow, label = None, None, None
        if self.need_data[self.T_MOISTURE_DATA] and have_or_not[self.T_MOISTURE_DATA]:
            T_moisture = torch.tensor(self.T_moistures[idx], dtype=torch.float32) 
        if self.need_data[self.SAP_FLOW_DATA] and have_or_not[self.SAP_FLOW_DATA]:
            sapflow = torch.tensor(self.sap_flows[idx], dtype=torch.float32)
        if self.need_data[self.LABEL_DATA] and have_or_not[self.LABEL_DATA]:
            label = torch.tensor(self.labels[idx], dtype=torch.float32) 
        return_data = []
        if self.need_data[self.RGB_IMAGE]:
            return_data.append(rgb_image)
        if self.need_data[self.INFRARED_IMAGE]:
            return_data.append(infrared_image)
        if self.need_data[self.T_MOISTURE_DATA]:
            return_data.append(T_moisture)
        if self.need_data[self.SAP_FLOW_DATA]:
            return_data.append(sapflow)
        if self.need_data[self.LABEL_DATA]:
            return_data.append(label)
        return_data.append(have_or_not)
        return return_data

if __name__ == '__main__':
    dataset = CropInfoDataset('preprocess/rgb_images_list.json', 'preprocess/t_T_mo_data.json', transform=ToTensor())

    print(dataset[0])
    print(len(dataset))
    print(dataset[0][0].size)
    print(dataset[0][1])
    print(dataset[0][2])

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched)
        break