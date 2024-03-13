from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import json
from torchvision.transforms import ToTensor
import torch

class CropInfoDataset(Dataset):
    def __init__(self, rgb_images_list_path, t_T_data_file_path, labels=None, transform=None):
        rgb_images, t_T_moistures = [], []
        labels = []
        with open(rgb_images_list_path, 'r', encoding='utf-8') as f:
            images_list = json.load(f)
        with open(t_T_data_file_path, 'r', encoding='utf-8') as f:
            t_T_moistures_data = json.load(f)
        
        for date in images_list:
            if not date in t_T_moistures_data:
                continue
            for image in images_list[date]:
                rgb_images.append(image)
                t_T_moistures.append(t_T_moistures_data[date])
                labels.append([0.5])
        
        self.rgb_images = rgb_images
        self.t_T_moistures = t_T_moistures
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        image = Image.open(self.rgb_images[idx])
        if self.transform:
            image = self.transform(image)
        t_T_moisture = torch.tensor(self.t_T_moistures[idx], dtype=torch.float32) 
        label = torch.tensor(self.labels[idx], dtype=torch.float32) 
        return image, t_T_moisture, label

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