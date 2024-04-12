from pathlib import Path
from dataset.dataset import CropInfoDataset
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from torch.utils.data.dataset import random_split
import timm
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.early_stopping import EarlyStopping
from utils.utils import get_next_subdir_name, save_results
from models.soil_water_predict import SoilWaterPredictModel
from models.water_predict_model import WaterPredictModel
import methods.soil_moisture_predict as smp_methods
from tqdm import tqdm
def evaluate(
    model: nn.Module, 
    validation_dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    output_func: callable
):
    model.eval()  # 将模型设置为评估模式
    validation_losses = []
    with torch.no_grad():  # 在评估阶段不计算梯度
        for data in tqdm(validation_dataloader, desc='validate'):
            # images = images.to(device)
            # labels = labels.to(device)
            # outputs = model(images)
            outputs, labels = output_func(model, device, data)
            loss = criterion(outputs, labels)
            validation_losses.append(loss.item())

    model.train()  # 将模型设置回训练模式
    return np.mean(validation_losses)

def train(
    model: nn.Module,
    device: torch.device,
    dataset: Dataset,
    results_dir: Path,
    output_func: callable,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.001,
    train_ratio: float = 0.8,
    val_epoches: int = 5,
    patience: int = 4,
    draw_skip_epoches: int = 6,
):
    results_dir.mkdir(exist_ok=True, parents=True)
    new_dir_path = results_dir / get_next_subdir_name(results_dir)
    new_dir_path.mkdir(exist_ok=False)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model = model.to(device)
    model.train()
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    validation_dataset.dataset.transform = model.get_image_transform(is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}'):  # 假设dataloader已准备好
            # images = images.to(device)
            # labels = labels.to(device)
            # outputs = model(images)
            outputs, labels = output_func(model, device, data)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
        if (epoch+1) % val_epoches == 0:  # 每10个epoch后在验证集上进行评估
            validation_loss = evaluate(
                                        model=model, 
                                        validation_dataloader=val_loader, 
                                        criterion=criterion, 
                                        device=device,
                                        output_func=output_func
                                    )
            val_losses.append(validation_loss)
            print(f'After epoch {epoch+1}, Validation Loss: {validation_loss:.4f}')
            early_stopping(validation_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

    save_results(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=new_dir_path,
        val_epoches=val_epoches,
        skip_epoches=draw_skip_epoches
    )

# 示例: 如何调用validate_model函数
# 假设 validation_dataset 是你的验证数据集
# validation_dataset = # 你的验证数据集
# validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False, num_workers=1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = # 你的模型实例
# model.to(device)

# validate_model(model, validation_dataloader, device)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SoilWaterPredictModel()
    src_dir = Path('./main.py').resolve().parent
    rgb_image_dir = src_dir / 'data' / 'rgb_images'
    infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    T_moisture_data_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    sapflow_data_file_path = src_dir / 'data' / 'series_data' / 'sapflow_data.CSV'
    labels_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'
    dataset = smp_methods.rgb_dataset(
        rgb_images_dir=rgb_image_dir,
        labels_file_path=labels_file_path,
        transform=model.get_image_transform(is_training=True)
    )
    train(
        model=model,
        device=device,
        dataset=dataset,
        results_dir=src_dir / 'train' /'soil_moisture_predict' / 'rgb',
        num_epochs=1,
        batch_size=32,
        lr=0.001,
        train_ratio=0.8,
        val_epoches=1,
        patience=4,
        draw_skip_epoches=0,
        output_func=smp_methods.rgb_output
    )

    
    