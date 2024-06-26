from pathlib import Path
from dataset.dataset import CropInfoDataset
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from torch.utils.data.dataset import random_split
import timm
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable
from utils.early_stopping import EarlyStopping
from utils.utils import get_next_subdir_name, save_results
import models.soil_moisture_predict as smp_models
import methods.soil_moisture_predict as smp_methods
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
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
    val_mapes = []
    with torch.no_grad():  # 在评估阶段不计算梯度
        for data in tqdm(validation_dataloader, desc='validate'):
            # images = images.to(device)
            # labels = labels.to(device)
            # outputs = model(images)
            outputs, labels = output_func(model, device, data)
            loss = criterion(outputs, labels)
            validation_losses.append(loss.item())
            val_mapes.append(torch.mean(torch.abs(labels-outputs)/labels).item())

    model.train()  # 将模型设置回训练模式
    return np.mean(validation_losses), np.mean(val_mapes)


def train(
    model: nn.Module,
    device: torch.device,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    results_dir: Path,
    output_func: Callable,
    collate_fn: Callable=None,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.001,
    val_epoches: int = 5,
    patience: int = 4,
    draw_skip_epoches: int = 0,
    num_workers:int = 8,
    save_models: bool = True
):
    results_dir.mkdir(exist_ok=True, parents=True)
    new_dir_path = results_dir / get_next_subdir_name(results_dir)
    new_dir_path.mkdir(exist_ok=False)
    writer = SummaryWriter(new_dir_path)

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model = model.to(device)
    model.train()
    
    if collate_fn is None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # adjust learning rates
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    val_losses = []
    train_mapes = []
    val_mapes = []
    best_model_weights = None
    for epoch in range(num_epochs):
        batch_losses = []
        batch_mapes = []
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}'):  # 假设dataloader已准备好
            # images = images.to(device)
            # labels = labels.to(device)
            # outputs = model(images)
            outputs, labels = output_func(model, device, data)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            batch_mapes.append(torch.mean(torch.abs(labels-outputs)/labels).item())
            batch_losses.append(loss.item())
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        train_mapes.append(np.mean(batch_mapes))
        writer.add_scalar('Loss/train', train_loss, epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')
        if (epoch+1) % val_epoches == 0:  # 每10个epoch后在验证集上进行评估
            validation_loss, val_mape = evaluate(
                                        model=model, 
                                        validation_dataloader=val_loader, 
                                        criterion=criterion, 
                                        device=device,
                                        output_func=output_func
                                    )
            val_losses.append(validation_loss)
            val_mapes.append(val_mape)
            print(f'After epoch {epoch+1}, Validation Loss: {validation_loss:.4f}')
            writer.add_scalar('Loss/Validate', validation_loss, epoch)
            if early_stopping(validation_loss):
                best_model_weights = model.state_dict().copy()
            if early_stopping.early_stop:
                print("Early stopping triggered! Best validation loss:", early_stopping.best_score)
                break
    writer.close()
    with open(new_dir_path / 'hyper_parameters.txt', 'w') as f:
        f.write(f'num_epochs: {num_epochs}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'lr: {lr}\n')
        f.write(f'patience: {patience}\n')
        f.write(f'num_workers: {num_workers}\n')
        f.write(f'val_epochs: {val_epoches}\n')

    model.output_net_settings(new_dir_path)
    save_results(
        best_model_weights=best_model_weights,
        model=model,
        train_mapes=train_mapes,
        val_mapes=val_mapes,
        train_losses=train_losses,
        val_losses=val_losses,
        output_dir=new_dir_path,
        val_epoches=val_epoches,
        skip_epoches=draw_skip_epoches,
        save_models=save_models
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
    TRAIN_RATIO = 0.8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = smp_models.RGB_Model()
    model = smp_models.RgbVgg16Model()
    # model = smp_models.RGB_TM_Model()
    # model = smp_models.RgbVgg16TmModel()

    src_dir = Path('./main.py').resolve().parent
    rgb_image_dir = src_dir / 'data' / 'rgb_images'
    infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    T_moisture_data_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    sapflow_data_file_path = src_dir / 'data' / 'series_data' / 'sapflow_data.CSV'
    labels_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'

    # train_dataset = smp_methods.rgb_dataset(
    #     rgb_images_dir=rgb_image_dir,
    #     labels_file_path=labels_file_path,
    #     transform=model.get_image_transform(is_training=True)
    # )
    # validation_dataset = smp_methods.rgb_dataset(
    #     rgb_images_dir=rgb_image_dir,
    #     labels_file_path=labels_file_path,
    #     transform=model.get_image_transform(is_training=False)
    # )
    train_dataset = smp_methods.rgb_TM_dataset(
        rgb_images_dir=rgb_image_dir,
        T_moisture_data_file_path=T_moisture_data_file_path,
        labels_file_path=labels_file_path,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = smp_methods.rgb_TM_dataset(
        rgb_images_dir=rgb_image_dir,
        T_moisture_data_file_path=T_moisture_data_file_path,
        labels_file_path=labels_file_path,
        transform=model.get_image_transform(is_training=False)
    )
    total_size = len(train_dataset)
    print('dataset size:', total_size)
    train_size = int(total_size * TRAIN_RATIO)
    indices = np.random.permutation(total_size)
    train_dataset = Subset(train_dataset, indices[:train_size])
    validation_dataset = Subset(validation_dataset, indices[train_size:])
    train(
        model=model,
        device=device,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        # results_dir=src_dir / 'train' /'soil_moisture_predict' / 'rgb',
        results_dir=src_dir / 'train' /'soil_moisture_predict' / 'rgb_tm',
        num_epochs=300,
        batch_size=32,
        lr=0.05,
        val_epoches=2,
        patience=10,
        draw_skip_epoches=0,
        # output_func=smp_methods.rgb_output
        output_func=smp_methods.rgb_and_TM_output,
        # collate_fn=smp_methods.rgb_TM_collate_fn
    )

    
    