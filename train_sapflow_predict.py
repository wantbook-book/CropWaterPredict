from utils.train import train
import torch
import models.sapflow_predict as sfp_models
import dataset.sapflow_predict as sfp_datasets
import methods.sapflow_predict as sfp_methods
from torch.utils.data import Subset
from pathlib import Path
import numpy as np
TRAIN_RATIO = 0.8
def rgb_vgg16_train(
    rgb_images_dir: Path, 
    sapflow_dir: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8
):
    model = sfp_models.RgbVgg16Model()
    train_dataset = sfp_datasets.RgbDataset(
        rgb_images_dir=rgb_images_dir,
        sapflow_dir=sapflow_dir,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = sfp_datasets.RgbDataset(
        rgb_images_dir=rgb_images_dir,
        sapflow_dir=sapflow_dir,
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
        results_dir=Path('train/sapflow_predict/rgb_vgg16'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=sfp_methods.rgb_output,
        num_workers=num_workers
    )


def rgb_resnet18_train(
    rgb_images_dir: Path, 
    sapflow_dir: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8
):
    model = sfp_models.RgbResNet18Model()
    train_dataset = sfp_datasets.RgbDataset(
        rgb_images_dir=rgb_images_dir,
        sapflow_dir=sapflow_dir,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = sfp_datasets.RgbDataset(
        rgb_images_dir=rgb_images_dir,
        sapflow_dir=sapflow_dir,
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
        results_dir=Path('train/sapflow_predict/rgb_resnet18'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=sfp_methods.rgb_output,
        num_workers=num_workers
    )

def rgb_vgg16_tm_trian(
    rgb_images_dir: Path, 
    temp_moisture_filepath: Path,
    sapflow_dir: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8
):
    model = sfp_models.RgbVgg16TmModel()
    train_dataset = sfp_datasets.RgbTmDataset(
        rgb_images_dir=rgb_images_dir,
        tm_file_path=temp_moisture_filepath,
        sapflow_dir=sapflow_dir,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = sfp_datasets.RgbTmDataset(
        rgb_images_dir=rgb_images_dir,
        tm_file_path=temp_moisture_filepath,
        sapflow_dir=sapflow_dir,
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
        results_dir=Path('train/sapflow_predict/rgb_vgg16_tm'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=sfp_methods.rgb_and_tm_output,
        num_workers=num_workers
    )

def rgb_resnet18_tm_train(
    rgb_images_dir: Path, 
    temp_moisture_filepath: Path,
    sapflow_dir: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8
):
    model = sfp_models.RgbResNet18TmModel()
    train_dataset = sfp_datasets.RgbTmDataset(
        rgb_images_dir=rgb_images_dir,
        tm_file_path=temp_moisture_filepath,
        sapflow_dir=sapflow_dir,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = sfp_datasets.RgbTmDataset(
        rgb_images_dir=rgb_images_dir,
        tm_file_path=temp_moisture_filepath,
        sapflow_dir=sapflow_dir,
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
        results_dir=Path('train/sapflow_predict/rgb_resnet18_tm'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=sfp_methods.rgb_and_tm_output,
        num_workers=num_workers
    )