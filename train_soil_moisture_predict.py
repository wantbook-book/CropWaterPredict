from utils.train import train
import torch
import models.soil_moisture_predict as smp_models
import dataset.soil_moisture_predict as smp_datasets
import methods.soil_moisture_predict as smp_methods
from torch.utils.data import Subset
from pathlib import Path
import numpy as np
TRAIN_RATIO = 0.8
def rgb_vgg16_train(
    rgb_images_dir: Path, 
    soil_moisture_filepath: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8,
    save_models: bool = True,
    results_dir: Path = Path('train/soil_moisture_predict/rgb_vgg16')
):
    model = smp_models.RgbVgg16Model()
    train_dataset = smp_datasets.rgb_dataset(
        rgb_images_dir=rgb_images_dir,
        labels_file_path=soil_moisture_filepath,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = smp_datasets.rgb_dataset(
        rgb_images_dir=rgb_images_dir,
        labels_file_path=soil_moisture_filepath,
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
        results_dir=results_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=smp_methods.rgb_output,
        num_workers=num_workers,
        save_models=save_models
    )


def rgb_resnet18_train(
    rgb_images_dir: Path, 
    soil_moisture_filepath: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8,
    save_models: bool = True
):
    model = smp_models.RgbResNet18Model()
    train_dataset = smp_datasets.rgb_dataset(
        rgb_images_dir=rgb_images_dir,
        labels_file_path=soil_moisture_filepath,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = smp_datasets.rgb_dataset(
        rgb_images_dir=rgb_images_dir,
        labels_file_path=soil_moisture_filepath,
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
        results_dir=Path('train/soil_moisture_predict/rgb_resnet18'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=smp_methods.rgb_output,
        num_workers=num_workers,
        save_models=save_models
    )

def rgb_vgg16_tm_trian(
    rgb_images_dir: Path, 
    temp_moisture_filepath: Path,
    soil_moisture_filepath: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8,
    save_models: bool = True,
    results_dir=Path('train/soil_moisture_predict/rgb_vgg16_tm')
):
    model = smp_models.RgbVgg16TmModel()
    train_dataset = smp_datasets.rgb_tm_dataset(
        rgb_images_dir=rgb_images_dir,
        temp_moisture_filepath=temp_moisture_filepath,
        soil_moisture_filepath=soil_moisture_filepath,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = smp_datasets.rgb_tm_dataset(
        rgb_images_dir=rgb_images_dir,
        temp_moisture_filepath=temp_moisture_filepath,
        soil_moisture_filepath=soil_moisture_filepath,
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
        results_dir=results_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=smp_methods.rgb_and_tm_output,
        num_workers=num_workers,
        save_models=save_models
    )

def rgb_resnet18_tm_train(
    rgb_images_dir: Path, 
    temp_moisture_filepath: Path,
    soil_moisture_filepath: Path,
    device: torch.device,
    num_epochs: int = 300,
    batch_size: int = 32,
    lr: float = 0.01,
    val_epochs: int = 2,
    patience: int = 10,
    num_workers: int = 8,
    save_models: bool = True
):
    model = smp_models.RgbResNet18TmModel()
    train_dataset = smp_datasets.rgb_tm_dataset(
        rgb_images_dir=rgb_images_dir,
        temp_moisture_filepath=temp_moisture_filepath,
        soil_moisture_filepath=soil_moisture_filepath,
        transform=model.get_image_transform(is_training=True)
    )
    validation_dataset = smp_datasets.rgb_tm_dataset(
        rgb_images_dir=rgb_images_dir,
        temp_moisture_filepath=temp_moisture_filepath,
        soil_moisture_filepath=soil_moisture_filepath,
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
        results_dir=Path('train/soil_moisture_predict/rgb_resnet18_tm'),
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        val_epoches=val_epochs,
        patience=patience,
        output_func=smp_methods.rgb_and_tm_output,
        num_workers=num_workers,
        save_models=save_models
    )