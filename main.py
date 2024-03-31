from models.vgg16 import VGG16
from models.mlp import MLP
from models.encoder import TransformerEncoder
from utils.config_manager import ConfigManager
from pathlib import Path
from dataset.dataset import CropInfoDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.data.dataset import random_split
import timm
import numpy as np
import matplotlib.pyplot as plt
import time
from utils.early_stopping import EarlyStopping
config = ConfigManager()
VGG_WEIGHTS_PATH = str(Path(__file__).parent / 'model_weights/vgg16.bin')
class WaterPredictModel(nn.Module):
    def __init__(self):
        super(WaterPredictModel, self).__init__()
        vgg16_conf = config.get('vgg16')
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=vgg16_conf['output_dim'])
        self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        self.infrared_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=vgg16_conf['output_dim'])

        
        encoder_conf = config.get('T_moisture_encoder')
        input_dim, d_model, nhead, nhid, nlayers, dropout = \
            encoder_conf['input_dim'], encoder_conf['d_model'], encoder_conf['nhead'],\
                 encoder_conf['nhid'], encoder_conf['nlayers'], encoder_conf['dropout']
        self.T_moisture_transformer = TransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout
        )
        encoder_conf = config.get('sap_flow_encoder')
        input_dim, d_model, nhead, nhid, nlayers, dropout = \
            encoder_conf['input_dim'], encoder_conf['d_model'], encoder_conf['nhead'],\
                 encoder_conf['nhid'], encoder_conf['nlayers'], encoder_conf['dropout']
        self.sap_flow_transformer = TransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout
        )
        mlp_conf = config.get('mlp')
        input_dim, hidden_dim, output_dim = \
            mlp_conf['input_dim'], mlp_conf['hidden_dim'], mlp_conf['output_dim']
        self.mlp = MLP(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim)

    def get_image_transform(self, is_training=False):
        return timm.data.create_transform(**self.data_config, is_training=is_training)

    def forward(self, rgb_image, infrared_image, T_moisture_data, sap_flow_data):
        image_embd = self.rgb_vgg16(rgb_image)
        infrared_embd = self.infrared_vgg16(infrared_image)
        t_T_embd = self.T_moisture_transformer(T_moisture_data)
        sap_flow_embd = self.sap_flow_transformer(sap_flow_data)
        embd = torch.cat((image_embd, infrared_embd, t_T_embd, sap_flow_embd), dim=1)
        # print(image_embd.shape)
        # print(t_T_embd.shape)
        return self.mlp(embd)


class SoilWaterPredictModel(nn.Module):
    def __init__(self):
        super(SoilWaterPredictModel, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'])
        self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        self.mlp = MLP(
            input_size=soil_water_conf['mlp']['input_dim'], 
            hidden_size=soil_water_conf['mlp']['hidden_dim'], 
            num_classes=soil_water_conf['mlp']['output_dim']
        )

    def get_image_transform(self, is_training=False):
        return timm.data.create_transform(**self.data_config, is_training=is_training)

    def forward(self, rgb_image):
        image_embd = self.rgb_vgg16(rgb_image)
        return self.mlp(image_embd)

def evaluate_soil_water_predict_model(model, validation_dataloader, criterion, device):
    model.eval()  # 将模型设置为评估模式
    validation_losses = []

    with torch.no_grad():  # 在评估阶段不计算梯度
        for images, labels, _ in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_losses.append(loss.item())

    model.train()  # 将模型设置回训练模式
    return np.mean(validation_losses)

def train_soil_water_predict_model():
    num_epochs = 300
    batch_size = 32
    lr = 0.001
    train_ratio = 0.8
    val_epoches = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=10, verbose=True)
    model = SoilWaterPredictModel()
    model = model.to(device)
    model.train()
    src_dir = Path('./main.py').resolve().parent
    rgb_image_dir = src_dir / 'data' / 'rgb_images'
    # infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    T_moisture_data_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    # sapflow_data_file_path = src_dir / 'data' / 'series_data' / 'sapflow_data.CSV'
    labels_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'
    dataset = CropInfoDataset(
            rgb_images_directory=rgb_image_dir, 
            # infrared_images_directory=infrared_image_dir, 
            # T_moisture_data_file_path=T_moisture_data_file_path,
            # sap_flow_data_file_path=sapflow_data_file_path,
            labels_file_path=labels_file_path,
            transform=model.get_image_transform(is_training=True)
        )
    dataset_size = len(dataset)
    print('dataset_size:', dataset_size)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        for images, labels, _ in train_loader:  # 假设dataloader已准备好
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        if (epoch+1) % val_epoches == 0:  # 每10个epoch后在验证集上进行评估
            validation_loss = evaluate_soil_water_predict_model(model, val_loader, criterion, device)
            val_losses.append(validation_loss)
            print(f'After epoch {epoch+1}, Validation Loss: {validation_loss:.4f}')
            early_stopping(validation_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        
    train_results = src_dir/'train_soil_water_predict_model_results'
    train_results.mkdir(exist_ok=True)
    new_dir_path = train_results / get_next_subdir_name(train_results)
    new_dir_path.mkdir(exist_ok=False)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(new_dir_path/'loss_curve.png')
    plt.clf()
    plt.plot(range(val_epoches, num_epochs+1, val_epoches), val_losses, marker='o', linestyle='-', color='r')
    plt.title('Validation Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True)
    plt.savefig(new_dir_path/'val_loss_curve.png')
    # 保存模型
    torch.save(model.state_dict(), new_dir_path/'soil_water_predict_model.pth')
    
def get_next_subdir_name(dir_path: Path)->str:
    num_dirs = [int(p.name) for p in dir_path.iterdir() if p.is_dir() and p.name.isdigit()]
    max_num = max(num_dirs) if num_dirs else 0
    return str(max_num+1)

def train_water_predict_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WaterPredictModel()
    # print(model)
    # return
    model = model.to(device)
    model.train()
    num_epochs = 5
    dataset = CropInfoDataset('dataset/preprocess/rgb_images_list.json', 'dataset/preprocess/t_T_mo_data.json', transform=model.get_transformer(is_training=True))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for images, t_T_moistures, labels in dataloader:  # 假设dataloader已准备好
            images = images.to(device)
            t_T_moistures = t_T_moistures.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, t_T_moistures)
            print(outputs.shape)
            print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WaterPredictModel().to(device)
    dataset = CropInfoDataset('dataset/preprocess/rgb_images_list.json', 'dataset/preprocess/t_T_mo_data.json', transform=model.get_transformer(is_training=False))
    test_dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    # 示例代码
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for images, t_T_moisture, labels in test_dataloader:  # 假设test_dataloader是测试数据
            images, t_T_moisture = images.to(device), t_T_moisture.to(device)
            outputs = model(images, t_T_moisture)
            print(outputs)
            break
            # 进行预测处理，例如取最大概率的标签等

def validate_model(model, validation_dataloader, device):
    model.eval()  # 设置模型为评估模式
    total_loss_mse = 0.0
    total_loss_mae = 0.0
    total_count = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for images, t_T_data, labels in validation_dataloader:
            images = images.to(device)
            t_T_data = t_T_data.to(device)
            labels = labels.to(device)
            
            outputs = model(images, t_T_data)
            # 计算MSE和MAE损失
            loss_mse = nn.MSELoss()(outputs, labels)
            loss_mae = nn.L1Loss()(outputs, labels)
            
            total_loss_mse += loss_mse.item() * labels.size(0)
            total_loss_mae += loss_mae.item() * labels.size(0)
            total_count += labels.size(0)
    
    # 计算平均损失
    avg_loss_mse = total_loss_mse / total_count
    avg_loss_mae = total_loss_mae / total_count
    
    print(f'Validation - Average MSE: {avg_loss_mse:.4f}, Average MAE: {avg_loss_mae:.4f}')

# 示例: 如何调用validate_model函数
# 假设 validation_dataset 是你的验证数据集
# validation_dataset = # 你的验证数据集
# validation_dataloader = DataLoader(validation_dataset, batch_size=2, shuffle=False, num_workers=1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = # 你的模型实例
# model.to(device)

# validate_model(model, validation_dataloader, device)


if __name__ == '__main__':
    train_soil_water_predict_model()