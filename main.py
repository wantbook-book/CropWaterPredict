from models.vgg16 import VGG16
from models.mlp import MLP
from models.encoder import TransformerEncoder
from utils.config_manager import ConfigManager
from pathlib import Path
from dataset.dataset import CropInfoDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import timm
config = ConfigManager()
VGG_WEIGHTS_PATH = str(Path(__file__).parent / 'model_weights/vgg16.bin')
class WaterPredictModel(nn.Module):
    def __init__(self):
        super(WaterPredictModel, self).__init__()
        vgg16_conf = config.get('vgg16')
        self.vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=vgg16_conf['output_dim'])
        self.data_config = timm.data.resolve_model_data_config(self.vgg16.model)
        
        encoder_conf = config.get('encoder')
        input_dim, d_model, nhead, nhid, nlayers, dropout = \
            encoder_conf['input_dim'], encoder_conf['d_model'], encoder_conf['nhead'],\
                 encoder_conf['nhid'], encoder_conf['nlayers'], encoder_conf['dropout']
        self.transformer = TransformerEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout
        )
        mlp_conf = config.get('mlp')
        input_dim, hidden_dim, output_dim = \
            mlp_conf['input_dim'], mlp_conf['hidden_dim'], mlp_conf['output_dim']
        self.mlp = MLP(input_size=input_dim, hidden_size=hidden_dim, num_classes=output_dim)
    def get_transformer(self, is_training=False):
        return timm.data.create_transform(**self.data_config, is_training=is_training)
    def forward(self, image, t_T_data):
        image_embd = self.vgg16(image)
        t_T_embd = self.transformer(t_T_data)
        # print(image_embd.shape)
        # print(t_T_embd.shape)
        embd = torch.cat((image_embd, t_T_embd), dim=1)
        return self.mlp(embd)


def train():
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
    inference()