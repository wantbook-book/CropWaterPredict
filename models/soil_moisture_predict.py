from models.vgg16 import VGG16
from models.mlp import MLP
from models.encoder import TransformerEncoder
from utils.config_manager import ConfigManager
from pathlib import Path
import torch.nn as nn
import torch
import timm
config = ConfigManager()
VGG_WEIGHTS_PATH = str(Path(__file__).parent.parent / 'model_weights/vgg16.bin')
class RGB_Model(nn.Module):
    def __init__(self):
        super(RGB_Model, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')['rgb']
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'], finetune=soil_water_conf['rgb_vgg16']['finetune'])
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

class RGB_TM_Model(nn.Module):
    def __init__(self):
        super(RGB_TM_Model, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')['rgb_tm']
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'], finetune=soil_water_conf['rgb_vgg16']['finetune'])
        self.TM_encoder = TransformerEncoder(
            input_dim=soil_water_conf['T_moisture_encoder']['input_dim'], 
            d_model=soil_water_conf['T_moisture_encoder']['d_model'],
            nhead=soil_water_conf['T_moisture_encoder']['nhead'],
            nhid=soil_water_conf['T_moisture_encoder']['nhid'],
            nlayers=soil_water_conf['T_moisture_encoder']['nlayers'],
            dropout=soil_water_conf['T_moisture_encoder']['dropout']
        )
        self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        self.mlp = MLP(
            input_size=soil_water_conf['mlp']['input_dim'], 
            hidden_size=soil_water_conf['mlp']['hidden_dim'], 
            num_classes=soil_water_conf['mlp']['output_dim']
        )

    def get_image_transform(self, is_training=False):
        return timm.data.create_transform(**self.data_config, is_training=is_training)

    def forward(self, rgb_image, T_moisture):
        image_embd = self.rgb_vgg16(rgb_image)
        T_moisture_embd = self.TM_encoder(T_moisture)
        embd = torch.cat([image_embd, T_moisture_embd], dim=1)
        return self.mlp(embd)