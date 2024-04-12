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
class SoilWaterPredictModel(nn.Module):
    def __init__(self):
        super(SoilWaterPredictModel, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')
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