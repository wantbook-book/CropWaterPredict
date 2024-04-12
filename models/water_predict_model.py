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