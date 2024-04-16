from models.vgg16 import VGG16
from models.mlp import MLP
from models.encoder import TransformerEncoder
from models.resnet import ResNet, BasicBlock
from utils.config_manager import ConfigManager
from pathlib import Path
import torch.nn as nn
import torch
import timm
from torchvision import transforms
config = ConfigManager()
VGG_WEIGHTS_PATH = str(Path(__file__).parent.parent / 'model_weights/vgg16.bin')
RESNET18_WEIGHTS_PATH = str(Path(__file__).parent.parent / 'model_weights/resnet18.bin')
class RGB_Model(nn.Module):
    def __init__(self):
        super(RGB_Model, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')['rgb']
        # self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'], finetune=soil_water_conf['rgb_vgg16']['finetune'])
        # self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        self.rgb_resnet18 = ResNet(
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=soil_water_conf['rgb_resnet18']['output_dim']
            # num_classes=1000
        )
        # self.rgb_resnet18.fc = nn.Linear(self.rgb_resnet18.fc.in_features, soil_water_conf['rgb_resnet18']['output_dim'])
        if soil_water_conf['rgb_resnet18']['finetune']:
            state_dict = torch.load(RESNET18_WEIGHTS_PATH)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.final_mlp = MLP(
            input_size=soil_water_conf['mlp']['input_dim'], 
            hidden_size=soil_water_conf['mlp']['hidden_dim'], 
            num_classes=soil_water_conf['mlp']['output_dim']
        )

    def get_image_transform(self, is_training=False):
        # return timm.data.create_transform(**self.data_config, is_training=is_training)
        if is_training:
            _transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            _transform = transforms.Compose([
                transforms.Resize(256),            # 将图像大小调整一致
                transforms.CenterCrop(224),        # 中心裁剪以匹配模型输入
                transforms.ToTensor(),             # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理
                                    std=[0.229, 0.224, 0.225])
            ])
        return _transform  

    def forward(self, rgb_image):
        image_embd = self.rgb_resnet18(rgb_image)
        return self.final_mlp(image_embd)

class RGB_TM_Model(nn.Module):
    def __init__(self):
        super(RGB_TM_Model, self).__init__()
        soil_water_conf = config.get('soil_water_predict_model')['rgb_tm']
        # self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'], finetune=soil_water_conf['rgb_vgg16']['finetune'])
        # self.TM_encoder = TransformerEncoder(
        #     input_dim=soil_water_conf['T_moisture_encoder']['input_dim'], 
        #     d_model=soil_water_conf['T_moisture_encoder']['d_model'],
        #     nhead=soil_water_conf['T_moisture_encoder']['nhead'],
        #     nhid=soil_water_conf['T_moisture_encoder']['nhid'],
        #     nlayers=soil_water_conf['T_moisture_encoder']['nlayers'],
        #     dropout=soil_water_conf['T_moisture_encoder']['dropout']
        # )
        self.rgb_resnet18 = ResNet(
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=soil_water_conf['rgb_resnet18']['output_dim']
            # num_classes=1000
        )
        if soil_water_conf['rgb_resnet18']['finetune']:
            state_dict = torch.load(RESNET18_WEIGHTS_PATH)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.tm_mlp = MLP(
            input_size=soil_water_conf['tm_mlp']['input_dim'], 
            hidden_size=soil_water_conf['tm_mlp']['hidden_dim'], 
            num_classes=soil_water_conf['tm_mlp']['output_dim']
        )
        self.final_mlp = MLP(
            input_size=soil_water_conf['mlp']['input_dim'], 
            hidden_size=soil_water_conf['mlp']['hidden_dim'], 
            num_classes=soil_water_conf['mlp']['output_dim']
        )
        # self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        

    # def get_image_transform(self, is_training=False):
    #     return timm.data.create_transform(**self.data_config, is_training=is_training)
    def get_image_transform(self, is_training=False):
        # return timm.data.create_transform(**self.data_config, is_training=is_training)
        if is_training:
            _transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            _transform = transforms.Compose([
                transforms.Resize(256),            # 将图像大小调整一致
                transforms.CenterCrop(224),        # 中心裁剪以匹配模型输入
                transforms.ToTensor(),             # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化处理
                                    std=[0.229, 0.224, 0.225])
            ])
        return _transform  

    def forward(self, rgb_image, T_moisture):
        # rgb_image: [B, C, H, W]
        # T_moisture: [B, num_per_day, 3]
        image_embd = self.rgb_resnet18(rgb_image)
        T_moisture = T_moisture.transpose(1,2)
        # tm_embd: [B, mlp_output_dim*3]
        tm_embd = self.tm_mlp(T_moisture)
        tm_embd = tm_embd.flatten(1)
        # tm_embd = torch.sum(tm_embd, dim=1) / torch.norm(tm_embd, p=2, dim=1, keepdim=True)
        # 不确定要不要归一化
        # tm_embd = torch.sum(tm_embd, dim=1)
        # breakpoint()
        embd = torch.cat([image_embd, tm_embd], dim=1)
        return self.final_mlp(embd)