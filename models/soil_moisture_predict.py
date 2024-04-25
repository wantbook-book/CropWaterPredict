import functools
from utils.utils import act 
from models.vgg16 import VGG16
from models.mlp import MLP
from models.encoder import TransformerEncoder
from models.resnet import ResNet, BasicBlock
from utils.config_manager import ConfigManager
from pathlib import Path
import json
import torch.nn as nn
import torch
import timm
from torchvision import transforms
SRC_PATH = Path(__file__).resolve().parent.parent
VGG_WEIGHTS_PATH = str(SRC_PATH / 'model_weights/vgg16.bin')
RESNET18_WEIGHTS_PATH = str(SRC_PATH / 'model_weights/resnet18.bin')
class RgbResNet18Model(nn.Module):
    def __init__(self):
        super(RgbResNet18Model, self).__init__()
        config = ConfigManager(SRC_PATH / 'conf/soil_moisture_predict.json')['rgb']
        self.net_settings = {}
        self.net_settings['rgb_resnet18'] = {
            'output_dim': config['rgb_resnet18']['output_dim'],
            'finetune': config['rgb_resnet18']['finetune']
        }
        self.net_settings['mlp'] = {
            'input_dim': config['mlp']['input_dim'], 
            'hidden_dim': config['mlp']['hidden_dim'], 
            'output_dim': config['mlp']['output_dim'],
            'nhidlayer': config['mlp']['nhidlayer'],
            'hidactive': config['mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
        # self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=soil_water_conf['rgb_vgg16']['output_dim'], finetune=soil_water_conf['rgb_vgg16']['finetune'])
        # self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        self.rgb_resnet18 = ResNet(
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=config['rgb_resnet18']['output_dim']
            # num_classes=1000
        )

        # self.rgb_resnet18.fc = nn.Linear(self.rgb_resnet18.fc.in_features, soil_water_conf['rgb_resnet18']['output_dim'])
        if config['rgb_resnet18']['finetune']:
            state_dict = torch.load(RESNET18_WEIGHTS_PATH)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.final_mlp = MLP(
            insize=config['mlp']['input_dim'], 
            hidsize=config['mlp']['hidden_dim'], 
            outsize=config['mlp']['output_dim'],
            nhidlayer=config['mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['mlp']['hidactive']),
        )


    def get_image_transform(self, is_training=False):
        # return timm.data.create_transform(**self.data_config, is_training=is_training)
        if is_training:
            _transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.CenterCrop(224),
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
    
    def output_net_settings(self, output_dir:Path):
        with open(output_dir/'net_settings.json', 'w') as f:
            json.dump(self.net_settings, f, indent=4)


class RgbVgg16Model(nn.Module):
    def __init__(self):
        super(RgbVgg16Model, self).__init__()
        config = ConfigManager(SRC_PATH / 'conf/soil_moisture_predict.json')['rgb']
        self.net_settings = {}
        self.net_settings['rgb_vgg16'] = {
            'output_dim': config['rgb_vgg16']['output_dim'],
            'finetune': config['rgb_vgg16']['finetune']
        }
        self.net_settings['mlp'] = {
            'input_dim': config['mlp']['input_dim'], 
            'hidden_dim': config['mlp']['hidden_dim'], 
            'output_dim': config['mlp']['output_dim'],
            'nhidlayer': config['mlp']['nhidlayer'],
            'hidactive': config['mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=config['rgb_vgg16']['output_dim'], finetune=config['rgb_vgg16']['finetune'])
        # self.data_config = timm.data.resolve_model_data_config(self.rgb_vgg16.model)
        # self.rgb_resnet18 = ResNet(
        #     block=BasicBlock,
        #     layers=[2,2,2,2],
        #     num_classes=soil_water_conf['rgb_resnet18']['output_dim']
        #     # num_classes=1000
        # )
        # self.rgb_resnet18.fc = nn.Linear(self.rgb_resnet18.fc.in_features, soil_water_conf['rgb_resnet18']['output_dim'])
        # if soil_water_conf['rgb_resnet18']['finetune']:
        #     state_dict = torch.load(RESNET18_WEIGHTS_PATH)
        #     state_dict.pop('fc.weight', None)
        #     state_dict.pop('fc.bias', None)
        #     self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.final_mlp = MLP(
            insize=config['mlp']['input_dim'], 
            hidsize=config['mlp']['hidden_dim'], 
            outsize=config['mlp']['output_dim'],
            nhidlayer=config['mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['mlp']['hidactive']),
        )

    def get_image_transform(self, is_training=False):
        # return timm.data.create_transform(**self.data_config, is_training=is_training)
        if is_training:
            _transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.RandomCrop(224),
                transforms.CenterCrop(224),
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
        image_embd = self.rgb_vgg16(rgb_image)
        return self.final_mlp(image_embd)
    
    def output_net_settings(self, output_dir:Path):
        with open(output_dir/'net_settings.json', 'w') as f:
            json.dump(self.net_settings, f, indent=4)

class RgbResNet18TmModel(nn.Module):
    def __init__(self):
        super(RgbResNet18TmModel, self).__init__()
        config = ConfigManager(SRC_PATH / 'conf/soil_moisture_predict.json')['rgb_tm']
        self.net_settings = {}
        self.net_settings['rgb_resnet18'] = {
            'output_dim': config['rgb_resnet18']['output_dim'],
            'finetune': config['rgb_resnet18']['finetune']
        }
        self.net_settings['tm_mlp'] = {
            'input_dim': config['tm_mlp']['input_dim'], 
            'hidden_dim': config['tm_mlp']['hidden_dim'], 
            'output_dim': config['tm_mlp']['output_dim'],
            'nhidlayer': config['tm_mlp']['nhidlayer'],
            'hidactive': config['tm_mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
        
        self.net_settings['mlp'] = {
            'input_dim': config['mlp']['input_dim'], 
            'hidden_dim': config['mlp']['hidden_dim'], 
            'output_dim': config['mlp']['output_dim'],
            'nhidlayer': config['mlp']['nhidlayer'],
            'hidactive': config['mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
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
            num_classes=config['rgb_resnet18']['output_dim']
            # num_classes=1000
        )
        if config['rgb_resnet18']['finetune']:
            state_dict = torch.load(RESNET18_WEIGHTS_PATH)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.tm_mlp = MLP(
            insize=config['tm_mlp']['input_dim'], 
            hidsize=config['tm_mlp']['hidden_dim'], 
            outsize=config['tm_mlp']['output_dim'],
            nhidlayer=config['tm_mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['tm_mlp']['hidactive']),
        )
        self.final_mlp = MLP(
            insize=config['mlp']['input_dim'], 
            hidsize=config['mlp']['hidden_dim'], 
            outsize=config['mlp']['output_dim'],
            nhidlayer=config['mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['mlp']['hidactive']),
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
    
    def output_net_settings(self, output_dir:Path):
        with open(output_dir/'net_settings.json', 'w') as f:
            json.dump(self.net_settings, f, indent=4)


class RgbVgg16TmModel(nn.Module):
    def __init__(self):
        super(RgbVgg16TmModel, self).__init__()
        config = ConfigManager(SRC_PATH / 'conf/soil_moisture_predict.json')['rgb_tm']
        self.net_settings = {}
        self.net_settings['rgb_vgg16'] = {
            'output_dim': config['rgb_vgg16']['output_dim'],
            'finetune': config['rgb_vgg16']['finetune']
        }
        self.net_settings['tm_mlp'] = {
            'input_dim': config['tm_mlp']['input_dim'], 
            'hidden_dim': config['tm_mlp']['hidden_dim'], 
            'output_dim': config['tm_mlp']['output_dim'],
            'nhidlayer': config['tm_mlp']['nhidlayer'],
            'hidactive': config['tm_mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
        
        self.net_settings['mlp'] = {
            'input_dim': config['mlp']['input_dim'], 
            'hidden_dim': config['mlp']['hidden_dim'], 
            'output_dim': config['mlp']['output_dim'],
            'nhidlayer': config['mlp']['nhidlayer'],
            'hidactive': config['mlp']['hidactive'],
            'norm': config['tm_mlp']['norm'],
        }
        self.rgb_vgg16 = VGG16(VGG_WEIGHTS_PATH, num_classes=config['rgb_vgg16']['output_dim'], finetune=config['rgb_vgg16']['finetune'])
        # self.TM_encoder = TransformerEncoder(
        #     input_dim=soil_water_conf['T_moisture_encoder']['input_dim'], 
        #     d_model=soil_water_conf['T_moisture_encoder']['d_model'],
        #     nhead=soil_water_conf['T_moisture_encoder']['nhead'],
        #     nhid=soil_water_conf['T_moisture_encoder']['nhid'],
        #     nlayers=soil_water_conf['T_moisture_encoder']['nlayers'],
        #     dropout=soil_water_conf['T_moisture_encoder']['dropout']
        # )
        # self.rgb_resnet18 = ResNet(
        #     block=BasicBlock,
        #     layers=[2,2,2,2],
        #     num_classes=soil_water_conf['rgb_resnet18']['output_dim']
        #     # num_classes=1000
        # )
        # if soil_water_conf['rgb_resnet18']['finetune']:
        #     state_dict = torch.load(RESNET18_WEIGHTS_PATH)
        #     state_dict.pop('fc.weight', None)
        #     state_dict.pop('fc.bias', None)
        #     self.rgb_resnet18.load_state_dict(state_dict, strict=False)
        self.tm_mlp = MLP(
            insize=config['tm_mlp']['input_dim'], 
            hidsize=config['tm_mlp']['hidden_dim'], 
            outsize=config['tm_mlp']['output_dim'],
            nhidlayer=config['tm_mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['tm_mlp']['hidactive']),
        )
        self.final_mlp = MLP(
            insize=config['mlp']['input_dim'], 
            hidsize=config['mlp']['hidden_dim'], 
            outsize=config['mlp']['output_dim'],
            nhidlayer=config['mlp']['nhidlayer'],
            norm=config['tm_mlp']['norm'],
            hidactive=functools.partial(act, config['mlp']['hidactive']),
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
        # image_embd = self.rgb_resnet18(rgb_image)
        image_embd = self.rgb_vgg16(rgb_image)
        T_moisture = T_moisture.transpose(1,2)
        # tm_embd: [B, mlp_output_dim*3]
        tm_embd = self.tm_mlp(T_moisture)
        tm_embd = tm_embd.flatten(1)
        breakpoint()
        # tm_embd = torch.sum(tm_embd, dim=1) / torch.norm(tm_embd, p=2, dim=1, keepdim=True)
        # 不确定要不要归一化
        # tm_embd = torch.sum(tm_embd, dim=1)
        # breakpoint()
        embd = torch.cat([image_embd, tm_embd], dim=1)
        return self.final_mlp(embd)
    
    def output_net_settings(self, output_dir:Path):
        with open(output_dir/'net_settings.json', 'w') as f:
            json.dump(self.net_settings, f, indent=4)