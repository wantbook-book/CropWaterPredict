import timm
import torch
import torch.nn as nn
class VGG16(nn.Module):
    def __init__(self, weights_path, num_classes):
        super(VGG16, self).__init__()
        self.model = timm.create_model(
            'vgg16.tv_in1k',
            pretrained=False,
            num_classes=0,  # remove classifier nn.Linear
        )
        # weights_path = 'models/vgg16.bin'  # 替换为你的权重文件路径
        # 容忍不匹配的键，因为我们删除了分类器
        self.model.load_state_dict(torch.load(weights_path), strict=False)
        # 替换分类器
        # num_features:4096
        self.fc = nn.Linear(self.model.num_features, num_classes) 
    def forward(self, x):
        x = self.model(x) 
        return self.fc(x)