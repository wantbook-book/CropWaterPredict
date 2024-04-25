import torch
from torch import nn
from utils.utils import parse_dtype
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(MLP, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_classes = num_classes
#         self.layer1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(hidden_size, hidden_size) 
#         self.output_layer = nn.Linear(hidden_size, num_classes) 
    
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.relu(out)
#         out = self.layer2(out)
#         out = self.relu(out)
#         out = self.output_layer(out)
#         return out

def NormedLinear(*args, scale=1.0, norm:bool=True, dtype=torch.float32, **kwargs):
    dtype = parse_dtype(dtype)
    if dtype == torch.float32:
        out = nn.Linear(*args, **kwargs)
    else:
        raise ValueError(dtype)
    if norm:
        out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    # if kwargs.get('bias', True):
    #     out.bias.data *= 0
    return out

class MLP(nn.Module):
    def __init__(self, insize, nhidlayer, outsize, hidsize, hidactive, dtype=torch.float32, norm:bool=True):
        super().__init__()
        self.insize = insize
        self.nhidlayer = nhidlayer
        self.outsize = outsize
        in_sizes = [insize] + [hidsize]*nhidlayer
        out_sizes = [hidsize]*nhidlayer + [outsize]
        self.layers = nn.ModuleList(
            [NormedLinear(insize, outsize, dtype=dtype, norm=norm) for (insize, outsize) in zip(in_sizes, out_sizes)]
        )
        self.hidactive = hidactive
    def forward(self, x):
        *hidlayers, final_layer = self.layers
        for layer in hidlayers:
            x = layer(x)
            x = self.hidactive(x)
        x = final_layer(x)
        return x
    
    @property
    def output_shape(self):
        return (self.outsize,)

