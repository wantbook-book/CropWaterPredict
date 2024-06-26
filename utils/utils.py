from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.nn import functional as F
def act(act_name, x):
    if act_name == 'relu':
        return F.relu(x)
    elif act_name == 'gelu':
        return F.quick_gelu(x)
    elif act_name == 'none':
        return x
    else:
        raise NotImplementedError(act_name)
    
def parse_dtype(x):
    if isinstance(x, torch.dtype):
        return x
    elif isinstance(x, str):
        if x == 'float32' or x == 'float':
            return torch.float32
        else:
            pass
        
def get_next_subdir_name(dir_path: Path)->str:
    num_dirs = [int(p.name) for p in dir_path.iterdir() if p.is_dir() and p.name.isdigit()]
    max_num = max(num_dirs) if num_dirs else 0
    return str(max_num+1)

def save_results(
    best_model_weights,
    model: nn.Module,
    train_mapes: list[float],
    val_mapes: list[float],
    train_losses: list[float],
    val_losses: list[float],
    output_dir: Path,
    val_epoches: int,
    # skip epoches for drawing loss curve
    skip_epoches: int = 5,
    save_models: bool = True
):
    train_org_len = len(train_losses)
    val_org_len = len(val_losses)
    min_val_loss = min(val_losses)
    min_val_loss_index = val_losses.index(min_val_loss)
    with open(output_dir/'train_losses.txt', 'w') as f:
        for i in range(len(train_losses)):
            f.write(f'{i+1}:{train_losses[i]}\n')
    with open(output_dir/'val_losses.txt', 'w') as f:
        for i in range(len(val_losses)):
            f.write(f'{(i+1)*val_epoches}:{val_losses[i]}\n')
        f.write(f'min_loss:{(min_val_loss_index+1)*val_epoches}:{min_val_loss}')
    
    with open(output_dir/'train_mapes.txt', 'w') as f:
        for i in range(len(train_mapes)):
            f.write(f'{i+1}:{train_mapes[i]}\n')
    with open(output_dir/'val_mapes.txt', 'w') as f:
        for i in range(len(val_mapes)):
            f.write(f'{(i+1)*val_epoches}:{val_mapes[i]}\n')
    
    train_losses = train_losses[skip_epoches:]
    val_losses = val_losses[skip_epoches//val_epoches:]
    plt.figure(figsize=(10, 6))
    plt.plot(range(skip_epoches+1, train_org_len+1), train_losses, marker='', linestyle='-', color='b')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.grid(True)
    # plt.savefig(new_dir_path/'loss_curve.png')
    # plt.clf()
    plt.plot(range(val_epoches*(skip_epoches//val_epoches+1), (val_org_len+1)*val_epoches, val_epoches), val_losses, marker='', linestyle='-', color='r')
    # plt.title('Validation Loss vs. Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Loss')
    # plt.grid(True)
    plt.savefig(output_dir/'loss_curve.png')
    if save_models:
        torch.save(model.state_dict(), output_dir/'last_weights.pth')
        torch.save(best_model_weights, output_dir/'best_weights.pth')
