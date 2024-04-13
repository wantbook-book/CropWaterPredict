from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
def get_next_subdir_name(dir_path: Path)->str:
    num_dirs = [int(p.name) for p in dir_path.iterdir() if p.is_dir() and p.name.isdigit()]
    max_num = max(num_dirs) if num_dirs else 0
    return str(max_num+1)

def save_results(
    model: nn.Module,
    train_losses: list[float],
    val_losses: list[float],
    output_dir: Path,
    val_epoches: int,
    # skip epoches for drawing loss curve
    skip_epoches: int = 5,
):
    train_org_len = len(train_losses)
    val_org_len = len(val_losses)
    with open(output_dir/'train_losses.txt', 'w') as f:
        for i in range(len(train_losses)):
            f.write(f'{i+1}: {train_losses[i]}\n')
    with open(output_dir/'val_losses.txt', 'w') as f:
        for i in range(len(val_losses)):
            f.write(f'{(i+1)*val_epoches}: {val_losses[i]}\n')
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
    torch.save(model.state_dict(), output_dir/'weights.pth')