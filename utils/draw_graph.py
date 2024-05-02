import matplotlib.pyplot as plt
from pathlib import Path

def get_train_val_losses(result_path: Path):
    train_losses_file = result_path / 'train_losses.txt'
    val_losses_file = result_path / 'val_losses.txt'
    train_losses, train_epochs = [], []
    val_losses, val_epochs = [], []

    with open(train_losses_file, 'r') as f:
        for line in f:
            epoch, loss = line.strip().split(':')
            train_epochs.append(int(epoch))
            train_losses.append(float(loss))
    with open(val_losses_file, 'r') as f:
        lines = f.readlines()[:-1]
        for line in lines:
            epoch, loss = line.strip().split(':')
            val_epochs.append(int(epoch))
            val_losses.append(float(loss))
    
    return train_losses, train_epochs, val_losses, val_epochs

def compare_train_val_curves(result1_path: Path, result2_path: Path, label1: str, label2: str, output_path: Path):
    res1_train_losses_file = result1_path / 'train_losses.txt'
    res1_val_losses_file = result1_path / 'val_losses.txt'
    res1_train_losses, res1_train_epochs, res1_val_losses, res1_val_epochs = get_train_val_losses(result1_path)
    
    res2_train_losses_file = result2_path / 'train_losses.txt'
    res2_val_losses_file = result2_path / 'val_losses.txt'
    res2_train_losses, res2_train_epochs, res2_val_losses, res2_val_epochs = get_train_val_losses(result2_path)
    fig, ax = plt.subplots()
    ax.plot(res1_train_epochs, res1_train_losses, label=label1, color='blue', linestyle='dashed', linewidth=2)
    ax.plot(res1_val_epochs, res1_val_losses, label=label1, color='blue', linestyle='solid', linewidth=4)
    ax.plot(res2_train_epochs, res2_train_losses, label=label2, color='red', linestyle='dashed', linewidth=2)
    ax.plot(res2_val_epochs, res2_val_losses, label=label2, color='red', linestyle='solid', linewidth=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    plt.savefig(output_path)
    

if __name__ == '__main__':
    result1_path = Path('../train/sapflow_predict/rgb_vgg16/2best')
    result2_path = Path('../train/sapflow_predict/rgb_resnet18/best1')
    output_path = Path('../graphs/sapflow_predict/only_rgb_resnet18_vs_vgg16.png')
    compare_train_val_curves(result1_path, result2_path, 'vgg16', 'resnet18', output_path)


