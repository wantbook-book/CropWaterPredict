import matplotlib.pyplot as plt
from pathlib import Path

SKIP_LOSS_LG = 20
def get_train_val_losses(result_path: Path):
    train_losses_file = result_path / 'train_losses.txt'
    val_losses_file = result_path / 'val_losses.txt'
    train_losses, train_epochs = [], []
    val_losses, val_epochs = [], []

    with open(train_losses_file, 'r') as f:
        for line in f:
            epoch, loss = line.strip().split(':')
            loss = float(loss)
            if loss > SKIP_LOSS_LG:
                continue 
            train_epochs.append(int(epoch))
            train_losses.append(float(loss))
    with open(val_losses_file, 'r') as f:
        lines = f.readlines()[:-1]
        for line in lines:
            epoch, loss = line.strip().split(':')
            loss = float(loss)
            if loss > SKIP_LOSS_LG:
                continue 
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
    ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.plot(res1_train_epochs, res1_train_losses, color='blue', linestyle='dashed', linewidth=2)
    ax.plot(res1_val_epochs, res1_val_losses, label=label1, color='blue', linestyle='solid', linewidth=3)
    ax.plot(res2_train_epochs, res2_train_losses, color='red', linestyle='dashed', linewidth=2)
    ax.plot(res2_val_epochs, res2_val_losses, label=label2, color='red', linestyle='solid', linewidth=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    plt.savefig(output_path)
    

if __name__ == '__main__':
    result1_paths = [
        Path('../train/sapflow_predict/rgb_vgg16/2best'),
        Path('../train/sapflow_predict/rgb_vgg16/2best'),
        Path('../train/sapflow_predict/rgb_resnet18/best1'),
        Path('../train/sapflow_predict/rgb_vgg16_tm/best2'),
        Path('../train/sapflow_predict/rgb_resnet18_tm/best1'),


        Path('../train/soil_moisture_predict/rgb_vgg16/11'),
        Path('../train/soil_moisture_predict/rgb_vgg16/11'),
        Path('../train/soil_moisture_predict/rgb_resnet18/6'),
        Path('../train/soil_moisture_predict/rgb_vgg16_tm/12'),
        Path('../train/soil_moisture_predict/rgb_vgg16/11'),
        Path('../train/soil_moisture_predict/rgb_vgg16_tm/12'),
    ]
    result2_paths = [
        Path('../train/sapflow_predict/rgb_resnet18/best1'),
        Path('../train/sapflow_predict/rgb_vgg16_tm/best2'),
        Path('../train/sapflow_predict/rgb_resnet18_tm/best1'),
        Path('../train/sapflow_predict/rgb_resnet18_tm/best1'),
        Path('../train/sapflow_predict/rgb_resnet18_tm_transformer/5'),

        Path('../train/soil_moisture_predict/rgb_resnet18/6'),
        Path('../train/soil_moisture_predict/rgb_vgg16_tm/12'),
        Path('../train/soil_moisture_predict/rgb_resnet18_tm/4'),
        Path('../train/soil_moisture_predict/rgb_resnet18_tm/4'),
        Path('../train/soil_moisture_predict/segment_rgb_vgg16/6'),
        Path('../train/soil_moisture_predict/segment_rgb_vgg16_tm/6'),
    ]
    output_paths = [
        Path('../graphs/sapflow_predict/only_rgb_resnet18_vs_vgg16.png'),
        Path('../graphs/sapflow_predict/only_rgb_vgg16_vs_rgb_tm_vgg16.png'),
        Path('../graphs/sapflow_predict/only_rgb_resnet18_vs_rgb_tm_resnet18.png'),
        Path('../graphs/sapflow_predict/rgb_tm_vgg16_tm_vs_rgb_tm_resnet18.png'),
        Path('../graphs/sapflow_predict/rgb_tm_resnet18_vs_rgb_tm_resnet18_transformer.png'),

        Path('../graphs/soil_moisture_predict/only_rgb_resnet18_vs_vgg16.png'),
        Path('../graphs/soil_moisture_predict/only_rgb_vgg16_vs_rgb_tm_vgg16.png'),
        Path('../graphs/soil_moisture_predict/only_rgb_resnet18_vs_rgb_tm_resnet18.png'),
        Path('../graphs/soil_moisture_predict/rgb_tm_vgg16_tm_vs_rgb_tm_resnet18.png'),
        Path('../graphs/soil_moisture_predict/rgb_vgg16_vs_segment_rgb_vgg16.png'),
        Path('../graphs/soil_moisture_predict/rgb_tm_vgg16_vs_segment_rgb_tm_vgg16.png'),
    ]

    labels = [
        ('vgg16', 'resnet18'),
        ('vgg16', 'vgg16_tm'),
        ('resnet18', 'resnet18_tm'),
        ('vgg16_tm', 'resnet18_tm'),
        ('resnet18_mlp', 'resnet18_transformer'),

        ('vgg16', 'resnet18'),
        ('vgg16', 'vgg16_tm'),
        ('resnet18', 'resnet18_tm'),
        ('vgg16_tm', 'resnet18_tm'),
        ('vgg16', 'segment_rgb_vgg16'),
        ('vgg16_tm', 'segment_rgb_vgg16_tm'),
    ]
    for result1_path, result2_path, output_path, label in zip(result1_paths, result2_paths, output_paths, labels):
        compare_train_val_curves(result1_path, result2_path, label[0], label[1], output_path)


