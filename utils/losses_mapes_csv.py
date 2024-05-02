import pandas as pd
from pathlib import Path

def write_losses_mapes_to_csv(losses_mapes_dir: Path):
    csv_filepath = losses_mapes_dir.parent / f'{losses_mapes_dir.name}_losses_mapes.csv'
    rows = []
    for _dir in losses_mapes_dir.iterdir():
        row = {
            'dir':'0',
            'epoch':0,
            'train_losses': 0,
            'val_losses': 0,
            'train_mapes': 0,
            'val_mapes': 0,
        }
        if _dir.is_dir():
            row['dir'] = str(_dir)
            val_losses_file = _dir / 'val_losses.txt'
            train_losses_file = _dir / 'train_losses.txt'
            val_mapes_file = _dir / 'val_mapes.txt'
            train_mapes_file = _dir / 'train_mapes.txt'
            if val_losses_file.exists():
                with open(val_losses_file, 'r') as val_losses_f:
                    val_losses_lines = val_losses_f.readlines() 
                    _, best_epoch, best_val_loss = val_losses_lines[-1].split(':')
                    row['epoch'] = int(best_epoch)
                    row['val_losses'] = float(best_val_loss)

            if train_losses_file.exists():
                with open(train_losses_file, 'r') as train_losses_f:
                    train_losses_lines = train_losses_f.readlines()
                    for line in train_losses_lines:
                        epoch, train_loss = line.split(':')
                        if epoch == best_epoch:
                            row['train_losses'] = float(train_loss)
            
            if val_mapes_file.exists():
                with open(val_mapes_file, 'r') as val_mapes_f:
                    val_mapes_lines = val_mapes_f.readlines()
                    for line in val_mapes_lines:
                        epoch, val_mape = line.split(':')
                        if epoch == best_epoch:
                            row['val_mapes'] = float(val_mape)
                            
            if train_mapes_file.exists():
                with open(train_mapes_file, 'r') as train_mapes_f:
                    train_mapes_lines = train_mapes_f.readlines()
                    for line in train_mapes_lines:
                        epoch, train_mape = line.split(':')
                        if epoch == best_epoch:
                            row['train_mapes'] = float(train_mape)
            rows.append(row)
                
        df = pd.DataFrame(rows)
        df.to_csv(csv_filepath, index=False)


if __name__ == '__main__':
    dirs = [
        # Path('../train/sapflow_predict/rgb_resnet18'),
        # Path('../train/sapflow_predict/rgb_resnet18_tm'),
        # Path('../train/sapflow_predict/rgb_vgg16'),
        # Path('../train/sapflow_predict/rgb_vgg16_tm'),
        # Path('../train/soil_moisture_predict/rgb_resnet18'),
        # Path('../train/soil_moisture_predict/rgb_resnet18_tm'),
        # Path('../train/soil_moisture_predict/rgb_vgg16'),
        # Path('../train/soil_moisture_predict/rgb_vgg16_tm'),
        Path('../train/soil_moisture_predict/segment_rgb_vgg16'),
        Path('../train/soil_moisture_predict/segment_rgb_vgg16_tm'),
        Path('../train/sapflow_predict/rgb_resnet18_tm_transformer'),
    ]
    for _dir in dirs:
        write_losses_mapes_to_csv(_dir)