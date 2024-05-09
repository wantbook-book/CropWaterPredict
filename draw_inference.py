import models.sapflow_predict as sfp_models
import models.soil_moisture_predict as smp_models
import dataset.sapflow_predict as sfp_datasets
import dataset.soil_moisture_predict as smp_datasets
import torch
from pathlib import Path
from utils.inference import compare_models, read_labels, draw_infer_graph, cal_r2
from torch.utils.data import DataLoader
def infer_and_draw():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 12
    src_dir = Path(__file__).resolve().parent
    model1 = sfp_models.RgbVgg16TmModel()
    model1.load_state_dict(torch.load(src_dir/'train/sapflow_predict/rgb_vgg16_tm/best2/best_weights.pth', map_location=device))
    model2 = sfp_models.RgbResNet18TmModel()
    model2.load_state_dict(torch.load(src_dir/'train/sapflow_predict/rgb_resnet18_tm/best1/best_weights.pth', map_location=device))

    rgb_images_dir = src_dir / 'data' / 'rgb_images'
    segment_images_dir = src_dir / 'data' / 'segment_rgb_images'
    # infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    temp_moisture_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    soil_moisture_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'
    sapflow_rgb_images_dir = src_dir / 'data' / 'sapflow_predict_data' / 'rgb_images'
    sapflow_dir = src_dir / 'data' / 'sapflow_predict_data' / 'sapflow'

    dataset = sfp_datasets.RgbTmDataset(
        rgb_images_dir=sapflow_rgb_images_dir,
        tm_file_path=temp_moisture_file_path,
        sapflow_dir=sapflow_dir,
        transform=model1.get_image_transform(is_training=False)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    output_dir = Path(src_dir/'graphs/true_vs_predict/sapflow')
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_models(
        model1=model1, model1_use_tm=False,
        model2=model2, model2_use_tm=False,
        dataloader=dataloader,
        output_file=output_dir / 'rgb_tm_vgg16_vs_resnet18.png',
        y_label='Stem Flow (g/h)',
        device=device,
        label1='vgg16_tm',
        label2='resnet18_tm'
    )
  

def draw():
    src_dir = Path(__file__).resolve().parent
    labels = read_labels(label_file=src_dir/'graphs/true_vs_predict/sapflow/rgb_tm_vgg16_vs_resnet18.txt')
    labels = labels
    draw_infer_graph(
        labels=labels, 
        output_file=src_dir/'graphs/true_vs_predict/sapflow/rgb_vgg16_vs_resnet18.png', 
        y_label='Stem Flow (g/h)',
        label1='vgg16',
        label2='resnet18'
    )
    cal_r2(labels)
    

    

if __name__ == '__main__':
    infer_and_draw()