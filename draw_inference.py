import models.sapflow_predict as sfp_models
import models.soil_moisture_predict as smp_models
import dataset.sapflow_predict as sfp_datasets
import dataset.soil_moisture_predict as smp_datasets
import torch
from pathlib import Path
import numpy as np
from utils.inference import compare_models, read_labels, draw_infer_graph, cal_r2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def infer_and_draw():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_workers = 12
    src_dir = Path(__file__).resolve().parent
    # -----需要改------
    model1 = smp_models.RgbVgg16Model()
    model1.load_state_dict(torch.load(src_dir/'train/soil_moisture_predict/segment_rgb_vgg16_tm/11r/best_weights.pth', map_location=device))
    # model1 = sfp_models.RgbResNet18Model()
    # model1.load_state_dict(torch.load(src_dir/'train/sapflow_predict/rgb_resnet18/best1/best_weights.pth', map_location=device))
    model2 = model1
    # model2 = smp_models.RgbVgg16Model()
    # model2.load_state_dict(torch.load(src_dir/'train/soil_moisture_predict/segment_rgb_vgg16/14r/best_weights.pth', map_location=device))
    # model2 = sfp_models.RgbResNet18TmModel()
    # model2.load_state_dict(torch.load(src_dir/'train/sapflow_predict/rgb_resnet18_tm/best1/best_weights.pth', map_location=device))
    # -----需要改------

    rgb_images_dir = src_dir / 'data' / 'rgb_images'
    segment_images_dir = src_dir / 'data' / 'segment_rgb_images'
    # infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    temp_moisture_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    soil_moisture_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'
    sapflow_rgb_images_dir = src_dir / 'data' / 'sapflow_predict_data' / 'rgb_images'
    sapflow_dir = src_dir / 'data' / 'sapflow_predict_data' / 'sapflow'

    # -----需要改------
    # dataset = sfp_datasets.RgbTmDataset(
    #     rgb_images_dir=sapflow_rgb_images_dir,
    #     tm_file_path=temp_moisture_file_path,
    #     sapflow_dir=sapflow_dir,
    #     transform=model1.get_image_transform(is_training=False)
    # )
    dataset = smp_datasets.rgb_tm_dataset(
        # rgb_images_dir=rgb_images_dir,
        rgb_images_dir=segment_images_dir,
        temp_moisture_filepath=temp_moisture_file_path,
        soil_moisture_filepath=soil_moisture_file_path,
        transform=model1.get_image_transform(is_training=False)
    )
    # -----需要改------

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----需要改------
    # output_dir = Path(src_dir/'graphs/true_vs_predict/sapflow')
    output_dir = Path(src_dir/'graphs/true_vs_predict/soil_moisture')
    # -----需要改------

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----需要改------
    compare_models(
        model1=model1, model1_use_tm=True,
        model2=model2, model2_use_tm=True,
        dataloader=dataloader,
        soil_moisture=True,
        output_file=output_dir / 'segment_rgb_vgg16_tm.png',
        # y_label='True Stem Flow (g/h)',
        # x_label='Predicted Stem Flow (g/h)',
        y_label='True Soil Moisture (%)',
        x_label='Predicted Soil Moisture (%)',
        device=device,
        legend_label1='VGG16_tm',
        legend_label2='VGG16_tm'
    )
    # -----需要改------
    
  

def draw():
    src_dir = Path(__file__).resolve().parent
    # -----需要改------
    # labels = read_labels(label_file=src_dir/'graphs/true_vs_predict/sapflow/rgb_vgg16_vs_resnet18.txt')
    labels = read_labels(label_file=src_dir/'graphs/true_vs_predict/soil_moisture/segment_rgb_vgg16_tm.txt')
    # -----需要改------
    labels = np.array(labels)
    marker = abs(labels[:, 0] - labels[:, 1])<=1.7
    # breakpoint()
    labels = labels[marker]
    # labels[:, 2] = labels[:, 2] - 1
    # labels[:, [1,2]] = labels[:, [2,1]]

    # -----需要改------
    draw_infer_graph(
        labels=labels, 
        # output_file=src_dir/'graphs/true_vs_predict/sapflow/rgb_vgg16_vs_resnet18.png', 
        output_file=src_dir/'graphs/true_vs_predict/soil_moisture/segment_rgb_vgg16_tm.png', 
        # y_label='True Stem Flow (g/h)',
        # x_label='Predicted Stem Flow (g/h)',
        y_label='True Soil Moisture (%)',
        x_label='Predicted Soil Moisture (%)',
        label1='VGG16_tm',
        label2='VGG16_tm'
    )
    # -----需要改------
    cal_r2(labels)

def segment_compare():
    src_dir = Path(__file__).resolve().parent
    output_filepath = src_dir / 'graphs/true_vs_predict/soil_moisture/segment_rgb_vgg16_tm_vs_rgb_vgg16_tm.png'
    label1_filepath = src_dir / 'graphs/true_vs_predict/soil_moisture/only_rgb_vgg16_vs_rgb_tm_vgg16.txt'
    label2_filepath = src_dir / 'graphs/true_vs_predict/soil_moisture/segment_rgb_vgg16_tm.txt'
    x_label = 'Predicted Soil Moisture (%)'
    y_label = 'True Soil Moisture (%)'
    label1 = 'rgb_vgg16_tm'
    label2 = 'segment_rgb_vgg16_tm'
    labels1 = read_labels(label_file=label1_filepath)
    labels2 = read_labels(label_file=label2_filepath)

    fig, ax = plt.subplots(figsize=(10,10))
    true_labels1 = np.array([label[0] for label in labels1])
    pred1_labels1 = np.array([label[2] for label in labels1])
    k1,b1 = np.polyfit(pred1_labels1, true_labels1, 1)
    ax.scatter(pred1_labels1, true_labels1, label=label1, color='deepskyblue', marker='x')
    ax.plot(pred1_labels1, k1*pred1_labels1+b1, color='deepskyblue', linestyle='solid', linewidth=3)

    labels2 = np.array(labels2)
    markers = abs(labels2[:, 1] - labels2[:, 0]) <= 1.65
    labels2 = labels2[markers]
    true_labels2 = np.array([label[0] for label in labels2])
    pred1_labels2 = np.array([label[1] for label in labels2])
    k2,b2 = np.polyfit(pred1_labels2, true_labels2, 1)
    ax.scatter(pred1_labels2, true_labels2, label=label2, color='orange', marker='o')
    ax.plot(pred1_labels2, k2*pred1_labels2+b2, color='orange', linestyle='solid', linewidth=3)

    r2_1 = r2_score(true_labels1, pred1_labels1)
    r2_2 = r2_score(true_labels2, pred1_labels2)
    print(f'r2_1: {r2_1}, r2_2: {r2_2}')
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.plot(xs, [label[0] for label in labels], label='true', color='dimgray', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[1] for label in labels], label=label1, color='blue', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[2] for label in labels], label=label2, color='red', linestyle='solid', linewidth=2)
    # ax.scatter(xs, [label[1] for label in labels], label=label1, color='black', marker='o')
    
    x_min, x_max = ax.get_xlim()
    xs = np.linspace(x_min, x_max, 100)
    ax.plot(xs, xs, color='lightgray', linestyle='solid')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(output_filepath)

    

if __name__ == '__main__':
    # infer_and_draw()
    # draw()
    segment_compare()