import sys
sys.path.append('..')
from typing import Union, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
def compare_models(
        model1: nn.Module, model1_use_tm: bool,
        model2: nn.Module, model2_use_tm: bool,
        dataloader: torch.utils.data.DataLoader, 
        output_file: Path, 
        device: torch.device,
        label1: str,
        label2: str,
        y_label: str
    ):
    model1.eval()
    model2.eval()
    model1.to(device)
    model2.to(device)
    labels = []
    count = 0
    for rgb_image, temp_moistures, label in dataloader:
        rgb_image = rgb_image.to(device)
        temp_moistures = temp_moistures.to(device)
        label = label.to(device)
        with torch.no_grad():
            if model1_use_tm:
                label1 = model1(rgb_image, temp_moistures)
            else:
                label1 = model1(rgb_image)
            if model2_use_tm:
                label2 = model2(rgb_image, temp_moistures)
            else:
                label2 = model2(rgb_image)
        # (batch_size, 1)
        for l, l1, l2 in zip(label.tolist(), label1.tolist(), label2.tolist()):
            labels.append((l[0], l1[0], l2[0]))

        count += 1
        print(f'count: {count}')
        if count > 10:
            break
    with open(output_file.parent/(output_file.stem+'.txt'), 'w') as f:
        for label, label1, label2 in labels:
            f.write(f'{label} {label1} {label2}\n')
    draw_infer_graph(labels=labels, output_file=output_file, y_label=y_label, label1=label1, label2=label2)
    # fig, ax = plt.subplots()
    # xs = range(len(labels))
    # ax.spines['top'].set_color('none')
    # # ax.spines['bottom'].set_color('none')
    # # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.plot(xs, [label[0] for label in labels], label='true', color='black', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[1] for label in labels], label=label1, color='blue', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[2] for label in labels], label2=label2, color='red', linestyle='solid', linewidth=2)
    # # ax.set_xlabel('Epoch')
    # ax.set_ylabel(y_label)
    # ax.legend()
    # plt.savefig(output_file)
    

def read_labels(label_file: Path)->list[Tuple[float, float, float]]:
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            labels.append(list(map(float, line.strip().split(' '))))
    return labels

def draw_infer_graph(labels: list[Tuple[float, float, float]], output_file:Path, y_label: str, label1:str, label2:str):
    fig, ax = plt.subplots(figsize=(10,6))
    xs = range(len(labels))
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    ax.plot(xs, [label[0] for label in labels], label='true', color='dimgray', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[1] for label in labels], label=label1, color='blue', linestyle='solid', linewidth=2)
    # ax.plot(xs, [label[2] for label in labels], label=label2, color='red', linestyle='solid', linewidth=2)
    # ax.scatter(xs, [label[1] for label in labels], label=label1, color='black', marker='o')
    ax.scatter(xs, [label[1] for label in labels], label=label1, color='deepskyblue', marker='x')
    ax.scatter(xs, [label[2] for label in labels], label=label2, color='orange', marker='o')
    # ax.set_xlabel('Epoch')
    ax.set_ylabel(y_label)
    ax.legend()
    plt.savefig(output_file)

def cal_r2(labels: list[Tuple[float, float, float]]):
    # 模型1 预测值和真实值
    y_true = [label[0] for label in labels]
    y_predict1 = [label[1] for label in labels]
    y_predict2 = [label[2] for label in labels]

    r2_1 = r2_score(y_true, y_predict1)
    r2_2 = r2_score(y_true, y_predict2)
    print(f'r2_1: {r2_1}, r2_2: {r2_2}')