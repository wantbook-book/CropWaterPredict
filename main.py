
import train_soil_moisture_predict as smp_train
import train_sapflow_predict as sfp_train
from pathlib import Path
import torch
if __name__ == '__main__':
    src_dir = Path('./main.py').resolve().parent
    rgb_images_dir = src_dir / 'data' / 'rgb_images'
    # infrared_image_dir = src_dir / 'data' / 'thermal_data_processed'
    temp_moisture_file_path = src_dir / 'data' / 'series_data' / 'T_moisture_data.csv'
    soil_moisture_file_path = src_dir / 'data' / 'labels' / 'soil_water_content.CSV'
    sapflow_rgb_images_dir = src_dir / 'data' / 'sapflow_predict_data' / 'rgb_images'
    sapflow_dir = src_dir / 'data' / 'sapflow_predict_data' / 'sapflow'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print('='*10+'soil_moisture_predict: rgb_vgg16_train'+'='*10)
    # smp_train.rgb_vgg16_train(
    #     rgb_images_dir=rgb_images_dir, 
    #     soil_moisture_filepath=soil_moisture_file_path,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    # print('='*10+'soil_moisture_predict: rgb_vgg16_tm_train'+'='*10)
    # smp_train.rgb_vgg16_tm_trian(
    #     rgb_images_dir=rgb_images_dir, 
    #     temp_moisture_filepath=temp_moisture_file_path,
    #     soil_moisture_filepath=soil_moisture_file_path,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    # print('='*10+'soil_moisture_predict: rgb_renet18_train'+'='*10)
    # smp_train.rgb_resnet18_train(
    #     rgb_images_dir=rgb_images_dir, 
    #     soil_moisture_filepath=soil_moisture_file_path,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    # print('='*10+'soil_moisture_predict: rgb_renet18_tm_train'+'='*10)
    # smp_train.rgb_resnet18_tm_train(
    #     rgb_images_dir=rgb_images_dir, 
    #     temp_moisture_filepath=temp_moisture_file_path,
    #     soil_moisture_filepath=soil_moisture_file_path,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    # print('='*10+'sapflow_predict: rgb_vgg16_train'+'='*10)
    # sfp_train.rgb_vgg16_train(
    #     rgb_images_dir=sapflow_rgb_images_dir, 
    #     sapflow_dir=sapflow_dir,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    # print('='*10+'sapflow_predict: rgb_vgg16_tm_train'+'='*10)
    # sfp_train.rgb_vgg16_tm_trian(
    #     rgb_images_dir=sapflow_rgb_images_dir, 
    #     sapflow_dir=sapflow_dir,
    #     temp_moisture_filepath=temp_moisture_file_path,
    #     device=device,
    #     num_epochs=1,
    #     batch_size=32,
    #     lr=0.01,
    #     val_epochs=1,
    #     patience=10
    # )

    print('='*10+'sapflow_predict: rgb_renet18_train'+'='*10)
    sfp_train.rgb_resnet18_train(
        rgb_images_dir=sapflow_rgb_images_dir, 
        sapflow_dir=sapflow_dir,
        device=device,
        num_epochs=1,
        batch_size=32,
        lr=0.01,
        val_epochs=1,
        patience=10
    )

    print('='*10+'sapflow_predict: rgb_renet18_tm_train'+'='*10)
    sfp_train.rgb_resnet18_tm_train(
        rgb_images_dir=sapflow_rgb_images_dir, 
        sapflow_dir=sapflow_dir,
        temp_moisture_filepath=temp_moisture_file_path,
        device=device,
        num_epochs=1,
        batch_size=32,
        lr=0.01,
        val_epochs=1,
        patience=10
    )



