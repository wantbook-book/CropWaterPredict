import cv2
import numpy as np
import sys
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
IMAGE_SUFFIXES = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

IMAGE = 'image'
IMAGE_PROCESSED = 'image_processed'
image = None
input_points = []
input_labels = []
def resize(image):
    # 原始图像的尺寸
    original_height, original_width = image.shape[:2]
    # 目标宽度
    target_width = 600
    # 计算缩放因子，以保持长宽比不变
    scale_factor = target_width / original_width
    # 计算新的高度
    target_height = int(original_height * scale_factor)
    # 使用计算出的新尺寸对图像进行缩放
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
def process(image_file: Path, output_dir: Path, predictor):
    global image, input_points, input_labels
    image_org = cv2.imread(str(image_file))
    image_org = resize(image_org)
    image = image_org.copy()
    cv2.imshow(IMAGE, image)
    image_tosave = None
    mask_tosave = None
    clip_image = None
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # 这里可以添加处理图片的逻辑
            print("process...")
            image_processed, image_tosave, mask_tosave, clip_image = segment_mask(predictor, image_org)
            cv2.imshow(IMAGE_PROCESSED, image_processed)
            print('process succeed!')

        elif key == ord('n'):
            # 这里可以添加处理图片的逻辑
            print("save...")
            input_points = []
            input_labels = []
            # cv2.destroyAllWindows()
            print('save succeed!')
            return
        elif key == ord('r'):
            print('reset...')
            input_points = []
            input_labels = []
            image = image_org.copy()
            cv2.imshow(IMAGE, image)
            cv2.imshow(IMAGE_PROCESSED, image)
            print('reset succeed!')
        elif key == ord('s'):
            print('save...')
            output_file = output_dir/ 'images' / image_file.name
            cv2.imwrite(str(output_file), image_tosave)
            print(f'save to {output_file} succeed!')
            output_file = output_dir / 'masks' / f'mask_{image_file.name}'
            cv2.imwrite(str(output_file), mask_tosave.astype(np.uint8)*255)
            print(f'save to {output_file} succeed!')
            output_file = output_dir / 'clip_images' / f'clip_{image_file.name}'
            cv2.imwrite(str(output_file), clip_image)
            print(f'save to {output_file} succeed!')
        elif key == 27:  # 按下 'ESC' 键退出
            cv2.destroyAllWindows()
            print('exit')
            sys.exit(0)

def segment_mask(predictor, image_org):
    global input_points, input_labels
    print('input_points', input_points)
    print('input_labels', input_labels)
    predictor.set_image(image_org)
    masks, scores, logits = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        multimask_output=True,
    )
    mask = np.any(masks, axis=0)
    # 找到所有True值的索引
    padding = 20
    true_indices = np.argwhere(mask)
    # 找到最小矩形框的左上角和右下角坐标
    top_left = true_indices.min(axis=0)
    bottom_right = true_indices.max(axis=0)



    # mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask_image = np.copy(image_org)
    color = np.array([255, 144, 30], dtype=np.uint8)
    mask_image[mask] = color
    image_processed = cv2.addWeighted(image_org, 0.4, mask_image, 0.6, 0)
    image_tosave = np.zeros_like(image_org)
    image_tosave[mask] = image_org[mask]

    clip_image = image_tosave[top_left[0]-padding:bottom_right[0]+padding, top_left[1]-padding:bottom_right[1]+padding]
    # for mask in masks:
    #     color = np.array([30/255, 144/255, 255/255], dtype=np.float32)
    #     h, w = mask.shape[-2:]
    #     mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    #     image = cv2.addWeighted(image.astype(np.float32)/255, 0.5, mask_image, 0.5, 0)
    return image_processed, image_tosave, mask, clip_image

def click_event(event, x, y, flags, param):
    global image,input_points, input_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在左键点击的位置绘制绿色点
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        input_points.append([x, y])
        input_labels.append(1)
        cv2.imshow('image', image)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 在右键点击的位置绘制红色点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        input_points.append([x, y])
        input_labels.append(0)
        cv2.imshow('image', image)


if __name__ == '__main__':
    # 设置鼠标回调函数
    cv2.namedWindow(IMAGE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(IMAGE_PROCESSED, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(IMAGE, click_event)
    
    images_dir = Path('../data/rgb_images')
    output_dir = Path('../data/segmented_images')
    output_dir.mkdir(exist_ok=True)
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    # device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    predictor = SamPredictor(sam)

    for path in images_dir.iterdir():
        if path.is_dir():
            date = path.name
            subdir_path = output_dir/date
            subdir_path.mkdir(exist_ok=True)
            mask_subdir_path = subdir_path / 'masks'
            mask_subdir_path.mkdir(exist_ok=True)
            clip_subdir_path = subdir_path / 'clip_images'
            clip_subdir_path.mkdir(exist_ok=True)
            images_subdir_path = subdir_path / 'images'
            images_subdir_path.mkdir(exist_ok=True)
            for image_file in path.iterdir():
                if image_file.is_file():
                    if image_file.suffix.lower() in IMAGE_SUFFIXES:
                        process(image_file, output_dir=subdir_path, predictor=predictor)