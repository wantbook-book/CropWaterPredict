import gradio as gr
from PIL import Image, ImageDraw

def process_image_and_points(image, points):
    # 创建一个ImageDraw对象来在图片上绘图
    draw = ImageDraw.Draw(image)
    
    # 解析点坐标
    points = points.split(';')  # 分隔不同的点
    for point in points:
        if point:  # 确保点坐标不为空
            x, y = point.split(',')  # 分隔x,y坐标
            x, y = int(x), int(y)  # 转换为整数
            # 在图片上标点（这里以绘制小圆圈为例）
            draw.ellipse((x-5, y-5, x+5, y+5), fill='red')

    # 返回修改后的图片
    return image

# 创建Gradio界面
iface = gr.Interface(
    fn=process_image_and_points,
    inputs=[
        gr.ImageEditor(type="pil", eraser=gr.Eraser(), brush=gr.Brush()),  # 允许用户上传和编辑图片
        gr.Textbox(label="Points", placeholder="Enter points like 'x1,y1;x2,y2'"),
    ],
    outputs=gr.Image(type="pil"),
    title="Mark Points on Image",
    description="Upload an image and enter points as 'x1,y1;x2,y2;...' to mark them on the image."
)

iface.launch()
