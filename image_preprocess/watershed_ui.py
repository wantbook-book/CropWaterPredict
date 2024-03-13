import numpy as np
import cv2

# 全局变量
image = None
markers = None
current_marker = 1  # 当前标记
marks_updated = False  # 标记是否更新

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # 用户点击左键，在当前位置标记
        cv2.circle(markers, (x, y), 10, (current_marker), -1)

        # 更新图像显示
        marks_updated = True

# 加载图像
image = cv2.imread(r'.\images\Canon\01 - Nov - 23\JPG\DSC_0537 (Pot 2) .JPG')
image = cv2.resize(image, (600, 400))  # 调整图像大小
markers = np.zeros(image.shape[:2], dtype=np.int32)

# 创建窗口并绑定鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    # 显示图像
    cv2.imshow('Image', image)

    # 每次标记更新时，重新运行分水岭算法
    if marks_updated:
        markers_copy = markers.copy()
        cv2.watershed(image, markers_copy)
        segments = np.zeros(image.shape, dtype=np.uint8)

        for color_ind in range(current_marker):
            # 给不同区域上色
            segments[markers_copy == (color_ind)] = [int(np.random.randint(0, 255)) for _ in range(3)]

        cv2.imshow('Segments', segments)
        marks_updated = False

    # 按键操作
    k = cv2.waitKey(1)
    if k == ord('q'):  # 按'q'键退出
        break
    elif k == ord('n'):  # 按'n'键切换到下一个标记
        current_marker += 1

cv2.destroyAllWindows()
