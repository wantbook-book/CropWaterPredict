import cv2
import numpy as np
from pathlib import Path

directory_path = Path(r'.\images\Canon\01 - Nov - 23\JPG')
image_path = directory_path / 'DSC_0536 (Pot 1).JPG'
image = cv2.imread(str(image_path))

cv2.namedWindow("image", 0)
cv2.resizeWindow("image", 300, 300)  
cv2.imshow('image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()