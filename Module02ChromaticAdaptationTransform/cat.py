import cv2
from pathlib import Path
import numpy as np

image = f"{Path(__file__).parent}/test1.png"

data = cv2.imread(image)

mean_bgr = data.mean((0,1))
mean_abs = mean_bgr.mean()
print(mean_bgr)
print(mean_abs)

data_cat = (data * (mean_abs / mean_bgr)).round().clip(0, 255).astype(np.uint8)
print(mean_abs / mean_bgr)

winname = "cat viewer"
cv2.imshow("cat viewer", data)
cv2.setWindowTitle(winname, "cat viewer [pre-cat] [q to quit, 1 - 3 to change image]")
while True:
    key = chr(cv2.waitKeyEx())
    match key:
        case 'q':
            break
        case '1':
            cv2.imshow(winname, data)
            cv2.setWindowTitle(winname, "cat viewer [pre-cat]")
        case '2':
            cv2.imshow(winname, data_cat)
            cv2.setWindowTitle(winname, "cat viewer [post-cat]")
        case '3':
            cv2.imshow(winname, ((data.astype(np.int16) - data_cat.astype(np.int16)) / 2 + 128).round().astype(np.uint8))
            cv2.setWindowTitle(winname, "cat viewer [diff]")