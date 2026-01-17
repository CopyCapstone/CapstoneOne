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

# print(cat_image)
cv2.imshow("cat", data)
cv2.waitKey()
cv2.imshow("cat", data_cat)
cv2.waitKey()