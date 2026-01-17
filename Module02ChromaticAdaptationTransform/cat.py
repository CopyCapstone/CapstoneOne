import cv2
from pathlib import Path
import numpy as np

image = f"{Path(__file__).parent}/test.png"

data = cv2.imread(image)

data_b = data.copy()
data_b[..., [1, 2]] = 0

data_g = data.copy()
data_g[..., [0, 2]] = 0

data_r = data.copy()
data_r[..., [0, 1]] = 0

cv2.imshow("image", data)
cv2.waitKey()

cv2.imshow("image", data_b)
cv2.waitKey()

cv2.imshow("image", data_g)
cv2.waitKey()

cv2.imshow("image", data_r)
cv2.waitKey()

cv2.destroyWindow("image")