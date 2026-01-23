import cv2
import numpy as np

def chromatic_adaptation_transform(image):
    data = cv2.imread(image)

    mean_bgr = data.mean((0,1))
    mean_abs = mean_bgr.mean()

    data_cat = (data * (mean_abs / mean_bgr)).round().clip(0, 255).astype(np.uint8)

    print(mean_bgr, mean_abs / mean_bgr)
    return data_cat