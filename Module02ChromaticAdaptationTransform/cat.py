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
cv2.setWindowTitle(winname, "cat viewer [pre-cat] [q to quit, 1 - 9 to change image]")
while True:
    key = cv2.waitKeyEx()
    try:
        key = chr(key)
    except ValueError as e:
        if key == -1: # this is the signal to close the window
            break
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
            cv2.setWindowTitle(winname, "cat viewer [diff-cat]")
        case '4':
            temp = np.zeros(data.shape, np.uint8)
            temp[:, :, 2] = data[:, :, 2]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-red] [avg. {mean_bgr[2].round(2)}]")
        case '5':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 2] = data_cat[:, :, 2]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-red] [cor. {(mean_abs / mean_bgr[2] * 100).round(2)}%]")
        case '6':
            temp = np.zeros(data.shape, np.uint8)
            temp[:, :, 1] = data[:, :, 1]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-green] [avg. {mean_bgr[1].round(2)}]")
        case '7':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 1] = data_cat[:, :, 1]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-green] [cor. {(mean_abs / mean_bgr[1] * 100).round(2)}%]")
        case '8':
            temp = np.zeros(data.shape, np.uint8)
            temp[:, :, 0] = data[:, :, 0]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-blue] [avg. {mean_bgr[0].round(2)}]")
        case '9':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 0] = data_cat[:, :, 0]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-blue] [cor. {(mean_abs / mean_bgr[0] * 100).round(2)}%]")
cv2.destroyAllWindows()