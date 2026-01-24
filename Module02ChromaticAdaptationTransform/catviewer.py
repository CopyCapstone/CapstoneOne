import cv2
import argparse
import numpy as np
from cat import chromatic_adaptation_transform

parser = argparse.ArgumentParser()

parser.add_argument("image_path")
parser.add_argument("-m", "--method", default="grey_world", required=False)
parser.add_argument("-o", "--options", nargs="*", default=None, required=False)

args = parser.parse_args()
image = args.image_path

data_raw = cv2.imread(image)
data_cat = chromatic_adaptation_transform(image, args.method, args.options)

print("[q]quit\n[1]pre-cat [2]post-cat [3]diff\n[4]red pre-cat [5]red post-cat\n[6]green pre-cat [7]green post-cat\n[8]blue pre-cat [9]blue post-cat")

winname = "cat viewer"
cv2.imshow("cat viewer", data_raw)
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
            cv2.imshow(winname, data_raw)
            cv2.setWindowTitle(winname, "cat viewer [pre-cat]")
        case '2':
            cv2.imshow(winname, data_cat)
            cv2.setWindowTitle(winname, "cat viewer [post-cat]")
        case '3':
            rgb_diff = (data_raw.mean((0,1)) - data_cat.mean((0,1))).round(2)
            cv2.imshow(winname, ((data_raw.astype(np.int16) - data_cat.astype(np.int16)) / 2 + 128).round().astype(np.uint8))
            cv2.setWindowTitle(winname, "cat viewer [diff-cat] [r. {:+0.2f} g. {:+0.2f} b. {:+0.2f}]".format(*np.flip(rgb_diff)))
        case '4':
            temp = np.zeros(data_raw.shape, np.uint8)
            temp[:, :, 2] = data_raw[:, :, 2]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-red] [avg. {temp[:,:,2].mean().round(2)}]")
        case '5':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 2] = data_cat[:, :, 2]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-red] [cor. {(temp[:,:,2].mean() / data_raw[:,:,2].mean() * 100).round(2)}%]")
        case '6':
            temp = np.zeros(data_raw.shape, np.uint8)
            temp[:, :, 1] = data_raw[:, :, 1]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-green] [avg. {temp[:,:,1].mean().round(2)}]")
        case '7':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 1] = data_cat[:, :, 1]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-green] [cor. {(temp[:,:,1].mean() / data_raw[:,:,1].mean() * 100).round(2)}%]")
        case '8':
            temp = np.zeros(data_raw.shape, np.uint8)
            temp[:, :, 0] = data_raw[:, :, 0]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [pre-blue] [avg. {temp[:,:,0].mean().round(2)}]")
        case '9':
            temp = np.zeros(data_cat.shape, np.uint8)
            temp[:, :, 0] = data_cat[:, :, 0]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, f"cat viewer [post-blue] [cor. {(temp[:,:,0].mean() / data_raw[:,:,0].mean() * 100).round(2)}%]")
cv2.destroyAllWindows()