import cv2
import argparse
import numpy as np
from .kmeans import kmeans
from gloss import detect_gloss

parser = argparse.ArgumentParser()

parser.add_argument("image_path")
parser.add_argument("-t", "--threshold", default=0.2, type=float, required=False)
parser.add_argument("-i", "--iterations", default=1, type=int, required=False)
parser.add_argument("-l", "--light", nargs=3, default=[255, 255, 255], type=int, required=False)

args = parser.parse_args()
image = args.image_path

data_raw = cv2.imread(image)
centroids, labels = kmeans(image, float(args.threshold), int(args.iterations))
gloss_percent, gloss_label = detect_gloss(centroids, labels)

print("[q]quit\n[1]raw [2]centroid [3]gloss")

winname = "kmeans viewer"
cv2.imshow(winname, data_raw)
cv2.setWindowTitle(winname, "kmeans viewer [raw] [q to quit, 1 - 3 to change image]")
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
            cv2.setWindowTitle(winname, "kmeans viewer [raw]")
        case '2':
            cv2.imshow(winname, centroids[labels])
            cv2.setWindowTitle(winname, "kmeans viewer [centroids]")
        case '3':
            temp = np.zeros((data_raw.shape), np.uint8)
            temp[np.nonzero(labels == gloss_label)] = [255, 255, 255]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, "kmeans viewer [gloss] [tot. {:0.2f}%]".format(gloss_percent * 100))
cv2.destroyAllWindows()