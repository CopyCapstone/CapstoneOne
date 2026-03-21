import cv2
import argparse
import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from Module_03_PixelSegmentation.kmeans import kmeans
from Module_03_PixelSegmentation.gloss import detect_gloss

parser = argparse.ArgumentParser()

parser.add_argument("image_path")
parser.add_argument("-t", "--threshold", default=0.2, required=False)
parser.add_argument("-i", "--iterations", default=1, required=False)

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
            temp[np.nonzero(labels == gloss_label)] = centroids[gloss_label]
            cv2.imshow(winname, temp)
            cv2.setWindowTitle(winname, "kmeans viewer [gloss] [tot. {:0.2f}%]".format(gloss_percent * 100))
            # Print the results to the terminal
            print(f"--- Analysis Results ---")
            print(f"Specular Percentage: {gloss_percent * 100:.2f}%")
            print(f"Average RGB of Specular Pixels: {centroids[gloss_label]}")
            # 1. Isolate your BGR centroid value
            bgr_color = centroids[gloss_label]

            # 2. Reshape it into a 1x1 pixel "image" for OpenCV
            # Note: KMeans usually outputs floats. We convert to uint8 (0-255) for standard OpenCV conversion.
            pixel_bgr = np.array([[bgr_color]], dtype=np.uint8)

            # 3. Convert from BGR to CIE XYZ
            pixel_xyz = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2XYZ)

            # 4. Extract the 1D array back out of the 1x1 image structure
            xyz_color = pixel_xyz[0][0]

            # 5. Print your new format
            print(f"Average XYZ of Specular Pixels: {xyz_color}")
        case '4':
            # # Identify indices of all non-glossy centroids
            # non_gloss_indices = [i for i in range(len(centroids)) if i != gloss_label]
            
            # print(f"--- Diffuse (Non-Glossy) Analysis ---")
            
            # if not non_gloss_indices:
            #     print("No non-glossy areas detected.")
            # else:
            #     # Calculate the total percentage of non-glossy area
            #     non_gloss_percent = 1.0 - gloss_percent
            #     print(f"Total Diffuse Percentage: {non_gloss_percent * 100:.2f}%")

            #     # Create a mask for all non-glossy pixels
            #     diffuse_mask = (labels != gloss_label)
                
            #     # Option A: Calculate the overall average RGB of all non-glossy pixels combined
            #     # We use the original raw data filtered by the diffuse mask
            #     avg_rgb_diffuse = cv2.mean(data_raw, mask=diffuse_mask.astype(np.uint8))[:3]
            #     # Note: cv2.mean returns BGR, we flip to RGB for printing
            #     print(f"Combined Average RGB (Diffuse): [{avg_rgb_diffuse[2]:.2f}, {avg_rgb_diffuse[1]:.2f}, {avg_rgb_diffuse[0]:.2f}]")

            #     # Option B: Print individual RGB values for each non-glossy cluster
            #     print("Individual Non-Glossy Clusters:")
            #     for idx in non_gloss_indices:
            #         cluster_percent = np.mean(labels == idx) * 100
            #         print(f"  - Cluster [{idx}]: RGB {centroids[idx][::-1]} ({cluster_percent:.2f}%)")

            # # Visual Feedback: Show all non-glossy areas on screen
            # temp_diffuse = np.zeros_like(data_raw)
            # temp_diffuse[labels != gloss_label] = data_raw[labels != gloss_label]
            # cv2.imshow(winname, temp_diffuse)
            # cv2.setWindowTitle(winname, "kmeans viewer [diffuse] [tot. {:0.2f}%]".format(non_gloss_percent * 100))
            # 1. Create a mask for all pixels NOT identified as gloss
            non_gloss_mask = (labels != gloss_label)
            
            # 2. Calculate the average BGR of all diffuse (non-glossy) pixels
            # cv2.mean calculates the average of the pixels in data_raw using the mask
            avg_bgr = cv2.mean(data_raw, mask=non_gloss_mask.astype(np.uint8))[:3]
            
            # 3. Convert BGR to RGB for printing
            avg_rgb = [round(avg_bgr[2], 2), round(avg_bgr[1], 2), round(avg_bgr[0], 2)]
            
            # 4. Visualization: Show the non-glossy area
            temp = np.zeros_like(data_raw)
            temp[non_gloss_mask] = data_raw[non_gloss_mask]
            cv2.imshow(winname, temp)
            
            non_gloss_percent = (1.0 - gloss_percent)
            cv2.setWindowTitle(winname, "kmeans viewer [non-gloss] [tot. {:0.2f}%]".format(non_gloss_percent * 100))
            
            # 5. Print Results
            print(f"--- Non-Gloss (Diffuse) Analysis ---")
            print(f"Non-Gloss Percentage: {non_gloss_percent * 100:.2f}%")
            print(f"Average RGB of Non-Gloss Pixels: {avg_rgb}")
cv2.destroyAllWindows()