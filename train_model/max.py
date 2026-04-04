from PIL import Image
import numpy as np

# 1. Load the image and convert it to RGB 
# (This step ensures any Alpha/transparency channel in the is ignored)
img = Image.open("test.jpg").convert("RGB")

# 2. Convert the image to a NumPy array for fast mathematical operations
img_array = np.array(img)

# 3. Find the maximum value along the spatial dimensions (height=0, width=1)
# This leaves us with an array of 3 values: [max_R, max_G, max_B]
max_r, max_g, max_b = img_array.max(axis=(0, 1))

print(f"Max Red: {max_r}")
print(f"Max Green: {max_g}")
print(f"Max Blue: {max_b}")

# Bonus: If you want the single highest value across ALL channels (0-255)
overall_max = img_array.max()
print(f"Absolute highest value in the image: {overall_max}")