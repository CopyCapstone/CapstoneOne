
from Module_07_BatchProcessing.cat_batch import chromatic_adaptation_transform
from Module_02_ChromaticAdaptationTransform.custom_CAT import custom_CAT
import cv2
from skimage.color import rgb2xyz, xyz2rgb
from colour.colorimetry import CCS_ILLUMINANTS
ILLUMINANTS_xy = CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
import numpy as np

def process_cat_logic(img_bgr, method, lower, upper, src_light, tgt_light):
    if img_bgr is None:
        return None
    if method == "custom":
        # 1. เตรียมค่า xy ของแสง
        xy_src = ILLUMINANTS_xy[src_light]
        xy_tgt = ILLUMINANTS_xy[tgt_light]
        # 2. แปลง BGR -> RGB -> XYZ (ใช้ skimage.color)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_xyz = rgb2xyz(img_rgb)
        # 3. เรียกใช้ฟังก์ชัน custom_CAT
        processed_xyz = custom_CAT(img_xyz, xy_src, xy_tgt, method='Von Kries')
        # 4. แปลง XYZ กลับเป็น RGB -> BGR
        processed_rgb_float = xyz2rgb(processed_xyz)
        processed_rgb_float = np.clip(processed_rgb_float, 0, 1)
        processed_rgb_uint8 = (processed_rgb_float * 255).astype(np.uint8)
        return cv2.cvtColor(processed_rgb_uint8, cv2.COLOR_RGB2BGR)
    else:
        return chromatic_adaptation_transform(img_bgr, method, [lower, upper])
