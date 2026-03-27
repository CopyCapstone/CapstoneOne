import numpy as np
from skimage import color

def bt709_to_linear(c):
    return np.where(c < 0.081, c / 4.5, ((c + 0.099) / 1.099) ** (1 / 0.45))

def calculate_lab(row):
    r, g, b = row[0] / 255.0, row[1] / 255.0, row[2] / 255.0
    rgb_linear = bt709_to_linear(np.array([r, g, b]))
    # แปลง linear RGB → XYZ ด้วย BT.709/sRGB matrix
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = M @ rgb_linear
    # XYZ → Lab (D65)
    lab_pixel = color.xyz2lab(xyz.reshape(1,1,3), illuminant='D65')
    L, a, b = lab_pixel[0,0]
    return round(L, 2), round(a, 2), round(b, 2)

def prepare_input(rgb_values):
    """
    แปลงค่า RGB เป็น features สำหรับโมเดล (Quadratic + Lab)
    พร้อมปรับ Shape และ Normalize ข้อมูลให้ตรงกับตอน Train
    """
    # 1. รับค่า RGB และ Normalize ให้อยู่ในช่วง [0, 1]
    r = rgb_values[0] / 255.0
    g = rgb_values[1] / 255.0
    b = rgb_values[2] / 255.0
    
    # 2. แปลงเป็น Lab
    L_cal, a_cal, b_cal = calculate_lab(rgb_values)
    
    # 3. *** สำคัญมาก ***: Normalize ค่า Lab ให้เหมือนตอน Train
    L_cal_norm = L_cal / 100.0
    a_cal_norm = (a_cal + 120.0) / 240.0
    b_cal_norm = (b_cal + 120.0) / 240.0
    
    # 4. เรียงลำดับ Features ให้ตรงกับ DataFrame ตอน Train
    # ลำดับใน Train Data: 'R', 'G', 'B', 'L_cal', 'a_cal', 'b_cal', 'R*G', 'R*B', 'G*B', 'R**2', 'G**2', 'B**2'
    features = [
        r, g, b, 
        L_cal_norm, a_cal_norm, b_cal_norm,
        r * g, r * b, g * b,
        r ** 2, g ** 2, b ** 2
    ]
    
    # 5. แปลงเป็น Numpy Array และปรับ Shape เป็น 2D (1, 12) 
    features_array = np.array([features]) 
    
    return features_array