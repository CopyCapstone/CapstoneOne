from email.policy import default
import json
import tensorflow as tf
import streamlit as st
import cv2
import os
from Module_03_PixelSegmentation.kmeans import kmeans
from Module_03_PixelSegmentation.gloss import detect_gloss
import numpy as np
from pathlib import Path
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
    
    # 4. เรียงลำดับ Features ให้ตรงกับ DataFrame ตอน Train เป๊ะๆ
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


# --- Configuration & Constants ---
st.set_page_config(page_title="Capstone Project", layout="wide")
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

st.title("🔍 Module 4")
st.divider()
# --- Initialize session state ---
def init_session_state():
    """รวมการประกาศ Session State """
    defaults = {
        # "stored_threshold": 0.2,
        # "stored_iterations": 1,
        # "stored_gloss_percent": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val  
init_session_state()

image_path = rf"tmp\glossReplaced_frames\GlossReplaced_frame_{st.session_state.stored_frame_num}.jpg"

if os.path.exists(image_path):
    # 1. โหลดภาพและประมวลผล
    data_raw = cv2.imread(image_path)
    h, w, _ = data_raw.shape 
    
    RGB_mean = st.session_state.stored_average_RGB_diffuse 
    
    st.image(cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB), caption=f"GlossReplaced_frame_{st.session_state.stored_frame_num}")
    st.metric(label="Gloss Replaced by Mean Diffuse", value=f"RBG: {RGB_mean}")
    # 1. โหลดโมเดล
    try:
        model = tf.keras.models.load_model('my_model.keras')
        # 2. เตรียมข้อมูล Input
        # RGB_mean มาในรูปแบบ [R, G, B]
        input_data = prepare_input(RGB_mean)
        
        # 3. ทำการ Predict
        prediction = model.predict(input_data)
        
        # 4. แสดงผลลัพธ์
        pred_L = prediction[0][0] * 100.0
        pred_a = (prediction[0][1] * 240.0) - 120.0
        pred_b = (prediction[0][2] * 240.0) - 120.0
        st.metric(label=f"AI Prediction Result", value=f"Lab: [{pred_L:.2f} {pred_a:.2f} {pred_b:.2f}]")
        
    except Exception as e:
        st.error(f"Error loading or predicting: {e}")
    
    with st.expander("ℹ️ Details"):
        st.write(f"Processing Frame: {st.session_state.stored_frame_num}")
        st.write(f"Object Detection Pad_x: {st.session_state.stored_pad_x}")
        st.write(f"Object Detection Pad_y: {st.session_state.stored_pad_y}")
        st.write(f"Object Detection Shrink: {st.session_state.stored_shrink}")
        st.write(f"Chromatic Adaptation Transform Method: {st.session_state.stored_cat_method}")
        if st.session_state.stored_cat_method == "custom":
            st.write(f"Custom CAT: {st.session_state.stored_light_source} to {st.session_state.stored_light_target}")
        if st.session_state.stored_cat_method == "white_patch":
            st.write(f"White Patch Percentile Setting: Lower={st.session_state.stored_lower}%, Upper={st.session_state.stored_upper}%")
        st.write(f"Clustering Threshold: {st.session_state.stored_threshold}")
        st.write(f"Max K-means Iterations: {st.session_state.stored_iterations}")
        st.divider()
        st.write(f"Pixel Segmentation Gloss Percentage: {st.session_state.stored_gloss_percent * 100:.2f}%")
        st.write(f"The average RGB value of specular pixels: {st.session_state.stored_average_RGB_gloss}")
        st.write(f"The average RGB value of diffuse pixels: {st.session_state.stored_average_RGB_diffuse}")
            
else:
    st.error(f"ไม่พบไฟล์ภาพใน Path: {image_path}")