import tensorflow as tf
import streamlit as st
import cv2
import os
from pathlib import Path
from Module_04_Color_Measurement.prepare_input import prepare_input

# --- Configuration & Constants ---
st.set_page_config(page_title="Capstone Project", layout="wide")
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

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

st.title("🤖 Module 4: Color Measurement")
st.divider()

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