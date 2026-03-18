from email.policy import default
import json

import streamlit as st
import cv2
import os
from Module_03_PixelSegmentation.kmeans import kmeans
from Module_03_PixelSegmentation.gloss import detect_gloss
import numpy as np
from pathlib import Path

# --- Configuration & Constants ---
st.set_page_config(page_title="Capstone Project", layout="wide")
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

# ฟังก์ชันช่วยสร้างกล่องสี HTML
def create_color_box(rgb):
    r, g, b = rgb
    return f'<span style="display:inline-block; width:70px; height:70px; background-color:rgb({r},{g},{b}); border:1px solid #ccc; border-radius:4px; vertical-align:middle; margin-left:10px;"></span>'

st.title("🔍 Module 3 & 5: Pixel Segmentation and Glossiness Measurement")
st.divider()
# --- Initialize session state ---
def init_session_state():
    """รวมการประกาศ Session State """
    defaults = {
        "stored_threshold": 0.2,
        "stored_iterations": 1,
        "stored_gloss_percent": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val  
init_session_state()

with st.sidebar:
    st.header("⚙️ Segmentation Settings")
    
    # แปลง threshold เป็น Slider (0.0 - 1.0)
    # Default 0.20 ตามที่เขียนไว้ใน docstring ของฟังก์ชัน kmeans
    st.write("Default Clustering Threshold: 0.20 Default Max Iterations: 1")
    input1 , input2 = st.columns([0.7, 0.3])
    with input1:
        input_threshold = st.slider(
            "Clustering Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            step=0.01,
            help="จะหยุดเพิ่ม Cluster ถ้าค่า Inertia ไม่ดีขึ้นเกิน % นี้",
            value=st.session_state.stored_threshold,
            key="input_threshold",
            on_change=lambda: st.session_state.update({
                "stored_threshold": st.session_state.input_threshold
            })
        )
        st.session_state.stored_threshold = st.session_state.input_threshold
    with input2:
        input_iterations = st.number_input(
            "Max Iterations", 
            min_value=1, 
            max_value=10, 
            step=1,
            help="จำนวนรอบที่จะรัน K-means เพื่อหาผลลัพธ์ที่ดีที่สุด",
            value=st.session_state.stored_iterations,
            key="input_iterations",
            on_change=lambda: st.session_state.update({
                "stored_iterations": st.session_state.input_iterations 
            })
        )
        st.session_state.stored_iterations = st.session_state.input_iterations

image_path = rf"tmp\CAT_frames\CAT_frame_{st.session_state.stored_frame_num}.jpg"

if os.path.exists(image_path):
    # 1. โหลดภาพและประมวลผล
    data_raw = cv2.imread(image_path)
    h, w, _ = data_raw.shape 
    
    # รัน K-means และค้นหา Gloss
    centroids, labels = kmeans(image_path, input_threshold, input_iterations)
    gloss_percent, gloss_label = detect_gloss(centroids, labels)

    # เตรียม Mask
    mask_gloss = (labels == gloss_label)
    mask_not_gloss = (labels != gloss_label)

    # ค่า Gloss Percentage และ ค่าเฉลี่ย RGB ของ Gloss
    st.session_state.stored_gloss_percent = gloss_percent
    st.metric(label="Gloss Percentage", value=f"{gloss_percent * 100:.2f}%")
    gloss_pixels_bgr = data_raw[mask_gloss]
    gloss_pixels_rgb = gloss_pixels_bgr[:, ::-1]
    average_BGR_gloss = gloss_pixels_bgr.mean(axis=0).astype(np.uint8)
    average_RGB_gloss = average_BGR_gloss[::-1]
    st.session_state.stored_average_RGB_gloss = average_RGB_gloss
    
    # ค่า ค่าเฉลี่ย RGB ของ Diffuse (ส่วนที่ไม่ใช่ Gloss)
    diffuse_pixels_bgr = data_raw[mask_not_gloss]
    diffuse_pixels_rgb = diffuse_pixels_bgr[:, ::-1]
    average_BGR_diffuse = diffuse_pixels_bgr.mean(axis=0).astype(np.uint8)
    average_RGB_diffuse = average_BGR_diffuse[::-1]
    st.session_state.stored_average_RGB_diffuse = average_RGB_diffuse

    col1, col2, col3, col4 = st.columns([0.3, 0.2, 0.3, 0.2])
    with col1:
        st.metric(label="The average RGB value of specular pixels", value=f"{average_RGB_gloss}")
    with col2:
        st.write(f"{create_color_box(average_RGB_gloss)}", unsafe_allow_html=True)
    with col3:
        st.metric(label="The average RGB value of diffuse pixels", value=f"{average_RGB_diffuse}")  
    with col4:
        st.write(f"{create_color_box(average_RGB_diffuse)}", unsafe_allow_html=True)

    # --- สร้างรูปที่ 2: บริเวณเฉพาะที่เป็น Gloss (Masked Image) ---
    gloss_only = np.zeros_like(data_raw)
    gloss_only[mask_gloss] = data_raw[mask_gloss]

    # --- สร้างรูปที่ 3: แทนที่ Gloss ด้วย Mean ของ Not-Gloss ---
    replaced_img = data_raw.copy()
    if np.any(mask_not_gloss):
        # คำนวณค่าสีเฉลี่ยของส่วนที่ไม่ใช่ Gloss (BGR) ไว้แล้ว = average_BGR_diffuse
        # แทนที่บริเวณ Gloss ด้วยค่าเฉลี่ยนั้น
        replaced_img[mask_gloss] = average_BGR_diffuse
    else:
        st.warning("ไม่สามารถหาค่าเฉลี่ยส่วนอื่นได้เนื่องจากภาพเป็น Gloss ทั้งหมด")


    # แสดง 3 รูปเรียงกัน (หรือใช้ columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(data_raw, cv2.COLOR_BGR2RGB), 
                 caption="1. Original Image", width='stretch')
    with col2:
        st.image(cv2.cvtColor(gloss_only, cv2.COLOR_BGR2RGB), 
                 caption="2. Gloss Area Only", width='stretch')
        debug_dir = TMP_DIR / "glossArea_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"GlossArea_frame_{st.session_state.stored_frame_num}.jpg"
        cv2.imwrite(str(debug_path), gloss_only)
    with col3:
        st.image(cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB), 
                 caption="3. Gloss Replaced by Mean Diffuse", width='stretch')
        debug_dir = TMP_DIR / "glossReplaced_frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"GlossReplaced_frame_{st.session_state.stored_frame_num}.jpg"
        cv2.imwrite(str(debug_path), replaced_img)
        
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

    with st.expander("🕵️‍♂️ เจาะลึกค่าสีบริเวณ Gloss"):
        if len(gloss_pixels_rgb) > 0:
            unique_colors = np.unique(gloss_pixels_rgb, axis=0)
            st.write(f"จำนวนพิกเซล Gloss ทั้งหมด: {len(gloss_pixels_bgr)} พิกเซล จาก พิกเซลของภาพทั้งหมด {h*w} พิกเซล")
            st.write(f"จำนวนเฉดสีที่พบ (Unique Colors): {len(unique_colors)} เฉดสี จาก Unique Colors ของภาพทั้งหมด {len(np.unique(data_raw, axis=0))} เฉดสี")
            st.write(f"ค่าสี RGB ของพิกเซล Gloss: {gloss_pixels_rgb}")
        if len(diffuse_pixels_rgb) > 0:
            st.write(f"ค่าสี RGB ของพิกเซล Diffuse: {diffuse_pixels_rgb}")
            
    # Prepare the dictionary
    settings_data = {
        "frame_num": int(st.session_state.stored_frame_num),
        "pad_x": int(st.session_state.stored_pad_x),
        "pad_y": int(st.session_state.stored_pad_y),
        "shrink": int(st.session_state.stored_shrink),
        "cat_method": st.session_state.stored_cat_method,
        "clustering_threshold": float(st.session_state.stored_threshold),
        "max_kmeans_iterations": int(st.session_state.stored_iterations),
    }
    
    # Add conditional variables based on CAT method
    if st.session_state.stored_cat_method == "custom":
        settings_data["light_source"] = st.session_state.stored_light_source
        settings_data["light_target"] = st.session_state.stored_light_target
    elif st.session_state.stored_cat_method == "white_patch":
        settings_data["white_patch_lower"] = float(st.session_state.stored_lower)
        settings_data["white_patch_upper"] = float(st.session_state.stored_upper)

    # Convert dictionary to a formatted JSON string
    json_string = json.dumps(settings_data, indent=4)
    
    # Create a download button
    with st.sidebar:
        st.divider()
        st.download_button(
            label="💾 Save Settings as JSON",
            data=json_string,
            file_name=f"settings.json",
            mime="application/json",
            width='stretch'
        )
else:
    st.error(f"ไม่พบไฟล์ภาพใน Path: {image_path}")