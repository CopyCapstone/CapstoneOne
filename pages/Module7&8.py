import os
import cv2
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pathlib import Path
from Module_01_ObjectDetection.detect_rotate_crop import detect_rotate_crop
from Module_03_PixelSegmentation.gloss import detect_gloss
from Module_04_ColorMeasurement.prepare_input import prepare_input
from Module_07_BatchProcessing.kmeans_batch import kmeans
from Module_07_BatchProcessing.process_cat_logic_batch import process_cat_logic

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
SETTING_FILE = TMP_DIR / 'settings.json'
VIDEO_PATH = TMP_DIR/ "uploaded_video" / "uploaded_video.mp4"

def load_config():
    if os.path.exists(SETTING_FILE):
        with open(SETTING_FILE, 'r') as f:
            return json.load(f)
    return {}

st.title("🔥 Module 7 & 8: Batch Processing and Data Aggregation")
st.divider()

with st.sidebar:
    config = load_config()
    pad_x = float(config.get("pad_x", -0.1))
    pad_y = float(config.get("pad_y", -0.05))
    shrink_val = float(config.get("shrink", 0.2))
    cat_method = str(config.get("cat_method", "custom"))
    light_source = str(config.get("light_source", "D65"))
    light_target = str(config.get("light_target", "D65"))
    
    val_lower = config.get("white_patch_lower")
    white_patch_lower = int(val_lower) if val_lower is not None else 95
    val_upper = config.get("white_patch_upper")
    white_patch_upper = int(val_upper) if val_upper is not None else 99
    
    clustering_threshold = float(config.get("clustering_threshold", 0.20))
    max_kmeans_iterations = int(config.get("max_kmeans_iterations", 1))

    st.header("⚙️ System Settings")
    st.subheader("Current Parameters")
    st.json(config) # แสดงโครงสร้าง JSON ให้ดูแบบสวยงาม
    st.divider()
    batchButton = st.button("GO❕❕❕", type="primary", width="stretch")

# โหลดโมเดล
try:
    model = tf.keras.models.load_model('my_model.keras')
except Exception as e:
    st.error(f"Error loading or predicting: {e}") 

if os.path.exists(str(VIDEO_PATH)):
    st.write(f"✅ Video found: {str(VIDEO_PATH)}") 
    # Show Video Player FIRST (so it's always visible)
    try:
        with open(str(VIDEO_PATH), "rb") as f:
            st.video(f.read())
    except Exception as e:
        st.error(f"Error loading video player: {e}")
        
    # Loop through the range of frames you want to display
    if batchButton:
        data = pd.DataFrame()
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        if not cap.isOpened():
            st.error("OpenCV could not open the file. Check if it's a valid video.")
        else:
            # Get actual frame count from the opened file
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # ดึงค่า FPS ของวิดีโอ
            video_fps = cap.get(cv2.CAP_PROP_FPS) 
            # คำนวณจำนวนวินาทีทั้งหมด
            total_seconds = int(total_frames / video_fps)
            st.markdown(f"Info: {total_seconds:.2f}s | FPS: {video_fps} | {total_frames} frames")
            #สร้าง UI Elements เตรียมไว้ก่อน
            col_a, col_b, col_c= st.columns([0.4,0.3,0.3])
            with col_a:image_placeholder = st.empty()
            with col_b:crop_image_placeholder = st.empty()
            with col_c:cat_image_placeholder = st.empty()
            col_d, col_e= st.columns([0.5,0.5])
            with col_d:
                specular_only_placeholder = st.empty()
                gloss_percent_placeholder = st.empty()
                specular_rgb_placeholder = st.empty()
            with col_e:
                replaced_img_placeholder = st.empty()
                rgb_placeholder = st.empty()
                lab_placeholder = st.empty()
                    
            progress_bar = st.progress(0)
            table_placeholder = st.empty()
                
            for sec in range(total_seconds+1):
                # คำนวณหา index ของเฟรมที่อยู่ที่วินาทีนั้นๆ
                frame_id = int(sec * video_fps)
                # สั่งให้ OpenCV กระโดดไปที่เฟรมนั้น
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, frame_bgr = cap.read()
                if success:
                        # อัปเดต Progress ตามสัดส่วนวินาที
                        progress = sec / total_seconds
                        progress_bar.progress(progress,text=f"Processing: seconds {sec} / {total_seconds} (Frame {frame_id})")

                        # ประมวลผล Image
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        cropped_bgr = detect_rotate_crop(frame_bgr, pad_x_pct=pad_x, pad_y_pct=pad_y, shrink=shrink_val)
                        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
                        if cat_method == "custom":
                            cat_bgr = process_cat_logic(cropped_bgr, cat_method,None,None,light_source,light_target)        
                        else:
                            processed_full_bgr = process_cat_logic(frame_bgr, cat_method,white_patch_lower,white_patch_upper,None,None)            
                            if processed_full_bgr is not None:
                            # ทำ Object Detection (Crop) 
                                cat_bgr = detect_rotate_crop(processed_full_bgr, pad_x_pct=pad_x, pad_y_pct=pad_y, shrink=shrink_val)
                            # ตรวจสอบกรณี Detect ไม่เจอ
                            if cat_bgr is None:
                                st.error("🎯 Detection failed on processed image. Show full frame instead.")
                                cat_bgr = processed_full_bgr
                        cat_rgb = cv2.cvtColor(cat_bgr, cv2.COLOR_BGR2RGB)
                        # รัน K-means และค้นหา Gloss
                        centroids, labels = kmeans(cat_bgr, clustering_threshold, max_kmeans_iterations)
                        gloss_percent, gloss_label = detect_gloss(centroids, labels)
                        
                        # เตรียม Mask
                        mask_gloss = (labels == gloss_label)
                        mask_not_gloss = (labels != gloss_label)
                        
                        # ค่า Gloss Percentage และ ค่าเฉลี่ย RGB ของ Gloss (specular)
                        specular_pixels_bgr = cat_bgr[mask_gloss]
                        specular_pixels_rgb = specular_pixels_bgr[:, ::-1]
                        average_BGR_specular = specular_pixels_bgr.mean(axis=0).astype(np.uint8)
                        average_RGB_specular = average_BGR_specular[::-1]
                        
                        # ค่า ค่าเฉลี่ย RGB ของ Diffuse (ส่วนที่ไม่ใช่ Gloss)
                        diffuse_pixels_bgr = cat_bgr[mask_not_gloss]
                        diffuse_pixels_rgb = diffuse_pixels_bgr[:, ::-1]
                        average_BGR_diffuse = diffuse_pixels_bgr.mean(axis=0).astype(np.uint8)
                        average_RGB_diffuse = average_BGR_diffuse[::-1]
                        
                        # --- สร้างรูปที่: บริเวณเฉพาะที่เป็น Gloss (Masked Image) ---
                        specular_only = np.zeros_like(cat_bgr)
                        specular_only[mask_gloss] = cat_bgr[mask_gloss]
                        specular_only_rgb = cv2.cvtColor(specular_only, cv2.COLOR_BGR2RGB)
                        # --- สร้างรูปที่: แทนที่ Gloss ด้วย Mean ของ Not-Gloss ---
                        replaced_img = cat_bgr.copy()
                        if np.any(mask_not_gloss):
                            # คำนวณค่าสีเฉลี่ยของส่วนที่ไม่ใช่ Gloss (BGR) ไว้แล้ว = average_BGR_diffuse
                            # แทนที่บริเวณ Gloss ด้วยค่าเฉลี่ยนั้น
                            replaced_img[mask_gloss] = average_BGR_diffuse
                            replaced_img_rgb = cv2.cvtColor(replaced_img, cv2.COLOR_BGR2RGB)


                        input_data = prepare_input(average_RGB_diffuse)
                        # ทำการ Predict
                        prediction = model.predict(input_data)
                        # แสดงผลลัพธ์
                        pred_L = round(float(prediction[0][0] * 100.0), 2)
                        pred_a = round(float((prediction[0][1] * 240.0) - 120.0), 2)
                        pred_b = round(float((prediction[0][2] * 240.0) - 120.0), 2)
                        
                        rgb_placeholder.metric(label="Specular Replaced by Mean Diffuse", value=f"RBG: {average_RGB_diffuse}")
                        lab_placeholder.metric(label=f"AI Prediction Result", value=f"L\*a\*b\*: [{pred_L} {pred_a} {pred_b}]")
                        
                        gloss_percent_placeholder.metric(label="Gloss Percentage", value=f"{gloss_percent * 100:.2f}%")
                        specular_rgb_placeholder.metric(label="Mean Specular Area", value=f"RBG: {average_RGB_specular}")
                        
                        # แสดงผลแบบเขียนทับที่เดิม (Placeholder)
                        image_placeholder.image(frame_rgb, caption=f"Time: {sec}s (Frame {frame_id})", width="stretch")                    
                        crop_image_placeholder.image(cropped_rgb, caption=f"crop_image")                    
                        cat_image_placeholder.image(cat_rgb, caption=f"cat_image")                    
                        specular_only_placeholder.image(specular_only_rgb, caption=f"specular_only_image")  
                        replaced_img_placeholder.image(replaced_img_rgb, caption=f"replaced_specular_image")  


                        new_row_data = [{
                            'sec': sec, 
                            'frame_index': frame_id, 
                            'diffuse_rgb_avg':average_RGB_diffuse,
                            'predict_CIELAB':[pred_L,pred_a,pred_b],
                            'specular_rgb_avg': average_RGB_specular,
                            'gloss_percent': gloss_percent
                            }]
                        new_row_df = pd.DataFrame(new_row_data)
                        data = pd.concat([data,new_row_df], ignore_index=True)

                        # 2. อัปเดตตารางในตำแหน่งเดิม
                        # สมมติ confusion_matrix ของคุณมีการเปลี่ยนแปลงใน loop นี้
                        table_placeholder.dataframe(data,hide_index=True)
                else:
                    break

            st.sidebar.success("✅ ประมวลผลเสร็จสิ้น!")            
            cap.release()
            
else:
    st.info("👈 Please start at Module1 to begin.")
    
    



