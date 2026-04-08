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
OUTPUT_CSV_PATH = TMP_DIR / "dataframe" / "batch_processing_results.csv"
os.makedirs(OUTPUT_CSV_PATH.parent, exist_ok=True)
df = pd.read_csv(OUTPUT_CSV_PATH)

st.title("📊 Module 9 & 10: Forecasting and Visualization")
st.divider()

# ตรวจสอบว่ามีคอลัมน์ 'sec' หรือไม่
if 'sec' in df.columns:
    df = df.set_index('sec') # ตั้งค่าให้ sec เป็นแกน X
    
    st.markdown("### 🔴🟢🔵 Diffuse RGB Values")
    st.line_chart(df[['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b']],
                  color=['red', 'green', 'blue'])

    st.markdown("### 🎨 Predicted CIELAB")
    st.line_chart(df[['predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b']],
                  color=['gray', 'red', 'yellow'])

    st.markdown("### ✨ Specular RGB Values and Gloss Percent")
    st.scatter_chart(
        df,
        y=['specular_avg_r', 'specular_avg_g', 'specular_avg_b'],
        color=['red', 'green', 'blue'], 
        size='gloss_percent'
    )
    st.markdown("### 📈 Color Difference (dE)")
    st.line_chart(df[['dE']])

    st.markdown("### 📋 ตารางข้อมูลดิบ (Raw Data)")
    st.dataframe(df)

else:
    st.error("ไม่พบคอลัมน์ 'sec' ในชุดข้อมูล กรุณาตรวจสอบไฟล์ CSV ของคุณ")