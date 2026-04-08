import os
import pandas as pd
import streamlit as st
from pathlib import Path
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    forecast_steps = st.slider("เลือกเวลาที่ต้องการพยากรณ์ล่วงหน้า (วินาที / Sec)", min_value=0, max_value=60, value=10)

    # ==========================================
    # ส่วนของ FORECASTING
    # ==========================================
    st.header("🔮 Forecasting (พยากรณ์แนวโน้มในอนาคต)")
    st.info("ใช้อัลกอริทึม **Holt's Linear Trend (Exponential Smoothing)** ข้อมูลจริงจะเป็น **สีเข้ม** ส่วนข้อมูลพยากรณ์จะเป็น **สีสว่าง**")

    target_cols = [
        'diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b',
        'predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b',
        'specular_avg_r', 'specular_avg_g', 'specular_avg_b',
        'gloss_percent', 'dE'
    ]
    
    if forecast_steps > 0:
        with st.spinner("กำลังคำนวณโมเดล Forecasting..."):
            
            last_sec = df.index[-1]
            future_index = np.arange(last_sec + 1, last_sec + 1 + forecast_steps)
            forecast_df = pd.DataFrame(index=future_index, columns=target_cols)

            for col in target_cols:
                series = df[col].dropna()
                
                if len(series) < 3:
                    forecast_df[col] = [series.iloc[-1]] * forecast_steps
                else:
                    try:
                        model = ExponentialSmoothing(
                            series, 
                            trend='add', 
                            seasonal=None, 
                            initialization_method="heuristic"
                        )
                        fit_model = model.fit()
                        forecast_df[col] = fit_model.forecast(forecast_steps).values
                    except Exception as e:
                        st.warning(f"ไม่สามารถคำนวณ Forecasting สำหรับ {col} ได้")
                        forecast_df[col] = [series.iloc[-1]] * forecast_steps

            # 1. ดึงจุดสุดท้ายของข้อมูลจริงมาเป็นจุดเริ่มต้นของข้อมูลพยากรณ์ เพื่อให้เส้นกราฟเชื่อมต่อกัน
            last_actual_data = df.loc[[last_sec], target_cols]
            forecast_df = pd.concat([last_actual_data, forecast_df])
            
            # 2. เปลี่ยนชื่อคอลัมน์ส่วนที่เป็นพยากรณ์ (เพื่อแยกสีตอนพล็อต)
            forecast_cols_mapping = {col: f"{col}_forecast" for col in target_cols}
            forecast_df = forecast_df.rename(columns=forecast_cols_mapping)
            
            # 3. Join ข้อมูลเข้าด้วยกัน
            plot_df = df[target_cols].join(forecast_df, how='outer')
            
            # รวม gloss_percent เพื่อใช้กำหนดขนาดจุดใน scatter plot
            plot_df['gloss_percent_combined'] = plot_df['gloss_percent'].fillna(plot_df['gloss_percent_forecast'])
            df = plot_df  # อัปเดต df ให้รวมข้อมูลพยากรณ์ด้วย

        # --- วาดกราฟแบบแยกสี (ข้อมูลจริง vs ข้อมูลพยากรณ์) ---
        st.markdown("### 🔴🟢🔵 Diffuse RGB Values")
        st.line_chart(df[[
            'diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b',
            'diffuse_avg_r_forecast', 'diffuse_avg_g_forecast', 'diffuse_avg_b_forecast'
        ]], color=['red', 'green', 'blue', "#FF6666", '#66FF66', '#6666FF']) 

        st.markdown("### 🎨 Predicted CIELAB")
        st.line_chart(df[[
            'predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b',
            'predict_CIELAB_L_forecast', 'predict_CIELAB_a_forecast', 'predict_CIELAB_b_forecast'
        ]], color=['gray', 'red', 'yellow', '#CCCCCC', '#FF6666', '#FFFF66'])

        st.markdown("### ✨ Specular RGB Values and Gloss Percent")
        st.scatter_chart(df,
            y=[
                'specular_avg_r', 'specular_avg_g', 'specular_avg_b',
                'specular_avg_r_forecast', 'specular_avg_g_forecast', 'specular_avg_b_forecast'
            ],
            color=['red', 'green', 'blue', "#FF6666", '#66FF66', '#6666FF'], 
            size='gloss_percent_combined'
        )
        df.drop(columns=['gloss_percent_combined'], inplace=True)  # ลบคอลัมน์ชั่วคราวออกหลังใช้แล้ว

        st.markdown("### 📈 Color Difference (dE)")
        st.line_chart(df[['dE', 'dE_forecast']], color=['#0000FF', '#FF00FF'])

        st.markdown("### 📋 ตารางข้อมูล (รวมข้อมูลพยากรณ์)")
        st.dataframe(df)
        
    else:    
        st.markdown("### 🔴🟢🔵 Diffuse RGB Values")
        st.line_chart(df[['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b']],color=['red', 'green', 'blue'])

        st.markdown("### 🎨 Predicted CIELAB")
        st.line_chart(df[['predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b']],color=['gray', 'red', 'yellow'])

        st.markdown("### ✨ Specular RGB Values and Gloss Percent")
        st.scatter_chart(df,y=['specular_avg_r', 'specular_avg_g', 'specular_avg_b'],color=['red', 'green', 'blue'], size='gloss_percent')
        
        st.markdown("### 📈 Color Difference (dE)")
        st.line_chart(df[['dE']])

        st.markdown("### 📋 ตารางข้อมูลดิบ (Raw Data)")
        st.dataframe(df)

else:
    st.error("ไม่พบคอลัมน์ 'sec' ในชุดข้อมูล กรุณาตรวจสอบไฟล์ CSV ของคุณ")