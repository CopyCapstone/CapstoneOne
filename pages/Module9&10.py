import os
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
from Module_09_Forecasting.forecast import forecast_series, compute_metrics
warnings.filterwarnings("ignore")

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
SETTING_FILE = TMP_DIR / 'settings.json'
VIDEO_PATH = TMP_DIR / "uploaded_video" / "uploaded_video.mp4"
DATA_CSV_PATH = TMP_DIR / "dataframe" / "batch_processing_results.csv"
FORECAST_OUTPUT_CSV_PATH = TMP_DIR / "dataframe" / "forecast_results.csv"
MIN_DEGREE, MAX_DEGREE = 1, 5
LONGGEST_FORECAST = 600 # วินาที (10 นาที)

try:
    os.makedirs(DATA_CSV_PATH.parent, exist_ok=True)
    os.makedirs(FORECAST_OUTPUT_CSV_PATH.parent, exist_ok=True)
    
    df = pd.read_csv(DATA_CSV_PATH)
    
    st.title("📊 Module 9 & 10: Forecasting and Visualization")
    st.divider()
    
    FORECAST_COLUMNS = [
        'diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b',
        'estimated_CIELAB_L', 'estimated_CIELAB_a', 'estimated_CIELAB_b',
        'specular_avg_r', 'specular_avg_g', 'specular_avg_b',
        'gloss_percent', 'dE'
    ]
    
    COLUMN_LABELS = {
        'diffuse_avg_r':     'Diffuse R',
        'diffuse_avg_g':     'Diffuse G',
        'diffuse_avg_b':     'Diffuse B',
        'estimated_CIELAB_L':  'CIELAB L*',
        'estimated_CIELAB_a':  'CIELAB a*',
        'estimated_CIELAB_b':  'CIELAB b*',
        'specular_avg_r':    'Specular R',
        'specular_avg_g':    'Specular G',
        'specular_avg_b':    'Specular B',
        'gloss_percent':     'Gloss (%)',
        'dE':                'ΔE (Color Diff)',
    }

    # ════════════════════════════════════════════════════════════════════════════
    if 'sec' not in df.columns:
        st.error("ไม่พบคอลัมน์ 'sec' ในชุดข้อมูล กรุณาตรวจสอบไฟล์ CSV ของคุณ")
        st.stop()

    df = df.set_index('sec')

    # ─── Section 1: Original Charts ─────────────────────────────────────────────
    st.markdown("## 📈 Observed Data")
    st.markdown("### 🔴🟢🔵 Diffuse RGB Values")
    st.line_chart(df[['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b']],
                color=['#FF0000', '#00CC00', '#0000FF'])
    st.markdown("### 🎨 Estimated CIELAB")
    st.line_chart(df[['estimated_CIELAB_L', 'estimated_CIELAB_a', 'estimated_CIELAB_b']],
                color=['#888888', '#FF0000', '#DDCC00'])
    st.markdown("### ✨ Specular RGB Values and Gloss Percent")
    st.scatter_chart(df,y=['specular_avg_r', 'specular_avg_g', 'specular_avg_b'],color=['#FF0000', '#00CC00', '#0000FF'],size='gloss_percent')
    st.markdown("### 📈 Color Difference (ΔE)")
    st.line_chart(df[['dE']])
    st.markdown("### 📋 ตารางข้อมูลดิบ (Raw Data)")
    st.dataframe(df, width='stretch')
    st.divider()

    # ─── Section 2: Forecasting ─────────────────────────────────────────────────
    st.markdown("## 🔮 Forecasting")
    # --- ส่วนทฤษฎี ---
    with st.expander("อธิบาย Polynomial Regression"):
        st.write("### 1. Polynomial Features")
        st.write("คือการสร้าง 'ตัวแปรใหม่' จากเวลา ($x$) เดิมที่มีอยู่ เพื่อให้โมเดลสามารถสร้างเส้นโค้งได้:")
        st.latex(r"X_{transformed} = [x^0, x^1, x^2, x^3, ..., x^n]")
        
        st.write("### 2. Linear Regression")
        st.write("เมื่อเราได้ตัวแปรยกกำลังมาแล้ว โมเดลจะหาค่า beta (weights) ที่เหมาะสมที่สุด:")
        st.latex(r"y = \beta_0 + \beta_1x + \beta_2x^2 + ... + \beta_nx^n")
        
        st.info("""
        💡 **หลักการง่ายๆ:** 
        - **Degree 1:** คือเส้นตรงธรรมดา ($y = ax + b$)
        - **Degree 2:** คือเส้นโค้งพาราโบลา (U-shape หรือคว่ำ)
        - **Degree 3 ขึ้นไป:** เส้นจะเริ่มมีความหยักมากขึ้นตามข้อมูล
        """)

        # --- Sidebar สำหรับปรับค่า Parameter ---
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🕹️ ปรับแต่งค่า Parameters")
            dump_degree = st.slider("Degree (ระดับความโค้ง)", MIN_DEGREE, MAX_DEGREE, 2)
            st.caption("ยิ่ง Degree สูง เส้นจะยิ่งโค้งตามจุดข้อมูลได้มากขึ้น")
            if dump_degree == 1:
                st.caption("- **โหมดเส้นตรง:** เน้นดูแนวโน้ม (Trend) ภาพรวมว่าขึ้นหรือลงแบบคงที่")
            elif dump_degree <= 3:
                st.caption("- **โหมดเส้นโค้ง:** เริ่มปรับตัวตามความเร่งของข้อมูลได้มากขึ้น")
            else:
                st.caption("- **โหมดซับซ้อน:** ระวังการเกิด *Overfitting* (เส้นพยายามผ่านทุกจุดจนทำนายอนาคตเพี้ยน)")

        with col2:
            st.subheader("⚙️ ตั้งค่าข้อมูล")
            # ปรับปรุง input ให้รองรับข้อมูลที่สะท้อนความโค้ง
            actual_data_str = st.text_input("Actual Data (Yt) แยกด้วยเครื่องหมายคอมมา", "0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181")
            dump_forecast_steps = st.number_input("forecast_steps", 1, 10, 3)
            
        st.divider()

        # --- ส่วนการคำนวณ (ใช้ Logic ของ Sklearn) ---
        Y_actual = [float(x.strip()) for x in actual_data_str.split(",")]
        n = len(Y_actual)

        X = np.arange(n).reshape(-1, 1)
        y = np.array(Y_actual)

        # 2. สร้าง Features และ Fit Model
        poly = PolynomialFeatures(degree=dump_degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)

        # 3. สร้างข้อมูลสำหรับการแสดงผล (Historical + Forecast)
        X_all = np.arange(n + dump_forecast_steps).reshape(-1, 1)
        X_all_poly = poly.transform(X_all)
        y_all_pred = model.predict(X_all_poly)

        # เตรียม DataFrame สำหรับตาราง
        results = []
        for t in range(n + dump_forecast_steps):
            actual = Y_actual[t] if t < n else None
            pred = y_all_pred[t]
            
            results.append({
                "Time (t)": t,
                "Actual (Yt)": actual,
                "Forecast (Ft)": round(pred, 4),
                "Type": "Historical" if t < n else "Forecast"
            })

        tmp_df = pd.DataFrame(results)

        # --- ส่วนแสดงผล ---
        col1, col2 = st.columns([1.5, 1.2])

        with col1:
            st.write("### 📊 ตารางข้อมูล")
            st.dataframe(tmp_df.style.highlight_null(color="#f0f0f0"), width='stretch', hide_index=True)
            
            # แสดงสมการที่โมเดลสร้างขึ้น
            b0 = model.intercept_
            coeffs = model.coef_ # Note: b0 จะอยู่ใน intercept_ ถ้า include_bias=True ใน poly อาจต้องระวัง
            # เพื่อความชัดเจนในการสอน:
            eq_text = f"y = {model.intercept_:.2f}"
            for i, c in enumerate(model.coef_, 1):
                eq_text += f" + ({c:.4f} \cdot x^{i})"
            st.latex(eq_text)
            # R-squared
            st.latex(r"R^2 = {:.4f}".format(model.score(X_poly, y)))

        with col2:
            st.write("### 📈 กราฟแสดงผล")
            fig = go.Figure()

            # เส้นค่าจริง
            fig.add_trace(go.Scatter(x=tmp_df["Time (t)"], y=tmp_df["Actual (Yt)"], 
                                    mode='markers', name='ค่าจริง (Actual)',
                                    marker=dict(color='blue', size=10)))

            # เส้นพยากรณ์ (ลากผ่านทั้งอดีตและอนาคตเพื่อให้เห็น Regression Line)
            fig.add_trace(go.Scatter(x=tmp_df["Time (t)"], y=tmp_df["Forecast (Ft)"], 
                                    mode='lines+markers', name='Polynomial Fit',
                                    line=dict(color='orange', dash='dash')))

            fig.update_layout(xaxis_title="เวลา (t)", yaxis_title="ค่าที่ได้",
                            hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig)
            
    actual_step = np.median(np.diff(df.index)) if len(df) > 1 else 1 # การันตี step size เท่ากันทุกแถว 

    with st.sidebar:
        # --- Controls ---
        st.header("⚙️ Forecasting Settings")
        st.markdown("""
            **Forecasting Method : Polynomial Regression:**
            """
        )
        st.markdown(f"Details of processing (seconds/processing): {actual_step}")
        forecast_iterations = st.slider(f"Forecasting Target (step) n*{actual_step}", 1, int(LONGGEST_FORECAST/actual_step), 1)
        st.caption(f"หมายถึง Forecasting ถึง {forecast_iterations * actual_step} วินาทีข้างหน้า")
        degree = st.slider("Degree of Forecasting Equation", MIN_DEGREE, MAX_DEGREE, 2)
        selected_cols = st.multiselect(
            "Variables to Forecast",
            FORECAST_COLUMNS,
            default=['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b', 'dE']
        )

        if not selected_cols:
            st.warning("กรุณาเลือกอย่างน้อย 1 ตัวแปร")
            st.stop()

    # Time axis
    last_sec   = df.index[-1]
    future_idx = [last_sec + (i + 1) * actual_step for i in range(forecast_iterations)]
    full_idx   = list(df.index) + future_idx

    # ─── Run forecasts & plot ────────────────────────────────────────────────────
    metrics_rows = []
    # Create a dict to store forecasted values for the table
    forecast_dict = {"sec (forecast)": full_idx}

    for col in selected_cols:
        series = df[col] # การันตีไม่มี .dropna() ใน df

        # 1. พยากรณ์อนาคตของจริง (ใช้ข้อมูล 100% เต็ม) ตั้งแต่จุดเริ่มต้นจนถึงอนาคตที่ต้องการ (0-N) เพื่อให้เห็นแนวโน้มของเส้นโค้ง
        full_forecast_values, forecast_values, model_info = forecast_series(series, forecast_iterations, degree=degree)
        # Store forecasted values in output dict for table
        forecast_dict[COLUMN_LABELS.get(col, col)] = np.round(full_forecast_values, 2)

        # Confidence band (±2 std)
        # 1. ดึงค่าพยากรณ์เฉพาะช่วงที่เป็นข้อมูลในอดีต (Historical)
        hist_predicted = full_forecast_values[:len(series)]
        
        # 2. คำนวณ Residuals (ความคลาดเคลื่อน)
        residuals = series.values - hist_predicted
        
        # 3. คำนวณ STD ของ Residuals
        residual_std = np.std(residuals)
        
        # 4. สร้าง Upper / Lower Bound (±2 STD ของ Residuals)
        upper = full_forecast_values + (2 * residual_std)
        lower = full_forecast_values - (2 * residual_std)

        # ── Plotly chart ──────────────────────────────────────────────────────
        fig = go.Figure()

        # Observed
        fig.add_trace(go.Scatter(
            x=list(series.index), y=series.values,
            mode='lines+markers',
            name='Observed',
            line=dict(color='royalblue', width=2),
            marker=dict(size=7)
        ))

        # Forecast 0-N (ลากผ่านทั้งอดีตและอนาคตเพื่อให้เห็น Regression Line)
        fig.add_trace(go.Scatter(
            # x=future_idx, y=fcast_vals,
            x=full_idx, y=full_forecast_values, 
            mode='lines+markers',
            name='Forecast',
            line=dict(color='tomato', width=2, dash='dash'),
            marker=dict(size=7, symbol='diamond')
        ))
        

        fig.add_trace(go.Scatter(
            x=full_idx + full_idx[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(255,99,71,0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=True
        ))
        # Divider line
        fig.add_vline(x=last_sec, line_dash="dot", line_color="gray", annotation_text="Last Observed", annotation_position="top right")

        fig.update_layout(
            title=f"{COLUMN_LABELS.get(col, col)}  |  Polynomial Regression",
            xaxis_title="Time (sec)",
            yaxis_title=COLUMN_LABELS.get(col, col),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, width='stretch')

    # 2. ทำ Back-testing ด้วยข้อมูล 30% สุดท้าย เพื่อหา MAE, RMSE, MAPE
    
        # compare degree ที่ใช้ในการพยากรณ์กับจำนวนข้อมูลที่มี เพื่อป้องกัน Overfitting หรือ Underfitting
        for d in range(MIN_DEGREE, MAX_DEGREE + 1):
            try:
                # หาจุดตัด (Split Index) ที่ 70%
                split_idx = int(len(series) * 0.7)
                
                train_series = series.iloc[:split_idx]
                test_series = series.iloc[split_idx:]
                test_steps = len(test_series)
                
                full_preds, bt_predicted, model_info = forecast_series(train_series, test_steps, degree=d)
                
                bt_actual = test_series.values
                
                # เตรียมข้อมูลสำหรับหา Train R-squared
                train_actual = train_series.values
                train_predicted = full_preds[:len(train_actual)] # ดึงผลทำนายเฉพาะช่วงข้อมูล Train
                
                # ป้องกันกรณีเกิด Error จากความยาวของ Array ไม่เท่ากัน
                if len(bt_actual) == len(bt_predicted):
                    mae, rmse, mape = compute_metrics(bt_actual, bt_predicted)
                    
                    # คำนวณ R-squared สำหรับ Train และ Test
                    train_r2 = r2_score(train_actual, train_predicted)
                    test_r2 = r2_score(bt_actual, bt_predicted)
                else:
                    mae, rmse, mape = float('nan'), float('nan'), float('nan')
                    train_r2, test_r2 = float('nan'), float('nan')
                    
            except Exception as e:
                mae, rmse, mape = float('nan'), float('nan'), float('nan')
                train_r2, test_r2 = float('nan'), float('nan')
                model_info = "Error" # ป้องกัน Error กรณีที่ model_info ยังไม่ถูกสร้าง
                
            metrics_rows.append({
                "Variable":  COLUMN_LABELS.get(col, "Unknown"), # แนะนำให้ใส่ default value ไว้เผื่อหาไม่เจอ
                "MAE":       round(mae, 2) if not np.isnan(mae) else "N/A",
                "RMSE":      round(rmse, 2) if not np.isnan(rmse) else "N/A",
                "MAPE (%)":  round(mape, 2) if not np.isnan(mape) else "N/A",
                "Train R-squared": round(train_r2, 4) if not np.isnan(train_r2) else "N/A",
                "Test R-squared":  round(test_r2, 4) if not np.isnan(test_r2) else "N/A",
                "Degree Used": d,
                "Model Info": model_info

            })

    # ─── Forecast Table ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Forecasted Table")
    forecast_out_df = pd.DataFrame(forecast_dict).set_index("sec (forecast)")
    st.dataframe(forecast_out_df, width='stretch')
    forecast_out_df.to_csv(FORECAST_OUTPUT_CSV_PATH)

    # ─── Metrics Table ───────────────────────────────────────────────────────────
    if metrics_rows:
        st.divider()
        st.markdown("### 📐 Forecast Accuracy Metrics \n"
            "- **MAE** (Mean Absolute Error)\n"
            "- **RMSE** (Root Mean Squared Error)\n"
            "- **MAPE** (Mean Absolute Percentage Error)\n"
            "- **R²** (Coefficient of Determination)"
        )

        with st.expander("📏 Forecast Accuracy Metrics"):
            metrics_df = pd.DataFrame(metrics_rows).set_index("Variable")

            # show sec ที่ใช้ใน train (70% ของข้อมูล)
            # show sec ที่ใช้ใน test (30% ของข้อมูล)
            total_len = len(df)
            split_idx = int(total_len * 0.7)
            train_sec_str = ", ".join(map(str, df.index[:split_idx]))
            test_sec_str = ", ".join(map(str, df.index[split_idx:]))

            st.markdown(f"**Train Seconds (70%):** {train_sec_str}")
            st.markdown(f"**Test Seconds (30%):** {test_sec_str}")
            
            # split to multi table by "Degree Used" เพื่อให้ดูง่ายขึ้น
            for degree_used in metrics_df["Degree Used"].unique():
                st.markdown(f"Degree {degree_used}")
                st.dataframe(metrics_df[metrics_df["Degree Used"] == degree_used], width='stretch')  
except Exception as e:
    st.error(f"ERROR: {str(e)}")
    # st.switch_page("pages/Module7&8.py")
