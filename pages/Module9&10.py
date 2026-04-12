import os
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import warnings
warnings.filterwarnings("ignore")

# --- Configuration & Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"
SETTING_FILE = TMP_DIR / 'settings.json'
VIDEO_PATH = TMP_DIR / "uploaded_video" / "uploaded_video.mp4"
DATA_CSV_PATH = TMP_DIR / "dataframe" / "batch_processing_results.csv"
FORECAST_OUTPUT_CSV_PATH = TMP_DIR / "dataframe" / "forecast_results.csv"

os.makedirs(DATA_CSV_PATH.parent, exist_ok=True)
os.makedirs(FORECAST_OUTPUT_CSV_PATH.parent, exist_ok=True)

df = pd.read_csv(DATA_CSV_PATH)

st.title("📊 Module 9 & 10: Forecasting and Visualization")
st.divider()

FORECAST_COLUMNS = [
    'diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b',
    'predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b',
    'specular_avg_r', 'specular_avg_g', 'specular_avg_b',
    'gloss_percent', 'dE'
]

COLUMN_LABELS = {
    'diffuse_avg_r':     'Diffuse R',
    'diffuse_avg_g':     'Diffuse G',
    'diffuse_avg_b':     'Diffuse B',
    'predict_CIELAB_L':  'CIELAB L*',
    'predict_CIELAB_a':  'CIELAB a*',
    'predict_CIELAB_b':  'CIELAB b*',
    'specular_avg_r':    'Specular R',
    'specular_avg_g':    'Specular G',
    'specular_avg_b':    'Specular B',
    'gloss_percent':     'Gloss (%)',
    'dE':                'ΔE (Color Diff)',
}

# ─── Helper: run one forecasting model ──────────────────────────────────────
def forecast_series(series: pd.Series, steps: int):
    """
    Returns (forecast_values, model_info_str)
    Handles short series gracefully.
    """
    n = len(series)
    trend   = "add" if n >= 4 else None
    model = ExponentialSmoothing(series, trend=trend, seasonal=None,initialization_method="estimated")
    fit = model.fit(optimized=True)
    forecast = fit.forecast(steps)
    info = (f"Holt-Winters" " | " f"α={fit.params.get('smoothing_level', float('nan')):.4f}" " | " f"β={fit.params.get('smoothing_trend', float('nan')):.4f}" )
    return forecast.values, info


# ─── Metrics helper ─────────────────────────────────────────────────────────
def compute_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else float('nan')
    return mae, rmse, mape


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
st.line_chart(df[['predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b']],
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
with st.expander("อธิบาย Holt-Winters Exponential Smoothing ดูสมการที่ใช้คำนวณ เฉพาะส่วน Level (α) และ Trend (β)"):
    st.write("### 1. สมการระดับ (Level - $L_t$)")
    st.latex(r"L_t = \alpha Y_t + (1 - \alpha)(L_{t-1} + T_{t-1})")
    
    st.write("### 2. สมการแนวโน้ม (Trend - $T_t$)")
    st.latex(r"T_t = \beta(L_t - L_{t-1}) + (1 - \beta)T_{t-1}")
    
    st.write("### 3. การพยากรณ์ไปข้างหน้า (Forecast - $F_{t+m}$)")
    st.latex(r"F_{t+m} = L_t + (m \times T_t)")

    # --- Sidebar สำหรับปรับค่า Parameter ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕹️ ปรับแต่งค่า Parameters")
        alpha = st.slider("ค่า Alpha (α) - สำหรับ Level", 0.0, 1.0, 0.5, 0.05)
        beta = st.slider("ค่า Beta (β) - สำหรับ Trend", 0.0, 1.0, 0.5, 0.05)
    with col2:
        st.subheader("🔍 **วิเคราะห์ผล:**")
        st.write(f"- ขณะนี้ใช้ **$\\alpha = {alpha}$**: {'ให้ความสำคัญกับข้อมูลใหม่สูง' if alpha > 0.7 else 'เน้นความเรียบของข้อมูล' if alpha < 0.3 else 'สมดุลระหว่างข้อมูลเก่าและใหม่'}")
        st.write(f"- ขณะนี้ใช้ **$\\beta = {beta}$**: {'ปรับตัวตาม Trend ได้ไวมาก' if beta > 0.7 else 'Trend ค่อนข้างคงที่' if beta < 0.3 else 'ปรับตัวตามแนวโน้มปานกลาง'}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ ค่าเริ่มต้น (Initial Values)")
        l0 = st.number_input("ค่าระดับเริ่มต้น (L0)", value=100.0)
        t0 = st.number_input("ค่าแนวโน้มเริ่มต้น (T0)", value=2.0)
    with col2:
        st.subheader("📊 **ข้อมูลตัวอย่าง:**")
        actual_data_str = st.text_input("ค่าจริง (Yt) แยกด้วยเครื่องหมายคอมมา", "110, 115")
        Y_actual = [float(x.strip()) for x in actual_data_str.split(",")]
        forecast_steps = st.number_input("จำนวนงวดที่พยากรณ์ล่วงหน้า", 1, 10, 2)

    # --- ส่วนการคำนวณ ---
    results = []
    # ค่าที่ t=0
    current_L = l0
    current_T = t0
    results.append({
        "Time (t)": 0,
        "Actual (Yt)": None,
        "Level (Lt)": current_L,
        "Trend (Tt)": current_T,
        "Forecast (Ft)": None,
        "Type": "Initial"
    })

    # คำนวณช่วงที่มีค่าจริง (Historical)
    for t, y_val in enumerate(Y_actual, 1):
        # Forecast สำหรับ t ปัจจุบัน (ใช้ค่าจาก t-1)
        prev_L = results[t-1]["Level (Lt)"]
        prev_T = results[t-1]["Trend (Tt)"]
        f_t = prev_L + prev_T
        
        # คำนวณ Level และ Trend ใหม่
        new_L = alpha * y_val + (1 - alpha) * (prev_L + prev_T)
        new_T = beta * (new_L - prev_L) + (1 - beta) * prev_T
        
        results.append({
            "Time (t)": t,
            "Actual (Yt)": y_val,
            "Level (Lt)": round(new_L, 4),
            "Trend (Tt)": round(new_T, 4),
            "Forecast (Ft)": round(f_t, 4),
            "Type": "Historical"
        })
        current_L, current_T = new_L, new_T

    # คำนวณช่วงอนาคต (Future Forecast)
    last_t = len(Y_actual)
    for m in range(1, forecast_steps + 1):
        f_future = current_L + (m * current_T)
        results.append({
            "Time (t)": last_t + m,
            "Actual (Yt)": None,
            "Level (Lt)": None,
            "Trend (Tt)": None,
            "Forecast (Ft)": round(f_future, 4),
            "Type": "Forecast"
        })

    tmp_df = pd.DataFrame(results)

    # --- ส่วนแสดงผล ---
    col1, col2 = st.columns([1.5, 1.2])

    with col1:
        st.write("### 📊 ตารางเปรียบเทียบ")
        st.dataframe(tmp_df.style.highlight_null(color="#f0f0f0"), width='stretch', hide_index = True)

    with col2:
        st.write("### 📈 กราฟแสดงผล")
        fig = go.Figure()

        # เส้นค่าจริง
        fig.add_trace(go.Scatter(x=tmp_df["Time (t)"], y=tmp_df["Actual (Yt)"], 
                                mode='lines+markers', name='ค่าจริง (Actual)',
                                line=dict(color='blue', width=3)))

        # เส้นค่าพยากรณ์
        fig.add_trace(go.Scatter(x=tmp_df["Time (t)"], y=tmp_df["Forecast (Ft)"], 
                                mode='lines+markers', name='ค่าพยากรณ์ (Forecast)',
                                line=dict(color='orange', dash='dash')))

        fig.update_layout(xaxis_title="เวลา (t)", yaxis_title="ยอดขาย / ค่า",
                        hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, width='stretch')


with st.sidebar:
    # --- Controls ---
    st.header("⚙️ Forecasting Settings")
    st.markdown("""
        **วิธีการ Forecasting ที่ใช้วิธี Holt-Winters Exponential Smoothing:**
        """
    )
    st.markdown(f"Details of processing (seconds/processing): {st.session_state['stored_step']}")
    forecast_steps = st.slider("Forecasting Target (seconds)", 1, 600, 6)
    st.markdown("Forecasting Method : Holt-Winters (Exponential Smoothing)")
    selected_cols = st.multiselect(
        "Variables to Forecast",
        FORECAST_COLUMNS,
        default=['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b', 'dE']
    )

    if not selected_cols:
        st.warning("กรุณาเลือกอย่างน้อย 1 ตัวแปร")
        st.stop()

# Time axis
actual_step = df.index[1] - df.index[0] if len(df) > 1 else 1
# คำนวณจำนวนแถวที่ต้องพยากรณ์ (Number of iterations)
# เช่น ถ้าอยากดูอนาคต 60 วินาที และข้อมูลห่างกันทีละ 10 วิ -> ต้องพยากรณ์ 6 แถว
forecast_iterations = int(max(1, forecast_steps / actual_step))
last_sec   = df.index[-1]
future_idx = [last_sec + (i + 1) * actual_step for i in range(forecast_iterations)]
full_idx   = list(df.index) + future_idx

# ─── Run forecasts & plot ────────────────────────────────────────────────────
metrics_rows = []
forecast_dict = {"sec (forecast)": future_idx}

for col in selected_cols:
    series = df[col].dropna()
    if len(series) < 2:
        st.warning(f"⚠️ {COLUMN_LABELS.get(col)}: ข้อมูลไม่เพียงพอสำหรับการพยากรณ์")
        continue

    # 1. พยากรณ์อนาคตของจริง (ใช้ข้อมูล 100% เต็ม)
    fcast_vals, model_info = forecast_series(series, forecast_iterations)
    # Store forecasted values in output dict for table
    forecast_dict[COLUMN_LABELS.get(col)] = np.round(fcast_vals, 2)

    # 2. ทำ Back-testing ด้วยข้อมูล 30% สุดท้าย เพื่อหา MAE, RMSE, MAPE
    try:
        # หาจุดตัด (Split Index) ที่ 70%
        split_idx = int(len(series) * 0.7)
        
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        test_steps = len(test_series)
        
        # ให้โมเดลเรียนรู้จากข้อมูล 70% แรก แล้วให้ลองพยากรณ์ไปข้างหน้าเท่ากับความยาวของช่วง 30%
        bt_predicted, _ = forecast_series(train_series, test_steps)
        
        bt_actual = test_series.values
        bt_predicted = np.array(bt_predicted)
        
        # ป้องกันกรณีเกิด Error จากความยาวของ Array ไม่เท่ากัน
        if len(bt_actual) == len(bt_predicted):
            mae, rmse, mape = compute_metrics(bt_actual, bt_predicted)
        else:
            mae, rmse, mape = float('nan'), float('nan'), float('nan')
    except Exception as e:
        mae, rmse, mape = float('nan'), float('nan'), float('nan')
        
    metrics_rows.append({
        "Variable":  COLUMN_LABELS.get(col),
        "Method":    "Holt-Winters (Exponential Smoothing)",
        "MAE":     round(mae, 2),
        "RMSE":    round(rmse, 2),
        "MAPE (%)": round(mape, 2) if not np.isnan(mape) else "N/A",
        "Model Info":   model_info,
    })

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

    # Forecast
    fig.add_trace(go.Scatter(
        x=future_idx, y=fcast_vals,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='tomato', width=2, dash='dash'),
        marker=dict(size=7, symbol='diamond')
    ))

    # Confidence band (±2 std)
    hist_std = series.std()
    upper = fcast_vals + 2 * hist_std
    lower = fcast_vals - 2 * hist_std

    fig.add_trace(go.Scatter(
        x=future_idx + future_idx[::-1],
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
        title=f"{COLUMN_LABELS.get(col)}  |  Holt-Winters (Exponential Smoothing)",
        xaxis_title="Time (sec)",
        yaxis_title=COLUMN_LABELS.get(col),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, width='stretch')

# ─── Metrics Table ───────────────────────────────────────────────────────────
if metrics_rows:
    st.divider()
    st.markdown("### 📐 Forecast Accuracy Metrics \n"
        "- **MAE** (Mean Absolute Error)\n"
        "- **RMSE** (Root Mean Squared Error)\n"
        "- **MAPE** (Mean Absolute Percentage Error)"
    )
    metrics_df = pd.DataFrame(metrics_rows).set_index("Variable")
    st.dataframe(metrics_df, width='stretch')

# ─── Forecast Table ──────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📋 Forecasted Table")
forecast_out_df = pd.DataFrame(forecast_dict).set_index("sec (forecast)")
st.dataframe(forecast_out_df, width='stretch')