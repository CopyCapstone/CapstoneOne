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
OUTPUT_CSV_PATH = TMP_DIR / "dataframe" / "batch_processing_results.csv"
os.makedirs(OUTPUT_CSV_PATH.parent, exist_ok=True)

df = pd.read_csv(OUTPUT_CSV_PATH)

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
def forecast_series(series: pd.Series, steps: int, method: str):
    """
    Returns (forecast_values, model_info_str)
    Handles short series gracefully.
    """
    n = len(series)

    if method == "Holt-Winters (Exponential Smoothing)":
        # Use simple exponential smoothing when series is too short for trend/season
        trend   = "add" if n >= 4 else None
        model = ExponentialSmoothing(series, trend=trend, seasonal=None,initialization_method="estimated")
        fit = model.fit(optimized=True)
        forecast = fit.forecast(steps)
        info = (f"Holt-Winters | trend={'additive' if trend else 'none'}" "|" f"α={fit.params.get('smoothing_level', float('nan')):.4f}" "|" f"β={fit.params.get('smoothing_trend', float('nan')):.4f}" )

    else : # "ARIMA"
        try:
            # ใช้ Auto-ARIMA ค้นหาพารามิเตอร์ (p,d,q) ที่ดีที่สุดโดยอัตโนมัติ
            auto_model = pm.auto_arima(
                series,
                start_p=0, start_q=0,
                max_p=3, max_q=3,        # จำกัดขอบเขตไม่ให้เกิน 3 เพื่อให้ระบบรันไวขึ้น ไม่หน่วงแอป
                d=None,                  # ใส่ None เพื่อให้ระบบรันการทดสอบทางสถิติ (ADF Test) หาค่า d ให้เอง
                seasonal=False,          # กำหนดเป็น False เพราะการเสื่อมสภาพของวัสดุไม่มีเรื่อง "ฤดูกาล (Seasonality)" แบบยอดขาย
                stepwise=True,           # ใช้ Stepwise Algorithm ช่วยให้หาค่าได้เร็วขึ้นมาก
                suppress_warnings=True,  # ซ่อนแจ้งเตือนยิบย่อย
                error_action="ignore"
            )
            
            # พยากรณ์ไปข้างหน้า
            forecast = auto_model.predict(n_periods=steps)
            
            # ดึงค่า p, d, q ที่ระบบเลือกมาเก็บไว้โชว์ (เพื่อเอาไปเขียนใน Report ได้)
            best_order = auto_model.order 
            info = f"Auto-ARIMA {best_order}"
        except Exception as e:
            # Fallback: แผนสำรอง กรณีข้อมูลช่วงนั้นนิ่งสนิทจนทำ Differencing ไม่ได้            
            model = ARIMA(series, order=(0, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps)
            info = "ARIMA(0,1,0) [fallback]"            
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

st.markdown("### 🎨 Predicted CIELAB")
st.line_chart(df[['predict_CIELAB_L', 'predict_CIELAB_a', 'predict_CIELAB_b']],
              color=['#888888', '#FF0000', '#DDCC00'])

st.markdown("### ✨ Specular RGB Values and Gloss Percent")
st.scatter_chart(df,
                 y=['specular_avg_r', 'specular_avg_g', 'specular_avg_b'],
                 color=['#FF0000', '#00CC00', '#0000FF'],
                 size='gloss_percent')

st.markdown("### 📈 Color Difference (ΔE)")
st.line_chart(df[['dE']])

st.markdown("### 📋 ตารางข้อมูลดิบ (Raw Data)")
st.dataframe(df, width='stretch')

st.divider()

# ─── Section 2: Forecasting ─────────────────────────────────────────────────
st.markdown("## 🔮 Forecasting")

with st.sidebar:
    # --- Controls ---
    st.header("⚙️ Forecasting Settings")
    # st.markdown("""
    #     **วิธีการ Forecasting ที่มีให้ใช้:**
    #     - **Holt-Winters Exponential Smoothing** — มาตรฐานสำหรับข้อมูล time-series ที่มี trend  
    #     *(ISO/ASTM color measurement, paint & coating industry)*
    #     - **ARIMA** — ใช้กว้างขวางในงานวิจัย degradation / aging ของวัสดุ  
    #     *(งานวิจัยวัสดุศาสตร์, color stability studies)*"""
    # )

    forecast_steps = st.slider("⏩ Forecasting Target (sec)", 1, 60, 6)
    method = st.selectbox("📐 Forecasting Method",["Holt-Winters (Exponential Smoothing)", "ARIMA"],index=0)
    selected_cols = st.multiselect(
        "📊 Variables to Forecast",
        FORECAST_COLUMNS,
        default=['diffuse_avg_r', 'diffuse_avg_g', 'diffuse_avg_b', 'dE']
    )

    if not selected_cols:
        st.warning("กรุณาเลือกอย่างน้อย 1 ตัวแปร")
        st.stop()

# Time axis
last_sec   = df.index[-1]
future_idx = [last_sec + (i + 1) for i in range(forecast_steps)]
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
    fcast_vals, model_info = forecast_series(series, forecast_steps, method)
    # Store forecasted values in output dict for table
    forecast_dict[COLUMN_LABELS.get(col)] = np.round(fcast_vals, 4)

    # 2. ทำ Back-testing ด้วยข้อมูล 30% สุดท้าย เพื่อหา MAE, RMSE, MAPE
    try:
        # หาจุดตัด (Split Index) ที่ 70%
        split_idx = int(len(series) * 0.7)
        
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        test_steps = len(test_series)
        
        # ให้โมเดลเรียนรู้จากข้อมูล 70% แรก แล้วให้ลองพยากรณ์ไปข้างหน้าเท่ากับความยาวของช่วง 30%
        bt_predicted, _ = forecast_series(train_series, test_steps, method)
        
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
        "Method":    method.split(" ")[0],
        "MAE":     round(mae, 4),
        "RMSE":    round(rmse, 4),
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
        title=f"{COLUMN_LABELS.get(col)}  |  {method}",
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

csv_bytes = forecast_out_df.to_csv().encode("utf-8")
st.download_button(
    label="⬇️ Download Forecasted Table (CSV)",
    data=csv_bytes,
    file_name="forecast_results.csv",
    mime="text/csv"
)