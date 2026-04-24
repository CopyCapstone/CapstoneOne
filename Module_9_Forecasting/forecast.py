import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def forecast_series(series: pd.Series, steps: int, degree: int = 2):
    """
    Uses actual time index (seconds) to forecast future intervals.
    """
    # 1. Use the actual index (0, 60, 120...) as X
    X = series.index.values.reshape(-1, 1)
    y = series.values

    # 2. Transform X
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # 3. Fit
    model = LinearRegression()
    model.fit(X_poly, y)

    # 4. Calculate future X values (e.g., 420, 480)
    step_size = np.median(np.diff(series.index)) # การัน step size เท่ากันทุกแถว
    last_val = series.index[-1]
    
    # Generate [420, 480, ...]
    X_future = np.array([last_val + (i + 1) * step_size for i in range(steps)]).reshape(-1, 1)

    X_full = np.vstack([X, X_future])
    X_full_poly = poly.transform(X_full)
    full_forecast_values = model.predict(X_full_poly)
    forecast_values = full_forecast_values[-steps:] # Get only the forecasted values
        
    # 5. Info String
    b0 = model.intercept_
    coeffs = model.coef_
    
    eq_terms = ["β0"] + [f"β{i+1}x^{i+1}" for i in range(len(coeffs))]
    eq_str = " + ".join(eq_terms)
    val_parts = [f"β0={b0:.2f}"] + [f"β{i+1}={val:.2e}" for i, val in enumerate(coeffs)]
    
    info = (f"y = {eq_str}\n" f"{' | '.join(val_parts)}\n") # f"Forecasted Intervals (seconds): {X_future.flatten().tolist()}
    
    return full_forecast_values, forecast_values, info

# ─── Metrics helper ─────────────────────────────────────────────────────────
def compute_metrics(actual, predicted):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mask = np.abs(actual) > 1e-8
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.any() else float('nan')
    return mae, rmse, mape