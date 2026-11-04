from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(title="Indoor Air Quality Forecasting API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_indoor_air_lstm_6feat.keras"
SCALER_PATH = BASE_DIR / "humidity_winner_scaler.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class SensorWindow(BaseModel):
  readings: list[dict]


@app.post("/predict")
def predict(payload: SensorWindow):
  try:
    df_input = pd.DataFrame(payload.readings)
    if len(df_input) < 60:
      raise HTTPException(status_code=400, detail="Exactly 60 rows required.")

    df_input = df_input.tail(60).copy()

    # --- RULE-BASED SAFETY CHECK ---
    # Look at the very last (most recent) MQ135 sensor reading
    latest_mq135 = float(df_input["MQ135 Value"].iloc[-1])
    spike_detected = latest_mq135 > 300.0

    # Compute difference features
    df_input["Temperature (C)_diff"] = (
        df_input["Temperature (C)"].diff().fillna(0)
    )
    df_input["Humidity (%)_diff"] = df_input["Humidity (%)"].diff().fillna(0)
    df_input["MQ135 Value_diff"] = df_input["MQ135 Value"].diff().fillna(0)

    feature_cols = [
        "Temperature (C)",
        "Temperature (C)_diff",
        "Humidity (%)",
        "Humidity (%)_diff",
        "MQ135 Value",
        "MQ135 Value_diff",
    ]

    input_values = df_input[feature_cols].values
    scaled_values = scaler.transform(input_values)
    X_inference = np.expand_dims(scaled_values, axis=0)

    y_pred_scaled = model.predict(X_inference, verbose=0)
    y_pred_original = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 6)
    ).reshape(y_pred_scaled.shape)

    forecast_list = []
    for step in range(15):
      forecast_list.append({
          "minute_ahead": step + 1,
          "temperature_c": round(float(y_pred_original[0, step, 0]), 2),
          "humidity_pct": round(float(y_pred_original[0, step, 2]), 2),
          "mq135_value": round(float(y_pred_original[0, step, 4]), 2),
      })

    # Return the forecast along with the safety trigger status
    return {
        "status": "success",
        "safety_alert": spike_detected,
        "message": (
            "⚠️ DANGER: Hazardous gas/smoke spike detected!"
            if spike_detected
            else "Air quality levels are normal."
        ),
        "forecast": forecast_list,
    }

  except HTTPException:
    raise
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))