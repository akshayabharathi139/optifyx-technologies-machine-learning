import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.losses import MeanSquaredError

def forecast_arima(model_path: str, steps: int, last_date: pd.Timestamp):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    forecast = model.forecast(steps=steps)
    dates = pd.date_range(start=last_date, periods=steps+1, freq='B')[1:]
    return pd.DataFrame({'Date': dates, 'Forecast': forecast})

def forecast_lstm(model_path: str, scaler_path: str, historical_data: np.ndarray, steps: int):
    scaler = pickle.load(open(scaler_path, 'rb'))
    model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})

    seq_len = 3
    scaled_data = scaler.transform(historical_data.reshape(-1,1))
    last_seq = scaled_data[-seq_len:].reshape(1, seq_len, 1)

    preds = []
    for _ in range(steps):
        pred = model.predict(last_seq, verbose=0)[0]
        preds.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[pred]], axis=1)

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return forecast
