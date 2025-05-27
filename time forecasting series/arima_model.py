import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA

def train_arima(df, order=(5,1,0)):
    model = ARIMA(df, order=order)
    fitted_model = model.fit()
    joblib.dump(fitted_model, 'models/arima_model.pkl')
    return fitted_model

def forecast_arima(steps=10):
    model = joblib.load('models/arima_model.pkl')
    forecast = model.forecast(steps=steps)
    return forecast
