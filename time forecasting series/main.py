import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Load dataset
df = pd.read_csv('yahoo_stock.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
data = df['Close'].sort_index()

# Forecast steps
forecast_steps = 30  # increase to make it visible

# ---------------- ARIMA Forecast ----------------
arima_model = ARIMA(data, order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=forecast_steps)

# Future dates
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]
arima_forecast_series = pd.Series(arima_forecast.values, index=future_dates)

# ---------------- Simulated LSTM Forecast ----------------
# Simulate basic trend (placeholder for real LSTM)
slope = (data.iloc[-1] - data.iloc[-30]) / 30
lstm_forecast = [data.iloc[-1] + slope * i for i in range(1, forecast_steps + 1)]
lstm_forecast_series = pd.Series(lstm_forecast, index=future_dates)

# ---------------- Plotting ----------------
plt.figure(figsize=(14, 6))

# Historical data
plt.plot(data.index, data, label='Historical', color='blue')

# ARIMA Forecast
plt.plot(arima_forecast_series.index, arima_forecast_series, label='ARIMA Forecast', color='green')

# Simulated LSTM Forecast
plt.plot(lstm_forecast_series.index, lstm_forecast_series, label='Simulated LSTM Forecast', color='red')

# Zoom in on last 6 months
plt.xlim([data.index[-120], future_dates[-1]])

# Labels and title
plt.title('Stock Price Forecast: Historical, ARIMA, Simulated LSTM')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
