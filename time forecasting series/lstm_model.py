import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

def preprocess_lstm_data(series, look_back=60):
    data = series.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=20, batch_size=16, verbose=1,
              callbacks=[EarlyStopping(monitor='loss', patience=3)])
    model.save('models/lstm_model.h5')
    return model

def forecast_lstm(model, X_last, steps, scaler):
    predictions = []
    current_input = X_last[-1]

    for _ in range(steps):
        pred = model.predict(current_input.reshape(1, current_input.shape[0], 1), verbose=0)
        predictions.append(pred[0][0])

        # Append and slide
        current_input = np.append(current_input[1:], pred, axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
