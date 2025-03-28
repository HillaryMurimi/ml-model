import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

# Feature Engineering
def create_features(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['rsi'] = compute_rsi(df['close'])
    df.dropna(inplace=True)
    return df

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Prepare Data for LSTM
def prepare_data(df, seq_len=50):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(seq_len, len(df_scaled)):
        X.append(df_scaled[i-seq_len:i])
        y.append(df_scaled[i, 0])  # Predicting close price
    return np.array(X), np.array(y), scaler

# Define LSTM Model
def build_model(input_shape):
    model = keras.Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train Model
def train_model(X_train, y_train):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    return model

# Save Model and Scaler
def save_model(model, scaler, model_path='model.h5', scaler_path='scaler.pkl'):
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

# Load Model and Scaler
def load_model(model_path='model.h5', scaler_path='scaler.pkl'):
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

if __name__ == "__main__":
    data = load_data('forex_data.csv')
    data = create_features(data)
    X, y, scaler = prepare_data(data)
    model = train_model(X, y)
    save_model(model, scaler)
