import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Günlük veri çek
today = datetime.today().strftime("%Y-%m-%d")
df = yf.download("AAPL", start="2003-01-01", end= today , interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Eğitim ve test ayır
split_date = df.index[-252]  # Son 1 yıl test (252 gün)
train = df.loc[df.index < split_date]
test = df.loc[df.index >= split_date]

# Ölçekleme
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 3])  # Close sütunu hedef
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# Model
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Tahmin
pred = model.predict(X_test)
pred_prices = scaler.inverse_transform(
    np.hstack((np.zeros((len(pred), 3)), pred, np.zeros((len(pred), 1))))
)[:, 3]

real_prices = test['Close'].values[look_back:]

# Grafik
plt.figure(figsize=(14,5))
plt.plot(test.index[look_back:], real_prices, label='Gerçek')
plt.plot(test.index[look_back:], pred_prices, label='Tahmin')
plt.legend()
plt.show()

# Performans
rmse = np.sqrt(mean_squared_error(real_prices, pred_prices))
mape = mean_absolute_percentage_error(real_prices, pred_prices)
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape*100:.2f}%")
model.save("model.h5")
