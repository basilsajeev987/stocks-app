import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib

import torch
import torch.nn as nn

# =========================
# LSTM MODEL
# =========================

class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=64,

            num_layers=2,

            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            1
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out

# =========================
# RSI
# =========================

def calculate_rsi(
    prices,
    period=14
):

    delta = prices.diff()

    gain = delta.where(
        delta > 0,
        0
    )

    loss = -delta.where(
        delta < 0,
        0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi

# =========================
# MACD
# =========================

def calculate_macd(prices):

    ema12 = prices.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = prices.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    return macd, signal

# =========================
# TRAIN
# =========================

ticker = "NVDA"

stock = yf.Ticker(
    ticker
)

hist = stock.history(
    period="5y"
)

# =========================
# FEATURES
# =========================

hist["EMA20"] = (
    hist["Close"]
    .ewm(
        span=20,
        adjust=False
    )
    .mean()
)

hist["RSI"] = calculate_rsi(
    hist["Close"]
)

macd, signal = calculate_macd(
    hist["Close"]
)

hist["MACD"] = macd

hist["MACD_SIGNAL"] = signal

hist["Returns"] = (
    hist["Close"]
    .pct_change()
)

hist["Volatility"] = (
    hist["Returns"]
    .rolling(10)
    .std()
)

hist.dropna(inplace=True)

features = [

    "Close",

    "Volume",

    "RSI",

    "MACD",

    "EMA20",

    "Volatility"
]

data = hist[
    features
].values

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    data
)

# SAVE SCALER

joblib.dump(
    scaler,
    "models/scaler.save"
)

# =========================
# SEQUENCES
# =========================

sequence_length = 60

X = []
y = []

for i in range(
    sequence_length,
    len(scaled_data)
):

    X.append(

        scaled_data[
            i-sequence_length:i
        ]
    )

    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)

y = np.array(y)

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    y,
    dtype=torch.float32
).view(-1, 1)

# =========================
# MODEL
# =========================

model = LSTMModel(
    input_size=len(features)
)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# TRAIN
# =========================

epochs = 50

for epoch in range(
    epochs
):

    model.train()

    outputs = model(
        X_tensor
    )

    loss = criterion(
        outputs,
        y_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}"
        f"/{epochs}"
        f" Loss: {loss.item()}"
    )

# =========================
# SAVE MODEL
# =========================

torch.save(

    model.state_dict(),

    "models/lstm_model.pth"
)

print(
    "MODEL SAVED"
)