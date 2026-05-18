import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import joblib

import torch
import torch.nn as nn

# =========================
# FEATURES
# =========================

FEATURES = [

    "Close",

    "Volume",

    "RSI",

    "MACD",

    "EMA20",

    "Volatility"
]

SEQUENCE_LENGTH = 60

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
# LOAD MODEL
# =========================

model = LSTMModel(
    input_size=len(FEATURES)
)

model.load_state_dict(

    torch.load(
        "models/lstm_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# =========================
# LOAD SCALER
# =========================

scaler = joblib.load(
    "models/scaler.save"
)

# =========================
# PREDICT FUNCTION
# =========================

def predict_stock(ticker):

    try:

        # =========================
        # FETCH DATA
        # =========================

        stock = yf.Ticker(
            ticker
        )

        hist = stock.history(
            period="1y"
        )

        if hist.empty:

            return {
                "error":
                    "No Data"
            }

        # =========================
        # INDICATORS
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

        # =========================
        # FEATURES
        # =========================

        data = hist[
            FEATURES
        ].values

        scaled_data = scaler.transform(
            data
        )

        # =========================
        # LAST SEQUENCE
        # =========================

        last_sequence = scaled_data[
            -SEQUENCE_LENGTH:
        ]

        last_sequence = np.expand_dims(
            last_sequence,
            axis=0
        )

        last_tensor = torch.tensor(

            last_sequence,

            dtype=torch.float32
        )

        # =========================
        # PREDICTION
        # =========================

        with torch.no_grad():

            prediction = model(
                last_tensor
            )

        # =========================
        # INVERSE SCALE
        # =========================

        dummy_array = np.zeros(
            (
                1,
                len(FEATURES)
            )
        )

        dummy_array[0][0] = (
            prediction.item()
        )

        inverse_prediction = scaler.inverse_transform(
            dummy_array
        )

        predicted_price = float(
            inverse_prediction[0][0]
        )

        current_price = float(
            hist["Close"].iloc[-1]
        )

        # =========================
        # SIGNAL
        # =========================

        signal = (

            "BUY"

            if predicted_price
            > current_price

            else "SELL"
        )

        trend = (

            "Bullish"

            if signal == "BUY"

            else "Bearish"
        )

        # =========================
        # CONFIDENCE
        # =========================

        difference = abs(

            predicted_price
            - current_price
        )

        confidence = 100 - (

            difference
            / current_price
        ) * 100

        confidence = max(
            min(confidence, 99),
            50
        )

        # =========================
        # RETURN
        # =========================

        return {

            "ticker":
                ticker,

            "currentPrice":
                round(
                    current_price,
                    2
                ),

            "predictedPrice":
                round(
                    predicted_price,
                    2
                ),

            "signal":
                signal,

            "trend":
                trend,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "model":
                "Production LSTM",

            "features":
                FEATURES,
        }

    except Exception as e:

        print(
            "AI ERROR:",
            e
        )

        return {
            "error":
                str(e)
        }