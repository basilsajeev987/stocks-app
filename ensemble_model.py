import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import MinMaxScaler

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
# ENSEMBLE PREDICTION
# =========================

def ensemble_predict(ticker):

    try:

        stock = yf.Ticker(
            ticker
        )

        hist = stock.history(
            period="2y"
        )

        if hist.empty:

            return {
                "error":
                    "No Data"
            }

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

            "Open",

            "High",

            "Low",

            "Close",

            "Volume",

            "EMA20",

            "RSI",

            "MACD",

            "Volatility"
        ]

        X = hist[
            features
        ]

        y = hist[
            "Close"
        ].shift(-1)

        X = X[:-1]

        y = y[:-1]

        # =========================
        # SCALE
        # =========================

        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(
            X
        )

        # =========================
        # RANDOM FOREST
        # =========================

        rf_model = RandomForestRegressor(

            n_estimators=100,

            random_state=42
        )

        rf_model.fit(
            X_scaled,
            y
        )

        # =========================
        # XGBOOST
        # =========================

        xgb_model = XGBRegressor(

            n_estimators=100,

            learning_rate=0.05,

            max_depth=5,

            objective="reg:squarederror"
        )

        xgb_model.fit(
            X_scaled,
            y
        )

        # =========================
        # LAST ROW
        # =========================

        latest_data = X.iloc[-1:]

        latest_scaled = scaler.transform(
            latest_data
        )

        # =========================
        # PREDICTIONS
        # =========================

        rf_prediction = float(

            rf_model.predict(
                latest_scaled
            )[0]
        )

        xgb_prediction = float(

            xgb_model.predict(
                latest_scaled
            )[0]
        )

        # =========================
        # ENSEMBLE
        # =========================

        final_prediction = (

            rf_prediction
            + xgb_prediction

        ) / 2

        current_price = float(
            hist["Close"].iloc[-1]
        )

        # =========================
        # SIGNAL
        # =========================

        signal = (

            "BUY"

            if final_prediction
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

            final_prediction
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
                    final_prediction,
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

            "models": [

                "Random Forest",

                "XGBoost"
            ],

            "ensemble":
                "Averaged Prediction",
        }

    except Exception as e:

        print(
            "ENSEMBLE ERROR:",
            e
        )

        return {
            "error":
                str(e)
        }