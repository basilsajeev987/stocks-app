from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rl_agent import (
    train_rl_agent
)
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import time
import ta
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from pymongo import MongoClient
from ensemble_model import (
    ensemble_predict
)
from textblob import TextBlob

from ai_model import predict_stock
import asyncio

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# MONGODB
# =========================

client = MongoClient(
    "mongodb://localhost:27017/"
)

db = client["stock_app"]

portfolio_collection = db["portfolio"]

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CACHE
# =========================

cached_data = {}

last_updated = 0

# =========================
# WEBSOCKET MANAGER
# =========================

class ConnectionManager:

    def __init__(self):

        self.active_connections = []

    async def connect(
        self,
        websocket: WebSocket
    ):

        await websocket.accept()

        self.active_connections.append(
            websocket
        )

    def disconnect(
        self,
        websocket: WebSocket
    ):

        self.active_connections.remove(
            websocket
        )

    async def broadcast(
        self,
        data
    ):

        for connection in self.active_connections:

            await connection.send_json(
                data
            )

manager = ConnectionManager()

# =========================
# STOCKS
# =========================

STOCKS = [
    "AAPL",
    "TSLA",
    "NVDA",
    "MSFT",
    "AMZN",
    "META",
    "GOOGL",
    "NFLX",
    "AMD",
    "PLTR",
]
# =========================
# LIVE WEBSOCKET
# =========================

@app.websocket("/ws/stocks")
async def websocket_endpoint(
    websocket: WebSocket
):

    await manager.connect(
        websocket
    )

    try:

        while True:

            global last_updated

            if (
                time.time()
                - last_updated
                > 5
            ):

                update_stock_cache()

            await manager.broadcast(
                list(
                    cached_data.values()
                )
            )

            await asyncio.sleep(2)

    except WebSocketDisconnect:

        manager.disconnect(
            websocket
        )

        
# =========================
# UPDATE CACHE
# =========================
@app.get("/rl/{ticker}")
def rl_prediction(
    ticker: str
):

    return train_rl_agent(
        ticker.upper()
    )
def update_stock_cache():

    global cached_data
    global last_updated

    try:

        data = yf.download(
            tickers=" ".join(STOCKS),
            period="1d",
            interval="1m",
            group_by="ticker",
            progress=False,
            threads=True
        )

        updated = {}

        for ticker in STOCKS:

            try:

                stock_data = data[ticker]

                if stock_data.empty:
                    continue

                latest = stock_data.dropna()

                price = round(
                    latest["Close"].iloc[-1],
                    2
                )

                updated[ticker] = {

                    "ticker": ticker,

                    "price": price,

                    "name": ticker,
                }

            except Exception as e:

                print(
                    f"{ticker} ERROR:",
                    e
                )

        cached_data = updated

        last_updated = time.time()

        print("Stock cache updated")

    except Exception as e:

        print(
            "CACHE ERROR:",
            e
        )

@app.get("/ensemble/{ticker}")
def ensemble(ticker: str):

    return ensemble_predict(
        ticker.upper()
    )
# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "message": "Backend Running"
    }

# =========================
# GET ALL STOCKS
# =========================

@app.get("/stocks")
def get_stocks():

    global last_updated

    if time.time() - last_updated > 15:

        update_stock_cache()

    return list(
        cached_data.values()
    )
# =========================
# BACKTEST ENGINE
# =========================

@app.get("/backtest/{ticker}")
def backtest(
    ticker: str,
    strategy: str = "ema"
):

    try:

        stock = yf.Ticker(
            ticker.upper()
        )

        hist = stock.history(
            period="1y"
        )

        if hist.empty:
            return {
                "error": "No data"
            }

        # =========================
        # EMA STRATEGY
        # =========================

        hist["EMA20"] = (
            hist["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        hist["EMA50"] = (
            hist["Close"]
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
        )

        hist["Signal"] = 0

        hist.loc[
            hist["EMA20"]
            > hist["EMA50"],
            "Signal"
        ] = 1

        hist["Position"] = (
            hist["Signal"]
            .diff()
        )

        # =========================
        # RETURNS
        # =========================

        hist["MarketReturn"] = (
            hist["Close"]
            .pct_change()
        )

        hist["StrategyReturn"] = (

            hist["MarketReturn"]

            * hist["Signal"]
            .shift(1)
        )

        hist.dropna(inplace=True)

        cumulative_market = (
            (
                1 +
                hist["MarketReturn"]
            )
            .cumprod()
            .iloc[-1]
        )

        cumulative_strategy = (
            (
                1 +
                hist["StrategyReturn"]
            )
            .cumprod()
            .iloc[-1]
        )

        total_return = (
            (
                cumulative_strategy
                - 1
            )
            * 100
        )

        volatility = (
            hist[
                "StrategyReturn"
            ]
            .std()
            * np.sqrt(252)
        )

        sharpe_ratio = (
            hist[
                "StrategyReturn"
            ]
            .mean()
            * 252
        ) / volatility

        # =========================
        # WIN RATE
        # =========================

        winning_trades = hist[
            hist["StrategyReturn"]
            > 0
        ]

        win_rate = (
            len(winning_trades)
            / len(hist)
        ) * 100

        # =========================
        # MAX DRAWDOWN
        # =========================

        cumulative = (
            (
                1 +
                hist["StrategyReturn"]
            )
            .cumprod()
        )

        peak = cumulative.cummax()

        drawdown = (
            (
                cumulative - peak
            )
            / peak
        )

        max_drawdown = (
            drawdown.min() * 100
        )

        # =========================
        # EQUITY CURVE
        # =========================

        equity_curve = []

        running = 10000

        for value in hist[
            "StrategyReturn"
        ]:

            running *= (
                1 + value
            )

            equity_curve.append(
                round(running, 2)
            )

        return {

            "ticker":
                ticker,

            "strategy":
                "EMA Crossover",

            "totalReturn":
                round(
                    total_return,
                    2
                ),

            "marketReturn":
                round(
                    (
                        cumulative_market
                        - 1
                    )
                    * 100,
                    2
                ),

            "sharpeRatio":
                round(
                    sharpe_ratio,
                    2
                ),

            "volatility":
                round(
                    volatility * 100,
                    2
                ),

            "winRate":
                round(
                    win_rate,
                    2
                ),

            "maxDrawdown":
                round(
                    max_drawdown,
                    2
                ),

            "equityCurve":
                equity_curve,
        }

    except Exception as e:

        print(
            "BACKTEST ERROR:",
            e
        )

        return {
            "error": str(e)
        }
# =========================
# GET SINGLE STOCK
# =========================

@app.get("/stock/{ticker}")
def get_stock(ticker: str):

    ticker = ticker.upper()

    if ticker in cached_data:

        return cached_data[ticker]

    return {

        "ticker": ticker,

        "price": 0,

        "name": ticker,
    }

# =========================
# HISTORY + TECHNICALS
# =========================

@app.get("/history/{ticker}")
def history(
    ticker: str,
    period: str = "1mo"
):

    try:

        interval = "1d"

        if period == "1d":
            interval = "5m"

        elif period == "1wk":
            interval = "30m"

        elif period == "1mo":
            interval = "1d"

        elif period == "3mo":
            interval = "1d"

        elif period == "1y":
            interval = "1wk"

        stock = yf.Ticker(
            ticker.upper()
        )

        hist = stock.history(
            period=period,
            interval=interval
        )

        if hist.empty:
            return []

        # =========================
        # SMA
        # =========================

        hist["SMA20"] = (
            hist["Close"]
            .rolling(window=20)
            .mean()
        )

        # =========================
        # EMA
        # =========================

        hist["EMA20"] = (
            hist["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        # =========================
        # RSI
        # =========================

        delta = hist["Close"].diff()

        gain = (
            delta.where(
                delta > 0,
                0
            )
            .rolling(14)
            .mean()
        )

        loss = (
            -delta.where(
                delta < 0,
                0
            )
            .rolling(14)
            .mean()
        )

        rs = gain / loss

        hist["RSI"] = (
            100 -
            (100 / (1 + rs))
        )

        # =========================
        # MACD
        # =========================

        macd = ta.trend.MACD(
            hist["Close"]
        )

        hist["MACD"] = (
            macd.macd()
        )

        hist["MACD_SIGNAL"] = (
            macd.macd_signal()
        )

        # =========================
        # BOLLINGER BANDS
        # =========================

        bb = ta.volatility.BollingerBands(
            hist["Close"]
        )

        hist["BB_HIGH"] = (
            bb.bollinger_hband()
        )

        hist["BB_LOW"] = (
            bb.bollinger_lband()
        )

        # =========================
        # VWAP
        # =========================

        vwap = ta.volume.VolumeWeightedAveragePrice(

            high=hist["High"],

            low=hist["Low"],

            close=hist["Close"],

            volume=hist["Volume"],
        )

        hist["VWAP"] = (
            vwap.volume_weighted_average_price()
        )

        candles = []

        for index, row in hist.iterrows():

            candles.append({

                "date":
                    str(index),

                "open":
                    round(
                        row["Open"],
                        2
                    ),

                "high":
                    round(
                        row["High"],
                        2
                    ),

                "low":
                    round(
                        row["Low"],
                        2
                    ),

                "close":
                    round(
                        row["Close"],
                        2
                    ),

                "sma20":
                    round(
                        row["SMA20"],
                        2
                    ) if pd.notna(
                        row["SMA20"]
                    ) else None,

                "ema20":
                    round(
                        row["EMA20"],
                        2
                    ) if pd.notna(
                        row["EMA20"]
                    ) else None,

                "rsi":
                    round(
                        row["RSI"],
                        2
                    ) if pd.notna(
                        row["RSI"]
                    ) else None,

                "macd":
                    round(
                        row["MACD"],
                        2
                    ) if pd.notna(
                        row["MACD"]
                    ) else None,

                "macdSignal":
                    round(
                        row["MACD_SIGNAL"],
                        2
                    ) if pd.notna(
                        row["MACD_SIGNAL"]
                    ) else None,

                "bbHigh":
                    round(
                        row["BB_HIGH"],
                        2
                    ) if pd.notna(
                        row["BB_HIGH"]
                    ) else None,

                "bbLow":
                    round(
                        row["BB_LOW"],
                        2
                    ) if pd.notna(
                        row["BB_LOW"]
                    ) else None,

                "vwap":
                    round(
                        row["VWAP"],
                        2
                    ) if pd.notna(
                        row["VWAP"]
                    ) else None,
            })

        return candles

    except Exception as e:

        print(
            "HISTORY ERROR:",
            e
        )

        return []

# =========================
# AI PREDICTION
# =========================

@app.get("/predict/{ticker}")
def predict(ticker: str):

    return predict_stock(
        ticker.upper()
    )

# =========================
# QUANT METRICS
# =========================

@app.get("/quant/{ticker}")
def quant_metrics(ticker: str):

    try:

        stock = yf.Ticker(
            ticker.upper()
        )

        hist = stock.history(
            period="1y"
        )

        hist["Returns"] = (
            hist["Close"]
            .pct_change()
        )

        returns = hist[
            "Returns"
        ].dropna()

        expected_return = (
            returns.mean() * 252
        )

        volatility = (
            returns.std()
            * np.sqrt(252)
        )

        risk_free_rate = 0.02

        sharpe_ratio = (
            expected_return
            - risk_free_rate
        ) / volatility

        var_95 = np.percentile(
            returns,
            5
        )

        market = yf.Ticker("SPY")

        market_hist = market.history(
            period="1y"
        )

        market_hist["Returns"] = (
            market_hist["Close"]
            .pct_change()
        )

        merged = pd.concat(
            [
                returns,
                market_hist["Returns"]
            ],
            axis=1
        ).dropna()

        merged.columns = [
            "stock",
            "market"
        ]

        covariance = np.cov(
            merged["stock"],
            merged["market"]
        )[0][1]

        market_variance = np.var(
            merged["market"]
        )

        beta = (
            covariance /
            market_variance
        )

        return {

            "ticker":
                ticker,

            "expectedReturn":
                round(
                    expected_return * 100,
                    2
                ),

            "volatility":
                round(
                    volatility * 100,
                    2
                ),

            "sharpeRatio":
                round(
                    sharpe_ratio,
                    2
                ),

            "beta":
                round(
                    beta,
                    2
                ),

            "var95":
                round(
                    var_95 * 100,
                    2
                ),
        }

    except Exception as e:

        print(
            "QUANT ERROR:",
            e
        )

        return {
            "error": str(e)
        }

# =========================
# MONTE CARLO
# =========================

@app.get("/montecarlo/{ticker}")
def monte_carlo(ticker: str):

    try:

        stock = yf.Ticker(
            ticker.upper()
        )

        hist = stock.history(
            period="1y"
        )

        returns = (
            hist["Close"]
            .pct_change()
            .dropna()
        )

        simulations = 100

        days = 30

        last_price = (
            hist["Close"].iloc[-1]
        )

        simulated_paths = []

        for _ in range(simulations):

            prices = [last_price]

            for _ in range(days):

                simulated_return = (
                    np.random.choice(
                        returns
                    )
                )

                prices.append(

                    prices[-1] *
                    (
                        1 +
                        simulated_return
                    )
                )

            simulated_paths.append(
                prices
            )

        final_prices = [
            path[-1]
            for path in simulated_paths
        ]

        min_price = round(
            np.percentile(
                final_prices,
                5
            ),
            2
        )

        max_price = round(
            np.percentile(
                final_prices,
                95
            ),
            2
        )

        avg_price = round(
            np.mean(
                final_prices
            ),
            2
        )

        return {

            "ticker":
                ticker,

            "minPrice":
                min_price,

            "maxPrice":
                max_price,

            "averagePrice":
                avg_price,

            "paths":
                simulated_paths,
        }

    except Exception as e:

        print(
            "MONTE CARLO ERROR:",
            e
        )

        return {
            "error": str(e)
        }

# =========================
# NEWS + SENTIMENT
# =========================

@app.get("/news/{ticker}")
def stock_news(ticker: str):

    try:

        API_KEY = (
            "d85gfahr01qitd91t5bgd85gfahr01qitd91t5c0"
        )

        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={ticker}"
            f"&from=2025-01-01"
            f"&to=2026-12-31"
            f"&token={API_KEY}"
        )

        response = requests.get(url)

        data = response.json()

        news_items = []

        sentiments = []

        for item in data[:5]:

            headline = item.get(
                "headline",
                ""
            )

            summary = item.get(
                "summary",
                ""
            )

            full_text = (
                headline +
                " " +
                summary
            )

            sentiment_score = (
                TextBlob(
                    full_text
                )
                .sentiment
                .polarity
            )

            sentiments.append(
                sentiment_score
            )

            news_items.append({

                "title":
                    headline,

                "summary":
                    summary,

                "url":
                    item.get("url"),

                "source":
                    item.get("source"),

                "sentiment":
                    round(
                        sentiment_score,
                        2
                    ),
            })

        avg_sentiment = (
            sum(sentiments)
            / len(sentiments)
        ) if sentiments else 0

        sentiment_label = "Neutral"

        if avg_sentiment > 0.1:

            sentiment_label = "Bullish"

        elif avg_sentiment < -0.1:

            sentiment_label = "Bearish"

        return {

            "ticker":
                ticker,

            "overallSentiment":
                sentiment_label,

            "score":
                round(
                    avg_sentiment,
                    2
                ),

            "news":
                news_items,
        }

    except Exception as e:

        print(
            "NEWS ERROR:",
            e
        )

        return {
            "error": str(e)
        }

# =========================
# CORRELATION MATRIX
# =========================

@app.get("/correlation")
def correlation():

    try:

        tickers = [
            "AAPL",
            "TSLA",
            "NVDA",
            "MSFT",
            "AMZN",
            "META",
            "AMD",
            "NFLX"
        ]

        data = yf.download(
            tickers,
            period="1y",
            progress=False
        )["Close"]

        returns = (
            data
            .pct_change()
            .dropna()
        )

        corr = returns.corr()

        return corr.to_dict()

    except Exception as e:

        print(
            "CORRELATION ERROR:",
            e
        )

        return {
            "error": str(e)
        }

# =========================
# EFFICIENT FRONTIER
# =========================

@app.get("/efficient-frontier")
def efficient_frontier():

    try:

        tickers = [
            "AAPL",
            "TSLA",
            "NVDA",
            "MSFT"
        ]

        data = yf.download(
            tickers,
            period="1y",
            progress=False
        )["Close"]

        returns = (
            data
            .pct_change()
            .dropna()
        )

        mean_returns = (
            returns.mean() * 252
        )

        cov_matrix = (
            returns.cov() * 252
        )

        portfolios = []

        for _ in range(500):

            weights = np.random.random(
                len(tickers)
            )

            weights /= np.sum(
                weights
            )

            portfolio_return = np.sum(
                mean_returns * weights
            )

            portfolio_volatility = np.sqrt(

                np.dot(
                    weights.T,

                    np.dot(
                        cov_matrix,
                        weights
                    )
                )
            )

            sharpe = (
                portfolio_return
                / portfolio_volatility
            )

            portfolios.append({

                "return":
                    round(
                        portfolio_return * 100,
                        2
                    ),

                "risk":
                    round(
                        portfolio_volatility * 100,
                        2
                    ),

                "sharpe":
                    round(
                        sharpe,
                        2
                    ),

                "weights": {

                    tickers[i]:
                    round(
                        weights[i] * 100,
                        2
                    )

                    for i in range(
                        len(tickers)
                    )
                }
            })

        return portfolios

    except Exception as e:

        print(
            "FRONTIER ERROR:",
            e
        )

        return {
            "error": str(e)
        }

# =========================
# ADD PORTFOLIO
# =========================

@app.post("/portfolio")
def add_portfolio(data: dict):

    portfolio_collection.insert_one({

        "ticker":
            data["ticker"],

        "quantity":
            data["quantity"],

        "buyPrice":
            data["buyPrice"],
    })

    return {
        "message": "Added"
    }

# =========================
# GET PORTFOLIO
# =========================

@app.get("/portfolio")
def get_portfolio():

    items = list(

        portfolio_collection.find(
            {},
            {"_id": 0}
        )
    )

    updated = []

    for item in items:

        ticker = item["ticker"]

        stock = yf.Ticker(ticker)

        hist = stock.history(
            period="1d"
        )

        current_price = round(
            hist["Close"].iloc[-1],
            2
        )

        invested = (
            item["quantity"]
            * item["buyPrice"]
        )

        current_value = (
            item["quantity"]
            * current_price
        )

        profit = round(
            current_value - invested,
            2
        )

        updated.append({

            "ticker":
                ticker,

            "quantity":
                item["quantity"],

            "buyPrice":
                item["buyPrice"],

            "currentPrice":
                current_price,

            "profit":
                profit,

            "value":
                round(
                    current_value,
                    2
                ),
        })

    return updated

# =========================
# INITIAL CACHE
# =========================

update_stock_cache()