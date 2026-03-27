from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from utils import multi_step_prediction
from indicators import compute_rsi, compute_macd

# optional live
try:
    from nsepython import nsefetch
    LIVE_AVAILABLE = True
except:
    LIVE_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model("model/transformer_model.h5",compile=False)
scaler = joblib.load("scaler/scaler.pkl")


def get_live_data(symbol):
    try:
        url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}"
        data = nsefetch(url)

        prices = [i['close'] for i in data['grapthData']]
        df = pd.DataFrame(prices, columns=["Close"])
        return df
    except:
        return None


@app.get("/predict")
def predict(stock: str = "RELIANCE"):
    try:
        df = None

        # 🔥 Try live data
        if LIVE_AVAILABLE:
            df = get_live_data(stock)

        # 🔥 Fallback to CSV
        if df is None or df.empty:
            df = pd.read_csv(f"data/{stock}.csv")

        close_prices = df[['Close']]

        scaled = scaler.transform(close_prices)

        predictions = multi_step_prediction(model, scaled, scaler)

        # Indicators
        rsi = compute_rsi(close_prices['Close']).dropna().tolist()
        macd, signal = compute_macd(close_prices['Close'])

        return {
            "history": close_prices.tail(100)['Close'].tolist(),
            "prediction": predictions,
            "rsi": rsi[-50:],
            "macd": macd.tail(50).tolist(),
            "signal": signal.tail(50).tolist()
        }

    except Exception as e:
        return {"error": str(e)}