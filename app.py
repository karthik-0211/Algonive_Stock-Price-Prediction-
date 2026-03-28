import gradio as gr
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
from tensorflow.keras.models import load_model

# =========================
# STOCK LIST
# =========================
def get_stock_list():
    return sorted([f.replace(".csv", "") for f in os.listdir("data") if f.endswith(".csv")])

stock_list = get_stock_list()

# =========================
# INDICATORS
# =========================
def calculate_rsi(data, period=14):
    delta = pd.Series(data).diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    exp1 = pd.Series(data).ewm(span=12).mean()
    exp2 = pd.Series(data).ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal

# =========================
# SIGNAL LOGIC
# =========================
def get_signal(rsi, macd, signal):
    if rsi.iloc[-1] < 30 and macd.iloc[-1] > signal.iloc[-1]:
        return "🟢 BUY SIGNAL"
    elif rsi.iloc[-1] > 70 and macd.iloc[-1] < signal.iloc[-1]:
        return "🔴 SELL SIGNAL"
    else:
        return "🟡 HOLD"

# =========================
# MAIN FUNCTION
# =========================
def analyze(stock):

    try:
        # 🔥 Load model INSIDE function (fixes "Starting stuck")
        model = load_model("model/transformer_model.keras", compile=False)
        scaler = joblib.load("scaler/scaler.pkl")

        df = pd.read_csv(f"data/{stock}.csv")

        prices = df['Close'].values

        # SCALE
        scaled = scaler.transform(prices.reshape(-1, 1))
        last_60 = scaled[-60:]
        X = np.array([last_60])

        pred = model.predict(X, verbose=0)[0][0]
        pred = scaler.inverse_transform([[pred]])[0][0]

        # INDICATORS
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)

        trade_signal = get_signal(rsi, macd, signal)

        # =========================
        # CANDLESTICK
        # =========================
        fig_candle = go.Figure(data=[go.Candlestick(
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])

        fig_candle.update_layout(
            template="plotly_dark",
            title=f"{stock} Candlestick Chart",
            height=400
        )

        # =========================
        # RSI
        # =========================
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(y=rsi, name="RSI"))
        fig_rsi.update_layout(template="plotly_dark", height=250)

        # =========================
        # MACD
        # =========================
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(y=macd, name="MACD"))
        fig_macd.add_trace(go.Scatter(y=signal, name="Signal"))
        fig_macd.update_layout(template="plotly_dark", height=250)

        return (
            f"📈 Predicted Price: ₹{pred:.2f}",
            trade_signal,
            fig_candle,
            fig_rsi,
            fig_macd
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", "", None, None, None


# =========================
# UI DESIGN (PRO LEVEL)
# =========================
with gr.Blocks() as app:

    gr.Markdown("""
    # 🚀 AI Stock Trading Dashboard  
    ### Transformer-Based Multi-Stock Prediction System  
    """)

    with gr.Row():
        stock = gr.Dropdown(stock_list, value=stock_list[0], label="Select Stock")
        btn = gr.Button("Analyze 🚀")

    prediction = gr.Textbox(label="Prediction Result")
    signal_box = gr.Textbox(label="Trading Signal")

    candle_chart = gr.Plot(label="Candlestick Chart")

    with gr.Row():
        rsi_chart = gr.Plot(label="RSI Indicator")
        macd_chart = gr.Plot(label="MACD Indicator")

    btn.click(
        analyze,
        inputs=stock,
        outputs=[prediction, signal_box, candle_chart, rsi_chart, macd_chart]
    )


# =========================
# LAUNCH (FIXED FOR HF)
# =========================
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    css="""
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .gradio-container {
        font-family: 'Poppins';
    }
    button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white !important;
        border-radius: 8px !important;
        transition: 0.3s;
    }
    button:hover {
        transform: scale(1.05);
    }
    """
)
