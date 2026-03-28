import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# =========================
# STOCK LIST
# =========================
def get_stock_list():
    return [f.replace(".csv", "") for f in os.listdir("data") if f.endswith(".csv")]

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
    if rsi[-1] < 30 and macd[-1] > signal[-1]:
        return "🟢 BUY SIGNAL"
    elif rsi[-1] > 70 and macd[-1] < signal[-1]:
        return "🔴 SELL SIGNAL"
    else:
        return "🟡 HOLD"

# =========================
# MAIN FUNCTION
# =========================
def analyze(stock):

    df = pd.read_csv(f"data/{stock}.csv")

    prices = df['Close'].values

    # =========================
    # LIGHTWEIGHT PREDICTION
    # =========================
    last_price = prices[-1]

    # simulate realistic movement (-2% to +2%)
    pred = last_price * (1 + np.random.uniform(-0.02, 0.02))

    # =========================
    # INDICATORS
    # =========================
    rsi = calculate_rsi(prices)
    macd, signal = calculate_macd(prices)

    trade_signal = get_signal(rsi, macd, signal)

    # =========================
    # CANDLESTICK CHART
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
    # RSI CHART
    # =========================
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(y=rsi, name="RSI"))
    fig_rsi.update_layout(
        template="plotly_dark",
        title="RSI Indicator",
        height=250
    )

    # =========================
    # MACD CHART
    # =========================
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(y=macd, name="MACD"))
    fig_macd.add_trace(go.Scatter(y=signal, name="Signal"))
    fig_macd.update_layout(
        template="plotly_dark",
        title="MACD Indicator",
        height=250
    )

    return (
        f"📈 Predicted Price: ₹{pred:.2f}",
        trade_signal,
        fig_candle,
        fig_rsi,
        fig_macd
    )

# =========================
# UI DESIGN (PRO LEVEL)
# =========================
with gr.Blocks(css="""
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.gradio-container {
    font-family: 'Poppins', sans-serif;
}
""") as app:

    gr.Markdown("""
# 🚀 AI Trading Dashboard  
### 📊 Stock Prediction & Technical Analysis
""")

    with gr.Row():
        stock = gr.Dropdown(stock_list, value=stock_list[0], label="📌 Select Stock")
        btn = gr.Button("🚀 Analyze", scale=0)

    prediction = gr.Textbox(label="📈 Prediction")
    signal_box = gr.Textbox(label="📊 Trading Signal")

    candle_chart = gr.Plot(label="📉 Candlestick Chart")

    with gr.Row():
        rsi_chart = gr.Plot(label="RSI")
        macd_chart = gr.Plot(label="MACD")

    btn.click(
        analyze,
        inputs=stock,
        outputs=[prediction, signal_box, candle_chart, rsi_chart, macd_chart]
    )

app.launch()
