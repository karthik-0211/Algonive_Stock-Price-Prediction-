# 🚀 AI Stock Trading Dashboard
### Transformer-Based Multi-Stock Price Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?style=for-the-badge&logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-UI-green?style=for-the-badge&logo=gradio)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-yellow?style=for-the-badge)

---

## 🌐 Live Demo

**🔗 Try the app here:** [https://huggingface.co/spaces/karthik1102/stock_price_prediction](https://huggingface.co/spaces/karthik1102/stock_price_prediction)

---

## 📌 Project Overview

This project is an **AI-powered stock trading dashboard** that uses a **Transformer-based deep learning model** to predict stock closing prices. The application provides real-time technical analysis with **RSI**, **MACD**, and **Candlestick charts**, along with automated **Buy / Sell / Hold trading signals** — all wrapped in a sleek Gradio web interface.

Built as part of the **Algonive Internship**, this project demonstrates end-to-end ML deployment from model training to a live interactive web app.

---

## ✨ Features

- 📈 **Next-day closing price prediction** using a pre-trained Transformer model
- 🕯️ **Interactive Candlestick chart** for historical OHLC price visualization
- 📊 **RSI (Relative Strength Index)** indicator with overbought/oversold zones
- 📉 **MACD (Moving Average Convergence Divergence)** indicator with signal line
- 🟢🔴🟡 **Automated Trading Signals** — BUY, SELL, or HOLD based on RSI + MACD logic
- 🎨 **Dark-themed professional UI** built with Gradio + custom CSS
- ☁️ **Deployed on Hugging Face Spaces** for zero-setup access

---

## 🧠 Model Architecture

The core prediction engine is a **Transformer-based neural network** trained on historical stock data.

| Component | Details |
|---|---|
| Model Type | Transformer (Deep Learning) |
| Input | Last 60 days of scaled closing prices |
| Output | Predicted next closing price |
| Framework | TensorFlow 2.20.0 / Keras |
| Scaler | MinMaxScaler (saved via `joblib`) |
| Model Format | `.keras` (saved in `model/` directory) |

---

## 📐 Technical Indicators

### RSI — Relative Strength Index
- Period: **14 days**
- **RSI < 30** → Oversold (potential BUY zone)
- **RSI > 70** → Overbought (potential SELL zone)

### MACD — Moving Average Convergence Divergence
- Fast EMA: **12 periods**
- Slow EMA: **26 periods**
- Signal Line: **9-period EMA of MACD**

### Trading Signal Logic
```
if RSI < 30 AND MACD > Signal  →  🟢 BUY SIGNAL
if RSI > 70 AND MACD < Signal  →  🔴 SELL SIGNAL
else                           →  🟡 HOLD
```

---

## 🗂️ Repository Structure

```
Algonive_Stock-Price-Prediction-/
│
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
│
├── data/                   # Stock CSV files (one per stock ticker)
│   └── <TICKER>.csv        # Columns: Open, High, Low, Close, Volume
│
├── model/
│   └── transformer_model.keras   # Pre-trained Transformer model
│
└── scaler/
    └── scaler.pkl          # Fitted MinMaxScaler for price normalization
```

---

## ⚙️ How It Works

1. **User selects a stock** from the dropdown (auto-populated from `data/` folder)
2. **Historical CSV data** is loaded for the selected ticker
3. **Last 60 closing prices** are scaled using the saved `MinMaxScaler`
4. The **Transformer model** predicts the next closing price
5. **RSI and MACD** are computed from the full price history
6. A **trading signal** is generated based on RSI + MACD crossover logic
7. **Three interactive Plotly charts** are rendered — Candlestick, RSI, and MACD

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/karthik-0211/Algonive_Stock-Price-Prediction-.git
cd Algonive_Stock-Price-Prediction-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
python app.py
```

The app will start at `http://localhost:7860`

---

## 📦 Dependencies

```
gradio
numpy
pandas
scikit-learn
joblib
plotly
tensorflow==2.20.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## ☁️ Deployment

This application is deployed on **Hugging Face Spaces** using a Gradio SDK environment.

| Platform | Link |
|---|---|
| 🤗 Hugging Face Space | [karthik1102/stock_price_prediction](https://huggingface.co/spaces/karthik1102/stock_price_prediction) |
| 💻 GitHub Repository | [karthik-0211/Algonive_Stock-Price-Prediction-](https://github.com/karthik-0211/Algonive_Stock-Price-Prediction-) |

---

## 📊 Sample Output

After selecting a stock and clicking **Analyze 🚀**, the dashboard displays:

- `📈 Predicted Price: ₹XXXX.XX` — Model's predicted closing price
- `🟢 BUY SIGNAL / 🔴 SELL SIGNAL / 🟡 HOLD` — Trading recommendation
- **Candlestick Chart** — Full OHLC price history
- **RSI Chart** — Momentum oscillator
- **MACD Chart** — Trend-following momentum indicator

---

## ⚠️ Disclaimer

> This project is developed for **educational and research purposes only**.
> The predictions and trading signals generated by this model are **not financial advice**.
> Do not use this tool for actual investment decisions without consulting a certified financial advisor.

---

## 👨‍💻 Author

**Karthik**
- GitHub: [@karthik-0211](https://github.com/karthik-0211)
- Hugging Face: [@karthik1102](https://huggingface.co/karthik1102)

---

## 🏢 Internship

This project was built as part of the **Algonive Internship Program**.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
