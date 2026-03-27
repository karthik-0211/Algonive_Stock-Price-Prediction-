import pandas as pd

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(data):
    exp1 = data.ewm(span=12).mean()
    exp2 = data.ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal