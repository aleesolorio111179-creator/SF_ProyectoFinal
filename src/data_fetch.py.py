import yfinance as yf
import pandas as pd

def download_prices(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
    return df["Adj Close"]
