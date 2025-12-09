import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def compute_all_metrics(returns, rf):
    mu = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sk = returns.apply(skew)
    kt = returns.apply(kurtosis)

    sharpe = (mu - rf) / vol
    dd = {}
    for c in returns.columns:
        wealth = (1 + returns[c]).cumprod()
        dd[c] = (wealth / wealth.cummax() - 1).min()

    df = pd.DataFrame({
        "Retorno": mu,
        "Volatilidad": vol,
        "Skew": sk,
        "Kurtosis": kt,
        "Sharpe": sharpe,
        "MaxDrawdown": dd
    })

    return df
