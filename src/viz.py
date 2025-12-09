import matplotlib.pyplot as plt
import numpy as np

def plot_price_series(df):
    fig, ax = plt.subplots(figsize=(10,5))
    df.plot(ax=ax)
    return fig

def plot_efficient_frontier(mu, cov, rf):
    fig, ax = plt.subplots(figsize=(8,5))
    n = 100
    vols = []
    rets = []
    for i in range(n):
        w = np.random.dirichlet(np.ones(len(mu)))
        rets.append(w@mu)
        vols.append(np.sqrt(w.T@cov@w))
    ax.scatter(vols, rets)
    ax.set_title("Efficient Frontier (Muestreo Random)")
    ax.set_xlabel("Volatilidad")
    ax.set_ylabel("Retorno")
    return fig
