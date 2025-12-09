import numpy as np
from scipy.optimize import minimize

def min_variance_portfolio(cov):
    n = len(cov)
    w0 = np.ones(n)/n
    bounds = [(0,1)]*n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}

    res = minimize(lambda w: w.T@cov@w, w0, bounds=bounds, constraints=cons)
    return res.x

def max_sharpe_portfolio(mu, cov, rf):
    n = len(mu)
    w0 = np.ones(n)/n
    bounds = [(0,1)]*n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}

    def neg_sharpe(w):
        ret = w@mu
        vol = np.sqrt(w.T@cov@w)
        return -(ret - rf)/vol

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return res.x

def markowitz_target_return(mu, cov, target):
    n = len(mu)
    w0 = np.ones(n)/n
    bounds = [(0,1)]*n

    cons = (
        {"type": "eq", "fun": lambda w: w.sum() - 1},
        {"type": "eq", "fun": lambda w: w@mu - target}
    )

    res = minimize(lambda w: w.T@cov@w, w0, bounds=bounds, constraints=cons)
    return res.x
