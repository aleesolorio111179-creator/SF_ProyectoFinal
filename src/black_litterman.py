import numpy as np

def implied_returns(cov, w_market, risk_aversion=2.5):
    return risk_aversion * cov.values @ w_market

def black_litterman_posterior(pi, cov, P, q, omega, tau):
    cov_tau = tau * cov

    middle = np.linalg.inv(P @ cov_tau @ P.T + omega)

    mu_bl = np.linalg.inv(np.linalg.inv(cov_tau) + P.T @ middle @ P) @ \
            (np.linalg.inv(cov_tau) @ pi + P.T @ middle @ q)

    cov_bl = cov + np.linalg.inv(np.linalg.inv(cov_tau) + P.T @ middle @ P)

    return mu_bl, cov_bl
