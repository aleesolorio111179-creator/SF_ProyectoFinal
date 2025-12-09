import streamlit as st
import pandas as pd
import numpy as np

from src.data_fetch import download_prices
from src.metrics import compute_all_metrics
from src.optimization import (
    min_variance_portfolio,
    max_sharpe_portfolio,
    markowitz_target_return
)
from src.black_litterman import (
    implied_returns,
    black_litterman_posterior
)
from src.viz import (
    plot_price_series,
    plot_efficient_frontier
)

# ---------------------------------------------------------
# CONFIG STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Gestor Cuantitativo de Portafolios",
    layout="wide"
)

st.title("Proyecto Final — Optimización Cuantitativa de Portafolios")
st.write("Aplicación profesional desarrollada para análisis, optimización y construcción de portafolios modernos.")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Configuración")

estrategia = st.sidebar.selectbox("Selecciona estrategia:", ["Regiones", "Sectores"])

tickers_regiones = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]
tickers_sectores = [
    "XLC","XLY","XLP","XLE","XLF","XLV","XLI",
    "XLB","XLRE","XLK","XLU"
]

if estrategia == "Regiones":
    tickers = tickers_regiones
else:
    tickers = tickers_sectores

start = st.sidebar.date_input("Fecha inicio", pd.to_datetime("2015-01-01"))
end   = st.sidebar.date_input("Fecha fin", pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre de riesgo (anual)", 0.00, 0.20, 0.02)

tipo_portafolio = st.sidebar.radio(
    "Tipo de cartera:",
    ["Arbitrario", "Optimizado", "Black–Litterman"]
)

# ---------------------------------------------------------
# DESCARGA DE DATOS
# ---------------------------------------------------------
with st.spinner("Descargando precios..."):
    prices = download_prices(tickers, start, end)

st.subheader("Precios de cierre ajustados")
st.line_chart(prices)

# ---------------------------------------------------------
# CÁLCULO DE MÉTRICAS
# ---------------------------------------------------------
returns = prices.pct_change().dropna()
metrics = compute_all_metrics(returns, rf)

st.subheader("Métricas por activo")
st.dataframe(metrics)

cov = returns.cov() * 252
mu  = returns.mean() * 252

# ---------------------------------------------------------
# PORTAFOLIO ARBITRARIO
# ---------------------------------------------------------
if tipo_portafolio == "Arbitrario":
    st.subheader("Portafolio Arbitrario")
    w = []
    for t in tickers:
        val = st.number_input(f"Peso {t}", 0.0, 1.0, 1/len(tickers))
        w.append(val)
    w = np.array(w)
    w = w / w.sum()

    st.write("### Pesos finales normalizados")
    st.write(w)

    ret_p = w @ mu
    vol_p = np.sqrt(w.T @ cov @ w)
    sharpe_p = (ret_p - rf) / vol_p

    st.metric("Rendimiento esperado anual", f"{ret_p:.2%}")
    st.metric("Volatilidad anual", f"{vol_p:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_p:.2f}")

# ---------------------------------------------------------
# PORTAFOLIOS OPTIMIZADOS
# ---------------------------------------------------------
elif tipo_portafolio == "Optimizado":
    st.subheader("Optimización Moderna de Portafolios")

    opcion = st.selectbox("Elige método:", ["Min Var", "Max Sharpe", "Markowitz Target"])

    if opcion == "Min Var":
        w = min_variance_portfolio(cov)
    elif opcion == "Max Sharpe":
        w = max_sharpe_portfolio(mu, cov, rf)
    else:
        target = st.slider("Target Return", 0.00, float(mu.max()), float(mu.mean()))
        w = markowitz_target_return(mu, cov, target)

    st.write("### Pesos óptimos")
    df_w = pd.DataFrame({"ticker": tickers, "peso": w})
    st.dataframe(df_w)

    ret_p = w @ mu
    vol_p = np.sqrt(w.T @ cov @ w)
    sharpe_p = (ret_p - rf) / vol_p

    st.metric("Rendimiento esperado anual", f"{ret_p:.2%}")
    st.metric("Volatilidad anual", f"{vol_p:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_p:.2f}")

    fig = plot_efficient_frontier(mu, cov, rf)
    st.pyplot(fig)

# ---------------------------------------------------------
# BLACK–LITTERMAN
# ---------------------------------------------------------
else:
    st.subheader("Black–Litterman")

    # Mercado = pesos iguales
    w_market = np.ones(len(tickers)) / len(tickers)

    pi = implied_returns(cov, w_market, risk_aversion=2.5)
    st.write("Retornos implícitos de equilibrio π:")
    st.write(pi)

    st.write("### Views (P y Q)")
    view_ticker1 = st.selectbox("Ticker A (A sobre B)", tickers)
    view_ticker2 = st.selectbox("Ticker B", tickers)

    q = st.number_input("¿Cuánto crees que A tendrá mayor retorno que B? (en %)", -20.0, 20.0, 2.0) / 100

    # P matrix
    P = np.zeros((1, len(tickers)))
    P[0, tickers.index(view_ticker1)] = 1
    P[0, tickers.index(view_ticker2)] = -1

    # Ω
    tau = 0.05
    omega = np.array([[P @ cov.values @ P.T]])[0] * tau

    # Posterior
    mu_bl, cov_bl = black_litterman_posterior(pi, cov.values, P, q, omega, tau)

    st.write("### Retornos Black–Litterman")
    st.write(mu_bl)

    st.write("### Optimización con BL")
    w = max_sharpe_portfolio(mu_bl, cov_bl, rf)
    df_w = pd.DataFrame({"ticker": tickers, "peso": w})
    st.dataframe(df_w)
