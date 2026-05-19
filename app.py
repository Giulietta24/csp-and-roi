import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from math import log, sqrt, exp
from scipy.stats import norm

st.set_page_config(page_title="Options Dashboard (Stable C Model)", layout="wide")

st.title("📊 Options Dashboard (Option C - Stable Version)")

# =========================
# CACHING LAYER (IMPORTANT)
# =========================
@st.cache_data(ttl=300)
def get_ticker(symbol):
    return yf.Ticker(symbol)

@st.cache_data(ttl=300)
def get_price(symbol):
    return yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]

# =========================
# BLACK SCHOLES HELPERS
# =========================
def bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_theta(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return 0
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    first = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T))

    if option_type == "call":
        second = r*K*exp(-r*T)*norm.cdf(d2)
        return first - second
    else:
        second = r*K*exp(-r*T)*norm.cdf(-d2)
        return first + second

# =========================
# INPUTS
# =========================
ticker_input = st.sidebar.text_input("Ticker", "AAPL").upper()
target_price = st.sidebar.number_input("Target Price (Calls)", value=200.0)

ticker = get_ticker(ticker_input)
current_price = get_price(ticker_input)

st.subheader(f"{ticker_input} Price: ${current_price:.2f}")

# =========================
# EXPIRATIONS (SAFE)
# =========================
try:
    expirations = ticker.options
except:
    st.error("Yahoo blocked request. Try again in 1–2 minutes.")
    st.stop()

expiry = st.sidebar.selectbox("Expiration", expirations)

chain = ticker.option_chain(expiry)
calls = chain.calls.copy()
puts = chain.puts.copy()

days = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days, 1)
T = days / 365
r = 0.04
sigma = 0.25  # fallback IV assumption

# =========================
# CSP MODEL (OPTION C)
# =========================
puts["premium"] = puts["lastPrice"]
puts["collateral"] = puts["strike"] * 100
puts["roc"] = (puts["premium"] * 100) / puts["collateral"]

puts["delta"] = puts["strike"].apply(lambda k: bs_delta(current_price, k, T, r, sigma, "put"))
puts["theta"] = puts["strike"].apply(lambda k: bs_theta(current_price, k, T, r, sigma, "put"))

puts["theta_score"] = abs(puts["theta"]) * 100
puts["otm_score"] = np.clip(0.3 - abs(puts["delta"]), 0, 0.3)

# OPTION C WEIGHTS (CLEAN VERSION)
puts["income_score"] = (
    puts["theta_score"] * 0.45 +
    puts["roc"] * 100 * 0.35 +
    puts["otm_score"] * 100 * 0.20
)

best_csp = puts.sort_values("income_score", ascending=False).head(1)

# =========================
# CALL MODEL (OPTION C)
# =========================
calls["premium"] = calls["lastPrice"]
calls["cost"] = calls["premium"] * 100

calls["profit_at_target"] = (target_price - calls["strike"] - calls["premium"]) * 100
calls["roi"] = calls["profit_at_target"] / calls["cost"]

calls["delta"] = calls["strike"].apply(lambda k: bs_delta(current_price, k, T, r, sigma, "call"))
calls["theta"] = calls["strike"].apply(lambda k: bs_theta(current_price, k, T, r, sigma, "call"))

calls["theta_penalty"] = abs(calls["theta"]) * 100
calls["delta_score"] = calls["delta"]

# OPTION C WEIGHTS
calls["growth_score"] = (
    calls["roi"].fillna(-1) * 50 +
    calls["delta_score"] * 30 +
    (1 - calls["theta_penalty"].fillna(0)) * 20
)

best_call = calls.sort_values("growth_score", ascending=False).head(1)

# =========================
# OUTPUT
# =========================
st.header("🏆 Best Trades (Stable Option C)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Best CSP")
    st.dataframe(best_csp[["strike","premium","roc","delta","theta","income_score"]])

with col2:
    st.subheader("🚀 Best Call")
    st.dataframe(best_call[["strike","premium","roi","delta","theta","growth_score"]])

st.success("Stable Option C model running (no reliance on Yahoo Greeks).")
