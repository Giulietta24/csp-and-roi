import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Options Dashboard (Income + Growth)", layout="wide")

st.title("📊 Options Income + Growth Dashboard (Option C Model)")

# ---------------------------
# Inputs
# ---------------------------
ticker_input = st.sidebar.text_input("Ticker", "AAPL").upper()
target_price = st.sidebar.number_input("Target Price (for Calls)", value=200.0)

ticker = yf.Ticker(ticker_input)

try:
    current_price = ticker.history(period="1d")["Close"].iloc[-1]
except:
    st.error("Invalid ticker or no data available.")
    st.stop()

st.subheader(f"{ticker_input} Price: ${current_price:.2f}")

# ---------------------------
# Expirations
# ---------------------------
expirations = ticker.options
if not expirations:
    st.error("No options available.")
    st.stop()

expiry = st.sidebar.selectbox("Expiration", expirations)

chain = ticker.option_chain(expiry)
calls = chain.calls.copy()
puts = chain.puts.copy()

days_to_expiry = max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days, 1)

# =========================================================
# SAFE GREKS HANDLING (Yahoo fallback)
# =========================================================
def safe_col(df, col):
    if col in df.columns:
        return df[col]
    return np.nan

for df in [calls, puts]:
    if "delta" not in df.columns:
        df["delta"] = np.nan
    if "theta" not in df.columns:
        df["theta"] = np.nan

# Fallback estimates if missing
puts["delta"] = puts["delta"].fillna(-0.25)
calls["delta"] = calls["delta"].fillna(0.30)

puts["theta"] = puts["theta"].fillna(-0.05)
calls["theta"] = calls["theta"].fillna(-0.05)

# =========================================================
# ===================== CSP MODEL =========================
# =========================================================
puts["premium"] = puts["lastPrice"]
puts["collateral"] = puts["strike"] * 100
puts["max_income"] = puts["premium"] * 100

puts["breakeven"] = puts["strike"] - puts["premium"]

# ROC
puts["roc"] = puts["max_income"] / puts["collateral"]

# Theta income (positive benefit)
puts["theta_income"] = abs(puts["theta"]) * 100

# Probability proxy (OTM safety)
puts["otm_score"] = np.clip(0.30 - abs(puts["delta"]), 0, 0.30)

# ================= OPTION C WEIGHTS =================
# Theta 45%, ROC 35%, OTM safety 20%
puts["income_score"] = (
    puts["theta_income"] * 0.45 +
    puts["roc"] * 100 * 0.35 +
    puts["otm_score"] * 100 * 0.20
)

best_csp = puts.sort_values("income_score", ascending=False).head(1)

# =========================================================
# ===================== CALL MODEL ========================
# =========================================================
calls["premium"] = calls["lastPrice"]
calls["cost"] = calls["premium"] * 100

calls["break_even"] = calls["strike"] + calls["premium"]

calls["profit_at_target"] = (target_price - calls["strike"] - calls["premium"]) * 100

calls["roi"] = calls["profit_at_target"] / calls["cost"]

calls["theta_penalty"] = abs(calls["theta"]) * 100

calls["delta_score"] = np.clip(calls["delta"], 0, 1)

# ================= OPTION C WEIGHTS =================
# ROI 50%, Delta 30%, Theta penalty 20%
calls["growth_score"] = (
    calls["roi"].fillna(-1) * 50 +
    calls["delta_score"] * 30 +
    (1 - calls["theta_penalty"].fillna(0)) * 20
)

best_call = calls.sort_values("growth_score", ascending=False).head(1)

# =========================================================
# ===================== RESULTS ===========================
# =========================================================

st.header("🏆 Recommended Trades (Auto-Selected)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 Best Income Trade (CSP)")
    st.dataframe(best_csp[[
        "contractSymbol", "strike", "premium", "breakeven",
        "roc", "theta_income", "delta", "income_score"
    ]])

    st.success("Selected using: Theta (45%), ROC (35%), OTM safety (20%)")

with col2:
    st.subheader("🚀 Best Growth Trade (Call)")
    st.dataframe(best_call[[
        "contractSymbol", "strike", "premium",
        "break_even", "profit_at_target",
        "roi", "delta", "growth_score"
    ]])

    st.success("Selected using: ROI (50%), Delta (30%), Theta penalty (20%)")

# =========================================================
# FULL TABLES
# =========================================================

st.header("📊 Full Option Chains")

st.subheader("Puts (Income)")
st.dataframe(puts.sort_values("income_score", ascending=False))

st.subheader("Calls (Growth)")
st.dataframe(calls.sort_values("growth_score", ascending=False))

# =========================================================
# EXPLANATION PANEL
# =========================================================

st.header("🧠 How the Model Thinks")

st.markdown("""
### 💰 Income (CSP)
- **Theta (45%)** → how fast you collect premium decay  
- **ROC (35%)** → capital efficiency  
- **OTM safety (20%)** → probability of expiring worthless  

👉 Result: finds *high yield + high probability* cash-secured puts

---

### 🚀 Growth (Calls)
- **ROI (50%)** → leverage at your target price  
- **Delta (30%)** → directional strength  
- **Theta penalty (20%)** → avoids rapid decay contracts  

👉 Result: finds *best asymmetric upside trades*
""")