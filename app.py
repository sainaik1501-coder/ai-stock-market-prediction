
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="AI Stock Project", layout="wide")

st.title("AI-Based Stock Market Prediction")
st.write("Welcome to my first-year B.Tech AI/ML project!")

st.info("This project is for educational purposes, not financial advice.")

ticker = st.selectbox(
    "Select a company",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

if st.button("Load Stock Data"):
    with st.spinner("Downloading stock data..."):
        data = yf.download(
            ticker,
            period="1y",
            auto_adjust=True,
            progress=False
        )

    if data.empty:
        st.error("No data found. Please try again.")
    else:
        prices = data["Close"]

        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        prices = prices.dropna()

        st.subheader("Historical Closing Prices")
        st.line_chart(prices)

        st.subheader("Latest Closing Price")
        st.write(round(float(prices.iloc[-1]), 2))
      
