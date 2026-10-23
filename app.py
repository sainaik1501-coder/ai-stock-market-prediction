
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np


from scipy.optimize import minimize


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="AI Stock Prediction", layout="wide")
st.title("AI-Based Stock Market Prediction")
st.write("First-Year B.Tech AI/ML Project")
st.warning("Educational use only — not financial advice.")

ticker = st.selectbox(
    "Select a company",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

@st.cache_data(ttl=3600)
def get_prices(symbol):
    data = yf.download(
        symbol, period="5y", auto_adjust=True, progress=False
    )
    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna().sort_index()

if st.button("Compare Models"):
    try:
        prices = get_prices(ticker)

        if len(prices) < 100:
            st.error("Not enough historical data.")
            st.stop()

        df = pd.DataFrame({"Close": prices})

        for lag in range(1, 6):
            df[f"Lag_{lag}"] = df["Close"].shift(lag)

        df = df.dropna()
        features = [f"Lag_{i}" for i in range(1, 6)]

        X = df[features]
        y = df["Close"]

        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        }

        results = []
        predictions_df = pd.DataFrame(index=y_test.index)
        predictions_df["Actual"] = y_test

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            predictions_df[name] = pred

            results.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred))
            })

        result_table = pd.DataFrame(results).sort_values("MAE")

        st.success("All three models trained successfully!")
        st.subheader("Model Comparison")
        st.dataframe(result_table, use_container_width=True)

        st.subheader("MAE Comparison")
        st.bar_chart(result_table.set_index("Model")["MAE"])

        st.subheader("Actual vs Predicted Prices")
        st.line_chart(predictions_df)

        best_name = result_table.iloc[0]["Model"]
        st.write("Lowest test MAE:", best_name)

        st.download_button(
            "Download Model Results",
            result_table.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

        st.caption(
            "Lower MAE and RMSE indicate smaller errors on this test set. "
            "Past performance does not guarantee future results."
        )

    except Exception as e:
        st.error(f"Error: {e}")


# ==========================================
# PORTFOLIO OPTIMIZATION
# ==========================================

st.divider()
st.header("Portfolio Optimization")
st.write(
    "Compare an equal-weight portfolio with a "
    "mathematically optimized portfolio."
)

if st.button("Optimize Portfolio"):
    with st.spinner("Analyzing portfolio..."):
        try:
            tickers = [
                "RELIANCE.NS",
                "TCS.NS",
                "INFY.NS"
            ]

            price_data = {}

            for symbol in tickers:
                stock = yf.download(
                    symbol,
                    period="5y",
                    auto_adjust=True,
                    progress=False
                )

                if not stock.empty:
                    close = stock["Close"]

                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]

                    price_data[symbol] = close

            prices = pd.DataFrame(price_data).dropna()

            if len(prices) < 100 or len(price_data) != 3:
                st.error(
                    "Insufficient data. Please try again later."
                )
                st.stop()

            # Calculate daily percentage returns
            returns = prices.pct_change().dropna()

            # Annualized estimates using 252 trading days
            mean_returns = returns.mean() * 252
            covariance = returns.cov() * 252

            number_of_stocks = len(tickers)
            equal_weights = np.repeat(
                1 / number_of_stocks,
                number_of_stocks
            )

            risk_free_rate = 0.04

            def portfolio_stats(weights):
                annual_return = np.dot(
                    weights, mean_returns.to_numpy()
                )

                annual_volatility = np.sqrt(
                    weights @ covariance.to_numpy() @ weights
                )

                sharpe = (
                    (annual_return - risk_free_rate)
                    / annual_volatility
                    if annual_volatility > 0
                    else 0
                )

                return (
                    annual_return,
                    annual_volatility,
                    sharpe
                )

            def negative_sharpe(weights):
                return -portfolio_stats(weights)[2]

            result = minimize(
                negative_sharpe,
                equal_weights,
                method="SLSQP",
                bounds=[(0, 1)] * number_of_stocks,
                constraints={
                    "type": "eq",
                    "fun": lambda weights: np.sum(weights) - 1
                }
            )

            if not result.success:
                st.error(
                    "Optimization did not converge. "
                    "Please try again."
                )
                st.stop()

            optimized_weights = result.x

            equal_stats = portfolio_stats(equal_weights)
            optimized_stats = portfolio_stats(optimized_weights)

            comparison = pd.DataFrame({
                "Equal Weight": equal_weights,
                "Optimized": optimized_weights
            }, index=tickers)

            st.subheader("Portfolio Weights")
            st.dataframe(
                comparison.style.format("{:.1%}"),
                use_container_width=True
            )

            st.bar_chart(comparison)

            summary = pd.DataFrame({
                "Portfolio": [
                    "Equal Weight",
                    "Optimized"
                ],
                "Annualized Return": [
                    equal_stats[0],
                    optimized_stats[0]
                ],
                "Annualized Volatility": [
                    equal_stats[1],
                    optimized_stats[1]
                ],
                "Sharpe Ratio": [
                    equal_stats[2],
                    optimized_stats[2]
                ]
            })

            st.subheader("Portfolio Comparison")
            st.dataframe(
                summary.style.format({
                    "Annualized Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.3f}"
                }),
                use_container_width=True
            )

            st.download_button(
                "Download Portfolio Results",
                summary.to_csv(index=False).encode("utf-8"),
                file_name="portfolio_results.csv",
                mime="text/csv"
            )

            st.caption(
                "Estimates use historical daily returns and a "
                "4% assumed annual risk-free rate. They exclude "
                "transaction costs, taxes and slippage. "
                "Optimized weights are not investment advice."
            )

        except Exception as e:
            st.error(f"Portfolio analysis failed: {e}")


# ==========================================
# FINANCIAL NEWS SENTIMENT ANALYSIS
# ==========================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.divider()
st.header("Financial News Sentiment Analysis")

st.write(
    "Enter a financial headline to estimate its sentiment. "
    "This starter feature does not fetch live news."
)

@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_sentiment_analyzer()

headline = st.text_area(
    "Enter a financial news headline",
    placeholder="Example: The company reports stronger quarterly profits."
)

if st.button("Analyze Sentiment"):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        scores = analyzer.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        st.subheader(f"Sentiment: {sentiment}")
        st.metric("Sentiment score", f"{compound:.3f}")

        st.write("Detailed scores")
        st.json(scores)

        st.caption(
            "This is a basic text-analysis tool, not a "
            "financial recommendation. VADER may misread "
            "financial language, sarcasm, or context."
)
        
            
    
