
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("AI-Based Stock Market Prediction")
st.write("First-Year B.Tech AI/ML Project")

st.warning(
    "Educational project only. Predictions are experimental "
    "and are not financial advice."
)

ticker = st.selectbox(
    "Select a company",
    ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
)

@st.cache_data(ttl=3600)
def get_stock_data(symbol):
    data = yf.download(
        symbol,
        period="5y",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.Series(dtype=float)

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    close = close.dropna()
    close = close[~close.index.duplicated(keep="last")]

    return close.sort_index()

if st.button("Train Model and Predict"):
    with st.spinner("Downloading data and training the model..."):
        try:
            prices = get_stock_data(ticker)

            if len(prices) < 100:
                st.error("Not enough historical data to train the model.")
                st.stop()

            # Create five previous-day price features
            dataset = pd.DataFrame({"Close": prices})

            for i in range(1, 6):
                dataset[f"Lag_{i}"] = dataset["Close"].shift(i)

            dataset = dataset.dropna()

            feature_columns = [
                f"Lag_{i}" for i in range(1, 6)
            ]

            X = dataset[feature_columns]
            y = dataset["Close"]

            # Keep the newest 20% for testing
            split = int(len(dataset) * 0.8)

            X_train = X.iloc[:split]
            X_test = X.iloc[split:]
            y_train = y.iloc[:split]
            y_test = y.iloc[split:]

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            # Simple baseline: predict tomorrow equals today's price
            baseline = X_test["Lag_1"].to_numpy()

            model_mae = mean_absolute_error(
                y_test, predictions
            )
            model_rmse = np.sqrt(
                mean_squared_error(y_test, predictions)
            )

            baseline_mae = mean_absolute_error(
                y_test, baseline
            )

            actual_direction = np.sign(
                y_test.to_numpy() - X_test["Lag_1"].to_numpy()
            )

            predicted_direction = np.sign(
                predictions - X_test["Lag_1"].to_numpy()
            )

            direction_accuracy = np.mean(
                actual_direction == predicted_direction
            ) * 100

            st.success("Model training and testing completed!")

            st.subheader("Historical Closing Prices")
            st.line_chart(prices)

            st.subheader("Test Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Model MAE", f"{model_mae:.2f}")
            c2.metric("Model RMSE", f"{model_rmse:.2f}")
            c3.metric(
                "Direction Accuracy",
                f"{direction_accuracy:.1f}%"
            )

            st.write(
                "Baseline MAE (predicting no price change):",
                round(baseline_mae, 2)
            )

            if model_mae < baseline_mae:
                st.info(
                    "The model had lower MAE than the baseline "
                    "on this test period."
                )
            else:
                st.info(
                    "The model did not beat the baseline on MAE "
                    "in this test period."
                )

            results = pd.DataFrame({
                "Actual Price": y_test,
                "Model Prediction": predictions,
                "Baseline Prediction": baseline
            }, index=y_test.index)

            st.subheader("Actual vs Predicted Prices")
            st.line_chart(results)

            # Estimate the next trading day's closing price
            latest_features = pd.DataFrame(
                [[prices.iloc[-i] for i in range(1, 6)]],
                columns=feature_columns
            )

            next_estimate = model.predict(latest_features)[0]

            st.subheader("Experimental Next-Day Estimate")
            st.metric(
                "Estimated closing price",
                f"₹{next_estimate:,.2f}"
            )

            st.caption(
                "This is a model estimate, not a reliable forecast "
                "or a recommendation to buy or sell."
            )

            st.download_button(
                "Download Test Results (CSV)",
                data=results.to_csv().encode("utf-8"),
                file_name="stock_prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info(
                "Please take a screenshot of this error and send it."
)
            
