
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
    
