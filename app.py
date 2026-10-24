
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
     
# ==========================================
# DEEP LEARNING: LSTM STOCK MODEL
# ==========================================

st.divider()
st.header("Deep Learning: LSTM Model")

st.write(
    "This model learns from sequences of 20 historical "
    "closing prices to estimate the next price."
)

if st.button("Train LSTM Model"):
    with st.spinner(
        "Training the neural network. This may take several minutes..."
    ):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data for LSTM training.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)

            # Split chronologically before fitting the scaler
            split = int(len(values) * 0.8)
            train_values = values[:split]

            scaler = MinMaxScaler()
            scaler.fit(train_values)
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            if len(X_test) == 0:
                st.error("Not enough test data.")
                st.stop()

            model = Sequential([
                Input(shape=(window, 1)),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            predicted_scaled = model.predict(X_test, verbose=0)

            predicted = scaler.inverse_transform(predicted_scaled).ravel()
            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            lstm_mae = mean_absolute_error(actual, predicted)

            st.success("LSTM training and testing completed!")
            st.metric("LSTM Test MAE", f"₹{lstm_mae:.2f}")

            comparison = pd.DataFrame({
                "Actual Price": actual,
                "LSTM Prediction": predicted
            })

            st.line_chart(comparison)

            latest_sequence = scaled[-window:].reshape(1, window, 1)
            next_scaled = model.predict(latest_sequence, verbose=0)
            next_price = scaler.inverse_transform(next_scaled)[0, 0]

            st.metric(
                "Experimental LSTM next-day estimate",
                f"₹{next_price:,.2f}"
            )

            st.download_button(
                "Download LSTM Test Results",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="lstm_test_results.csv",
                mime="text/csv"
            )

            st.caption(
                "Experimental educational model only. "
                "Predictions are uncertain and are not financial advice."
            )

        except Exception as e:
            st.error(f"LSTM failed: {e}")
            st.info(
                "If TensorFlow fails to install or run, send me "
                "a screenshot of the full error."
)
            
# ==========================================
# DEEP LEARNING: GRU MODEL
# ==========================================

st.divider()
st.header("Deep Learning: GRU Model")

st.write(
    "GRU learns patterns from the previous 20 stock closing prices."
)

if st.button("Train GRU Model"):
    with st.spinner("Training GRU model. Please wait..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, GRU, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            model = Sequential([
                Input(shape=(window, 1)),
                GRU(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            scaled_predictions = model.predict(X_test, verbose=0)

            predictions = scaler.inverse_transform(
                scaled_predictions
            ).ravel()

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            mae = mean_absolute_error(actual, predictions)

            st.success("GRU training completed!")
            st.metric("GRU Test MAE", f"₹{mae:.2f}")

            results = pd.DataFrame({
                "Actual Price": actual,
                "GRU Prediction": predictions
            })

            st.line_chart(results)

            latest_sequence = scaled[-window:].reshape(
                1, window, 1
            )

            next_scaled = model.predict(
                latest_sequence, verbose=0
            )

            next_price = scaler.inverse_transform(
                next_scaled
            )[0, 0]

            st.metric(
                "Experimental GRU next-day estimate",
                f"₹{next_price:,.2f}"
            )

            st.download_button(
                "Download GRU Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="gru_results.csv",
                mime="text/csv"
            )

            st.caption(
                "Educational experiment only. This estimate is "
                "not financial advice."
            )

        except Exception as e:
            st.error(f"GRU error: {e}")


# ==========================================
# STEP 11: LSTM vs GRU COMPARISON
# ==========================================

st.divider()
st.header("LSTM vs GRU Model Comparison")

st.write(
    "Both models are trained using the same historical prices "
    "and chronological train/test split."
)

if st.button("Compare LSTM and GRU"):
    with st.spinner("Training both models. Please wait..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, GRU, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            # Train LSTM
            lstm_model = Sequential([
                Input(shape=(window, 1)),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            lstm_model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            lstm_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Train GRU
            gru_model = Sequential([
                Input(shape=(window, 1)),
                GRU(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            gru_model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            gru_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Evaluate both models on the same test data
            lstm_scaled = lstm_model.predict(X_test, verbose=0)
            gru_scaled = gru_model.predict(X_test, verbose=0)

            lstm_predictions = scaler.inverse_transform(
                lstm_scaled
            ).ravel()

            gru_predictions = scaler.inverse_transform(
                gru_scaled
            ).ravel()

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            lstm_mae = mean_absolute_error(
                actual, lstm_predictions
            )
            gru_mae = mean_absolute_error(
                actual, gru_predictions
            )

            comparison = pd.DataFrame({
                "Model": ["LSTM", "GRU"],
                "Test MAE (₹)": [lstm_mae, gru_mae]
            })

            st.success("Both models trained successfully!")

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                comparison.set_index("Model")["Test MAE (₹)"]
            )

            if lstm_mae < gru_mae:
                st.write("LSTM had the lower test MAE in this run.")
            elif gru_mae < lstm_mae:
                st.write("GRU had the lower test MAE in this run.")
            else:
                st.write("Both models had the same test MAE.")

            results = pd.DataFrame({
                "Actual Price": actual,
                "LSTM Prediction": lstm_predictions,
                "GRU Prediction": gru_predictions
            })

            st.line_chart(results)

            st.download_button(
                "Download LSTM vs GRU Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="lstm_gru_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "Historical test results only. Lower MAE does not "
                "guarantee better future predictions. "
                "This is an educational project, not financial advice."
            )

        except Exception as e:
            st.error(f"Comparison error: {e}")



# ==========================================
# STEP 11: LSTM vs GRU COMPARISON
# ==========================================

st.divider()
st.header("LSTM vs GRU Model Comparison")

st.write(
    "Both models are trained using the same historical prices "
    "and chronological train/test split."
)

if st.button("Compare LSTM and GRU", key="compare_lstm_gru_btn"):
    with st.spinner("Training both models. Please wait..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, LSTM, GRU, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            # Train LSTM
            lstm_model = Sequential([
                Input(shape=(window, 1)),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            lstm_model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            lstm_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Train GRU
            gru_model = Sequential([
                Input(shape=(window, 1)),
                GRU(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            gru_model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            gru_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            # Evaluate both models on the same test data
            lstm_scaled = lstm_model.predict(X_test, verbose=0)
            gru_scaled = gru_model.predict(X_test, verbose=0)

            lstm_predictions = scaler.inverse_transform(
                lstm_scaled
            ).ravel()

            gru_predictions = scaler.inverse_transform(
                gru_scaled
            ).ravel()

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            lstm_mae = mean_absolute_error(
                actual, lstm_predictions
            )
            gru_mae = mean_absolute_error(
                actual, gru_predictions
            )

            comparison = pd.DataFrame({
                "Model": ["LSTM", "GRU"],
                "Test MAE (₹)": [lstm_mae, gru_mae]
            })

            st.success("Both models trained successfully!")

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                comparison.set_index("Model")["Test MAE (₹)"]
            )

            if lstm_mae < gru_mae:
                st.write("LSTM had the lower test MAE in this run.")
            elif gru_mae < lstm_mae:
                st.write("GRU had the lower test MAE in this run.")
            else:
                st.write("Both models had the same test MAE.")

            results = pd.DataFrame({
                "Actual Price": actual,
                "LSTM Prediction": lstm_predictions,
                "GRU Prediction": gru_predictions
            })

            st.line_chart(results)

            st.download_button(
                "Download LSTM vs GRU Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="lstm_gru_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "Historical test results only. Lower MAE does not "
                "guarantee better future predictions. "
                "This is an educational project, not financial advice."
            )

        except Exception as e:
            st.error(f"Comparison error: {e}")



# ==========================================
# STEP 12: DEEP LEARNING - 1D CNN MODEL
# ==========================================

st.divider()
st.header("Deep Learning: 1D CNN Model")

st.write(
    "The CNN model learns short-term patterns from the "
    "previous 20 trading-day closing prices."
)

if st.button("Train CNN Model", key="train_cnn_model_btn"):
    with st.spinner("Training CNN model. Please wait..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Input, Conv1D, MaxPooling1D,
                Flatten, Dense
            )
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window

            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            model = Sequential([
                Input(shape=(window, 1)),
                Conv1D(
                    filters=32,
                    kernel_size=3,
                    activation="relu"
                ),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(32, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            scaled_predictions = model.predict(
                X_test, verbose=0
            )

            predictions = scaler.inverse_transform(
                scaled_predictions
            ).ravel()

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            mae = mean_absolute_error(actual, predictions)

            st.success("CNN training completed!")
            st.metric("CNN Test MAE", f"₹{mae:.2f}")

            results = pd.DataFrame({
                "Actual Price": actual,
                "CNN Prediction": predictions
            })

            st.line_chart(results)

            st.download_button(
                "Download CNN Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="cnn_results.csv",
                mime="text/csv",
                key="download_cnn_results_btn"
            )

            st.caption(
                "Educational experiment using historical prices. "
                "This model does not guarantee future performance "
                "and is not financial advice."
            )

        except Exception as e:
            st.error(f"CNN model error: {e}")



# ==========================================
# STEP 13: DEEP LEARNING - SIMPLE RNN MODEL
# ==========================================

st.divider()
st.header("Deep Learning: Simple RNN Model")

st.write(
    "Simple RNN learns sequential patterns from the "
    "previous 20 trading-day closing prices."
)

if st.button("Train Simple RNN Model", key="train_simple_rnn_btn"):
    with st.spinner("Training Simple RNN. Please wait..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, SimpleRNN, Dense
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            model = Sequential([
                Input(shape=(window, 1)),
                SimpleRNN(32),
                Dense(16, activation="relu"),
                Dense(1)
            ])

            model.compile(
                optimizer="adam",
                loss="mean_squared_error"
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=[early_stop],
                verbose=0
            )

            scaled_predictions = model.predict(X_test, verbose=0)

            predictions = scaler.inverse_transform(
                scaled_predictions
            ).ravel()

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            mae = mean_absolute_error(actual, predictions)

            st.success("Simple RNN training completed!")
            st.metric("Simple RNN Test MAE", f"₹{mae:.2f}")

            results = pd.DataFrame({
                "Actual Price": actual,
                "Simple RNN Prediction": predictions
            })

            st.line_chart(results)

            st.download_button(
                "Download Simple RNN Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="simple_rnn_results.csv",
                mime="text/csv",
                key="download_simple_rnn_results_btn"
            )

            st.caption(
                "Educational experiment using historical prices. "
                "Results do not guarantee future performance "
                "and are not financial advice."
            )

        except Exception as e:
            st.error(f"Simple RNN model error: {e}")

            
# ==========================================
# STEP 14: COMPARE ALL FOUR DEEP LEARNING MODELS
# ==========================================

st.divider()
st.header("Deep Learning Models: Final Comparison")

st.write(
    "Compare LSTM, GRU, 1D CNN, and Simple RNN using "
    "the same stock data and chronological test period."
)

if st.button(
    "Compare All Four Models",
    key="compare_all_four_dl_models_btn"
):
    with st.spinner("Training four models. This may take a few minutes..."):
        try:
            import tensorflow as tf
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.metrics import mean_absolute_error
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Input, LSTM, GRU, SimpleRNN,
                Conv1D, MaxPooling1D, Flatten, Dense
            )
            from tensorflow.keras.callbacks import EarlyStopping

            tf.keras.utils.set_random_seed(42)

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float).reshape(-1, 1)
            split = int(len(values) * 0.8)

            scaler = MinMaxScaler()
            scaler.fit(values[:split])
            scaled = scaler.transform(values)

            window = 20
            X, y = [], []

            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i, 0])
                y.append(scaled[i, 0])

            X = np.array(X).reshape(-1, window, 1)
            y = np.array(y)

            train_end = split - window
            X_train, X_test = X[:train_end], X[train_end:]
            y_train, y_test = y[:train_end], y[train_end:]

            actual = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).ravel()

            model_builders = {
                "LSTM": lambda: Sequential([
                    Input(shape=(window, 1)),
                    LSTM(32),
                    Dense(16, activation="relu"),
                    Dense(1)
                ]),
                "GRU": lambda: Sequential([
                    Input(shape=(window, 1)),
                    GRU(32),
                    Dense(16, activation="relu"),
                    Dense(1)
                ]),
                "1D CNN": lambda: Sequential([
                    Input(shape=(window, 1)),
                    Conv1D(32, 3, activation="relu"),
                    MaxPooling1D(2),
                    Flatten(),
                    Dense(32, activation="relu"),
                    Dense(1)
                ]),
                "Simple RNN": lambda: Sequential([
                    Input(shape=(window, 1)),
                    SimpleRNN(32),
                    Dense(16, activation="relu"),
                    Dense(1)
                ])
            }

            comparison_rows = []
            prediction_columns = {"Actual Price": actual}

            for name, build_model in model_builders.items():
                tf.keras.backend.clear_session()

                model = build_model()
                model.compile(
                    optimizer="adam",
                    loss="mean_squared_error"
                )

                early_stop = EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True
                )

                model.fit(
                    X_train,
                    y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1,
                    shuffle=False,
                    callbacks=[early_stop],
                    verbose=0
                )

                scaled_pred = model.predict(X_test, verbose=0)

                predicted = scaler.inverse_transform(
                    scaled_pred
                ).ravel()

                mae = mean_absolute_error(actual, predicted)

                comparison_rows.append({
                    "Model": name,
                    "Test MAE (₹)": round(float(mae), 2)
                })

                prediction_columns[name + " Prediction"] = predicted

            comparison = pd.DataFrame(comparison_rows)
            comparison = comparison.sort_values("Test MAE (₹)")

            st.success("All four models trained successfully!")

            st.subheader("Test Performance")
            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("MAE Comparison")
            st.bar_chart(
                comparison.set_index("Model")["Test MAE (₹)"]
            )

            st.subheader("Actual vs Predicted Prices")
            predictions_df = pd.DataFrame(prediction_columns)
            st.line_chart(predictions_df)

            best_model = comparison.iloc[0]["Model"]
            best_mae = comparison.iloc[0]["Test MAE (₹)"]

            st.write(
                f"Lowest test MAE in this run: {best_model} "
                f"(₹{best_mae:.2f})."
            )

            st.download_button(
                "Download Four-Model Comparison",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name="four_model_comparison.csv",
                mime="text/csv",
                key="download_four_model_comparison_btn"
            )

            st.download_button(
                "Download All Test Predictions",
                predictions_df.to_csv(index=False).encode("utf-8"),
                file_name="four_model_predictions.csv",
                mime="text/csv",
                key="download_four_model_predictions_btn"
            )

            st.caption(
                "This comparison uses one historical test period. "
                "The lowest MAE does not guarantee future success. "
                "No transaction costs are included. Educational use only; "
                "not financial advice."
            )

        except Exception as e:
            st.error(f"Four-model comparison error: {e}")




# ==========================================
# STEP 15: CLASSICAL ML - SUPPORT VECTOR REGRESSION
# ==========================================

st.divider()
st.header("Classical ML: Support Vector Regression (SVR)")

st.write(
    "SVR predicts stock closing prices using the previous "
    "20 trading-day closing prices as input features."
)

if st.button("Train SVR Model", key="train_svr_model_btn"):
    with st.spinner("Training SVR model. Please wait..."):
        try:
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            prices = get_prices(ticker)

            if len(prices) < 150:
                st.error("Not enough historical data.")
                st.stop()

            values = prices.to_numpy(dtype=float)
            window = 20

            X, y = [], []

            for i in range(window, len(values)):
                X.append(values[i-window:i])
                y.append(values[i])

            X = np.array(X)
            y = np.array(y)

            split = int(len(X) * 0.8)

            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Scale features using training data only
            x_scaler = StandardScaler()
            X_train_scaled = x_scaler.fit_transform(X_train)
            X_test_scaled = x_scaler.transform(X_test)

            # Scale target values using training data only
            y_scaler = StandardScaler()
            y_train_scaled = y_scaler.fit_transform(
                y_train.reshape(-1, 1)
            ).ravel()

            model = SVR(
                kernel="rbf",
                C=10,
                epsilon=0.1
            )

            model.fit(X_train_scaled, y_train_scaled)

            scaled_predictions = model.predict(X_test_scaled)

            predictions = y_scaler.inverse_transform(
                scaled_predictions.reshape(-1, 1)
            ).ravel()

            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))

            st.success("SVR training completed!")

            col1, col2 = st.columns(2)
            col1.metric("SVR Test MAE", f"₹{mae:.2f}")
            col2.metric("SVR Test RMSE", f"₹{rmse:.2f}")

            results = pd.DataFrame({
                "Actual Price": y_test,
                "SVR Prediction": predictions
            })

            st.line_chart(results)

            st.download_button(
                "Download SVR Results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="svr_results.csv",
                mime="text/csv",
                key="download_svr_results_btn"
            )

            st.caption(
                "SVR uses the previous 20 closing prices as features. "
                "Results are historical test estimates, not guaranteed "
                "future prices or financial advice."
            )

        except Exception as e:
            st.error(f"SVR model error: {e}")




            
            
    
