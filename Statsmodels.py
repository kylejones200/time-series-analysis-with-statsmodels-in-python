"""Generated from Jupyter notebook: Statsmodels

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.layers import LSTM


def create_lagged_features(data, lag):
    X, y = ([], [])
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return (np.array(X), np.array(y))


def main() -> None:
    "\nGenerate or Load Time Series\xa0Data\nSimulate a time series with trend and seasonality\n"

    np.random.seed(42)

    n = 200

    time = pd.date_range(start="2023-01-01", periods=n, freq="D")

    trend = np.linspace(10, 50, n)

    seasonality = 10 * np.sin(np.linspace(0, 2 * np.pi, n))

    noise = np.random.normal(0, 2, n)

    data = trend + seasonality + noise

    df = pd.DataFrame({"date": time, "value": data})

    df.set_index("date", inplace=True)

    hold_out_days = 30

    train = df.iloc[:-hold_out_days]

    hold_out = df.iloc[-hold_out_days:]

    plt.figure(figsize=(10, 6))

    plt.plot(df.index, df["value"], label="Full Dataset", color="Blue")

    plt.plot(
        hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green"
    )

    plt.title("Simulated Time Series with Training and Hold-Out Sets")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid()

    plt.savefig("simulated_time_series.png")

    plt.show()

    "\nARIMA\n"

    model = ARIMA(train["value"], order=(2, 1, 2), freq="D")

    arima_result = model.fit()

    forecast = arima_result.get_forecast(steps=hold_out_days)

    forecast_index = hold_out.index

    forecast_mean = forecast.predicted_mean

    forecast_ci = forecast.conf_int()

    mape = mean_absolute_percentage_error(hold_out["value"], forecast_mean)

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.3%}")

    plt.figure(figsize=(10, 6))

    plt.plot(train.index, train["value"], label="Training Data", color="Blue")

    plt.plot(
        hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green"
    )

    plt.plot(forecast_index, forecast_mean, label="Forecast", color="Red")

    plt.fill_between(
        forecast_index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color="Red",
        alpha=0.2,
        label="Confidence Interval",
    )

    plt.title(f"ARIMA Forecast (MAPE: {mape:.3%})")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid()

    plt.savefig("arima_forecast_holdout.png")

    plt.show()

    "\nTime Series Decomposition\nUse seasonal_decompose to split the series into trend, seasonal, and residual components.\n"

    decomposition = seasonal_decompose(df["value"], model="additive", period=30)

    fig = decomposition.plot()

    fig.set_size_inches(10, 8)

    plt.suptitle("Time Series Decomposition", fontsize=16, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("time_series_decomposition.png")

    plt.show()

    "\nCheck for Stationarity\nUse the Augmented Dickey-Fuller (ADF) test to assess stationarity.\n"

    result = adfuller(df["value"])

    print(f"ADF Statistic: {result[0]:.4f}")

    print(f"P-Value: {result[1]:.4f}")

    if result[1] > 0.05:
        print("The time series is non-stationary.")
    else:
        print("The time series is stationary.")

    "\nAutocorrelation and Partial Autocorrelation\nVisualize the ACF and PACF to determine lag dependencies.\n"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_acf(df["value"], lags=30, ax=axes[0])

    plot_pacf(df["value"], lags=30, ax=axes[1])

    plt.suptitle("ACF and PACF Plots", fontsize=16)

    plt.savefig("acf_pacf_plots.png")

    plt.show()

    "\nFit an ARIMA\xa0Model\nFit an ARIMA model to the data for forecasting.\n"

    model = ARIMA(df["value"], order=(2, 1, 2))

    arima_result = model.fit()

    print(arima_result.summary())

    arima_result.plot_diagnostics(figsize=(10, 6))

    plt.savefig("arima_residuals_diagnostics.png")

    plt.show()

    "\nHolt-Winters Exponential Smoothing\n"

    hw_model = ExponentialSmoothing(
        train["value"], seasonal="add", seasonal_periods=30
    ).fit()

    hw_forecast = hw_model.forecast(steps=hold_out_days)

    mape_hw = mean_absolute_percentage_error(hold_out["value"], hw_forecast)

    print(f"Holt-Winters MAPE: {mape_hw:.3%}")

    plt.figure(figsize=(10, 6))

    plt.plot(train.index, train["value"], label="Training Data", color="Blue")

    plt.plot(
        hold_out.index, hold_out["value"], label="Hold-Out (True Values)", color="Green"
    )

    plt.plot(hold_out.index, hw_forecast, label="Holt-Winters Forecast", color="Red")

    plt.title(f"Holt-Winters Forecast \n MAPE: {mape_hw:.3%}")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid()

    plt.savefig("holt_winters_forecast.png")

    plt.show()

    lag = 10

    train_size = int(0.85 * (len(df) - lag))

    df_train = df.iloc[: train_size + lag].copy()

    df_test = df.iloc[train_size + lag :].copy()

    scaler = MinMaxScaler()

    scaler.fit(df_train["value"].values.reshape(-1, 1))

    df_train_scaled = scaler.transform(df_train["value"].values.reshape(-1, 1))

    df_test_scaled = scaler.transform(df_test["value"].values.reshape(-1, 1))

    X_train, y_train = create_lagged_features(df_train_scaled.flatten(), lag)

    X_test, y_test = create_lagged_features(df_test_scaled.flatten(), lag)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = tf.keras.Sequential(
        [LSTM(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )

    model.compile(optimizer="adam", loss="mse")

    model.summary()

    model.fit(
        X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1
    )

    y_pred_lstm = model.predict(X_test)

    y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)

    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    train_predictions = model.predict(X_train)

    train_predictions_inverse = scaler.inverse_transform(train_predictions)

    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_lstm_inverse)

    print(f"LSTM MAPE: {mape:.3%}")

    plt.figure(figsize=(12, 8))

    plt.plot(df.index, df["value"].values, label="Actual Data", color="Blue")

    train_index = df_train.index[lag:]

    plt.plot(
        train_index,
        train_predictions_inverse,
        label="Training Predictions",
        color="Orange",
    )

    test_index = df_test.index[lag:]

    plt.plot(test_index, y_test_inverse, label="Hold-Out (True Values)", color="Green")

    plt.plot(test_index, y_pred_lstm_inverse, label="Testing Predictions", color="Red")

    plt.title(f"LSTM Forecast. MAPE: {mape:.3%}")

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.legend()

    plt.grid()

    plt.savefig("LSTM_forecast_with_holdout.png")

    plt.show()


if __name__ == "__main__":
    main()
