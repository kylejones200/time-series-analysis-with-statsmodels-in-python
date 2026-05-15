"""Generated from Jupyter notebook: Statsmodels

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

"""
Generate or Load Time Series Data
Simulate a time series with trend and seasonality
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate Simulated Time Series Data
np.random.seed(42)
n = 200
time = pd.date_range(start="2023-01-01", periods=n, freq="D")
trend = np.linspace(10, 50, n)
seasonality = 10 * np.sin(np.linspace(0, 2 * np.pi, n))
noise = np.random.normal(0, 2, n)
data = trend + seasonality + noise

# Create a DataFrame; split the data
df = pd.DataFrame({"date": time, "value": data})
df.set_index("date", inplace=True)

hold_out_days = 30
train = df.iloc[:-hold_out_days]
hold_out = df.iloc[-hold_out_days:]

# Plot the Data
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


# --- code cell ---

"""
ARIMA
"""


# Fit ARIMA Model on Training Data
model = ARIMA(train["value"], order=(2, 1, 2), freq="D")  # Explicitly set freq="D"
arima_result = model.fit()

# Forecast Future Values for Hold-Out Period
forecast = arima_result.get_forecast(steps=hold_out_days)
forecast_index = hold_out.index  # Use the same index as the hold-out set
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Calculate MAPE on Hold-Out Set
mape = mean_absolute_percentage_error(hold_out["value"], forecast_mean)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.3%}")

# Plot the Results
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


# --- code cell ---

"""
Time Series Decomposition
Use seasonal_decompose to split the series into trend, seasonal, and residual components.
"""
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series
decomposition = seasonal_decompose(df["value"], model="additive", period=30)

# Plot the components
fig = decomposition.plot()
fig.set_size_inches(10, 8)  # Adjust the figure size
plt.suptitle("Time Series Decomposition", fontsize=16, y=0.95)  # Adjust title position
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Prevent overlap of title with subplots
plt.savefig("time_series_decomposition.png")
plt.show()


# --- code cell ---

"""
Check for Stationarity
Use the Augmented Dickey-Fuller (ADF) test to assess stationarity.
"""

from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(df["value"])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"P-Value: {result[1]:.4f}")
if result[1] > 0.05:
    print("The time series is non-stationary.")
else:
    print("The time series is stationary.")


# --- code cell ---

"""
Autocorrelation and Partial Autocorrelation
Visualize the ACF and PACF to determine lag dependencies.
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(df["value"], lags=30, ax=axes[0])
plot_pacf(df["value"], lags=30, ax=axes[1])
plt.suptitle("ACF and PACF Plots", fontsize=16)
plt.savefig("acf_pacf_plots.png")
plt.show()


# --- code cell ---

"""
Fit an ARIMA Model
Fit an ARIMA model to the data for forecasting.
"""

from statsmodels.tsa.arima.model import ARIMA

# Fit an ARIMA(2,1,2) model
model = ARIMA(df["value"], order=(2, 1, 2))
arima_result = model.fit()

print(arima_result.summary())
# Plot the residuals
arima_result.plot_diagnostics(figsize=(10, 6))
plt.savefig("arima_residuals_diagnostics.png")
plt.show()


# --- code cell ---

"""
Holt-Winters Exponential Smoothing
"""

# Apply Holt-Winters Exponential Smoothing
hw_model = ExponentialSmoothing(
    train["value"], seasonal="add", seasonal_periods=30
).fit()

hw_forecast = hw_model.forecast(steps=hold_out_days)

# Calculate MAPE on Hold-Out Set
mape_hw = mean_absolute_percentage_error(hold_out["value"], hw_forecast)
print(f"Holt-Winters MAPE: {mape_hw:.3%}")

# Plot the Results
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


# --- code cell ---

import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM

# Prepare Data for LSTM
# Split data first before scaling
lag = 10  # Number of past observations to use for prediction
train_size = int(0.85 * (len(df) - lag))
df_train = df.iloc[: train_size + lag].copy()
df_test = df.iloc[train_size + lag :].copy()

# Fit scaler on training data only
scaler = MinMaxScaler()
scaler.fit(df_train["value"].values.reshape(-1, 1))
df_train_scaled = scaler.transform(df_train["value"].values.reshape(-1, 1))
df_test_scaled = scaler.transform(df_test["value"].values.reshape(-1, 1))


def create_lagged_features(data, lag):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return np.array(X), np.array(y)



def main():
    # Create lagged features on scaled data
    X_train, y_train = create_lagged_features(df_train_scaled.flatten(), lag)
    X_test, y_test = create_lagged_features(df_test_scaled.flatten(), lag)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build, Fit, Predict and Evaluate the LSTM Model
    model = tf.keras.Sequential(
        [LSTM(50, activation="relu", input_shape=(lag, 1)), tf.keras.layers.Dense(1)]
    )
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.1)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm_inverse = scaler.inverse_transform(
        y_pred_lstm
    )  # Inverse scaling for comparison
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Reconstruct training predictions for plotting
    train_predictions = model.predict(X_train)
    train_predictions_inverse = scaler.inverse_transform(train_predictions)

    # Calculate MAPE for the test set
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_lstm_inverse)
    print(f"LSTM MAPE: {mape:.3%}")

    # Plot the Results
    plt.figure(figsize=(12, 8))
    plt.plot(df.index, df["value"].values, label="Actual Data", color="Blue")
    train_index = df_train.index[lag:]
    plt.plot(
        train_index, train_predictions_inverse, label="Training Predictions", color="Orange"
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
