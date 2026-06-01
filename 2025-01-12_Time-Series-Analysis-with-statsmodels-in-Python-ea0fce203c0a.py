# Description: Short example for Time Series Analysis with statsmodels in Python.

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader, TensorDataset

np.random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

"""
Generate or Load Time Series Data
Simulate a time series with trend and seasonality
"""

# Generate Simulated Time Series Data
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


def _run_statsmodels_demos() -> None:
    result = adfuller(df["value"])
    logger.info("ADF p-value: %.4f", result[1])
    arima_result = ARIMA(train["value"], order=(1, 1, 0)).fit()
    forecast_mean = arima_result.get_forecast(steps=hold_out_days).predicted_mean
    mape = mean_absolute_percentage_error(hold_out["value"], forecast_mean)
    logger.info("ARIMA hold-out MAPE: %.3f%%", mape * 100)
    hw_model = ExponentialSmoothing(
        train["value"], trend="add", seasonal="add", seasonal_periods=30
    ).fit()
    hw_forecast = hw_model.forecast(steps=hold_out_days)
    mape_hw = mean_absolute_percentage_error(hold_out["value"], hw_forecast)
    logger.info("Holt-Winters MAPE: %.3f%%", mape_hw * 100)


scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled["value"] = scaler.fit_transform(df["value"].values.reshape(-1, 1))
df = df_scaled


class _LSTMForecaster(nn.Module):
    """LSTM forecaster (auto-generated PyTorch replacement for Keras Sequential)."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 50,
        output_size: int = 1,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


def _train_torch(
    model: nn.Module,
    X_train,
    y_train,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    validation_split: float = 0.2,
    patience: int = 15,
) -> nn.Module:
    """Standard training loop replacing  + model.fit()."""
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    if y_t.dim() == 1:
        y_t = y_t.unsqueeze(1)
    n_val = max(1, int(len(X_t) * validation_split))
    X_val, y_val = X_t[-n_val:], y_t[-n_val:]
    X_tr, y_tr = X_t[:-n_val], y_t[:-n_val]
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def _predict_torch(model: nn.Module, X_test) -> "np.ndarray":
    """Replace model.predict()."""
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()


def create_lagged_features(data, lag):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i : i + lag])
        y.append(data[i + lag])
    return np.array(X), np.array(y)


def main():
    lag = 10  # Number of past observations to use for prediction
    X, y = create_lagged_features(df["value"].values, lag)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # Split into training and testing sets
    train_size = int(0.85 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    # Build, Fit, Predict and Evaluate the LSTM Model
    model = _LSTMForecaster(n_features=1, hidden=50, n_layers=1)
    _train_torch(model, X_train, y_train, epochs=5)
    y_pred_lstm = _predict_torch(model, X_test)
    y_pred_lstm_inverse = scaler.inverse_transform(
        y_pred_lstm
    )  # Inverse scaling for comparison
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1))
    # Reconstruct training predictions for plotting
    train_predictions = _predict_torch(model, X_train)
    train_predictions_inverse = scaler.inverse_transform(train_predictions)
    # Calculate MAPE for the test set
    mape = mean_absolute_percentage_error(y_test_inverse, y_pred_lstm_inverse)
    logger.info(f"LSTM MAPE: {mape:.3%}")
    # Plot the Results
    plt.figure(figsize=(12, 8))
    plt.plot(
        df.index,
        scaler.inverse_transform(df["value"].values.reshape(-1, 1)),
        label="Actual Data",
        color="Blue",
    )
    train_index = df.index[lag : train_size + lag]
    plt.plot(
        train_index,
        train_predictions_inverse,
        label="Training Predictions",
        color="Orange",
    )
    test_index = df.index[train_size + lag :]
    plt.plot(test_index, y_test_inverse, label="Hold-Out (True Values)", color="Green")
    plt.plot(test_index, y_pred_lstm_inverse, label="Testing Predictions", color="Red")
    plt.title(f"LSTM Forecast. MAPE: {mape:.3%}")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("LSTM_forecast_with_holdout.png")
    plt.close()


if __name__ == "__main__":
    _run_statsmodels_demos()
    main()
