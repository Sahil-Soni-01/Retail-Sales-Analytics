"""
forecast.py
-----------
Monthly sales forecasting with honest evaluation.

Models compared on a hold-out period (the last BACKTEST_MONTHS months):
  * Seasonal naive   - "same month last year" (the baseline to beat)
  * Holt-Winters     - additive trend + additive seasonality
The better model (lower MAE, which is stabler than MAPE on small monthly values) is then refit on ALL data to forecast ahead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config
from .analysis import monthly_sales


def _series(df: pd.DataFrame) -> pd.Series:
    m = monthly_sales(df).set_index("Month")["Sales"]
    m.index.freq = "MS"
    return m


def _metrics(actual: pd.Series, pred: pd.Series) -> dict:
    err = actual.values - pred.values
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE %": float(np.mean(np.abs(err) / np.where(actual.values == 0, np.nan, actual.values)) * 100),
    }


def _holt_winters(train: pd.Series) -> ExponentialSmoothing:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                                    initialization_method="estimated").fit()


def forecast_sales(df: pd.DataFrame, horizon: int = config.FORECAST_HORIZON,
                   holdout: int = config.BACKTEST_MONTHS) -> dict:
    """Returns {'history','backtest','metrics','best_model','forecast'}."""
    y = _series(df)
    if len(y) < 24 + holdout:
        raise ValueError(f"Need at least {24 + holdout} months of data, have {len(y)}.")

    train, test = y.iloc[:-holdout], y.iloc[-holdout:]

    naive_pred = pd.Series(y.shift(12).iloc[-holdout:].values, index=test.index)
    hw_pred = _holt_winters(train).forecast(holdout)
    hw_pred.index = test.index

    metrics = {"Seasonal naive": _metrics(test, naive_pred), "Holt-Winters": _metrics(test, hw_pred)}
    best = min(metrics, key=lambda k: metrics[k]["MAE"])

    future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if best == "Holt-Winters":
        fc = _holt_winters(y).forecast(horizon)
        fc.index = future_idx
    else:
        fc = pd.Series(y.iloc[-12:].values[:horizon], index=future_idx)
    fc = fc.clip(lower=0)

    return {
        "history": y,
        "backtest": pd.DataFrame({"Actual": test, "Seasonal naive": naive_pred, "Holt-Winters": hw_pred}),
        "metrics": pd.DataFrame(metrics).T.round(2),
        "best_model": best,
        "forecast": fc.rename("Forecast"),
    }
