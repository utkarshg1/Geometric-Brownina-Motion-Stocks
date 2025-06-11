import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@dataclass(frozen=True)
class StockData:
    ticker: str
    data: pd.Series = field(init=False, repr=False)
    mu: float = field(init=False)
    sigma: float = field(init=False)

    def __post_init__(self) -> None:
        if type(self.ticker) != str:
            raise TypeError("Ticker should be a string")
        data = self.get_stock_data()
        object.__setattr__(self, "data", data)
        mu, sigma = self.estimate_mu_sigma()
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    def get_stock_data(self) -> pd.DataFrame:
        return yf.download(
            tickers=self.ticker, period="5y", interval="1d", auto_adjust=True
        ).loc[:, "Close"]

    def estimate_mu_sigma(self) -> tuple[float, float]:
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        mu_daily = log_returns.mean()
        sigma_daily = log_returns.std(ddof=1)

        trading_days = 252
        mu = (mu_daily + 0.5 * sigma_daily**2) * trading_days
        sigma = sigma_daily * np.sqrt(trading_days)
        return float(mu.values.flatten()[0]), float(sigma.values.flatten()[0])

    def plot_data(self) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.data)
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.set_title(
            f"{self.ticker} Closing Price, mu = {self.mu:.4f}, sigma = {self.sigma:.4f}"
        )
        ax.grid()
        return fig

    def simulate_gbm(
        self, M: int = 1000, T: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        start = time.perf_counter()
        S0 = float(self.data.values.flatten()[-1])
        mu = self.mu
        sigma = self.sigma
        dt = T / 252
        N = int(T / dt)

        # Generate all random increments at once: shape (M, N)
        Z = np.random.standard_normal((M, N))
        increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        # Build the paths: cumulative sum in each row
        log_S_paths = np.cumsum(increments, axis=1)
        log_S_paths = np.hstack((np.zeros((M, 1)), log_S_paths))
        S_paths = S0 * np.exp(log_S_paths)

        # Time grid
        times = np.linspace(0, T, N + 1)
        stop = time.perf_counter()
        elapsed = (stop - start) * 1000
        st.write(f"Elapsed time to generate values : **{elapsed:.2f} milli seconds**")
        return times, S_paths

    def plot_simulations(self) -> Figure:
        times, paths = self.simulate_gbm()
        M = paths.shape[0]
        fig, ax = plt.subplots(figsize=(10, 4))

        median = np.percentile(paths, 50, axis=0)
        lower = np.percentile(paths, 2.5, axis=0)
        upper = np.percentile(paths, 97.5, axis=0)

        for i in range(M):
            ax.plot(times, paths[i], lw=0.8, alpha=0.6)

        ax.plot(times, median, color="blue", lw=3, label="Median")
        ax.plot(times, lower, color="blue", lw=3, ls="--", label="2.5th percentile")
        ax.plot(times, upper, color="blue", lw=3, ls="--", label="97.5th percentile")
        ax.set_xlabel("Days")
        ax.set_ylabel("Simulated Prices")
        ax.set_title(f"GBM Simulation for {self.ticker}")
        ax.legend()
        ax.grid()
        st.subheader("Forecast 1 year")
        S0 = paths[0, 0]
        st.write(f"Current Price : {S0:.2f}")
        st.write(f"Forecasted Median : {median[-1]:.2f}")
        st.write(f"Lower 2.5 Percentile : {lower[-1]:.2f}")
        st.write(f"Upper 97.5 percentile : {upper[-1]:.2f}")
        pct_chg = (median[-1] - S0) / S0
        st.write(f"Percentage change from forecasted median 1 year : {pct_chg:.2%}")
        return fig
