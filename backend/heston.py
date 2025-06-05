import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class HestonModel:
    def __init__(self, S0, v0, kappa, theta, xi, rho, r, T):
        self.S0 = S0
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.r = r
        self.T = T

    def simulate_paths(self, n_paths=1000, n_steps=252):
        dt = self.T / n_steps
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for i in range(n_steps):
            Z1 = np.random.standard_normal(n_paths)
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * np.random.standard_normal(n_paths)

            v[:, i + 1] = np.maximum(
                v[:, i] + self.kappa * (self.theta - v[:, i]) * dt + 
                self.xi * np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z2,
                0
            )
            S[:, i + 1] = S[:, i] * np.exp(
                (self.r - 0.5 * v[:, i]) * dt + 
                np.sqrt(np.maximum(v[:, i], 0)) * np.sqrt(dt) * Z1
            )

        return S, v

def get_stock_data(symbol, period="3y"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    data['Returns'] = data['Close'].pct_change().dropna()
    historical_vol = data['Returns'].std() * np.sqrt(252)
    return data, historical_vol

def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        tnx_data = tnx.history(period="5d")
        return tnx_data['Close'].iloc[-1] / 100
    except:
        return 0.03

def calibrate_heston_simple(stock_data, historical_vol):
    v0 = historical_vol**2
    theta = historical_vol**2
    kappa = 2.0
    xi = 0.3
    rho = -0.7
    return v0, kappa, theta, xi, rho

def analyze_stock_with_heston(symbol="PLTR", forecast_months=6):
    print(f"🔍 Analyzing {symbol} with Heston Model\n{'=' * 50}")
    stock_data, historical_vol = get_stock_data(symbol)
    current_price = stock_data['Close'].iloc[-1]
    risk_free_rate = get_risk_free_rate()
    v0, kappa, theta, xi, rho = calibrate_heston_simple(stock_data, historical_vol)
    T = forecast_months / 12
    heston = HestonModel(current_price, v0, kappa, theta, xi, rho, risk_free_rate, T)
    S_paths, v_paths = heston.simulate_paths(n_paths=1000, n_steps=int(252 * T))
    return stock_data, S_paths, v_paths, heston

def create_visualizations(symbol, stock_data, S_paths, v_paths, heston):
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = np.linspace(0, heston.T, S_paths.shape[1])
    for i in range(min(100, S_paths.shape[0])):
        ax.plot(time_steps, S_paths[i], alpha=0.05, color='blue')
    ax.set_title(f"Heston Model Price Simulation for {symbol}")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Stock Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Placeholder class for Streamlit Cloud compatibility
class EnhancedHestonML:
    def __init__(self, *args, **kwargs):
        print("⚠️ Torch-based EnhancedHestonML class is disabled for deployment.")
    def train_networks(self, *args, **kwargs):
        print("[Mock] Training disabled")
    def enhanced_forecast(self, *args, **kwargs):
        return np.zeros((1, 1)), np.zeros((1, 1)), 0.0
    def create_enhanced_visualizations(self, *args, **kwargs):
        print("[Mock] Visualization disabled")

if __name__ == "__main__":
    symbol = "PLTR"
    stock_data, S_paths, v_paths, heston_model = analyze_stock_with_heston(symbol, forecast_months=12)
    create_visualizations(symbol, stock_data, S_paths, v_paths, heston_model)
    print("\n🔄 Torch-based ML analysis has been disabled for Streamlit deployment.")
