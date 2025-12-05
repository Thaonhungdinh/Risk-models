# ================================
# Efficient Frontier, CML, SML Visualization & Interpretation
# ================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# 1. DATA DOWNLOAD
# --------------------------

stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'PG']
market_ticker = 'SPY'
rf_rate = 0.04  # 4% risk-free rate

end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)

# Download data
data = yf.download(stocks + [market_ticker], start=start_date, end=end_date, interval='1mo', progress=False)

# Use Adjusted Close if available, else Close
if 'Adj Close' in data.columns:
    stock_data = data['Adj Close'][stocks].dropna(axis=1, how='all')
    market_data = data['Adj Close'][market_ticker].dropna()
else:
    stock_data = data['Close'][stocks].dropna(axis=1, how='all')
    market_data = data['Close'][market_ticker].dropna()

print("Stocks with valid data:", list(stock_data.columns))

# Monthly returns
returns = stock_data.pct_change().dropna()
market_returns = market_data.pct_change().dropna()

# Annualized stats
mean_returns = returns.mean() * 12
cov_matrix = returns.cov() * 12
std_devs = returns.std() * np.sqrt(12)

# --------------------------
# 2. PORTFOLIO FUNCTIONS
# --------------------------

def portfolio_stats(weights):
    ret = np.sum(weights * mean_returns)
    std = np.sqrt(weights.T @ cov_matrix @ weights)
    return ret, std

def portfolio_variance(weights):
    return weights.T @ cov_matrix @ weights

def negative_sharpe(weights):
    ret, std = portfolio_stats(weights)
    return -(ret - rf_rate) / std

# --------------------------
# 3. EFFICIENT FRONTIER
# --------------------------

n_assets = len(stock_data.columns)
weights_init = np.array([1/n_assets]*n_assets)
bounds = tuple((0,1) for _ in range(n_assets))
constraints_sum = ({'type':'eq', 'fun': lambda w: np.sum(w)-1})

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
frontier_returns = []
frontier_risks = []
frontier_weights = []

for target in target_returns:
    constraints = (
        constraints_sum,
        {'type':'eq', 'fun': lambda w: portfolio_stats(w)[0]-target}
    )
    result = minimize(portfolio_variance, weights_init, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        frontier_returns.append(portfolio_stats(result.x)[0])
        frontier_risks.append(portfolio_stats(result.x)[1])
        frontier_weights.append(result.x)

# --------------------------
# 4. TANGENCY PORTFOLIO (Max Sharpe)
# --------------------------

tangency_result = minimize(negative_sharpe, weights_init, method='SLSQP', bounds=bounds, constraints=constraints_sum)
tangency_weights = tangency_result.x
tangency_return, tangency_risk = portfolio_stats(tangency_weights)
tangency_sharpe = (tangency_return - rf_rate) / tangency_risk

# --------------------------
# 5. CAPITAL MARKET LINE (CML)
# --------------------------

cml_risks = np.linspace(0, max(frontier_risks)*1.2, 100)
cml_returns = rf_rate + tangency_sharpe * cml_risks

# --------------------------
# 6. SECURITY MARKET LINE (SML)
# --------------------------

betas = {}
alphas = {}
actual_returns = mean_returns.to_dict()
market_annual_return = market_returns.mean() * 12

for stock in stock_data.columns:
    cov = np.cov(returns[stock], market_returns[:len(returns[stock])])[0,1]
    var = np.var(market_returns[:len(returns[stock])])
    beta = cov/var
    betas[stock] = beta
    capm_ret = rf_rate + beta*(market_annual_return - rf_rate)
    alphas[stock] = actual_returns[stock] - capm_ret

beta_range = np.linspace(-0.5, 2.5, 100)
sml_returns = rf_rate + beta_range*(market_annual_return - rf_rate)

# --------------------------
# 7. VISUALIZATION
# --------------------------

plt.figure(figsize=(16,6))

# EF + CML
plt.subplot(1,2,1)
plt.plot(np.array(frontier_risks)*100, np.array(frontier_returns)*100, 'b-', label='Efficient Frontier', linewidth=2)
for stock in stock_data.columns:
    plt.scatter(std_devs[stock]*100, mean_returns[stock]*100, s=80, label=stock)
plt.scatter(tangency_risk*100, tangency_return*100, marker='*', s=300, c='red', label='Tangency Portfolio', edgecolors='black')
plt.plot(cml_risks*100, cml_returns*100, 'g--', linewidth=2, label='CML')
plt.scatter(0, rf_rate*100, c='yellow', s=150, edgecolors='black', label='Risk-Free')
plt.xlabel('Risk (σ) %'); plt.ylabel('Expected Return %')
plt.title('Efficient Frontier & CML'); plt.legend(); plt.grid(True, alpha=0.3)

# SML
plt.subplot(1,2,2)
plt.plot(beta_range, sml_returns*100, 'r-', label='SML', linewidth=2)
for stock in stock_data.columns:
    color = 'green' if alphas[stock]>0 else 'red'
    plt.scatter(betas[stock], actual_returns[stock]*100, c=color, s=100, edgecolors='black')
    plt.text(betas[stock]+0.02, actual_returns[stock]*100, stock, fontsize=9)
plt.scatter(0, rf_rate*100, c='yellow', s=150, edgecolors='black', label='Risk-Free')
plt.scatter(1, market_annual_return*100, c='blue', s=150, edgecolors='black', label='Market')
plt.xlabel('Beta'); plt.ylabel('Expected Return %')
plt.title('Security Market Line'); plt.legend(); plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------
# 8. AUTOMATIC INTERPRETATION
# --------------------------

print("\n📊 Tangency Portfolio:")
print(f"  Expected Return: {tangency_return*100:.2f}%")
print(f"  Risk (σ): {tangency_risk*100:.2f}%")
print(f"  Sharpe Ratio: {tangency_sharpe:.4f}\n")

print("📊 Alpha Analysis (Stock Valuation):")
for stock in stock_data.columns:
    status = "Undervalued ✓" if alphas[stock]>0 else "Overvalued ✗" if alphas[stock]<0 else "Fair"
    print(f"  {stock}: Alpha={alphas[stock]*100:+.2f}%, {status}")

print("\n✅ All charts generated: EF, CML, SML. Interpretation ready.")


