import numpy as np
import pandas as pd

# === Parameters ===
portfolio_value = 1_000_000       # total portfolio
mu = 0.0005                        # daily expected return
sigma = 0.02                       # daily volatility (2%)
n_sims = 100000                    # Monte Carlo paths

# === Step 1: simulate returns ===
simulated_returns = np.random.normal(mu, sigma, n_sims)

# === Step 2: convert returns to portfolio values ===
portfolio_values = portfolio_value * (1 + simulated_returns)

# === Step 3: calculate losses ===
losses = portfolio_value - portfolio_values

# === Step 4: compute VaR ===
VaR_95 = np.percentile(losses, 95)     # loss at 95% confidence
VaR_99 = np.percentile(losses, 99)     # loss at 99% confidence

print(f"Monte Carlo 95% VaR: {VaR_95:,.0f}")
print(f"Monte Carlo 99% VaR: {VaR_99:,.0f}")
import matplotlib.pyplot as plt

# === Plot histogram of portfolio losses ===
plt.figure(figsize=(10,6))
plt.hist(losses, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

# Mark 95% and 99% VaR
plt.axvline(VaR_95, color='red', linestyle='--', label=f'95% VaR: {VaR_95:,.0f}')
plt.axvline(VaR_99, color='darkred', linestyle='--', label=f'99% VaR: {VaR_99:,.0f}')

plt.title('Monte Carlo Simulated Portfolio Loss Distribution')
plt.xlabel('Loss')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
