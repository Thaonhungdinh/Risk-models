import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Add this import

# === Portfolio Parameters ===
weights = np.array([0.4, 0.35, 0.25])          # weights of 3 assets
mu = np.array([0.0005, 0.0002, 0.0001])        # daily expected returns
sigma = np.array([0.02, 0.015, 0.03])          # daily volatilities

# Correlation matrix
corr_matrix = np.array([
    [1.00, 0.30, 0.10],
    [0.30, 1.00, 0.25],
    [0.10, 0.25, 1.00]
])

portfolio_value = 1_000_000
n_sims = 100000

# === Step 1: build covariance matrix ===
cov_matrix = np.outer(sigma, sigma) * corr_matrix

# === Step 2: Cholesky decomposition ===
L = np.linalg.cholesky(cov_matrix)

# === Step 3: simulate correlated returns ===
Z = np.random.normal(size=(n_sims, 3))     # independent random shocks
correlated_returns = Z @ L.T + mu          # add expected returns

# === Step 4: compute portfolio return ===
portfolio_returns = correlated_returns @ weights

# === Step 5: convert to losses ===
portfolio_values = portfolio_value * (1 + portfolio_returns)
losses = portfolio_value - portfolio_values

# === Step 6: compute VaR ===
VaR_95 = np.percentile(losses, 95)
VaR_99 = np.percentile(losses, 99)

print(f"Multivariate Monte Carlo 95% VaR: {VaR_95:,.0f}")
print(f"Multivariate Monte Carlo 99% VaR: {VaR_99:,.0f}")

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