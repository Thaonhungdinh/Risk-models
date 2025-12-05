import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------
# 1. SIMULATE FUNDAMENTAL SCORES
# --------------------------
n_assets = 30
n_months = 60
factors = ['Value', 'Momentum', 'Quality']

# Simulate fundamental scores for each stock (e.g., book-to-market, ROE)
scores = pd.DataFrame(np.random.normal(0, 1, size=(n_assets, len(factors))),
                      index=[f'Asset_{i+1}' for i in range(n_assets)],
                      columns=factors)

# Normalize scores to get betas
betas_fund = scores.div(scores.std(axis=0), axis=1)  # standardized exposure

print("=== Asset Factor Betas (Fundamental Model) ===")
print(betas_fund.head())

# --------------------------
# 2. SIMULATE ASSET RETURNS
# --------------------------
# Factor returns (monthly)
factor_returns = pd.DataFrame(np.random.normal(0, 0.01, size=(n_months, len(factors))),
                              columns=factors)

# Idiosyncratic noise
specific_vol = 0.02
returns = pd.DataFrame(0, index=range(n_months), columns=scores.index)

for i, asset in enumerate(scores.index):
    # asset return = sum(beta*factor_return) + noise
    returns[asset] = factor_returns.values @ betas_fund.loc[asset].values + np.random.normal(0, specific_vol, n_months)

# --------------------------
# 3. PORTFOLIO METRICS
# --------------------------
weights = np.array([1/n_assets]*n_assets)

# Total portfolio return
portfolio_returns = returns @ weights

# Covariance of factor returns
factor_cov = factor_returns.cov().values

# Systematic covariance: B Σ_f B'
B = betas_fund.values
D = np.diag([specific_vol**2]*n_assets)  # idiosyncratic variance
cov_sys = B @ factor_cov @ B.T
cov_total = cov_sys + D

total_vol = np.sqrt(weights.T @ cov_total @ weights)
systematic_vol = np.sqrt(weights.T @ cov_sys @ weights)
specific_port_vol = np.sqrt(total_vol**2 - systematic_vol**2)

print("\n=== Portfolio Metrics ===")
print(f"Total Volatility: {total_vol:.4f}")
print(f"Systematic Volatility: {systematic_vol:.4f}")
print(f"Specific Volatility: {specific_port_vol:.4f}")

# --------------------------
# 4. VISUALIZE BETAS
# --------------------------
plt.figure(figsize=(12,6))
for factor in factors:
    plt.bar(betas_fund.index, betas_fund[factor], alpha=0.7, label=factor)
plt.xticks(rotation=90)
plt.ylabel("Beta")
plt.title("Asset Factor Exposures (Fundamental Factor Model)")
plt.legend()
plt.tight_layout()
plt.show()
