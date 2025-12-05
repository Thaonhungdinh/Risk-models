import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# --------------------------
# 1. SIMULATE DATA
# --------------------------

np.random.seed(42)

n_assets = 30
n_months = 60  # 5 years monthly
n_factors = 3

# Macro factors: GDP, Inflation, Interest Rate
factors = pd.DataFrame(
    np.random.normal(0, 0.01, size=(n_months, n_factors)),
    columns=['GDP', 'Inflation', 'IR']
)

# True betas and alphas
betas_true = np.random.uniform(-1, 1, size=(n_assets, n_factors))
alphas_true = np.random.uniform(-0.01, 0.01, size=n_assets)

# Simulate asset returns: R = alpha + beta*F + noise
returns = pd.DataFrame(
    np.zeros((n_months, n_assets)),
    columns=[f'Asset_{i + 1}' for i in range(n_assets)]
)

for i in range(n_assets):
    returns.iloc[:, i] = alphas_true[i] + factors.values @ betas_true[i, :] + np.random.normal(0, 0.02, n_months)

# Risk-free rate
rf = 0.002  # 0.2% per month
returns_excess = returns - rf

# --------------------------
# 2. ESTIMATE BETAS, ALPHA, RSQUARED, SPECIFIC VOLATILITY
# --------------------------

betas_est = pd.DataFrame(index=returns.columns, columns=factors.columns)
alphas_est = pd.Series(index=returns.columns)
r_squared = pd.Series(index=returns.columns)
specific_vol = pd.Series(index=returns.columns)

for asset in returns.columns:
    y = returns_excess[asset]
    X = sm.add_constant(factors)  # add alpha term
    model = sm.OLS(y, X).fit()

    alphas_est[asset] = model.params['const']
    betas_est.loc[asset, :] = model.params[factors.columns]
    r_squared[asset] = model.rsquared
    specific_vol[asset] = np.sqrt(model.mse_resid)  # idiosyncratic volatility

# --------------------------
# 3. COMPUTE PORTFOLIO METRICS
# --------------------------

# Example: equal-weighted portfolio
weights = np.array([1 / n_assets] * n_assets)

# Portfolio total return series
portfolio_returns = (returns_excess * weights).sum(axis=1)

# Portfolio covariance using factor model: Σ = BΣfB' + D
factor_cov = factors.cov()  # factor covariance
B = betas_est.values.astype(float)
D = np.diag(specific_vol.values ** 2)

# Systematic (factor-driven) covariance
cov_sys = B @ factor_cov.values @ B.T
# Total portfolio variance
cov_total = cov_sys + D
# Portfolio volatility
total_vol = np.sqrt(weights.T @ cov_total @ weights)

# Systematic volatility (factor contribution)
systematic_vol = np.sqrt(weights.T @ cov_sys @ weights)
# Specific volatility
specific_port_vol = np.sqrt(total_vol ** 2 - systematic_vol ** 2)

print("========== PORTFOLIO METRICS ==========")
print(f"Total Portfolio Volatility: {total_vol:.4f}")
print(f"Systematic Volatility: {systematic_vol:.4f}")
print(f"Specific Volatility: {specific_port_vol:.4f}")

# --------------------------
# 4. VISUALIZE BETAS
# --------------------------

plt.figure(figsize=(12, 6))
for factor in factors.columns:
    plt.bar(betas_est.index, betas_est[factor].astype(float), alpha=0.7, label=factor)
plt.xticks(rotation=90)
plt.ylabel("Beta")
plt.title("Asset Factor Exposures (Betas)")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 5. INTERPRETATION
# --------------------------

"""
Interpretation Guide:

1. Betas:
   - Positive beta -> asset moves in the same direction as the factor.
   - Negative beta -> asset moves opposite to the factor.
   - Magnitude shows sensitivity.

2. R-squared:
   - High R^2 (~1) means factor model explains most of the asset's return.
   - Low R^2 (~0) means most return is idiosyncratic.

3. Portfolio Volatility:
   - Total volatility combines systematic + specific risk.
   - Systematic volatility = portion explained by factors.
   - Specific volatility = portion due to asset-specific risk.
   - Weighting can reduce idiosyncratic risk (diversification).

4. Factor Visualization:
   - Shows which factors dominate the risk of individual assets.
   - Helps in risk management and factor tilting.
"""

