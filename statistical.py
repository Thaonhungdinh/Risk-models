import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

# ================================
# 1. Simulate synthetic returns
# ================================
n_assets = 30
n_obs = 120  # 10 years of monthly returns
n_factors = 5  # number of statistical factors

# Simulate factor returns
factor_returns = np.random.normal(0, 0.02, size=(n_obs, n_factors))

# Simulate factor loadings (betas)
betas = np.random.uniform(-1, 1, size=(n_assets, n_factors))

# Residual (specific) returns
specific_returns = np.random.normal(0, 0.01, size=(n_obs, n_assets))

# Generate asset returns: R = B * F + epsilon
asset_returns = factor_returns @ betas.T + specific_returns
asset_returns = pd.DataFrame(asset_returns, columns=[f'Asset{i+1}' for i in range(n_assets)])

# ================================
# 2. Statistical Factor Model (PCA)
# ================================
pca = PCA(n_components=n_factors)
pca.fit(asset_returns)

# Factor loadings (betas)
factor_loadings = pca.components_.T  # shape: (assets, factors)

# Factor returns (principal components)
factor_scores = pca.transform(asset_returns)

# ================================
# 3. Volatility decomposition
# ================================
# Total variance
total_var = asset_returns.var().values

# Systematic variance (from factors)
sys_var = np.var(factor_scores @ factor_loadings.T, axis=0)

# Specific variance (residual)
spec_var = total_var - sys_var

# R-squared
r_squared = sys_var / total_var

# ================================
# 4. Visualization
# ================================
assets = asset_returns.columns
factors = [f'Factor{i+1}' for i in range(n_factors)]

# Bar chart: factor loadings per asset
fig, ax = plt.subplots(figsize=(16,6))
width = 0.15
x = np.arange(n_assets)
for i in range(n_factors):
    ax.bar(x + i*width, factor_loadings[:, i], width=width, label=factors[i])

ax.set_xticks(x + width*(n_factors/2))
ax.set_xticklabels(assets, rotation=45)
ax.set_ylabel("Factor Loading")
ax.set_title("Statistical Factor Loadings per Asset")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Multi-dimensional factor vectors (scatter in 3D for first 3 factors)
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(np.zeros(n_assets), np.zeros(n_assets), np.zeros(n_assets),
          factor_loadings[:,0], factor_loadings[:,1], factor_loadings[:,2],
          length=0.1, normalize=True, color='blue')
ax.set_xlabel('Factor1')
ax.set_ylabel('Factor2')
ax.set_zlabel('Factor3')
ax.set_title('Assets as Multi-dimensional Factor Vectors')
plt.show()

# ================================
# 5. Display volatility decomposition
# ================================
summary = pd.DataFrame({
    'Asset': assets,
    'Total_Vol': np.sqrt(total_var),
    'Systematic_Vol': np.sqrt(sys_var),
    'Specific_Vol': np.sqrt(spec_var),
    'R_squared': r_squared
})

pd.set_option('display.float_format', '{:.4f}'.format)
print(summary)
