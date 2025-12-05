import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ====== Simulate data ======
np.random.seed(42)
n_assets = 30
n_obs = 120  # e.g., 10 years of monthly returns
n_factors = 5

# Simulate asset returns
returns = np.random.normal(0, 0.02, (n_obs, n_assets))

# ====== Statistical factor extraction (PCA) ======
pca = PCA(n_components=n_factors)
pca.fit(returns)
factor_loadings = pca.components_.T  # shape: (n_assets, n_factors)

# Total variance per asset
total_var = returns.var(axis=0)

# ====== 3D Factor Vectors (first 3 factors) ======
x, y, z = factor_loadings[:, 0], factor_loadings[:, 1], factor_loadings[:, 2]

# Normalize colors to total volatility
colors = (total_var - total_var.min()) / (total_var.max() - total_var.min())

fig = go.Figure()

for i in range(n_assets):
    fig.add_trace(go.Scatter3d(
        x=[0, x[i]],
        y=[0, y[i]],
        z=[0, z[i]],
        mode='lines+markers',
        marker=dict(size=5, color=colors[i], colorscale='Viridis', cmin=0, cmax=1),
        line=dict(width=4, color=f'rgba({int(colors[i]*255)}, {int((1-colors[i])*255)}, 150, 0.8)'),
        name=f'Asset {i+1}'
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='Factor 1',
        yaxis_title='Factor 2',
        zaxis_title='Factor 3'
    ),
    title='Multi-dimensional Factor Vectors (Statistical Factor Model)',
    width=900,
    height=700,
)

fig.show()

# ====== Bar Chart of Factor Loadings ======
asset_labels = [f'Asset {i+1}' for i in range(n_assets)]
factor_labels = [f'Factor {i+1}' for i in range(n_factors)]

bar_fig = go.Figure()

for i in range(n_assets):
    bar_fig.add_trace(go.Bar(
        x=factor_labels,
        y=factor_loadings[i, :],
        name=asset_labels[i]
    ))

bar_fig.update_layout(
    barmode='group',
    title='Factor Loadings per Asset',
    xaxis_title='Factors',
    yaxis_title='Loading Value',
    width=1000,
    height=500
)

bar_fig.show()

