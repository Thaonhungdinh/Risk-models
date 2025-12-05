import numpy as np
import pandas as pd

# ============================
# 1. SIMULATE DATA
# ============================

np.random.seed(0)

n_port = 50
n_bench = 250

# Benchmark universe
assets = [f"A{i}" for i in range(n_bench)]

beta = np.random.uniform(0.8, 1.3, n_bench)               # betas
spec_vol = np.random.uniform(0.10, 0.30, n_bench)         # specific vol
market_vol = 0.20                                         # market vol (annual)

# Benchmark weights
w_bench = np.random.random(n_bench)
w_bench /= w_bench.sum()

# Portfolio is a subset of benchmark
port_indices = np.random.choice(n_bench, n_port, replace=False)
w_port = np.zeros(n_bench)
w_port[port_indices] = 1 / n_port

# ============================
# 2. SYSTEMATIC RISK
# ============================

beta_port = np.sum(w_port * beta)
beta_bench = np.sum(w_bench * beta)

sys_vol_port = abs(beta_port) * market_vol
sys_vol_bench = abs(beta_bench) * market_vol

# ============================
# 3. SPECIFIC RISK
# ============================

spec_vol_port = np.sqrt(np.sum((w_port * spec_vol)**2))
spec_vol_bench = np.sqrt(np.sum((w_bench * spec_vol)**2))

# ============================
# 4. TOTAL APT VOL
# ============================

apt_vol_port = np.sqrt(sys_vol_port**2 + spec_vol_port**2)
apt_vol_bench = np.sqrt(sys_vol_bench**2 + spec_vol_bench**2)

# ============================
# 5. TRACKING ERROR
# ============================

# systematic TE
delta_beta = beta_port - beta_bench
te_systematic = abs(delta_beta) * market_vol

# specific TE
te_specific = np.sqrt(np.sum(((w_port - w_bench) * spec_vol)**2))

# total TE
te_total = np.sqrt(te_systematic**2 + te_specific**2)

# ============================
# 6. TRACKING AT RISK (68%)
# ============================

tar_68 = te_total * 1.0   # 1 SD

# ============================
# 7. SYSTEMATIC BETA & CORR
# ============================

systematic_beta = beta_port
systematic_corr = sys_vol_port / apt_vol_port

# ============================
# 8. NUMBER OF ASSETS & OVERLAP
# ============================

overlap = np.sum(np.minimum(w_port, w_bench))

# ============================
# 9. MC TO TE
# ============================

MC = ((w_port - w_bench) * (spec_vol**2)) / (te_total + 1e-12)

mc_table = pd.DataFrame({
    "Asset": assets,
    "MC_to_TE": MC,
    "Abs_MC_to_TE": np.abs(MC)
}).sort_values("Abs_MC_to_TE", ascending=False)

preferences_table = mc_table.head(10)[["Asset", "MC_to_TE"]]

# ============================
# 10. BUILD FINAL REPORT TABLE
# ============================

report = pd.DataFrame({
    "Portfolio": [
        apt_vol_port,
        sys_vol_port,
        spec_vol_port,
        te_total,
        te_systematic,
        te_specific,
        tar_68,
        systematic_beta,
        systematic_corr,
        n_port,
        n_bench,
        overlap
    ],
    "Benchmark": [
        apt_vol_bench,
        sys_vol_bench,
        spec_vol_bench,
        "",
        "",
        "",
        "",
        beta_bench,
        sys_vol_bench / apt_vol_bench,
        "",
        "",
        ""
    ]
},
index=[
    "APT Volatility (Total)",
    "Systematic Vol",
    "Specific Vol",
    "Tracking Error (Total)",
    "TE Systematic",
    "TE Specific",
    "Tracking-at-Risk (68%)",
    "Systematic Beta",
    "Systematic Correlation",
    "Portfolio #Assets",
    "Benchmark #Assets",
    "Overlap"
])

print("\n========== RISK REPORT ==========\n")
print(report.round(6))

print("\n========== TOP 10 MC TO TE ==========\n")
print(preferences_table.round(6))
