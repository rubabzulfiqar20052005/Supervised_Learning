"""
Task 2 — Metrics Dashboard + Gradient Descent from Scratch
Streamlit Interactive Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metrics + Gradient Descent",
    page_icon="🎯",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    color: #f0f0f0;
}

.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.3rem;
}

.metric-tile {
    background: linear-gradient(145deg, #1e1b4b, #1e293b);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 0.6rem;
}

.tile-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.tile-manual {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    color: #a78bfa;
    font-weight: 600;
}

.tile-sklearn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: #60a5fa;
}

.tile-diff {
    font-size: 0.72rem;
    color: #4ade80;
    font-family: 'IBM Plex Mono', monospace;
}

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    color: #a78bfa;
    background: #1e1b4b;
    border-left: 3px solid #7c3aed;
    padding: 0.5rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.4rem 0 1rem 0;
}

.gd-stat {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}

.gd-stat span {
    color: #60a5fa;
}

div[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

.stSlider label {
    color: #c7d2fe !important;
    font-family: 'Outfit', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    names = housing.feature_names
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_tr, y_te if False else y_tr)
    y_pred = model.predict(X_te)
    return X_te, y_te, y_pred, len(y_te), len(names)


@st.cache_data
def run_gd(lr, n_iter, noise_level, seed):
    np.random.seed(seed)
    n = 150
    X = np.random.uniform(0, 10, n)
    y = 2.5 * X + 8.0 + np.random.randn(n) * noise_level
    X_norm = (X - X.mean()) / X.std()

    w, b = 0.0, 0.0
    cost_history = []
    w_history    = []
    b_history    = []

    for _ in range(n_iter):
        y_hat = w * X_norm + b
        res   = y_hat - y
        cost  = np.mean(res ** 2)
        dw    = (2 / n) * np.dot(res, X_norm)
        db    = (2 / n) * np.sum(res)
        w    -= lr * dw
        b    -= lr * db
        cost_history.append(cost)
        w_history.append(w)
        b_history.append(b)

    sk = LinearRegression().fit(X_norm.reshape(-1, 1), y)
    return X_norm, y, cost_history, w_history, b_history, w, b, sk.coef_[0], sk.intercept_


def styled_plot(fig):
    fig.patch.set_facecolor('#0f172a')
    for ax in fig.axes:
        ax.set_facecolor('#1e1b4b')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#312e81')
    return fig


# ── Manual Metrics ────────────────────────────────────────────────────────────
def compute_all_metrics(y_true, y_pred, n, p):
    mae  = np.mean(np.abs(y_true - y_pred))
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_r = np.sum((y_true - y_pred) ** 2)
    ss_t = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_r / ss_t
    adj  = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return mae, mse, rmse, r2, adj


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">🎯 Metrics Dashboard + GD from Scratch</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">MAE · MSE · RMSE · R² · Adjusted R² · Batch Gradient Descent</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["📊  Metrics Dashboard", "⬇️  Gradient Descent"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — METRICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    X_te, y_te, y_pred, n, p = load_data()

    mae_m, mse_m, rmse_m, r2_m, adj_m = compute_all_metrics(y_te, y_pred, n, p)

    mae_sk   = mean_absolute_error(y_te, y_pred)
    mse_sk   = mean_squared_error(y_te, y_pred)
    rmse_sk  = np.sqrt(mse_sk)
    r2_sk    = r2_score(y_te, y_pred)
    adj_sk   = 1 - (1 - r2_sk) * (n - 1) / (n - p - 1)

    st.markdown('<div class="section-header">// METRIC TILES — Manual NumPy vs Sklearn</div>', unsafe_allow_html=True)

    metrics = [
        ("MAE",         mae_m,  mae_sk),
        ("MSE",         mse_m,  mse_sk),
        ("RMSE",        rmse_m, rmse_sk),
        ("R²",          r2_m,   r2_sk),
        ("Adjusted R²", adj_m,  adj_sk),
    ]

    cols = st.columns(5)
    for col, (label, manual_v, sk_v) in zip(cols, metrics):
        diff = abs(manual_v - sk_v)
        col.markdown(f"""
        <div class="metric-tile">
            <div class="tile-label">{label}</div>
            <div class="tile-manual">{manual_v:.4f}</div>
            <div class="tile-sklearn">sk: {sk_v:.4f}</div>
            <div class="tile-diff">Δ {diff:.2e}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Purple = Manual NumPy · Blue = Sklearn · Green Δ = Absolute Difference")

    # ── Formula Expander ──────────────────────────────────────────────────
    with st.expander("📐 View Metric Formulas"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **MAE** = (1/n) Σ |yᵢ - ŷᵢ|

            **MSE** = (1/n) Σ (yᵢ - ŷᵢ)²

            **RMSE** = √MSE
            """)
        with c2:
            st.markdown("""
            **R²** = 1 - SS_res / SS_tot

            **Adjusted R²** = 1 - (1-R²)(n-1)/(n-p-1)

            where p = number of features
            """)

    # ── Comparison Table ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">// COMPARISON TABLE</div>', unsafe_allow_html=True)

    cmp_df = pd.DataFrame({
        'Metric'        : [m[0] for m in metrics],
        'Manual (NumPy)': [m[1] for m in metrics],
        'Sklearn'       : [m[2] for m in metrics],
        'Difference'    : [abs(m[1] - m[2]) for m in metrics],
    })

    st.dataframe(
        cmp_df.style.format({
            'Manual (NumPy)': '{:.8f}',
            'Sklearn'       : '{:.8f}',
            'Difference'    : '{:.2e}',
        }).highlight_min(subset=['Difference'], color='#14532d'),
        use_container_width=True,
        hide_index=True,
    )

    # ── Plots ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">// VISUAL DASHBOARD</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Plot 1: Error metrics bar chart
    ax = axes[0]
    err_labels  = ['MAE', 'MSE', 'RMSE']
    err_manual  = [mae_m, mse_m, rmse_m]
    err_sklearn = [mae_sk, mse_sk, rmse_sk]
    x = np.arange(3)
    ax.bar(x - 0.2, err_manual,  0.38, color='#a78bfa', alpha=0.9, label='Manual')
    ax.bar(x + 0.2, err_sklearn, 0.38, color='#60a5fa', alpha=0.9, label='Sklearn')
    ax.set_xticks(x); ax.set_xticklabels(err_labels)
    ax.set_title('Error Metrics')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3, color='#312e81')

    # Plot 2: R² comparison
    ax2 = axes[1]
    r2_labels  = ['R²', 'Adj R²']
    r2_man_v   = [r2_m, adj_m]
    r2_sk_v    = [r2_sk, adj_sk]
    x2 = np.arange(2)
    bars_m = ax2.bar(x2 - 0.2, r2_man_v, 0.38, color='#a78bfa', alpha=0.9, label='Manual')
    bars_s = ax2.bar(x2 + 0.2, r2_sk_v,  0.38, color='#60a5fa', alpha=0.9, label='Sklearn')
    ax2.set_xticks(x2); ax2.set_xticklabels(r2_labels)
    ax2.set_ylim(0, 1); ax2.set_title('R² Scores')
    for bar in list(bars_m) + list(bars_s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8, color='#e2e8f0')
    ax2.legend(fontsize=8)
    ax2.grid(True, axis='y', alpha=0.3, color='#312e81')

    # Plot 3: Actual vs Predicted
    ax3 = axes[2]
    ax3.scatter(y_te, y_pred, alpha=0.2, s=8, color='#818cf8')
    ax3.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()],
             color='#f472b6', lw=2, ls='--', label='Perfect Fit')
    ax3.set_xlabel('Actual'); ax3.set_ylabel('Predicted')
    ax3.set_title('Actual vs Predicted')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, color='#312e81')

    st.pyplot(styled_plot(fig), use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GRADIENT DESCENT
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown('<div class="section-header">// BATCH GRADIENT DESCENT — INTERACTIVE</div>', unsafe_allow_html=True)

    # ── Sidebar Controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ GD Controls")
        lr          = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2], value=0.05)
        n_iter      = st.slider("Iterations", 50, 500, 300, step=50)
        noise_level = st.slider("Noise Level", 1.0, 8.0, 3.0, step=0.5)
        seed        = st.number_input("Random Seed", value=42, step=1)

    X_norm, y_gd, cost_history, w_hist, b_hist, w_f, b_f, sk_w, sk_b = run_gd(
        lr, n_iter, noise_level, int(seed)
    )

    # ── Current Iteration Slider ──────────────────────────────────────────
    st.markdown("#### Animate Regression Line — Drag Slider")
    iter_idx = st.slider("Current Iteration", 0, n_iter - 1, n_iter - 1, step=1)

    w_cur = w_hist[iter_idx]
    b_cur = b_hist[iter_idx]
    cost_cur = cost_history[iter_idx]

    # ── GD Stats ──────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Iteration",    f"{iter_idx + 1} / {n_iter}")
    g2.metric("Current Cost", f"{cost_cur:.4f}")
    g3.metric("Weight (w)",   f"{w_cur:.4f}")
    g4.metric("Bias (b)",     f"{b_cur:.4f}")

    # ── Main Plots ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x_range = np.linspace(X_norm.min(), X_norm.max(), 200)

    # Left: Regression Line Animation
    ax_line = axes[0]
    ax_line.scatter(X_norm, y_gd, color='#818cf8', alpha=0.5, s=18, label='Data Points', zorder=2)

    # Final converged line
    ax_line.plot(x_range, w_f * x_range + b_f, color='#4ade80', lw=2, ls='--',
                 label=f'Final (iter {n_iter})', zorder=3)
    # Sklearn line
    ax_line.plot(x_range, sk_w * x_range + sk_b, color='#fbbf24', lw=1.5, ls=':',
                 label='Sklearn OLS', zorder=3)
    # Current GD line
    ax_line.plot(x_range, w_cur * x_range + b_cur, color='#f472b6', lw=2.8,
                 label=f'GD @ iter {iter_idx + 1}', zorder=4)

    ax_line.set_xlabel('X (normalized)')
    ax_line.set_ylabel('y')
    ax_line.set_title(f'Regression Line @ Iteration {iter_idx + 1}\nw={w_cur:.3f}  b={b_cur:.3f}  Cost={cost_cur:.2f}')
    ax_line.legend(fontsize=8, loc='upper left')
    ax_line.grid(True, ls='--', alpha=0.3, color='#312e81')

    # Right: Cost Curve
    ax_cost = axes[1]
    ax_cost.plot(range(n_iter), cost_history, color='#818cf8', lw=2, label='Cost (MSE)')
    ax_cost.axvline(iter_idx, color='#f472b6', ls='--', lw=1.5, label=f'Iter {iter_idx + 1}')
    ax_cost.scatter([iter_idx], [cost_cur], color='#f472b6', s=60, zorder=5)
    ax_cost.axhline(cost_history[-1], color='#4ade80', ls=':', lw=1.2,
                    label=f'Final Cost = {cost_history[-1]:.2f}')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('MSE Cost')
    ax_cost.set_title('Cost Curve — Gradient Descent')
    ax_cost.legend(fontsize=8)
    ax_cost.grid(True, ls='--', alpha=0.3, color='#312e81')

    st.pyplot(styled_plot(fig), use_container_width=True)
    plt.close()

    # ── Learning Rate Comparison ──────────────────────────────────────────
    st.markdown('<div class="section-header">// LEARNING RATE COMPARISON</div>', unsafe_allow_html=True)

    lr_list    = [0.001, 0.01, 0.05, 0.1]
    lr_colors  = ['#60a5fa', '#a78bfa', '#4ade80', '#f87171']

    fig2, ax_lr = plt.subplots(figsize=(11, 4))
    for lr_val, clr in zip(lr_list, lr_colors):
        _, _, ch, _, _, _, _, _, _ = run_gd(lr_val, n_iter, noise_level, int(seed))
        ax_lr.plot(ch, color=clr, lw=2, label=f'LR = {lr_val}')

    ax_lr.set_xlabel('Iteration')
    ax_lr.set_ylabel('MSE Cost')
    ax_lr.set_title('Effect of Learning Rate on Convergence')
    ax_lr.legend(fontsize=10)
    ax_lr.set_ylim(bottom=0)
    ax_lr.grid(True, ls='--', alpha=0.3, color='#312e81')

    st.pyplot(styled_plot(fig2), use_container_width=True)
    plt.close()

    # ── GD vs Sklearn comparison ──────────────────────────────────────────
    st.markdown('<div class="section-header">// GD vs SKLEARN — FINAL COMPARISON</div>', unsafe_allow_html=True)

    cmp = pd.DataFrame({
        'Parameter'        : ['Weight (w)', 'Bias (b)'],
        'Gradient Descent' : [w_f,  b_f],
        'Sklearn OLS'      : [sk_w, sk_b],
        'Difference'       : [abs(w_f - sk_w), abs(b_f - sk_b)],
    })
    st.dataframe(
        cmp.style.format({
            'Gradient Descent': '{:.8f}',
            'Sklearn OLS'     : '{:.8f}',
            'Difference'      : '{:.2e}',
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.success(f"GD converged in {n_iter} iterations with LR={lr}. Final cost: {cost_history[-1]:.4f}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family: IBM Plex Mono, monospace; font-size: 0.7rem; color: #334155;'>
Task 2 · Metrics Dashboard + Batch Gradient Descent · NumPy from Scratch + Sklearn Verification
</div>
""", unsafe_allow_html=True)
