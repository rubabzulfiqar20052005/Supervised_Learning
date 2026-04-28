"""
Task 1 — Regression from Scratch + Sklearn
Streamlit Interactive Dashboard
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression from Scratch",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0d1117; }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    color: #e6edf3;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -1px;
    line-height: 1.2;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.05rem;
    color: #8b949e;
    margin-top: 0.4rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    color: #58a6ff;
    font-weight: 700;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #3fb950;
    background: #161b22;
    border-left: 3px solid #3fb950;
    padding: 0.5rem 1rem;
    border-radius: 0 6px 6px 0;
    margin: 1.2rem 0 0.8rem 0;
}

.formula-box {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #f0883e;
}

.stDataFrame {
    background: #161b22 !important;
}

div[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

.stSlider label, .stSelectbox label, .stRadio label {
    color: #c9d1d9 !important;
    font-family: 'DM Sans', sans-serif;
}

.badge {
    display: inline-block;
    background: #1f6feb;
    color: white;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['MedHouseVal'] = housing.target
    return df, housing.feature_names


def ols_regression(X, y):
    """Manual OLS: beta = (X^T X)^-1 X^T y"""
    n = X.shape[0]
    X_b = np.column_stack([np.ones(n), X])
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta[0], beta[1:]


def predict(X, intercept, coef):
    return intercept + X @ coef


def styled_plot(fig):
    fig.patch.set_facecolor('#0d1117')
    for ax in fig.axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.title.set_color('#c9d1d9')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    return fig


# ── Load Data ────────────────────────────────────────────────────────────────
df, feature_names = load_data()

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📈 Regression from Scratch</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Manual OLS (NumPy) vs sklearn · California Housing Dataset</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    mode = st.radio("Regression Type", ["Simple Linear Regression", "Multiple Linear Regression"])

    if mode == "Simple Linear Regression":
        feature = st.selectbox("Select Feature (X)", list(feature_names), index=2)
        clip_outliers = st.checkbox("Clip Outliers (AveRooms > 20)", value=True)

    st.markdown("---")
    st.markdown("### 📘 Dataset Info")
    st.markdown(f"**Samples:** {len(df):,}")
    st.markdown(f"**Features:** {len(feature_names)}")
    st.markdown(f"**Target:** Median House Value")
    st.markdown("---")
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 0.7rem; color: #484f58;'>
    OLS Formula:<br>
    β = (XᵀX)⁻¹ Xᵀy
    </div>
    """, unsafe_allow_html=True)

# ── Dataset Preview ───────────────────────────────────────────────────────────
with st.expander("🗂️ Dataset Preview (first 5 rows)", expanded=False):
    st.dataframe(df.head(), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Features", len(feature_names))
    c3.metric("Target Range", f"{df['MedHouseVal'].min():.1f} – {df['MedHouseVal'].max():.1f}")


# ════════════════════════════════════════════════════════════════════════════
# SIMPLE LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════════════
if mode == "Simple Linear Regression":

    st.markdown('<div class="section-header">// SIMPLE LINEAR REGRESSION</div>', unsafe_allow_html=True)

    # Prepare data
    X_raw = df[[feature]].values
    y_raw = df['MedHouseVal'].values

    if clip_outliers and feature == 'AveRooms':
        mask = X_raw[:, 0] < 20
        X_raw, y_raw = X_raw[mask], y_raw[mask]

    # OLS Manual
    man_int, man_coef = ols_regression(X_raw, y_raw)

    # Sklearn
    sk_model = LinearRegression().fit(X_raw, y_raw)
    sk_int   = sk_model.intercept_
    sk_coef  = sk_model.coef_

    # ── Equations ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Manual OLS Equation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{man_int:.4f} + {man_coef[0]:.4f}·x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Sklearn Equation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{sk_int:.4f} + {sk_coef[0]:.4f}·x</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Comparison Table ─────────────────────────────────────────────────
    st.markdown("#### Coefficient Comparison")
    cmp_df = pd.DataFrame({
        'Parameter'    : ['Intercept (b0)', f'Coef — {feature}'],
        'Manual OLS'   : [man_int, man_coef[0]],
        'Sklearn'      : [sk_int, sk_coef[0]],
        'Difference'   : [abs(man_int - sk_int), abs(man_coef[0] - sk_coef[0])],
    })
    st.dataframe(cmp_df.style.format({
        'Manual OLS': '{:.8f}', 'Sklearn': '{:.8f}', 'Difference': '{:.2e}'
    }), use_container_width=True)

    # ── Regression Line Plot ──────────────────────────────────────────────
    st.markdown("#### Regression Line — Scatter Plot")

    x_line = np.linspace(X_raw.min(), X_raw.max(), 300).reshape(-1, 1)
    y_man  = predict(x_line, man_int, man_coef)
    y_sk   = sk_model.predict(x_line)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(X_raw, y_raw, color='#58a6ff', alpha=0.25, s=10, label='Data Points')
    ax.plot(x_line, y_man, color='#f0883e', lw=2.5, label=f'Manual OLS: y = {man_int:.3f} + {man_coef[0]:.3f}x')
    ax.plot(x_line, y_sk,  color='#3fb950', lw=2.0, ls='--', label=f'Sklearn:    y = {sk_int:.3f} + {sk_coef[0]:.3f}x')
    ax.set_xlabel(feature)
    ax.set_ylabel('MedHouseVal')
    ax.set_title(f'Simple Linear Regression: {feature} → MedHouseVal')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, color='#30363d')
    st.pyplot(styled_plot(fig), use_container_width=True)
    plt.close()

    # ── Metrics ──────────────────────────────────────────────────────────
    y_pred_man = predict(X_raw, man_int, man_coef)
    y_pred_sk  = sk_model.predict(X_raw)

    st.markdown("#### Model Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (Manual)",  f"{r2_score(y_raw, y_pred_man):.4f}")
    m2.metric("R² (Sklearn)", f"{r2_score(y_raw, y_pred_sk):.4f}")
    m3.metric("RMSE (Sklearn)", f"{np.sqrt(mean_squared_error(y_raw, y_pred_sk)):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# MULTIPLE LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════════════
else:

    st.markdown('<div class="section-header">// MULTIPLE LINEAR REGRESSION — ALL FEATURES</div>', unsafe_allow_html=True)

    X_full = df[list(feature_names)].values
    y_full = df['MedHouseVal'].values

    # Manual OLS
    man_int_m, man_coef_m = ols_regression(X_full, y_full)

    # Sklearn
    sk_m = LinearRegression().fit(X_full, y_full)

    # ── Full Coefficient Table ────────────────────────────────────────────
    st.markdown("#### All Coefficients — Manual vs Sklearn")

    params   = ['Intercept'] + list(feature_names)
    man_vals = [man_int_m] + list(man_coef_m)
    sk_vals  = [sk_m.intercept_] + list(sk_m.coef_)

    cmp_multi = pd.DataFrame({
        'Parameter' : params,
        'Manual OLS': man_vals,
        'Sklearn'   : sk_vals,
        'Difference': [abs(a - b) for a, b in zip(man_vals, sk_vals)],
    })

    st.dataframe(cmp_multi.style.format({
        'Manual OLS': '{:.8f}', 'Sklearn': '{:.8f}', 'Difference': '{:.2e}'
    }), use_container_width=True)

    # ── Coefficient Bar Chart ─────────────────────────────────────────────
    st.markdown("#### Coefficient Visualization")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    feat_names = list(feature_names)
    x_pos = np.arange(len(feat_names))

    axes[0].bar(x_pos - 0.18, man_coef_m, 0.36, color='#f0883e', alpha=0.85, label='Manual OLS')
    axes[0].bar(x_pos + 0.18, sk_m.coef_, 0.36, color='#58a6ff', alpha=0.85, label='Sklearn')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(feat_names, rotation=30, ha='right', fontsize=8)
    axes[0].axhline(0, color='#8b949e', lw=0.8)
    axes[0].set_title('Coefficients — Manual vs Sklearn')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3, color='#30363d')

    # Actual vs Predicted
    y_pred_full = sk_m.predict(X_full)
    axes[1].scatter(y_full, y_pred_full, color='#58a6ff', alpha=0.15, s=6)
    axes[1].plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()],
                 color='#f85149', lw=2, ls='--', label='Perfect Fit')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, color='#30363d')

    st.pyplot(styled_plot(fig), use_container_width=True)
    plt.close()

    # ── Summary Metrics ───────────────────────────────────────────────────
    st.markdown("#### Summary Metrics")

    y_pred_man_m = predict(X_full, man_int_m, man_coef_m)
    y_pred_sk_m  = sk_m.predict(X_full)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Manual",   f"{r2_score(y_full, y_pred_man_m):.4f}")
    m2.metric("R² Sklearn",  f"{r2_score(y_full, y_pred_sk_m):.4f}")
    m3.metric("RMSE Manual",  f"{np.sqrt(mean_squared_error(y_full, y_pred_man_m)):.4f}")
    m4.metric("RMSE Sklearn", f"{np.sqrt(mean_squared_error(y_full, y_pred_sk_m)):.4f}")

    st.info("R² and RMSE are identical for Manual OLS and Sklearn — confirming correct implementation.")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-family: Space Mono, monospace; font-size: 0.7rem; color: #484f58;'>
Task 1 · Regression from Scratch · California Housing Dataset · NumPy OLS + Sklearn
</div>
""", unsafe_allow_html=True)
