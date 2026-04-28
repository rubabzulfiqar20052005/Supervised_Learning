import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Regularization Dashboard", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background: #1c1f26;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🚀 Regularization Dashboard")
st.markdown("### Compare Linear, Ridge & Lasso with Interactive Visualizations")

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

dataset_option = st.sidebar.radio(
    "Dataset",
    ["Synthetic", "California Housing"]
)

model_option = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Ridge", "Lasso"]
)

alpha = st.sidebar.slider("Alpha", 0.001, 10.0, 1.0)

show_all_models = st.sidebar.checkbox("Compare All Models", True)

# ------------------ DATA ------------------
if dataset_option == "Synthetic":
    X, y = make_regression(
        n_samples=250,
        n_features=20,
        n_informative=10,
        noise=15,
        random_state=42
    )
    X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(20)])
    y = pd.Series(y)

    # Add multicollinearity
    X["dup1"] = X["Feature_0"] + np.random.normal(0, 0.1, len(X))
    X["dup2"] = X["Feature_1"] + np.random.normal(0, 0.1, len(X))

else:
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

# ------------------ SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ MODEL FUNCTION ------------------
def get_model(name, alpha):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Ridge":
        return Ridge(alpha=alpha)
    else:
        return Lasso(alpha=alpha, max_iter=10000)

def evaluate(model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return (
        mean_absolute_error(y_test, preds),
        np.sqrt(mean_squared_error(y_test, preds)),
        r2_score(y_test, preds),
        model.coef_
    )

# ------------------ SINGLE MODEL ------------------
model = get_model(model_option, alpha)
mae, rmse, r2, coefs = evaluate(model)

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)

col1.metric("📉 MAE", f"{mae:.2f}")
col2.metric("📊 RMSE", f"{rmse:.2f}")
col3.metric("🎯 R² Score", f"{r2:.3f}")

# ------------------ COEFFICIENT BAR ------------------
st.subheader("📊 Feature Importance")

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": coefs
}).sort_values(by="Coefficient")

fig = px.bar(
    coef_df,
    x="Coefficient",
    y="Feature",
    orientation='h',
    color="Coefficient",
    color_continuous_scale="RdBu"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------ COEFFICIENT PATH ------------------
st.subheader("📈 Coefficient Path")

alphas = np.logspace(-3, 2, 50)
coefs_path = []

for a in alphas:
    m = get_model(model_option, a)
    m.fit(X_train, y_train)
    coefs_path.append(m.coef_)

coefs_path = np.array(coefs_path)

fig2 = go.Figure()

for i in range(coefs_path.shape[1]):
    fig2.add_trace(go.Scatter(
        x=alphas,
        y=coefs_path[:, i],
        mode='lines',
        line=dict(width=1),
        showlegend=False
    ))

fig2.update_layout(
    xaxis_type="log",
    title="Coefficient Shrinkage",
    xaxis_title="Alpha",
    yaxis_title="Coefficient Value"
)

st.plotly_chart(fig2, use_container_width=True)

# ------------------ MODEL COMPARISON ------------------
if show_all_models:
    st.subheader("⚖️ Model Comparison")

    results = []

    for m_name in ["Linear Regression", "Ridge", "Lasso"]:
        m = get_model(m_name, alpha)
        mae, rmse, r2, _ = evaluate(m)
        results.append([m_name, mae, rmse, r2])

    df_results = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])

    fig3 = px.bar(
        df_results,
        x="Model",
        y=["MAE", "RMSE", "R2"],
        barmode="group",
        title="Performance Comparison"
    )

    st.plotly_chart(fig3, use_container_width=True)

# ------------------ LASSO CV ------------------
st.subheader("🎯 LassoCV Optimization")

lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)

best_alpha = lasso_cv.alpha_

st.success(f"Optimal Alpha: {best_alpha:.4f}")

lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train, y_train)

coef_series = pd.Series(lasso_best.coef_, index=X.columns)
removed = coef_series[coef_series == 0].index.tolist()

st.write("❌ Eliminated Features:")
st.write(removed if removed else "None")

# ------------------ CORRELATION HEATMAP ------------------
st.subheader("🔥 Correlation Heatmap")

corr = pd.DataFrame(X).corr()

fig4 = px.imshow(
    corr,
    color_continuous_scale="RdBu",
    title="Feature Correlation"
)

st.plotly_chart(fig4, use_container_width=True)

# ------------------ DOWNLOAD ------------------
st.subheader("📥 Download Results")

csv = df_results.to_csv(index=False).encode()
st.download_button(
    "Download Results CSV",
    csv,
    "results.csv",
    "text/csv"
)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("💡 Built for Machine Learning Regularization Lab | Premium UI")