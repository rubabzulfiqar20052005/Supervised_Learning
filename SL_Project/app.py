import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ------------------------
# LOAD MODEL
# ------------------------
model = pickle.load(open("best_model.pkl", "rb"))

# ------------------------
# CUSTOM CSS (🔥 UI BOOST)
# ------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# TITLE
# ------------------------
st.title("🏠 Advanced House Price Prediction System")
st.markdown("### Predict house prices using Machine Learning")

# ------------------------
# SIDEBAR INPUTS
# ------------------------
st.sidebar.header("🔧 Input Features")

rm = st.sidebar.slider("Rooms (RM)", 1.0, 10.0, 5.0)
lstat = st.sidebar.slider("LSTAT (%)", 1.0, 40.0, 10.0)
ptratio = st.sidebar.slider("PTRATIO", 10.0, 25.0, 18.0)
tax = st.sidebar.slider("TAX", 100, 800, 300)

# ------------------------
# MAIN LAYOUT
# ------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Summary")

    data = pd.DataFrame({
        "RM": [rm],
        "LSTAT": [lstat],
        "PTRATIO": [ptratio],
        "TAX": [tax]
    })

    st.dataframe(data, use_container_width=True)

with col2:
    st.subheader("📈 Feature Visualization")

    fig, ax = plt.subplots()
    ax.bar(["RM", "LSTAT", "PTRATIO", "TAX"], [rm, lstat, ptratio, tax])
    st.pyplot(fig)

# ------------------------
# PREDICTION
# ------------------------
st.markdown("---")

if st.button("🚀 Predict Price"):

    # Dummy full feature vector (Boston dataset expects 13 features)
    features = np.zeros(13)

    features[5] = rm        # RM index
    features[12] = lstat    # LSTAT index
    features[10] = ptratio  # PTRATIO
    features[9] = tax       # TAX

    prediction = model.predict([features])[0]

    # ------------------------
    # OUTPUT SECTION
    # ------------------------
    st.subheader("💰 Prediction Result")

    col3, col4, col5 = st.columns(3)

    col3.metric("Estimated Price", f"${prediction*1000:,.0f}")
    col4.metric("Rooms", rm)
    col5.metric("LSTAT", lstat)

    # ------------------------
    # INTERPRETATION
    # ------------------------
    st.markdown("### 🧠 Model Insight")

    if prediction > 25:
        st.success("🏡 High-value property")
    elif prediction > 15:
        st.info("🏠 متوسط (medium range house)")
    else:
        st.warning("🏚️ Low-value property")

# ------------------------
# FOOTER
# ------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | ML Project")