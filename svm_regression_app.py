import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config(
    page_title="SVM Regression App",
    page_icon="🩸",
    layout="wide"
)

# Custom Theme CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e1b4b, #311042, #111827);
}
* {
    color: white !important;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4 {
    color: white !important;
}
[data-testid="stSidebar"]{
    background: #111827;
}
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
}
.stDataFrame {
    background-color: white !important;
    color: black !important;
}
.stDataFrame table {
    color: black !important;
}
.stButton > button {
    background: #a855f7;
    color: white !important;
    border-radius: 10px;
    padding: 8px 18px;
    border: none;
}
.stButton > button:hover {
    background: #7e22ce;
}
.main-title{
    background: linear-gradient(90deg, #a855f7, #ec4899);
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
<h1>SVM Regression (SVR)</h1>
<h3>Diabetes Progression Prediction System</h3>
</div>
""", unsafe_allow_html=True)

# Load Model & Scaler
model = pickle.load(open("model_svr.pkl", "rb"))
scaler = pickle.load(open("scaler_svr.pkl", "rb"))

# Load Dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["Progression"] = diabetes.target

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "EDA", "Model Evaluation", "Prediction"]
)

if page == "Dashboard":
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Features", len(df.columns) - 1)
    c4.metric("Target", "Progression Value")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    tab1, tab2 = st.tabs(["Distribution", "Correlation"])

    with tab1:
        feature = st.selectbox("Select Feature", df.columns[:-1])
        fig, ax = plt.subplots()
        ax.hist(df[feature], bins=25, color="#a855f7", edgecolor="black")
        ax.set_title(feature)
        st.pyplot(fig)

    with tab2:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="plasma")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar(im)
        st.pyplot(fig)

elif page == "Model Evaluation":
    X = df.drop("Progression", axis=1)
    y = df["Progression"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test_scaled = scaler.transform(X_test)
    pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", round(mae, 3))
    c2.metric("MSE", round(mse, 3))
    c3.metric("RMSE", round(rmse, 3))
    c4.metric("R² Score", round(r2, 3))

    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, color="#a855f7", alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

elif page == "Prediction":
    st.subheader("Enter Patient Measurements (Normalized Scaled Inputs)")
    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age (scaled)", -0.15, 0.15, 0.0)
        sex = st.selectbox("Sex (scaled value)", [0.0507, -0.0446])
        bmi = st.slider("BMI (scaled)", -0.1, 0.17, 0.0)
        bp = st.slider("Blood Pressure (scaled)", -0.11, 0.13, 0.0)
        s1 = st.slider("Blood Serum 1 (s1)", -0.13, 0.15, 0.0)

    with c2:
        s2 = st.slider("Blood Serum 2 (s2)", -0.12, 0.20, 0.0)
        s3 = st.slider("Blood Serum 3 (s3)", -0.10, 0.16, 0.0)
        s4 = st.slider("Blood Serum 4 (s4)", -0.08, 0.19, 0.0)
        s5 = st.slider("Blood Serum 5 (s5)", -0.13, 0.13, 0.0)
        s6 = st.slider("Blood Serum 6 (s6)", -0.14, 0.14, 0.0)

    if st.button("Predict Progression"):
        data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)[0]

        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #a855f7, #ec4899);
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: white;">
            Predicted Disease Progression Index:<br>
            {prediction:.2f}
        </div>
        """, unsafe_allow_html=True)