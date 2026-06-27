import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# --------------------
# Load Model and Data
# --------------------
@st.cache_resource
def load_model():
    with open("kmeans.pkl", "rb") as f:
        model = pickle.load(f)
    return model

st.set_page_config(page_title="KMeans Clustering App", layout="wide")
st.title("📊 KMeans Clustering on Age & Income")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV with Age and Income)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # Load trained model
    model = load_model()

    # Scale the data (same as training)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["Age", "Income"]])

    # Predict clusters
    clusters = model.predict(df_scaled)
    df["Cluster"] = clusters

    # --------------------
    # Data Insights
    # --------------------
    st.subheader("Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Cluster Count")
        st.bar_chart(df["Cluster"].value_counts())

    with col2:
        st.write("### Summary Statistics")
        st.write(df.groupby("Cluster")[["Age", "Income"]].mean())

    # --------------------
    # Scatter Plot
    # --------------------
    st.subheader("Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=df["Age"], y=df["Income"], hue=df["Cluster"],
        palette="Set2", s=80, ax=ax
    )
    centers = scaler.inverse_transform(model.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Centers")
    ax.legend()
    st.pyplot(fig)

    # --------------------
    # Inertia Value
    # --------------------
    st.subheader("Model Inertia")
    st.write(f"🧮 Inertia Value: **{model.inertia_:.2f}**")

    # --------------------
    # Prediction Module
    # --------------------
    st.subheader("🔮 Predict Cluster for New Data")

    age = st.number_input("Enter Age", min_value=0, max_value=100, value=30)
    income = st.number_input("Enter Income", min_value=0, value=150000)

    if st.button("Predict Cluster"):
        new_point_scaled = scaler.transform([[age, income]])
        cluster_pred = model.predict(new_point_scaled)[0]

        st.success(f"✅ The input (Age={age}, Income={income}) belongs to **Cluster {cluster_pred}**")

        # Plot with new point
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=df["Age"], y=df["Income"], hue=df["Cluster"],
            palette="Set2", s=80, ax=ax2
        )
        ax2.scatter(age, income, c="black", s=200, marker="*", label="New Point")
        ax2.legend()
        st.pyplot(fig2)

else:
    st.warning("⚠️ Please upload a CSV file with Age and Income columns to continue.")
