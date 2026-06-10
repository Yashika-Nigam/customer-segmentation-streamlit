import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI Customer Segmentation App",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Model Folder Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"


# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_assets():
    encoder = load_model(MODEL_DIR / "encoder_phase2.keras")

    scaler = joblib.load(MODEL_DIR / "scaler_phase2.pkl")
    kmeans = joblib.load(MODEL_DIR / "kmeans_phase2.pkl")

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "tuned_xgboost_model.json"))

    with open(MODEL_DIR / "phase2_feature_columns.json", "r") as f:
        phase2_feature_cols = json.load(f)

    with open(MODEL_DIR / "xgboost_feature_columns.json", "r") as f:
        xgb_feature_cols = json.load(f)

    with open(MODEL_DIR / "phase2_cluster_names.json", "r") as f:
        cluster_names = json.load(f)

    return encoder, scaler, kmeans, xgb_model, phase2_feature_cols, xgb_feature_cols, cluster_names


encoder, scaler, kmeans, xgb_model, phase2_feature_cols, xgb_feature_cols, cluster_names = load_assets()


# -----------------------------
# Recommendation Logic
# -----------------------------
def get_recommendation(current_cluster, next_cluster):
    cluster_actions = {
        0: (
            "This customer belongs to the Active Regular Customers segment. "
            "They purchase regularly and show stable engagement. "
            "Recommended strategy: use regular engagement campaigns, personalized product suggestions, festive offers, and small loyalty rewards."
        ),
        1: (
            "This customer belongs to the High-Value Churn-Risk Customers segment. "
            "They have high spending value but have not purchased recently. "
            "Recommended strategy: use premium win-back offers, personalized discounts, urgency-based campaigns, and direct reactivation messages."
        ),
        2: (
            "This customer belongs to the Premium At-Risk Customers segment. "
            "They purchase higher-value products but show reduced recent activity. "
            "Recommended strategy: use exclusive product recommendations, premium offers, reminder campaigns, and personalized attention."
        ),
        3: (
            "This customer belongs to the Frequent Low-Spend Explorers segment. "
            "They interact frequently and explore many products, but their spending is comparatively low. "
            "Recommended strategy: use bundle offers, cross-selling, combo discounts, and minimum-order-value coupons."
        ),
        4: (
            "This customer belongs to the Loyal Regular Customers segment. "
            "They show regular buying behavior and good order consistency. "
            "Recommended strategy: use loyalty benefits, early access offers, personalized recommendations, and retention campaigns."
        )
    }

    if current_cluster == next_cluster:
        movement_message = (
            " The model predicts that this customer will remain in the same segment next month, "
            "so the business should focus on retention and consistent engagement."
        )
    else:
        movement_message = (
            f" The model predicts that this customer may move from Cluster {current_cluster} "
            f"to Cluster {next_cluster} next month. This indicates a behavioral shift, "
            "so proactive targeted marketing is recommended."
        )

    return cluster_actions.get(current_cluster, "No specific recommendation available.") + movement_message

# -----------------------------
# App Title
# -----------------------------
st.title("📊 AI-Driven Customer Personality Analysis and Behavioral Segmentation")

st.write(
    "This application predicts the customer's current behavioral segment using "
    "**Autoencoder + KMeans**, and predicts the next-month customer cluster using "
    "**Tuned XGBoost**."
)

st.markdown("---")


# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("Enter Customer Behavioral Details")

recency = st.sidebar.slider(
    "Recency: Days since last purchase",
    min_value=0,
    max_value=365,
    value=30,
    step=1
)

frequency = st.sidebar.slider(
    "Frequency: Number of transactions/orders",
    min_value=1,
    max_value=100,
    value=5,
    step=1
)

monetary = st.sidebar.slider(
    "Monetary: Total spending amount",
    min_value=0,
    max_value=10000,
    value=1000,
    step=100
)

total_quantity = st.sidebar.slider(
    "Total Quantity Purchased",
    min_value=0,
    max_value=5000,
    value=100,
    step=10
)

unique_products = st.sidebar.slider(
    "Unique Products Purchased",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

unique_orders = st.sidebar.slider(
    "Unique Orders",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

avg_unit_price = st.sidebar.slider(
    "Average Unit Price",
    min_value=0,
    max_value=500,
    value=20,
    step=1
)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Customer Segment"):

    input_data = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "TotalQuantity": total_quantity,
        "UniqueProducts": unique_products,
        "UniqueOrders": unique_orders,
        "AvgUnitPrice": avg_unit_price
    }

    input_df = pd.DataFrame([input_data])

    # Arrange columns exactly like training
    input_df = input_df[phase2_feature_cols]

    # Same preprocessing as training
    input_log = np.log1p(input_df)
    input_scaled = scaler.transform(input_log)

    # Encode customer features
    input_encoded = encoder.predict(input_scaled, verbose=0)

    # Predict current cluster
    current_cluster = int(kmeans.predict(input_encoded)[0])

    # Prepare input for XGBoost
    xgb_input = input_df.copy()
    xgb_input["Cluster"] = current_cluster
    xgb_input = xgb_input[xgb_feature_cols]

    # Predict next-month cluster
    next_cluster = int(xgb_model.predict(xgb_input)[0])

    current_cluster_name = cluster_names.get(str(current_cluster), f"Cluster {current_cluster}")
    next_cluster_name = cluster_names.get(str(next_cluster), f"Cluster {next_cluster}")

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("Customer Input Data")
    st.dataframe(input_df, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Current Customer Segment",
            value=current_cluster_name
        )

    with col2:
        st.metric(
            label="Predicted Next-Month Segment",
            value=next_cluster_name
        )

    st.subheader("Marketing Recommendation")
    st.success(get_recommendation(current_cluster, next_cluster))

    st.subheader("Technical Summary")
    st.write({
        "Current Cluster Number": current_cluster,
        "Next-Month Cluster Number": next_cluster,
        "Current Segmentation Model": "Autoencoder + KMeans",
        "Next-Month Prediction Model": "Tuned XGBoost"
    })


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed as part of M.Tech Data Science Thesis Project.")