import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

@st.cache_data
def load_feature_ranges():
    with open("saved_models/feature_ranges.json", "r") as f:
        return json.load(f)
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

feature_ranges = load_feature_ranges()

recency = st.sidebar.slider(
    "Recency: Days since last purchase",
    min_value=int(feature_ranges["Recency"]["min"]),
    max_value=int(feature_ranges["Recency"]["max"]),
    value=int(feature_ranges["Recency"]["median"]),
    step=1
)

frequency = st.sidebar.slider(
    "Frequency: Number of orders",
    min_value=int(feature_ranges["Frequency"]["min"]),
    max_value=int(feature_ranges["Frequency"]["max"]),
    value=int(feature_ranges["Frequency"]["median"]),
    step=1
)

monetary = st.sidebar.slider(
    "Monetary: Total spending amount",
    min_value=float(feature_ranges["Monetary"]["min"]),
    max_value=float(feature_ranges["Monetary"]["max"]),
    value=float(feature_ranges["Monetary"]["median"]),
    step=100.0
)

total_quantity = st.sidebar.slider(
    "Total Quantity Purchased",
    min_value=int(feature_ranges["TotalQuantity"]["min"]),
    max_value=int(feature_ranges["TotalQuantity"]["max"]),
    value=int(feature_ranges["TotalQuantity"]["median"]),
    step=1
)

unique_products = st.sidebar.slider(
    "Unique Products Purchased",
    min_value=int(feature_ranges["UniqueProducts"]["min"]),
    max_value=int(feature_ranges["UniqueProducts"]["max"]),
    value=int(feature_ranges["UniqueProducts"]["median"]),
    step=1
)

customer_lifetime = st.sidebar.slider(
    "Customer Lifetime: Active customer duration in days",
    min_value=int(feature_ranges["CustomerLifetime"]["min"]),
    max_value=int(feature_ranges["CustomerLifetime"]["max"]),
    value=int(feature_ranges["CustomerLifetime"]["median"]),
    step=1
)

avg_order_value = monetary / (frequency + 1)

purchase_rate = frequency / (recency + 1)

quantity_per_order = total_quantity / (frequency + 1)

spend_per_product = monetary / (unique_products + 1)

purchase_frequency_variance = float(
    feature_ranges["PurchaseFrequencyVariance"]["median"]
)
unique_orders = frequency

avg_unit_price = monetary / total_quantity if total_quantity != 0 else 0

# -----------------------------
# Segment Average Profiles
# -----------------------------
segment_average_profiles = {
    0: {
        "Recency": 57.18,
        "Frequency": 69.92,
        "Monetary": 1142.51,
        "TotalQuantity": 647.14,
        "UniqueProducts": 54.49,
        "UniqueOrders": 3.14,
        "AvgUnitPrice": 3.55
    },
    1: {
        "Recency": 123.67,
        "Frequency": 20.49,
        "Monetary": 3500.02,
        "TotalQuantity": 2263.02,
        "UniqueProducts": 8.50,
        "UniqueOrders": 2.69,
        "AvgUnitPrice": 3.46
    },
    2: {
        "Recency": 102.95,
        "Frequency": 52.18,
        "Monetary": 2439.32,
        "TotalQuantity": 1518.74,
        "UniqueProducts": 25.59,
        "UniqueOrders": 3.80,
        "AvgUnitPrice": 6.89
    },
    3: {
        "Recency": 84.88,
        "Frequency": 78.94,
        "Monetary": 939.42,
        "TotalQuantity": 481.49,
        "UniqueProducts": 67.50,
        "UniqueOrders": 2.45,
        "AvgUnitPrice": 3.58
    },
    4: {
        "Recency": 67.45,
        "Frequency": 70.07,
        "Monetary": 1584.47,
        "TotalQuantity": 899.46,
        "UniqueProducts": 45.04,
        "UniqueOrders": 3.97,
        "AvgUnitPrice": 4.77
    }
}

unique_orders = frequency
avg_unit_price = monetary / total_quantity if total_quantity != 0 else 0
# -----------------------------
# Prediction
# -----------------------------

if st.sidebar.button("Predict Customer Segment"):
    input_df = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "TotalQuantity": total_quantity,
        "UniqueProducts": unique_products,
        "CustomerLifetime": customer_lifetime,
        "AvgOrderValue": avg_order_value,
        "PurchaseRate": purchase_rate,
        "QuantityPerOrder": quantity_per_order,
        "SpendPerProduct": spend_per_product,
        "PurchaseFrequencyVariance": purchase_frequency_variance,
        "UniqueOrders": unique_orders,
        "AvgUnitPrice": avg_unit_price
    }])
    
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

    # -----------------------------
    # Confidence / Certainty Scores
    # -----------------------------
    cluster_distances = kmeans.transform(input_encoded)[0]
    distance_similarity = np.exp(-(cluster_distances - cluster_distances.min()))
    cluster_certainty_scores = distance_similarity / distance_similarity.sum()
    current_cluster_confidence = float(cluster_certainty_scores[current_cluster]) * 100

    try:
        next_cluster_probabilities = xgb_model.predict_proba(xgb_input)[0]
        next_cluster_confidence = float(np.max(next_cluster_probabilities)) * 100
    except Exception:
        next_cluster_confidence = None

    segmentation_silhouette_score = 0.5706
    xgboost_test_accuracy = 78.58

    # Get business names of clusters
    current_cluster_name = cluster_names.get(str(current_cluster), f"Cluster {current_cluster}")
    next_cluster_name = cluster_names.get(str(next_cluster), f"Cluster {next_cluster}")

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("Customer vs Segment Average")

    segment_avg = segment_average_profiles[current_cluster]

    comparison_rows = []

    for metric in phase2_feature_cols:
        customer_value = float(input_df[metric].iloc[0])
        segment_value = float(segment_avg[metric])
        difference = customer_value - segment_value

        comparison_rows.append({
            "Metric": metric,
            "Customer Value": round(customer_value, 2),
            "Segment Average": round(segment_value, 2),
            "Difference": round(difference, 2)
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(comparison_df, width="stretch", hide_index=True)

    st.caption(
        "This table compares the entered customer values with the average behavior of the predicted customer segment."
    )

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        with st.container(border=True):
            st.markdown("### Current Segment")
            st.markdown(f"**{current_cluster_name}**")
            st.caption(f"Cluster Number: {current_cluster}")
            st.write("Model Used: Autoencoder + KMeans")

            st.metric("Segment Certainty", f"{current_cluster_confidence:.2f}%")
            st.progress(current_cluster_confidence / 100)

            st.caption(f"Segmentation Silhouette Score: {segmentation_silhouette_score}")

    with result_col2:
        with st.container(border=True):
            st.markdown("### Predicted Next-Month Segment")
            st.markdown(f"**{next_cluster_name}**")
            st.caption(f"Cluster Number: {next_cluster}")
            st.write("Model Used: Tuned XGBoost")

            if next_cluster_confidence is not None:
                st.metric("Prediction Confidence", f"{next_cluster_confidence:.2f}%")
                st.progress(next_cluster_confidence / 100)
            else:
                st.warning("Prediction confidence is not available.")

            st.caption(f"XGBoost Test Accuracy: {xgboost_test_accuracy:.2f}%")

    st.info(
        "Note: Current segment certainty is distance-based because KMeans is an unsupervised model. "
        "Next-month prediction confidence is based on XGBoost probability output."
    )
    st.subheader("Marketing Recommendation")
    st.success(get_recommendation(current_cluster, next_cluster))

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed as part of M.Tech Data Science Thesis Project.")