# AI-Driven Customer Personality Analysis and Behavioral Segmentation

This project is a Streamlit-based machine learning web application developed as part of an M.Tech Data Science thesis. The system performs customer behavioral segmentation and predicts the customer's next-month behavioral segment using machine learning and deep learning techniques.

The application allows users to enter customer behavioral details such as recency, frequency, monetary value, total quantity purchased, unique products, unique orders, and average unit price. Based on these inputs, the system predicts the customer's current segment, forecasts the next-month segment, and provides marketing recommendations for personalized digital marketing decisions.

## Live App

The deployed Streamlit application is available here:

https://customer-segmentation-app-ai.streamlit.app/

## Project Objective

The main objective of this project is to help businesses understand customer behavior and support personalized digital marketing strategies. The system identifies different customer segments, predicts possible future movement between segments, and suggests marketing actions for retention, reactivation, cross-selling, and customer engagement.

## Key Features

* Customer behavioral segmentation
* Current customer segment prediction
* Next-month customer segment prediction
* Business-friendly customer segment names
* Cluster-specific marketing recommendations
* Interactive Streamlit web interface
* Real-time prediction using saved trained models

## Models and Techniques Used

* Autoencoder for deep feature extraction
* KMeans clustering for customer segmentation
* Markov transition analysis for understanding customer movement
* Tuned XGBoost for next-month segment prediction
* Streamlit for web application deployment

## Input Features

The application uses the following customer-level behavioral features:

* Recency: Days since last purchase
* Frequency: Number of transactions or orders
* Monetary: Total customer spending
* TotalQuantity: Total quantity purchased
* UniqueProducts: Number of unique products purchased
* UniqueOrders: Number of unique orders
* AvgUnitPrice: Average unit price of purchased items

## Customer Segments

The system classifies customers into the following behavioral segments:

* Active Regular Customers
* High-Value Churn-Risk Customers
* Premium At-Risk Customers
* Frequent Low-Spend Explorers
* Loyal Regular Customers

## Project Workflow

1. Data preprocessing and cleaning
2. Feature engineering using customer behavioral variables
3. Autoencoder-based feature learning
4. KMeans clustering for customer segmentation
5. Markov transition analysis for customer movement understanding
6. XGBoost model training for next-month segment prediction
7. Model saving and loading
8. Streamlit app development
9. Deployment on Streamlit Community Cloud

## Project Structure

```text
customer-segmentation-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── saved_models/
    ├── encoder_phase2.keras
    ├── scaler_phase2.pkl
    ├── kmeans_phase2.pkl
    ├── tuned_xgboost_model.json
    ├── phase2_feature_columns.json
    ├── xgboost_feature_columns.json
    └── phase2_cluster_names.json
```

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Joblib
* Streamlit
* GitHub
* Streamlit Community Cloud

## How to Run Locally

Clone the repository:

```bash
git clone https://github.com/Yashika-Nigam/customer-segmentation-streamlit.git
```

Move into the project folder:

```bash
cd customer-segmentation-streamlit
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment

The application is deployed using Streamlit Community Cloud. The app loads the saved trained models from the `saved_models` folder and performs real-time prediction through the web interface.

## Academic Context

This project was developed as part of an M.Tech Data Science thesis titled:

**AI-Driven Dynamic Customer Behavioral Intelligence Framework using Deep Learning** 

The project demonstrates how machine learning and deep learning can be used to transform customer transaction data into actionable marketing intelligence.

## Author

**Yashika Nigam**
M.Tech Data Science
RGPV, Bhopal
