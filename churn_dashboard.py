import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------
# Page Config
# -----------------------
st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

# -----------------------
# Load Files
# -----------------------
model = joblib.load("Customer_Churn_model.pkl")
label_encoders = joblib.load("encoded_columns.pkl")
model_columns = joblib.load("training_columns.pkl")

# -----------------------
# Company Branding
# -----------------------
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        🚀 Telecom Customer Churn Analytics
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# SIDEBAR INPUT SECTION
# -----------------------
st.sidebar.header("📋 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.number_input("Tenure (Months)", min_value=0)
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0)

predict_button = st.sidebar.button("🔮 Predict Churn")

# -----------------------
# MAIN AREA
# -----------------------
if predict_button:

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "InternetService": [InternetService],
        "Contract": [Contract],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    # Encode categorical columns
    for col in input_data.select_dtypes(include="object").columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Align columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Convert to float
    input_data = input_data.astype(float)

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns(2)

    # -----------------------
    # Prediction Result
    # -----------------------
    with col1:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ Customer Likely to Churn")
        else:
            st.success(f"✅ Customer Likely to Stay")

        st.metric("Churn Probability", f"{probability:.2%}")

        # Probability Bar Chart
        fig, ax = plt.subplots()
        ax.bar(["Stay", "Churn"], [1-probability, probability])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability")
        st.pyplot(fig)

    # -----------------------
    # Feature Importance Chart
    # -----------------------
    with col2:
        st.subheader("Feature Importance")

        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": model_columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(10)

        fig2, ax2 = plt.subplots()
        ax2.barh(feature_importance_df["Feature"], feature_importance_df["Importance"])
        ax2.invert_yaxis()
        ax2.set_title("Top 10 Important Features")
        st.pyplot(fig2)
