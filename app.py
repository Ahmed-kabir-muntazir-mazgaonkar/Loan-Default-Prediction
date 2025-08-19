import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Load model and scaler
# ---------------------------
@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("💰 Loan Default Predictor")
    st.markdown("""
    Predict loan default risk based on:
    - Monthly Income
    - Loan Amount
    - Employment Status
    """)
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Enter applicant details")
    st.markdown("2. Enter loan period (months) & interest rate (%)")
    st.markdown("3. Click 'Predict Risk'")
    st.divider()
    st.markdown("Built with ❤️ using Streamlit")

# ---------------------------
# Main app
# ---------------------------
st.title("Loan Default Prediction Dashboard")

col1, col2 = st.columns([1, 2])

with col1:
    with st.form("prediction_form"):
        st.subheader("Applicant Details")
        
        income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=5000, step=100)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=500000, value=20000, step=1000)
        employment_status = st.selectbox("Employment Status", ["Employed", "Unemployed"])
        
        # Extra inputs for repayment calculation
        loan_period = st.number_input("Loan Period (Months)", min_value=1, max_value=360, value=12, step=1)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        
        submitted = st.form_submit_button("Predict Risk")

with col2:
    if submitted:
        # Prepare data for model
        emp = 1 if employment_status == "Employed" else 0
        data = np.array([[income, loan_amount, emp]])
        data_scaled = scaler.transform(data)
        
        # Prediction
        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]
        
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"⚠️ High Default Risk (Probability: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Low Default Risk (Probability: {proba[0]*100:.1f}%)")
        
        # Calculate final repayable amount
        final_amount = loan_amount * (1 + (interest_rate/100) * (loan_period/12))
        st.subheader("💵 Loan Repayment Details")
        st.write(f"Loan Period: {loan_period} months")
        st.write(f"Interest Rate: {interest_rate}% per year")
        st.write(f"**Total Amount to Repay: ${final_amount:,.2f}**")
        
        # Optional: probability bar
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(['Default Risk'], [proba[1]], color='#ff6b6b' if prediction == 1 else '#51cf66')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0, f"{proba[1]*100:.1f}% risk", ha='center', va='center', color='white', fontsize=12)
        st.pyplot(fig)
