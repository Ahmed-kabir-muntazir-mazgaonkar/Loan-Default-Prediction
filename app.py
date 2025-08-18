import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logging import warning

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Load Model and Scaler (With Caching)
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("loan_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, scaler = load_model()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("💰 Loan Default Predictor")
    st.markdown("""
    This app predicts whether a loan applicant is likely to default based on:
    - **Income**
    - **Loan Amount**
    - **Employment Status**
    - **Loan-to-Income Ratio**
    """)
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Enter applicant details")
    st.markdown("2. Click 'Predict'")
    st.markdown("3. View results and analysis")
    st.divider()
    st.markdown("Built with ❤️ using Streamlit")

# ---------------------------
# Main Title
# ---------------------------
st.title("Loan Default Prediction Dashboard")

# Two-column layout
col1, col2 = st.columns([1, 2])

# ---------------------------
# Input Form
# ---------------------------
with col1:
    with st.form("prediction_form"):
        st.subheader("Applicant Details")

        income = st.number_input(
            "Monthly Income (₹)",
            min_value=1000,  # More reasonable minimum
            max_value=200000,
            value=15000,    # More realistic default
            step=100,
            help="Gross monthly income in rupees"
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=1000,
            max_value=5000000,
            value=100000,
            step=1000,
            help="Requested loan amount in rupees"
        )

        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Unemployed"],
            index=0,
            help="Current employment status"
        )

        submitted = st.form_submit_button("Predict Risk")

# ---------------------------
# Prediction Section
# ---------------------------
with col2:
    if submitted:
        # Validate inputs
        if income <= 0 or loan_amount <= 0:
            st.error("Income and loan amount must be positive values")
            st.stop()

        # Encode employment status
        emp = 1 if employment_status == "Employed" else 0

        # Calculate loan-to-income ratio with safeguard
        loan_to_income_ratio = loan_amount / max(income, 1)  # Avoid division by zero

        # Prepare input data
        try:
            data = np.array([[income, loan_amount, emp, loan_to_income_ratio]])
            data_scaled = scaler.transform(data)
        except Exception as e:
            st.error(f"Error processing input data: {str(e)}")
            st.stop()

        # Make prediction
        try:
            prediction = model.predict(data_scaled)[0]
            proba = model.predict_proba(data_scaled)[0]
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.stop()

        # Display results
        st.subheader("Prediction Results")

        if prediction == 1:
            st.error(f"⚠️ High Default Risk (Probability: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ Low Default Risk (Probability: {proba[0]*100:.1f}%)")

        # Business rule warnings
        if loan_amount > income * 50:
            st.warning("⚠️ Loan amount is extremely high compared to income. "
                      "Real-world risk is likely HIGH even if model shows low risk.")

        if employment_status == "Unemployed" and prediction == 0:
            st.warning("⚠️ Note: Unemployed applicants usually carry higher risk in real scenarios.")

        # Probability visualization
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Default Risk'], [proba[1]], color='#ff6b6b' if prediction == 1 else '#51cf66')
        ax.barh(['Default Risk'], [proba[0]], left=[proba[1]], color='#51cf66' if prediction == 0 else '#ff6b6b')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0, f"{proba[1]*100:.1f}% risk",
                ha='center', va='center', color='white', fontsize=12)
        st.pyplot(fig)

        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Key Decision Factors")
            features = ['Income', 'Loan Amount', 'Employment Status', 'Loan-to-Income Ratio']
            importances = model.feature_importances_

            fig2, ax2 = plt.subplots()
            sns.barplot(x=importances, y=features, palette='viridis')
            ax2.set_title('Feature Importance')
            ax2.set_xlabel('Importance Score')
            st.pyplot(fig2)

# ---------------------------
# Additional Information Sections
# ---------------------------
with st.expander("📊 Model Performance Metrics"):
    st.subheader("Model Evaluation")
    st.markdown("""
    **Performance on Test Data:**  
    - Accuracy: 69%  
    - Precision (Repaid / Class 0): 0.76  
    - Recall (Repaid / Class 0): 0.69  
    - F1 Score (Repaid / Class 0): 0.72  

    - Precision (Default / Class 1): 0.61  
    - Recall (Default / Class 1): 0.69  
    - F1 Score (Default / Class 1): 0.65  
    """)

    cm = np.array([[81, 36],
                   [26, 57]])
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Repaid', 'Default'],
                yticklabels=['Repaid', 'Default'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix')
    st.pyplot(fig3)

with st.expander("ℹ️ About This App"):
    st.markdown("""
    ### How It Works
    This application uses a Random Forest machine learning model trained on historical loan data to predict:
    - The probability of loan default
    - Key factors influencing the decision

    ### Data Used
    The model was trained on a dataset containing:
    - Historical loan applications
    - Balanced representation of default/non-default cases
    - Features including income, loan amount, employment status, and loan-to-income ratio

    ### Limitations
    - Predictions are based solely on the provided financial information
    - Does not consider credit history or other potential factors
    - Accuracy may vary with extreme values
    """)

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
    .st-b7 { background-color: #f0f2f6; }
    .st-bb { background-color: #ffffff; }
    .css-18e3th9 {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stAlert { margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)
