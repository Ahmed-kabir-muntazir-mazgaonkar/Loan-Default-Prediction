import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
# Load Model and Scaler
# ---------------------------
@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:   # 🔹 Notebook me bhi yahi naam rakho
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
    This app predicts whether a loan applicant is likely to default based on:
    - **Income**
    - **Loan Amount**
    - **Employment Status**
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
            min_value=0,
            max_value=10000000,
            value=5000,
            step=1000
        )

        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=0,
            max_value=50000000,
            value=20000,
            step=1000
        )

        employment_status = st.selectbox(
            "Employment Status",
            ["Employed", "Unemployed"],
            index=0
        )

        submitted = st.form_submit_button("Predict Risk")

# ---------------------------
# Prediction Section
# ---------------------------
with col2:
    if submitted:
        # Encode employment status (same as training)
        emp = 1 if employment_status == "Employed" else 0

        # Match feature order from training: ['income', 'loan_amount', 'employment_status']
        data = np.array([[income, loan_amount, emp]])
        data_scaled = scaler.transform(data)

        # Prediction
        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]

        # Results
        st.subheader("Prediction Results")

        if prediction == 1:
            st.error("⚠️ High Default Risk (Probability: {:.1f}%)".format(proba[1]*100))
        else:
            st.success("✅ Low Default Risk (Probability: {:.1f}%)".format(proba[0]*100))

        # Business Rule Adjustment (Optional)
        if employment_status == "Unemployed" and prediction == 0:
            st.warning("⚠️ Note: Unemployed applicants usually carry higher risk in real scenarios.")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Default Risk'], [proba[1]], color='#ff6b6b' if prediction == 1 else '#51cf66')
        ax.barh(['Default Risk'], [proba[0]], left=[proba[1]], color='#51cf66' if prediction == 0 else '#ff6b6b')
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0, f"{proba[1]*100:.1f}% risk",
                ha='center', va='center', color='white', fontsize=12)
        st.pyplot(fig)

        # Feature importance (Random Forest / Tree models)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Key Decision Factors")
            features = ['Income', 'Loan Amount', 'Employment Status']
            importances = model.feature_importances_

            fig2, ax2 = plt.subplots()
            sns.barplot(x=importances, y=features, palette='viridis')
            ax2.set_title('Feature Importance')
            ax2.set_xlabel('Importance Score')
            st.pyplot(fig2)

# ---------------------------
# Expanders for Metrics & Info
# ---------------------------
with st.expander("📊 Model Performance Metrics"):
    st.subheader("Model Evaluation")
    st.markdown("""
    **Performance on Test Data:**  
    (Replace with your Notebook results)
    - Accuracy: 92%  
    - Precision: 89%  
    - Recall: 85%  
    - F1 Score: 87%  
    """)
    cm = np.array([[850, 50], [75, 425]])  # Replace with your confusion matrix
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
    - Features including income, loan amount, and employment status

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
</style>
""", unsafe_allow_html=True)
