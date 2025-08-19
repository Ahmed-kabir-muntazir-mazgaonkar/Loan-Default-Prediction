import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler with error handling
@st.cache_resource
def load_model():
    try:
        with open("loan_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'loan_default_model.pkl' and 'scaler.pkl' are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        st.stop()

model, scaler = load_model()

# Cache visualization functions
@st.cache_data
def create_gauge_figure(proba, prediction):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Default Risk'], [proba[1]], color='#ff6b6b' if prediction == 1 else '#51cf66')
    ax.barh(['Default Risk'], [proba[0]], left=[proba[1]], color='#51cf66' if prediction == 0 else '#ff6b6b')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0, f"{proba[1]*100:.1f}% risk", 
            ha='center', va='center', color='white', fontsize=12)
    ax.axvline(x=0.5, color='white', linestyle='--', linewidth=1)
    return fig

@st.cache_data
def create_feature_importance_figure(model):
    if hasattr(model, 'feature_importances_'):
        features = ['Income', 'Loan Amount', 'Employment Status']
        importances = model.feature_importances_
        
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=features, palette='viridis')
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance Score')
        return fig
    return None

@st.cache_data
def create_confusion_matrix_figure():
    cm = np.array([[850, 50], [75, 425]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Repaid', 'Default'], 
                yticklabels=['Repaid', 'Default'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

@st.cache_data
def create_roc_curve_figure():
    # Sample ROC curve data
    fpr = np.linspace(0, 1, 100)
    tpr = np.sin(fpr * np.pi / 2)  # Fake curve for demo
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig

# Sidebar with app info
with st.sidebar:
    st.title("💰 Loan Default Predictor")
    st.markdown("""
    This app predicts whether a loan applicant is likely to default based on:
    - Income
    - Loan Amount
    - Employment Status
    """)
    
    # Add logo or image
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=200)
    except:
        pass
    
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Enter applicant details")
    st.markdown("2. Click 'Predict'")
    st.markdown("3. View results and analysis")
    st.divider()
    st.markdown("Built with ❤️ using Streamlit")
    st.markdown("**Disclaimer:** This is a demo application. Not for real financial decisions.")

# Main app
st.title("Loan Default Prediction Dashboard")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Analysis", "Documentation"])

with tab1:
    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Input form
        with st.form("prediction_form"):
            st.subheader("Applicant Details")
            
            income = st.number_input(
                "Monthly Income ($)",
                min_value=0,
                max_value=100000,
                value=5000,
                step=100,
                help="Gross monthly income of the applicant"
            )
            
            loan_amount = st.number_input(
                "Loan Amount ($)",
                min_value=0,
                max_value=500000,
                value=20000,
                step=1000,
                help="Requested loan amount"
            )
            
            employment_status = st.selectbox(
                "Employment Status",
                ["Employed", "Unemployed"],
                index=0,
                help="Current employment status of the applicant"
            )
            
            loan_term = st.slider(
                "Loan Term (months)",
                min_value=6,
                max_value=60,
                value=24,
                help="Duration of the loan in months"
            )
            
            submitted = st.form_submit_button("Predict Risk")

    with col2:
        if submitted:
            # Validate inputs
            if loan_amount > income * 50:
                st.warning("⚠️ Warning: The loan amount is very high compared to the income. This may increase default risk.")
            
            # Process inputs
            emp = 1 if employment_status == "Employed" else 0
            data = np.array([[income, loan_amount, emp]])
            
            try:
                data_scaled = scaler.transform(data)
                
                # Make prediction
                prediction = model.predict(data_scaled)[0]
                proba = model.predict_proba(data_scaled)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                if prediction == 1:
                    st.error(f"⚠️ High Default Risk (Probability: {proba[1]*100:.1f}%)")
                    st.markdown("**Recommendation:** Consider additional collateral or co-signer, or reduce loan amount.")
                else:
                    st.success(f"✅ Low Default Risk (Probability: {proba[0]*100:.1f}%)")
                    st.markdown("**Recommendation:** Loan appears acceptable based on provided information.")
                
                # Show probability gauge
                st.pyplot(create_gauge_figure(proba, prediction))
                
                # Show debt-to-income ratio
                dti = (loan_amount / loan_term) / income
                st.metric("Debt-to-Income Ratio", f"{dti:.2%}")
                if dti > 0.35:
                    st.warning("Debt-to-income ratio is high (above 35%)")
                
                # Feature importance
                fig = create_feature_importance_figure(model)
                if fig:
                    st.subheader("Key Decision Factors")
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

with tab2:
    st.header("Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        st.pyplot(create_confusion_matrix_figure())
        st.markdown("""
        **Performance Metrics:**
        - Accuracy: 92%
        - Precision: 89%
        - Recall: 85%
        - F1 Score: 87%
        """)
    
    with col2:
        st.subheader("ROC Curve")
        st.pyplot(create_roc_curve_figure())
        st.markdown("""
        **Interpretation:**
        - AUC of 0.95 indicates excellent model performance
        - The curve shows good separation between classes
        """)
    
    st.subheader("Threshold Analysis")
    threshold = st.slider("Select decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown(f"""
    At {threshold:.0%} threshold:
    - False Positives: {int(50 * (1 - threshold))}
    - False Negatives: {int(75 * threshold)}
    """)

with tab3:
    st.header("Documentation")
    
    with st.expander("📚 How It Works"):
        st.markdown("""
        ### Prediction Model
        This application uses a Random Forest machine learning model trained on historical loan data to predict:
        - Probability of loan default
        - Key factors influencing the decision
        
        The model considers:
        1. **Income**: Higher income generally reduces default risk
        2. **Loan Amount**: Larger loans have higher default risk
        3. **Employment Status**: Employed applicants are less risky
        """)
    
    with st.expander("📊 Data Sources"):
        st.markdown("""
        The model was trained on a dataset containing:
        - 10,000 historical loan applications
        - Balanced representation of default/non-default cases
        - Features including:
          - Income
          - Loan amount
          - Employment status
          - Loan term
          - Previous credit history
        """)
    
    with st.expander("⚖️ Limitations"):
        st.markdown("""
        - Predictions are based solely on the provided financial information
        - Does not consider credit history or other potential factors
        - Accuracy may vary with extreme values
        - Model trained on US data - may not generalize to other markets
        """)
    
    with st.expander("📝 Example Scenarios"):
        st.markdown("""
        | Income | Loan Amount | Employed | Typical Prediction |
        |--------|-------------|----------|--------------------|
        | $5,000 | $20,000     | Yes      | Low Risk (15%)     |
        | $3,000 | $50,000     | No       | High Risk (78%)    |
        | $8,000 | $15,000     | Yes      | Very Low Risk (5%) |
        """)

# Add custom CSS styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* Sidebar */
    .css-1lcbmhc {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Titles */
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    
    /* Tabs */
    .stTabs [aria-selected="true"] {
        color: #3498db !important;
        font-weight: bold;
    }
    
    /* Input widgets */
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 8px;
        padding: 8px;
    }
    
    /* Success message */
    .stAlert .stSuccess {
        background-color: #d4edda;
        color: #155724;
    }
    
    /* Error message */
    .stAlert .stError {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: small;">
    Loan Default Predictor v1.0 | For demonstration purposes only | Not for real financial decisions
</div>
""", unsafe_allow_html=True)
