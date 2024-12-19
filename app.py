import streamlit as st
import pandas as pd
from src.config import Config
from src.utils import *
from src.predict import prediction
from src.model_store import initialize_models

# Initialize models only once using session state
if "models_initialized" not in st.session_state:
    initialize_models()
    st.session_state.models_initialized = True

st.set_page_config(
    page_title="Green Finance Risk Prediction",
    page_icon="🌱"
)

# Cache data loading to avoid reloading on interaction
@st.cache_data
def load_data():
    print("Dataset loaded successfully...")
    return pd.read_csv(Config.DATA_PATH)

def display_sidebar(company_details):
    st.sidebar.header("Company Details")
    markdown_content = sidebar_markdown(company_details)
    st.sidebar.markdown(markdown_content, unsafe_allow_html=True)

def main():    
    # Load data only once
    df = load_data()

    # Sidebar: Select Company
    companies = df["company_name"].unique()
    selected_company = st.sidebar.selectbox("Select a Company", companies)

    # Fetch company details based on selection
    company_details = df[df['company_name'] == selected_company].iloc[0]

    display_sidebar(company_details)

    st.title("Green Finance Risk Prediction 🌱")

    # Create two-column layout for inputs
    col1, col2 = st.columns(2)

    with col1:
        impact_area_community = st.number_input("Impact Area - Community", value=25.0, min_value=0.0, max_value=100.0, step=0.1)
        impact_area_environment = st.number_input("Impact Area - Environment", value=30.0, min_value=0.0, max_value=100.0, step=0.1)

    with col2:
        impact_area_customers = st.number_input("Impact Area - Customers", value=40.0, min_value=0.0, max_value=100.0, step=0.1)
        impact_area_governance = st.number_input("Impact Area - Governance", value=50.0, min_value=0.0, max_value=100.0, step=0.1)

    certification_cycle = st.slider("Certification Cycle", min_value=0, max_value=10, step=1)
    
    # Predict button in a separate row and center it
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("Predict", type="primary", use_container_width=True)

    # Handle button click event
    if predict_button:
        input_raw_data = f"""
Company Name: {company_details['company_name']}
Place: {company_details['country']}  
Industry Category: {company_details['industry_category']}  
Sector: {company_details['sector']}  
Industry: {company_details['industry']}  
Products and Services: {company_details['products_and_services']}  
Description: {company_details['description']}
Impact Area Community Value: {impact_area_community}
Impact Area Environment Value: {impact_area_environment}
Impact Area Customers Value: {impact_area_customers}
Impact Area Governance Value: {impact_area_governance}
Certification Cycle: {certification_cycle}
        """
        response = prediction(impact_area_community, impact_area_environment, impact_area_customers, impact_area_governance, certification_cycle, input_raw_data)
        st.info(response)


if __name__ == "__main__":
    main()