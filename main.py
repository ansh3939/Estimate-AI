import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="RE",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import uuid
import warnings
warnings.filterwarnings('ignore')

# Check Python version compatibility
def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        st.error(f"Python 3.11+ required. Current version: {version.major}.{version.minor}.{version.micro}")
        st.info("Please upgrade Python to 3.11 or higher")
        st.stop()
    elif version.minor < 13:
        st.info(f"Running Python {version.major}.{version.minor}.{version.micro}. For optimal performance, consider upgrading to Python 3.13+")

check_python_version()

from database import db_manager
from fast_ml_model import FastRealEstatePredictor
from investment_analyzer import InvestmentAnalyzer
from emi_calculator import EMICalculator
from real_estate_chatbot import RealEstateChatbot
from portfolio_analyzer import PropertyPortfolioAnalyzer
from appreciation_analyzer import PropertyAppreciationAnalyzer

# Enhanced Professional CSS
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        gap: 10px;
        margin-bottom: 2rem;
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: white;
        color: #667eea;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none;
        border: 2px solid #667eea;
        transition: all 0.3s ease;
        cursor: pointer;
        font-weight: 600;
    }
    
    .nav-pill:hover {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
    }
    
    .nav-pill.active {
        background: #667eea;
        color: white;
    }
    
    /* Prediction Results */
    .prediction-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .investment-score {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Feature Importance */
    .feature-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .feature-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 5px 0;
    }
    
    /* Info Boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Success Box */
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Warning Box */
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Floating Chat Icon */
    .floating-chat {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 1000;
        transition: all 0.3s ease;
    }
    
    .floating-chat:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Price Range Card */
    .price-range-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .price-range-card h4 {
        margin-bottom: 0.5rem;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    .price-range-card h2 {
        margin: 0.5rem 0;
        font-weight: bold;
        font-size: 1.4rem;
    }
    
    .price-range-card p {
        margin-top: 0.5rem;
        opacity: 0.8;
        font-size: 0.9rem;
    }
    
    /* Animation */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'prediction'
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'show_chatbot' not in st.session_state:
    st.session_state.show_chatbot = False

@st.cache_data
def load_database_data():
    """Load and validate data from database only"""
    try:
        # Load data from database
        data = db_manager.get_properties_from_db()
        
        if data.empty:
            st.error("No data found in database")
            st.info("Please contact administrator to import property data")
            return None
        
        # Validate required columns (database returns title case column names)
        required_columns = ['City', 'District', 'Sub_District', 'Area_SqFt', 'BHK', 'Property_Type', 'Furnishing', 'Price_INR']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns in database: {missing_columns}")
            return None
        
        # Clean and validate data
        data = data.dropna(subset=required_columns)
        data['Price_INR'] = pd.to_numeric(data['Price_INR'], errors='coerce')
        data['Area_SqFt'] = pd.to_numeric(data['Area_SqFt'], errors='coerce')
        data = data.dropna(subset=['Price_INR', 'Area_SqFt'])
        
        # Filter realistic values
        data = data[
            (data['Price_INR'] > 100000) & 
            (data['Price_INR'] < 100000000) &
            (data['Area_SqFt'] > 100) & 
            (data['Area_SqFt'] < 10000)
        ]
        
        if len(data) < 100:
            st.warning(f"Limited data available: {len(data)} properties")
        else:
            st.success(f"✅ Successfully loaded {len(data)} properties from database")
        
        return data
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("Please check database configuration")
        return None

def get_session_id():
    """Get or create session ID for user tracking"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_districts(data, city):
    """Get districts for a city"""
    if data is None:
        return []
    return sorted(data[data['City'] == city]['District'].unique().tolist())

def get_sub_districts(data, city, district):
    """Get sub-districts for a city and district"""
    if data is None:
        return []
    filtered_data = data[(data['City'] == city) & (data['District'] == district)]
    return sorted(filtered_data['Sub_District'].unique().tolist())

def main():
    # Load data
    data = load_database_data()
    
    if data is None:
        st.stop()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Real Estate Price Predictor</h1>
        <p>Professional Property Analytics Platform with ML-Powered Predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation (Visual indicator only)
    st.markdown("""
    <div class="nav-pills">
        <div class="nav-pill {}">🏠 Property Prediction</div>
        <div class="nav-pill {}">📊 Portfolio Tracker</div>
        <div class="nav-pill {}">💰 Investment Analyzer</div>
        <div class="nav-pill {}">📈 Market Trends</div>
        <div class="nav-pill {}">🧮 EMI Calculator</div>
        <div class="nav-pill {}">🤖 AI Assistant</div>
    </div>
    """.format(
        'active' if st.session_state.page == 'prediction' else '',
        'active' if st.session_state.page == 'portfolio' else '',
        'active' if st.session_state.page == 'investment' else '',
        'active' if st.session_state.page == 'trends' else '',
        'active' if st.session_state.page == 'emi' else '',
        'active' if st.session_state.page == 'chatbot' else ''
    ), unsafe_allow_html=True)
    
    # Page selection
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("🏠 Property Prediction"):
            st.session_state.page = 'prediction'
    with col2:
        if st.button("📊 Portfolio Tracker"):
            st.session_state.page = 'portfolio'
    with col3:
        if st.button("💰 Investment Analyzer"):
            st.session_state.page = 'investment'
    with col4:
        if st.button("📈 Market Trends"):
            st.session_state.page = 'trends'
    with col5:
        if st.button("🧮 EMI Calculator"):
            st.session_state.page = 'emi'
    with col6:
        if st.button("🤖 AI Assistant"):
            st.session_state.page = 'chatbot'
    
    # Show selected page
    if st.session_state.page == 'prediction':
        show_prediction_interface(data)
    elif st.session_state.page == 'portfolio':
        show_portfolio_tracker(data)
    elif st.session_state.page == 'investment':
        show_investment_analyzer(data)
    elif st.session_state.page == 'trends':
        show_appreciation_trends()
    elif st.session_state.page == 'emi':
        show_emi_calculator()
    elif st.session_state.page == 'chatbot':
        show_chatbot_interface()
    
    # Show floating chat icon
    show_floating_chat_icon()
    
    # Show prediction results if available
    if st.session_state.prediction_results and st.session_state.page == 'prediction':
        show_prediction_results()

def show_emi_calculator():
    """Display EMI calculator interface"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    st.markdown("## 🧮 EMI Calculator")
    st.markdown("Calculate your monthly EMI and analyze loan details")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Loan Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            loan_amount = st.number_input("Loan Amount (₹)", 
                                        min_value=100000, 
                                        max_value=50000000, 
                                        value=2500000,
                                        step=100000,
                                        help="Enter the total loan amount")
            
            interest_rate = st.number_input("Annual Interest Rate (%)", 
                                          min_value=1.0, 
                                          max_value=20.0, 
                                          value=8.5,
                                          step=0.1,
                                          help="Current home loan rates range from 8.5% to 11%")
        
        with col_b:
            tenure_years = st.number_input("Loan Tenure (Years)", 
                                         min_value=1, 
                                         max_value=30, 
                                         value=20,
                                         step=1,
                                         help="Typical home loan tenure is 15-25 years")
            
            # Optional prepayment
            prepayment_amount = st.number_input("One-time Prepayment (₹)", 
                                              min_value=0, 
                                              max_value=10000000, 
                                              value=0,
                                              step=50000,
                                              help="Optional prepayment amount")
        
        if prepayment_amount > 0:
            prepayment_month = st.number_input("Prepayment Month", 
                                             min_value=1, 
                                             max_value=tenure_years*12, 
                                             value=12,
                                             step=1,
                                             help="Month when prepayment will be made")
        else:
            prepayment_month = 0
    
    with col2:
        st.markdown("### Quick Info")
        st.markdown("""
        <div class="info-box">
        <h4>💡 Tips:</h4>
        <ul>
        <li>Lower interest rates save lakhs over tenure</li>
        <li>Shorter tenure = less total interest</li>
        <li>Prepayments significantly reduce total cost</li>
        <li>Consider tax benefits under Section 80C & 24</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Calculate EMI
    if st.button("📊 Calculate EMI", key="calc_emi"):
        calculator = EMICalculator()
        
        # Basic EMI calculation
        emi_result = calculator.calculate_emi(loan_amount, interest_rate, tenure_years)
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 EMI Calculation Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Monthly EMI</h3>
                <h2 style="color: #667eea;">₹{emi_result['emi']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Amount</h3>
                <h2 style="color: #764ba2;">₹{emi_result['total_amount']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Interest</h3>
                <h2 style="color: #f093fb;">₹{emi_result['total_interest']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Interest %</h3>
                <h2 style="color: #4facfe;">{ (emi_result['total_interest']/loan_amount*100):.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Prepayment analysis
        if prepayment_amount > 0:
            prepayment_result = calculator.calculate_prepayment_savings(
                loan_amount, interest_rate, tenure_years, prepayment_amount, prepayment_month
            )
            
            st.markdown("### 💰 Prepayment Benefits")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="success-box">
                    <h4>Interest Saved</h4>
                    <h3>₹{prepayment_result['interest_saved']:,.0f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="success-box">
                    <h4>Time Saved</h4>
                    <h3>{prepayment_result['time_saved_months']:.0f} months</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="success-box">
                    <h4>New Tenure</h4>
                    <h3>{prepayment_result['new_tenure_months']:.0f} months</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Amortization schedule
        st.markdown("### 📋 First Year Payment Schedule")
        
        schedule = calculator.generate_amortization_schedule(loan_amount, interest_rate, tenure_years, 12)
        
        schedule_df = pd.DataFrame(schedule)
        
        # Rename columns to match expected format
        schedule_df = schedule_df.rename(columns={
            'month': 'Month',
            'principal': 'Principal', 
            'interest': 'Interest',
            'emi': 'Total Payment',
            'outstanding': 'Balance'
        })
        
        schedule_df = schedule_df[['Month', 'Principal', 'Interest', 'Total Payment', 'Balance']]
        
        # Format numbers
        for col in ['Principal', 'Interest', 'Total Payment', 'Balance']:
            schedule_df[col] = schedule_df[col].astype(float).astype(int)
        
        st.dataframe(schedule_df, use_container_width=True)
        
        # Payment breakdown chart

    
    st.markdown('</div>', unsafe_allow_html=True)

def show_chatbot_interface():
    """Display the AI chatbot interface"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RealEstateChatbot()
        st.session_state.chatbot.initialize_chat_history()
    
    # Render chatbot interface
    st.session_state.chatbot.render_chatbot_interface()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_prediction_results():
    """Display stored prediction results"""
    if not st.session_state.prediction_results:
        return
    
    results = st.session_state.prediction_results
    
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main prediction with range
        base_prediction = results['prediction']
        lower_bound = base_prediction * 0.90
        upper_bound = base_prediction * 1.10
        
        st.markdown(f"""
        <div class="prediction-result">
            <h2>Predicted Property Value Range</h2>
            <h1>₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}</h1>
            <p>Best Estimate: ₹{base_prediction:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Investment score
        if 'investment_score' in results:
            score_color = "#43e97b" if results['investment_score'] >= 70 else "#f093fb" if results['investment_score'] >= 50 else "#ff6b6b"
            st.markdown(f"""
            <div class="investment-score" style="background: linear-gradient(135deg, {score_color} 0%, #38f9d7 100%);">
                <h3>Investment Score</h3>
                <h1>{results['investment_score']}/100</h1>
                <p>{results.get('investment_recommendation', 'Good Investment')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Price range estimate
        if 'prediction' in results:
            st.markdown("### 🎯 Price Range Estimate")
            
            base_prediction = results['prediction']
            # Calculate a realistic range based on ±10-15% variation
            lower_bound = base_prediction * 0.90
            upper_bound = base_prediction * 1.10
            
            st.markdown(f"""
            <div class="price-range-card">
                <h4>Estimated Property Value Range</h4>
                <h2>₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}</h2>
                <p>Best Estimate: ₹{base_prediction:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence level
            confidence_level = "High" if results.get('training_metrics', {}).get('r2_score', 0) > 0.85 else "Medium"
            st.info(f"Confidence Level: {confidence_level}")
        
        # Feature importance
        if 'feature_importance' in results and results['feature_importance']:
            st.markdown("### 🎯 Key Factors")
            
            # Sort by importance
            importance_items = sorted(results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            
            for feature, importance in importance_items:
                st.markdown(f"""
                <div class="feature-item">
                    <span>{feature}</span>
                    <span>{importance:.1%}</span>
                </div>
                <div class="feature-bar" style="width: {importance*100}%;"></div>
                """, unsafe_allow_html=True)

def show_prediction_interface(data):
    """Display the main property prediction interface"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🏠 Property Details")
        
        # Location inputs
        col_a, col_b = st.columns(2)
        
        with col_a:
            cities = sorted(data['City'].unique().tolist())
            selected_city = st.selectbox("Select City", cities, key="pred_city")
            
            districts = get_districts(data, selected_city)
            selected_district = st.selectbox("Select District", districts, key="pred_district")
        
        with col_b:
            sub_districts = get_sub_districts(data, selected_city, selected_district)
            selected_sub_district = st.selectbox("Select Sub-District", sub_districts, key="pred_sub_district")
            
            property_types = sorted(data['Property_Type'].unique().tolist())
            selected_property_type = st.selectbox("Property Type", property_types, key="pred_property_type")
        
        # Property specifications
        col_c, col_d = st.columns(2)
        
        with col_c:
            area_sqft = st.number_input("Area (Square Feet)", 
                                      min_value=100, 
                                      max_value=10000, 
                                      value=1000,
                                      step=50,
                                      key="pred_area")
            
            bhk_options = sorted(data['BHK'].unique().tolist())
            selected_bhk = st.selectbox("BHK", bhk_options, key="pred_bhk")
        
        with col_d:
            furnishing_options = sorted(data['Furnishing'].unique().tolist())
            selected_furnishing = st.selectbox("Furnishing", furnishing_options, key="pred_furnishing")
        
        # Predict button
        if st.button("🎯 Predict Property Value", key="predict_btn", use_container_width=True):
            with st.spinner("Analyzing property data..."):
                try:
                    # Prepare input data
                    input_data = {
                        'City': selected_city,
                        'District': selected_district,
                        'Sub_District': selected_sub_district,
                        'Area_SqFt': area_sqft,
                        'BHK': selected_bhk,
                        'Property_Type': selected_property_type,
                        'Furnishing': selected_furnishing
                    }
                    
                    # Initialize and train predictor
                    predictor = FastRealEstatePredictor()
                    
                    # Train model with current data
                    st.info("Training three ML models: Decision Tree, Random Forest, and XGBoost...")
                    training_metrics = predictor.train_model(data)
                    
                    # Display only the best model result
                    if not training_metrics.get('cached', False):
                        best_model = training_metrics.get('best_model', 'Unknown')
                        r2_score = training_metrics.get('r2_score', 0)
                        mae = training_metrics.get('mae', 0)
                        
                        st.success(f"Model trained successfully! Accuracy: {r2_score:.1%}")
                    
                    # Make prediction
                    prediction, all_predictions = predictor.predict(input_data)
                    
                    # Get feature importance
                    feature_importance = predictor.get_feature_importance()
                    
                    # Investment analysis
                    investment_analyzer = InvestmentAnalyzer()
                    investment_score, investment_recommendation = investment_analyzer.analyze(input_data, prediction)
                    
                    # Store results
                    st.session_state.prediction_results = {
                        'prediction': prediction,
                        'all_predictions': all_predictions,
                        'feature_importance': feature_importance,
                        'investment_score': investment_score,
                        'investment_recommendation': investment_recommendation,
                        'input_data': input_data,
                        'training_metrics': training_metrics,
                        'model_used': 'FastRealEstatePredictor'
                    }
                    
                    # Save to database
                    session_id = get_session_id()
                    db_manager.save_prediction(session_id, input_data, st.session_state.prediction_results)
                    
                    st.success("✅ Prediction completed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.info("Please check your inputs and try again")
    
    with col2:
        st.markdown("### 📊 Market Overview")
        
        # Quick stats for selected city
        city_data = data[data['City'] == selected_city]
        
        if not city_data.empty:
            avg_price = city_data['Price_INR'].mean()
            avg_price_per_sqft = city_data['Price_per_SqFt'].mean()
            total_properties = len(city_data)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Price</h4>
                <h3>₹{avg_price:,.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Price per Sq.Ft</h4>
                <h3>₹{avg_price_per_sqft:,.0f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Available Properties</h4>
                <h3>{total_properties:,}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent predictions
        st.markdown("### 🕒 Recent Predictions")
        session_id = get_session_id()
        recent_predictions = db_manager.get_prediction_history(session_id, limit=5)
        
        if recent_predictions:
            for i, pred in enumerate(recent_predictions[:3]):
                st.markdown(f"""
                <div class="info-box">
                    <small>{pred['city']} - {pred['bhk']} BHK</small><br>
                    <strong>₹{pred['predicted_price']:,.0f}</strong>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent predictions found")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_floating_chat_icon():
    """Display static floating chat icon in bottom right corner"""
    if not st.session_state.show_chatbot:
        st.markdown("""
        <div class="floating-chat" onclick="alert('Click the AI Assistant button above to access the chatbot!')">
            <span style="font-size: 24px;">🤖</span>
        </div>
        """, unsafe_allow_html=True)

def show_portfolio_tracker(data):
    """Display portfolio tracking interface for existing properties"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    st.markdown("## 📊 Portfolio Tracker")
    st.markdown("Track your existing property investments and get market insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏠 Your Property Details")
        
        # Property purchase details
        col_a, col_b = st.columns(2)
        
        with col_a:
            cities = sorted(data['City'].unique().tolist())
            property_city = st.selectbox("Property City", cities, key="portfolio_city")
            
            districts = get_districts(data, property_city)
            property_district = st.selectbox("Property District", districts, key="portfolio_district")
            
            sub_districts = get_sub_districts(data, property_city, property_district)
            property_sub_district = st.selectbox("Property Sub-District", sub_districts, key="portfolio_sub_district")
        
        with col_b:
            property_types = sorted(data['Property_Type'].unique().tolist())
            property_type = st.selectbox("Property Type", property_types, key="portfolio_property_type")
            
            furnishing_options = sorted(data['Furnishing'].unique().tolist())
            property_furnishing = st.selectbox("Furnishing", furnishing_options, key="portfolio_furnishing")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            property_area = st.number_input("Area (Sq.Ft)", 
                                          min_value=100, 
                                          max_value=10000, 
                                          value=1200,
                                          step=50,
                                          key="portfolio_area")
            
            bhk_options = sorted(data['BHK'].unique().tolist())
            property_bhk = st.selectbox("BHK", bhk_options, key="portfolio_bhk")
        
        with col_d:
            purchase_price = st.number_input("Purchase Price (₹)", 
                                           min_value=100000, 
                                           max_value=50000000, 
                                           value=2500000,
                                           step=100000,
                                           key="portfolio_purchase_price")
            
            purchase_year = st.number_input("Purchase Year", 
                                          min_value=2000, 
                                          max_value=2024, 
                                          value=2020,
                                          step=1,
                                          key="portfolio_purchase_year")
    
    with col2:
        st.markdown("### 💡 Portfolio Tips")
        st.markdown("""
        <div class="info-box">
        <h4>Track Performance:</h4>
        <ul>
        <li>Monitor current market value vs purchase price</li>
        <li>Track appreciation rates in your area</li>
        <li>Compare with market benchmarks</li>
        <li>Get hold/sell recommendations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("📈 Analyze Portfolio Performance", key="analyze_portfolio"):
        with st.spinner("Analyzing your property portfolio..."):
            try:
                # Prepare property data
                purchase_data = {
                    'city': property_city,
                    'district': property_district,
                    'sub_district': property_sub_district,
                    'area_sqft': property_area,
                    'bhk': property_bhk,
                    'property_type': property_type,
                    'furnishing': property_furnishing,
                    'purchase_price': purchase_price,
                    'purchase_year': purchase_year
                }
                
                # Initialize analyzers
                portfolio_analyzer = PropertyPortfolioAnalyzer()
                predictor = FastRealEstatePredictor()
                
                # Train current predictor
                predictor.train_model(data)
                
                # Analyze current property value
                try:
                    current_analysis = portfolio_analyzer.analyze_current_property_value(purchase_data, predictor)
                except Exception as e:
                    st.error(f"Portfolio analysis error: {str(e)}")
                    st.write("Debug info - Purchase data keys:", list(purchase_data.keys()))
                    st.write("Debug info - Purchase data:", purchase_data)
                    return
                
                # Generate recommendations
                property_data = {k: v for k, v in purchase_data.items() if k != 'purchase_price' and k != 'purchase_year'}
                recommendation = portfolio_analyzer.generate_hold_sell_recommendation(current_analysis, property_data)
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Portfolio Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Current Value</h3>
                        <h2 style="color: #667eea;">₹{current_analysis['current_value']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    gain_loss = current_analysis['total_appreciation']
                    color = "#43e97b" if gain_loss >= 0 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Gain/Loss</h3>
                        <h2 style="color: {color};">₹{abs(gain_loss):,.0f}</h2>
                        <p>{'Gain' if gain_loss >= 0 else 'Loss'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    appreciation = current_analysis['total_growth_percent']
                    color = "#43e97b" if appreciation >= 0 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Appreciation</h3>
                        <h2 style="color: {color};">{appreciation:+.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    annual_return = current_analysis['annual_growth_percent']
                    color = "#43e97b" if annual_return >= 5 else "#f093fb" if annual_return >= 0 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Annual Return</h3>
                        <h2 style="color: {color};">{annual_return:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation
                rec_color = "#43e97b" if recommendation['recommendation'] == "HOLD" else "#f093fb" if recommendation['recommendation'] == "SELL" else "#4facfe"
                st.markdown(f"""
                <div class="chart-container">
                    <h3>🎯 Investment Recommendation: <span style="color: {rec_color};">{recommendation['recommendation']}</span></h3>
                    <p><strong>Reasoning:</strong> {recommendation['reasoning']}</p>
                    <p><strong>Confidence:</strong> {recommendation['confidence_score']:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Market timing analysis
                st.markdown("### ⏰ Market Timing Analysis")
                timing_analysis = portfolio_analyzer.generate_market_timing_analysis(property_city)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Market Phase: {timing_analysis['market_phase']}</h4>
                        <p>{timing_analysis['phase_description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Best Action: {timing_analysis['recommended_action']}</h4>
                        <p>{timing_analysis['action_reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Performance chart
                years_held = 2024 - purchase_year
                if years_held > 0:
                    st.markdown("### 📈 Investment Performance Over Time")
                    
                    # Create performance chart
                    years = list(range(purchase_year, 2025))
                    values = []
                    
                    for year in years:
                        if year == purchase_year:
                            values.append(purchase_price)
                        else:
                            years_diff = year - purchase_year
                            annual_growth = current_analysis['annualized_return'] / 100
                            projected_value = purchase_price * ((1 + annual_growth) ** years_diff)
                            values.append(projected_value)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', 
                                           name='Property Value', line=dict(color='#667eea', width=3)))
                    
                    fig.update_layout(
                        title='Property Value Growth Over Time',
                        xaxis_title='Year',
                        yaxis_title='Value (₹)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Portfolio analysis failed: {str(e)}")
                st.info("Please check your inputs and try again")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_investment_analyzer(data):
    """Display investment opportunity analyzer"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    st.markdown("## 💰 Investment Opportunity Analyzer")
    st.markdown("Analyze potential investment properties and opportunities")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Target Property Analysis")
        
        # Investment criteria
        col_a, col_b = st.columns(2)
        
        with col_a:
            investment_budget = st.number_input("Investment Budget (₹)", 
                                              min_value=500000, 
                                              max_value=50000000, 
                                              value=3000000,
                                              step=100000,
                                              key="investment_budget")
            
            cities = sorted(data['City'].unique().tolist())
            target_city = st.selectbox("Target City", cities, key="investment_city")
            
            districts = get_districts(data, target_city)
            target_district = st.selectbox("Target District", districts, key="investment_district")
        
        with col_b:
            investment_horizon = st.number_input("Investment Horizon (Years)", 
                                               min_value=1, 
                                               max_value=20, 
                                               value=5,
                                               step=1,
                                               key="investment_horizon")
            
            risk_tolerance = st.selectbox("Risk Tolerance", 
                                        ["Conservative", "Moderate", "Aggressive"], 
                                        index=1,
                                        key="risk_tolerance")
        
        # Property specifications for analysis
        st.markdown("### 🏠 Property Specifications")
        
        col_c, col_d = st.columns(2)
        
        with col_c:
            sub_districts = get_sub_districts(data, target_city, target_district)
            target_sub_district = st.selectbox("Target Sub-District", sub_districts, key="investment_sub_district")
            
            property_types = sorted(data['Property_Type'].unique().tolist())
            target_property_type = st.selectbox("Property Type", property_types, key="investment_property_type")
        
        with col_d:
            target_area = st.number_input("Desired Area (Sq.Ft)", 
                                        min_value=500, 
                                        max_value=5000, 
                                        value=1200,
                                        step=50,
                                        key="investment_area")
            
            bhk_options = sorted(data['BHK'].unique().tolist())
            target_bhk = st.selectbox("BHK", bhk_options, key="investment_bhk")
            
            furnishing_options = sorted(data['Furnishing'].unique().tolist())
            target_furnishing = st.selectbox("Furnishing", furnishing_options, key="investment_furnishing")
    
    with col2:
        st.markdown("### 💡 Investment Tips")
        st.markdown("""
        <div class="info-box">
        <h4>Smart Investing:</h4>
        <ul>
        <li>Location is the most important factor</li>
        <li>Consider future infrastructure development</li>
        <li>Analyze rental yield potential</li>
        <li>Factor in maintenance costs</li>
        <li>Monitor market cycles</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Market insights for selected city
        city_data = data[data['City'] == target_city]
        if not city_data.empty:
            avg_price_per_sqft = city_data['Price_per_SqFt'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h4>{target_city} Avg Price</h4>
                <h3>₹{avg_price_per_sqft:,.0f}/sq.ft</h3>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Investment Opportunity", key="analyze_investment"):
        with st.spinner("Analyzing investment opportunity..."):
            try:
                # Prepare target property data
                target_property = {
                    'City': target_city,
                    'District': target_district,
                    'Sub_District': target_sub_district,
                    'Area_SqFt': target_area,
                    'BHK': target_bhk,
                    'Property_Type': target_property_type,
                    'Furnishing': target_furnishing
                }
                
                # Initialize analyzers
                portfolio_analyzer = PropertyPortfolioAnalyzer()
                predictor = FastRealEstatePredictor()
                
                # Train current predictor
                predictor.train_model(data)
                
                # Analyze investment opportunity
                try:
                    investment_analysis = portfolio_analyzer.analyze_investment_opportunity(
                        target_property, investment_budget, predictor
                    )
                except Exception as analysis_error:
                    st.error(f"❌ Investment analysis failed: {str(analysis_error)}")
                    st.info("Please check your inputs and try again")
                    return
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Investment Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Fair Value</h3>
                        <h2 style="color: #667eea;">₹{investment_analysis['fair_market_value']:,.0f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    budget_fit = investment_analysis['budget_adequacy']
                    color = "#43e97b" if budget_fit >= 100 else "#f093fb" if budget_fit >= 80 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Budget Fit</h3>
                        <h2 style="color: {color};">{budget_fit:.0f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    roi = investment_analysis['projected_roi_annual']
                    color = "#43e97b" if roi >= 8 else "#f093fb" if roi >= 5 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Expected ROI</h3>
                        <h2 style="color: {color};">{roi:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    score = investment_analysis['investment_attractiveness_score']
                    color = "#43e97b" if score >= 75 else "#f093fb" if score >= 50 else "#ff6b6b"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Investment Score</h3>
                        <h2 style="color: {color};">{score}/100</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Investment recommendation
                recommendation = investment_analysis['investment_recommendation']
                rec_color = "#43e97b" if "BUY" in recommendation.upper() else "#f093fb" if "CONSIDER" in recommendation.upper() else "#ff6b6b"
                
                st.markdown(f"""
                <div class="chart-container">
                    <h3>🎯 Investment Recommendation</h3>
                    <h2 style="color: {rec_color};">{recommendation}</h2>
                    <p><strong>Analysis:</strong> {investment_analysis['detailed_analysis']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk assessment
                st.markdown("### ⚠️ Risk Assessment")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h4>Risk Level: {investment_analysis['risk_level']}</h4>
                        <ul>
                        <li>Market volatility risk</li>
                        <li>Liquidity considerations</li>
                        <li>Location-specific factors</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Key Factors:</h4>
                        <ul>
                        <li>Budget adequacy: {budget_fit:.0f}%</li>
                        <li>Expected annual return: {roi:.1f}%</li>
                        <li>Investment timeline: {investment_horizon} years</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Projected returns over time
                if investment_horizon > 1:
                    st.markdown("### 📈 Projected Investment Returns")
                    
                    years = list(range(1, investment_horizon + 1))
                    investment_values = []
                    
                    initial_value = investment_analysis['fair_market_value']
                    annual_growth = roi / 100
                    
                    for year in years:
                        projected_value = initial_value * ((1 + annual_growth) ** year)
                        investment_values.append(projected_value)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=years, y=investment_values, mode='lines+markers',
                                           name='Projected Value', line=dict(color='#667eea', width=3)))
                    fig.add_hline(y=investment_budget, line_dash="dash", line_color="red",
                                annotation_text="Initial Investment")
                    
                    fig.update_layout(
                        title=f'Projected Investment Growth ({investment_horizon} Years)',
                        xaxis_title='Year',
                        yaxis_title='Property Value (₹)',
                        hovermode='x unified',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Investment analysis failed: {str(e)}")
                st.info("Please check your inputs and try again")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_appreciation_trends():
    """Display property appreciation trends and historical analysis"""
    st.markdown('<div class="slide-in">', unsafe_allow_html=True)
    
    st.markdown("## 📈 Property Appreciation Trends")
    st.markdown("Analyze historical property appreciation and market trends")
    
    # Initialize appreciation analyzer
    appreciation_analyzer = PropertyAppreciationAnalyzer()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Analysis Parameters")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # City selection for analysis
            major_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]
            selected_cities = st.multiselect("Select Cities for Comparison", 
                                           major_cities, 
                                           default=["Mumbai", "Delhi", "Bangalore"],
                                           key="appreciation_cities")
            
            analysis_years = st.number_input("Analysis Period (Years)", 
                                           min_value=1, 
                                           max_value=20, 
                                           value=5,
                                           step=1,
                                           key="appreciation_years")
        
        with col_b:
            focus_city = st.selectbox("Focus City for Detailed Analysis", 
                                    selected_cities if selected_cities else major_cities[:1],
                                    key="focus_city")
            
            investment_amount = st.number_input("Hypothetical Investment (₹)", 
                                              min_value=1000000, 
                                              max_value=50000000, 
                                              value=5000000,
                                              step=500000,
                                              key="appreciation_investment")
    
    with col2:
        st.markdown("### 📊 Market Insights")
        st.markdown("""
        <div class="info-box">
        <h4>Historical Trends:</h4>
        <ul>
        <li>Mumbai: Premium market leader</li>
        <li>Bangalore: Tech hub growth</li>
        <li>Delhi NCR: Consistent performer</li>
        <li>Chennai: Stable appreciation</li>
        <li>Pune: Emerging hotspot</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("📈 Generate Appreciation Analysis", key="analyze_appreciation"):
        with st.spinner("Generating comprehensive market analysis..."):
            try:
                if not selected_cities:
                    st.error("Please select at least one city for analysis")
                    return
                
                # City comparison analysis
                st.markdown("---")
                st.markdown("## 📊 City Performance Comparison")
                
                comparison_df = appreciation_analyzer.compare_cities_performance(selected_cities, analysis_years)
                
                # Display comparison table
                st.dataframe(comparison_df, use_container_width=True)
                
                # Performance metrics for each city
                col1, col2, col3, col4 = st.columns(4)
                
                for i, city in enumerate(selected_cities[:4]):
                    city_metrics = appreciation_analyzer.calculate_appreciation_metrics(city, analysis_years)
                    
                    with [col1, col2, col3, col4][i]:
                        color = "#43e97b" if city_metrics['average_annual_growth'] > 8 else "#f093fb" if city_metrics['average_annual_growth'] > 5 else "#ff6b6b"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>{city}</h4>
                            <h3 style="color: {color};">{city_metrics['average_annual_growth']:.1f}%</h3>
                            <small>Annual Growth</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Appreciation trends chart
                st.markdown("### 📈 Historical Appreciation Trends")
                
                trends_chart = appreciation_analyzer.create_appreciation_trends_chart(selected_cities)
                st.plotly_chart(trends_chart, use_container_width=True)
                
                # Detailed analysis for focus city
                if focus_city:
                    st.markdown(f"### 🎯 Detailed Analysis: {focus_city}")
                    
                    focus_metrics = appreciation_analyzer.calculate_appreciation_metrics(focus_city, analysis_years)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Average Growth</h4>
                            <h3>{focus_metrics['average_annual_growth']:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Market Phase</h4>
                            <h3>{focus_metrics['market_phase']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Risk Level</h4>
                            <h3>{focus_metrics['risk_level']}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Investment recommendations
                    recommendations = appreciation_analyzer.get_investment_recommendations(focus_city, analysis_years)
                    
                    st.markdown(f"""
                    <div class="chart-container">
                        <h3>🎯 Investment Recommendation: <span style="color: #667eea;">{recommendations['overall_rating']}</span></h3>
                        <p><strong>Best Investment Strategy:</strong> {recommendations['best_strategy']}</p>
                        <p><strong>Expected Return:</strong> {recommendations['expected_annual_return']:.1f}% annually</p>
                        <p><strong>Recommendation:</strong> {recommendations['recommendation_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Future projection chart
                    st.markdown(f"### 🔮 Future Value Projection ({focus_city})")
                    
                    projection_chart = appreciation_analyzer.create_future_projection_chart(
                        focus_city, investment_amount, 10
                    )
                    st.plotly_chart(projection_chart, use_container_width=True)
                
                # Market insights and recommendations
                st.markdown("### 💡 Market Insights & Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="success-box">
                        <h4>🏆 Top Performing Markets</h4>
                        <p>Based on historical data and growth potential:</p>
                        <ul>
                        <li><strong>Bangalore:</strong> Tech sector driven growth</li>
                        <li><strong>Mumbai:</strong> Financial capital premium</li>
                        <li><strong>Pune:</strong> Emerging IT hub</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Investment Considerations</h4>
                        <ul>
                        <li>Market cycles vary by city</li>
                        <li>Infrastructure development impacts</li>
                        <li>Economic policy changes</li>
                        <li>Local demand-supply dynamics</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Appreciation analysis failed: {str(e)}")
                st.info("Please try again with different parameters")
                
                # Add debugging information for development
                if str(e).find("expected_annual_return") != -1:
                    st.info("Debug: Issue with investment recommendations - checking data structure")
                    
                # Fallback: Show basic city information
                st.markdown("### 📊 Basic City Information")
                for city in selected_cities[:3]:
                    try:
                        basic_metrics = appreciation_analyzer.calculate_appreciation_metrics(city, analysis_years)
                        st.info(f"**{city}**: Average growth {basic_metrics.get('average_annual_growth', 7.5):.1f}% annually")
                    except:
                        st.info(f"**{city}**: Market data temporarily unavailable")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()