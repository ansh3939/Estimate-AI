import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from database import db_manager
from fast_ml_model import FastRealEstatePredictor
from investment_analyzer import InvestmentAnalyzer
from emi_calculator import EMICalculator

from real_estate_chatbot import RealEstateChatbot
from portfolio_analyzer import PropertyPortfolioAnalyzer
import uuid
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Real Estate Price Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS
st.markdown("""
<style>
    /* Global App Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Main Header with Advanced Effects */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Enhanced Price Display */
    .price-display {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 3rem 2rem;
        background-color: white;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Modern Card Design */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Recommendation Cards */
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border: none;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 30px rgba(168, 237, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .success-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: none;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 30px rgba(252, 182, 159, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .warning-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
    }
    
    .error-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border: none;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 30px rgba(250, 177, 160, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .error-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
    }
    
    .info-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 15px;
        padding: 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        box-shadow: 0 10px 25px rgba(118, 75, 162, 0.4);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Form Controls */
    .stSelectbox > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.2);
    }
    
    .stNumberInput > div > div {
        background: white;
        border-radius: 10px;
        border: 2px solid #e1e8ed;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Metric Container Styling */
    [data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart Styling */
    .stPlotlyChart {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    .stMarkdown p {
        color: #34495e;
        line-height: 1.6;
    }
    
    /* Loading Animation */
    .stSpinner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
    }
    
    /* Floating Chat Button */
    .stButton > button[data-testid="baseButton-primary"] {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        z-index: 1000 !important;
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        font-size: 24px !important;
        padding: 0 !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_database_data():
    """Load and validate data from database only"""
    try:
        # Get data directly from database
        db_data = db_manager.get_properties_from_db()
        
        if db_data.empty:
            raise ValueError("No data available in database")
        
        # Explicit data type conversion to prevent errors
        safe_data = pd.DataFrame({
            'City': db_data['City'].astype(str),
            'District': db_data['District'].astype(str),
            'Sub_District': db_data['Sub_District'].astype(str),
            'Area_SqFt': pd.to_numeric(db_data['Area_SqFt'], errors='coerce'),
            'BHK': pd.to_numeric(db_data['BHK'], errors='coerce').astype(int),
            'Property_Type': db_data['Property_Type'].astype(str),
            'Furnishing': db_data['Furnishing'].astype(str),
            'Price_INR': pd.to_numeric(db_data['Price_INR'], errors='coerce'),
            'Price_per_SqFt': pd.to_numeric(db_data['Price_per_SqFt'], errors='coerce')
        })
        
        # Remove any invalid data
        safe_data = safe_data.dropna()
        safe_data = safe_data[safe_data['Price_INR'] > 0]
        safe_data = safe_data[safe_data['Area_SqFt'] > 0]
        
        print(f"Successfully loaded {len(safe_data)} properties from database")
        return safe_data
        
    except Exception as e:
        st.error(f"Database loading error: {str(e)}")
        return pd.DataFrame()

def get_session_id():
    """Get or create session ID for user tracking"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_districts(data, city):
    """Get districts for a city"""
    if data.empty:
        return []
    city_data = data[data['City'] == city]
    return sorted(city_data['District'].unique().tolist())

def get_sub_districts(data, city, district):
    """Get sub-districts for a city and district"""
    if data.empty:
        return []
    filtered_data = data[(data['City'] == city) & (data['District'] == district)]
    return sorted(filtered_data['Sub_District'].unique().tolist())

def main():
    # Initialize session
    session_id = get_session_id()
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 3.2rem; font-weight: 700; position: relative; z-index: 2;">
            AI Real Estate Intelligence Platform
        </h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.4rem; opacity: 0.95; position: relative; z-index: 2;">
            Professional Investment Analysis & Market Intelligence Platform
        </p>
        <div style="margin-top: 1.5rem; font-size: 1rem; opacity: 0.8; position: relative; z-index: 2;">
            Powered by Advanced Machine Learning • Real-time Market Analysis • Investment Insights
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Portfolio Tracker", "Investment Analyzer"])
    
    with tab1:
        show_prediction_interface()
    
    with tab2:
        show_portfolio_tracker()
    
    with tab3:
        show_investment_analyzer()
    
    # Floating AI assistant icon
    show_floating_chat_icon()

def show_chatbot_interface():
    """Display the AI chatbot interface"""
    try:
        chatbot = RealEstateChatbot()
        
        # Render the chatbot interface
        chatbot.render_chatbot_interface()
        
        # Add suggested questions
        st.markdown("---")
        chatbot.render_suggested_questions()
        
    except Exception as e:
        st.error(f"Chatbot initialization error: {str(e)}")
        st.info("Please ensure the OpenAI API key is properly configured.")

def show_prediction_interface():
    """Display the main property prediction interface"""
    
    # Load data from database
    data = load_database_data()
    
    if data.empty:
        st.error("Unable to load data from database. Please contact support.")
        st.stop()
    
    # Initialize models
    try:
        fast_predictor = FastRealEstatePredictor()
        investment_analyzer = InvestmentAnalyzer()
        emi_calculator = EMICalculator()
        
        # Train fast model with database data
        performance = fast_predictor.train_model(data)
        
        # Model loaded silently for fast predictions
        
    except Exception as e:
        st.error(f"Model training error: {str(e)}")
        st.stop()
    
    # Enhanced sidebar with professional styling
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600;">Property Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Location selection with enhanced styling
        st.markdown("""
        <div class="sidebar-section">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem;">Location Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        city = st.selectbox("Select City", ["Mumbai", "Delhi", "Gurugram", "Noida", "Bangalore"], 
                           help="Choose the city for property analysis")
        
        districts = get_districts(data, city)
        district = st.selectbox("Select District", districts, 
                               help="District within the selected city")
        
        sub_districts = get_sub_districts(data, city, district)
        sub_district = st.selectbox("Select Sub-District", sub_districts,
                                   help="Specific area within the district")
        
        # Property details with enhanced styling
        st.markdown("""
        <div class="sidebar-section">
            <h4 style="color: #2c3e50; margin-bottom: 0.8rem;">Property Specifications</h4>
        </div>
        """, unsafe_allow_html=True)
        
        area_sqft = st.number_input("Area (Square Feet)", min_value=100, max_value=10000, 
                                   value=1000, step=50, help="Property area in square feet")
        bhk = st.selectbox("BHK Configuration", [1, 2, 3, 4, 5, 6], index=1,
                          help="Number of bedrooms, hall, and kitchen")
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House", "Studio"], 
                                    help="Type of property")
        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
        
        # Predict button
        predict_button = st.button("Predict Property Price", type="primary", use_container_width=True)
    
    # Main content area
    if predict_button:
        # Prepare input data
        input_data = {
            'City': city,
            'District': district,
            'Sub_District': sub_district,
            'Area_SqFt': area_sqft,
            'BHK': bhk,
            'Property_Type': property_type,
            'Furnishing': furnishing
        }
        
        try:
            # Make fast prediction
            predicted_price, confidence_scores = fast_predictor.predict(input_data)
            
            # Investment analysis
            investment_score, recommendation = investment_analyzer.analyze(input_data, predicted_price)
            
            # Save prediction to database
            try:
                session_id = get_session_id()
                prediction_result = {
                    'predicted_price': predicted_price,
                    'investment_score': investment_score,
                    'model_used': 'Fast Random Forest',
                    'all_predictions': confidence_scores or {}
                }
                prediction_id = db_manager.save_prediction(session_id, input_data, prediction_result)
            except:
                pass  # Continue without database save if error
            
            # Enhanced results display
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h3 style="color: #667eea; font-weight: 600; margin-bottom: 1rem;">Property Valuation Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced price display with professional styling
            st.markdown(f"""
            <div class="price-display">
                <div style="font-size: 1.2rem; margin-bottom: 0.5rem; opacity: 0.8;">Estimated Property Value</div>
                ₹{predicted_price:,.0f}
                <div style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.9; background: rgba(255,255,255,0.1); padding: 0.5rem; border-radius: 10px;">
                    ₹{predicted_price/area_sqft:,.0f} per Sq Ft
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Results columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Investment analysis
                if investment_score >= 7:
                    card_class = "success-card"
                    score_color = "#2E7D32"
                    status_text = "Excellent Investment"
                elif investment_score >= 5:
                    card_class = "warning-card"
                    score_color = "#F57C00"
                    status_text = "Good Investment"
                else:
                    card_class = "error-card"
                    score_color = "#C62828"
                    status_text = "High Risk"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4 style="margin-top: 0; color: #2c3e50; font-weight: 600;">Investment Analysis</h4>
                    <div style="font-size: 2.5rem; font-weight: 700; color: {score_color}; text-align: center; margin: 1rem 0;">
                        {investment_score}/10
                    </div>
                    <div style="background: rgba(255,255,255,0.3); padding: 0.8rem; border-radius: 10px; margin: 1rem 0;">
                        <p style="margin: 0; font-weight: 600; color: #2c3e50;">{status_text}</p>
                        <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #34495e; line-height: 1.4;">
                            {recommendation}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Enhanced EMI Calculator
                st.markdown("""
                <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06); margin-bottom: 1rem;">
                    <h4 style="color: #667eea; margin-top: 0; font-weight: 600;">💰 EMI Calculator</h4>
                </div>
                """, unsafe_allow_html=True)
                
                loan_percentage = st.slider("💳 Loan Amount (%)", 20, 90, 80, 
                                           help="Percentage of property value as loan")
                loan_amount = predicted_price * (loan_percentage / 100)
                
                interest_rate = st.slider("📈 Interest Rate (%)", 6.0, 15.0, 8.5, 0.1,
                                         help="Annual interest rate")
                tenure_years = st.slider("📅 Tenure (Years)", 5, 30, 20,
                                        help="Loan repayment period")
                
                emi_details = emi_calculator.calculate_emi(loan_amount, interest_rate, tenure_years)
                
                st.markdown(f"""
                <div class="info-section">
                    <div style="text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 2rem; font-weight: 700; color: white; margin-bottom: 0.5rem;">
                            ₹{emi_details['emi']:,.0f}
                        </div>
                        <div style="font-size: 1rem; opacity: 0.9;">Monthly EMI</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
                        <div style="display: flex; justify-content: space-between; margin: 0.3rem 0;">
                            <span>Loan Amount:</span> <strong>₹{loan_amount:,.0f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.3rem 0;">
                            <span>Total Interest:</span> <strong>₹{emi_details['total_interest']:,.0f}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin: 0.3rem 0;">
                            <span>Total Payment:</span> <strong>₹{emi_details['total_amount']:,.0f}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            

        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    else:
        # Enhanced default dashboard
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h3 style="color: #667eea; font-weight: 600; margin-bottom: 0.5rem;">Welcome to AI Real Estate Intelligence</h3>
            <p style="color: #34495e; font-size: 1.1rem; margin: 0;">Configure your property parameters in the sidebar to get instant AI-powered valuations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced market insights with professional styling
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = data['Price_INR'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #667eea; margin-bottom: 0.5rem;">
                        ₹{avg_price:,.0f}
                    </div>
                    <div style="color: #2c3e50; font-weight: 600;">Average Property Price</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.3rem;">Across all cities</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_sqft_price = data['Price_per_SqFt'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #764ba2; margin-bottom: 0.5rem;">
                        ₹{avg_sqft_price:,.0f}
                    </div>
                    <div style="color: #2c3e50; font-weight: 600;">Price per Sq Ft</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.3rem;">Market average</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_properties = len(data)
            st.markdown(f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem;">
                        {total_properties:,}
                    </div>
                    <div style="color: #2c3e50; font-weight: 600;">Properties Analyzed</div>
                    <div style="color: #7f8c8d; font-size: 0.9rem; margin-top: 0.3rem;">Database coverage</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced city-wise analysis
        st.markdown("""
        <div style="margin: 3rem 0 1rem 0;">
            <h4 style="color: #667eea; font-weight: 600; text-align: center;">Market Distribution by City</h4>
        </div>
        """, unsafe_allow_html=True)
        
        city_counts = data['City'].value_counts()
        fig = px.bar(x=city_counts.index, y=city_counts.values, 
                     title="", color=city_counts.values,
                     color_continuous_scale=['#667eea', '#764ba2'])
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            xaxis=dict(title="Cities", showgrid=False),
            yaxis=dict(title="Number of Properties", showgrid=True, gridcolor='#ecf0f1')
        )
        fig.update_traces(hovertemplate='<b>%{x}</b><br>Properties: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)

def show_floating_chat_icon():
    """Display floating chat icon in bottom right corner"""
    # Initialize chat state
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    
    # Enhanced floating chat button CSS
    st.markdown("""
    <style>
    .floating-chat-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        color: white;
        font-size: 28px;
        text-decoration: none;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    .floating-chat-btn:hover {
        transform: scale(1.15);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .floating-chat-btn:active {
        transform: scale(1.05);
    }
    
    .chat-sidebar {
        position: fixed;
        top: 0;
        right: -400px;
        width: 400px;
        height: 100vh;
        background: white;
        box-shadow: -5px 0 20px rgba(0,0,0,0.1);
        transition: right 0.3s ease;
        z-index: 1001;
        border-left: 3px solid #2E7D32;
    }
    
    .chat-sidebar.open {
        right: 0;
    }
    
    .stButton > button[data-testid="baseButton-primary"] {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        z-index: 1000 !important;
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #2E7D32, #4CAF50) !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(46, 125, 50, 0.3) !important;
        font-size: 24px !important;
        padding: 0 !important;
    }
    
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 6px 25px rgba(46, 125, 50, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Fixed position chat button
    chat_button_container = st.empty()
    
    with chat_button_container.container():
        if st.button("💬", key="floating_chat", help="Open AI Assistant"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
    
    # Show chat interface in sidebar when open
    if st.session_state.chat_open:
        with st.sidebar:
            st.markdown("### AI Real Estate Assistant")
            
            # Close button
            if st.button("✕ Close Chat", key="close_chat"):
                st.session_state.chat_open = False
                st.rerun()
            
            st.markdown("---")
            
            # Chat interface
            show_chatbot_interface()

def show_portfolio_tracker():
    """Display portfolio tracking interface for existing properties"""
    st.header("📊 Property Portfolio Tracker")
    st.markdown("Track your existing property values, analyze growth performance, and get buy/sell/hold recommendations.")
    
    # Initialize portfolio analyzer
    portfolio_analyzer = PropertyPortfolioAnalyzer()
    predictor = FastRealEstatePredictor()
    
    # Load data for training
    data = load_database_data()
    if data is not None and not data.empty:
        predictor.train_model(data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Details")
        
        # Property information
        city = st.selectbox("City", sorted(data['City'].unique()) if data is not None else ['Mumbai'])
        district = st.selectbox("District", get_districts(data, city) if data is not None else ['Central'])
        sub_district = st.selectbox("Sub District", get_sub_districts(data, city, district) if data is not None else ['Area 1'])
        
        area_sqft = st.number_input("Property Area (Sq Ft)", min_value=100, max_value=10000, value=1000, step=50)
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)
        property_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Builder Floor'])
        furnishing = st.selectbox("Furnishing", ['Unfurnished', 'Semi-Furnished', 'Furnished'], index=1)
        
        st.subheader("Purchase Information")
        purchase_price = st.number_input("Purchase Price (₹)", min_value=100000, max_value=50000000, value=5000000, step=100000)
        purchase_date = st.date_input("Purchase Date", value=pd.to_datetime('2020-01-01'))
        
        analyze_button = st.button("📈 Analyze My Property", type="primary")
    
    with col2:
        if analyze_button:
            with st.spinner("Analyzing your property portfolio..."):
                # Prepare property data for prediction
                property_data = {
                    'city': city,
                    'district': district,
                    'sub_district': sub_district,
                    'area_sqft': area_sqft,
                    'bhk': bhk,
                    'property_type': property_type,
                    'furnishing': furnishing,
                    'purchase_price': purchase_price,
                    'purchase_date': purchase_date.strftime('%Y-%m-%d')
                }
                
                # Analyze current value
                property_analysis = portfolio_analyzer.analyze_current_property_value(property_data, predictor)
                
                # Generate recommendation
                recommendation = portfolio_analyzer.generate_hold_sell_recommendation(property_analysis, property_data)
                
                # Display results
                st.subheader("📊 Property Analysis Results")
                
                # Key metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.metric(
                        "Current Value", 
                        f"₹{property_analysis['current_value']:,.0f}",
                        f"₹{property_analysis['total_appreciation']:,.0f}"
                    )
                
                with col_metric2:
                    st.metric(
                        "Total Growth", 
                        f"{property_analysis['total_growth_percent']:.1f}%",
                        f"{property_analysis['annual_growth_percent']:.1f}% annually"
                    )
                
                with col_metric3:
                    st.metric(
                        "vs Market", 
                        f"{property_analysis['performance_vs_market']:+.1f}%",
                        "Performance difference"
                    )
                
                # Recommendation
                st.subheader("🎯 Investment Recommendation")
                
                rec_color = {
                    "STRONG HOLD": "🟢", "HOLD": "🟢", 
                    "CONDITIONAL HOLD": "🟡", "CONSIDER SELLING": "🟠", 
                    "SELL": "🔴"
                }.get(recommendation['recommendation'], "🟡")
                
                st.markdown(f"""
                <div style="padding: 1rem; background-color: #f0f8f0; border-radius: 10px; border-left: 5px solid #2E7D32;">
                    <h4>{rec_color} {recommendation['recommendation']}</h4>
                    <p>{recommendation['reasoning']}</p>
                    <p><strong>Confidence Score:</strong> {recommendation['confidence_score']:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Future projections
                st.subheader("📈 Future Value Projections")
                
                proj_col1, proj_col2 = st.columns(2)
                with proj_col1:
                    st.metric("Next Year", f"₹{recommendation['next_year_projection']:,.0f}")
                with proj_col2:
                    st.metric("5-Year Projection", f"₹{recommendation['five_year_projection']:,.0f}")

def show_investment_analyzer():
    """Display investment opportunity analyzer"""
    st.header("💰 Investment Opportunity Analyzer")
    st.markdown("Evaluate whether a property at a specific price is a good investment opportunity.")
    
    # Initialize components
    portfolio_analyzer = PropertyPortfolioAnalyzer()
    predictor = FastRealEstatePredictor()
    
    # Load data for training
    data = load_database_data()
    if data is not None and not data.empty:
        predictor.train_model(data)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Property Under Consideration")
        
        # Property details
        city = st.selectbox("Target City", sorted(data['City'].unique()) if data is not None else ['Mumbai'], key="inv_city")
        district = st.selectbox("Target District", get_districts(data, city) if data is not None else ['Central'], key="inv_district")
        sub_district = st.selectbox("Target Sub District", get_sub_districts(data, city, district) if data is not None else ['Area 1'], key="inv_sub_district")
        
        area_sqft = st.number_input("Area (Sq Ft)", min_value=100, max_value=10000, value=1000, step=50, key="inv_area")
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1, key="inv_bhk")
        property_type = st.selectbox("Property Type", ['Apartment', 'Independent House', 'Builder Floor'], key="inv_type")
        furnishing = st.selectbox("Furnishing", ['Unfurnished', 'Semi-Furnished', 'Furnished'], index=1, key="inv_furnishing")
        
        st.subheader("Investment Parameters")
        asking_price = st.number_input("Asking Price (₹)", min_value=100000, max_value=50000000, value=5000000, step=100000, key="inv_price")
        
        analyze_investment_button = st.button("🎯 Analyze Investment", type="primary")
    
    with col2:
        if analyze_investment_button:
            with st.spinner("Analyzing investment opportunity..."):
                # Prepare target property data
                target_property = {
                    'city': city,
                    'district': district,
                    'sub_district': sub_district,
                    'area_sqft': area_sqft,
                    'bhk': bhk,
                    'property_type': property_type,
                    'furnishing': furnishing
                }
                
                # Analyze investment opportunity
                investment_analysis = portfolio_analyzer.analyze_investment_opportunity(
                    target_property, asking_price, predictor
                )
                
                # Display results
                st.subheader("💡 Investment Analysis")
                
                # Value comparison
                col_val1, col_val2 = st.columns(2)
                with col_val1:
                    st.metric("Asking Price", f"₹{investment_analysis['asking_price']:,.0f}")
                with col_val2:
                    st.metric(
                        "Market Value", 
                        f"₹{investment_analysis['predicted_market_value']:,.0f}",
                        f"₹{investment_analysis['value_gap']:,.0f}"
                    )
                
                # Value gap indicator
                if investment_analysis['value_gap_percent'] > 0:
                    gap_color = "🟢"
                    gap_text = f"Undervalued by {investment_analysis['value_gap_percent']:.1f}%"
                else:
                    gap_color = "🔴"
                    gap_text = f"Overvalued by {abs(investment_analysis['value_gap_percent']):.1f}%"
                
                st.markdown(f"**Value Assessment:** {gap_color} {gap_text}")
                
                # Investment recommendation
                st.subheader("🎯 Investment Recommendation")
                
                rec_color = {
                    "STRONG BUY": "🟢", "BUY": "🟢", 
                    "CONDITIONAL BUY": "🟡", "AVOID": "🟠", 
                    "STRONG AVOID": "🔴"
                }.get(investment_analysis['investment_recommendation'], "🟡")
                
                st.markdown(f"""
                <div style="padding: 1rem; background-color: #f0f8f0; border-radius: 10px; border-left: 5px solid #2E7D32;">
                    <h4>{rec_color} {investment_analysis['investment_recommendation']}</h4>
                    <p>{investment_analysis['reasoning']}</p>
                    <p><strong>Confidence Score:</strong> {investment_analysis['confidence_score']:.0f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # ROI projections
                st.subheader("📊 ROI Projections")
                
                roi_col1, roi_col2, roi_col3 = st.columns(3)
                with roi_col1:
                    st.metric("3-Year ROI", f"{investment_analysis['roi_projections']['three_year_roi']:.1f}%")
                with roi_col2:
                    st.metric("5-Year ROI", f"{investment_analysis['roi_projections']['five_year_roi']:.1f}%")
                with roi_col3:
                    st.metric("Annual Growth", f"{investment_analysis['roi_projections']['annual_growth_rate']:.1f}%")
                
                # Growth projections chart
                years = [0, 1, 3, 5]
                values = [
                    asking_price,
                    investment_analysis['growth_projections']['one_year'],
                    investment_analysis['growth_projections']['three_year'],
                    investment_analysis['growth_projections']['five_year']
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years, y=values,
                    mode='lines+markers',
                    name='Projected Value',
                    line=dict(color='#2E7D32', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Investment Growth Projection",
                    xaxis_title="Years",
                    yaxis_title="Property Value (₹)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()