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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E7D32 0%, #4CAF50 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .price-display {
        background: linear-gradient(135deg, #1976D2, #42A5F5);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #E8F5E8, #C8E6C9);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #FFF3E0, #FFCC02);
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #FFEBEE, #EF5350);
        border-left: 4px solid #F44336;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-section {
        background: #E3F2FD;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .sidebar-section {
        background: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
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
        <h1 style="margin: 0; font-size: 2.5rem;">AI Real Estate Intelligence Platform</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Professional Investment Analysis & Market Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 Portfolio Tracker", "💰 Investment Analyzer"])
    
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
    
    # Sidebar for property input
    with st.sidebar:
        st.markdown("### Property Configuration")
        
        # Location selection
        st.markdown("#### Location Details")
        city = st.selectbox("Select City", ["Mumbai", "Delhi", "Gurugram", "Noida", "Bangalore"])
        
        districts = get_districts(data, city)
        district = st.selectbox("Select District", districts)
        
        sub_districts = get_sub_districts(data, city, district)
        sub_district = st.selectbox("Select Sub-District", sub_districts)
        
        # Property details
        st.markdown("#### Property Specifications")
        area_sqft = st.number_input("Area (Square Feet)", min_value=100, max_value=10000, value=1000, step=50)
        bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])
        property_type = st.selectbox("Property Type", ["Apartment", "Villa", "House", "Studio"])
        furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"])
        
        # Predict button
        predict_button = st.button("🔮 Predict Property Price", type="primary", use_container_width=True)
    
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
            
            # Display results
            st.markdown("### Property Valuation Results")
            
            # Price display
            st.markdown(f"""
            <div class="price-display">
                ₹{predicted_price:,.0f}
                <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.8;">
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
                    <h4 style="margin-top: 0; color: {score_color};">Investment Analysis</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {score_color};">
                        {investment_score}/10
                    </div>
                    <p style="margin: 0.5rem 0 0 0;">{status_text}</p>
                    <p style="font-size: 0.9rem; margin: 0.5rem 0 0 0; color: #666;">
                        {recommendation}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # EMI Calculator
                st.markdown("#### EMI Calculator")
                loan_percentage = st.slider("Loan Amount (%)", 20, 90, 80)
                loan_amount = predicted_price * (loan_percentage / 100)
                interest_rate = st.slider("Interest Rate (%)", 6.0, 15.0, 8.5, 0.1)
                tenure_years = st.slider("Tenure (Years)", 5, 30, 20)
                
                emi_details = emi_calculator.calculate_emi(loan_amount, interest_rate, tenure_years)
                
                st.markdown(f"""
                <div class="info-section">
                    <h5 style="margin-top: 0;">Monthly EMI: ₹{emi_details['emi']:,.0f}</h5>
                    <p style="margin: 0.25rem 0;">Loan Amount: ₹{loan_amount:,.0f}</p>
                    <p style="margin: 0.25rem 0;">Total Interest: ₹{emi_details['total_interest']:,.0f}</p>
                    <p style="margin: 0.25rem 0;">Total Payment: ₹{emi_details['total_amount']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            

        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    else:
        # Default dashboard
        st.markdown("### Welcome to AI Real Estate Intelligence")
        
        # Show market insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = data['Price_INR'].mean()
            st.metric("Average Property Price", f"₹{avg_price:,.0f}")
        
        with col2:
            avg_sqft_price = data['Price_per_SqFt'].mean()
            st.metric("Average Price per Sq Ft", f"₹{avg_sqft_price:,.0f}")
        
        with col3:
            total_properties = len(data)
            st.metric("Total Properties in Database", f"{total_properties:,}")
        
        # City-wise analysis
        st.markdown("#### City-wise Property Distribution")
        city_counts = data['City'].value_counts()
        fig = px.bar(x=city_counts.index, y=city_counts.values, 
                     title="Properties by City", color=city_counts.values,
                     color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

def show_floating_chat_icon():
    """Display floating chat icon in bottom right corner"""
    # Initialize chat state
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    
    # Floating chat button CSS
    st.markdown("""
    <style>
    .floating-chat-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #2E7D32, #4CAF50);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
        color: white;
        font-size: 24px;
        text-decoration: none;
        border: none;
    }
    
    .floating-chat-btn:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(46, 125, 50, 0.4);
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