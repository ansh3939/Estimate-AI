import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from ml_model import RealEstatePredictor
from investment_analyzer import InvestmentAnalyzer
from emi_calculator import EMICalculator
from advanced_ml_models import AdvancedRealEstatePredictor
from live_data_scraper import LivePropertyDataScraper
from market_analysis import ComparativeMarketAnalyzer
from database import db_manager
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

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
    
    .investment-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #C8E6C9;
        margin: 1rem 0;
    }
    
    .price-display {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #2E7D32;
    }
    
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #C8E6C9;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #C8E6C9;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .info-section {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFF8E1 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #FFE0B2;
        margin: 1rem 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
        border: 1px solid #C8E6C9;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFF8E1 100%);
        border: 1px solid #FFE0B2;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .error-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FCE4EC 100%);
        border: 1px solid #F8BBD9;
        border-radius: 10px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_components():
    data_processor = DataProcessor()
    predictor = RealEstatePredictor()
    advanced_predictor = AdvancedRealEstatePredictor()
    investment_analyzer = InvestmentAnalyzer()
    emi_calculator = EMICalculator()
    market_analyzer = ComparativeMarketAnalyzer()
    live_scraper = LivePropertyDataScraper()
    
    # Load and process data
    data_processor.load_all_data()
    combined_data = data_processor.get_combined_data()
    
    # Train basic model
    predictor.train_model(combined_data)
    
    # Train advanced models
    if combined_data is not None and not combined_data.empty:
        advanced_predictor.train_models(combined_data)
    
    return data_processor, predictor, advanced_predictor, investment_analyzer, emi_calculator, market_analyzer, live_scraper

def get_session_id():
    """Get or create session ID for user tracking"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def main():
    # Initialize session
    session_id = get_session_id()
    
    # Professional Header with Database Status
    try:
        analytics_data = db_manager.get_analytics_data()
        db_status = "✅ Connected"
        total_properties = analytics_data.get('total_properties', 0)
        total_predictions = analytics_data.get('total_predictions', 0)
    except:
        db_status = "⚠️ Offline"
        total_properties = 0
        total_predictions = 0
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">AI Real Estate Price Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Professional Investment Analysis & Market Intelligence Platform
        </p>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;">
            Database: {db_status} | Properties: {total_properties:,} | Predictions Made: {total_predictions:,}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load components with enhanced error handling
    try:
        data_processor, predictor, advanced_predictor, investment_analyzer, emi_calculator, market_analyzer, live_scraper = load_components()
    except Exception as e:
        st.error(f"Error loading application components: {str(e)}")
        
        # Try to load data from database only
        try:
            st.info("Attempting to load data from database...")
            db_data = db_manager.get_properties_from_db()
            if not db_data.empty:
                # Create minimal components for database-only mode
                data_processor = DataProcessor()
                data_processor.combined_data = db_data
                predictor = RealEstatePredictor()
                advanced_predictor = AdvancedRealEstatePredictor()
                investment_analyzer = InvestmentAnalyzer()
                emi_calculator = EMICalculator()
                market_analyzer = ComparativeMarketAnalyzer()
                live_scraper = LivePropertyDataScraper()
                
                # Train models with database data
                predictor.train_model(db_data)
                advanced_predictor.train_models(db_data)
                
                st.success("Successfully loaded from database!")
            else:
                st.error("No data available in database. Please contact support.")
                st.stop()
        except Exception as db_error:
            st.error(f"Database connection failed: {str(db_error)}")
            st.error("Unable to load application. Please refresh the page or contact support.")
            st.stop()
    
    # Enhanced Sidebar with Professional Styling
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style="color: #2E7D32; margin-top: 0;">Property Configuration</h3>
            <p style="color: #666; font-size: 0.9rem;">Configure your property details for accurate prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
        # Location Selection Section
        st.markdown("#### 📍 Location Details")
        city = st.selectbox(
            "Select City",
            ["Mumbai", "Delhi", "Gurugram", "Noida", "Bangalore"],
            help="Choose the city where you want to buy property"
        )
        
        # Get districts and sub-districts based on city
        districts = data_processor.get_districts(city)
        district = st.selectbox("Select District", districts, help="Select the district within the city")
        
        sub_districts = data_processor.get_sub_districts(city, district)
        sub_district = st.selectbox("Select Sub-District", sub_districts, help="Choose the specific area")
        
        st.markdown("---")
        
        # Property Configuration Section
        st.markdown("#### 🏢 Property Specifications")
        
        col1, col2 = st.columns(2)
        with col1:
            area_sqft = st.number_input(
                "Area (Sq Ft)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=50,
                help="Total carpet area in square feet"
            )
            
            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Builder Floor", "Independent House"],
                help="Type of property construction"
            )
        
        with col2:
            bhk = st.selectbox(
                "BHK Configuration",
                [1, 2, 3, 4, 5, 6],
                index=1,
                help="Number of bedrooms"
            )
            
            furnishing = st.selectbox(
                "Furnishing Status",
                ["Unfurnished", "Semi-Furnished", "Fully Furnished"],
                help="Current furnishing condition"
            )
        
        st.markdown("---")
        
        # Advanced ML Model Selection
        st.markdown("#### 🤖 AI Model Selection")
        model_choice = st.selectbox(
            "Choose Prediction Model",
            ["Advanced Ensemble (Recommended)", "Decision Tree", "Random Forest", "XGBoost"],
            help="Select the AI model for price prediction"
        )
        
        # Live Data Integration Toggle
        use_live_data = st.checkbox(
            "Include Live Market Data",
            value=False,
            help="Integrate real-time property data from online sources"
        )
        
        st.markdown("---")
    
        # Prediction Button with Enhanced Styling
        predict_button = st.button(
            "🔍 Analyze Property Value", 
            type="primary",
            use_container_width=True,
            help="Generate AI-powered price prediction and investment analysis"
        )
    
    # Main Content Area
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
            # Handle live data integration
            if use_live_data:
                with st.spinner("Fetching live market data..."):
                    live_data = live_scraper.get_live_market_data([city])
                    if not live_data.empty:
                        st.success(f"Integrated {len(live_data)} live properties from market sources")
            
            # Make prediction based on model choice
            if model_choice == "Advanced Ensemble (Recommended)":
                predicted_price, all_predictions = advanced_predictor.predict(input_data)
                
                # Show model comparison
                st.markdown("#### 🎯 AI Model Predictions Comparison")
                model_col1, model_col2, model_col3 = st.columns(3)
                
                with model_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Decision Tree</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{all_predictions.get('decision_tree', 0):,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with model_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Random Forest</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{all_predictions.get('random_forest', 0):,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with model_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">XGBoost</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{all_predictions.get('xgboost', 0):,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Best model indicator
                best_model = advanced_predictor.best_model_name
                if best_model:
                    model_display_name = best_model.replace('_', ' ').title()
                    st.info(f"🏆 Best performing model: **{model_display_name}** (Selected for final prediction)")
                else:
                    st.info("🏆 Using ensemble prediction from multiple AI models")
                
            else:
                # Use basic predictor
                predicted_price = predictor.predict(input_data)
            
            # Analyze investment
            investment_score, recommendation = investment_analyzer.analyze(
                input_data, predicted_price
            )
            
            # Save prediction to database
            try:
                prediction_result = {
                    'predicted_price': predicted_price,
                    'investment_score': investment_score,
                    'model_used': model_choice,
                    'all_predictions': confidence_scores if model_choice == "Advanced Ensemble (Recommended)" and 'confidence_scores' in locals() else {}
                }
                prediction_id = db_manager.save_prediction(session_id, input_data, prediction_result)
            except Exception as e:
                pass  # Continue without database if error occurs
            
            # Enhanced Results Display
            st.markdown("### 📊 Property Valuation Results")
            
            # Price Display with Professional Card
            st.markdown(f"""
            <div class="price-display">
                ₹{predicted_price:,.0f}
                <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.8;">
                    ₹{predicted_price/area_sqft:,.0f} per Sq Ft
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Main Results in Two Columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Investment Analysis Card
                if investment_score >= 7:
                    card_class = "success-card"
                    score_color = "#2E7D32"
                    status_text = "Excellent Investment Opportunity"
                elif investment_score >= 5:
                    card_class = "warning-card"
                    score_color = "#F57C00"
                    status_text = "Good Investment Potential"
                else:
                    card_class = "error-card"
                    score_color = "#C62828"
                    status_text = "High Risk Investment"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4 style="margin-top: 0; color: {score_color};">Investment Analysis</h4>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {score_color};">
                        {status_text}
                    </div>
                    <div style="font-size: 1.2rem; margin-top: 0.5rem;">
                        Score: {investment_score}/10
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Investment Details
                st.markdown("#### 📈 Investment Insights")
                st.markdown(recommendation)
            
            with col2:
                # EMI Calculator Section with Professional Styling
                st.markdown("""
                <div class="info-section">
                    <h4 style="margin-top: 0; color: #1976D2;">💰 EMI Calculator</h4>
                    <p style="margin-bottom: 1rem; color: #666;">Calculate your monthly payment details</p>
                </div>
                """, unsafe_allow_html=True)
                
                # EMI Input Parameters
                col2a, col2b = st.columns(2)
                with col2a:
                    loan_amount = st.number_input(
                        "Loan Amount (₹)",
                        min_value=100000,
                        max_value=int(predicted_price * 0.9),
                        value=int(predicted_price * 0.8),
                        step=100000,
                        help="Total loan amount from bank"
                    )
                    
                    tenure_years = st.number_input(
                        "Loan Tenure (Years)",
                        min_value=1,
                        max_value=30,
                        value=20,
                        help="Loan repayment period"
                    )
                
                with col2b:
                    interest_rate = st.number_input(
                        "Interest Rate (%)",
                        min_value=5.0,
                        max_value=15.0,
                        value=8.5,
                        step=0.1,
                        help="Annual interest rate from bank"
                    )
                    
                    down_payment = predicted_price - loan_amount
                    st.metric("Down Payment", f"₹{down_payment:,.0f}")
                
                # Calculate EMI
                emi_details = emi_calculator.calculate_emi(
                    loan_amount, interest_rate, tenure_years
                )
                
                # EMI Results in Professional Cards
                st.markdown("#### 📊 Payment Breakdown")
                
                emi_col1, emi_col2, emi_col3 = st.columns(3)
                with emi_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Monthly EMI</h5>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #2E7D32;">
                            ₹{emi_details['emi']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with emi_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Total Interest</h5>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #F57C00;">
                            ₹{emi_details['total_interest']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with emi_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Total Amount</h5>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #7B1FA2;">
                            ₹{emi_details['total_amount']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced EMI Breakdown Chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Principal Amount', 'Interest Amount'],
                        values=[loan_amount, emi_details['total_interest']],
                        hole=0.5,
                        marker_colors=['#2E7D32', '#F57C00'],
                        textinfo='label+percent',
                        textfont=dict(size=12)
                    )
                ])
                fig.update_layout(
                    title={
                        'text': "Loan Composition",
                        'x': 0.5,
                        'font': {'size': 16, 'color': '#1A1A1A'}
                    },
                    height=300,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced Market Analysis Section
            st.markdown("---")
            st.markdown("### 📈 Market Analysis & Trends")
            
            market_data = data_processor.get_market_analysis(city, district)
            
            if not market_data.empty:
                # Market Overview Cards
                st.markdown("#### Market Overview")
                market_col1, market_col2, market_col3, market_col4 = st.columns(4)
                
                with market_col1:
                    avg_price = market_data['Price_INR'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Average Price</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{avg_price:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with market_col2:
                    avg_price_sqft = market_data['Price_per_SqFt'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Avg Price/SqFt</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{avg_price_sqft:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with market_col3:
                    total_properties = len(market_data)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Properties</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            {total_properties}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with market_col4:
                    price_range = market_data['Price_INR'].max() - market_data['Price_INR'].min()
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5 style="margin: 0; color: #1976D2;">Price Range</h5>
                        <div style="font-size: 1.3rem; font-weight: bold; color: #2E7D32;">
                            ₹{price_range:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced Charts Section
                st.markdown("#### Market Trends & Analysis")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Price trends by property type
                    fig = px.box(
                        market_data,
                        x='Property_Type',
                        y='Price_INR',
                        title="Price Distribution by Property Type",
                        color='Property_Type',
                        color_discrete_sequence=['#2E7D32', '#1976D2', '#F57C00']
                    )
                    fig.update_layout(
                        title_font_size=16,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with chart_col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Average price by BHK
                    avg_price_bhk = market_data.groupby('BHK')['Price_INR'].mean().reset_index()
                    fig = px.bar(
                        avg_price_bhk,
                        x='BHK',
                        y='Price_INR',
                        title="Average Price by BHK Configuration",
                        color='Price_INR',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(
                        title_font_size=16,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Area-wise pricing analysis
                if len(market_data['Sub_District'].unique()) > 1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    avg_price_area = market_data.groupby('Sub_District')['Price_per_SqFt'].mean()
                    sorted_areas = avg_price_area.nlargest(10)
                    fig = px.bar(
                        x=sorted_areas.values,
                        y=sorted_areas.index,
                        orientation='h',
                        title="Top 10 Sub-Districts by Price per Sq Ft",
                        color=sorted_areas.values,
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(
                        title_font_size=16,
                        height=400,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Furnishing Analysis
                furnishing_analysis = market_data.groupby('Furnishing')['Price_INR'].mean().reset_index()
                if len(furnishing_analysis) > 1:
                    furnish_col1, furnish_col2 = st.columns(2)
                    
                    with furnish_col1:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        fig = px.pie(
                            furnishing_analysis,
                            values='Price_INR',
                            names='Furnishing',
                            title="Average Price by Furnishing Status",
                            color_discrete_sequence=['#2E7D32', '#1976D2', '#F57C00']
                        )
                        fig.update_layout(title_font_size=16)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with furnish_col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        # Area vs Price scatter plot
                        fig = px.scatter(
                            market_data.sample(min(100, len(market_data))),
                            x='Area_SqFt',
                            y='Price_INR',
                            color='BHK',
                            size='Price_per_SqFt',
                            title="Area vs Price Analysis",
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            title_font_size=16,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced Market Analysis Section
            st.markdown("---")
            st.markdown("### 📈 Advanced Market Intelligence")
            
            # Create tabs for different analysis
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏙️ Comparative Analysis", 
                "📊 Historical Trends", 
                "🎯 Investment Heatmap", 
                "🔮 Price Forecasting"
            ])
            
            with tab1:
                st.markdown("#### Multi-City Market Comparison")
                combined_data = data_processor.get_combined_data()
                if combined_data is not None and not combined_data.empty:
                    comparison_charts = market_analyzer.create_comparative_analysis(combined_data)
                    
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        if 'price_comparison' in comparison_charts:
                            st.plotly_chart(comparison_charts['price_comparison'], use_container_width=True)
                        
                        if 'property_type_analysis' in comparison_charts:
                            st.plotly_chart(comparison_charts['property_type_analysis'], use_container_width=True)
                    
                    with comp_col2:
                        if 'price_sqft_distribution' in comparison_charts:
                            st.plotly_chart(comparison_charts['price_sqft_distribution'], use_container_width=True)
                        
                        if 'bhk_analysis' in comparison_charts:
                            st.plotly_chart(comparison_charts['bhk_analysis'], use_container_width=True)
                    
                    if 'area_price_scatter' in comparison_charts:
                        st.plotly_chart(comparison_charts['area_price_scatter'], use_container_width=True)
            
            with tab2:
                st.markdown("#### Historical Price Trends & Patterns")
                if combined_data is not None and not combined_data.empty:
                    with st.spinner("Generating historical trends..."):
                        historical_data = market_analyzer.generate_historical_trends(combined_data, years_back=5)
                        
                        if not historical_data.empty:
                            trend_chart = market_analyzer.create_trend_analysis(historical_data)
                            st.plotly_chart(trend_chart, use_container_width=True)
                            
                            # Historical insights
                            st.markdown("##### Key Historical Insights")
                            trend_insights = market_analyzer.calculate_appreciation_trends(combined_data)
                            
                            insight_cols = st.columns(len(trend_insights))
                            for i, (city, trends) in enumerate(trend_insights.items()):
                                with insight_cols[i]:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <h5 style="margin: 0; color: #1976D2;">{city}</h5>
                                        <div style="font-size: 1.1rem; color: #2E7D32;">
                                            Growth: {trends['annual_growth_rate']*100:.1f}%
                                        </div>
                                        <div style="font-size: 0.9rem; color: #666;">
                                            {trends['market_cycle'].title()} Market
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
            
            with tab3:
                st.markdown("#### Investment Attractiveness Analysis")
                if combined_data is not None and not combined_data.empty:
                    heatmap_chart = market_analyzer.create_investment_heatmap(combined_data)
                    st.plotly_chart(heatmap_chart, use_container_width=True)
                    
                    # Investment recommendations
                    market_insights = market_analyzer.generate_market_insights(combined_data)
                    
                    st.markdown("##### Investment Recommendations")
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.markdown(f"""
                        <div class="success-card">
                            <h5 style="margin-top: 0; color: #2E7D32;">Best Investment City</h5>
                            <p><strong>{market_insights.get('best_investment_city', 'N/A')}</strong></p>
                            <p style="font-size: 0.9rem; color: #666;">
                                Highest growth potential and market stability
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="info-section">
                            <h5 style="margin-top: 0; color: #1976D2;">Most Affordable</h5>
                            <p><strong>{market_insights.get('most_affordable_city', 'N/A')}</strong></p>
                            <p style="font-size: 0.9rem; color: #666;">
                                Entry-level investment opportunity
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with rec_col2:
                        st.markdown(f"""
                        <div class="warning-card">
                            <h5 style="margin-top: 0; color: #F57C00;">Premium Market</h5>
                            <p><strong>{market_insights.get('most_expensive_city', 'N/A')}</strong></p>
                            <p style="font-size: 0.9rem; color: #666;">
                                Highest property values
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="info-section">
                            <h5 style="margin-top: 0; color: #1976D2;">Best Value BHK</h5>
                            <p><strong>{market_insights.get('most_value_bhk', 'N/A')} BHK</strong></p>
                            <p style="font-size: 0.9rem; color: #666;">
                                Optimal price per square foot
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab4:
                st.markdown("#### 5-Year Price Appreciation Forecast")
                if combined_data is not None and not combined_data.empty:
                    forecast_chart = market_analyzer.create_appreciation_forecast(combined_data)
                    st.plotly_chart(forecast_chart, use_container_width=True)
                    
                    # Forecast summary
                    appreciation_trends = market_analyzer.calculate_appreciation_trends(combined_data)
                    
                    st.markdown("##### Projected Returns (5 Years)")
                    forecast_cols = st.columns(len(appreciation_trends))
                    
                    for i, (city, trends) in enumerate(appreciation_trends.items()):
                        with forecast_cols[i]:
                            current_price = trends['current_avg_price']
                            projected_5yr = trends['projected_5_year']
                            total_return = ((projected_5yr - current_price) / current_price) * 100
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5 style="margin: 0; color: #1976D2;">{city}</h5>
                                <div style="font-size: 1.2rem; font-weight: bold; color: #2E7D32;">
                                    +{total_return:.1f}%
                                </div>
                                <div style="font-size: 0.9rem; color: #666;">
                                    ₹{current_price:,.0f} → ₹{projected_5yr:,.0f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Model Performance Section
            if model_choice == "Advanced Ensemble (Recommended)":
                st.markdown("---")
                st.markdown("### 🎯 AI Model Performance Analysis")
                
                model_performance = advanced_predictor.get_model_comparison()
                if not model_performance.empty:
                    perf_col1, perf_col2 = st.columns([2, 1])
                    
                    with perf_col1:
                        st.dataframe(
                            model_performance,
                            use_container_width=True,
                            hide_index=True
                        )
                    
                    with perf_col2:
                        feature_importance = advanced_predictor.get_feature_importance()
                        if feature_importance:
                            st.markdown("#### Feature Importance")
                            
                            # Create feature importance chart
                            features = list(feature_importance.keys())[:6]  # Top 6 features
                            importance_values = [feature_importance[f] for f in features]
                            
                            fig_importance = go.Figure(data=[
                                go.Bar(
                                    y=features,
                                    x=importance_values,
                                    orientation='h',
                                    marker_color='#2E7D32'
                                )
                            ])
                            fig_importance.update_layout(
                                title="Top Features Impact",
                                height=300,
                                showlegend=False
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    else:
        # Default Dashboard when no prediction is made
        st.markdown("### 🏠 Welcome to AI Real Estate Intelligence Platform")
        
        # User Analytics Dashboard
        st.markdown("---")
        
        # Create tabs for different sections
        dash_tab1, dash_tab2, dash_tab3 = st.tabs([
            "📊 Your Analytics", 
            "📈 Market Overview", 
            "🔍 Recent Activity"
        ])
        
        with dash_tab1:
            st.markdown("#### Personal Analytics Dashboard")
            
            # Get user prediction history
            try:
                user_history = db_manager.get_prediction_history(session_id, limit=20)
                
                if user_history:
                    st.markdown(f"**Total Predictions Made:** {len(user_history)}")
                    
                    # Create summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_predictions = len(user_history)
                    avg_price = sum(pred['predicted_price'] for pred in user_history) / total_predictions
                    avg_investment_score = sum(pred.get('investment_score', 0) for pred in user_history if pred.get('investment_score')) / max(1, len([p for p in user_history if p.get('investment_score')]))
                    favorite_city = max(set([pred['city'] for pred in user_history]), key=[pred['city'] for pred in user_history].count)
                    
                    with col1:
                        st.metric("Total Predictions", total_predictions)
                    
                    with col2:
                        st.metric("Average Price", f"₹{avg_price:,.0f}")
                    
                    with col3:
                        st.metric("Avg Investment Score", f"{avg_investment_score:.1f}/10")
                    
                    with col4:
                        st.metric("Favorite City", favorite_city)
                    
                    # Recent predictions table
                    st.markdown("##### Recent Predictions")
                    history_df = pd.DataFrame(user_history[:10])
                    history_df['predicted_price'] = history_df['predicted_price'].apply(lambda x: f"₹{x:,.0f}")
                    history_df = history_df[['city', 'district', 'bhk', 'area_sqft', 'predicted_price', 'investment_score', 'created_at']]
                    history_df.columns = ['City', 'District', 'BHK', 'Area (SqFt)', 'Predicted Price', 'Investment Score', 'Date']
                    st.dataframe(history_df, use_container_width=True)
                    
                else:
                    st.info("No predictions made yet. Start by making your first property prediction above!")
                    
            except Exception as e:
                st.warning("Analytics temporarily unavailable")
        
        with dash_tab2:
            st.markdown("#### Market Overview")
            
            try:
                analytics_data = db_manager.get_analytics_data()
                
                # Platform statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Properties", f"{analytics_data.get('total_properties', 0):,}")
                
                with col2:
                    st.metric("Total Predictions", f"{analytics_data.get('total_predictions', 0):,}")
                
                with col3:
                    st.metric("Active Users", analytics_data.get('active_users', 0))
                
                # Popular cities chart
                if analytics_data.get('popular_cities'):
                    st.markdown("##### Most Popular Cities")
                    cities_data = analytics_data['popular_cities']
                    cities_df = pd.DataFrame(cities_data, columns=['City', 'Predictions'])
                    
                    fig = px.bar(
                        cities_df, 
                        x='City', 
                        y='Predictions',
                        title="Prediction Activity by City",
                        color='Predictions',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning("Market data temporarily unavailable")
        
        with dash_tab3:
            st.markdown("#### Recent Platform Activity")
            
            # Show sample market insights
            combined_data = data_processor.get_combined_data()
            if combined_data is not None and not combined_data.empty:
                market_insights = market_analyzer.generate_market_insights(combined_data)
                
                insight_col1, insight_col2 = st.columns(2)
                
                with insight_col1:
                    st.markdown(f"""
                    <div class="info-section">
                        <h5 style="margin-top: 0; color: #2E7D32;">Top Investment City</h5>
                        <p><strong>{market_insights.get('best_investment_city', 'Mumbai')}</strong></p>
                        <p style="font-size: 0.9rem; color: #666;">Highest growth potential</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="warning-card">
                        <h5 style="margin-top: 0; color: #F57C00;">Most Expensive</h5>
                        <p><strong>{market_insights.get('most_expensive_city', 'Mumbai')}</strong></p>
                        <p style="font-size: 0.9rem; color: #666;">Premium market</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with insight_col2:
                    st.markdown(f"""
                    <div class="success-card">
                        <h5 style="margin-top: 0; color: #1976D2;">Most Affordable</h5>
                        <p><strong>{market_insights.get('most_affordable_city', 'Noida')}</strong></p>
                        <p style="font-size: 0.9rem; color: #666;">Entry-level opportunity</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="info-section">
                        <h5 style="margin-top: 0; color: #2E7D32;">Best Value BHK</h5>
                        <p><strong>{market_insights.get('most_value_bhk', 2)} BHK</strong></p>
                        <p style="font-size: 0.9rem; color: #666;">Optimal investment size</p>
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-section">
            <h4 style="margin-top: 0; color: #1976D2;">How to Use This Platform</h4>
            <ol style="color: #666;">
                <li><strong>Select Location:</strong> Choose your preferred city, district, and sub-district</li>
                <li><strong>Configure Property:</strong> Set area, BHK, property type, and furnishing details</li>
                <li><strong>Analyze:</strong> Click "Analyze Property Value" to get AI-powered predictions</li>
                <li><strong>Review Results:</strong> Get price predictions, investment analysis, and EMI calculations</li>
                <li><strong>Market Insights:</strong> Explore comprehensive market trends and comparisons</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall Market Insights Dashboard
        st.markdown("---")
        st.markdown("### 📊 Overall Market Insights")
        insights_data = data_processor.get_market_insights()
        
        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        
        with insight_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin: 0; color: #1976D2;">Total Properties</h5>
                <div style="font-size: 1.8rem; font-weight: bold; color: #2E7D32;">
                    {insights_data['total_properties']:,}
                </div>
                <div style="font-size: 0.9rem; color: #666;">Across all cities</div>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin: 0; color: #1976D2;">Average Price</h5>
                <div style="font-size: 1.8rem; font-weight: bold; color: #2E7D32;">
                    ₹{insights_data['avg_price']:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #666;">National average</div>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col3:
            st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin: 0; color: #1976D2;">Price Range</h5>
                <div style="font-size: 1.8rem; font-weight: bold; color: #2E7D32;">
                    ₹{insights_data['price_range']:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #666;">Market spread</div>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col4:
            st.markdown(f"""
            <div class="metric-card">
                <h5 style="margin: 0; color: #1976D2;">Avg Price/SqFt</h5>
                <div style="font-size: 1.8rem; font-weight: bold; color: #2E7D32;">
                    ₹{insights_data['avg_price_per_sqft']:,.0f}
                </div>
                <div style="font-size: 0.9rem; color: #666;">Per square foot</div>
            </div>
            """, unsafe_allow_html=True)
        
        # City-wise Overview
        st.markdown("#### 🌆 City-wise Market Overview")
        if data_processor.combined_data is not None:
            city_overview = data_processor.combined_data.groupby('City').agg({
                'Price_INR': ['mean', 'count'],
                'Price_per_SqFt': 'mean',
                'Area_SqFt': 'mean'
            }).round(0)
            
            city_overview.columns = ['Avg Price', 'Properties', 'Avg Price/SqFt', 'Avg Area']
            city_overview = city_overview.reset_index()
            
            # Create an interactive city comparison chart
            overview_col1, overview_col2 = st.columns(2)
            
            with overview_col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = px.bar(
                    city_overview,
                    x='City',
                    y='Avg Price',
                    title="Average Property Prices by City",
                    color='Avg Price',
                    color_continuous_scale='Greens',
                    text='Properties'
                )
                fig.update_layout(
                    title_font_size=16,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_traces(texttemplate='%{text} properties', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with overview_col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = px.scatter(
                    city_overview,
                    x='Avg Area',
                    y='Avg Price/SqFt',
                    size='Properties',
                    color='City',
                    title="Price per SqFt vs Average Area by City",
                    hover_data=['Avg Price']
                )
                fig.update_layout(
                    title_font_size=16,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Property Type Distribution
            st.markdown("#### Property Type Distribution")
            if 'Property_Type' in data_processor.combined_data.columns:
                prop_type_series = data_processor.combined_data['Property_Type'].value_counts()
                prop_type_dist = pd.DataFrame({
                    'Property_Type': prop_type_series.index,
                    'Count': prop_type_series.values
                })
                
                type_col1, type_col2 = st.columns(2)
                
                with type_col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = px.pie(
                        prop_type_dist,
                        values='Count',
                        names='Property_Type',
                        title="Property Type Distribution",
                        color_discrete_sequence=['#2E7D32', '#1976D2', '#F57C00']
                    )
                    fig.update_layout(title_font_size=16)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with type_col2:
                    # BHK Distribution
                    if 'BHK' in data_processor.combined_data.columns:
                        bhk_series = data_processor.combined_data['BHK'].value_counts()
                        bhk_series = bhk_series.sort_index()
                        bhk_dist = pd.DataFrame({
                            'BHK': bhk_series.index,
                            'Count': bhk_series.values
                        })
                        
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        fig = px.bar(
                            bhk_dist,
                            x='BHK',
                            y='Count',
                            title="BHK Configuration Distribution",
                            color='Count',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(
                            title_font_size=16,
                            showlegend=False,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
