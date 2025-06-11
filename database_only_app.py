import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from database import db_manager
from ml_model import RealEstatePredictor
from advanced_ml_models import AdvancedRealEstatePredictor
from investment_analyzer import InvestmentAnalyzer
from emi_calculator import EMICalculator
from market_analysis import ComparativeMarketAnalyzer
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
    
    # Database status header
    try:
        analytics_data = db_manager.get_analytics_data()
        db_status = "Connected"
        total_properties = analytics_data.get('total_properties', 0)
        total_predictions = analytics_data.get('total_predictions', 0)
    except:
        db_status = "Offline"
        total_properties = 0
        total_predictions = 0
    
    st.markdown(f"""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">AI Real Estate Price Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Professional Investment Analysis & Market Intelligence Platform
        </p>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.7;">
            Database: {db_status} | Properties: {total_properties:,} | Predictions: {total_predictions:,}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data from database
    data = load_database_data()
    
    if data.empty:
        st.error("Unable to load data from database. Please contact support.")
        st.stop()
    
    # Initialize models
    try:
        predictor = RealEstatePredictor()
        advanced_predictor = AdvancedRealEstatePredictor()
        investment_analyzer = InvestmentAnalyzer()
        emi_calculator = EMICalculator()
        market_analyzer = ComparativeMarketAnalyzer()
        
        # Train models with database data
        predictor.train_model(data)
        advanced_predictor.train_models(data)
        
        st.success(f"Models trained successfully with {len(data)} properties")
        
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
        
        # Model selection
        st.markdown("#### AI Model Selection")
        model_choice = st.selectbox(
            "Choose Prediction Model",
            ["Advanced Ensemble (Recommended)", "Basic Decision Tree"]
        )
        
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
            # Make prediction based on model choice
            if model_choice == "Advanced Ensemble (Recommended)":
                predicted_price, confidence_scores = advanced_predictor.predict(input_data)
            else:
                predicted_price = predictor.predict(input_data)
                confidence_scores = None
            
            # Investment analysis
            investment_score, recommendation = investment_analyzer.analyze(input_data, predicted_price)
            
            # Save prediction to database
            try:
                prediction_result = {
                    'predicted_price': predicted_price,
                    'investment_score': investment_score,
                    'model_used': model_choice,
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
                    <p style="margin: 0.25rem 0;">Total Payment: ₹{emi_details['total_payment']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model confidence (for advanced model)
            if model_choice == "Advanced Ensemble (Recommended)" and confidence_scores:
                st.markdown("#### Model Confidence Analysis")
                conf_col1, conf_col2 = st.columns([2, 1])
                
                with conf_col1:
                    # Confidence chart
                    models = list(confidence_scores.keys())
                    predictions = list(confidence_scores.values())
                    
                    fig = go.Figure(data=[
                        go.Bar(x=models, y=predictions, marker_color='#2E7D32')
                    ])
                    fig.update_layout(
                        title="Model Predictions Comparison",
                        xaxis_title="ML Models",
                        yaxis_title="Predicted Price (₹)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with conf_col2:
                    # Best model info
                    best_model = advanced_predictor.best_model_name
                    if best_model:
                        model_display_name = best_model.replace('_', ' ').title()
                        st.info(f"Best Model: **{model_display_name}**")
                    
                    # Model performance
                    model_performance = advanced_predictor.get_model_comparison()
                    if not model_performance.empty:
                        st.dataframe(model_performance, use_container_width=True)
        
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

if __name__ == "__main__":
    main()