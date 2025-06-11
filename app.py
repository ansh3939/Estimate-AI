import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_processor import DataProcessor
from ml_model import RealEstatePredictor
from investment_analyzer import InvestmentAnalyzer
from emi_calculator import EMICalculator
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def load_components():
    data_processor = DataProcessor()
    predictor = RealEstatePredictor()
    investment_analyzer = InvestmentAnalyzer()
    emi_calculator = EMICalculator()
    
    # Load and process data
    data_processor.load_all_data()
    predictor.train_model(data_processor.get_combined_data())
    
    return data_processor, predictor, investment_analyzer, emi_calculator

def main():
    st.title("AI Real Estate Price Predictor & Investment Analyzer")
    st.markdown("---")
    
    # Load components
    try:
        data_processor, predictor, investment_analyzer, emi_calculator = load_components()
    except Exception as e:
        st.error(f"Error loading application components: {str(e)}")
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.header("Property Details")
    
    # Location selection
    city = st.sidebar.selectbox(
        "Select City",
        ["Mumbai", "Delhi", "Gurugram", "Noida", "Bangalore"]
    )
    
    # Get districts and sub-districts based on city
    districts = data_processor.get_districts(city)
    district = st.sidebar.selectbox("Select District", districts)
    
    sub_districts = data_processor.get_sub_districts(city, district)
    sub_district = st.sidebar.selectbox("Select Sub-District", sub_districts)
    
    # Property details
    area_sqft = st.sidebar.number_input(
        "Area (Sq Ft)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=50
    )
    
    bhk = st.sidebar.selectbox(
        "BHK",
        [1, 2, 3, 4, 5, 6]
    )
    
    property_type = st.sidebar.selectbox(
        "Property Type",
        ["Apartment", "Builder Floor", "Independent House"]
    )
    
    furnishing = st.sidebar.selectbox(
        "Furnishing",
        ["Unfurnished", "Semi-Furnished", "Fully Furnished"]
    )
    
    # Prediction button
    if st.sidebar.button("Predict Price", type="primary"):
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
            # Make prediction
            predicted_price = predictor.predict(input_data)
            
            # Analyze investment
            investment_score, recommendation = investment_analyzer.analyze(
                input_data, predicted_price
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price Prediction")
                st.metric(
                    "Predicted Price",
                    f"₹{predicted_price:,.0f}",
                    delta=f"₹{predicted_price/area_sqft:,.0f} per Sq Ft"
                )
                
                st.subheader("Investment Analysis")
                if investment_score >= 7:
                    st.success(f"**Excellent Investment** (Score: {investment_score}/10)")
                elif investment_score >= 5:
                    st.warning(f"**Good Investment** (Score: {investment_score}/10)")
                else:
                    st.error(f"**Risky Investment** (Score: {investment_score}/10)")
                
                st.write(recommendation)
            
            with col2:
                st.subheader("EMI Calculator")
                loan_amount = st.number_input(
                    "Loan Amount (₹)",
                    min_value=100000,
                    max_value=int(predicted_price * 0.9),
                    value=int(predicted_price * 0.8)
                )
                
                interest_rate = st.number_input(
                    "Interest Rate (%)",
                    min_value=5.0,
                    max_value=15.0,
                    value=8.5,
                    step=0.1
                )
                
                tenure_years = st.number_input(
                    "Loan Tenure (Years)",
                    min_value=1,
                    max_value=30,
                    value=20
                )
                
                # Calculate EMI
                emi_details = emi_calculator.calculate_emi(
                    loan_amount, interest_rate, tenure_years
                )
                
                st.metric("Monthly EMI", f"₹{emi_details['emi']:,.0f}")
                st.metric("Total Interest", f"₹{emi_details['total_interest']:,.0f}")
                st.metric("Total Amount", f"₹{emi_details['total_amount']:,.0f}")
                
                # EMI Breakdown Chart
                fig = go.Figure(data=[
                    go.Pie(
                        labels=['Principal', 'Interest'],
                        values=[loan_amount, emi_details['total_interest']],
                        hole=0.4
                    )
                ])
                fig.update_layout(title="Loan Breakdown", height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Market Analysis
            st.subheader("Market Analysis")
            market_data = data_processor.get_market_analysis(city, district)
            
            if not market_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Price trends by property type
                    fig = px.box(
                        market_data,
                        x='Property_Type',
                        y='Price_INR',
                        title="Price Distribution by Property Type"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Average price by BHK
                    avg_price_bhk = market_data.groupby('BHK')['Price_INR'].mean().reset_index()
                    fig = px.bar(
                        avg_price_bhk,
                        x='BHK',
                        y='Price_INR',
                        title="Average Price by BHK"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Area-wise pricing
                if len(market_data['Sub_District'].unique()) > 1:
                    avg_price_area = market_data.groupby('Sub_District')['Price_per_SqFt'].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=avg_price_area.values,
                        y=avg_price_area.index,
                        orientation='h',
                        title="Top 10 Areas by Price per Sq Ft"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    # Data insights
    st.subheader("Market Insights")
    insights_data = data_processor.get_market_insights()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{insights_data['total_properties']:,}")
    
    with col2:
        st.metric("Average Price", f"₹{insights_data['avg_price']:,.0f}")
    
    with col3:
        st.metric("Price Range", f"₹{insights_data['price_range']:,.0f}")
    
    with col4:
        st.metric("Avg Price/SqFt", f"₹{insights_data['avg_price_per_sqft']:,.0f}")

if __name__ == "__main__":
    main()
