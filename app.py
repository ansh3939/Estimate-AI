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
    investment_analyzer = InvestmentAnalyzer()
    emi_calculator = EMICalculator()
    
    # Load and process data
    data_processor.load_all_data()
    predictor.train_model(data_processor.get_combined_data())
    
    return data_processor, predictor, investment_analyzer, emi_calculator

def main():
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">AI Real Estate Price Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Professional Investment Analysis & Market Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load components
    try:
        data_processor, predictor, investment_analyzer, emi_calculator = load_components()
    except Exception as e:
        st.error(f"Error loading application components: {str(e)}")
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
            # Make prediction
            predicted_price = predictor.predict(input_data)
            
            # Analyze investment
            investment_score, recommendation = investment_analyzer.analyze(
                input_data, predicted_price
            )
            
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
                    avg_price_area = market_data.groupby('Sub_District')['Price_per_SqFt'].mean().sort_values(ascending=False).head(10)
                    fig = px.bar(
                        x=avg_price_area.values,
                        y=avg_price_area.index,
                        orientation='h',
                        title="Top 10 Sub-Districts by Price per Sq Ft",
                        color=avg_price_area.values,
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
        
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    else:
        # Default Dashboard when no prediction is made
        st.markdown("### 🏠 Welcome to AI Real Estate Intelligence Platform")
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
            st.markdown("#### 🏢 Property Type Distribution")
            prop_type_dist = data_processor.combined_data['Property_Type'].value_counts().reset_index()
            prop_type_dist.columns = ['Property_Type', 'Count']
            
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
                bhk_dist = data_processor.combined_data['BHK'].value_counts().sort_index().reset_index()
                bhk_dist.columns = ['BHK', 'Count']
                
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
