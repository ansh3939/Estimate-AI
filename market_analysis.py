import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ComparativeMarketAnalyzer:
    def __init__(self):
        self.historical_growth_rates = {
            'Mumbai': {
                'annual_growth': 0.08,
                'quarterly_growth': 0.02,
                'volatility': 0.15,
                'market_cycle': 'stable'
            },
            'Delhi': {
                'annual_growth': 0.07,
                'quarterly_growth': 0.0175,
                'volatility': 0.12,
                'market_cycle': 'stable'
            },
            'Gurugram': {
                'annual_growth': 0.12,
                'quarterly_growth': 0.03,
                'volatility': 0.20,
                'market_cycle': 'growth'
            },
            'Noida': {
                'annual_growth': 0.10,
                'quarterly_growth': 0.025,
                'volatility': 0.18,
                'market_cycle': 'growth'
            },
            'Bangalore': {
                'annual_growth': 0.11,
                'quarterly_growth': 0.0275,
                'volatility': 0.16,
                'market_cycle': 'growth'
            }
        }
        
    def generate_historical_trends(self, data: pd.DataFrame, years_back: int = 5) -> pd.DataFrame:
        """Generate historical price trends based on current data and growth patterns"""
        historical_data = []
        current_date = datetime.now()
        
        for city in data['City'].unique():
            city_data = data[data['City'] == city]
            growth_info = self.historical_growth_rates.get(city, self.historical_growth_rates['Mumbai'])
            
            for months_back in range(0, years_back * 12, 3):  # Quarterly data
                date = current_date - timedelta(days=months_back * 30)
                
                # Calculate historical prices based on growth rates
                growth_factor = (1 + growth_info['annual_growth']) ** (months_back / 12)
                volatility_factor = 1 + np.random.normal(0, growth_info['volatility'] * 0.1)
                
                for _, row in city_data.iterrows():
                    historical_price = row['Price_INR'] / growth_factor * volatility_factor
                    
                    historical_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'City': city,
                        'District': row['District'],
                        'Sub_District': row['Sub_District'],
                        'Property_Type': row['Property_Type'],
                        'BHK': row['BHK'],
                        'Area_SqFt': row['Area_SqFt'],
                        'Price_INR': max(historical_price, 100000),
                        'Price_per_SqFt': max(historical_price / row['Area_SqFt'], 100),
                        'Quarter': f"Q{((current_date.month - 1) // 3) + 1} {date.year}",
                        'Year': date.year
                    })
        
        return pd.DataFrame(historical_data)
    
    def calculate_appreciation_trends(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate property appreciation trends for each city"""
        trends = {}
        
        for city in data['City'].unique():
            city_data = data[data['City'] == city]
            growth_info = self.historical_growth_rates.get(city, self.historical_growth_rates['Mumbai'])
            
            # Calculate various metrics
            avg_price = city_data['Price_INR'].mean()
            avg_price_sqft = city_data['Price_per_SqFt'].mean()
            
            trends[city] = {
                'current_avg_price': avg_price,
                'current_avg_price_sqft': avg_price_sqft,
                'annual_growth_rate': growth_info['annual_growth'],
                'projected_1_year': avg_price * (1 + growth_info['annual_growth']),
                'projected_3_year': avg_price * ((1 + growth_info['annual_growth']) ** 3),
                'projected_5_year': avg_price * ((1 + growth_info['annual_growth']) ** 5),
                'market_volatility': growth_info['volatility'],
                'market_cycle': growth_info['market_cycle'],
                'investment_score': self._calculate_investment_score(growth_info, avg_price_sqft)
            }
        
        return trends
    
    def _calculate_investment_score(self, growth_info: Dict, avg_price_sqft: float) -> int:
        """Calculate investment score based on growth potential and affordability"""
        base_score = 5
        
        # Growth rate bonus
        if growth_info['annual_growth'] >= 0.10:
            base_score += 2
        elif growth_info['annual_growth'] >= 0.08:
            base_score += 1
        
        # Volatility penalty
        if growth_info['volatility'] > 0.18:
            base_score -= 1
        
        # Affordability factor
        if avg_price_sqft < 8000:
            base_score += 1
        elif avg_price_sqft > 15000:
            base_score -= 1
        
        return max(1, min(10, base_score))
    
    def create_comparative_analysis(self, data: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create comprehensive comparative analysis charts"""
        charts = {}
        
        # 1. City-wise Price Comparison
        city_stats = data.groupby('City').agg({
            'Price_INR': ['mean', 'median', 'std'],
            'Price_per_SqFt': ['mean', 'median'],
            'Area_SqFt': 'mean'
        }).round(0)
        
        city_stats.columns = ['Avg_Price', 'Median_Price', 'Price_Std', 'Avg_Price_SqFt', 'Median_Price_SqFt', 'Avg_Area']
        city_stats = city_stats.reset_index()
        
        # Price comparison chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Bar(
            x=city_stats['City'],
            y=city_stats['Avg_Price'],
            name='Average Price',
            marker_color='#2E7D32',
            text=[f'₹{x:,.0f}' for x in city_stats['Avg_Price']],
            textposition='outside'
        ))
        fig_price.add_trace(go.Bar(
            x=city_stats['City'],
            y=city_stats['Median_Price'],
            name='Median Price',
            marker_color='#4CAF50',
            text=[f'₹{x:,.0f}' for x in city_stats['Median_Price']],
            textposition='outside'
        ))
        fig_price.update_layout(
            title='City-wise Property Price Comparison',
            xaxis_title='Cities',
            yaxis_title='Price (₹)',
            barmode='group',
            height=400
        )
        charts['price_comparison'] = fig_price
        
        # 2. Price per SqFt Analysis
        fig_sqft = px.box(
            data,
            x='City',
            y='Price_per_SqFt',
            title='Price per Square Foot Distribution by City',
            color='City'
        )
        fig_sqft.update_layout(height=400, showlegend=False)
        charts['price_sqft_distribution'] = fig_sqft
        
        # 3. Property Type Analysis across Cities
        prop_type_city = data.groupby(['City', 'Property_Type'])['Price_INR'].mean().reset_index()
        fig_prop_type = px.bar(
            prop_type_city,
            x='City',
            y='Price_INR',
            color='Property_Type',
            title='Average Price by Property Type across Cities',
            barmode='group'
        )
        fig_prop_type.update_layout(height=400)
        charts['property_type_analysis'] = fig_prop_type
        
        # 4. BHK Analysis across Cities
        bhk_city = data.groupby(['City', 'BHK'])['Price_INR'].mean().reset_index()
        fig_bhk = px.line(
            bhk_city,
            x='BHK',
            y='Price_INR',
            color='City',
            title='Price Trend by BHK Configuration across Cities',
            markers=True
        )
        fig_bhk.update_layout(height=400)
        charts['bhk_analysis'] = fig_bhk
        
        # 5. Area vs Price Scatter
        fig_scatter = px.scatter(
            data.sample(min(500, len(data))),
            x='Area_SqFt',
            y='Price_INR',
            color='City',
            size='BHK',
            title='Area vs Price Analysis across Cities',
            hover_data=['Property_Type', 'District']
        )
        fig_scatter.update_layout(height=400)
        charts['area_price_scatter'] = fig_scatter
        
        return charts
    
    def create_appreciation_forecast(self, data: pd.DataFrame) -> go.Figure:
        """Create property appreciation forecast chart"""
        trends = self.calculate_appreciation_trends(data)
        
        cities = list(trends.keys())
        years = [0, 1, 3, 5]
        
        fig = go.Figure()
        
        for city in cities:
            trend_data = trends[city]
            values = [
                trend_data['current_avg_price'],
                trend_data['projected_1_year'],
                trend_data['projected_3_year'],
                trend_data['projected_5_year']
            ]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=f"{city} (Growth: {trend_data['annual_growth_rate']*100:.1f}%)",
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Property Price Appreciation Forecast (5 Years)',
            xaxis_title='Years from Now',
            yaxis_title='Average Property Price (₹)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_investment_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create investment attractiveness heatmap"""
        city_district_stats = data.groupby(['City', 'District']).agg({
            'Price_per_SqFt': 'mean',
            'Price_INR': 'mean'
        }).reset_index()
        
        # Calculate investment scores
        investment_scores = []
        for _, row in city_district_stats.iterrows():
            city = row['City']
            growth_info = self.historical_growth_rates.get(city, self.historical_growth_rates['Mumbai'])
            score = self._calculate_investment_score(growth_info, row['Price_per_SqFt'])
            investment_scores.append(score)
        
        city_district_stats['Investment_Score'] = investment_scores
        
        # Create pivot table for heatmap
        heatmap_data = city_district_stats.pivot(
            index='District',
            columns='City',
            values='Investment_Score'
        ).fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            colorbar=dict(title='Investment Score'),
            text=heatmap_data.values,
            texttemplate='%{text}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title='Investment Attractiveness Heatmap (District-wise)',
            xaxis_title='Cities',
            yaxis_title='Districts',
            height=600
        )
        
        return fig
    
    def generate_market_insights(self, data: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive market insights"""
        insights = {}
        
        # Overall market analysis
        insights['total_properties'] = len(data)
        insights['cities_covered'] = data['City'].nunique()
        insights['avg_national_price'] = data['Price_INR'].mean()
        insights['avg_national_price_sqft'] = data['Price_per_SqFt'].mean()
        
        # City rankings
        city_stats = data.groupby('City').agg({
            'Price_INR': 'mean',
            'Price_per_SqFt': 'mean'
        }).round(0)
        
        insights['most_expensive_city'] = city_stats['Price_INR'].idxmax()
        insights['most_affordable_city'] = city_stats['Price_INR'].idxmin()
        insights['highest_price_sqft_city'] = city_stats['Price_per_SqFt'].idxmax()
        insights['lowest_price_sqft_city'] = city_stats['Price_per_SqFt'].idxmin()
        
        # Growth potential analysis
        trends = self.calculate_appreciation_trends(data)
        growth_scores = {city: info['investment_score'] for city, info in trends.items()}
        insights['best_investment_city'] = max(growth_scores, key=growth_scores.get)
        insights['growth_potential'] = trends
        
        # Property type insights
        prop_type_stats = data.groupby('Property_Type')['Price_INR'].mean().round(0)
        insights['most_expensive_property_type'] = prop_type_stats.idxmax()
        insights['most_affordable_property_type'] = prop_type_stats.idxmin()
        
        # Size analysis
        bhk_stats = data.groupby('BHK')['Price_per_SqFt'].mean().round(0)
        insights['most_value_bhk'] = bhk_stats.idxmin()
        insights['premium_bhk'] = bhk_stats.idxmax()
        
        return insights
    
    def create_trend_analysis(self, historical_data: pd.DataFrame) -> go.Figure:
        """Create historical trend analysis"""
        if historical_data.empty:
            return go.Figure()
        
        # Group by city and quarter
        quarterly_trends = historical_data.groupby(['City', 'Quarter', 'Year'])['Price_INR'].mean().reset_index()
        quarterly_trends['Date'] = pd.to_datetime(quarterly_trends[['Year', 'Quarter']].apply(
            lambda x: f"{x['Year']}-{(int(x['Quarter'][1:]) - 1) * 3 + 1:02d}-01", axis=1
        ))
        quarterly_trends = quarterly_trends.sort_values('Date')
        
        fig = go.Figure()
        
        for city in quarterly_trends['City'].unique():
            city_data = quarterly_trends[quarterly_trends['City'] == city]
            
            fig.add_trace(go.Scatter(
                x=city_data['Date'],
                y=city_data['Price_INR'],
                mode='lines+markers',
                name=city,
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='Historical Property Price Trends (5 Years)',
            xaxis_title='Date',
            yaxis_title='Average Property Price (₹)',
            height=500,
            hovermode='x unified'
        )
        
        return fig