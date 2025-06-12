import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st

class PropertyAppreciationAnalyzer:
    def __init__(self):
        """Initialize the Property Appreciation Analyzer"""
        self.historical_growth_rates = {
            'Mumbai': {
                '2019': 5.2, '2020': -2.1, '2021': 8.7, '2022': 12.3, '2023': 9.8, '2024': 7.5
            },
            'Delhi': {
                '2019': 4.8, '2020': -1.8, '2021': 7.2, '2022': 10.5, '2023': 8.9, '2024': 6.8
            },
            'Bangalore': {
                '2019': 6.1, '2020': 1.2, '2021': 11.4, '2022': 15.2, '2023': 12.7, '2024': 9.3
            },
            'Gurugram': {
                '2019': 3.9, '2020': -3.2, '2021': 9.8, '2022': 13.1, '2023': 10.4, '2024': 8.1
            },
            'Noida': {
                '2019': 4.2, '2020': -2.5, '2021': 8.9, '2022': 11.7, '2023': 9.6, '2024': 7.2
            }
        }
        
        self.price_index_data = {
            'Mumbai': [100, 105.2, 103.0, 111.9, 125.7, 138.0, 148.4],
            'Delhi': [100, 104.8, 102.9, 110.3, 121.9, 132.8, 141.8],
            'Bangalore': [100, 106.1, 107.4, 119.6, 137.8, 155.3, 169.8],
            'Gurugram': [100, 103.9, 100.6, 110.5, 125.0, 138.0, 149.2],
            'Noida': [100, 104.2, 101.6, 110.6, 123.5, 135.4, 145.2]
        }
        
        self.years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024']
        
    def calculate_appreciation_metrics(self, city: str, years: int = 5) -> Dict:
        """Calculate comprehensive appreciation metrics for a city"""
        if city not in self.historical_growth_rates:
            return self._get_default_metrics()
        
        growth_data = self.historical_growth_rates[city]
        price_index = self.price_index_data[city]
        
        # Calculate metrics
        recent_years = list(growth_data.keys())[-years:]
        recent_growth = [growth_data[year] for year in recent_years]
        
        avg_growth = np.mean(recent_growth)
        growth_volatility = np.std(recent_growth)
        max_growth = max(recent_growth)
        min_growth = min(recent_growth)
        
        # Calculate CAGR
        start_index = price_index[0]
        end_index = price_index[-1]
        cagr = ((end_index / start_index) ** (1/6) - 1) * 100
        
        # Calculate cumulative returns
        cumulative_return = ((end_index - start_index) / start_index) * 100
        
        return {
            'average_annual_growth': round(avg_growth, 2),
            'cagr_6_year': round(cagr, 2),
            'volatility': round(float(growth_volatility), 2),
            'max_annual_growth': round(max_growth, 2),
            'min_annual_growth': round(min_growth, 2),
            'cumulative_return_6_year': round(cumulative_return, 2),
            'current_trend': 'Positive' if avg_growth > 0 else 'Negative',
            'risk_level': self._assess_risk_level(float(growth_volatility)),
            'market_phase': self._determine_market_phase(recent_growth[-1])
        }
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess risk level based on volatility"""
        if volatility < 3:
            return 'Low'
        elif volatility < 6:
            return 'Medium'
        else:
            return 'High'
    
    def _determine_market_phase(self, latest_growth: float) -> str:
        """Determine current market phase"""
        if latest_growth > 10:
            return 'Growth Phase'
        elif latest_growth > 5:
            return 'Stable Growth'
        elif latest_growth > 0:
            return 'Slow Growth'
        else:
            return 'Correction Phase'
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics for unknown cities"""
        return {
            'average_annual_growth': 7.5,
            'cagr_6_year': 8.2,
            'volatility': 4.5,
            'max_annual_growth': 12.0,
            'min_annual_growth': 2.0,
            'cumulative_return_6_year': 45.0,
            'current_trend': 'Positive',
            'risk_level': 'Medium',
            'market_phase': 'Stable Growth'
        }
    
    def create_appreciation_trends_chart(self, cities: List[str] = None) -> go.Figure:
        """Create comprehensive appreciation trends chart"""
        if cities is None:
            cities = ['Mumbai', 'Delhi', 'Bangalore', 'Gurugram', 'Noida']
        
        # Ensure cities is not empty
        if not cities:
            cities = ['Mumbai', 'Delhi', 'Bangalore']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Index Trends', 'Annual Growth Rates', 'Volatility Analysis', 'CAGR Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
        
        # Price Index Trends
        for i, city in enumerate(cities):
            if city in self.price_index_data:
                fig.add_trace(
                    go.Scatter(
                        x=self.years,
                        y=self.price_index_data[city],
                        mode='lines+markers',
                        name=city,
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
        
        # Annual Growth Rates
        for i, city in enumerate(cities):
            if city in self.historical_growth_rates:
                growth_years = list(self.historical_growth_rates[city].keys())
                growth_values = list(self.historical_growth_rates[city].values())
                
                fig.add_trace(
                    go.Bar(
                        x=growth_years,
                        y=growth_values,
                        name=f"{city} Growth",
                        marker_color=colors[i % len(colors)],
                        opacity=0.8
                    ),
                    row=1, col=2
                )
        
        # Volatility Analysis
        volatility_data = []
        for city in cities:
            if city in self.historical_growth_rates:
                growth_values = list(self.historical_growth_rates[city].values())
                volatility = np.std(growth_values)
                volatility_data.append({'City': city, 'Volatility': volatility})
        
        if volatility_data:
            df_vol = pd.DataFrame(volatility_data)
            fig.add_trace(
                go.Bar(
                    x=df_vol['City'],
                    y=df_vol['Volatility'],
                    name='Volatility',
                    marker_color='rgba(255, 99, 132, 0.8)'
                ),
                row=2, col=1
            )
        
        # CAGR Comparison
        cagr_data = []
        for city in cities:
            if city in self.price_index_data:
                start_price = self.price_index_data[city][0]
                end_price = self.price_index_data[city][-1]
                cagr = ((end_price / start_price) ** (1/6) - 1) * 100
                cagr_data.append({'City': city, 'CAGR': cagr})
        
        if cagr_data:
            df_cagr = pd.DataFrame(cagr_data)
            fig.add_trace(
                go.Bar(
                    x=df_cagr['City'],
                    y=df_cagr['CAGR'],
                    name='CAGR (%)',
                    marker_color='rgba(54, 162, 235, 0.8)'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Property Appreciation Analysis Dashboard",
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Price Index", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Growth Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="City", row=2, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="City", row=2, col=2)
        fig.update_yaxes(title_text="CAGR (%)", row=2, col=2)
        
        return fig
    
    def create_future_projection_chart(self, city: str, property_value: float, years: int = 10) -> go.Figure:
        """Create future value projection chart"""
        metrics = self.calculate_appreciation_metrics(city)
        avg_growth = metrics['average_annual_growth'] / 100
        
        # Create multiple scenarios
        future_years = list(range(1, years + 1))
        
        # Conservative scenario (50% of average growth)
        conservative_values = [property_value * ((1 + avg_growth * 0.5) ** year) for year in future_years]
        
        # Realistic scenario (average growth)
        realistic_values = [property_value * ((1 + avg_growth) ** year) for year in future_years]
        
        # Optimistic scenario (120% of average growth)
        optimistic_values = [property_value * ((1 + avg_growth * 1.2) ** year) for year in future_years]
        
        fig = go.Figure()
        
        # Add current value
        fig.add_trace(go.Scatter(
            x=[0] + future_years,
            y=[property_value] + conservative_values,
            mode='lines+markers',
            name='Conservative (Low Growth)',
            line=dict(color='#ff6b6b', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0] + future_years,
            y=[property_value] + realistic_values,
            mode='lines+markers',
            name='Realistic (Average Growth)',
            line=dict(color='#4ecdc4', width=4),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0] + future_years,
            y=[property_value] + optimistic_values,
            mode='lines+markers',
            name='Optimistic (High Growth)',
            line=dict(color='#45b7d1', width=3),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f"Property Value Projection - {city}",
            xaxis_title="Years from Now",
            yaxis_title="Property Value (₹)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        return fig
    
    def compare_cities_performance(self, cities: List[str], years: int = 5) -> pd.DataFrame:
        """Compare performance metrics across cities"""
        comparison_data = []
        
        for city in cities:
            metrics = self.calculate_appreciation_metrics(city, years)
            comparison_data.append({
                'City': city,
                'Avg Annual Growth (%)': metrics['average_annual_growth'],
                'CAGR (%)': metrics['cagr_6_year'],
                'Volatility': metrics['volatility'],
                'Risk Level': metrics['risk_level'],
                'Market Phase': metrics['market_phase'],
                'Cumulative Return (%)': metrics['cumulative_return_6_year']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_investment_recommendations(self, city: str, investment_horizon: int) -> Dict:
        """Get investment recommendations based on historical analysis"""
        metrics = self.calculate_appreciation_metrics(city)
        
        recommendations = {
            'overall_rating': self._calculate_investment_rating(metrics, investment_horizon),
            'expected_annual_return': metrics['average_annual_growth'],
            'recommendation_text': f"Based on {investment_horizon}-year analysis",
            'key_insights': [],
            'recommendations': [],
            'risk_factors': [],
            'best_strategy': ''
        }
        
        # Generate insights based on metrics
        if metrics['average_annual_growth'] > 8:
            recommendations['key_insights'].append(f"Strong historical performance with {metrics['average_annual_growth']}% average annual growth")
        
        if metrics['volatility'] < 4:
            recommendations['key_insights'].append("Low volatility indicates stable market conditions")
        elif metrics['volatility'] > 6:
            recommendations['risk_factors'].append("High volatility suggests market uncertainty")
        
        # Investment horizon specific recommendations
        if investment_horizon <= 3:
            if metrics['current_trend'] == 'Positive' and metrics['volatility'] < 5:
                recommendations['best_strategy'] = 'Short-term speculation with careful monitoring'
            else:
                recommendations['best_strategy'] = 'Consider waiting for better market conditions'
        else:
            recommendations['best_strategy'] = 'Long-term buy and hold strategy recommended'
        
        # Generate detailed recommendation text
        if metrics['average_annual_growth'] > 8 and metrics['volatility'] < 5:
            recommendations['recommendation_text'] = f"Excellent investment opportunity with {metrics['average_annual_growth']:.1f}% average growth and stable market conditions"
        elif metrics['average_annual_growth'] > 5:
            recommendations['recommendation_text'] = f"Good investment potential with {metrics['average_annual_growth']:.1f}% historical growth"
        else:
            recommendations['recommendation_text'] = f"Conservative market with {metrics['average_annual_growth']:.1f}% growth - suitable for risk-averse investors"
        
        return recommendations
    
    def _calculate_investment_rating(self, metrics: Dict, horizon: int) -> str:
        """Calculate overall investment rating"""
        score = 0
        
        # Growth score
        if metrics['average_annual_growth'] > 10:
            score += 3
        elif metrics['average_annual_growth'] > 6:
            score += 2
        elif metrics['average_annual_growth'] > 0:
            score += 1
        
        # Volatility score (lower is better)
        if metrics['volatility'] < 3:
            score += 2
        elif metrics['volatility'] < 6:
            score += 1
        
        # CAGR score
        if metrics['cagr_6_year'] > 8:
            score += 2
        elif metrics['cagr_6_year'] > 5:
            score += 1
        
        # Horizon adjustment
        if horizon > 5:
            score += 1
        
        if score >= 7:
            return 'Excellent'
        elif score >= 5:
            return 'Good'
        elif score >= 3:
            return 'Average'
        else:
            return 'Below Average'