# API Documentation

## Core Components

### FastRealEstatePredictor
Machine learning predictor for property valuations.

**Methods:**
- `train_model(data: DataFrame) -> Dict[str, float]`
- `predict(input_data: Dict) -> Tuple[float, Dict[str, float]]`
- `get_feature_importance() -> Dict[str, float]`

### PropertyAppreciationAnalyzer
Historical market analysis and trend prediction.

**Methods:**
- `calculate_appreciation_metrics(city: str, years: int) -> Dict`
- `create_appreciation_trends_chart(cities: List[str]) -> go.Figure`
- `compare_cities_performance(cities: List[str], years: int) -> DataFrame`

### InvestmentAnalyzer
Property investment scoring and recommendations.

**Methods:**
- `analyze(property_data: Dict, predicted_price: float) -> Tuple[int, str]`

### PropertyPortfolioAnalyzer
Portfolio tracking and management analytics.

**Methods:**
- `analyze_current_property_value(purchase_data: Dict, predictor) -> Dict`
- `generate_hold_sell_recommendation(analysis: Dict, property_data: Dict) -> Dict`
- `analyze_investment_opportunity(target_property: Dict, budget: float, predictor) -> Dict`

### RealEstateChatbot (ARIA)
AI-powered real estate assistant with context awareness.

**Methods:**
- `get_response(user_message: str, chat_history: List[Dict]) -> str`
- `extract_user_context(message: str) -> Dict`
- `analyze_sentiment_and_urgency(message: str) -> Dict`

### EMICalculator
Loan calculation and financial planning utilities.

**Methods:**
- `calculate_emi(principal: float, annual_rate: float, tenure_years: int) -> Dict`
- `generate_amortization_schedule(principal: float, annual_rate: float, tenure_years: int) -> List`

## Data Models

### Property Schema
```python
{
    "City": str,
    "District": str, 
    "Sub_District": str,
    "Area_SqFt": float,
    "BHK": int,
    "Property_Type": str,
    "Furnishing": str,
    "Price_INR": float,
    "Price_per_SqFt": float
}
```

### Prediction Response
```python
{
    "predicted_price": float,
    "confidence_intervals": {
        "lower_bound": float,
        "upper_bound": float
    },
    "investment_score": int,
    "recommendation": str
}
```