"""
Configuration settings for AI Real Estate Intelligence Platform
"""
import os
from typing import Dict, List

# Application Configuration
APP_NAME = "AI Real Estate Intelligence Platform"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Professional Investment Analysis & Market Intelligence Platform"

# Database Configuration
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL"),
    "pool_size": 10,
    "max_overflow": 20,
    "pool_timeout": 30,
    "pool_recycle": 1800
}

# API Configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4o",  # Latest OpenAI model
    "max_tokens": 2000,
    "temperature": 0.7
}

# ML Model Configuration
ML_CONFIG = {
    "model_cache_path": "public/data/fast_model_cache.pkl",
    "validation_split": 0.2,
    "random_state": 42,
    "n_estimators": 100,
    "max_depth": 15
}

# City Configuration
SUPPORTED_CITIES: List[str] = [
    "Mumbai", "Delhi", "Bangalore", "Gurugram", "Noida"
]

# Market Data Configuration
MARKET_DATA_CONFIG = {
    "historical_years": 6,
    "growth_rates": {
        "Mumbai": 0.08,
        "Delhi": 0.07,
        "Bangalore": 0.11,
        "Gurugram": 0.12,
        "Noida": 0.10
    },
    "price_index_base_year": 2018
}

# UI Configuration
UI_CONFIG = {
    "theme_color": "#667eea",
    "secondary_color": "#764ba2",
    "accent_color": "#4ecdc4",
    "page_title": "AI Real Estate Intelligence",
    "page_icon": "🏠"
}

# Data Paths
DATA_PATHS = {
    "csv_data": "public/data/",
    "model_cache": "public/data/",
    "assets": "public/assets/"
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "cache_ttl": 3600,  # 1 hour
    "max_prediction_time": 500,  # milliseconds
    "batch_size": 1000
}