import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from database import db_manager
import warnings
warnings.filterwarnings('ignore')

def create_safe_training_data() -> pd.DataFrame:
    """Create safe, validated training data from database"""
    try:
        # Get data from database with explicit type handling
        db_data = db_manager.get_properties_from_db()
        
        if db_data.empty:
            raise ValueError("No data available in database")
        
        # Create a clean copy with explicit data types
        clean_data = pd.DataFrame({
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
        
        # Remove any rows with NaN values after conversion
        clean_data = clean_data.dropna()
        
        # Validate realistic ranges
        clean_data = clean_data[
            (clean_data['Area_SqFt'] > 0) & 
            (clean_data['BHK'] > 0) & 
            (clean_data['Price_INR'] > 0)
        ]
        
        print(f"Created safe training data with {len(clean_data)} records")
        return clean_data
        
    except Exception as e:
        print(f"Error creating training data: {e}")
        raise

def safe_encode_categorical(data: pd.DataFrame, encoders: Dict = None, fit: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """Safely encode categorical features without conversion errors"""
    if encoders is None:
        encoders = {}
    
    data_encoded = data.copy()
    categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
    
    from sklearn.preprocessing import LabelEncoder
    
    for column in categorical_columns:
        if column in data_encoded.columns:
            # Ensure column is string type first
            data_encoded[column] = data_encoded[column].astype(str)
            
            if fit:
                encoders[column] = LabelEncoder()
                data_encoded[column] = encoders[column].fit_transform(data_encoded[column])
            else:
                if column in encoders:
                    # Handle unseen categories safely
                    unique_values = set(encoders[column].classes_)
                    data_encoded[column] = data_encoded[column].apply(
                        lambda x: x if x in unique_values else encoders[column].classes_[0]
                    )
                    data_encoded[column] = encoders[column].transform(data_encoded[column])
    
    return data_encoded, encoders

def safe_model_predict(model, encoders: Dict, input_data: Dict[str, Any]) -> float:
    """Safely make predictions with proper data validation"""
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Ensure proper data types
    input_df['Area_SqFt'] = pd.to_numeric(input_df['Area_SqFt'])
    input_df['BHK'] = pd.to_numeric(input_df['BHK']).astype(int)
    
    # Encode categorical features
    encoded_input, _ = safe_encode_categorical(input_df, encoders, fit=False)
    
    # Make prediction
    prediction = model.predict(encoded_input)[0]
    return float(prediction)