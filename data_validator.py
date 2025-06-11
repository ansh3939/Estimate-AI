import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class DataValidator:
    """Comprehensive data validation and cleaning for real estate data"""
    
    def __init__(self):
        self.required_columns = [
            'City', 'District', 'Sub_District', 'Area_SqFt', 
            'BHK', 'Property_Type', 'Furnishing', 'Price_INR'
        ]
        self.numeric_columns = ['Area_SqFt', 'BHK', 'Price_INR']
        self.categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
    
    def validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data validation and cleaning"""
        if data is None or data.empty:
            raise ValueError("No data provided for validation")
        
        # Create a copy to avoid modifying original
        clean_data = data.copy()
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in clean_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean numeric columns with robust conversion
        for col in self.numeric_columns:
            if col in clean_data.columns:
                # Convert to numeric, handling any string values
                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
                # Remove rows with invalid numeric values
                invalid_mask = clean_data[col].isna() | (clean_data[col] <= 0)
                if invalid_mask.any():
                    print(f"Removing {invalid_mask.sum()} rows with invalid {col} values")
                    clean_data = clean_data[~invalid_mask]
        
        # Clean categorical columns
        for col in self.categorical_columns:
            if col in clean_data.columns:
                # Convert to string and handle missing values
                clean_data[col] = clean_data[col].astype(str).str.strip()
                # Replace problematic values
                problematic_values = ['nan', 'None', 'null', '', 'NaN', 'NULL']
                clean_data[col] = clean_data[col].replace(problematic_values, 'Unknown')
                # Remove rows where essential categorical fields are unknown
                if col in ['City', 'Property_Type']:
                    unknown_mask = clean_data[col] == 'Unknown'
                    if unknown_mask.any():
                        print(f"Removing {unknown_mask.sum()} rows with unknown {col}")
                        clean_data = clean_data[~unknown_mask]
        
        # Calculate Price_per_SqFt if missing
        if 'Price_per_SqFt' not in clean_data.columns:
            clean_data['Price_per_SqFt'] = clean_data['Price_INR'] / clean_data['Area_SqFt']
        
        # Validate data ranges
        clean_data = self._validate_ranges(clean_data)
        
        # Final validation
        if clean_data.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        print(f"Data validation complete: {len(clean_data)} valid records")
        return clean_data
    
    def _validate_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data ranges to ensure realistic values"""
        # Remove unrealistic values
        valid_mask = (
            (data['Area_SqFt'] >= 100) & (data['Area_SqFt'] <= 10000) &  # 100 to 10,000 sq ft
            (data['BHK'] >= 1) & (data['BHK'] <= 10) &  # 1 to 10 BHK
            (data['Price_INR'] >= 100000) & (data['Price_INR'] <= 1000000000)  # 1 lakh to 100 crores
        )
        
        removed_count = (~valid_mask).sum()
        if removed_count > 0:
            print(f"Removing {removed_count} rows with unrealistic values")
        
        return data[valid_mask]
    
    def validate_model_input(self, input_data: Dict) -> Dict:
        """Validate and clean model input data"""
        clean_input = input_data.copy()
        
        # Ensure numeric fields are numbers
        for field in ['Area_SqFt', 'BHK']:
            if field in clean_input:
                try:
                    clean_input[field] = float(clean_input[field])
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid {field}: must be a number")
        
        # Ensure categorical fields are strings
        for field in self.categorical_columns:
            if field in clean_input:
                clean_input[field] = str(clean_input[field]).strip()
                if not clean_input[field] or clean_input[field].lower() in ['nan', 'none', 'null']:
                    clean_input[field] = 'Unknown'
        
        return clean_input

# Global validator instance
data_validator = DataValidator()