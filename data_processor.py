import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

class DataProcessor:
    def __init__(self):
        self.data = {}
        self.combined_data = None
        self.data_dir = "data"
        
    def load_all_data(self):
        """Load all city data from CSV files"""
        cities = ["mumbai", "delhi", "gurugram", "noida", "bangalore"]
        
        for city in cities:
            file_path = os.path.join(self.data_dir, f"{city}_properties.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['City'] = city.title()
                    self.data[city.title()] = df
                except Exception as e:
                    print(f"Error loading {city} data: {e}")
        
        # Combine all data
        if self.data:
            self.combined_data = pd.concat(list(self.data.values()), ignore_index=True)
            self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the combined data"""
        if self.combined_data is not None and not self.combined_data.empty:
            # Ensure numeric columns are properly typed
            numeric_columns = ['Area_SqFt', 'BHK', 'Price_INR']
            for col in numeric_columns:
                if col in self.combined_data.columns:
                    self.combined_data[col] = pd.to_numeric(self.combined_data[col], errors='coerce')
            
            # Ensure categorical columns are strings
            categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
            for col in categorical_columns:
                if col in self.combined_data.columns:
                    self.combined_data[col] = self.combined_data[col].astype(str)
            
            # Calculate price per sq ft
            if 'Price_per_SqFt' not in self.combined_data.columns:
                self.combined_data['Price_per_SqFt'] = (
                    self.combined_data['Price_INR'] / self.combined_data['Area_SqFt']
                )
            
            # Handle missing values
            self.combined_data = self.combined_data.dropna()
            
            # Remove outliers using IQR method for numeric data
            if len(self.combined_data) > 0:
                Q1 = self.combined_data['Price_INR'].quantile(0.25)
                Q3 = self.combined_data['Price_INR'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.combined_data = self.combined_data[
                    (self.combined_data['Price_INR'] >= lower_bound) &
                    (self.combined_data['Price_INR'] <= upper_bound)
                ]
    
    def get_districts(self, city: str) -> List[str]:
        """Get districts for a given city"""
        if city in self.data:
            return sorted(self.data[city]['District'].unique().tolist())
        return []
    
    def get_sub_districts(self, city: str, district: str) -> List[str]:
        """Get sub-districts for a given city and district"""
        if city in self.data:
            filtered_data = self.data[city][self.data[city]['District'] == district]
            return sorted(filtered_data['Sub_District'].unique().tolist())
        return []
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get the combined preprocessed data"""
        return self.combined_data
    
    def get_market_analysis(self, city: str, district: str) -> pd.DataFrame:
        """Get market analysis data for a specific city and district"""
        if city in self.data:
            return self.data[city][self.data[city]['District'] == district]
        return pd.DataFrame()
    
    def get_market_insights(self) -> Dict:
        """Get overall market insights"""
        if self.combined_data is not None and not self.combined_data.empty:
            return {
                'total_properties': len(self.combined_data),
                'avg_price': self.combined_data['Price_INR'].mean(),
                'price_range': self.combined_data['Price_INR'].max() - self.combined_data['Price_INR'].min(),
                'avg_price_per_sqft': self.combined_data['Price_per_SqFt'].mean()
            }
        return {
            'total_properties': 0,
            'avg_price': 0,
            'price_range': 0,
            'avg_price_per_sqft': 0
        }
