import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from typing import Dict, Any

class RealEstatePredictor:
    def __init__(self):
        self.model = DecisionTreeRegressor(
            random_state=42,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5
        )
        self.label_encoders = {}
        self.feature_columns = [
            'City', 'District', 'Sub_District', 'Area_SqFt', 
            'BHK', 'Property_Type', 'Furnishing'
        ]
        self.is_trained = False
        
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features"""
        data_encoded = data.copy()
        categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
        
        for column in categorical_columns:
            if column in data_encoded.columns:
                if fit:
                    self.label_encoders[column] = LabelEncoder()
                    data_encoded[column] = self.label_encoders[column].fit_transform(data_encoded[column])
                else:
                    if column in self.label_encoders:
                        # Handle unseen categories
                        unique_values = self.label_encoders[column].classes_
                        data_encoded[column] = data_encoded[column].apply(
                            lambda x: x if x in unique_values else unique_values[0]
                        )
                        data_encoded[column] = self.label_encoders[column].transform(data_encoded[column])
        
        return data_encoded
    
    def train_model(self, data: pd.DataFrame):
        """Train the Decision Tree model"""
        if data is None or data.empty:
            raise ValueError("No data provided for training")
        
        # Prepare features and target
        X = data[self.feature_columns].copy()
        y = data['Price_INR'].copy()
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: ₹{mae:,.0f}")
        print(f"R² Score: {r2:.3f}")
        
        self.is_trained = True
        
        return {
            'mae': mae,
            'r2_score': r2,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict(self, input_data: Dict[str, Any]) -> float:
        """Make price prediction for given input"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value
        
        # Select and order columns
        input_df = input_df[self.feature_columns]
        
        # Encode categorical features
        input_encoded = self._encode_categorical_features(input_df, fit=False)
        
        # Make prediction
        prediction = self.model.predict(input_encoded)[0]
        
        # Ensure prediction is positive
        return max(prediction, 100000)  # Minimum price of 1 lakh
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_trained:
            return {}
        
        return dict(zip(self.feature_columns, self.model.feature_importances_))
