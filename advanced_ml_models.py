import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
import joblib
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedRealEstatePredictor:
    def __init__(self):
        self.models = {
            'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=15, min_samples_split=10, min_samples_leaf=5),
            'random_forest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20, min_samples_split=5),
            'xgboost': xgb.XGBRegressor(random_state=42, n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.8)
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'City', 'District', 'Sub_District', 'Area_SqFt', 
            'BHK', 'Property_Type', 'Furnishing'
        ]
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self.model_performance = {}
        
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using label encoders"""
        data_encoded = data.copy()
        categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
        
        # Convert all categorical data to string first to handle mixed types
        for column in categorical_columns:
            if column in data_encoded.columns:
                data_encoded[column] = data_encoded[column].astype(str)
        
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
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better prediction"""
        data_features = data.copy()
        
        # Price per square foot (if price exists)
        if 'Price_INR' in data_features.columns:
            data_features['Price_per_SqFt'] = data_features['Price_INR'] / data_features['Area_SqFt']
        
        # Area categories
        data_features['Area_Category'] = pd.cut(data_features['Area_SqFt'], 
                                               bins=[0, 800, 1200, 1800, 2500, 10000], 
                                               labels=['Small', 'Medium', 'Large', 'XLarge', 'Luxury'])
        
        # BHK to Area ratio
        data_features['BHK_Area_Ratio'] = data_features['BHK'] / data_features['Area_SqFt'] * 1000
        
        # Encode new categorical features
        if 'Area_Category' in data_features.columns:
            data_features['Area_Category'] = data_features['Area_Category'].astype(str)
        
        return data_features
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train all models and select the best performer"""
        if data is None or data.empty:
            raise ValueError("No data provided for training")
        
        print("Creating enhanced features...")
        data_enhanced = self._create_features(data)
        
        # Prepare features and target
        feature_cols = self.feature_columns + ['Area_Category', 'BHK_Area_Ratio']
        X = data_enhanced[feature_cols].copy()
        y = data_enhanced['Price_INR'].copy()
        
        # Encode categorical features
        X_encoded = self._encode_categorical_features(X, fit=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scale features for XGBoost
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = float('-inf')
        
        print("Training and evaluating models...")
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            if model_name == 'xgboost':
                # Use scaled features for XGBoost
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Use original encoded features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Cross-validation score
            if model_name == 'xgboost':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            cv_mean = cv_scores.mean()
            
            self.model_performance[model_name] = {
                'mae': mae,
                'r2_score': r2,
                'rmse': rmse,
                'cv_r2_mean': cv_mean,
                'cv_r2_std': cv_scores.std()
            }
            
            print(f"{model_name} - R²: {r2:.3f}, MAE: ₹{mae:,.0f}, CV R²: {cv_mean:.3f}±{cv_scores.std():.3f}")
            
            # Select best model based on R² score
            if r2 > best_score:
                best_score = r2
                self.best_model = model
                self.best_model_name = model_name
        
        self.is_trained = True
        print(f"\nBest model: {self.best_model_name} with R² score: {best_score:.3f}")
        
        return self.model_performance
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Make prediction using the best model and return confidence scores"""
        if not self.is_trained:
            raise ValueError("Models are not trained yet")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Create enhanced features
        input_enhanced = self._create_features(input_df)
        
        # Ensure all required columns are present
        feature_cols = self.feature_columns + ['Area_Category', 'BHK_Area_Ratio']
        for col in feature_cols:
            if col not in input_enhanced.columns:
                if col == 'Area_Category':
                    input_enhanced[col] = 'Medium'
                elif col == 'BHK_Area_Ratio':
                    input_enhanced[col] = input_data.get('BHK', 2) / input_data.get('Area_SqFt', 1000) * 1000
                else:
                    input_enhanced[col] = 0
        
        # Select and order columns
        input_enhanced = input_enhanced[feature_cols]
        
        # Encode categorical features
        input_encoded = self._encode_categorical_features(input_enhanced, fit=False)
        
        # Get predictions from all models
        predictions = {}
        
        for model_name, model in self.models.items():
            if model_name == 'xgboost':
                input_scaled = self.scaler.transform(input_encoded)
                pred = model.predict(input_scaled)[0]
            else:
                pred = model.predict(input_encoded)[0]
            
            predictions[model_name] = max(pred, 100000)  # Minimum price of 1 lakh
        
        # Use best model prediction as primary
        best_prediction = predictions[self.best_model_name]
        
        return best_prediction, predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the best model"""
        if not self.is_trained or self.best_model is None:
            return {}
        
        feature_cols = self.feature_columns + ['Area_Category', 'BHK_Area_Ratio']
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance_dict = dict(zip(feature_cols, self.best_model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all model performances"""
        if not self.model_performance:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, metrics in self.model_performance.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'R² Score': metrics['r2_score'],
                'MAE (₹)': metrics['mae'],
                'RMSE (₹)': metrics['rmse'],
                'CV R² Mean': metrics['cv_r2_mean'],
                'CV R² Std': metrics['cv_r2_std']
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('R² Score', ascending=False)