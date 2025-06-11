import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
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
        # Advanced ensemble with four optimized models for superior accuracy
        self.models = {
            'random_forest': RandomForestRegressor(
                random_state=42, 
                n_estimators=250,  # Increased for better performance
                max_depth=28,      # Deeper trees for complex patterns
                min_samples_split=3,
                min_samples_leaf=2,
                max_features=0.8,  # Use more features
                bootstrap=True,
                oob_score=True,
                n_jobs=-1          # Use all cores
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=42, 
                n_estimators=400,  # More estimators for better accuracy
                max_depth=12,      # Deeper for complex relationships
                learning_rate=0.06, # Lower for better convergence
                subsample=0.87,
                colsample_bytree=0.85,
                reg_alpha=0.05,    # Fine-tuned regularization
                reg_lambda=0.8,    
                min_child_weight=2,
                gamma=0.05,
                objective='reg:squarederror',
                eval_metric='rmse',
                early_stopping_rounds=75,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                random_state=42,
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                loss='squared_error'
            ),
            'decision_tree': DecisionTreeRegressor(
                random_state=42, 
                max_depth=22,      # Increased depth
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.9,  # Use most features
                splitter='best'
            )
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
        self.ensemble_weights = {}  # For weighted averaging
        
    def _encode_categorical_features(self, data: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical features using label encoders"""
        data_encoded = data.copy()
        categorical_columns = ['City', 'District', 'Sub_District', 'Property_Type', 'Furnishing']
        
        # Convert all categorical data to string first to handle mixed types
        for column in categorical_columns:
            if column in data_encoded.columns:
                data_encoded[column] = data_encoded[column].astype(str)
                # Handle any NaN values
                data_encoded[column] = data_encoded[column].fillna('Unknown')
        
        for column in categorical_columns:
            if column in data_encoded.columns:
                if fit:
                    self.label_encoders[column] = LabelEncoder()
                    data_encoded[column] = self.label_encoders[column].fit_transform(data_encoded[column])
                else:
                    if column in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[column].classes_)
                        data_encoded[column] = data_encoded[column].apply(
                            lambda x: x if x in unique_values else self.label_encoders[column].classes_[0]
                        )
                        data_encoded[column] = self.label_encoders[column].transform(data_encoded[column])
                    else:
                        # If encoder doesn't exist, create a simple numeric mapping
                        unique_vals = data_encoded[column].unique()
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        data_encoded[column] = data_encoded[column].map(mapping)
        
        return data_encoded
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for superior prediction accuracy"""
        print("Creating enhanced features...")
        data_features = data.copy()
        
        # Price per square foot (if price exists)
        if 'Price_INR' in data_features.columns:
            data_features['Price_per_SqFt'] = data_features['Price_INR'] / data_features['Area_SqFt']
        
        # Advanced area categorization with more granularity (numeric encoding)
        def categorize_area(area):
            if area <= 600:
                return 0  # Compact
            elif area <= 1000:
                return 1  # Medium
            elif area <= 1500:
                return 2  # Large
            elif area <= 2200:
                return 3  # Premium
            elif area <= 3000:
                return 4  # Luxury
            else:
                return 5  # Ultra_Luxury
        
        data_features['Area_Category'] = data_features['Area_SqFt'].apply(categorize_area)
        
        # Sophisticated feature engineering
        data_features['BHK_Area_Ratio'] = data_features['BHK'] / data_features['Area_SqFt'] * 1000
        data_features['Area_Per_Room'] = data_features['Area_SqFt'] / data_features['BHK']
        data_features['Area_Squared'] = data_features['Area_SqFt'] ** 2
        data_features['BHK_Squared'] = data_features['BHK'] ** 2
        
        # Market premium factors
        city_premium = {
            'Mumbai': 1.8,
            'Delhi': 1.5, 
            'Gurugram': 1.4,
            'Bangalore': 1.3,
            'Noida': 1.1
        }
        data_features['Location_Premium'] = data_features['City'].map(city_premium).fillna(1.0)
        
        # Property type multipliers
        property_scores = {
            'Villa': 1.4,
            'House': 1.2,
            'Apartment': 1.0,
            'Studio': 0.7
        }
        data_features['Property_Score'] = data_features['Property_Type'].map(property_scores).fillna(1.0)
        
        # Furnishing impact
        furnishing_scores = {
            'Fully Furnished': 1.15,
            'Semi-Furnished': 1.05,
            'Unfurnished': 0.95
        }
        data_features['Furnishing_Score'] = data_features['Furnishing'].map(furnishing_scores).fillna(1.0)
        
        # Composite features
        data_features['Premium_Factor'] = (data_features['Location_Premium'] * 
                                         data_features['Property_Score'] * 
                                         data_features['Furnishing_Score'])
        
        # Advanced transformations
        data_features['Value_Index'] = data_features['Premium_Factor'] * data_features['Area_SqFt']
        data_features['Log_Area'] = np.log1p(data_features['Area_SqFt'])
        data_features['Sqrt_Area'] = np.sqrt(data_features['Area_SqFt'])
        
        # Encode new categorical features
        categorical_new = ['Area_Category']
        for col in categorical_new:
            if col in data_features.columns:
                data_features[col] = data_features[col].astype(str)
        
        return data_features
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train all models and select the best performer"""
        if data is None or data.empty:
            raise ValueError("No data provided for training")
        
        print("Creating enhanced features...")
        data_enhanced = self._create_features(data)
        
        # Prepare features and target with all advanced features
        advanced_features = [
            'Area_Category', 'BHK_Area_Ratio', 'Area_Per_Room', 'Area_Squared', 
            'BHK_Squared', 'Location_Premium', 'Property_Score', 'Furnishing_Score',
            'Premium_Factor', 'Value_Index', 'Log_Area', 'Sqrt_Area'
        ]
        feature_cols = self.feature_columns + advanced_features
        X = data_enhanced[feature_cols].copy()
        y = data_enhanced['Price_INR'].copy()
        
        # Encode categorical features (ensure Area_Category is properly encoded)
        X_encoded = self._encode_categorical_features(X, fit=True)
        
        # Verify all data is numeric
        print(f"Data types after encoding: {X_encoded.dtypes}")
        print(f"Any non-numeric values: {X_encoded.select_dtypes(include=['object']).columns.tolist()}")
        
        # Force convert any remaining object columns to numeric
        for col in X_encoded.columns:
            if X_encoded[col].dtype == 'object':
                print(f"Converting {col} from object to numeric")
                X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        X_encoded = X_encoded.fillna(0)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Scale features for XGBoost
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        best_score = float('-inf')
        
        print("Training and evaluating models with hyperparameter optimization...")
        
        # Hyperparameter tuning for Random Forest
        if 'random_forest' in self.models:
            rf_params = {
                'n_estimators': [200, 250, 300],
                'max_depth': [25, 28, 30],
                'min_samples_split': [2, 3, 4],
                'max_features': [0.7, 0.8, 0.9]
            }
            rf_search = RandomizedSearchCV(
                RandomForestRegressor(random_state=42, n_jobs=-1),
                rf_params, cv=3, n_iter=10, random_state=42, scoring='r2'
            )
            rf_search.fit(X_train, y_train)
            self.models['random_forest'] = rf_search.best_estimator_
            print(f"Random Forest optimized parameters: {rf_search.best_params_}")
        
        # Hyperparameter tuning for Gradient Boosting
        if 'gradient_boosting' in self.models:
            gb_params = {
                'n_estimators': [150, 200, 250],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.08, 0.1, 0.12],
                'subsample': [0.75, 0.8, 0.85]
            }
            gb_search = RandomizedSearchCV(
                GradientBoostingRegressor(random_state=42),
                gb_params, cv=3, n_iter=10, random_state=42, scoring='r2'
            )
            gb_search.fit(X_train_scaled, y_train)
            self.models['gradient_boosting'] = gb_search.best_estimator_
            print(f"Gradient Boosting optimized parameters: {gb_search.best_params_}")
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            try:
                if model_name == 'xgboost':
                    # XGBoost with proper validation set for early stopping
                    model.fit(X_train_scaled, y_train, 
                             eval_set=[(X_test_scaled, y_test)], 
                             verbose=False)
                    y_pred = model.predict(X_test_scaled)
                elif model_name == 'gradient_boosting':
                    # Use scaled features for Gradient Boosting as well
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    # Use original encoded features for tree-based models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
            
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
        
        # Calculate ensemble weights based on R² performance
        r2_scores = {name: perf['r2_score'] for name, perf in self.model_performance.items()}
        total_r2 = sum(max(0, score) for score in r2_scores.values())
        
        if total_r2 > 0:
            self.ensemble_weights = {
                model: max(0, r2_scores[model]) / total_r2 
                for model in r2_scores.keys()
            }
        else:
            # Equal weights fallback
            self.ensemble_weights = {
                model: 1.0 / len(r2_scores) 
                for model in r2_scores.keys()
            }
        
        print(f"Ensemble weights: {self.ensemble_weights}")
        
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
        
        # Ensure all required columns are present with advanced features
        advanced_features = [
            'Area_Category', 'BHK_Area_Ratio', 'Area_Per_Room', 'Area_Squared', 
            'BHK_Squared', 'Location_Premium', 'Property_Score', 'Furnishing_Score',
            'Premium_Factor', 'Value_Index', 'Log_Area', 'Sqrt_Area'
        ]
        feature_cols = self.feature_columns + advanced_features
        
        # Fill missing advanced features if not created properly
        for col in advanced_features:
            if col not in input_enhanced.columns:
                if col == 'Area_Category':
                    input_enhanced[col] = 'Medium'
                elif col == 'BHK_Area_Ratio':
                    input_enhanced[col] = input_data.get('BHK', 2) / input_data.get('Area_SqFt', 1000) * 1000
                else:
                    input_enhanced[col] = 1.0
        
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
        
        # Implement weighted ensemble averaging for better accuracy
        if len(self.ensemble_weights) == len(predictions):
            # Use performance-based weights
            weighted_prediction = sum(
                predictions[model] * self.ensemble_weights[model] 
                for model in predictions.keys()
            )
        else:
            # Fallback to equal weights if not available
            weighted_prediction = np.mean(list(predictions.values()))
        
        # Use weighted ensemble as primary prediction for better accuracy
        ensemble_prediction = max(weighted_prediction, 100000)
        
        return ensemble_prediction, predictions
    
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