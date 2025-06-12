"""
Unit tests for ML models and predictors
"""
import unittest
import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.fast_ml_model import FastRealEstatePredictor


class TestFastRealEstatePredictor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = FastRealEstatePredictor()
        
        # Sample test data
        self.test_data = pd.DataFrame({
            'City': ['Mumbai', 'Delhi', 'Bangalore'] * 10,
            'District': ['Andheri', 'CP', 'Koramangala'] * 10,
            'Sub_District': ['East', 'Central', 'Block 1'] * 10,
            'Area_SqFt': [1000, 1200, 800] * 10,
            'BHK': [2, 3, 2] * 10,
            'Property_Type': ['Apartment', 'Villa', 'Apartment'] * 10,
            'Furnishing': ['Furnished', 'Semi-Furnished', 'Unfurnished'] * 10,
            'Price_INR': [5000000, 8000000, 4000000] * 10,
            'Price_per_SqFt': [5000, 6667, 5000] * 10
        })
    
    def test_model_training(self):
        """Test model training functionality"""
        metrics = self.predictor.train_model(self.test_data)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('r2_score', metrics)
        self.assertIn('mae', metrics)
    
    def test_prediction(self):
        """Test prediction functionality"""
        # Train model first
        self.predictor.train_model(self.test_data)
        
        test_input = {
            'City': 'Mumbai',
            'District': 'Andheri',
            'Sub_District': 'East',
            'Area_SqFt': 1000,
            'BHK': 2,
            'Property_Type': 'Apartment',
            'Furnishing': 'Furnished'
        }
        
        prediction, confidence = self.predictor.predict(test_input)
        
        self.assertIsInstance(prediction, float)
        self.assertIsInstance(confidence, dict)
        self.assertGreater(prediction, 0)


if __name__ == '__main__':
    unittest.main()