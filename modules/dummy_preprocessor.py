import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DummyMarketPreprocessor:
    """
    A dummy preprocessor for market data when the real preprocessor is not available.
    It accepts the input data and passess it through to be used with fallback prediction methods.
    """
    def __init__(self):
        pass
    
    def transform(self, X):
        """
        Return an identity transformation of the input
        """
        # If it's a pandas DataFrame, return the values
        if hasattr(X, 'values'):
            return X.values
        return X
    
    def fit_transform(self, X, y=None):
        """
        Return an identity transformation of the input
        """
        return self.transform(X)

class DummyResourcePreprocessor:
    """
    A dummy preprocessor for resource generator model when the real preprocessor is not available.
    """
    def __init__(self, crops=None):
        """
        Initialize with supported crops
        """
        if crops is None:
            self.crops = ["Basil", "Cilantro", "Kale", "Lettuce", "Spinach"]
        else:
            self.crops = crops
    
    def transform(self, X):
        """
        Transform crop names to one-hot encoded representation
        """
        # Create an empty array of shape (n_samples, n_crops)
        n_samples = len(X)
        one_hot = np.zeros((n_samples, len(self.crops)))
        
        # One-hot encode the crop column
        for i, crop in enumerate(X['Crop']):
            if crop in self.crops:
                crop_idx = self.crops.index(crop)
                one_hot[i, crop_idx] = 1
        
        return one_hot
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform
        """
        return self.transform(X)

class DummyYieldPreprocessor:
    """
    A dummy preprocessor for yield prediction model when the real preprocessor is not available.
    """
    def __init__(self, crops=None):
        """
        Initialize with supported crops
        """
        if crops is None:
            self.crops = ["Basil", "Cilantro", "Kale", "Lettuce", "Spinach"]
        else:
            self.crops = crops
            
        # Initialize a scaler for numerical features
        self.scaler = StandardScaler()
    
    def transform(self, X):
        """
        Transform input data (crop names and resources) for yield prediction
        """
        # Extract the crop column for one-hot encoding
        crop_col = X['Crop']
        
        # Create one-hot encoding for crops
        n_samples = len(X)
        crop_one_hot = np.zeros((n_samples, len(self.crops)))
        
        for i, crop in enumerate(crop_col):
            if crop in self.crops:
                crop_idx = self.crops.index(crop)
                crop_one_hot[i, crop_idx] = 1
        
        # Get numerical columns
        numerical_cols = ['Light', 'Temperature', 'Humidity', 'CO2', 
                         'Soil Moisture', 'pH', 'EC']
        
        # Extract numerical values
        numerical_values = []
        
        for col in numerical_cols:
            if col in X.columns:
                values = X[col].values.reshape(-1, 1)
            else:
                # If column is missing, use default values based on crop
                values = np.zeros((n_samples, 1))
                for i, crop in enumerate(crop_col):
                    if crop == "Basil":
                        values[i] = 0.6
                    elif crop == "Cilantro":
                        values[i] = 0.5
                    elif crop == "Kale":
                        values[i] = 0.7
                    elif crop == "Lettuce":
                        values[i] = 0.4
                    elif crop == "Spinach":
                        values[i] = 0.5
                    else:
                        values[i] = 0.5
            
            # Apply simple scaling to normalize the values
            if col == 'Light':
                values = values / 400.0
            elif col == 'Temperature':
                values = (values - 15) / 15.0
            elif col == 'Humidity':
                values = (values - 40) / 50.0
            elif col == 'CO2':
                values = (values - 400) / 1100.0
            elif col == 'Soil Moisture':
                values = values / 100.0
            elif col == 'pH':
                values = (values - 5.5) / 2.0
            elif col == 'EC':
                values = (values - 0.5) / 3.0
                
            numerical_values.append(values)
            
        # Combine numerical features into one array
        numerical_array = np.concatenate(numerical_values, axis=1)
        
        # Combine categorical and numerical features
        combined = np.concatenate([crop_one_hot, numerical_array], axis=1)
        
        return combined
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform
        """
        return self.transform(X)

def create_dummy_market_preprocessor():
    """
    Create a dummy preprocessor for market data
    """
    return DummyMarketPreprocessor()

def create_dummy_resource_preprocessor():
    """
    Create a dummy preprocessor for resource generation
    """
    return DummyResourcePreprocessor()

def create_dummy_yield_preprocessor():
    """
    Create a dummy preprocessor for yield prediction
    """
    return DummyYieldPreprocessor()

def create_dummy_preprocessor():
    """
    Generic function to create a dummy preprocessor,
    used when the specific type is not specified
    """
    return DummyMarketPreprocessor()