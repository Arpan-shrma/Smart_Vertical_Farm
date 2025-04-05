import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import sys
import os

# Add module directory to path
if 'modules' not in sys.path:
    sys.path.append('modules')
if 'utils' not in sys.path:
    sys.path.append('utils')

# Try to import custom modules (with error handling)
try:
    from forecast import get_forecast_summary
    from recommender import recommend_optimal_conditions
    from dummy_preprocessor import create_dummy_market_preprocessor, create_dummy_preprocessor
    modules_imported = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    modules_imported = False
    # Define simplified versions if modules not found
    def get_forecast_summary(current_data, periods=30, model_path=None):
        """Simplified forecast function that returns dummy data if module not found"""
        # Create dummy forecast data
        crops = current_data['Crop'].unique() if 'Crop' in current_data else ['Basil', 'Cilantro', 'Kale', 'Lettuce', 'Spinach']
        start_date = datetime.now()
        
        # Create daily forecast
        daily_data = []
        for i in range(periods):
            for crop in crops:
                base_price = 14.5 if crop == "Basil" else 12.3 if crop == "Cilantro" else 15.7 if crop == "Kale" else 8.9 if crop == "Lettuce" else 10.2
                # Add some randomness and trend
                price = base_price * (1 + 0.05 * i/periods + np.random.normal(0, 0.03))
                daily_data.append({
                    'Timestamp': start_date + timedelta(days=i),
                    'Crop': crop,
                    'Predicted_Price': round(price, 2),
                    'Days_Ahead': i
                })
        
        daily_df = pd.DataFrame(daily_data)
        
        # Create weekly and monthly aggregations
        weekly_df = daily_df.copy()
        weekly_df['Week'] = weekly_df['Timestamp'].dt.isocalendar().week
        weekly_df = weekly_df.groupby(['Crop', 'Week']).agg(
            Avg_Weekly_Price=('Predicted_Price', 'mean'),
            First_Date=('Timestamp', 'min')
        ).reset_index()
        weekly_df['Avg_Weekly_Price'] = weekly_df['Avg_Weekly_Price'].round(2)
        
        monthly_df = daily_df.copy()
        monthly_df['Month'] = monthly_df['Timestamp'].dt.month
        monthly_df = monthly_df.groupby(['Crop', 'Month']).agg(
            Avg_Monthly_Price=('Predicted_Price', 'mean'),
            First_Date=('Timestamp', 'min')
        ).reset_index()
        monthly_df['Avg_Monthly_Price'] = monthly_df['Avg_Monthly_Price'].round(2)
        
        return {
            'daily': daily_df,
            'weekly': weekly_df,
            'monthly': monthly_df
        }

    def recommend_optimal_conditions(crop, yield_value, model=None):
        """Simplified recommendation function that returns reasonable defaults if module not found"""
        # Base values for each crop
        crop_defaults = {
            'Basil': {'Light': 0.75, 'Temperature': 0.6, 'Humidity': 0.65, 'CO2': 0.7, 'Soil Moisture': 0.7, 'pH': 0.45, 'EC': 0.4},
            'Cilantro': {'Light': 0.65, 'Temperature': 0.45, 'Humidity': 0.6, 'CO2': 0.6, 'Soil Moisture': 0.65, 'pH': 0.5, 'EC': 0.35},
            'Kale': {'Light': 0.75, 'Temperature': 0.4, 'Humidity': 0.55, 'CO2': 0.7, 'Soil Moisture': 0.65, 'pH': 0.45, 'EC': 0.45},
            'Lettuce': {'Light': 0.5, 'Temperature': 0.35, 'Humidity': 0.65, 'CO2': 0.55, 'Soil Moisture': 0.7, 'pH': 0.4, 'EC': 0.3},
            'Spinach': {'Light': 0.65, 'Temperature': 0.3, 'Humidity': 0.65, 'CO2': 0.7, 'Soil Moisture': 0.65, 'pH': 0.5, 'EC': 0.4}
        }
        
        # Get base values for the crop
        base_values = crop_defaults.get(crop, crop_defaults['Lettuce']).copy()
        
        # Adjust based on yield (higher yields need more resources)
        # Scale factor: default is 1.0, adjust up to 50% more for high yields
        scale_factor = 1.0 + (yield_value - 1.0) * 0.1  # 10% adjustment per kg over 1.0
        scale_factor = max(0.8, min(1.5, scale_factor))  # Limit scaling between 0.8 and 1.5
        
        # Apply scaling to resources that scale with yield
        scaling_resources = ['Light', 'CO2', 'EC']
        for resource in scaling_resources:
            base_values[resource] = min(1.0, base_values[resource] * scale_factor)
            
        return base_values

# Function to load market demand forecast model
def load_market_forecast_model():
    """Load the market demand forecast model from disk."""
    try:
        model_path = 'models/forecast_market_demand.pkl'
        if not os.path.exists(model_path):
            model_path = 'forecast_market_demand.pkl'  # Try current directory
        
        model = joblib.load(model_path)
        return model, True
    except Exception as e:
        st.warning(f"Market forecast model could not be loaded: {e}. Using fallback predictions.")
        return None, False

# Function to load resource generator model
def load_resource_generator_model():
    """Load the resource generator model from disk."""
    try:
        model_path = 'models/resource_generator_model.h5'
        if not os.path.exists(model_path):
            model_path = 'resource_generator_model.h5'  # Try current directory
            
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        st.warning(f"Resource generator model could not be loaded: {e}. Using fallback generators.")
        return None, False

# Function to load yield predictor model
def load_yield_predictor_model():
    """Load the yield predictor model from disk."""
    try:
        model_path = 'models/yield_predictor_model.h5'
        if not os.path.exists(model_path):
            model_path = 'yield_predictor_model.h5'  # Try current directory
            
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        st.warning(f"Yield predictor model could not be loaded: {e}. Using fallback predictions.")
        return None, False

# Function to load preprocessors
def load_preprocessors():
    """Load preprocessors for the models."""
    preprocessors = {}
    
    # Market forecast preprocessor
    try:
        preprocessor_path = 'preprocessors/market_forecast_preprocessor.pkl'
        if not os.path.exists(preprocessor_path):
            # Try alternative paths
            alternatives = [
                'market_forecast_preprocessor.pkl',
                'forecast_preprocessor.pkl'
            ]
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    preprocessor_path = alt_path
                    break
        
        if os.path.exists(preprocessor_path):
            preprocessors['market_forecast'] = joblib.load(preprocessor_path)
        else:
            # Create a dummy preprocessor if file not found
            if modules_imported:
                preprocessors['market_forecast'] = create_dummy_market_preprocessor()
            else:
                # Simple dummy preprocessor if module not imported
                class DummyPreprocessor:
                    def transform(self, X): return X
                preprocessors['market_forecast'] = DummyPreprocessor()
    except Exception as e:
        st.warning(f"Market forecast preprocessor could not be loaded: {e}")
        # Create a dummy preprocessor
        if modules_imported:
            preprocessors['market_forecast'] = create_dummy_market_preprocessor()
        else:
            # Simple dummy preprocessor if module not imported
            class DummyPreprocessor:
                def transform(self, X): return X
            preprocessors['market_forecast'] = DummyPreprocessor()
    
    # Resource generator preprocessor
    try:
        preprocessor_path = 'preprocessors/preprocessor_resource_model.pkl'
        if not os.path.exists(preprocessor_path):
            # Try alternative paths
            alternatives = [
                'preprocessor_resource_model.pkl',
                'resource_generator_preprocessor.pkl',
                'preprocessor_model2.pkl'
            ]
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    preprocessor_path = alt_path
                    break
                    
        if os.path.exists(preprocessor_path):
            preprocessors['resource_generator'] = joblib.load(preprocessor_path)
        else:
            # Create a dummy preprocessor if file not found
            if modules_imported:
                preprocessors['resource_generator'] = create_dummy_preprocessor()
            else:
                # Simple dummy preprocessor if module not imported
                class DummyPreprocessor:
                    def transform(self, X): return X
                preprocessors['resource_generator'] = DummyPreprocessor()
    except Exception as e:
        st.warning(f"Resource generator preprocessor could not be loaded: {e}")
        # Create a dummy preprocessor
        if modules_imported:
            preprocessors['resource_generator'] = create_dummy_preprocessor()
        else:
            # Simple dummy preprocessor if module not imported
            class DummyPreprocessor:
                def transform(self, X): return X
            preprocessors['resource_generator'] = DummyPreprocessor()
    
    # Yield predictor preprocessor
    try:
        preprocessor_path = 'preprocessors/preprocessor_yield_model.pkl'
        if not os.path.exists(preprocessor_path):
            # Try alternative paths
            alternatives = [
                'preprocessor_yield_model.pkl',
                'yield_predictor_preprocessor.pkl'
            ]
            
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    preprocessor_path = alt_path
                    break
            
        if os.path.exists(preprocessor_path):        
            preprocessors['yield_predictor'] = joblib.load(preprocessor_path)
        else:
            # Create a dummy preprocessor if file not found
            if modules_imported:
                preprocessors['yield_predictor'] = create_dummy_preprocessor()
            else:
                # Simple dummy preprocessor if module not imported
                class DummyPreprocessor:
                    def transform(self, X): return X
                preprocessors['yield_predictor'] = DummyPreprocessor()
    except Exception as e:
        st.warning(f"Yield predictor preprocessor could not be loaded: {e}")
        # Create a dummy preprocessor
        if modules_imported:
            preprocessors['yield_predictor'] = create_dummy_preprocessor()
        else:
            # Simple dummy preprocessor if module not imported
            class DummyPreprocessor:
                def transform(self, X): return X
            preprocessors['yield_predictor'] = DummyPreprocessor()
    
    return preprocessors

# Function to generate sample market data
def generate_sample_data(start_date, days=7):
    """Generate sample market data for demonstration purposes"""
    crops = ["Basil", "Cilantro", "Kale", "Lettuce", "Spinach"]
    dates = pd.date_range(start=start_date - timedelta(days=days), periods=days)
    sample_data = []
    
    for crop in crops:
        base_price = 14.5 if crop == "Basil" else 12.3 if crop == "Cilantro" else 15.7 if crop == "Kale" else 8.9 if crop == "Lettuce" else 10.2
        for date in dates:
            # Add some random variation plus a slight upward trend
            trend_factor = 1 + (0.01 * (date - dates[0]).days / days)
            price = base_price * trend_factor * (1 + np.random.normal(0, 0.05))
            sample_data.append({
                'Timestamp': date,
                'Crop': crop,
                'Price per kg': price,
                'Volume Sold per Cycle': np.random.randint(450, 550),
                'Dump Amount': np.random.randint(20, 35)
            })
    
    return pd.DataFrame(sample_data)
# Function to preprocess data for market forecast model
def preprocess_market_data(df):
    """Preprocess market data for the forecast model"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Ensure timestamp is datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Extract date features
    data['Year'] = data['Timestamp'].dt.year
    data['Month'] = data['Timestamp'].dt.month
    data['Quarter'] = data['Timestamp'].dt.quarter
    data['Day_of_Week'] = data['Timestamp'].dt.dayofweek
    data['Day_of_Month'] = data['Timestamp'].dt.day
    data['Is_Weekend'] = (data['Day_of_Week'] >= 5).astype(int)
    data['Week_of_Year'] = data['Timestamp'].dt.isocalendar().week
    
    # Calculate days since start
    start_date = data['Timestamp'].min()
    data['Days_Since_Start'] = (data['Timestamp'] - start_date).dt.days
    
    # Create season features
    data['month'] = data['Timestamp'].dt.month
    data['Season_Winter'] = ((data['month'] >= 12) | (data['month'] <= 2)).astype(int)
    data['Season_Spring'] = ((data['month'] >= 3) & (data['month'] <= 5)).astype(int)
    data['Season_Summer'] = ((data['month'] >= 6) & (data['month'] <= 8)).astype(int)
    data['Season_Fall'] = ((data['month'] >= 9) & (data['month'] <= 11)).astype(int)
    data.drop('month', axis=1, inplace=True)
    
    # Create one-hot encoding for crop
    crop_dummies = pd.get_dummies(data['Crop'], prefix='Crop')
    data = pd.concat([data, crop_dummies], axis=1)
    
    # Sort by timestamp for creating lag features
    data.sort_values(['Crop', 'Timestamp'], inplace=True)
    
    # Create lag features by crop
    for crop in data['Crop'].unique():
        crop_mask = data['Crop'] == crop
        
        # Create lag features (1, 2, 3 days)
        for lag in range(1, 4):
            data.loc[crop_mask, f'price_lag_{lag}'] = data.loc[crop_mask, 'Price per kg'].shift(lag)
            data.loc[crop_mask, f'volume_lag_{lag}'] = data.loc[crop_mask, 'Volume Sold per Cycle'].shift(lag)
            data.loc[crop_mask, f'dump_lag_{lag}'] = data.loc[crop_mask, 'Dump Amount'].shift(lag)
    
    # Create rolling statistics
    for crop in data['Crop'].unique():
        crop_mask = data['Crop'] == crop
        
        # 3-day rolling stats
        data.loc[crop_mask, 'price_roll_mean_3'] = data.loc[crop_mask, 'Price per kg'].rolling(3).mean().shift(1)
        data.loc[crop_mask, 'price_roll_std_3'] = data.loc[crop_mask, 'Price per kg'].rolling(3).std().shift(1)
        
        # 7-day rolling stats
        data.loc[crop_mask, 'price_roll_mean_7'] = data.loc[crop_mask, 'Price per kg'].rolling(7).mean().shift(1)
        data.loc[crop_mask, 'price_roll_std_7'] = data.loc[crop_mask, 'Price per kg'].rolling(7).std().shift(1)
    
    # Calculate derived features
    data['price_to_volume_ratio'] = data['Price per kg'] / data['Volume Sold per Cycle'].replace(0, 1)
    data['dump_to_volume_ratio'] = data['Dump Amount'] / data['Volume Sold per Cycle'].replace(0, 1)
    
    # Calculate price difference and percentage change
    data['price_diff'] = data.groupby('Crop')['Price per kg'].diff()
    data['price_pct_change'] = data.groupby('Crop')['Price per kg'].pct_change() * 100
    
    # Calculate aggregate statistics
    crop_avg_price = data.groupby('Crop')['Price per kg'].transform('mean')
    data['crop_avg_price'] = crop_avg_price
    
    crop_price_volatility = data.groupby('Crop')['Price per kg'].transform('std')
    data['crop_price_volatility'] = crop_price_volatility
    
    daily_avg_price = data.groupby('Timestamp')['Price per kg'].transform('mean')
    data['market_avg_price'] = daily_avg_price
    
    # Handle missing values from lag and rolling features
    data = data.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return data
# Function to forecast market demand using the ML model
def forecast_market_demand(historical_data, forecast_days=30, model=None, preprocessor=None):
    """Forecast market prices for crops using the model or fallback method"""
    # Check if we should use the actual model or fallback
    use_model = model is not None and preprocessor is not None
    
    # Try using the ML model if available
    if use_model:
        try:
            st.info("Using the XGBoost market forecast model...")
            
            # Preprocess historical data
            processed_data = preprocess_market_data(historical_data)
            
            # Get latest data for each crop to use as starting point
            latest_data = processed_data.sort_values('Timestamp').groupby('Crop').last().reset_index()
            
            # Get the last date in the dataset
            last_date = historical_data['Timestamp'].max()
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            # Create future dates for prediction
            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            # Store predictions
            predictions = []
            
            # Get the expected feature names from the model if available
            expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
            # For each crop, predict prices for all future dates
            for crop in historical_data['Crop'].unique():
                # Get the most recent data for this crop
                crop_latest = latest_data[latest_data['Crop'] == crop].copy()
                
                # For each future date
                for i, future_date in enumerate(future_dates):
                    # Create a new data point for prediction
                    future_row = crop_latest.copy()
                    
                    # Update temporal features
                    future_row['Year'] = future_date.year
                    future_row['Month'] = future_date.month
                    future_row['Quarter'] = (future_date.month - 1) // 3 + 1
                    future_row['Day_of_Week'] = future_date.weekday()
                    future_row['Day_of_Month'] = future_date.day
                    future_row['Is_Weekend'] = 1 if future_date.weekday() >= 5 else 0
                    future_row['Week_of_Year'] = future_date.isocalendar()[1]
                    future_row['Days_Since_Start'] = (future_date - pd.to_datetime(historical_data['Timestamp'].min())).days
                    
                    # Update season
                    month = future_date.month
                    future_row['Season_Winter'] = int(month == 12 or month <= 2)
                    future_row['Season_Spring'] = int(3 <= month <= 5)
                    future_row['Season_Summer'] = int(6 <= month <= 8)
                    future_row['Season_Fall'] = int(9 <= month <= 11)
                    
                    # Prepare features for prediction
                    future_row_for_prediction = future_row.copy()
                    
                    # Drop non-numeric and unwanted columns
                    columns_to_drop = ['Timestamp']
                    if 'Price per kg' in future_row_for_prediction.columns:
                        columns_to_drop.append('Price per kg')
                        
                    for col in columns_to_drop:
                        if col in future_row_for_prediction.columns:
                            future_row_for_prediction = future_row_for_prediction.drop(col, axis=1)
                    
                    # Additionally remove any other non-numeric columns
                    for col in list(future_row_for_prediction.columns):
                        if future_row_for_prediction[col].dtype == 'object' or future_row_for_prediction[col].dtype.name.startswith('datetime'):
                            future_row_for_prediction = future_row_for_prediction.drop(col, axis=1)
                    
                    # Handle feature alignment if we know what the model expects
                    if expected_features is not None:
                        # Add missing columns with zeros
                        for feature in expected_features:
                            if feature not in future_row_for_prediction.columns:
                                future_row_for_prediction[feature] = 0
                                
                        # Select only the expected columns in the right order
                        try:
                            future_row_for_prediction = future_row_for_prediction[expected_features]
                        except KeyError as e:
                            st.warning(f"Missing expected feature: {e}")
                            # Print available columns for debugging
                            st.write("Available columns:", future_row_for_prediction.columns.tolist())
                            # Cannot continue without the right features
                            raise
                    
                    try:
                        # Ensure we have a DataFrame with proper index
                        prediction_input = future_row_for_prediction.reset_index(drop=True)
                        # Make the prediction
                        price_prediction = model.predict(prediction_input)[0]
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        # Fallback to a simple trend-based prediction
                        base_price = crop_latest['Price per kg'].mean() if 'Price per kg' in crop_latest.columns else 12.0
                        trend = 0.01  # Small upward trend
                        days_ahead = i + 1
                        noise = np.random.normal(0, 0.02)
                        price_prediction = base_price * (1 + trend * days_ahead / 30 + noise)
                    
                    # Store prediction
                    predictions.append({
                        'Timestamp': future_date,
                        'Crop': crop,
                        'Predicted_Price': round(price_prediction, 2),
                        'Days_Ahead': i + 1
                    })
                    
                    # Update latest data with the prediction for next iteration
                    crop_latest['Price per kg'] = price_prediction
                    
                    # Update lag features
                    for lag in range(3, 0, -1):
                        if lag > 1:
                            future_row[f'price_lag_{lag}'] = future_row[f'price_lag_{lag-1}']
                        else:
                            future_row['price_lag_1'] = price_prediction
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            # Package the results
            daily_df = predictions_df
            
            # Create weekly aggregation
            weekly_df = daily_df.copy()
            weekly_df['Week'] = weekly_df['Timestamp'].dt.isocalendar().week
            weekly_df = weekly_df.groupby(['Crop', 'Week']).agg(
                Avg_Weekly_Price=('Predicted_Price', 'mean'),
                First_Date=('Timestamp', 'min')
            ).reset_index()
            weekly_df['Avg_Weekly_Price'] = weekly_df['Avg_Weekly_Price'].round(2)
            
            # Create monthly aggregation
            monthly_df = daily_df.copy()
            monthly_df['Month'] = monthly_df['Timestamp'].dt.month
            monthly_df = monthly_df.groupby(['Crop', 'Month']).agg(
                Avg_Monthly_Price=('Predicted_Price', 'mean'),
                First_Date=('Timestamp', 'min')
            ).reset_index()
            monthly_df['Avg_Monthly_Price'] = monthly_df['Avg_Monthly_Price'].round(2)
            
            return {
                'daily': daily_df,
                'weekly': weekly_df,
                'monthly': monthly_df
            }
        
        except Exception as e:
            st.error(f"Error using forecast model: {e}")
            st.info("Falling back to simulation-based forecasts")
    
    # If no model or error occurred, use the fallback method
    return get_forecast_summary(historical_data, periods=forecast_days)
# Function to generate resource configurations
def generate_resource_configurations(crop, target_yield, model=None, preprocessor=None, num_configs=5):
    """Generate multiple resource configurations using the model or fallback method"""
    # Try using the ML model if available
    if model is not None and preprocessor is not None:
        try:
            st.info(f"Using neural network to generate resource configurations for {crop}...")
            
            # Create input DataFrame with the crop name
            crop_df = pd.DataFrame({'Crop': [crop] * num_configs})
            
            # Transform with the preprocessor
            try:
                crop_input = preprocessor.transform(crop_df)
            except Exception as e:
                st.error(f"Error in preprocessing: {e}")
                st.warning("Model not working properly. Using predefined values.")
                raise ValueError("Preprocessing failed")
            
            # Generate multiple configurations using the model
            configurations = []
            
            # Define crop-specific parameters based on the actual data sample
            crop_params = {
                'Basil': {
                    'Light': 185,           # ~185 in sample data
                    'Temperature': 21,      # ~21 in sample data
                    'Humidity': 63,         # ~63 in sample data
                    'CO2': 700,             # ~700 in sample data
                    'Soil_Moisture': 60,    # ~60 in sample data
                    'pH': 6.2,              # ~6.2 in sample data
                    'EC': 1.5               # ~1.5 in sample data
                },
                'Cilantro': {
                    'Light': 115,           # ~115 in sample data
                    'Temperature': 20,      # ~20 in sample data
                    'Humidity': 65,         # ~65 in sample data
                    'CO2': 700,             # ~700 in sample data
                    'Soil_Moisture': 60,    # ~60 in sample data
                    'pH': 6.2,              # ~6.2 in sample data
                    'EC': 1.5               # ~1.5 in sample data
                },
                'Kale': {
                    'Light': 150,           # estimated based on other greens
                    'Temperature': 18,      # estimated
                    'Humidity': 60,         # estimated
                    'CO2': 650,             # estimated
                    'Soil_Moisture': 60,    # ~60 in sample data
                    'pH': 6.2,              # ~6.2 in sample data
                    'EC': 1.5               # ~1.5 in sample data
                },
                'Lettuce': {
                    'Light': 130,           # ~130 in sample data
                    'Temperature': 18.5,    # ~18.5 in sample data
                    'Humidity': 61,         # ~61 in sample data
                    'CO2': 600,             # ~600 in sample data
                    'Soil_Moisture': 60,    # ~60 in sample data
                    'pH': 6.2,              # ~6.2 in sample data
                    'EC': 1.35              # ~1.35 in sample data
                },
                'Spinach': {
                    'Light': 125,           # ~125 in sample data
                    'Temperature': 17.5,    # ~17.5 in sample data
                    'Humidity': 60,         # ~60 in sample data
                    'CO2': 600,             # ~600 in sample data
                    'Soil_Moisture': 60,    # ~60 in sample data
                    'pH': 6.2,              # ~6.2 in sample data
                    'EC': 1.45              # ~1.45 in sample data
                }
            }
            
            # Get the base parameters for this crop
            base_params = crop_params.get(crop, crop_params['Lettuce'])
            
            for i in range(num_configs):
                try:
                    # Call model predict with the transformed input
                    resources = model.predict(crop_input)
                    
                    # Extract resource values
                    if isinstance(resources, list):
                        resource_values = resources[0]
                    else:
                        # Get the correct row if we have a batch
                        resource_values = resources[i] if i < len(resources) else resources[0]
                    
                    # Build configuration based on model output but constrain to realistic ranges
                    config = {
                        'Configuration': i + 1,
                        'Light': float(max(100, min(200, float(resource_values[0]) * 200))),
                        'Temperature': float(max(15, min(25, float(resource_values[1]) * 25))),
                        'Humidity': float(max(50, min(70, float(resource_values[2]) * 70))),
                        'CO2': float(max(500, min(900, float(resource_values[3]) * 900))),
                        'Soil_Moisture': float(max(50, min(70, float(resource_values[4]) * 70))),
                        'pH': float(max(5.5, min(7.0, float(resource_values[5]) * 1.5 + 5.5))),
                        'EC': float(max(1.0, min(2.0, float(resource_values[6]) * 2.0)))
                    }
                
                except Exception as e:
                    st.error(f"Error in model prediction: {e}")
                    st.warning("Model prediction failed. Using predefined values.")
                    
                    # Create a configuration with parameters specific to this crop
                    config = {
                        'Configuration': i + 1,
                        'Light': float(base_params['Light'] + (i - 2) * 10),        # Vary light levels between configs
                        'Temperature': float(base_params['Temperature'] + (i - 2) * 0.5),  # Vary temperature
                        'Humidity': float(base_params['Humidity'] + (i - 2) * 1),   # Vary humidity
                        'CO2': float(base_params['CO2'] + (i - 2) * 25),            # Vary CO2
                        'Soil_Moisture': float(base_params['Soil_Moisture'] + (i - 2) * 1),  # Vary soil moisture
                        'pH': float(base_params['pH'] + (i - 2) * 0.05),            # Vary pH
                        'EC': float(base_params['EC'] + (i - 2) * 0.05)             # Vary EC
                    }
                
                # Adjust for target yield - scale based on target percentage difference
                # Most microgreen data is around 80-90 yield
                baseline_yield = 85
                if target_yield > baseline_yield:
                    # Scale certain parameters for higher yield targets
                    yield_factor = target_yield / baseline_yield  # Calculate how much higher the target is
                    scaling_factor = min(1.15, 1 + (yield_factor - 1) * 0.3)  # Cap scaling at 15%
                    
                    # Only adjust parameters that scale with yield (light, CO2, nutrients)
                    config['Light'] = min(200, config['Light'] * scaling_factor)
                    config['CO2'] = min(900, config['CO2'] * scaling_factor)
                    config['EC'] = min(2.0, config['EC'] * scaling_factor)
                
                configurations.append(config)
            
            return configurations
            
        except Exception as e:
            st.error(f"Error using resource generator model: {e}")
            st.warning("Model not working. Using predefined crop-specific values.")
    
    # If no model or error, use predefined values
    st.info("Using predefined crop-specific parameters")
    
    # Define deterministic crop-specific values based on the actual data sample
    crop_params = {
        'Basil': {
            'Light': 185,           # ~185 in sample data
            'Temperature': 21,      # ~21 in sample data
            'Humidity': 63,         # ~63 in sample data
            'CO2': 700,             # ~700 in sample data
            'Soil_Moisture': 60,    # ~60 in sample data
            'pH': 6.2,              # ~6.2 in sample data
            'EC': 1.5               # ~1.5 in sample data
        },
        'Cilantro': {
            'Light': 115,           # ~115 in sample data
            'Temperature': 20,      # ~20 in sample data
            'Humidity': 65,         # ~65 in sample data
            'CO2': 700,             # ~700 in sample data
            'Soil_Moisture': 60,    # ~60 in sample data
            'pH': 6.2,              # ~6.2 in sample data
            'EC': 1.5               # ~1.5 in sample data
        },
        'Kale': {
            'Light': 150,           # estimated based on other greens
            'Temperature': 18,      # estimated
            'Humidity': 60,         # estimated
            'CO2': 650,             # estimated
            'Soil_Moisture': 60,    # ~60 in sample data
            'pH': 6.2,              # ~6.2 in sample data
            'EC': 1.5               # ~1.5 in sample data
        },
        'Lettuce': {
            'Light': 130,           # ~130 in sample data
            'Temperature': 18.5,    # ~18.5 in sample data
            'Humidity': 61,         # ~61 in sample data
            'CO2': 600,             # ~600 in sample data
            'Soil_Moisture': 60,    # ~60 in sample data
            'pH': 6.2,              # ~6.2 in sample data
            'EC': 1.35              # ~1.35 in sample data
        },
        'Spinach': {
            'Light': 125,           # ~125 in sample data
            'Temperature': 17.5,    # ~17.5 in sample data
            'Humidity': 60,         # ~60 in sample data
            'CO2': 600,             # ~600 in sample data
            'Soil_Moisture': 60,    # ~60 in sample data
            'pH': 6.2,              # ~6.2 in sample data
            'EC': 1.45              # ~1.45 in sample data
        }
    }
    
    # Get base values for this crop
    base_params = crop_params.get(crop, crop_params['Lettuce'])
    
    # Create multiple configurations with systematic variations
    configs = []
    for i in range(num_configs):
        # Create a configuration with parameters specific to this crop
        # Each configuration varies systematically from the base parameters
        config = {
            'Configuration': i + 1,
            'Light': float(base_params['Light'] + (i - 2) * 10),        # Vary light levels between configs
            'Temperature': float(base_params['Temperature'] + (i - 2) * 0.5),  # Vary temperature
            'Humidity': float(base_params['Humidity'] + (i - 2) * 1),   # Vary humidity
            'CO2': float(base_params['CO2'] + (i - 2) * 25),            # Vary CO2
            'Soil_Moisture': float(base_params['Soil_Moisture'] + (i - 2) * 1),  # Vary soil moisture
            'pH': float(base_params['pH'] + (i - 2) * 0.05),            # Vary pH
            'EC': float(base_params['EC'] + (i - 2) * 0.05)             # Vary EC
        }
        
        # Adjust for target yield - scale based on target percentage difference
        # Most microgreen data is around 80-90 yield
        baseline_yield = 85
        if target_yield > baseline_yield:
            # Scale certain parameters for higher yield targets
            yield_factor = target_yield / baseline_yield  # Calculate how much higher the target is
            scaling_factor = min(1.15, 1 + (yield_factor - 1) * 0.3)  # Cap scaling at 15%
            
            # Only adjust parameters that scale with yield (light, CO2, nutrients)
            config['Light'] = min(200, config['Light'] * scaling_factor)
            config['CO2'] = min(900, config['CO2'] * scaling_factor)
            config['EC'] = min(2.0, config['EC'] * scaling_factor)
        
        configs.append(config)
    
    return configs


# Function to predict yield for resource configurations
def predict_yield(crop, resource_configs, model=None, preprocessor=None):
    """Predict yield for each resource configuration using the model or fallback method"""
    # Try using the ML model if available
    if model is not None and preprocessor is not None:
        try:
            st.info(f"Using neural network to predict yields for {crop}...")
            
            # Prepare input data for the model
            configs_with_yield = []
            
            for config in resource_configs:
                try:
                    # Create DataFrame with crop and resource values
                    input_df = pd.DataFrame({
                        'Crop': [crop],
                        'Light': [float(config['Light'])],
                        'Temperature': [float(config['Temperature'])],
                        'Humidity': [float(config['Humidity'])],
                        'CO2': [float(config['CO2'])],
                        'Soil Moisture': [float(config['Soil_Moisture'])],
                        'pH': [float(config['pH'])],
                        'EC': [float(config['EC'])]
                    })
                    
                    # Preprocess the input
                    model_input = preprocessor.transform(input_df)
                    
                    # Predict yield
                    predicted_yield = model.predict(model_input)
                    
                    # Extract scalar value if needed
                    if isinstance(predicted_yield, np.ndarray):
                        if predicted_yield.ndim > 1:
                            predicted_yield = predicted_yield[0][0]
                        else:
                            predicted_yield = predicted_yield[0]
                    
                    # Apply realistic constraints based on actual data
                    crop_yield_ranges = {
                        'Basil': (75, 105),
                        'Cilantro': (70, 90),
                        'Kale': (80, 105),
                        'Lettuce': (70, 90),
                        'Spinach': (75, 95)
                    }
                    
                    # Get realistic yield range for this crop
                    yield_range = crop_yield_ranges.get(crop, (70, 95))
                    min_yield, max_yield = yield_range
                    
                    # Constrain prediction to realistic values
                    constrained_yield = max(min_yield * 0.8, min(max_yield * 1.1, float(predicted_yield)))
                    
                    # Add yield to configuration
                    config_with_yield = config.copy()
                    config_with_yield['Predicted_Yield'] = round(constrained_yield, 2)
                    
                except Exception as e:
                    st.error(f"Error in yield prediction: {e}")
                    st.warning("Model prediction failed. Using predefined yield values.")
                    
                    # Use deterministic yield calculation when model fails
                    config_with_yield = config.copy()
                    config_with_yield['Predicted_Yield'] = calculate_deterministic_yield(crop, config)
                
                configs_with_yield.append(config_with_yield)
            
            # Sort by predicted yield
            configs_with_yield = sorted(configs_with_yield, 
                                      key=lambda x: x['Predicted_Yield'], 
                                      reverse=True)
            
            return configs_with_yield
            
        except Exception as e:
            st.error(f"Error using yield predictor model: {e}")
            st.warning("Yield prediction model not working. Using predefined values.")
    
    # If no model or error, use deterministic approach
    st.info("Using predefined yield calculation method")
    
    configs_with_yield = []
    for config in resource_configs:
        # Calculate yield deterministically
        yield_value = calculate_deterministic_yield(crop, config)
        
        # Add yield to configuration
        config_with_yield = config.copy()
        config_with_yield['Predicted_Yield'] = yield_value
        
        configs_with_yield.append(config_with_yield)
    
    # Sort by predicted yield
    configs_with_yield = sorted(configs_with_yield, 
                              key=lambda x: x['Predicted_Yield'], 
                              reverse=True)
    
    return configs_with_yield

def calculate_deterministic_yield(crop, config):
    """Calculate yield deterministically based on crop type and resource configuration"""
    # Define ideal ranges for each crop (optimal growing conditions)
    ideal_ranges = {
        "Basil": {
            'Light': (170, 200),
            'Temperature': (20, 22),
            'Humidity': (60, 68),
            'CO2': (650, 800),
            'Soil_Moisture': (60, 70),
            'pH': (6.0, 6.5),
            'EC': (1.4, 1.8)
        },
        "Cilantro": {
            'Light': (100, 125),
            'Temperature': (18, 20),
            'Humidity': (60, 68),
            'CO2': (600, 700),
            'Soil_Moisture': (60, 70),
            'pH': (6.0, 6.3),
            'EC': (1.4, 1.6)
        },
        "Kale": {
            'Light': (150, 200),
            'Temperature': (16, 19),
            'Humidity': (60, 68),
            'CO2': (600, 700),
            'Soil_Moisture': (60, 70),
            'pH': (6.0, 6.3),
            'EC': (1.4, 1.6)
        },
        "Lettuce": {
            'Light': (125, 150),
            'Temperature': (18, 20),
            'Humidity': (60, 68),
            'CO2': (600, 700),
            'Soil_Moisture': (60, 70),
            'pH': (6.0, 6.3),
            'EC': (1.3, 1.5)
        },
        "Spinach": {
            'Light': (125, 150),
            'Temperature': (17, 19),
            'Humidity': (55, 65),
            'CO2': (550, 650),
            'Soil_Moisture': (58, 65),
            'pH': (6.0, 6.3),
            'EC': (1.4, 1.6)
        }
    }
    
    # Define base yields for each crop at ideal conditions based on actual data
    base_yields = {
        "Basil": 90,
        "Cilantro": 80,
        "Kale": 95,
        "Lettuce": 80,
        "Spinach": 85
    }
    
    # Get ideal range for this crop (or default to lettuce)
    crop_range = ideal_ranges.get(crop, ideal_ranges["Lettuce"])
    
    # Calculate how close each parameter is to its ideal range (0.0 to 1.0)
    param_scores = {}
    for param, (min_val, max_val) in crop_range.items():
        current_val = config[param]
        if min_val <= current_val <= max_val:
            # Within ideal range - maximum score
            param_scores[param] = 1.0
        else:
            # Outside ideal range, calculate distance as a percentage of the range
            range_size = max_val - min_val
            if current_val < min_val:
                distance = (min_val - current_val) / range_size
            else:
                distance = (current_val - max_val) / range_size
            
            # Convert distance to score (0.0 to 1.0)
            param_scores[param] = max(0.0, 1.0 - distance)
    
    # Weight the importance of different parameters
    weights = {
        'Light': 0.25,         # Most important for photosynthesis
        'Temperature': 0.20,    # Very important for growth rates
        'Humidity': 0.10,       # Affects transpiration
        'CO2': 0.15,            # Important for photosynthesis
        'Soil_Moisture': 0.10,  # Water availability
        'pH': 0.10,             # Nutrient availability
        'EC': 0.10              # Nutrient concentration
    }
    
    # Calculate overall score (weighted average)
    overall_score = sum(param_scores.get(param, 0) * weights.get(param, 0) 
                        for param in weights.keys()) / sum(weights.values())
    
    # Get base yield for this crop (or default to 80)
    base_yield = base_yields.get(crop, 80)
    
    # Calculate final yield deterministically - simplified linear scaling
    # Convert overall_score (0-1) to a yield multiplier (0.7-1.1)
    yield_multiplier = 0.7 + (overall_score * 0.4)
    
    # Calculate final yield
    calculated_yield = base_yield * yield_multiplier
    
    # Round to 2 decimal places
    return round(calculated_yield, 2)

# Function to format resource values for display
def format_resource_display(resource_config):
    """Format resource values with appropriate units for display"""
    return {
        'Light': f"{resource_config['Light']:.1f} μmol/m²/s",
        'Temperature': f"{resource_config['Temperature']:.1f} °C",
        'Humidity': f"{resource_config['Humidity']:.1f} %",
        'CO2': f"{resource_config['CO2']:.0f} ppm",
        'Soil Moisture': f"{resource_config['Soil_Moisture']:.1f} %",
        'pH': f"{resource_config['pH']:.1f}",
        'EC': f"{resource_config['EC']:.2f} mS/cm"
    }

# Function to create radar chart for resource visualization
def create_resource_radar_chart(config, title=None):
    """Create radar chart for visualizing resource configurations"""
    # Categories for radar chart
    categories = ['Light', 'Temperature', 'Humidity', 
                 'CO2', 'Soil Moisture', 'pH', 'EC']
    
    # Normalize values for radar chart based on typical ranges
    normalized_values = [
        config['Light'] / 400,  # Light intensity (μmol/m²/s)
        (config['Temperature'] - 15) / 15,  # Temperature (°C)
        (config['Humidity'] - 40) / 50,  # Humidity (%)
        (config['CO2'] - 400) / 1100,  # CO2 concentration (ppm)
        config['Soil_Moisture'] / 100,  # Soil moisture (%)
        (config['pH'] - 5.5) / 2,  # pH level
        (config['EC'] - 0.5) / 3  # Electrical conductivity (mS/cm)
    ]
    
    # IMPORTANT: Make sure both lists have the same length before closing the loop
    # Close the loop for radar chart
    categories_closed = categories + [categories[0]]
    normalized_values_closed = normalized_values + [normalized_values[0]]
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    # Set the angles for each category (IMPORTANT: must match the length of categories)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    # Close the angle loop as well
    angles_closed = angles + [angles[0]]
    
    # Plot data - using the closed versions of both arrays
    ax.plot(angles_closed, normalized_values_closed, linewidth=2, linestyle='solid', color='#2E7D32')
    ax.fill(angles_closed, normalized_values_closed, alpha=0.25, color='#4CAF50')
    
    # Set category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, size=9)
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Add title if provided
    if title:
        plt.title(title)
    
    plt.tight_layout()
    return fig

# Apply custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
    }
    .subheader {
        font-size: 1.5rem;
        color: #33691E;
    }
    .step-header {
        font-size: 1.8rem;
        color: #2E7D32;
        padding-top: 1rem;
        padding-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        color: #333;
    }
    .instruction-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
        margin-bottom: 1rem;
        color: #333;
        border-left: 4px solid #2E7D32;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
        flex: 1;
        min-width: 150px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .rank-1 {
        background-color: #1B5E20;
        color: white;
        font-weight: bold;
    }
    .rank-2 {
        background-color: #2E7D32;
        color: white;
    }
    .rank-3 {
        background-color: #388E3C;
        color: white;
    }
    .rank-4 {
        background-color: #43A047;
        color: white;
    }
    .rank-5 {
        background-color: #4CAF50;
        color: white;
    }
    .inventory-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #2E7D32;
        color: white;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .in-stock {
        background-color: #E8F5E9;
        border-left: 4px solid #2E7D32;
    }
    .resource-table {
        width: 100%;
        border-collapse: collapse;
    }
    .resource-table th {
        background-color: #2E7D32;
        color: white;
        padding: 8px;
        text-align: left;
    }
    .resource-table td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .stButton button {
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .step-container {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 2rem;
        background-color: white;
    }
    .crop-ranking-item {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .crop-name {
        font-weight: bold;
    }
    .crop-price {
        font-size: 1.2rem;
    }
    .crop-inventory {
        background-color: #FFCDD2;
        color: #333;
        padding: 5px;
        border-radius: 5px;
        margin-left: 10px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Common crops in vertical farming
CROPS = ["Basil", "Cilantro", "Kale", "Lettuce", "Spinach"]
CROP_CYCLE_DAYS = 30  # Default crop cycle length in days

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'inventory' not in st.session_state:
    st.session_state.inventory = {}
if 'cycle_end_date' not in st.session_state:
    st.session_state.cycle_end_date = (datetime.now() + timedelta(days=7)).date()
if 'target_yields' not in st.session_state:
    st.session_state.target_yields = {crop: 85.0 for crop in CROPS}
if 'resource_configs' not in st.session_state:
    st.session_state.resource_configs = {}

# Define functions for navigation
def next_step():
    st.session_state.step += 1

def go_to_step(step):
    st.session_state.step = step

def restart_app():
    """Reset session state completely and reinitialize core variables"""
    # Store the current step
    current_step = st.session_state.step
    
    # Create a temporary dictionary of keys to delete
    keys_to_delete = [key for key in st.session_state.keys() if key != "step"]
    
    # Delete all keys except step
    for key in keys_to_delete:
        del st.session_state[key]
    
    # Reinitialize core variables
    st.session_state.inventory = {}
    st.session_state.cycle_end_date = (datetime.now() + timedelta(days=7)).date()
    st.session_state.target_yields = {crop: 85.0 for crop in CROPS}
    st.session_state.resource_configs = {}
    
    # Set step to 1 after reinitializing variables
    st.session_state.step = 1
    
    # Force a rerun to update the UI
    st.rerun()

# Sidebar
st.sidebar.markdown("<h2 class='subheader'>Vertical Farming Resource Optimizer</h2>", unsafe_allow_html=True)
try:
    st.sidebar.image("https://raw.githubusercontent.com/HarshTiwari1710/Resource-Competition-Modelling/refs/heads/main/logo.png", use_column_width=True)
except:
    # If image can't be loaded, just display text
    st.sidebar.markdown("### 🌱 Vertical Farming")

# Navigation in sidebar
st.sidebar.markdown("### Navigation")
current_step = st.session_state.step
steps = ["1. Set Cycle End Date", "2. View Price Forecasts", "3. Update Inventory", "4. Get Recommendations"]

for i, step_name in enumerate(steps, 1):
    if i < current_step:
        # Steps that can be navigated to
        if st.sidebar.button(f"✓ {step_name}", key=f"nav_{i}"):
            go_to_step(i)
    elif i == current_step:
        # Current step (highlighted)
        st.sidebar.markdown(f"**→ {step_name}**", unsafe_allow_html=True)
    else:
        # Future steps (disabled)
        st.sidebar.markdown(f"○ {step_name}", unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Start Over", key="restart_sidebar"):
    restart_app()

# Load models
market_model, market_model_loaded = load_market_forecast_model()
resource_model, resource_model_loaded = load_resource_generator_model()
yield_model, yield_model_loaded = load_yield_predictor_model()
preprocessors = load_preprocessors()

# Main Title
st.markdown("<h1 class='main-header'>Vertical Farming Resource Optimizer</h1>", unsafe_allow_html=True)

# Step 1: Cycle End Date Input
if st.session_state.step == 1:
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='step-header'>Step 1: Select Current Cycle End Date</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-card">
    <p>Please specify when your current growing cycle will end. This helps us provide accurate market forecasts and resource recommendations for your next growing cycle.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date input
    cycle_end_date = st.date_input(
        "When does your current growing cycle end?",
        value=st.session_state.cycle_end_date,
        min_value=datetime.now().date()
    )
    st.session_state.cycle_end_date = cycle_end_date
    
    # Crop cycle length
    cycle_length = st.slider(
        "Average crop cycle length (days)",
        min_value=20,
        max_value=45,
        value=CROP_CYCLE_DAYS,
        step=1
    )
    st.session_state.crop_cycle_days = cycle_length
    
    # Next button
    if st.button("Generate Price Forecast", key="generate_forecast"):
        # Convert to datetime for processing
        cycle_end_datetime = datetime.combine(cycle_end_date, datetime.min.time())
        
        # Generate forecast data
        with st.spinner("Generating price forecasts..."):
            # Generate sample data as a starting point
            historical_data = generate_sample_data(cycle_end_datetime)
            
            # Get forecast for next 90 days
            forecast_results = forecast_market_demand(
                historical_data,
                forecast_days=90,
                model=market_model if market_model_loaded else None,
                preprocessor=preprocessors.get('market_forecast') if preprocessors.get('market_forecast') else None
            )
            
            # Store in session state
            st.session_state.daily_forecast = forecast_results['daily']
            st.session_state.weekly_forecast = forecast_results['weekly']
            st.session_state.monthly_forecast = forecast_results['monthly']
            st.session_state.forecast_generated = True
            
            # Calculate next cycle dates
            next_cycle_start = cycle_end_datetime
            next_cycle_end = next_cycle_start + timedelta(days=cycle_length)
            
            # Filter forecast for next cycle
            next_cycle_forecast = st.session_state.daily_forecast[
                (st.session_state.daily_forecast['Timestamp'] >= pd.to_datetime(next_cycle_start)) &
                (st.session_state.daily_forecast['Timestamp'] <= pd.to_datetime(next_cycle_end))
            ]
            
            # Calculate average prices for next cycle
            next_cycle_avg = next_cycle_forecast.groupby('Crop')['Predicted_Price'].mean().reset_index()
            
            # Create ranking DataFrame
            ranking_data = []
            for _, row in next_cycle_avg.iterrows():
                crop = row['Crop']
                price = row['Predicted_Price']
                ranking_data.append({
                    'Crop': crop,
                    'Next Cycle Avg Price': round(price, 2),
                    'Overall Score': round(price, 2)
                })
            
            # Create DataFrame and sort by price
            ranking_df = pd.DataFrame(ranking_data).sort_values('Overall Score', ascending=False)
            
            # Save to session state
            st.session_state.crop_ranking = ranking_df
            st.session_state.next_cycle_start = next_cycle_start
            st.session_state.next_cycle_end = next_cycle_end
            
            # Move to next step
            next_step()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Step 2: Price Forecasts and Rankings
elif st.session_state.step == 2:
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='step-header'>Step 2: Review Price Forecasts & Crop Rankings</h2>", unsafe_allow_html=True)
    
    # Check if we have forecast data
    if 'forecast_generated' in st.session_state and st.session_state.forecast_generated:
        # Display cycle information
        cycle_end_date = st.session_state.cycle_end_date
        next_cycle_start = st.session_state.next_cycle_start
        next_cycle_end = st.session_state.next_cycle_end
        
        st.markdown(f"""
        <div class="instruction-card">
        <p><strong>Current Cycle End Date:</strong> {cycle_end_date.strftime('%B %d, %Y')}</p>
        <p><strong>Next Cycle Period:</strong> {next_cycle_start.strftime('%B %d, %Y')} to {next_cycle_end.strftime('%B %d, %Y')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display crop rankings
        st.markdown("<h3>Crop Rankings for Next Cycle</h3>", unsafe_allow_html=True)
        
        # Get ranking data
        ranking_df = st.session_state.crop_ranking.copy()
        
        # Display vertical bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Sort by overall score
        sorted_df = ranking_df.sort_values('Overall Score', ascending=True)
        
        # Define colors based on rank
        colors = ['#1B5E20', '#2E7D32', '#388E3C', '#43A047', '#4CAF50']
        
        # Create vertical bar chart
        bars = ax.barh(sorted_df['Crop'], sorted_df['Overall Score'], color=colors[:len(sorted_df)])
        
        # Add value labels
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + 0.2,
                bar.get_y() + bar.get_height()/2,
                f"${sorted_df['Overall Score'].iloc[i]:.2f}",
                va='center',
                fontweight='bold',
                color='black'
            )
        
        ax.set_xlabel('Predicted Price ($/kg)')
        ax.set_title('Crop Rankings by Predicted Price', fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Display ranking as styled list
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        for i, (_, row) in enumerate(ranking_df.iterrows()):
            crop = row['Crop']
            price = row['Overall Score']
            
            # Define rank color class
            rank_class = f"rank-{i+1}" if i < 5 else "rank-5"
            
            # Check if in inventory
            inventory_badge = ""
            if crop in st.session_state.inventory:
                inventory_badge = "<span class='crop-inventory'>IN STOCK</span>"
            
            st.markdown(f"""
            <div class='crop-ranking-item {rank_class}'>
                <div><span class='crop-name'>#{i+1}: {crop}</span> {inventory_badge}</div>
                <div class='crop-price'>${price:.2f}/kg</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display price forecast trends
        st.markdown("<h3>Price Forecast Trends</h3>", unsafe_allow_html=True)
        
        # Create a line chart of price forecasts over time
        daily_forecast = st.session_state.daily_forecast
        
        # Allow user to select crops to display
        selected_crops = st.multiselect(
            "Select crops to display forecast trends:",
            options=CROPS,
            default=CROPS[:3]
        )
        
        if selected_crops:
            # Filter forecast for selected crops
            filtered_forecast = daily_forecast[daily_forecast['Crop'].isin(selected_crops)]
            
            # Create line chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for crop in selected_crops:
                crop_data = filtered_forecast[filtered_forecast['Crop'] == crop]
                ax.plot(
                    crop_data['Timestamp'], 
                    crop_data['Predicted_Price'],
                    marker='o',
                    markersize=4,
                    linewidth=2,
                    label=crop
                )
            
            # Highlight next cycle period
            ax.axvspan(
                next_cycle_start, 
                next_cycle_end,
                alpha=0.2,
                color='green',
                label='Next Cycle'
            )
            
            # Add formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Price ($/kg)')
            ax.set_title('Price Forecast Trends', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            fig.tight_layout()
            
            st.pyplot(fig)
        
        # Next & Back buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Back", key="back_to_step1"):
                go_to_step(1)
        with col2:
            if st.button("Continue to Inventory →", key="continue_to_inventory"):
                next_step()
    else:
        st.warning("Please go back to Step 1 and generate price forecasts first.")
        if st.button("← Back to Step 1", key="back_to_step1_from_warning"):
            go_to_step(1)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Step 3: Inventory Management
elif st.session_state.step == 3:
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='step-header'>Step 3: Update Inventory & Set Yield Targets</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="instruction-card">
    <p>Select crops that you already have in stock. Lower priority will be given to in-stock crops since you're already growing them.</p>
    <p>Then set your target yield for each crop to get optimal growing parameters.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Crop Inventory")
        
        # Get ranking data if available, or create default
        if 'crop_ranking' in st.session_state:
            ranking_df = st.session_state.crop_ranking.copy()
        else:
            # Create default ranking
            ranking_data = []
            for crop in CROPS:
                base_price = 14.5 if crop == "Basil" else 12.3 if crop == "Cilantro" else 15.7 if crop == "Kale" else 8.9 if crop == "Lettuce" else 10.2
                ranking_data.append({
                    'Crop': crop,
                    'Next Cycle Avg Price': round(base_price, 2),
                    'Overall Score': round(base_price, 2)
                })
            ranking_df = pd.DataFrame(ranking_data).sort_values('Overall Score', ascending=False)
        
        # Display each crop with checkbox
        st.markdown("<div style='border: 1px solid #ddd; padding: 15px; border-radius: 5px;'>", unsafe_allow_html=True)
        
        # Clear inventory button
        if st.button("Clear All Inventory", key="clear_inventory"):
            st.session_state.inventory = {}
        
        for i, (_, row) in enumerate(ranking_df.iterrows()):
            crop = row['Crop']
            price = row['Overall Score'] if 'Overall Score' in row else 0
            
            # Apply rank-based styling
            rank_class = f"rank-{i+1}" if i < 5 else "rank-5"
            
            # Create columns for layout
            col_a, col_b = st.columns([3, 1])
            
            with col_a:
                # Create a container with conditional styling
                is_in_stock = crop in st.session_state.inventory
                stock_class = " in-stock" if is_in_stock else ""
                st.markdown(f"<div class='crop-item{stock_class}' style='padding: 10px; margin-bottom: 5px; border-radius: 5px;'>", unsafe_allow_html=True)
                st.markdown(f"<strong>{crop}</strong> (${price:.2f}/kg)", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_b:
                # Inventory checkbox
                inventory_status = crop in st.session_state.inventory
                if st.checkbox("In Stock", value=inventory_status, key=f"inv_{crop}"):
                    st.session_state.inventory[crop] = True
                else:
                    if crop in st.session_state.inventory:
                        del st.session_state.inventory[crop]
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("Target Yield Settings")
        
        # Display sliders for yield adjustment
        st.markdown("<div style='border: 1px solid #ddd; padding: 15px; border-radius: 5px;'>", unsafe_allow_html=True)
        
        # Get focus crops (prioritize crops NOT in inventory, then sort by ranking)
        focus_crops = []
        
        # First add crops not in inventory by ranking order
        for _, row in ranking_df.iterrows():
            crop = row['Crop']
            if crop not in st.session_state.inventory and crop not in focus_crops:
                focus_crops.append(crop)
        
        # Then add crops in inventory
        for crop in st.session_state.inventory:
            if crop not in focus_crops:
                focus_crops.append(crop)
        
        # Display yield sliders for each crop
        for crop in focus_crops:
            # Add visual indicator for inventory items
            if crop in st.session_state.inventory:
                st.markdown(f"<div style='padding: 10px; background-color: #FFCDD2; border-left: 4px solid #D32F2F; border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                st.markdown(f"<strong>{crop}</strong> (In Stock)", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='padding: 10px; background-color: #E8F5E9; border-left: 4px solid #2E7D32; border-radius: 5px; margin-bottom: 10px;'>", unsafe_allow_html=True)
                st.markdown(f"<strong>{crop}</strong>", unsafe_allow_html=True)
            
            # Yield slider
            st.session_state.target_yields[crop] = st.slider(
                f"Target Yield (g/tray)",
                min_value=50.0,
                max_value=150.0,
                value=st.session_state.target_yields.get(crop, 85.0),
                step=1.0,
                key=f"yield_{crop}"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Next & Back buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="back_to_step2"):
            go_to_step(2)
    with col2:
        if st.button("Generate Recommendations →", key="generate_recommendations"):
            # Calculate optimal focus crops (prioritize crops NOT in inventory)
            focus_crops = []
            
            # First add crops NOT in inventory by ranking order
            if 'crop_ranking' in st.session_state:
                sorted_ranking = st.session_state.crop_ranking.sort_values('Overall Score', ascending=False)
                for _, row in sorted_ranking.iterrows():
                    crop = row['Crop']
                    if crop not in st.session_state.inventory and crop not in focus_crops:
                        focus_crops.append(crop)
            
            # Then add crops in inventory at the end
            for crop in st.session_state.inventory:
                if crop not in focus_crops:
                    focus_crops.append(crop)
            
            # Limit to top 3 if more than 3
            if len(focus_crops) > 3:
                focus_crops = focus_crops[:3]
            
            # Store in session state
            st.session_state.focus_crops = focus_crops
            
            # Move to next step
            next_step()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Step 4: Resource Recommendations
elif st.session_state.step == 4:
    st.markdown("<div class='step-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='step-header'>Step 4: Resource Recommendations</h2>", unsafe_allow_html=True)
    
    # Get focus crops
    if 'focus_crops' in st.session_state:
        focus_crops = st.session_state.focus_crops
    else:
        # Fallback to all crops
        focus_crops = CROPS.copy()
    
    st.markdown(f"""
    <div class="instruction-card">
    <p>Based on market forecasts and your inventory status, we recommend focusing on these crops for your next growing cycle:</p>
    <p><strong>Focus Crops:</strong> {', '.join(focus_crops)}</p>
    <p>Below are the recommended growing parameters for each crop based on your target yields.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for each focus crop
    cols = st.columns(min(len(focus_crops), 3))
    
    # Get recommendations for each crop
    for i, crop in enumerate(focus_crops):
        with cols[i % len(cols)]:
            # Get target yield for this crop
            target_yield = st.session_state.target_yields.get(crop, 85.0)
            
            # Get price forecast if available
            price_info = ""
            if 'crop_ranking' in st.session_state:
                crop_row = st.session_state.crop_ranking[st.session_state.crop_ranking['Crop'] == crop]
                if not crop_row.empty:
                    price = crop_row['Overall Score'].values[0]
                    price_info = f"<br>Forecast Price: <strong>${price:.2f}/kg</strong>"
            
            # Header with status indicator
            bg_color = "#FFCDD2" if crop in st.session_state.inventory else "#E8F5E9"
            border_color = "#D32F2F" if crop in st.session_state.inventory else "#2E7D32"
            status = "IN STOCK" if crop in st.session_state.inventory else "NEW CROP"
            st.markdown(f"""
            <div style='background-color: {bg_color}; padding: 10px; border-radius: 5px; border-left: 4px solid {border_color}; margin-bottom: 10px; text-align: center;'>
            <h3>{crop}</h3>
            <p>{status}</p>
            <p>Target Yield: <strong>{target_yield:.1f} g/tray</strong>{price_info}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if we have resource configurations for this crop
            if crop in st.session_state.resource_configs:
                # Display the best configuration
                best_config = st.session_state.resource_configs[crop][0]
                
                # Format the values for display
                formatted_resources = format_resource_display(best_config)
                
                # Display as a formatted table
                st.markdown("<h4>Optimal Growing Parameters</h4>", unsafe_allow_html=True)
                st.markdown("""
                <table class="resource-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                """, unsafe_allow_html=True)
                
                for param, value in formatted_resources.items():
                    st.markdown(f"""
                    <tr>
                    <td>{param}</td>
                    <td>{value}</td>
                    </tr>
                    """, unsafe_allow_html=True)
                
                # Add yield if available
                if 'Predicted_Yield' in best_config:
                    st.markdown(f"""
                    <tr>
                    <td><strong>Predicted Yield</strong></td>
                    <td><strong>{best_config['Predicted_Yield']:.2f} g/tray</strong></td>
                    </tr>
                    """, unsafe_allow_html=True)
                
                st.markdown("</table>", unsafe_allow_html=True)
                
                # Create radar chart
                st.markdown("<h4>Resource Profile</h4>", unsafe_allow_html=True)
                radar_fig = create_resource_radar_chart(best_config)
                st.pyplot(radar_fig)
            else:
                # If no configurations yet, generate them from the model or fallback
                st.info(f"Generating optimal resource configurations for {crop}...")
                
                # Generate multiple configurations
                configs = generate_resource_configurations(
                    crop, 
                    target_yield,
                    model=resource_model if resource_model_loaded else None,
                    preprocessor=preprocessors.get('resource_generator') if preprocessors.get('resource_generator') else None,
                    num_configs=5
                )
                
                # Predict yield for each configuration
                configs_with_yield = predict_yield(
                    crop,
                    configs,
                    model=yield_model if yield_model_loaded else None,
                    preprocessor=preprocessors.get('yield_predictor') if preprocessors.get('yield_predictor') else None
                )
                
                # Store in session state
                st.session_state.resource_configs[crop] = configs_with_yield
                
                # Get the best configuration
                best_config = configs_with_yield[0]
                
                # Format the values for display
                formatted_resources = format_resource_display(best_config)
                
                # Display as a formatted table
                st.markdown("<h4>Optimal Growing Parameters</h4>", unsafe_allow_html=True)
                st.markdown("""
                <table class="resource-table">
                <tr><th>Parameter</th><th>Value</th></tr>
                """, unsafe_allow_html=True)
                
                for param, value in formatted_resources.items():
                    st.markdown(f"""
                    <tr>
                    <td>{param}</td>
                    <td>{value}</td>
                    </tr>
                    """, unsafe_allow_html=True)
                
                # Add yield
                st.markdown(f"""
                <tr>
                <td><strong>Predicted Yield</strong></td>
                <td><strong>{best_config['Predicted_Yield']:.2f} g/tray</strong></td>
                </tr>
                """, unsafe_allow_html=True)
                
                st.markdown("</table>", unsafe_allow_html=True)
                
                # Create radar chart
                st.markdown("<h4>Resource Profile</h4>", unsafe_allow_html=True)
                radar_fig = create_resource_radar_chart(best_config)
                st.pyplot(radar_fig)
    
    # Profit projection
    st.markdown("<h3>Profit Projection</h3>", unsafe_allow_html=True)
    
    # Initialize profit data
    profit_data = []
    for crop in focus_crops:
        # Get target yield (now in g/tray)
        target_yield = st.session_state.target_yields.get(crop, 85.0)

        # Get average forecast price ($/kg)
        if 'crop_ranking' in st.session_state:
            crop_row = st.session_state.crop_ranking[st.session_state.crop_ranking['Crop'] == crop]
            if not crop_row.empty:
                avg_price = crop_row['Overall Score'].values[0]
            else:
                # Default price if not in ranking
                avg_price = 12.0
        else:
            # Default price if no ranking
            avg_price = 12.0

        # Get the best configuration and predicted yield
        if crop in st.session_state.resource_configs:
            best_config = st.session_state.resource_configs[crop][0]
            predicted_yield = best_config['Predicted_Yield']
        else:
            predicted_yield = target_yield

        # Convert from g/tray to kg/tray for price calculations
        predicted_yield_kg = predicted_yield / 1000

        # Calculate costs
        # Base cost per tray varies by crop (material, seed, labor costs)
        base_tray_cost = 3.0 if crop == "Basil" else 2.5 if crop == "Cilantro" else 3.2 if crop == "Kale" else 2.0 if crop == "Lettuce" else 2.8
        # Add variable costs based on yield
        variable_cost = 0.01 * predicted_yield  # Higher yields use more resources
        cost_per_tray = base_tray_cost + variable_cost

        # Calculate revenue and profit per tray
        revenue_per_tray = avg_price * predicted_yield_kg  # Price is $/kg, yield is kg/tray
        profit_per_tray = revenue_per_tray - cost_per_tray

        # Calculate ROI
        roi = (profit_per_tray / cost_per_tray) * 100 if cost_per_tray > 0 else 0

        profit_data.append({
            'Crop': crop,
            'Target Yield': target_yield,
            'Predicted Yield': predicted_yield,
            'Forecast Price': avg_price,
            'Revenue': revenue_per_tray,
            'Cost': cost_per_tray,
            'Profit': profit_per_tray,
            'ROI': roi
        })
        
    
    # Create profit DataFrame
    profit_df = pd.DataFrame(profit_data)
    
    # Display styled table
    st.dataframe(profit_df.style.format({
        'Target Yield': '{:.1f} g/tray',
        'Predicted Yield': '{:.1f} g/tray',
        'Forecast Price': '${:.2f}/kg',
        'Revenue': '${:.2f}/tray',
        'Cost': '${:.2f}/tray',
        'Profit': '${:.2f}/tray',
        'ROI': '{:.1f}%'
        }))
    
    # Profit comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define colors based on inventory status
    colors = []
    for crop in profit_df['Crop']:
        if crop in st.session_state.inventory:
            colors.append('#D32F2F')  # Red for in-stock crops
        else:
            colors.append('#2E7D32')  # Green for new crops
    
    bars = ax.bar(profit_df['Crop'], profit_df['Profit'], color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'${height:.2f}',
               ha='center', va='bottom')
    
    ax.set_ylabel('Profit per m² ($)')
    ax.set_title('Projected Profit by Crop')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Multi-configuration comparison
    st.markdown("<h3>Configuration Comparison</h3>", unsafe_allow_html=True)
    
    # Allow user to select a crop to see all configurations
    selected_crop = st.selectbox("Select crop to compare configurations:", focus_crops)
    
    if selected_crop in st.session_state.resource_configs:
        configs = st.session_state.resource_configs[selected_crop]
        
        # Display as a table
        config_data = []
        for i, config in enumerate(configs):
            config_data.append({
                'Configuration': f"Config {i+1}",
                'Light': f"{config['Light']:.1f} μmol/m²/s",
                'Temperature': f"{config['Temperature']:.1f} °C",
                'Humidity': f"{config['Humidity']:.1f} %",
                'CO2': f"{config['CO2']:.0f} ppm",
                'pH': f"{config['pH']:.1f}",
                'EC': f"{config['EC']:.2f} mS/cm",
                'Predicted Yield': f"{config['Predicted_Yield']:.2f} kg/m²" 
            })
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
        
        # Create yield comparison chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        config_names = [f"Config {i+1}" for i in range(len(configs))]
        yields = [config['Predicted_Yield'] for config in configs]
        
        bars = ax.bar(config_names, yields, color='#388E3C')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel('Predicted Yield (kg/m²)')
        ax.set_title(f'Yield Comparison for {selected_crop} Configurations')
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.info(f"No configurations generated yet for {selected_crop}")
    
    # Final recommendations
    st.markdown("<h3>Strategic Recommendations</h3>", unsafe_allow_html=True)
    
    # Get most profitable crop
    if len(profit_df) > 0:
        most_profitable = profit_df.loc[profit_df['Profit'].idxmax()]['Crop']
        highest_profit = profit_df.loc[profit_df['Profit'].idxmax()]['Profit']
        highest_roi = profit_df.loc[profit_df['ROI'].idxmax()]['Crop']
    else:
        most_profitable = focus_crops[0] if focus_crops else CROPS[0]
        highest_profit = 0
        highest_roi = focus_crops[0] if focus_crops else CROPS[0]
    
    # Overall recommendation
    st.markdown(f"""
    <div class="instruction-card">
    <h4>Production Strategy</h4>
    <p>Based on your inputs and our analysis, we recommend:</p>
    <ul>
    <li><strong>Primary Focus:</strong> {most_profitable} (${highest_profit:.2f}/m² projected profit)</li>
    <li><strong>Best ROI:</strong> {highest_roi}</li>
    <li><strong>Resource Allocation:</strong> Use the parameters above to optimize your growing conditions</li>
    </ul>
    <p>For your next growing cycle starting after {st.session_state.cycle_end_date.strftime('%B %d, %Y')}, adjust your growing environment to match these recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Specific recommendations for inventory items
    if st.session_state.inventory:
        st.markdown("""
        <div class="instruction-card">
        <h4>Inventory Management</h4>
        """, unsafe_allow_html=True)
        
        for crop in st.session_state.inventory:
            crop_profit_row = profit_df[profit_df['Crop'] == crop]
            if not crop_profit_row.empty:
                profit = crop_profit_row['Profit'].values[0]
                roi = crop_profit_row['ROI'].values[0]
                
                if profit > 0 and roi > 10:
                    st.markdown(f"""
                    <p><strong>{crop} (In Stock):</strong> Continue growing with the recommended parameters. Projected profit: ${profit:.2f}/m².</p>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <p><strong>{crop} (In Stock):</strong> Consider reducing allocation or optimizing growing parameters to improve profit (currently ${profit:.2f}/m²).</p>
                    """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Next & Back buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back", key="back_to_step3"):
            go_to_step(3)
    with col2:
        if st.button("Restart", key="restart"):
            restart_app()
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 50px; text-align: center; color: #666;">
<p>Vertical Farming Resource Optimizer v1.0</p>
<p>Powered by Machine Learning | Developed by Harsh Tiwari, Arpan Sharma, Gurujit Randhawa</p>
</div>
""", unsafe_allow_html=True)