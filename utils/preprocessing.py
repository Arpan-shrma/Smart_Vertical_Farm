import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(df):
    """
    Transform raw data into features required by the trained model

    Parameters:
    -----------
    df : DataFrame
        Raw data with columns: Timestamp, Crop, Price per kg, Volume Sold per Cycle, Dump Amount

    Returns:
    --------
    processed_df : DataFrame
        DataFrame with all required features for prediction
    price : Series or None
        Original price data if available
    """
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

    # Create rolling statistics (3 and 7 days)
    for crop in data['Crop'].unique():
        crop_mask = data['Crop'] == crop

        # 3-day rolling stats
        data.loc[crop_mask, 'price_roll_mean_3'] = data.loc[crop_mask, 'Price per kg'].rolling(3).mean().shift(1)
        data.loc[crop_mask, 'price_roll_std_3'] = data.loc[crop_mask, 'Price per kg'].rolling(3).std().shift(1)

        # 7-day rolling stats
        data.loc[crop_mask, 'price_roll_mean_7'] = data.loc[crop_mask, 'Price per kg'].rolling(7).mean().shift(1)
        data.loc[crop_mask, 'price_roll_std_7'] = data.loc[crop_mask, 'Price per kg'].rolling(7).std().shift(1)

    # Calculate derived features
    data['price_to_volume_ratio'] = data['Price per kg'] / data['Volume Sold per Cycle'].replace(0, 1e-6)
    data['dump_to_volume_ratio'] = data['Dump Amount'] / data['Volume Sold per Cycle'].replace(0, 1e-6)

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

    # Store the target variable separately before dropping
    price = data['Price per kg'].copy() if 'Price per kg' in data.columns else None

    # Drop columns not needed for prediction
    drop_cols = ['Timestamp', 'Crop', 'Price per kg']
    data.drop([col for col in drop_cols if col in data.columns], axis=1, inplace=True)

    # Handle missing values from lag and rolling features
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)

    return data, price

def generate_sample_data(crops, num_days=7, start_date=None):
    """
    Generate sample market data for testing and demonstration purposes.
    
    Parameters:
    -----------
    crops : list
        List of crop names to include in the sample data
    num_days : int
        Number of days of data to generate
    start_date : datetime, optional
        Starting date for the data (defaults to 7 days ago)
        
    Returns:
    --------
    sample_df : DataFrame
        Sample market data with appropriate columns
    """
    if start_date is None:
        start_date = datetime.now() - pd.Timedelta(days=num_days)
    
    dates = pd.date_range(start=start_date, periods=num_days)
    sample_data = []
    
    # Base prices and characteristics for different crops
    crop_defaults = {
        'Basil': {'base_price': 14.5, 'volume': 500, 'dump': 25, 'volatility': 0.05},
        'Cilantro': {'base_price': 12.3, 'volume': 450, 'dump': 30, 'volatility': 0.04},
        'Kale': {'base_price': 15.7, 'volume': 520, 'dump': 28, 'volatility': 0.06},
        'Lettuce': {'base_price': 8.9, 'volume': 540, 'dump': 22, 'volatility': 0.03},
        'Spinach': {'base_price': 10.2, 'volume': 480, 'dump': 24, 'volatility': 0.04}
    }
    
    for crop in crops:
        defaults = crop_defaults.get(crop, crop_defaults['Lettuce'])
        base_price = defaults['base_price']
        base_volume = defaults['volume']
        base_dump = defaults['dump']
        volatility = defaults['volatility']
        
        # Add some trend to prices over time
        price_trend = np.linspace(-0.02, 0.08, num_days)  # Slight upward trend
        
        for i, date in enumerate(dates):
            # Add daily variability plus trend
            price = base_price * (1 + np.random.normal(0, volatility) + price_trend[i])
            volume = base_volume * (1 + np.random.normal(0, 0.1))
            dump = base_dump * (1 + np.random.normal(0, 0.15))
            
            sample_data.append({
                'Timestamp': date,
                'Crop': crop,
                'Price per kg': price,
                'Volume Sold per Cycle': volume,
                'Dump Amount': dump
            })
    
    sample_df = pd.DataFrame(sample_data)
    return sample_df