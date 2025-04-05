# Vertical Farming Resource Optimizer

This application helps vertical farm operators make data-driven decisions to optimize resource allocation, maximize crop yield, and align production with market demands.

## Features

- **Market Price Forecasting**: Predict future crop prices to plan planting cycles strategically
- **Resource Recommendation**: Get optimal environmental conditions for maximizing crop yields
- **Integrated Dashboard**: Comprehensive view of market trends and resource allocation strategies

## System Architecture

```
vertical-farming-app/
│
├── app.py                     ← Main Streamlit interface
├── requirements.txt           ← List of dependencies 
│
├── models/
│   ├── xgb_final_model.pkl    ← Trained XGBoost crop price model
│   └── my_model.h5            ← Trained Keras yield recommender model
│
├── data/
│   └── sample_data.csv        ← Example input data (optional)
│
├── modules/
│   ├── forecast.py            ← Handles price forecasting logic
│   └── recommender.py         ← Handles yield/environment recommendation logic
│
└── utils/
    └── preprocessing.py       ← Shared preprocessing and feature engineering
```

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/vertical-farming-app.git
   cd vertical-farming-app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**:
   ```bash
   mkdir -p models data
   ```

5. **Download pre-trained models**:
   Place the pre-trained XGBoost model (`xgb_final_model.pkl`) and Keras model (`my_model.h5`) in the `models/` directory.

## Running the Application

Start the Streamlit app with:

```bash
streamlit run app.py
```

The application will be available at [http://localhost:8501](http://localhost:8501).

## Using the Application

### Market Price Forecasting

1. Select the start date for historical data
2. Choose crops you want to forecast
3. Set the forecast horizon (days)
4. Optionally upload your custom historical data
5. Click "Generate Forecast" to view predictions

### Resource Recommendation

1. Select a crop from the dropdown
2. Adjust the target yield slider
3. Click "Get Recommendations" to view optimal growing conditions

### Integrated Dashboard

1. Select crops to analyze in the sidebar
2. Set the forecast horizon and target yields
3. Click "Update Dashboard" to generate a comprehensive view
4. Explore the market overview, resource allocation, and optimization summary tabs

## Data Format

If you're uploading custom historical data, it should be a CSV file with the following columns:
- Timestamp (in YYYY-MM-DD format)
- Crop (one of: Basil, Cilantro, Kale, Lettuce, Spinach)
- Price per kg (numerical value)
- Volume Sold per Cycle (numerical value)
- Dump Amount (numerical value)

## Technical Information

### Models

1. **Price Forecasting Model**: 
   - XGBoost regressor trained on historical market data
   - Accounts for seasonality, market trends, and crop-specific factors
   - Forecasts prices for up to 90 days ahead

2. **Resource Recommendation Model**:
   - Neural network model trained on sensor data from vertical farms
   - Recommends optimal growing conditions for desired crop yields
   - Parameters include light, temperature, humidity, CO2, soil moisture, pH, and EC

## Authors

- Arpan Sharma
- Harsh Tiwari

## Guide 
Gurujit Randhawa

## License

This project is licensed under the MIT License - see the LICENSE file for details.