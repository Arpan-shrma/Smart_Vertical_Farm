# Smart Vertical Farm: Resource Optimizer

## Overview

Smart Vertical Farm is an intelligent web application built with Streamlit and machine learning that helps vertical farmers optimize their growing resources and maximize profits. By combining market forecasting with crop-specific growing recommendations, the system helps farmers make data-driven decisions about which crops to grow and how to optimize growing conditions.

![Smart Vertical Farm Logo](https://raw.githubusercontent.com/HarshTiwari1710/Resource-Competition-Modelling/refs/heads/main/logo.png)

## Features

- **Market Price Forecasting**: Predicts future crop prices using XGBoost ML model
- **Crop Ranking**: Ranks crops by profitability based on market forecasts
- **Inventory Management**: Tracks current crop inventory 
- **Target Yield Settings**: Set realistic target yields for each crop
- **Resource Optimization**: Generates optimal growing parameters customized to each crop
- **Growing Strategy Options**: Provides multiple configuration strategies:
  - Balanced: Optimal general growing parameters
  - High Light: Increased light intensity for faster growth
  - Nutrient Rich: Enhanced nutrients for better nutrient content
  - Water Efficient: Optimized water consumption
  - Energy Efficient: Reduced energy use for sustainable growing
- **Profit Projections**: Calculates expected revenue, costs, profit and ROI
- **Visual Comparisons**: Radar charts and bar graphs for parameter visualization

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/smart-vertical-farm.git
   cd smart-vertical-farm
   ```

2. Create and activate a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Required Libraries

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn
- joblib
- xgboost

## Usage

1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Navigate through the 4-step process:
   - **Step 1**: Set your current growing cycle end date
   - **Step 2**: Review price forecasts & crop rankings
   - **Step 3**: Update inventory and set target yields
   - **Step 4**: Get optimized resource recommendations

## Machine Learning Models

This application uses three main ML models:

1. **Market Forecast Model** (`forecast_market_demand.pkl`):
   - XGBoost regressor trained on market data
   - Predicts crop prices based on historical patterns

2. **Resource Generator Model** (`resource_generator_model.h5`):
   - Neural network that generates optimal growing parameters
   - Trained on environmental condition data from successful crops

3. **Yield Predictor Model** (`yield_predictor_model.h5`):
   - Neural network that predicts yield based on growing parameters
   - Evaluates different resource configurations

## Project Structure

```
smart-vertical-farm/
├── app.py                           # Main application file
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── models/                          # Trained ML models
│   ├── forecast_market_demand.pkl   # Market forecast model
│   ├── resource_generator_model.h5  # Resource generator model
│   └── yield_predictor_model.h5     # Yield predictor model
├── preprocessors/                   # Model preprocessors
│   ├── market_forecast_preprocessor.pkl
│   ├── preprocessor_resource_model.pkl
│   └── preprocessor_yield_model.pkl
├── modules/                         # Helper modules
│   ├── forecast.py                  # Forecasting functions
│   └── recommender.py               # Recommendation functions
└── utils/                           # Utility functions
    └── dummy_preprocessor.py        # Fallback preprocessors
```

## ML Model Training

For information on how the models were trained, refer to the Jupyter notebooks in the `notebooks` directory:
- `market_forecast.ipynb`: Training process for price forecasting model
- `resource_optimization.ipynb`: Training for resource generator model
- `yield_prediction.ipynb`: Training for yield prediction model

## Demo Data

The application includes demo market data if no real data is available, allowing for testing without connecting to actual crop price databases.

## Profit Calculation

Profit for each crop is calculated using realistic microgreens industry costs:
- Seed costs (varying by crop)
- Growing medium
- Container costs
- Electricity (based on light intensity)
- Water (based on soil moisture)
- Labor costs
- Packaging costs

These are subtracted from revenue (yield * market price) to determine profit per tray.

## Development Team

- Harsh Tiwari
- Arpan Sharma
- Gurujit Randhawa

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the contributors of the market data and growing parameters used to train our models.