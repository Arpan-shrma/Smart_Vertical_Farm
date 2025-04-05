# modules/resource_generator.py

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and preprocessor
RESOURCE_MODEL_PATH = "models/resource_generator_model.h5"
RESOURCE_PREPROCESSOR_PATH = "preprocessors/preprocessor_model2.pkl"

# Load once globally
resource_model = load_model(RESOURCE_MODEL_PATH, compile=False)
resource_preprocessor = joblib.load(RESOURCE_PREPROCESSOR_PATH)


def generate_resource_configs(crop_name, num_configs=10):
    """
    Generate multiple environmental resource configurations for a given crop.

    Parameters:
    - crop_name (str): Crop to generate configs for
    - num_configs (int): Number of configurations to generate

    Returns:
    - pd.DataFrame: DataFrame of generated resource configurations
    """
    # Optionally include yield-like random noise for variation
    dummy_yields = np.random.uniform(0.8, 1.2, num_configs)
    input_df = pd.DataFrame({
        'Crop': [crop_name] * num_configs,
        'Yield': dummy_yields  # If yield is required for your model
    })

    # Preprocess input
    X_input = resource_preprocessor.transform(input_df)

    # Predict configurations
    predictions = resource_model.predict(X_input)
    config_df = pd.DataFrame(predictions, columns=['Light', 'Temperature', 'Humidity', 'CO2', 'Soil Moisture', 'pH', 'EC'])
    config_df.insert(0, 'Crop', crop_name)
    return config_df


if __name__ == "__main__":
    # Example test
    df = generate_resource_configs("Kale", num_configs=5)
    print(df)