import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

def recommend_optimal_conditions(crop, yield_value, model=None):
    """
    Generate optimal growing conditions for a specific crop and target yield
    
    Parameters:
    -----------
    crop : str
        Name of the crop (e.g., "Basil", "Kale", etc.)
    yield_value : float
        Target yield in kg/m²
    model : keras.Model or None
        Pre-loaded model for resource generation. If None, will attempt to load.
        
    Returns:
    --------
    dict
        Dictionary with recommended values for each parameter (Light, Temperature, etc.)
    """
    # Try to load the model if not provided
    if model is None:
        try:
            model_path = 'models/resource_generator_model.h5'
            if not os.path.exists(model_path):
                model_path = 'resource_generator_model.h5'  # Try current directory
                
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Try to load preprocessor
            try:
                preprocessor_path = 'preprocessors/resource_generator_preprocessor.pkl'
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
                            
                preprocessor = joblib.load(preprocessor_path)
                
                # Generate input data
                crop_df = pd.DataFrame({'Crop': [crop]})
                
                # Transform with preprocessor
                model_input = preprocessor.transform(crop_df)
                
                # Get predictions
                resources = model.predict(model_input)[0]
                
                # Scale resources based on target yield
                yield_factor = max(0.8, min(1.5, yield_value / 1.2))
                
                # Rescale to actual values
                result = {
                    'Light': min(0.95, resources[0] * yield_factor),
                    'Temperature': resources[1],
                    'Humidity': resources[2],
                    'CO2': min(0.95, resources[3] * yield_factor),
                    'Soil Moisture': resources[4],
                    'pH': resources[5],
                    'EC': min(0.95, resources[6] * yield_factor)
                }
                
                return result
                
            except Exception as e:
                print(f"Error with preprocessor: {e}")
                # Fall back to defaults
                model = None
                
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    
    # If model is None, use default values
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

def generate_resource_configurations(crop, target_yield, num_configs=5, model=None, preprocessor=None):
    """
    Generate multiple resource configurations for a given crop
    
    Parameters:
    -----------
    crop : str
        Name of the crop
    target_yield : float
        Target yield in kg/m²
    num_configs : int
        Number of configurations to generate
    model : keras.Model or None
        Pre-loaded model for resource generation
    preprocessor : object or None
        Preprocessor for the model
        
    Returns:
    --------
    list
        List of dictionaries with resource configurations
    """
    # Try using the ML model if available
    if model is not None and preprocessor is not None:
        try:
            # Create input DataFrame with the crop name
            crop_df = pd.DataFrame({'Crop': [crop] * num_configs})
            
            # Transform with the preprocessor
            crop_input = preprocessor.transform(crop_df)
            
            # Generate configurations using the model
            # Add some noise to generate variations
            configurations = []
            
            for i in range(num_configs):
                # Add random noise to the input to generate variations
                noise = np.random.normal(0, 0.1, crop_input.shape)
                crop_input_with_noise = crop_input + noise
                
                # Predict resource values 
                resource_values = model.predict(crop_input_with_noise)[i if i < len(crop_input_with_noise) else 0]
                
                # Scale resource values based on target yield
                yield_factor = max(0.8, min(1.5, target_yield / 1.2))
                
                # Create configuration dictionary with sensible ranges
                config = {
                    'Configuration': i + 1,
                    'Light': max(50, min(400, resource_values[0] * 400 * yield_factor)),
                    'Temperature': max(15, min(30, resource_values[1] * 30)),
                    'Humidity': max(40, min(90, resource_values[2] * 100)),
                    'CO2': max(400, min(1500, resource_values[3] * 2000 * yield_factor)),
                    'Soil_Moisture': max(50, min(90, resource_values[4] * 100)),
                    'pH': max(5.5, min(7.5, resource_values[5] * 2 + 5.5)),
                    'EC': max(0.5, min(3.5, resource_values[6] * 4 * yield_factor))
                }
                
                configurations.append(config)
            
            return configurations
            
        except Exception as e:
            print(f"Error using resource generator model: {e}")
            # Fall back to rule-based approach
    
    # If no model or error, use the fallback method
    # Get the base values
    base_values = recommend_optimal_conditions(crop, target_yield)
    
    # Create multiple variations around the base configuration
    configs = []
    for i in range(num_configs):
        # Apply small random variations to create multiple options
        variation = {k: v * (1 + np.random.normal(0, 0.1)) for k, v in base_values.items()}
        
        # Convert normalized values to actual ranges
        config = {
            'Configuration': i + 1,
            'Light': max(50, min(400, variation['Light'] * 400)),  # Scale to μmol/m²/s
            'Temperature': max(15, min(30, variation['Temperature'] * 30 + 15)),  # Scale to °C
            'Humidity': max(40, min(90, variation['Humidity'] * 60 + 40)),  # Scale to %
            'CO2': max(400, min(1500, variation['CO2'] * 1000 + 400)),  # Scale to ppm
            'Soil_Moisture': max(50, min(90, variation['Soil Moisture'] * 100)),  # Scale to %
            'pH': max(5.5, min(7.5, variation['pH'] * 7 + 5.5)),  # Scale to pH
            'EC': max(0.5, min(3.5, variation['EC'] * 3 + 0.5))  # Scale to mS/cm
        }
        
        configs.append(config)
    
    return configs

def predict_yield(crop, resource_configs, model=None, preprocessor=None):
    """
    Predict yield for each resource configuration
    
    Parameters:
    -----------
    crop : str
        Name of the crop
    resource_configs : list
        List of dictionaries with resource configurations
    model : keras.Model or None
        Pre-loaded model for yield prediction
    preprocessor : object or None
        Preprocessor for the model
        
    Returns:
    --------
    list
        List of dictionaries with resource configurations and predicted yields
    """
    # Try using the ML model if available
    if model is not None and preprocessor is not None:
        try:
            # Prepare input data for the model
            configs_with_yield = []
            
            for config in resource_configs:
                # Create DataFrame with crop and resource values
                input_df = pd.DataFrame({
                    'Crop': [crop],
                    'Light': [config['Light']],
                    'Temperature': [config['Temperature']],
                    'Humidity': [config['Humidity']],
                    'CO2': [config['CO2']],
                    'Soil Moisture': [config['Soil_Moisture']],
                    'pH': [config['pH']],
                    'EC': [config['EC']]
                })
                
                # Preprocess the input
                model_input = preprocessor.transform(input_df)
                
                # Predict yield
                predicted_yield = model.predict(model_input)[0][0]
                
                # Add yield to configuration
                config_with_yield = config.copy()
                config_with_yield['Predicted_Yield'] = max(0.5, min(5.0, predicted_yield))
                
                configs_with_yield.append(config_with_yield)
            
            # Sort by predicted yield
            configs_with_yield = sorted(configs_with_yield, 
                                      key=lambda x: x['Predicted_Yield'], 
                                      reverse=True)
            
            return configs_with_yield
            
        except Exception as e:
            print(f"Error using yield predictor model: {e}")
            # Fall back to rule-based approach
    
    # If no model or error, use a simple rule-based approach
    configs_with_yield = []
    for config in resource_configs:
        # Define ideal ranges for each crop
        ideal_ranges = {
            "Basil": {
                'Light': (200, 300),
                'Temperature': (21, 27),
                'Humidity': (60, 80),
                'CO2': (800, 1200),
                'Soil_Moisture': (60, 75),
                'pH': (5.8, 6.5),
                'EC': (1.0, 2.0)
            },
            "Cilantro": {
                'Light': (150, 250),
                'Temperature': (18, 24),
                'Humidity': (50, 70),
                'CO2': (800, 1200),
                'Soil_Moisture': (65, 80),
                'pH': (6.0, 6.7),
                'EC': (0.8, 1.5)
            },
            "Kale": {
                'Light': (250, 350),
                'Temperature': (15, 22),
                'Humidity': (50, 70),
                'CO2': (1000, 1400),
                'Soil_Moisture': (65, 75),
                'pH': (5.5, 6.5),
                'EC': (1.2, 2.5)
            },
            "Lettuce": {
                'Light': (150, 250),
                'Temperature': (16, 22),
                'Humidity': (60, 80),
                'CO2': (800, 1200),
                'Soil_Moisture': (70, 85),
                'pH': (5.8, 6.5),
                'EC': (0.8, 1.2)
            },
            "Spinach": {
                'Light': (200, 300),
                'Temperature': (15, 20),
                'Humidity': (60, 80),
                'CO2': (1000, 1400),
                'Soil_Moisture': (65, 75),
                'pH': (6.0, 7.0),
                'EC': (1.0, 2.0)
            }
        }
        
        # Get ideal range for this crop (or default to lettuce)
        crop_range = ideal_ranges.get(crop, ideal_ranges["Lettuce"])
        
        # Calculate how close each parameter is to its ideal range
        # 1.0 means perfectly in range, 0.0 means far outside range
        param_scores = {}
        for param, (min_val, max_val) in crop_range.items():
            current_val = config[param]
            if min_val <= current_val <= max_val:
                # Within ideal range
                param_scores[param] = 1.0
            else:
                # Outside ideal range, calculate distance
                if current_val < min_val:
                    distance = (min_val - current_val) / min_val
                else:
                    distance = (current_val - max_val) / max_val
                # Convert distance to score (0.0 to 1.0)
                param_scores[param] = max(0.0, 1.0 - distance)
        
        # Calculate overall score (weighted average)
        weights = {
            'Light': 0.25,
            'Temperature': 0.2,
            'Humidity': 0.1,
            'CO2': 0.15,
            'Soil_Moisture': 0.1,
            'pH': 0.1,
            'EC': 0.1
        }
        
        overall_score = sum(param_scores[param] * weights[param] for param in param_scores) / sum(weights.values())
        
        # Convert score to yield
        base_yield = {
            "Basil": 1.2,
            "Cilantro": 0.9,
            "Kale": 1.5,
            "Lettuce": 1.3,
            "Spinach": 1.1
        }.get(crop, 1.0)
        
        predicted_yield = base_yield * (0.5 + 1.5 * overall_score)  # Range from 0.5x to 2.0x base yield
        
        # Add yield to configuration
        config_with_yield = config.copy()
        config_with_yield['Predicted_Yield'] = round(predicted_yield, 2)
        
        configs_with_yield.append(config_with_yield)
    
    # Sort by predicted yield
    configs_with_yield = sorted(configs_with_yield, 
                              key=lambda x: x['Predicted_Yield'], 
                              reverse=True)
    
    return configs_with_yield

def get_best_resource_config(crop_name, yield_value, resource_model=None, yield_model=None):
    """
    Generate resource configurations, predict yields, and return the best configuration
    
    Parameters:
    -----------
    crop_name : str
        Name of the crop
    yield_value : float
        Target yield in kg/m²
    resource_model : keras.Model or None
        Pre-loaded model for resource generation
    yield_model : keras.Model or None
        Pre-loaded model for yield prediction
        
    Returns:
    --------
    dict
        Dictionary with optimal resource configuration and predicted yield
    """
    # Try to load models if not provided
    if resource_model is None:
        try:
            resource_model_path = 'models/resource_generator_model.h5'
            if not os.path.exists(resource_model_path):
                resource_model_path = 'resource_generator_model.h5'
            resource_model = tf.keras.models.load_model(resource_model_path, compile=False)
        except:
            resource_model = None
    
    if yield_model is None:
        try:
            yield_model_path = 'models/yield_predictor_model.h5'
            if not os.path.exists(yield_model_path):
                yield_model_path = 'yield_predictor_model.h5'
            yield_model = tf.keras.models.load_model(yield_model_path, compile=False)
        except:
            yield_model = None
    
    # Load preprocessors
    try:
        resource_preprocessor_path = 'preprocessors/resource_generator_preprocessor.pkl'
        if not os.path.exists(resource_preprocessor_path):
            # Try alternatives
            alternatives = ['preprocessor_resource_model.pkl', 'preprocessor_model2.pkl']
            for alt in alternatives:
                if os.path.exists(alt):
                    resource_preprocessor_path = alt
                    break
        resource_preprocessor = joblib.load(resource_preprocessor_path)
    except:
        resource_preprocessor = None
    
    try:
        yield_preprocessor_path = 'preprocessors/yield_predictor_preprocessor.pkl'
        if not os.path.exists(yield_preprocessor_path):
            # Try alternatives
            alternatives = ['preprocessor_yield_model.pkl']
            for alt in alternatives:
                if os.path.exists(alt):
                    yield_preprocessor_path = alt
                    break
        yield_preprocessor = joblib.load(yield_preprocessor_path)
    except:
        yield_preprocessor = None
    
    # Generate multiple configurations
    configs = generate_resource_configurations(
        crop_name, 
        yield_value,
        num_configs=10,
        model=resource_model,
        preprocessor=resource_preprocessor
    )
    
    # Predict yield for each configuration
    configs_with_yield = predict_yield(
        crop_name,
        configs,
        model=yield_model,
        preprocessor=yield_preprocessor
    )
    
    # Return the best configuration
    if configs_with_yield:
        return configs_with_yield[0]
    else:
        return None