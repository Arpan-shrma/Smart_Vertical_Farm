import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def format_resource_display(resource_config):
    """
    Format resource values with appropriate units for display
    
    Parameters:
    -----------
    resource_config : dict
        Dictionary containing resource configuration values
        
    Returns:
    --------
    dict
        Dictionary with formatted resource values including units
    """
    return {
        'Light': f"{resource_config['Light']:.1f} μmol/m²/s",
        'Temperature': f"{resource_config['Temperature']:.1f} °C",
        'Humidity': f"{resource_config['Humidity']:.1f} %",
        'CO2': f"{resource_config['CO2']:.0f} ppm",
        'Soil Moisture': f"{resource_config['Soil_Moisture']:.1f} %",
        'pH': f"{resource_config['pH']:.1f}",
        'EC': f"{resource_config['EC']:.2f} mS/cm"
    }

def create_resource_radar_chart(config, title=None):
    """
    Create radar chart for visualizing resource configurations
    
    Parameters:
    -----------
    config : dict
        Dictionary containing resource configuration values
    title : str, optional
        Chart title
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Radar chart figure
    """
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
    
    # Close the loop for radar chart
    categories = [*categories, categories[0]]
    normalized_values = [*normalized_values, normalized_values[0]]
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    # Set the angles for each category
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot data
    ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', color='#2E7D32')
    ax.fill(angles, normalized_values, alpha=0.25, color='#4CAF50')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1], size=9)
    
    # Remove radial labels
    ax.set_yticklabels([])
    
    # Add title if provided
    if title:
        plt.title(title)
    
    plt.tight_layout()
    return fig

def create_price_forecast_chart(forecast_data, selected_crops, next_cycle_start, next_cycle_end):
    """
    Create line chart of price forecasts over time
    
    Parameters:
    -----------
    forecast_data : DataFrame
        DataFrame with price forecast data
    selected_crops : list
        List of crops to display in the chart
    next_cycle_start : datetime
        Start date of the next growing cycle
    next_cycle_end : datetime
        End date of the next growing cycle
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Line chart figure
    """
    # Filter forecast for selected crops
    filtered_forecast = forecast_data[forecast_data['Crop'].isin(selected_crops)]
    
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
    
    return fig

def create_crop_ranking_chart(ranking_df):
    """
    Create vertical bar chart for crop rankings
    
    Parameters:
    -----------
    ranking_df : DataFrame
        DataFrame with crop ranking data
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Bar chart figure
    """
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
    
    fig.tight_layout()
    
    return fig

def create_yield_comparison_chart(configs):
    """
    Create bar chart comparing predicted yields for different configurations
    
    Parameters:
    -----------
    configs : list
        List of configuration dictionaries with predicted yields
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Bar chart figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    config_names = [f"Config {i+1}" for i in range(len(configs))]
    yields = [config['Predicted_Yield'] for config in configs]
    
    bars = ax.bar(config_names, yields, color='#388E3C')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{height:.2f}',
            ha='center', 
            va='bottom'
        )
    
    ax.set_ylabel('Predicted Yield (kg/m²)')
    ax.set_title('Yield Comparison for Configurations')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    fig.tight_layout()
    
    return fig

def create_profit_comparison_chart(profit_df, inventory):
    """
    Create profit comparison chart with color coding for inventory status
    
    Parameters:
    -----------
    profit_df : DataFrame
        DataFrame with profit data for each crop
    inventory : dict
        Dictionary of crops in inventory
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Bar chart figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define colors based on inventory status
    colors = []
    for crop in profit_df['Crop']:
        if crop in inventory:
            colors.append('#D32F2F')  # Red for in-stock crops
        else:
            colors.append('#2E7D32')  # Green for new crops
    
    bars = ax.bar(profit_df['Crop'], profit_df['Profit'], color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'${height:.2f}',
            ha='center', 
            va='bottom'
        )
    
    ax.set_ylabel('Profit per m² ($)')
    ax.set_title('Projected Profit by Crop')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    fig.tight_layout()
    
    return fig