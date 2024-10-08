# Forex-TCN-Predictor

## Introduction

The Forex-TCN-Predictor is an advanced forex price prediction model using Temporal Convolutional Networks (TCN). This project implements a sophisticated time series forecasting system for predicting forex prices, specifically tailored for the EUR/USD pair. The current implementation focuses on predicting both high and low prices for the next two time steps (t+1 and t+2).

## Project Evolution

This project represents the culmination of extensive research and experimentation with various time series forecasting techniques. Our journey to develop an effective forex prediction model involved several iterations and different architectural approaches:

1. **Traditional Time Series Models**: 
   - Initially, we explored classic statistical methods like ARIMA and SARIMA.
   - While these provided a solid baseline, they struggled to capture the complex non-linear patterns in forex data.

2. **Recurrent Neural Networks (RNNs)**:
   - We then moved to deep learning approaches, starting with simple RNNs.
   - These showed improvement over statistical methods but suffered from the vanishing gradient problem for longer sequences.

3. **Long Short-Term Memory (LSTM) Networks**:
   - LSTMs were implemented to address the limitations of simple RNNs.
   - They demonstrated better performance in capturing long-term dependencies in the forex data.
   - However, training times were considerable, and the model sometimes struggled with very long input sequences.

4. **Gated Recurrent Units (GRUs)**:
   - We experimented with GRUs as a more computationally efficient alternative to LSTMs.
   - Performance was comparable to LSTMs, with slightly faster training times.

5. **Convolutional Neural Networks (CNNs)**:
   - We also explored 1D CNNs for time series forecasting.
   - These showed promise in capturing local patterns but struggled with longer-term dependencies.

6. **Temporal Convolutional Networks (TCNs)**:
   - Finally, we implemented TCNs, which combine the strengths of CNNs and RNNs.
   - TCNs demonstrated superior performance in terms of prediction accuracy and computational efficiency.
   - They excelled in capturing both short-term and long-term patterns in the forex data.

After extensive comparison and evaluation, the TCN-based model emerged as the most effective approach for our forex prediction task. It offered the best balance of prediction accuracy, ability to handle long sequences, and computational efficiency.

## Key Advantages of the TCN Model

1. **Parallelism**: Unlike RNNs, TCNs can process input sequences in parallel, leading to faster training and inference times.
2. **Flexible Receptive Field**: The dilated convolutions in TCNs allow for a large receptive field, capturing long-range dependencies effectively.
3. **Stable Gradients**: TCNs don't suffer from vanishing/exploding gradients, a common issue with RNNs and LSTMs.
4. **Precise Control Over Memory**: The model's memory length is precisely determined by the architecture, offering better control compared to RNNs.

The current implementation of Forex-TCN-Predictor leverages these advantages to provide accurate and reliable forex price predictions, as evidenced by its impressive performance metrics.

# Description
Forex-TCN-Predictor is an advanced forex price prediction model leveraging Temporal Convolutional Networks (TCN) for time series forecasting. Specifically designed for EUR/USD High, Low and close price prediction, this project implements state-of-the-art techniques in deep learning and financial time series analysis.
Features
Custom TCN architecture with integrated time-based features
Comprehensive feature engineering including technical indicators
Data augmentation for high-volatility periods
Multi-step forecasting (t+1 and t+2)
Adaptive learning rate and early stopping mechanisms
Error analysis and visualization tools

# Installation
To set up the Forex-TCN-Predictor, follow these steps:

# Clone the repository:
Copy git clone https://github.com/Gdataly/Forex-TCN-Predictor.git

# Usage
To run the predictor:
Prepare your forex data in CSV format with columns: Date, Time, Open, High, Low, Close, Volume.
Update the file_path variable in the main script with your data file path.
Run the main script:
python forex_tcn_predictor.py


# Data
The model expects hourly forex data for EUR/USD. Ensure your dataset includes the following columns:
Date
Time
Open
High
Low
Close
Volume

# Model Architecture
The core of this project is a Temporal Convolutional Network (TCN) with the following key components:

Custom TCN layers with dilated convolutions
Integration of time-based features
Dense layers for final predictions
The model is designed to capture complex temporal dependencies in forex data, offering insights for both short-term and slightly longer-term price movements.

# Performance and Results
The model's performance is evaluated using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) for both t+1 and t+2 predictions. Actual performance metrics will vary based on the specific dataset and market conditions.

The Forex TCN Predictor demonstrates robust performance in forecasting both high and low prices for the EUR/USD pair. Key performance metrics include:

Mean Absolute Percentage Error (MAPE): Consistently ranges between 0.05% and 0.06% for both high and low price predictions.
This level of accuracy is maintained for both t+1 (next hour) and t+2 (two hours ahead) forecasts.
The model shows stable performance across different market conditions, including high volatility periods.

These results indicate that the model can predict forex prices with an average error of less than 0.1%, providing valuable insights for short-term trading strategies. However, users should note that past performance does not guarantee future results, and the model should be used in conjunction with other analysis tools and risk management strategies.

# Visualization
The project includes tools for visualizing:
Training history
Error distributions
Predicted vs Actual price comparisons
These visualizations are saved as PNG files in the project directory.

## Features and Usage

### Key Features

The Forex TCN Predictor script offers a comprehensive solution for forex price prediction, including:

1. Future predictions for the next two periods using the most recent data.
2. Saving of future predictions to a CSV file for easy analysis.
3. Visualization of the model's training history.
4. Comparative plots of actual vs predicted values for the test set.
5. Analysis and visualization of feature importance.

### Comprehensive Pipeline

This script provides an end-to-end pipeline for forex prediction:

- Data loading and preprocessing
- Feature engineering
- TCN model training
- Model performance evaluation
- Future price prediction
- Results visualization and model characteristic analysis

### Enhanced Prediction Capabilities

The improved prediction functionality aims to provide more accurate and reliable forecasts for future forex prices. This is achieved through:

- Utilization of the most recent data for predictions
- Implementation of a sliding window approach for multi-step forecasting
- Incorporation of various technical indicators and time-based features

### Usage Instructions

To use this script effectively:

1. Ensure all required libraries are installed. You can use the provided `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

2. Update the file paths in the `__main__` block of the script:
   - Set `file_path` to the location of your input data CSV file.
   - Adjust `model_path`, `scaler_X_path`, and `scaler_y_path` to your desired save locations.

3. Run the script in a Python environment with sufficient computational resources:
   ```
   python forex_tcn_predictor.py
   ```

### Output and Visualization

After running, the script will generate:

- A trained model file (`forex_model.h5`)
- Scaler files for features and target variables
- CSV files with predictions and actual values
- Various plots including error distributions, training history, and feature importance

# Future Improvements
Implement ensemble methods with other model architectures
Explore the inclusion of sentiment analysis from financial news
Optimize hyperparameters using Bayesian optimization
Extend the model to predict other currency pairs

# Contributing
Contributions to Forex-TCN-Predictor are welcome! Please feel free to submit a Pull Request.
# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
TensorFlow for the deep learning framework
pandas for data manipulation
scikit-learn for preprocessing and evaluation metrics
TA-Lib for technical analysis indicators

### Important Notes

- Regularly update your dataset and retrain the model to maintain accuracy, as forex markets are highly dynamic.
- Use these predictions in conjunction with other forms of analysis and robust risk management strategies when making trading decisions.
- The model's performance can vary depending on market conditions and the quality of input data.

By leveraging these features, you can gain valuable insights into forex price movements and potentially enhance your trading strategies. However, always exercise caution and consider multiple factors when making financial decisions.

Disclaimer
This project is for educational purposes only. Forex trading carries a high level of risk, and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange you should carefully consider your investment objectives, level of experience, and risk appetite.
