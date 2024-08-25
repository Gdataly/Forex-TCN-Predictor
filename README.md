# Forex-TCN-Predictor
Advanced forex price prediction model using Temporal Convolutional Networks (TCN). This project implements a sophisticated time series forecasting system for predicting forex close prices, specifically tailored for EUR/USD.

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

# Performance
The model's performance is evaluated using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) for both t+1 and t+2 predictions. Actual performance metrics will vary based on the specific dataset and market conditions.

# Visualization
The project includes tools for visualizing:
Training history
Error distributions
Predicted vs Actual price comparisons
These visualizations are saved as PNG files in the project directory.

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

Disclaimer
This project is for educational purposes only. Forex trading carries a high level of risk, and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange you should carefully consider your investment objectives, level of experience, and risk appetite.
