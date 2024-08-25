!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install

!pip install TA-Lib

#!pip install --upgrade keras-tcn(use this or the one below depends on your set up)
!pip install keras-tcn==3.4.0

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib
import tensorflow as tf
from tcn import TCN
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the forex data from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file containing forex data.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame with datetime index and additional features.
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Date and Time columns to a datetime index
    df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Date_Time', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    
    # Convert numeric columns to float
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    # Add time-based features
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Hour_of_Day'] = df.index.hour
    df['Week_of_Year'] = df.index.isocalendar().week
    
    # Perform data augmentation for high volatility hours
    augmented_data = []
    for idx, row in df.iterrows():
        if 9 <= idx.hour < 17:
            new_idx = idx + pd.Timedelta(minutes=30)
            new_row = row.copy()
            # Slightly modify features for augmented data
            new_row['Open'] += np.random.normal(0, 0.0001)
            new_row['Close'] += np.random.normal(0, 0.0001)
            new_row['Volume'] *= (1 + np.random.normal(0, 0.01))
            augmented_data.append((new_idx, new_row))
    
    # Add augmented data to the dataframe
    for idx, row in augmented_data:
        df.loc[idx] = row
    
    # Sort the index to ensure chronological order
    df = df.sort_index()
    
    return df

def create_features(df):
    """
    Create additional features for the forex data.
    
    Args:
    df (pd.DataFrame): Input DataFrame with basic forex data.
    
    Returns:
    pd.DataFrame: DataFrame with additional engineered features.
    """
    new_columns = {}
    
    # Create lagged features
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        for lag in range(1, 21):
            new_columns[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create moving average and standard deviation features
    for window in [5, 10, 20, 50]:
        new_columns[f'High_MA_{window}'] = df['High'].rolling(window=window).mean()
        new_columns[f'Low_MA_{window}'] = df['Low'].rolling(window=window).mean()
        new_columns[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
        new_columns[f'High_STD_{window}'] = df['High'].rolling(window=window).std()
        new_columns[f'Low_STD_{window}'] = df['Low'].rolling(window=window).std()
        new_columns[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
    
    # Add technical indicators
    new_columns['RSI'] = talib.RSI(df['Close'])
    new_columns['MACD'], new_columns['MACD_Signal'], _ = talib.MACD(df['Close'])
    new_columns['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    new_columns['Bollinger_Upper'], new_columns['Bollinger_Middle'], new_columns['Bollinger_Lower'] = talib.BBANDS(df['Close'])
    
    # Add price change features
    new_columns['High_Change_1'] = df['High'].pct_change(fill_method=None).fillna(0)
    new_columns['Low_Change_1'] = df['Low'].pct_change(fill_method=None).fillna(0)
    new_columns['High_Change_2'] = df['High'].pct_change(periods=2, fill_method=None).fillna(0)
    new_columns['Low_Change_2'] = df['Low'].pct_change(periods=2, fill_method=None).fillna(0)
    
    # Add time-based features
    new_columns['is_high_volatility'] = ((df.index.hour >= 9) & (df.index.hour < 17)).astype(int)
    new_columns['is_morning_spike'] = ((df.index.hour >= 9) & (df.index.hour <= 10)).astype(int)
    new_columns['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    new_columns['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Add price difference and ratio features
    new_columns['High_Low_Diff'] = df['High'] - df['Low']
    new_columns['High_Low_Ratio'] = df['High'] / df['Low']
    new_columns['High_Close_Ratio'] = df['High'] / df['Close']
    new_columns['Low_Close_Ratio'] = df['Low'] / df['Close']
    
    # Add moving averages of price differences and ratios
    for window in [5, 10, 20]:
        new_columns[f'High_Low_Diff_MA_{window}'] = new_columns['High_Low_Diff'].rolling(window=window).mean()
        new_columns[f'High_Low_Ratio_MA_{window}'] = new_columns['High_Low_Ratio'].rolling(window=window).mean()
    
    # Add price position features
    new_columns['Percent_Above_Close'] = (df['High'] - df['Close']) / (df['High'] - df['Low'])
    new_columns['Percent_Below_Close'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Combine all features
    df_with_features = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return df_with_features

def prepare_sequences(df, sequence_length=20, forecast_horizon=2):
    """
    Prepare input sequences and target values for the model.
    
    Args:
    df (pd.DataFrame): Input DataFrame with all features.
    sequence_length (int): Length of input sequences.
    forecast_horizon (int): Number of steps to forecast.
    
    Returns:
    tuple: Numpy arrays of input sequences (X), high targets, low targets, and timestamps.
    """
    feature_columns = [col for col in df.columns if col not in ['High', 'Low']]
    X = df[feature_columns].values
    y_high = df['High'].values
    y_low = df['Low'].values
    
    X_sequences, y_high_sequences, y_low_sequences, timestamps = [], [], [], []
    for i in range(len(X) - sequence_length - forecast_horizon + 1):
        X_sequences.append(X[i:(i + sequence_length)])
        y_high_sequences.append(y_high[(i + sequence_length):(i + sequence_length + forecast_horizon)])
        y_low_sequences.append(y_low[(i + sequence_length):(i + sequence_length + forecast_horizon)])
        timestamps.append(df.index[i + sequence_length])
    
    return np.array(X_sequences), np.array(y_high_sequences), np.array(y_low_sequences), np.array(timestamps)

def create_joint_tcn_model(input_shape, output_steps):
    """
    Create a joint TCN model for predicting both high and low prices.
    
    Args:
    input_shape (tuple): Shape of input data.
    output_steps (int): Number of output steps to predict.
    
    Returns:
    tf.keras.Model: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    
    x = TCN(
        nb_filters=64,
        kernel_size=3,
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16, 32],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.2,
        return_sequences=False
    )(inputs)
    
    high_output = Dense(output_steps, name='high_output')(x)
    low_output = Dense(output_steps, name='low_output')(x)
    
    model = Model(inputs=inputs, outputs=[high_output, low_output])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss={'high_output': 'mse', 'low_output': 'mse'},
                  loss_weights={'high_output': 1.0, 'low_output': 1.0})
    return model

def calculate_metrics(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
    
    Args:
    y_true (np.array): True values.
    y_pred (np.array): Predicted values.
    
    Returns:
    tuple: MAE and MAPE values.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mape

def process_and_train(file_path, sequence_length=20, forecast_horizon=2):
    """
    Main function to process data, create and train the model, and evaluate its performance.
    
    Args:
    file_path (str): Path to the input CSV file.
    sequence_length (int): Length of input sequences.
    forecast_horizon (int): Number of steps to forecast.
    
    Returns:
    tuple: Trained model, scalers, training history, test data, predictions, and timestamps.
    """
    # Load and preprocess data
    df = load_and_preprocess_data(file_path)
    print(f"Shape after augmentation: {df.shape}")
    df = create_features(df)
    df.dropna(inplace=True)
    
    # Prepare sequences
    X, y_high, y_low, timestamps = prepare_sequences(df, sequence_length, forecast_horizon)
    
    print(f"Total sequences: {len(X)}")
    print(f"Steps per epoch: {len(X) // 32}")  # Assuming batch_size = 32
    
    # Split data into train and test sets
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_high_train, y_high_test = y_high[:split_index], y_high[split_index:]
    y_low_train, y_low_test = y_low[:split_index], y_low[split_index:]
    timestamps_train, timestamps_test = timestamps[:split_index], timestamps[split_index:]
    
    print(f"Training sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    
    # Scale the data
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    scaler_y = MinMaxScaler()
    y_high_train_scaled = scaler_y.fit_transform(y_high_train.reshape(-1, 1)).reshape(y_high_train.shape)
    y_high_test_scaled = scaler_y.transform(y_high_test.reshape(-1, 1)).reshape(y_high_test.shape)
    y_low_train_scaled = scaler_y.transform(y_low_train.reshape(-1, 1)).reshape(y_low_train.shape)
    y_low_test_scaled = scaler_y.transform(y_low_test.reshape(-1, 1)).reshape(y_low_test.shape)
    
    # Create and train the model
    model = create_joint_tcn_model(input_shape=(sequence_length, X_train.shape[-1]), output_steps=forecast_horizon)
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    history = model.fit(
        X_train_scaled, 
        {'high_output': y_high_train_scaled, 'low_output': y_low_train_scaled},
        validation_split=0.1,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Make predictions
    y_high_pred_scaled, y_low_pred_scaled = model.predict(X_test_scaled)
    y_high_pred = scaler_y.inverse_transform(y_high_pred_scaled.reshape(-1, 1)).reshape(y_high_pred_scaled.shape)
    y_low_pred = scaler_y.inverse_transform(y_low_pred_scaled.reshape(-1, 1)).reshape(y_low_pred_scaled.shape)
    
    return model, scaler_X, scaler_y, history, y_high_test, y_low_test, y_high_pred, y_low_pred, timestamps_test

def plot_history(history):
    """
    Plot the training history of the model.
    
    Args:
    history (tf.keras.callbacks.History): Training history object.
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_predictions(actual, predicted, title):
    """
    Plot actual vs predicted values.
    
    Args:
    actual (np.array): Actual values.
    predicted (np.array): Predicted values.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def save_actual_vs_predicted(y_high_test, y_low_test, y_high_pred, y_low_pred, timestamps, forecast_horizon=2):
    """
    Save actual vs predicted results to a CSV file.
    
    Args:
    y_high_test, y_low_test (np.array): Actual high and low values.
    y_high_pred, y_low_pred (np.array): Predicted high and low values.
    timestamps (np.array): Timestamps for the predictions.
    forecast_horizon (int): Number of steps forecasted.
    """
    results_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual_High_t+1': y_high_test[:, 0],
        'Predicted_High_t+1': y_high_pred[:, 0],
        'Actual_Low_t+1': y_low_test[:, 0],
        'Predicted_Low_t+1': y_low_pred[:, 0],
        'Actual_High_t+2': y_high_test[:, 1],
        'Predicted_High_t+2': y_high_pred[:, 1],
        'Actual_Low_t+2': y_low_test[:, 1],
        'Predicted_Low_t+2': y_low_pred[:, 1]
    })
    
    results_df.to_csv('actual_vs_predicted_results.csv', index=False)
    print("Actual vs Predicted results saved to actual_vs_predicted_results.csv")
    print(f"Shape of saved results: {results_df.shape}")
    
    # Print the first few rows for verification
    print("\nFirst few rows of results:")
    print(results_df.head())

def analyze_prediction_error_over_time(y_true, y_pred, dates, title):
    """
    Analyze and plot prediction error over time.
    
    Args:
    y_true (np.array): Actual values.
    y_pred (np.array): Predicted values.
    dates (np.array): Timestamps for the predictions.
    title (str): Title for the plot.
    """
    errors = np.abs(y_true - y_pred)
    plt.figure(figsize=(15, 7))
    plt.plot(dates, errors)
    plt.title(f'Absolute Prediction Error Over Time - {title}')
    plt.xlabel('Date')
    plt.ylabel('Absolute Error')
    plt.savefig(f'error_over_time_{title.replace(" ", "_")}.png')
    plt.close()

def plot_error_distribution(y_true, y_pred, title):
    """
    Plot the distribution of prediction errors.
    
    Args:
    y_true (np.array): Actual values.
    y_pred (np.array): Predicted values.
    title (str): Title for the plot.
    """
    errors = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'Distribution of Prediction Errors - {title}')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(f'error_distribution_{title.replace(" ", "_")}.png')
    plt.close()

def analyze_feature_importance(model, feature_names):
    """
    Analyze and plot feature importance.
    
    Args:
    model (tf.keras.Model): Trained model.
    feature_names (list): List of feature names.
    
    Returns:
    pd.DataFrame: DataFrame with feature importance scores.
    """
    importance = np.abs(model.get_weights()[0]).sum(axis=1)
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_importance

if __name__ == "__main__":
    file_path = 'EURUSD23th15.csv'  # Update this path to your actual file location
    sequence_length = 20
    forecast_horizon = 2

    # Verify augmentation
    print("\nVerifying augmentation:")
    df = load_and_preprocess_data(file_path)
    print(df.head(10))  # Print first 10 rows to check augmented data

    # Process data and train model
    model, scaler_X, scaler_y, history, y_high_test, y_low_test, y_high_pred, y_low_pred, timestamps_test = process_and_train(file_path, sequence_length, forecast_horizon)

    # Calculate and print metrics
    for i in range(forecast_horizon):
        print(f"\nMetrics for t+{i+1}:")
        high_mae, high_mape = calculate_metrics(y_high_test[:, i], y_high_pred[:, i])
        low_mae, low_mape = calculate_metrics(y_low_test[:, i], y_low_pred[:, i])
        print(f"High - MAE: {high_mae:.5f}, MAPE: {high_mape:.2f}%")
        print(f"Low  - MAE: {low_mae:.5f}, MAPE: {low_mape:.2f}%")

    # Save model and scalers
    model.save('forex_model_joint_high_low.h5')
    joblib.dump(scaler_X, 'scaler_X_joint_high_low.joblib')
    joblib.dump(scaler_y, 'scaler_y_joint_high_low.joblib')
    print("Model and scalers saved.")

    # Generate and save various plots
    plot_history(history)
    plot_predictions(y_high_test[:, 0], y_high_pred[:, 0], 'High Price Predictions vs Actual (t+1)')
    plot_predictions(y_low_test[:, 0], y_low_pred[:, 0], 'Low Price Predictions vs Actual (t+1)')
    plot_predictions(y_high_test[:, 1], y_high_pred[:, 1], 'High Price Predictions vs Actual (t+2)')
    plot_predictions(y_low_test[:, 1], y_low_pred[:, 1], 'Low Price Predictions vs Actual (t+2)')

    # Check for logical consistency in predictions
    inconsistencies = np.sum(y_low_pred > y_high_pred)
    print(f"\nNumber of predictions where Low > High: {inconsistencies}")
    if inconsistencies > 0:
        print("Consider adjusting the model or implementing post-processing to ensure High >= Low")

    # Save actual vs predicted results
    save_actual_vs_predicted(y_high_test, y_low_test, y_high_pred, y_low_pred, timestamps_test, forecast_horizon)

    # Analyze prediction errors
    analyze_prediction_error_over_time(y_high_test[:, 0], y_high_pred[:, 0], timestamps_test, 'High (t+1)')
    analyze_prediction_error_over_time(y_low_test[:, 0], y_low_pred[:, 0], timestamps_test, 'Low (t+1)')
    plot_error_distribution(y_high_test[:, 0], y_high_pred[:, 0], 'High (t+1)')
    plot_error_distribution(y_low_test[:, 0], y_low_pred[:, 0], 'Low (t+1)')

    # Analyze feature importance
    feature_names = [col for col in df.columns if col not in ['High', 'Low']]
    feature_importance = analyze_feature_importance(model, feature_names)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Verify alignment of predictions
    print("\nVerifying alignment:")
    for i in range(5):  # Print first 5 entries
        print(f"Date: {timestamps_test[i]}")
        print(f"Actual High: {y_high_test[i, 0]:.5f}, Predicted High: {y_high_pred[i, 0]:.5f}")
        print(f"Actual Low: {y_low_test[i, 0]:.5f}, Predicted Low: {y_low_pred[i, 0]:.5f}")
        print()

    print("\nTraining and evaluation completed. Model, scalers, plots, and actual vs predicted results saved.")

    # List the contents of the working directory to verify
    print("\nFiles generated:")
    print(os.listdir())