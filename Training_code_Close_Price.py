!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install

!pip install TA-Lib

!pip install --upgrade keras-tcn

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib
import tensorflow as tf
from tcn import TCN
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class CustomTCN(TCN):
    def build(self, input_shape):
        super(CustomTCN, self).build(input_shape)
        if not self.return_sequences:
            build_output_shape = list(self.build_output_shape) if isinstance(self.build_output_shape, tuple) else self.build_output_shape
            self.slicer_layer.build(build_output_shape)

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the forex data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Date_Time', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Hour_of_Day'] = df.index.hour
    df['Week_of_Year'] = df.index.isocalendar().week
    
    return df

def create_features(df):
    """
    Create additional features for the forex data.
    """
    new_columns = {}
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        for lag in range(1, 21):
            new_columns[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    for window in [5, 10, 20, 50]:
        new_columns[f'Close_MA_{window}'] = df['Close'].rolling(window=window).mean()
        new_columns[f'Close_STD_{window}'] = df['Close'].rolling(window=window).std()
        new_columns[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
    
    new_columns['RSI'] = talib.RSI(df['Close'])
    new_columns['MACD'], new_columns['MACD_Signal'], _ = talib.MACD(df['Close'])
    new_columns['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
    new_columns['Bollinger_Upper'], new_columns['Bollinger_Middle'], new_columns['Bollinger_Lower'] = talib.BBANDS(df['Close'])
    
    new_columns['Price_Change_1'] = df['Close'].pct_change(fill_method=None).fillna(0)
    new_columns['Price_Change_2'] = df['Close'].pct_change(periods=2, fill_method=None).fillna(0)
    
    new_columns['is_high_volatility'] = ((df.index.hour >= 9) & (df.index.hour <= 17)).astype(int)
    new_columns['is_morning_spike'] = ((df.index.hour >= 9) & (df.index.hour <= 10)).astype(int)
    new_columns['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    new_columns['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    new_df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    return new_df

def prepare_sequences(df, sequence_length=20, forecast_horizon=2):
    """
    Prepare input sequences and target values for the model.
    """
    feature_columns = [col for col in df.columns if col != 'Close']
    X = df[feature_columns].values
    y = df['Close'].values
    
    X_sequences, y_sequences = [], []
    for i in range(len(X) - sequence_length - forecast_horizon + 1):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length : i + sequence_length + forecast_horizon])
    
    return np.array(X_sequences), np.array(y_sequences)

def create_enhanced_tcn_model(input_shape, output_steps):
    """
    Create an enhanced TCN model with additional dense layers for time features.
    """
    inputs = Input(shape=input_shape)
    
    main_features = inputs[:, :, :-8]
    time_features = inputs[:, :, -8:]
    
    x = CustomTCN(
        nb_filters=128,
        kernel_size=3,
        nb_stacks=2,
        dilations=[1, 2, 4, 8, 16, 32],
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.2,
        return_sequences=False
    )(main_features)
    
    t = Dense(32, activation='relu')(time_features[:, -1, :])
    t = Dense(16, activation='relu')(t)
    
    combined = Concatenate()([x, t])
    outputs = Dense(output_steps)(combined)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def calculate_metrics(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mape

def plot_error_distribution(errors, title):
    """
    Plot the distribution of errors and save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(title)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

def process_and_train(file_path, sequence_length=20, forecast_horizon=2):
    """
    Main function to process data, create and train the model, and evaluate its performance.
    """
    df = load_and_preprocess_data(file_path)
    df = create_features(df)
    df.dropna(inplace=True)
    
    high_error_mask = (df.index.hour >= 9) & (df.index.hour <= 17)
    high_error_data = df[high_error_mask]
    augmented_data = high_error_data.copy()
    augmented_data.index += pd.Timedelta(minutes=30)
    df = pd.concat([df, augmented_data]).sort_index()
    
    X, y = prepare_sequences(df, sequence_length, forecast_horizon)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(y_train.shape)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    model = create_enhanced_tcn_model(input_shape=(sequence_length, X_train.shape[-1]), output_steps=forecast_horizon)
    
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
        X_train_scaled, y_train_scaled,
        validation_split=0.1,
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).reshape(y_train_pred_scaled.shape)
    train_mae_t1, train_mape_t1 = calculate_metrics(y_train[:, 0], y_train_pred[:, 0])
    train_mae_t2, train_mape_t2 = calculate_metrics(y_train[:, 1], y_train_pred[:, 1])
    
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).reshape(y_test_pred_scaled.shape)
    test_mae_t1, test_mape_t1 = calculate_metrics(y_test[:, 0], y_test_pred[:, 0])
    test_mae_t2, test_mape_t2 = calculate_metrics(y_test[:, 1], y_test_pred[:, 1])
    
    plot_error_distribution(y_test[:, 0] - y_test_pred[:, 0], 'Test Error Distribution (t+1)')
    plot_error_distribution(y_test[:, 1] - y_test_pred[:, 1], 'Test Error Distribution (t+2)')
    
    test_dates = df.index[-len(X_test) - forecast_horizon + 1:-forecast_horizon + 1]
    results_df = pd.DataFrame(index=test_dates)
    results_df['Actual_t+1'] = y_test[:, 0]
    results_df['Predicted_t+1'] = y_test_pred[:, 0]
    results_df['Actual_t+2'] = y_test[:, 1]
    results_df['Predicted_t+2'] = y_test_pred[:, 1]
    results_df.to_csv('predicted_vs_actual.csv')
    
    return model, scaler_X, scaler_y, history, (train_mae_t1, train_mape_t1), (train_mae_t2, train_mape_t2), (test_mae_t1, test_mape_t1), (test_mae_t2, test_mape_t2), df

def save_model_and_scalers(model, scaler_X, scaler_y, model_path, scaler_X_path, scaler_y_path):
    """
    Save the trained model and scalers to disk.
    """
    model.save(model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)
    print(f"Model and scalers saved to {os.path.dirname(model_path)}")

def prepare_input_sequence(df, sequence_length, feature_columns):
    """
    Prepare the most recent input sequence for prediction.
    """
    recent_data = df[feature_columns].values[-sequence_length:]
    return recent_data.reshape(1, sequence_length, -1)

def predict_next_periods(model, scaler_X, scaler_y, input_sequence, periods=2):
    """
    Predict the next periods using the trained model.
    """
    current_sequence = input_sequence.copy()
    predictions = []
    
    for _ in range(periods):
        current_sequence_scaled = scaler_X.transform(current_sequence.reshape(-1, current_sequence.shape[-1])).reshape(current_sequence.shape)
        next_pred_scaled = model.predict(current_sequence_scaled)
        next_pred = scaler_y.inverse_transform(next_pred_scaled.reshape(-1, 1)).flatten()
        predictions.append(next_pred[0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, -1] = next_pred[0]  # Assuming 'Close' is the last feature
    
    return np.array(predictions)

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate the predictions using various metrics.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return {
        'MAE': mae,
        'MAPE': mape,
        'RMSE': rmse
    }

if __name__ == "__main__":
    # ... (previous code remains the same)

    save_model_and_scalers(model, scaler_X, scaler_y, model_path, scaler_X_path, scaler_y_path)
    
    print("\nTraining completed. Model and scalers saved.")
    print("Error distribution plots saved as PNG files.")
    print("Predicted vs Actual values saved to 'predicted_vs_actual.csv'.")

    # Prepare for future predictions
    feature_columns = [col for col in df.columns if col != 'Close']
    sequence_length = 20  # Make sure this matches the sequence_length used in training
    recent_sequence = prepare_input_sequence(df, sequence_length, feature_columns)
    
    # Make predictions for the next two periods
    future_predictions = predict_next_periods(model, scaler_X, scaler_y, recent_sequence, periods=2)
    
    print("\nPredictions for the next two periods:")
    print(f"t+1: {future_predictions[0]:.4f}")
    print(f"t+2: {future_predictions[1]:.4f}")

    # Create a DataFrame with the predictions and save it
    last_date = df.index[-1]
    prediction_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(len(future_predictions))]
    prediction_df = pd.DataFrame({
        'Predicted_Close': future_predictions
    }, index=prediction_dates)
    prediction_df.to_csv('future_predictions.csv')
    print("\nFuture predictions saved to 'future_predictions.csv'")

    # Plot the training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'.")

    # Plot actual vs predicted values for the test set
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(test_metrics_t1):], df['Close'].values[-len(test_metrics_t1):], label='Actual', alpha=0.7)
    plt.plot(df.index[-len(test_metrics_t1):], y_test_pred[:, 0], label='Predicted t+1', alpha=0.7)
    plt.title('Actual vs Predicted Close Price (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.savefig('actual_vs_predicted_test.png')
    plt.close()
    print("Actual vs Predicted plot for test set saved as 'actual_vs_predicted_test.png'.")

    # Additional analysis: Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.layers[0].get_weights()[0].sum(axis=(0,1))
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'.")

    print("\nScript execution completed successfully.")

# You can add more helper functions or analysis code here if needed