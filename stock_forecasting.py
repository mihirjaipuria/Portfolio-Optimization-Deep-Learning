# Import necessary libraries
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import ta
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import yfinance as yf

# Define functions for feature engineering
def add_log_features(df):
    """Add logarithmic features to the dataframe."""
    for i in range(1, 5):
        df[f'Log_Close_{i}'] = np.log(df['Close'] / df['Close'].shift(i))
    for i in range(4):
        df[f'Log_High_Open_{i}'] = np.log(df['High'] / df['Open'].shift(i))
    df.dropna(inplace=True)
    return df

def add_moving_averages(df):
    """Add moving average features to the dataframe."""
    df['EMA_9'] = df['Close'].ewm(span=9).mean().shift()
    for span in [5, 10, 15, 20, 25, 30]:
        df[f'SMA_{span}'] = df['Close'].rolling(span).mean().shift()
    df.dropna(inplace=True)
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df.dropna(inplace=True)
    return df

def add_more_features(df):
    """Add additional features to the dataframe."""
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['Volatility'] = df['Close'].rolling(window=30).std()
    df.dropna(inplace=True)
    return df

# Fetch data from Yahoo Finance
stock_symbol = 'V'
df = yf.download(stock_symbol, start='2011-12-01', end='2023-12-1')

# Apply feature engineering functions
df = add_log_features(df)
df = add_moving_averages(df)
df = add_technical_indicators(df)
df = add_more_features(df)

# Define features for the model
features = [
    'Close', 'EMA_9', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_20', 'SMA_25', 'SMA_30',
    'Log_Close_1', 'Log_Close_2', 'Log_Close_3', 'Log_Close_4',
    'Log_High_Open_0', 'Log_High_Open_1', 'Log_High_Open_2', 'Log_High_Open_3',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Diff', 'ATR_14', 'SMA_50', 'SMA_200', 'Volatility'
]

df_for_training = df[features].astype(float)

# Scale the features
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)

# Prepare data for LSTM model
trainX = []
trainY = []
n_future = 1
n_past = 14

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, :])
    trainY.append(df_for_training_scaled[i + n_future - 1, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, shuffle=False)

# Define the model building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=192, step=32),
                   activation='relu', return_sequences=True,
                   input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=192, step=32),
                   activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

# Set up Keras Tuner for hyperparameter optimization
TUNER_DIRECTORY = f'x7_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
tuner = kt.RandomSearch(build_model,
                        objective='val_mean_absolute_error',
                        max_trials=5,
                        executions_per_trial=1,
                        directory=TUNER_DIRECTORY,
                        project_name='stock_predictionfinal')

# Add EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform the hyperparameter search
tuner.search(trainX, trainY, epochs=150, validation_split=0.2, callbacks=[early_stopping])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Make predictions on the test set
predictions_test = best_model.predict(X_test)

# Calculate performance metrics
mae_test = mean_absolute_error(y_test, predictions_test)
mse_test = mean_squared_error(y_test, predictions_test)
r2_test = r2_score(y_test, predictions_test)
mape_test = mean_absolute_percentage_error(y_test, predictions_test)

print(f'Testing Set Metrics - MAE: {mae_test}, MSE: {mse_test}, R2: {r2_test}, MAPE: {mape_test}')

# Prepare data for future predictions
n_days_for_prediction = 30
n_past = 3
last_date = pd.to_datetime(df['Date'].iloc[-1])
predict_start_date = last_date + BDay(1)  
predict_period_dates = pd.date_range(start=predict_start_date, periods=n_days_for_prediction, freq='B')

trainX_reshaped = trainX[-n_past:]
prediction = best_model.predict(trainX_reshaped)
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 0]

# Create a dataframe with the predicted values
df_forecast = pd.DataFrame({'Date': predict_period_dates, 'Close': y_pred_future})
print(df_forecast)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df_combined['Date'], df_combined['Close'], label='Stock Prices', color='blue')
plt.axvline(df['Date'].iloc[-1], color='grey', linestyle='--', label='Forecast Start')
plt.title('Stock Price Prediction for 2024')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the forecasted data
base_path = 'forecasts_final_12_'
forecast_csv_path = f'{base_path}{stock_symbol}.csv'
df_forecast.to_csv(forecast_csv_path, index=False)
print(f'Forecasted data saved to {forecast_csv_path}')
