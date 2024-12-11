import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Stock Price Prediction with LSTM")

# File upload for the CSV data
uploaded_file = st.file_uploader("Upload your stock data CSV file:", type=["csv"])

if uploaded_file:
    # Load the CSV data
    data = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded data
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())

    # Ensure required columns exist in the dataset
    required_columns = [
        "Date", "Open", "High", "Low", "Close", "Adj Close", 
        "Volume", "Stock Name", "sentiment_score", "Negative", "Neutral", "Positive"
    ]
    if not all(col in data.columns for col in required_columns):
        st.error("The uploaded CSV does not contain all required columns.")
    else:
        # Convert the Date column to datetime
        data["Date"] = pd.to_datetime(data["Date"])

        # User input for stock name and prediction date
        stock_name = st.text_input("Enter the stock name (from the 'Stock Name' column):", "")
        prediction_date = st.date_input("Enter the prediction date:", datetime.today())

        # Submit button
        if st.button("Submit"):
            if stock_name:
                # Filter data for the selected stock name
                stock_data = data[data["Stock Name"] == stock_name]
                stock_data = stock_data.sort_values(by="Date")  # Ensure data is sorted by date

                # Check if data is sufficient
                if len(stock_data) < 60:
                    st.error("Not enough data available. At least 60 data points are required.")
                else:
                    # Extract the 'Close' price for modeling
                    close_prices = stock_data["Close"].values.reshape(-1, 1)

                    # Scale the data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(close_prices)

                    # Create feature and target datasets
                    x_input, y_input = [], []
                    for i in range(60, len(scaled_data)):
                        x_input.append(scaled_data[i - 60 : i, 0])  # Last 60 days
                        y_input.append(scaled_data[i, 0])  # Next day

                    x_input, y_input = np.array(x_input), np.array(y_input)

                    # Reshape the data for LSTM input
                    x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

                    # Train-test split
                    split = int(0.8 * len(x_input))  # 80% for training, 20% for testing
                    X_train, X_test = x_input[:split], x_input[split:]
                    y_train, y_test = y_input[:split], y_input[split:]

                    # Build LSTM model
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=1))  # Output layer for regression

                    # Compile the model
                    model.compile(optimizer="adam", loss="mean_squared_error")

                    # Train the model
                    model.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1)

                    # Predictions on train and test data
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Calculate evaluation metrics for training data
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    train_rmse = np.sqrt(train_mse)
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    train_r2 = r2_score(y_train, y_train_pred)

                    # Calculate evaluation metrics for testing data
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_r2 = r2_score(y_test, y_test_pred)

                    # Display evaluation metrics
                    st.subheader("Model Evaluation Metrics")

                   # st.write("### Training Data:")
                    #st.write(f"Mean Squared Error (MSE): {train_mse:.4f}")
                    #st.write(f"Root Mean Squared Error (RMSE): {train_rmse:.4f}")
                    #st.write(f"Mean Absolute Error (MAE): {train_mae:.4f}")
                    #st.write(f"R² Score: {train_r2:.4f}")

                    #st.write("### Testing Data:")
                    #st.write(f"Mean Squared Error (MSE): {test_mse:.4f}")
                    #st.write(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
                    #st.write(f"Mean Absolute Error (MAE): {test_mae:.4f}")
                    #st.write(f"R² Score: {test_r2:.4f}")

                    # Get the last 60 days of data for prediction
                    last_60_days = scaled_data[-60:]
                    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

                    # Predict the stock price for the specified date
                    predicted_price = model.predict(last_60_days)
                    predicted_price = scaler.inverse_transform(predicted_price)

                    # Display the prediction
                    st.subheader(
                        f"Predicted price for {stock_name} on {prediction_date.strftime('%Y-%m-%d')}: ${predicted_price[0][0]:.2f}"
                    )

                    # Plot the actual and predicted data
                    plt.figure(figsize=(12, 6))
                    plt.plot(close_prices[-200:], label="Actual Prices")
                    plt.scatter(
                        len(close_prices) - 1,
                        predicted_price[0][0],
                        color="red",
                        label="Predicted Price",
                        zorder=5,
                    )
                    plt.title(f"{stock_name} Stock Price Prediction")
                    plt.xlabel("Days")
                    plt.ylabel("Price (USD)")
                    plt.legend()
                    st.pyplot(plt)
            else:
                st.warning("Please enter a valid stock name.")
