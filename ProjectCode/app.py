import pandas as pd
import numpy as np
import streamlit as st
import tweepy
import nltk
import yfinance as yf
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Download VADER lexicon for sentiment analysis
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Twitter API credentials (replace with your keys and tokens)
API_KEY = "EDPeNzunuQSfyZg4fkorzFfo0"
API_SECRET = "z6JrMT8MJNon4TGdp7ab5J6JoL2ORelMqt4sAwWPN7XvzjCOlF"
ACCESS_TOKEN = "1864911274246836226-jyf7xyALdHD7JcxEDen6VJq1Ee0G8I"
ACCESS_TOKEN_SECRET = "Ccss4zIk7RyVEb8fFnnoeJEZSsQjQGUNO5iX7EbyWDdWF"
BEARER_TOKEN = f"AAAAAAAAAAAAAAAAAAAAAMCMxQEAAAAAhP5MkC7dK1o9Rt9M9P%2Fr64bA06s%3DAtKWvdAzKFOUn5reDAHqg0vkpoNhxkGdYdmU3t0BZWKbHYYhui"

# Authenticate with Twitter API
client = tweepy.Client(bearer_token=BEARER_TOKEN)


# Function to fetch tweets mentioning the company
def fetch_tweets(query, max_results=10):
    tweets = client.search_recent_tweets(
        query=query,
        tweet_fields=["created_at", "author_id", "text", "lang"],
        max_results=max_results,
    )
    tweet_data = []
    if tweets.data:
        for tweet in tweets.data:
            if tweet.lang == "en":
                tweet_data.append(
                    {
                        "created_at": tweet.created_at,
                        "author_id": tweet.author_id,
                        "Tweet content": tweet.text,
                    }
                )
    return pd.DataFrame(tweet_data)


# Function to fetch stock data from Yahoo Finance
def get_data_from_date(start_date, end_date, ticker_symbol):
    company_data = yf.download(
        ticker_symbol, start=start_date, end=end_date, interval="1d"
    )
    return company_data


# Streamlit app title
st.title("Company Sentiment & Stock Price Prediction")

# User input for company name and prediction date
company_name = st.text_input(
    "Enter the company or keyword to fetch tweets for:",
    "",
    placeholder="e.g. GOOG for Google",
)
prediction_date = st.date_input("Enter the prediction date:", datetime.today())

# Submit button for processing
if st.button("Submit"):
    if company_name and prediction_date:
        # Fetch tweets mentioning the company
        try:
            with st.spinner(f"Fetching tweets mentioning '{company_name}'..."):
                tweet_df = fetch_tweets(company_name, max_results=20)
        except Exception as e:
            st.warning(f"Error fetching tweets: {e}")
            tweet_df = pd.DataFrame()  # Set to an empty dataframe if error occurs

        if not tweet_df.empty:
            # Perform sentiment analysis on tweets
            tweet_df["sentiment_score"] = tweet_df["Tweet content"].apply(
                lambda tweet: sia.polarity_scores(tweet)["compound"]
            )
            tweet_df["predicted_sentiment"] = tweet_df["sentiment_score"].apply(
                lambda score: (
                    "Positive"
                    if score > 0.05
                    else ("Negative" if score < -0.05 else "Neutral")
                )
            )

            # Display tweet sentiment analysis results
            st.write("Sentiment Analysis of Tweets:")
            st.write(
                tweet_df[["Tweet content", "sentiment_score", "predicted_sentiment"]]
            )

            # Plot sentiment distribution
            st.write("Sentiment Distribution")
            st.bar_chart(tweet_df["predicted_sentiment"].value_counts())
        else:
            st.warning(f"No tweets found mentioning '{company_name}'.")

        # Predict stock prices
        if company_name:
            # Calculate the start date (2 years before the prediction date)
            start_date = prediction_date - timedelta(days=2 * 365)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = prediction_date.strftime("%Y-%m-%d")

            # Fetch the stock data
            company_data = get_data_from_date(
                start_date_str, end_date_str, company_name
            )

            if len(company_data) < 60:
                st.error(
                    "Error: Not enough data for the specified start date. The dataset must have at least 60 data points."
                )
            else:
                # Extract the 'Close' prices for model training
                data = company_data[["Close"]].values

                # Scale the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)

                # Prepare input data for LSTM (using the last 60 days to predict the next day's stock price)
                x_input = []
                y_input = []

                for i in range(60, len(scaled_data)):
                    x_input.append(scaled_data[i - 60 : i, 0])
                    y_input.append(scaled_data[i, 0])

                x_input, y_input = np.array(x_input), np.array(y_input)
                x_input = np.reshape(x_input, (x_input.shape[0], x_input.shape[1], 1))

                # Build the LSTM model
                model = Sequential()
                model.add(
                    LSTM(128, return_sequences=True, input_shape=(x_input.shape[1], 1))
                )
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer="adam", loss="mean_squared_error")

                # Train the model (you can adjust epochs if needed)
                model.fit(x_input, y_input, batch_size=1, epochs=1, verbose=1)

                # Get the data from the last 60 days for prediction
                last_60_days_data = company_data[["Close"]].iloc[-60:]

                # Scale the last 60 days data
                last_60_days_scaled = scaler.transform(last_60_days_data.values)
                last_60_days_scaled = np.reshape(last_60_days_scaled, (1, 60, 1))

                # Predict the stock price for the given date
                predicted_price = model.predict(last_60_days_scaled)

                # Inverse transform the predicted price to the original scale
                predicted_price = scaler.inverse_transform(predicted_price)

                # Display the predicted price
                st.subheader(
                    f"Predicted stock price for {prediction_date.strftime('%Y-%m-%d')}: ${predicted_price[0][0]:.2f}"
                )

                # Plot actual and predicted prices
                fig, ax = plt.subplots(figsize=(16, 6))
                ax.set_title(
                    f'{company_name} Stock Price Prediction for {prediction_date.strftime("%Y-%m-%d")}'
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Close Price USD ($)")
                ax.plot(
                    company_data.index, company_data["Close"], label="Actual Prices"
                )
                ax.plot(
                    prediction_date,
                    predicted_price[0][0],
                    marker="o",
                    color="red",
                    markersize=10,
                    label="Predicted Price",
                )
                ax.legend(["Actual", "Predicted"], loc="lower right")
                st.pyplot(fig)

                # Allow the user to download the data
                output_csv = f"{company_name}_predictions.csv"
                company_data.to_csv(output_csv, index=False)
                st.download_button(
                    label="Download CSV with Predictions",
                    data=company_data.to_csv(index=False).encode("utf-8"),
                    file_name=output_csv,
                    mime="text/csv",
                )

    else:
        st.info("Please enter a company or keyword and prediction date.")
