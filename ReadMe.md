
# Stock Prediction and Sentiment Analysis Project

## Project Structure

### ProjectCode

- Inside this folder, there are two files: `app.py` and `app_csv.py`.

  - **`app.py`**: 
    - A web application script that integrates stock prediction models and market sentiment analysis. It uses the Yahoo Finance API to retrieve company stock data and the Twitter API to fetch real-time tweets.

  - **`app_csv.py`**:
    - A web application script that integrates stock prediction models and market sentiment analysis. Here, you need to manually upload a `.csv` file provided in the `Dataset` folder.

### Dataset

- **File**: `data.csv`
  - Merged dataset for Tesla (TSLA) stock predictions.
  - Contains **daily sentiment scores** as feature columns.
  - Benchmark dataset for comparing model performances.

### ComparisonModels

- Contains 7 models:
  - XGBoost
  - ANN (Artificial Neural Network)
  - CNN (Convolutional Neural Network)
  - VAR (Vector AutoRegressive model)
  - LSTM (Long Short-Term Memory)
  - RandomForest
  - LinearRegression

- All files inside this folder follow a common structure `modelName.ipynb`. Running these files will provide MSE (Mean Squared Error), MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) scores, and a graph of predicted versus actual values.

## How to Run the Project

1. **Running `Models.ipynb`**:
   - **Steps**:
     1. Open the project folder in VS Code and navigate to the folder containing `ModelName.ipynb`.
     2. Execute the notebook using the run button with the Python 3.12 environment.
     3. Provide the required inputs:
        - **Ticker Symbol**: Enter the stock ticker symbol (e.g., TSLA for Tesla).
        - **Prediction Date**: Enter the desired prediction date in the format `YY-MM-DD`.

2. **Running `app.py`**:
   - **Steps**:
     1. Navigate to the folder containing `app.py`.
     2. Activate the virtual environment:
        - **Windows**: `myvenv\Scripts\activate`
        - **Ubuntu**: `source myvenv/bin/activate`
     3. Run the following command:
        ```bash
        streamlit run app.py
        ```
     4. A browser window will open, allowing you to:
        - Enter a company name and date for stock prediction.
        - View stock predictions and market sentiments.

3. **Running `app_csv.py`**:
   - **Steps**:
     1. Navigate to the folder containing `app_csv.py`.
     2. Activate the virtual environment:
        - **Windows**: `myvenv\Scripts\activate`
        - **Ubuntu**: `source myvenv/bin/activate`
     3. Run the following command:
        ```bash
        streamlit run app.py
        ```
     4. A browser window will open, allowing you to:
        - Upload the merged Dataset (`data.csv`) from the `Dataset` folder.
        - Enter a company name and date for stock prediction.

### Notes
- Ensure the following before running the project:
  - All required dependencies are installed.
  - The Python version is 3.12.
