# Financial Market Realized Volatility Prediction

## Overview

This project focuses on predicting short-term future realized volatility for selected stocks using high-frequency order book data. The pipeline encompasses several stages: initial data ingestion and processing, extensive feature engineering, stock selection through clustering and performance scoring, and finally, the application and evaluation of multiple machine learning models, including Weighted Least Squares (WLS), Random Forest, Long Short-Term Memory (LSTM) networks, and Transformer networks. The primary goal is to build robust models that accurately forecast volatility, a critical component in financial risk management and trading strategies.

---

## Workflow

The project follows a structured workflow, from raw data to model prediction and evaluation.

### 1. Data Ingestion and Initial Processing

- **Data Source**: The initial dataset consists of individual stock order book data stored in multiple CSV files.
- **Schema Definition**: A specific schema is defined for reading the CSV files, ensuring correct data types for columns like `time_id`, `seconds_in_bucket`, price levels (`bid_price1`, `ask_price1`, etc.), and size levels (`bid_size1`, `ask_size1`, etc.).
- **Data Aggregation**: Data from all individual stock CSV files are scanned and collected into a single Polars DataFrame.
- **Parquet Conversion**: The aggregated DataFrame is then written to a Parquet file (`112Stocks.parquet`) for efficient storage and faster subsequent reads. This Parquet file is then read using Pandas for further processing.

### 2. Preliminary Feature Engineering and Stock Filtering

This phase aims to derive initial features for each stock and `time_id` combination, filter out stocks with pathological volatility patterns, cluster similar stocks, and select a subset of stocks for more intensive modeling.

#### a. Snapshot Features Calculation

For each row (snapshot) in the order book data, the following features are calculated:

- **Micro-price**: A price derived from the best bid and ask prices and their corresponding sizes, designed to be more informative than the mid-price.
- **Spread1 & Spread2**: The difference between the best ask and bid prices (`ask_price1 - bid_price1`) and second-best ask and bid prices (`ask_price2 - bid_price2`).
- **Imbalance Size1**: The normalized difference between the best bid and ask sizes, indicating short-term order book pressure.
- **Book Pressure**: The ratio of total bid sizes (levels 1 and 2) to the total bid and ask sizes, indicating overall buy-side pressure.

#### b. Time Aggregated Features

The snapshot features are then aggregated per `stock_id` and `time_id`:

- **Realized Volatility**: Calculated as the standard deviation of log returns of the micro-price within each `time_id` window.
- **Realized Skewness**: The skewness of log returns of the micro-price.
- **Autocorrelation of Log Returns**: The Pearson correlation of log returns with their first lag.
- **Tick Frequency**: The number of data points (order book updates) within the `time_id`.
- **Mean of Snapshot Features**: Averages of micro-price, spreads, imbalance, book pressure, and bid/ask sizes over the `time_id` window.

#### c. Volatility-Based Stock Filtering

To remove stocks with unstable or extremely low volatility, a filtering process is applied:

- The mean realized volatility is calculated for each stock across all its `time_id` periods.
- An Interquartile Range (IQR) based outlier detection method is used. Stocks whose mean realized volatility falls outside `Q1 - 1.5*IQR` and `Q3 + 1.5*IQR` or below a small epsilon are filtered out.

#### d. Stock Clustering (K-Means)

The filtered stocks are then clustered based on their aggregated time-id features:

- **Meta-Features**: For each stock, the mean of its `time_id` features (e.g., mean realized volatility, mean spread1, mean tick frequency) are calculated to serve as meta-features representing the stock's typical behavior.
- **Scaling**: These meta-features are standardized using `StandardScaler`.
- **K-Means Clustering**: `KMeans` clustering is applied to the scaled meta-features to group stocks with similar characteristics. The number of clusters (`N_CLUSTERS`) is a predefined parameter.

#### e. Performance Scoring and Top Stock Selection

An expanding window approach is used to evaluate the predictability of each stock's realized volatility:

- **Features for Scoring**: A subset of features (`realized_volatility`, `mean_spread1`, `mean_imbalance_size1`, `mean_book_pressure`, `mean_micro_price`) are used as predictors, with their previous time*id's values (`prev*\*`) serving as inputs.
- **Target**: The current `realized_volatility` is the target.
- **R-squared Calculation**: For each stock, a `LinearRegression` model is trained on an expanding window of past `time_id` data to predict the next `time_id`'s realized volatility. The R-squared score is calculated based on these out-of-sample predictions.
- **QLIKE Calculation**: The QLIKE loss is calculated using the historical mean of realized volatility from the training window as the forecast for the next period.
- **Combined Score**: A combined score (`R2_WEIGHT * R-squared - QLIKE_WEIGHT * QLIKE`) is computed to rank stocks.
- **Selection**: Although a dynamic top N stock selection based on the combined score is implemented, the script subsequently uses a hardcoded list of `selected_stock_ids` to create the final dataset (`Data/30Stocks.parquet`) for the main modeling phase.

### 3. Advanced Feature Engineering for Predictive Modeling

The `make_features` function processes the data for the selected 30 stocks to generate a rich feature set for the predictive models:

- **Price and Spread Features**: `mid_price`, `spread`, `normalized_spread`.
- **Order Book Imbalance Features**: `imbalance` (level 1), `book_pressure` (levels 1 & 2), `OBI_L2` (Order Book Imbalance using level 2 data).
- **Entropy Features**: `LOB_entropy` (Limit Order Book entropy based on bid/ask sizes at levels 1 & 2), `LOB_entropy_normalized`.
- **Return Features**: `log_return` (based on mid-price), `log_wap_return` (based on Weighted Average Price - WAP).
- **Volatility Features**:
  - `realized_volatility`: Rolling sum of squared log returns (window of 30).
  - `bipower_var`: Bipower variation, a robust measure of volatility.
  - `rolling_vol_30`: Rolling standard deviation of log returns.
- **Target Variable**: `rv_future` (realized volatility shifted 30 periods into the future).
- **Lagged Features**: Lagged versions (lag 1 and 2) of `imbalance`, `book_pressure`, and `log_return`.
- **Rolling Mean Features**: `rolling_imbalance_mean_30`.
- **Time-Based Features**: `sec_sin`, `sec_cos` derived from `seconds_in_bucket` to capture cyclical patterns.
- **Size Transformations**: Log transformation (`np.log1p`) of bid/ask sizes for levels 1 & 2.
- The dataset is cleaned of NaNs and infinite values.

### 4. Feature Selection

After generating features, a selection process is applied:

- **Low Variance Filter**: `VarianceThreshold` is used to remove features with zero variance.
- **High Correlation Filter**: Features with a Spearman correlation greater than 0.98 with another feature (where the other feature has a higher sum of correlations) are dropped to reduce multicollinearity.
- The resulting feature set is saved to `Data/FE30Stocks.parquet`.

### 5. Predictive Modeling

Several models are trained to predict the `rv_future` (or its log transformation).

#### a. Weighted Least Squares (WLS)

- **Target**: `rv_future`.
- **Features**: A predefined list of engineered features.
- **Weighting**: Observations are weighted by the inverse of the rolling variance of the target variable (`y.rolling(2000, min_periods=1).var()`), giving more weight to periods with lower target variance.
- **Splitting**: Data is split into 80% training and 20% testing sets chronologically.
- **Model**: `statsmodels.WLS` is used.
- **Evaluation**: R-squared and QLIKE loss on the test set.

#### b. Random Forest Regressor

- **Target**: `rv_future_log` (logarithm of 1 + `rv_future`).
- **Features**: A predefined list of engineered features.
- **Splitting**: Data is split based on unique `time_id` values: 80% for train/validation, 20% for testing. The train/validation set is further split into 90% training and 10% validation. This ensures that data from the same `time_id` (session) does not leak across sets.
- **Model**: `sklearn.ensemble.RandomForestRegressor` with specified hyperparameters (e.g., `n_estimators=500`, `min_samples_leaf=3`).
- **Evaluation**: Root Mean Squared Error (RMSE), R-squared, and QLIKE loss on the test set (predictions are converted back to the original volatility scale before QLIKE calculation). Plots of predicted vs. actual volatility and residuals are generated.

#### c. Long Short-Term Memory (LSTM) Network

- **Target**: `rv_future_log`.
- **Features**: A predefined list, excluding the original target.
- **Preprocessing**:
  - Features are scaled using `MinMaxScaler`.
  - The target (`rv_future_log`) is also scaled using `MinMaxScaler`.
  - Data is transformed into sequences of length `SEQ_LEN` (30) using the `build_sequences` function. Each sequence of past `SEQ_LEN` feature vectors is used to predict the target at the end of the sequence.
- **Splitting**: Similar `time_id`-based split as Random Forest for train, validation, and test sets.
- **Model Architecture**:
  - `LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, num_features))`
  - `Dropout(0.2)`
  - `LSTM(32)`
  - `Dropout(0.2)`
  - `Dense(16, activation='relu')`
  - `Dense(1)` (output layer)
- **Training**: Compiled with Adam optimizer (learning rate 1e-4) and MSE loss. `EarlyStopping` is used to prevent overfitting.
- **Evaluation**: RMSE, R-squared, and QLIKE loss on the test set (predictions are inverse-transformed to the original volatility scale). Plots of predicted vs. actual volatility and training/validation loss curves are generated.

#### d. Transformer Network

- **Target**: `rv_future_log`.
- **Features and Preprocessing**: Similar to the LSTM setup (scaled features, scaled target, sequence generation).
- **Splitting**: Similar `time_id`-based split as LSTM/Random Forest.
- **Model Architecture (`build_transformer_model`)**:
  - Input layer.
  - `Dense` layer to project features to `d_model`.
  - Multiple Transformer blocks (default `num_layers=2`):
    - `MultiHeadAttention` (default `num_heads=4`).
    - Add & Norm (residual connection and Layer Normalization).
    - FeedForward Network (Dense layers).
    - Add & Norm.
  - `GlobalAveragePooling1D` to pool sequence information.
  - `Dense(1)` (output layer).
- **Training**: Compiled with Adam optimizer (learning rate 1e-3) and MSE loss. `EarlyStopping` is used.
- **Evaluation**: RMSE, R-squared, and QLIKE loss on the test set (predictions are inverse-transformed). Plots including actual vs. predicted volatility over time, scatter plot of predicted vs. actual, and residual distribution are generated.

### 6. Evaluation Metrics

The primary metrics used for evaluating model performance are:

- **R-squared ($R^2$)**: Proportion of the variance in the dependent variable that is predictable from the independent variables.
- **Root Mean Squared Error (RMSE)**: Square root of the average of squared differences between prediction and actual observation. Primarily used for models predicting log-transformed target.
- **QLIKE Loss**: A loss function commonly used for volatility prediction, defined as:
  $QLIKE(y_{true}, y_{pred}) = \text{mean}(y_{true} / y_{pred} - \log(y_{true} / y_{pred}) - 1)$
  Values are clipped to avoid division by zero or log of non-positive numbers.

---

## Key Libraries Used

- **Data Handling & Manipulation**: `pandas`, `polars`, `numpy`
- **Machine Learning (Classical)**: `scikit-learn` (for `MinMaxScaler`, `StandardScaler`, `VarianceThreshold`, `LinearRegression`, `RandomForestRegressor`, `KMeans`, metrics)
- **Statistical Modeling**: `statsmodels` (for `WLS`)
- **Deep Learning**: `tensorflow.keras` (for `LSTM`, `Dense`, `Dropout`, `MultiHeadAttention`, `LayerNormalization`, `Input`, model building, callbacks)
- **Scientific Computing**: `scipy.stats` (for `skew`, `pearsonr`)
- **Plotting**: `matplotlib.pyplot`
- **File System**: `glob`, `os`
- **Warnings Management**: `warnings`

---

## Code Structure

The project is contained within a single Python script (`Main.py`).

- **Data Paths**: The script uses relative paths for input data (e.g., `"Data/individual_book_train/*.csv"`) and output artifacts (e.g., `"Data/112Stocks.parquet"`, `"Data/FE30Stocks.parquet"`). Some local paths (`/Users/ayush/...`) are present in the later parts of the script for reading the feature-engineered data; these would need adjustment for different environments.
- **Functions**: The script is organized into several functions for distinct tasks like feature calculation (`calculate_basic_features_snapshot`, `calculate_time_id_features`, `make_features`), model building (`build_lstm_model`, `build_transformer_model`), and evaluation (`qlike_loss`, `qlike_safe`).
- **Workflow Sections**: The script progresses sequentially through the stages described in the Workflow section.

---

## Setup and Usage

1.  **Environment**: Ensure Python 3.x is installed along with all the libraries listed in "Key Libraries Used". These can typically be installed via pip:
    ```bash
    pip install pandas polars numpy scikit-learn statsmodels tensorflow matplotlib scipy
    ```
2.  **Data**:
    - Place the initial CSV files in the `Data/individual_book_train/` directory relative to where the script is run.
    - The script will generate intermediate Parquet files in the `Data/` directory.
    - If running parts of the script independently, ensure that the required Parquet files (e.g., `Data/FE30Stocks.parquet` for modeling) are available. **Note**: The script contains hardcoded absolute paths for reading `FE30Stocks.parquet` in the modeling sections; these must be updated to the correct local path.
3.  **Execution**:

    - Run the script from the command line: `python Main.py`
    - The script will print progress updates and evaluation results to the console.
    - Plots generated by `matplotlib` will be displayed if run in an environment that supports GUI windows.

4.  **Configuration**:
    - Key parameters are defined at the beginning of the script (e.g., `RANDOM_STATE`, `VOLATILITY_IQR_MULTIPLIER`, `N_CLUSTERS`, `MIN_PERIODS_FOR_MODEL`, `SEQ_LEN`). These can be adjusted to experiment with different settings.
    - The list of `selected_stock_ids` for the final modeling stage is hardcoded. To use a dynamically selected list of stocks, modifications would be needed to utilize the output of the `top_stocks` DataFrame.
