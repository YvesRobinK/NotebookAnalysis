#!/usr/bin/env python
# coding: utf-8

# ## 🚀 Optiver Trading Challenge: XGBoost vs LightGBM Overview 🚀
# 
# ### Unveiling the Competitors: XGBoost & LightGBM 🥊
# 
# In the world of machine learning, the algorithms we choose are pivotal in navigating the labyrinth of data towards robust predictions. Two titans stand out in this arena – XGBoost and LightGBM. Both have their unique strengths and are celebrated for their performance, but let's dissect their differences to understand when to use each.
# 
# #### 🌿 LightGBM: The Speedster for Massive Datasets
# LightGBM, standing tall with its efficiency, is designed to sprint through massive datasets. It boasts a leaf-wise tree growth and histogram-based computing, excelling in classification, regression, and ranking tasks.
# 
# ##### Advantages:
# - **Swift and Scalable**: Thanks to its histogram-based feature binning and leaf-wise growth, LightGBM is a speed demon, especially on large datasets. It converges faster and demands less memory, making it ideal for large-scale applications.
# - **Precision in Large Data**: Its distributed computing capability makes it a behemoth for handling vast amounts of high-dimensional data, ensuring accurate models even on the grandest of data stages.
# - **Memory Efficiency**: While efficient, its leaf-wise approach may use more memory compared to other gradient boosting algorithms, potentially making it a hungry beast when working with limited memory resources.
# - **Categorical Features**: LightGBM has a nifty trick up its sleeve – it can effectively handle categorical features by numerical transformation, sidestepping the need for one-hot encoding and keeping the memory usage lean.
# 
# ##### Disadvantages:
# - **Thirsty for Memory**: LightGBM might demand more memory due to its leaf-wise expansion, potentially making it less ideal for memory-constrained situations.
# - **Noise Sensitive**: Its keen focus on larger gradient samples can lead to overfitting noisy data, necessitating a careful approach to data preparation and feature engineering.
# - **Categorical Feature Caveat**: Despite its prowess with numerical conversion of categorical features, LightGBM doesn't support native categorical features, often requiring additional preprocessing.
# 
# #### 🌲 XGBoost: The Balanced Powerhouse
# XGBoost shines with its excellent performance and scalability. Renowned for its stellar prediction accuracy and suitability across a range of objectives and metrics, XGBoost is a force to be reckoned with in the machine learning competitions.
# 
# ##### Advantages:
# - **Regularization Champ**: With its L1 and L2 regularization, XGBoost keeps overfitting in check, ensuring that models don't just memorize but generalize well.
# - **Missing Values Magician**: XGBoost can intuitively handle missing data, saving precious preprocessing time and maintaining the integrity of real-world data handling.
# - **Categorical Nuance**: Through one-hot encoding, XGBoost gives due diligence to categorical features, preserving their intricate relationships with the target variables.
# 
# ##### Disadvantages:
# - **Patience Required**: Training on large datasets might test your patience as XGBoost's level-wise growth can be time and resource-intensive.
# - **Memory Considerations**: Stability comes at a cost – the level-wise method can be heavy on memory, making it a bit cumbersome for constrained environments.
# 
# ### 🏁 The Verdict
# Both LightGBM and XGBoost are stellar performers with their own sets of pros and cons. Your choice might hinge on the scale of the dataset, computational resources, and the importance of model interpretability. In the end, both are capable of delivering the gold in constructing reliable and precise models.
# 
# ### 🚀 Introduction to the Notebook 🚀
# Armed with the knowledge of these two algorithms, we proceed to use XGBoost in our notebook, leveraging its strengths for our prediction tasks. Follow along as we unravel the story the data tells through XGBoost's robust and precise modeling.
# 
# ### Inspiration and Credits 🙌
# This notebook is a continuation from this notebook https://www.kaggle.com/code/zulqarnainali/explained-singel-model-optiver, which draws inspiration from the remarkable work of Angle, which can be found in [this Kaggle project](https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model/notebook). Special thanks to Angle for sharing valuable insights and code.
# 
# 🙏 Your time and consideration are greatly appreciated. If you find this notebook valuable, please give it a thumbs-up! 👍
# 
# ## Purpose 🎯
# This notebook serves several primary purposes:
# - Load and preprocess the competition data 📁
# - Engineer pertinent features for training predictive models 🏋️‍♂️
# - Train models to predict the target variable 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook follows a structured approach:
# 1. **Data Preparation**: We load and preprocess the competition data in this section.
# 2. **Feature Engineering**: The generation and selection of relevant features for model training are covered here.
# 3. **Model Training**: Machine learning models are trained on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 
# ## How to Use 🛠️
# To make the most of this notebook, please adhere to these steps:
# 1. Ensure you have the competition data and environment ready.
# 2. Execute the cells in order for data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adjust the code as needed to enhance model performance or experiment with different approaches.
# 
# **Note**: Be sure to replace any placeholder paths or configurations with your specific information.

# ## 🧹 Importing necessary libraries

# In[1]:


import gc  # Garbage collection for memory management
import os  # Operating system-related functions
import time  # Time-related functions
import warnings  # Handling warnings
from itertools import combinations  # For creating combinations of elements
from warnings import simplefilter  # Simplifying warning handling

# 📦 Importing machine learning libraries
import joblib  # For saving and loading models
import xgboost as xgb  # XGBoost gradient boosting framework
import numpy as np  # Numerical operations
import pandas as pd  # Data manipulation and analysis
from sklearn.metrics import mean_absolute_error  # Metric for evaluation
from sklearn.model_selection import KFold, TimeSeriesSplit  # Cross-validation techniques

# 🤐 Disable warnings to keep the code clean
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# 📊 Define flags and variables
is_offline = False  # Flag for online/offline mode
is_train = True  # Flag for training mode
is_infer = True  # Flag for inference mode
max_lookback = np.nan  # Maximum lookback (not specified)
split_day = 435  # Split day for time series data


# ## 📊 Data Loading and Preprocessing 📊
# 
# 
# 
# 
# 

# **Explaination**
# 
# 1. `df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")`
#    - This line reads a CSV (Comma-Separated Values) file named "train.csv" using the Pandas library and assigns the resulting DataFrame to the variable `df`. The CSV file is expected to be located at the specified file path, "/kaggle/input/optiver-trading-at-the-close/train.csv". This line is loading a dataset from a file.
# 
# 2. `df = df.dropna(subset=["target"])`
#    - This line drops (removes) rows from the DataFrame `df` where there are missing values (NaN) in the "target" column. It uses the `dropna` method with the `subset` parameter set to "target" to specify that it should check for missing values in the "target" column and remove rows that have missing values. The updated DataFrame is assigned back to the variable `df`.
# 
# 3. `df.reset_index(drop=True, inplace=True)`
#    - This line resets the index of the DataFrame `df`. When data is removed from a DataFrame, the index labels may have gaps or may not be sequential. This line resets the index to be sequential, starting from 0, and the old index is dropped. The `drop=True` parameter indicates that the old index should be dropped, and `inplace=True` means that this operation modifies the DataFrame in place.
# 
# 4. `df_shape = df.shape`
#    - This line calculates the shape of the DataFrame `df`, which means it returns a tuple containing the number of rows and columns in the DataFrame. The result is assigned to the variable `df_shape`.
# 
# To summarize, the code reads a dataset from a CSV file, removes rows with missing values in a specific column ("target"), resets the index of the DataFrame to make it sequential, and finally, it calculates and stores the shape of the resulting DataFrame in the `df_shape` variable. 

# In[2]:


# 📂 Read the dataset from a CSV file using Pandas
df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")

# 🧹 Remove rows with missing values in the "target" column
df = df.dropna(subset=["target"])

# 🔁 Reset the index of the DataFrame and apply the changes in place
df.reset_index(drop=True, inplace=True)

# 📏 Get the shape of the DataFrame (number of rows and columns)
df_shape = df.shape


# ## 🚀 Memory Optimization Function with Data Type Conversion 🧹

# **Explaination**
# 
# This code defines a function `reduce_mem_usage` that is used to reduce the memory usage of a Pandas DataFrame by optimizing the data types of its columns. 
# 
# 1. `def reduce_mem_usage(df, verbose=0):`
#    - This line defines a function called `reduce_mem_usage` that takes two parameters: `df`, which is the input Pandas DataFrame that needs memory optimization, and `verbose` (defaulting to 0), which is a flag to control whether or not to provide memory optimization information.
# 
# 2. `start_mem = df.memory_usage().sum() / 1024**2`
#    - This line calculates the initial memory usage of the input DataFrame `df` and stores it in the `start_mem` variable. It does this by using the `memory_usage()` method, which returns the memory usage of each column, and then sums these values. The result is divided by 1024^2 to convert it to megabytes.
# 
# 3. The code then enters a loop that iterates through each column of the DataFrame using the `for col in df.columns:` loop.
# 
# 4. Inside the loop, it checks the data type of the column using `col_type = df[col].dtype`.
# 
# 5. If the column's data type is not 'object' (i.e., it's numeric), it proceeds with the optimization.
# 
# 6. For integer columns:
#    - It checks the minimum and maximum values in the column (c_min and c_max).
#    - Depending on the range of values, it converts the column to the smallest integer data type that can accommodate the data while reducing memory usage. It checks for `int8`, `int16`, `int32`, and `int64` data types based on the data range.
# 
# 7. For float columns:
#    - Similar to integer columns, it checks the minimum and maximum values.
#    - It converts the column to a `float32` data type if the range is within the limits of `np.finfo(np.float32)`. The `np.finfo()` function is used to get the floating-point type's limits.
# 
# 8. If the column's data type is neither integer nor float and falls outside the specified ranges, it defaults to `float32`.
# 
# 9. If `verbose` is set to a truthy value (e.g., 1), it provides information about memory optimization, including the initial and final memory usage, and the percentage reduction in memory usage.
# 
# 10. Finally, the function returns the DataFrame with optimized memory usage.
# 
# This function is useful for reducing the memory footprint of a DataFrame, especially when working with large datasets, by converting columns to the most memory-efficient data types based on the data they contain. It can help improve performance and reduce memory-related issues.

# In[3]:


# 🧹 Function to reduce memory usage of a Pandas DataFrame
def reduce_mem_usage(df, verbose=0):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    
    # 📏 Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2

    # 🔄 Iterate through each column in the DataFrame
    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column's data type is not 'object' (i.e., numeric)
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column's data type is an integer
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Check if the column's data type is a float
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    # ℹ️ Provide memory optimization information if 'verbose' is True
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")

    # 🔄 Return the DataFrame with optimized memory usage
    return df


#  ## 🏎️Parallel Triplet Imbalance Calculation with Numba

# **Explanation**
# 
# This notebook illustrates the enhancement of financial time series analysis by leveraging the power of Numba's njit Just-In-Time compilation to calculate rolling averages and their pairwise Euclidean distances, thus creating insightful rolling features. Let's explore the components:
# 
# 1. `from numba import njit, prange`
#    - This line imports `njit` for JIT compilation, enhancing execution speed, and `prange` for parallel loop processing, enabling concurrent computations.
# 
# 2. `@njit(parallel=True)`
#    - Applied to our function, this decorator instructs Numba to optimize for performance, allowing for fast, parallel computations of rolling features.
# 
# 3. `def compute_rolling_features(df_values, window_sizes):`
#    - Here, we define a function to compute rolling features using Numba, which accepts:
#      - `df_values`: A NumPy array of the DataFrame values, representing financial time series data.
#      - `window_sizes`: A list of window sizes for which rolling averages are computed.
# 
# 4. `num_rows, num_features = df_values.shape`
#    - We determine the shape of our input array to understand the scope of our calculations.
# 
# 5. `rolling_features = np.empty((num_rows, len(window_sizes) * num_features))`
#    - An empty array is initialized to hold the computed rolling averages for the different window sizes across all features.
# 
# 6. We iterate over the range of window sizes to compute rolling averages for each size.
# 
# 7. `for i in prange(len(window_sizes)):` 
#    - This loop uses `prange` to parallelize the computation of rolling averages for different window sizes.
# 
# 8. Within the loop, we use a sliding window approach to calculate the averages across each window size.
# 
# 9. We then calculate pairwise Euclidean distances between the rolling average features using `np.linalg.norm`.
# 
# 10. These distances provide a dynamic measure of how different rolling windows of averages compare with each other, capturing trends and volatilities in the data.
# 
# 11. The computed distances are structured and returned as a new array, enriching the feature set for the machine learning model.
# 
# 12. The resulting rolling features and their distances serve as an input to an XGBoost model, offering the model a nuanced view of temporal patterns and relationships in the data.
# 
# 13. `calculate_rolling_features_numba` is an additional function that orchestrates the process, preparing the input data and invoking `compute_rolling_features` to generate and return a DataFrame populated with these new rolling features.

# In[4]:


# 🏎️ Import Numba for just-in-time (JIT) compilation and parallel processing
from numba import njit, prange

# 📊 Function to compute triplet imbalance in parallel using Numba
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    # 🔁 Loop through all combinations of triplets
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        
        # 🔁 Loop through rows of the DataFrame
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            # 🚫 Prevent division by zero
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

# 📈 Function to calculate triplet imbalance for given price data and a DataFrame
def calculate_triplet_imbalance_numba(price, df):
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features

@njit(fastmath=True)
def rolling_average(arr, window):
    """
    Calculate the rolling average for a 1D numpy array.
    
    Parameters:
    arr (numpy.ndarray): Input array to calculate the rolling average.
    window (int): The number of elements to consider for the moving average.
    
    Returns:
    numpy.ndarray: Array containing the rolling average values.
    """
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan  # Padding with NaN for elements where the window is not full
    cumsum = np.cumsum(arr)

    for i in range(window, n):
        result[i] = (cumsum[i] - cumsum[i - window]) / window

    return result

@njit(parallel=True)
def compute_rolling_averages(df_values, window_sizes):
    """
    Calculate the rolling averages for multiple window sizes in parallel.
    
    Parameters:
    df_values (numpy.ndarray): 2D array of values to calculate the rolling averages.
    window_sizes (List[int]): List of window sizes for the rolling averages.
    
    Returns:
    numpy.ndarray: A 3D array containing the rolling averages for each window size.
    """
    num_rows, num_features = df_values.shape
    num_windows = len(window_sizes)
    rolling_features = np.empty((num_rows, num_features, num_windows))

    for feature_idx in prange(num_features):
        for window_idx, window in enumerate(window_sizes):
            rolling_features[:, feature_idx, window_idx] = rolling_average(df_values[:, feature_idx], window)

    return rolling_features


# ## 📊 Feature Generation Functions 📊
# 
# 
# 
# 
# 

# **Explaination**
# 
# 
# 
# 1. `imbalance_features(df)`:
#    - This function takes a DataFrame `df` as input.
#    - It calculates various features related to price and size data using Pandas' `eval` function, creating new columns in the DataFrame for each feature.
#    - It then creates pairwise price imbalance features for combinations of price columns.
#    - Next, it calculates triplet imbalance features using the Numba-optimized function `calculate_triplet_imbalance_numba`.
#    - It calculates the rolling features.
#    - Finally, it calculates additional features, including momentum, spread, intensity, pressure, market urgency, and depth pressure.
#    - It also calculates statistical aggregation features (mean, standard deviation, skewness, kurtosis) for both price and size columns.
#    - Shifted, return, and diff features are generated for specific columns.
#    - Infinite values in the DataFrame are replaced with 0.
# 
# 2. `other_features(df)`:
#    - This function adds time-related and stock-related features to the DataFrame.
#    - It calculates the day of the week, seconds, and minutes from the "date_id" and "seconds_in_bucket" columns.
#    - It maps global features from a predefined dictionary to the DataFrame based on the "stock_id."
# 
# 3. `generate_all_features(df)`:
#    - This function combines the features generated by the `imbalance_features` and `other_features` functions.
#    - It selects the relevant columns for feature generation, applies the `imbalance_features` function, adds time and stock-related features using the `other_features` function, and then performs garbage collection to free up memory.
#    - The function returns a DataFrame containing the generated features, excluding certain columns like "row_id," "target," "time_id," and "date_id."
# 

# In[5]:


# 📊 Function to generate imbalance features
def imbalance_features(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # V1 features
    # Calculate various features using Pandas eval function
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    # Create features for pairwise price imbalances
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    # Calculate triplet imbalance features using the Numba-optimized function
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
        
    # V2 features
    # Calculate additional features
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    # V3 features
    # Calculate shifted and return features for specific columns
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # V4 features - Rolling averages
    window_sizes = [4, 8, 10]  # Define your desired window sizes
    for price in prices:
        rolling_avg_features = compute_rolling_averages(df[price].values.reshape(-1, 1), window_sizes)

        # Assigning the rolling average results to the DataFrame
        for i, window in enumerate(window_sizes):
            column_name = f"{price}_rolling_avg_{window}"
            df[column_name] = rolling_avg_features[:, 0, i]

    # Patching the start-of-day values after all rolling averages are calculated
    for window in window_sizes:
        for idx, seconds in enumerate(df['seconds_in_bucket'].values):
            if seconds / 10 <= window:
                for price in prices:
                    column_name = f"{price}_rolling_avg_{window}"
                    df.at[idx, column_name] = df.at[idx, price]
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)

    # Replace infinite values with 0
    return df.replace([np.inf, -np.inf], 0)

# 📅 Function to generate time and stock-related features
def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    df["minute"] = df["seconds_in_bucket"] // 60  # Minutes

    # Map global features to the DataFrame
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

# 🚀 Function to generate all features by combining imbalance and other features
def generate_all_features(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    # Generate imbalance features
    df = imbalance_features(df)
    
    # Generate time and stock-related features
    df = other_features(df)
    gc.collect()  # Perform garbage collection to free up memory
    
    # Select and return the generated features
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_name]


# In[6]:


weights = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04 , 0.002, 0.002, 0.004, 0.04 , 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02 , 0.004, 0.006, 0.002, 0.02 , 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04 , 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04 , 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02 , 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]

weights = {int(k):v for k,v in enumerate(weights)}


# ## Data Splitting

# **Explaination**
# 
# Checks whether it is running in offline or online mode and takes different actions accordingly. Here's what each part of the code does:
# 
# 1. `if is_offline:`:
#    - This condition checks if the variable `is_offline` is `True`. If it is `True`, it means the code is running in offline mode. 
# 
# 2. In the offline mode block:
#    - The code splits the dataset into two parts: `df_train` and `df_valid` based on the value of the `split_day`. Data with "date_id" less than or equal to the `split_day` is assigned to `df_train`, while data with "date_id" greater than the `split_day` is assigned to `df_valid`.
#    - It then displays a message indicating that the code is running in offline mode and provides the shapes (number of rows and columns) of the training and validation sets using the `print` statements.
# 
# 3. In the online mode block:
#    - If the code is not running in offline mode (i.e., `is_offline` is `False`), it means it's running in online mode.
#    - In online mode, the entire dataset is used for training, and the entire dataset is assigned to `df_train`.
#    - It displays a message indicating that the code is running in online mode using the `print` statement.
# 
# The purpose of distinguishing between offline and online modes is often related to the context in which the code is used. In offline mode, you typically have historical data and can perform tasks like data splitting for training and validation, while in online mode, you might be working with real-time data and use the entire dataset for training. The choice of mode can impact the preprocessing and analysis steps that follow in the code.

# In[7]:


# Check if the code is running in offline or online mode
if is_offline:
    # In offline mode, split the data into training and validation sets based on the split_day
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    
    # Display a message indicating offline mode and the shapes of the training and validation sets
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
else:
    # In online mode, use the entire dataset for training
    df_train = df
    
    # Display a message indicating online mode
    print("Online mode")


# **Explaination**
# 
# 
# 
# 1. `if is_train:`
#    - This condition checks if the variable `is_train` is `True`. If it is `True`, it means that the code is being executed in a training context.
# 
# 2. Inside the `if is_train:` block:
#    - A dictionary named `global_stock_id_feats` is created. This dictionary contains various statistical summary features calculated for each stock_id. These features include the median, standard deviation, and range of bid sizes and ask sizes, as well as bid prices and ask prices. These statistics are computed based on the training data (`df_train`) using Pandas' `groupby` and aggregation functions.
# 
# 3. The code checks if the execution mode is offline (`is_offline`) by further nested conditions.
#    - If it is offline (`is_offline` is `True`):
#      - It generates features for the training set (`df_train`) using the `generate_all_features` function.
#      - It prints a message indicating that the process of building the training features is finished.
#      - It generates features for the validation set (`df_valid`) using the `generate_all_features` function.
#      - It prints a message indicating that the process of building the validation features is finished.
#      - It reduces memory usage of the validation features using the `reduce_mem_usage` function.
# 
#    - If it is not in offline mode (i.e., online mode):
#      - It generates features for the training set (`df_train`) using the `generate_all_features` function.
#      - It prints a message indicating that the process of building online training features is finished.
# 
# 4. After generating features, it reduces memory usage of the training features (`df_train_feats`) using the `reduce_mem_usage` function. This is done to optimize memory consumption and improve performance.
# 
# The code's purpose is to prepare and optimize the feature set for training, considering whether it is in offline or online mode and whether it's part of the training process. The generated features and memory optimization are important steps in machine learning workflows, as they impact the training process and the model's efficiency.

# In[8]:


if is_train:
    global_stock_id_feats = {
        "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
        "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
        "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
        "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
    }


# In[9]:


if is_train:
    if is_offline:
        df_train_feats = generate_all_features(df_train)
        print("Build Train Feats Finished.")
        df_valid_feats = generate_all_features(df_valid)
        print("Build Valid Feats Finished.")
        df_valid_feats = reduce_mem_usage(df_valid_feats)
    else:
        df_train_feats = generate_all_features(df_train)
        print("Build Online Train Feats Finished.")

    df_train_feats = reduce_mem_usage(df_train_feats)


# ### Training and Evaluation with XGBoost
# 
# **Explanation**
# 
# The code block outlined below is tasked with training an XGBoost model, provided the `is_train` flag is set to `True`, indicating a training scenario. Should the offline mode be active, the block will also handle model inference and performance assessment.
# 
# 1. **Training Condition (`if is_train:`):**
#    - This conditional statement confirms whether the code execution is meant for model training by checking the `is_train` flag.
# 
# 2. **Feature Preparation:**
#    - The list of feature names used for the model is retrieved from the training feature set (`df_train_feats`), under the training condition.
# 
# 3. **XGBoost Parameters:**
#    - Parameters for the XGBoost model are defined, outlining crucial hyperparameters which dictate the model's performance, such as the objective function, booster type, number of estimators, max depth, subsampling, learning rate, and tree method for GPU acceleration, among others.
#    - The count of features being used is logged, ensuring transparency in the model's input dimensions.
# 
# 4. **Offline Data Split:**
#    - An offline data split is created using a mask (`offline_split`) to segregate the dataset based on a predefined date (`split_day - 45`). This split aids in constructing offline training and validation datasets.
#    - Separate DataFrames for training and validation sets, both features and targets, are instantiated for offline training.
# 
# 5. **Train XGBoost Model:**
#    - An XGBoost Regressor model (`xgb_model`) is initialized with the set parameters and trained using the offline training feature set and its corresponding target.
#    - Early stopping and evaluation metrics are set to monitor and halt the training when no improvement is observed in validation scores.
# 
# 6. **Memory Management:**
#    - Post-training, memory cleanup is initiated to free resources by removing variables related to training data and invoking garbage collection.
# 
# 7. **Inference Setup:**
#    - The full training dataset's target variable (`df_train_target`) is delineated, and the script proceeds to fine-tune the model for inference purposes.
#    - An inference model (`infer_model`) is cloned from the primary model with adjustments made to the number of estimators as per the early stopping round's outcome.
# 
# 8. **Offline Evaluation (when `is_offline` is `True`):**
#    - In offline mode, the inference model's performance is benchmarked on the validation data (`df_valid_feats`). Predictions are evaluated against actual values using mean absolute error (`mean_absolute_error`).
#    - The evaluation score is presented, offering insight into the model's accuracy in an offline setting.
# 
# This framework ensures a robust model training and evaluation process, emphasizing early stopping and estimator adjustment for optimal performance, complemented by effective resource management practices.
# 

# In[10]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import gc

# Define is_gpu_enabled
is_gpu_enabled = True

if is_train:
    feature_name = list(df_train_feats.columns)
    print(f"Feature length = {len(feature_name)}")

    # Split data for offline training
    offline_split = df_train['date_id'] > (split_day - 45)
    X_offline_train = df_train_feats[~offline_split]
    X_offline_valid = df_train_feats[offline_split]
    y_offline_train = df_train['target'][~offline_split]
    y_offline_valid = df_train['target'][offline_split]

    print("Valid Model Training.")

    # Initialize the XGBRegressor with desired hyperparameters
    model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist' if is_gpu_enabled else 'auto',
        max_depth=8,
        subsample=0.6,
        colsample_bytree=0.5,
        learning_rate=0.01,
        reg_lambda=1,
        reg_alpha=0.5,
        n_estimators=3500,
        random_state=42,
        verbosity=0
    )

    # Fit the model with early stopping
    model.fit(
        X_offline_train[feature_name], y_offline_train,
        eval_set=[(X_offline_train[feature_name], y_offline_train),
                  (X_offline_valid[feature_name], y_offline_valid)],
        eval_metric='mae',
        early_stopping_rounds=100,
        verbose=True
    )

    # Cleanup to free memory
    del X_offline_train, X_offline_valid, y_offline_train, y_offline_valid
    gc.collect()

    # Inference
    print("Infer Model Training.")
    X_train = df_train_feats[feature_name]
    y_train = df_train['target']

    # Retrain the model on the full dataset with the optimal number of trees
    optimal_n_estimators = model.best_iteration + 1 if model.best_iteration else 3000

    infer_model = XGBRegressor(
        objective='reg:squarederror',
        tree_method='gpu_hist' if is_gpu_enabled else 'auto',
        max_depth=8,
        subsample=0.6,
        colsample_bytree=0.5,
        learning_rate=0.01,
        reg_lambda=1,
        reg_alpha=0.5,
        n_estimators=optimal_n_estimators,
        random_state=42,
        verbosity=0
    )

    # Fit the model on the full training data
    infer_model.fit(X_train, y_train, verbose=True)

    if is_offline:
        # Offline predictions and evaluation
        X_valid = df_valid_feats[feature_name]
        y_valid = df_valid['target']
        offline_predictions = infer_model.predict(X_valid)
        offline_score = mean_absolute_error(offline_predictions, y_valid)
        print(f"Offline Score: {np.round(offline_score, 4)}")


# # Bringing some LightGBM models to the mix

# In[11]:


import os
import lightgbm as lgb

def imbalance_features_lgbm(df):
    # Define lists of price and size-related column names
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
   
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum', 'size_imbalance']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    return df.replace([np.inf, -np.inf], 0)

def other_features_lgbm(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features_lgbm(df):
    # Select relevant columns for feature generation
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    # Generate imbalance features
    df = imbalance_features_lgbm(df)
    df = other_features_lgbm(df)
    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    return df[feature_name]

model_save_path = '/kaggle/input/lightgbm-models/modelitos_para_despues'
num_folds = 5  # The number of folds you used during training

loaded_models = [infer_model]

# Load each model
for i in range(1, num_folds + 1):
    model_filename = os.path.join(model_save_path, f'doblez_{i}.txt')
    if os.path.exists(model_filename):
        loaded_model = lgb.Booster(model_file=model_filename)
        loaded_models.append(loaded_model)
        print(f"Model for fold {i} loaded from {model_filename}")
    else:
        print(f"Model file {model_filename} not found.")

# Load the final model
final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
if os.path.exists(final_model_filename):
    final_model = lgb.Booster(model_file=final_model_filename)
    loaded_models.append(final_model)
    print(f"Final model loaded from {final_model_filename}")
else:
    print(f"Final model file {final_model_filename} not found.")

# Now 'loaded_models' contains the models loaded from the files


# ### Inference and Submission with XGBoost
# 
# **Explanation**
# 
# The following code block is specifically tailored for generating predictions in the inference mode and subsequently submitting these predictions to the Optiver 2023 competition platform.
# 
# 1. **Zero-Sum Function (`def zero_sum(prices, volumes):`)**
#    - This function accepts two NumPy arrays: `prices` representing price predictions, and `volumes` representing trade volumes.
#    - It calculates the standard error (`std_error`) by taking the square root of `volumes`.
#    - It computes a scaling factor `step` by dividing the total of `prices` by the cumulative sum of `std_error`.
#    - The function adjusts the `prices` using the calculated `step` and `std_error`, producing an output array `out` that ensures a zero-sum constraint.
#    - The `out` array is then returned, providing the adjusted price predictions.
# 
# 2. **Inference Mode Check (`if is_infer:`)**
#    - This conditional block is triggered if the `is_infer` variable is set to `True`, signaling that the model is operating in inference mode.
# 
# 3. **Competition Environment Setup**
#    - The `optiver2023` module is imported to interact with the competition environment.
#    - An environment object (`env`) is instantiated for interfacing with the Optiver 2023 competition's data and submission system.
# 
# 4. **Variable Initialization**
#    - Key variables for processing the test data are initialized, including an iterator (`iter_test`), a count variable (`counter`), and bounds for the predictions (`y_min` and `y_max`).
# 
# 5. **Data Preparation**
#    - Lists to track queries per second (`qps`) and a DataFrame (`cache`) for holding test data are set up.
# 
# 6. **Test Data Iteration**
#    - The code enters a loop, processing the test data batches as provided by the competition environment.
# 
# 7. **Loop Operations**
#    - Current time (`now_time`) is logged for performance tracking.
#    - Incoming test data is appended to the `cache`, maintaining a rolling window of the most recent entries.
#    - The `generate_all_features` function is called to extract predictive features from the test data.
# 
# 8. **Prediction Generation**
#    - Predictions are made using the XGBoost model (`infer_model`) trained earlier.
# 
# 9. **Prediction Adjustment**
#    - The `zero_sum` function is applied to the predictions, ensuring compliance with the zero-sum constraint of the competition.
# 
# 10. **Prediction Clipping**
#    - Predictions are clipped to the predefined bounds (`y_min` and `y_max`) to meet competition rules.
# 
# 11. **Prediction Submission Preparation**
#    - The sample submission is updated with the clipped prediction values.
# 
# 12. **Prediction Submission**
#    - The updated predictions are submitted through the competition environment's `env.predict()` method.
# 
# 13. **Performance Monitoring**
#    - The `counter` is incremented, and processing speed (`qps`) is calculated for each batch of predictions.
# 
# 14. **Performance Reporting**
#    - Iteration count and average queries per second are reported periodically to monitor and adjust the inference process.
# 
# 15. **Completion Time Estimation**
#    - Upon completion of the test data processing, the average qps is used to estimate and report the total time taken for inference.
# 
# This code has been configured for the purpose of executing predictions within the Optiver 2023 trading challenge, utilizing an XGBoost model for inference. It integrates a specific transformation procedure (`zero_sum`) and enforces a prediction range prior to submitting these predictions to the competition's evaluation system.
# 

# In[12]:


from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import time

def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices) / np.sum(std_error)
    out = prices - std_error * step
    return out

# Inference
if is_infer:
    import optiver2023
    env = optiver2023.make_env()  # Setting up the environment for the competition
    iter_test = env.iter_test()   # Getting the iterator for the test set
    counter = 0                   # Initializing a counter
    y_min, y_max = -64, 64        # Setting prediction boundaries
    qps = []                      # Queries per second tracking
    cache = pd.DataFrame()        # Initializing a cache to store test data
    
    model_weights = [1/len(loaded_models)] * len(loaded_models)
    
    for (test_df, revealed_targets, sample_prediction_df) in iter_test:
        now_time = time.time()    # Current time for performance measurement
        print('counter:', counter)
        # Concatenating new test data with the cache, keeping only the last 21 observations per stock_id
        cache = pd.concat([cache, test_df], ignore_index=True, axis=0)
        if counter > 0:
            cache = cache.groupby('stock_id').tail(21).reset_index(drop=True)
        
        # Generate features for the current test DataFrame
        feat = generate_all_features(cache)[-len(test_df):]  # Assuming generate_all_features is a predefined function
        
        # Make predictions using the trained XGBRegressor model
        pred = loaded_models[0].predict(feat) * model_weights[0]
        
        feat = generate_all_features_lgbm(cache)[-len(test_df):]

        # Generate predictions for each model and calculate the weighted average
        for model, weight in zip(loaded_models[1:], model_weights[1:]):
            pred += weight * model.predict(feat)
        
        # Apply your zero-sum and clipping operations
        pred = zero_sum(pred, test_df['bid_size'] + test_df['ask_size'])
        clipped_predictions = np.clip(pred, y_min, y_max)
        
        # Set the predictions in the sample_prediction_df
        sample_prediction_df['target'] = clipped_predictions
        
        # Use the environment to make predictions
        env.predict(sample_prediction_df)
        
        counter += 1
        qps.append(time.time() - now_time)
        
        if counter % 10 == 0:
            print(f"{counter} queries per second: {np.mean(qps)}")

    time_cost = 1.146 * np.mean(qps)
    print(f"The code will take approximately {np.round(time_cost, 2)} hours to reason about")


# In[ ]:





# In[ ]:




