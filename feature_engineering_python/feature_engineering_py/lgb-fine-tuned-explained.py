#!/usr/bin/env python
# coding: utf-8

# ##  singel model Advance perameters 🚀
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Optiver - Trading at the Close This notebook is designed to assist you in the competition, Predict US stocks closing movements
# 
# 
# ### Inspiration and Credits 🙌
# This notebook draws inspiration from the remarkable work of Angle, which can be found in [this Kaggle project](https://www.kaggle.com/code/lblhandsome/optiver-robust-best-single-model/notebook). Special thanks to Angle for sharing valuable insights and code.
# 
# 🌟 Dive into my profile and explore other public projects. Don't forget to share your feedback and experiences!
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
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
# 
# ## Acknowledgments 🙏
# We extend our gratitude to the Optiver organizers for providing the dataset and hosting the competition.
# 
# Let's embark on this journey! Don't hesitate to reach out if you have questions or require assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈

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
import lightgbm as lgb  # LightGBM gradient boosting framework
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

# **Explaination**
# 
# 
# This code includes functions for calculating triplet imbalance in a parallel and optimized manner using the Numba library. Let's break down each part of the code:
# 
# 1. `from numba import njit, prange`
#    - This line imports two important features from the Numba library: `njit` for Just-In-Time (JIT) compilation and `prange` for parallel processing. JIT compilation can significantly speed up the execution of code, and parallel processing allows for concurrent execution of code in a loop.
# 
# 2. `@njit(parallel=True)`
#    - This is a decorator applied to the `compute_triplet_imbalance` function, indicating that Numba should compile this function for speed optimization and parallel execution. This decorator makes use of Numba's features to enhance the performance of the code.
# 
# 3. `def compute_triplet_imbalance(df_values, comb_indices):`
#    - This function is designed to calculate triplet imbalance in a parallelized manner using Numba. It takes two parameters:
#      - `df_values`: A NumPy array containing the values of the DataFrame. It represents the price data.
#      - `comb_indices`: A list of combinations of three price indices (a, b, c) for which triplet imbalance needs to be computed.
# 
# 4. `num_rows = df_values.shape[0]`
#    - This line calculates the number of rows in the `df_values` array, which represents the number of rows in the DataFrame.
# 
# 5. `imbalance_features = np.empty((num_rows, num_combinations))`
#    - This line initializes an empty NumPy array `imbalance_features` with dimensions (number of rows, number of combinations). This array will store the computed triplet imbalance values.
# 
# 6. The code then enters a loop that iterates through all combinations of triplets specified by `comb_indices`.
# 
# 7. `for i in prange(num_combinations):`
#    - This loop is parallelized using `prange`, which allows for multiple combinations to be processed concurrently.
# 
# 8. Inside the loop, it extracts the indices (a, b, c) for the current combination.
# 
# 9. Another loop iterates through the rows of the DataFrame (`for j in range(num_rows)`) and calculates the triplet imbalance for each row.
# 
# 10. `max_val`, `min_val`, and `mid_val` are computed for each row.
# 
# 11. `if mid_val == min_val:` checks if division by zero would occur and sets the corresponding entry in `imbalance_features` to `np.nan` to prevent errors in such cases.
# 
# 12. The final imbalance value is calculated using the formula `(max_val - mid_val) / (mid_val - min_val)` and stored in the `imbalance_features` array.
# 
# 13. The function returns the `imbalance_features` array, which contains the computed triplet imbalance values for all combinations and rows.
# 
# 14. `calculate_triplet_imbalance_numba` is another function that takes a price column name and a DataFrame as input. It prepares the data and calculates triplet imbalance using the `compute_triplet_imbalance` function. It returns the result as a DataFrame with appropriately labeled columns.
# 

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


# **Exaplaination**
# 
# This code block is responsible for training a machine learning model (LightGBM) in a context where `is_train` is `True`. It also performs inference and evaluates the model if it's in offline mode. 
# 
# 1. `if is_train:`
#    - This condition checks if the variable `is_train` is `True`. If it is `True`, it means that the code is being executed in a training context.
# 
# 2. Inside the `if is_train:` block:
#    - It gets the list of feature names from the training features (`df_train_feats`).
# 
# 3. LightGBM Parameters:
#    - It defines LightGBM parameters for the model, specifying various hyperparameters such as the objective function, number of estimators, number of leaves, subsample ratio, learning rate, number of CPU cores to use, GPU acceleration, and others.
#    - It prints the length of the feature names to check the number of features used in the model.
# 
# 4. Offline Split:
#    - It creates a mask (`offline_split`) to split the training data into two sets based on a specific date (in this case, `(split_day - 45)`). Data with a "date_id" greater than this date is considered for offline validation, and data with a "date_id" less than or equal to this date is considered for offline training.
#    - It creates separate DataFrames for offline training (`df_offline_train` and `df_offline_train_target`) and offline validation (`df_offline_valid` and `df_offline_valid_target`).
#    - It prints a message indicating that offline model training is taking place.
# 
# 5. Train LightGBM Model:
#    - It creates a LightGBM Regressor model (`lgb_model`) with the specified parameters and fits it to the offline training data.
#    - It sets up early stopping and evaluation callbacks.
#    - This model is trained on the offline data.
# 
# 6. Memory Cleanup:
#    - It frees up memory by deleting variables related to offline training and performing garbage collection.
# 
# 7. Inference:
#    - It defines the target variable for the entire training dataset (`df_train_target`) and prints a message indicating that inference model training is taking place.
#    - It creates an inference model (`infer_lgb_model`) that is a copy of the initial model but with the number of estimators adjusted based on the best iteration from early stopping.
# 
# 8. If in offline mode (`is_offline` is `True`):
#    - It performs offline predictions using the inference model on the validation set (`df_valid_feats`) and evaluates the predictions using the mean absolute error (`mean_absolute_error`) against the true target values (`df_valid_target`).
#    - It prints the offline score as a measure of model performance on the validation set.
# 
# This code is designed to train and evaluate a machine learning model for offline data, and the training strategy includes early stopping and adjusting the number of estimators during inference to optimize model performance. It also includes memory management steps to improve the efficiency of the training process.

# In[9]:


import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import gc

# Assuming df_train_feats and df_train are already defined and df_train contains the 'date_id' column

# Set up parameters for LightGBM

lgb_params = {
        "objective": "mae",
        "n_estimators": 5500,
        "num_leaves": 256,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "learning_rate": 0.00877,
        "n_jobs": 4,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        "max_depth": 12,  # Maximum depth of the tree
        "min_child_samples": 15,  # Minimum number of data points in a leaf
        "reg_alpha": 0.1,  # L1 regularization term
        "reg_lambda": 0.3,  # L2 regularization term
        "min_split_gain": 0.2,  # Minimum loss reduction required for further partitioning
        "min_child_weight": 0.001,  # Minimum sum of instance weight (hessian) in a leaf
        "bagging_fraction": 0.9,  # Fraction of data to be used for training each tree
        "bagging_freq": 5,  # Frequency for bagging
        "feature_fraction": 0.9,  # Fraction of features to be used for training each tree
        "num_threads": 4,  # Number of threads for LightGBM to use
}
feature_name = list(df_train_feats.columns)
print(f"Feature length = {len(feature_name)}")

# The total number of date_ids is 480, we split them into 5 folds with a gap of 5 days in between
num_folds = 5
fold_size = 480 // num_folds
gap = 5

models = []
scores = []

model_save_path = 'modelitos_para_despues'  # Directory to save models
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# We need to use the date_id from df_train to split the data
date_ids = df_train['date_id'].values

for i in range(num_folds):
    start = i * fold_size
    end = start + fold_size
    
    # Define the purged set ranges
    purged_before_start = start - 2
    purged_before_end = start + 2
    purged_after_start = end - 2
    purged_after_end = end + 2
    
    # Exclude the purged ranges from the test set
    purged_set = ((date_ids >= purged_before_start) & (date_ids <= purged_before_end)) | \
                 ((date_ids >= purged_after_start) & (date_ids <= purged_after_end))
    
    # Define test_indices excluding the purged set
    test_indices = (date_ids >= start) & (date_ids < end) & ~purged_set
    train_indices = ~test_indices & ~purged_set
    
    df_fold_train = df_train_feats[train_indices]
    df_fold_train_target = df_train['target'][train_indices]
    df_fold_valid = df_train_feats[test_indices]
    df_fold_valid_target = df_train['target'][test_indices]

    print(f"Fold {i+1} Model Training")
    
    # Train a LightGBM model for the current fold
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        df_fold_train[feature_name],
        df_fold_train_target,
        eval_set=[(df_fold_valid[feature_name], df_fold_valid_target)],
        callbacks=[
            lgb.callback.early_stopping(stopping_rounds=100),
            lgb.callback.log_evaluation(period=100),
        ],
    )

    # Append the model to the list
    models.append(lgb_model)
    # Save the model to a file
    model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
    lgb_model.booster_.save_model(model_filename)
    print(f"Model for fold {i+1} saved to {model_filename}")

    # Evaluate model performance on the validation set
    fold_predictions = lgb_model.predict(df_fold_valid[feature_name])
    fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
    scores.append(fold_score)
    print(f"Fold {i+1} MAE: {fold_score}")

    # Free up memory by deleting fold specific variables
    del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
    gc.collect()

# Calculate the average best iteration from all regular folds
average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

# Update the lgb_params with the average best iteration
final_model_params = lgb_params.copy()
final_model_params['n_estimators'] = average_best_iteration

print(f"Training final model with average best iteration: {average_best_iteration}")

# Train the final model on the entire dataset
final_model = lgb.LGBMRegressor(**final_model_params)
final_model.fit(
    df_train_feats[feature_name],
    df_train['target'],
    callbacks=[
        lgb.callback.log_evaluation(period=100),
    ],
)

# Append the final model to the list of models
models.append(final_model)

# Save the final model to a file
final_model_filename = os.path.join(model_save_path, 'doblez-conjunto.txt')
final_model.booster_.save_model(final_model_filename)
print(f"Final model saved to {final_model_filename}")

# Now 'models' holds the trained models for each fold and 'scores' holds the validation scores
print(f"Average MAE across all folds: {np.mean(scores)}")


# **Explaination**
# 
# This code block is responsible for making predictions in inference mode and submitting them to the Optiver 2023 competition environment. 
# 
# 1. `def zero_sum(prices, volumes):`
#    - This function takes two NumPy arrays, `prices` and `volumes`, as input.
#    - It calculates the standard error as the square root of the `volumes`.
#    - It calculates a variable `step` by dividing the sum of `prices` by the sum of the standard errors (`std_error`).
#    - It calculates the `out` variable as the difference between the `prices` and the product of the `std_error` and the `step`.
#    - The function returns the `out` variable.
# 
# 2. `if is_infer:`
#    - This condition checks if the variable `is_infer` is `True`. If it is `True`, it means that the code is being executed in inference mode.
# 
# 3. Inside the `if is_infer:` block:
#    - The code imports the `optiver2023` module and creates an environment (`env`) for the Optiver 2023 competition.
# 
# 4. It initializes variables, including an iterator (`iter_test`), a counter (`counter`), and variables for specifying the lower and upper limits of predictions (`y_min` and `y_max`).
# 
# 5. It also initializes lists for recording queries per second (`qps`) and a DataFrame (`cache`) to store test data.
# 
# 6. The code enters a loop that iterates through the test data provided by the environment.
# 
# 7. Inside the loop:
#    - It records the current time (`now_time`) using the `time.time()` function.
#    - It concatenates the current test data with the existing cache of data.
#    - It keeps only the most recent 21 rows for each stock and sorts them.
#    - It generates features for the current test data using the `generate_all_features` function.
# 
# 8. It makes predictions using the previously trained inference LightGBM model (`infer_lgb_model`).
# 
# 9. It applies the `zero_sum` function to transform the predictions.
# 
# 10. It clips the transformed predictions to ensure they fall within the specified range defined by `y_min` and `y_max`.
# 
# 11. It updates the sample prediction with the clipped values.
# 
# 12. It submits the predictions to the environment using the `env.predict()` method.
# 
# 13. It updates the counter and records the time spent on each iteration in the `qps` list.
# 
# 14. It prints the current iteration number and the average queries per second (qps) if the counter is a multiple of 10.
# 
# 15. After processing all test data, it calculates the estimated time cost based on the average qps and prints the estimated time to reason about.
# 
# This code is designed for making predictions in an Optiver trading competition environment and uses a trained LightGBM model for inference. It also includes a transformation step (`zero_sum`) and clipping of predictions to ensure they are within a specified range before submitting them to the competition environment.

# In[10]:


def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices) / np.sum(std_error)
    out = prices - std_error * step
    return out

if is_infer:
    import optiver2023
    env = optiver2023.make_env()
    iter_test = env.iter_test()
    counter = 0
    y_min, y_max = -64, 64
    qps, predictions = [], []
    cache = pd.DataFrame()

    # Weights for each fold model
    model_weights = [1/len(models)] * len(models) 
    
    for (test, revealed_targets, sample_prediction) in iter_test:
        now_time = time.time()
        cache = pd.concat([cache, test], ignore_index=True, axis=0)
        if counter > 0:
            cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
        feat = generate_all_features(cache)[-len(test):]

        # Generate predictions for each model and calculate the weighted average
        lgb_predictions = np.zeros(len(test))
        for model, weight in zip(models, model_weights):
            lgb_predictions += weight * model.predict(feat)

        lgb_predictions = zero_sum(lgb_predictions, test['bid_size'] + test['ask_size'])
        clipped_predictions = np.clip(lgb_predictions, y_min, y_max)
        sample_prediction['target'] = clipped_predictions
        env.predict(sample_prediction)
        counter += 1
        qps.append(time.time() - now_time)
        if counter % 10 == 0:
            print(counter, 'qps:', np.mean(qps))

    time_cost = 1.146 * np.mean(qps)
    print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")


# ## Keep Exploring! 👀
# 
# If you enjoyed exploring this notebook and found it insightful, I encourage you to delve further into my portfolio.
# 
# 👉 [Explore My Portfolio](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Share Your Feedback and Gratitude 🙏
# 
# Your feedback is invaluable to us! We welcome your insights, suggestions, and questions as they fuel our continuous growth. If you have any comments or ideas to share, please feel free to get in touch with us.
# 
# 📬 Contact us via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# We extend our heartfelt gratitude for your time and engagement. Your support inspires us to generate more valuable content.
# 
# Wishing you a rewarding journey in the realm of data science and coding! 🚀
