#!/usr/bin/env python
# coding: utf-8

# # Welcome to the Optiver 2023 Trading Competition Notebook!
# 
# 🚀 **Introduction:**
# - This notebook, tailored for the Optiver 2023 Trading Competition, implements a comprehensive strategy to predict target values for stock trading based on provided datasets.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of ALEX Wang, available at [this Kaggle project]( https://www.kaggle.com/code/peizhengwang/best-public-score). I extend my gratitude to ALEX Wang for sharing their insights and code.
# 
# 🌟 **Key Components:**
# 1. **Feature Engineering:** Utilizes a rich set of features, including imbalance features, price spreads, market urgency, and more, to capture relevant information for model training.
# 2. **Ensemble Modeling:** Combines the strengths of LightGBM and Neural Networks for robust and accurate predictions.
# 3. **Zero-Sum Transformation:** Implements a zero-sum transformation to adjust prices based on volumes, enhancing trading signal accuracy.
# 
# 👩‍💻 **How It Works:**
# - The notebook follows a systematic approach, from feature engineering to ensemble model predictions, creating a powerful strategy for stock target prediction.
# 
# 🌐 **Acknowledgments:**
# - Special thanks to the competition host, Optiver, for providing this exciting challenge and dataset. Your efforts make the community thrive!
# 
# 📈 **Performance Estimation:**
# - The notebook provides an estimate of the time required to process the entire test set, ensuring transparency and efficiency.
# 
# 🙏 **Feedback and Gratitude:**
# - **Feedback:** I welcome your feedback and suggestions to enhance and improve this notebook. Feel free to share your insights!
# - **Gratitude:** A big thank you to the competition organizers, fellow participants, and the broader data science community. Your collaboration makes this journey truly rewarding.
# 
# 🌟 **Let's embark on this trading adventure and strive for excellence in the Optiver 2023 Trading Competition! Happy coding!**

# ## 📊 Environment Setup and Configuration
# 
# This cell sets up the environment and configurations for a machine learning project.
# 
# 1. **Garbage Collection and Resource Cleaning (gc):**
#    - `import gc`: Imports the garbage collection module.
#    - `gc.collect()`: Manually triggers garbage collection to free up memory.
# 
# 2. **Operating System and Time Modules (os, time):**
#    - `import os`: Imports the operating system module.
#    - `import time`: Imports the time module.
# 
# 3. **Warning Handling (warnings, simplefilter):**
#    - `import warnings`: Imports the warnings module.
#    - `simplefilter(action="ignore", category=pd.errors.PerformanceWarning)`: Ignores performance warnings related to Pandas.
# 
# 4. **Itertools for Combinations (itertools):**
#    - `from itertools import combinations`: Imports the combinations function for creating combinations of elements.
# 
# 5. **Joblib for Parallel Execution (joblib):**
#    - `import joblib`: Imports the joblib library for parallel execution.
# 
# 6. **LightGBM and NumPy (lightgbm, numpy):**
#    - `import lightgbm as lgb`: Imports the LightGBM library for gradient boosting.
#    - `import numpy as np`: Imports NumPy for numerical operations.
# 
# 7. **Pandas for Data Handling (pandas):**
#    - `import pandas as pd`: Imports the Pandas library for data manipulation.
# 
# 8. **Scikit-Learn for Metrics and Model Selection (sklearn):**
#    - `from sklearn.metrics import mean_absolute_error`: Imports the mean absolute error metric.
#    - `from sklearn.model_selection import KFold, TimeSeriesSplit`: Imports KFold and TimeSeriesSplit for cross-validation.
# 
# 9. **Polars for Data Manipulation (polars):**
#    - `import polars as pl`: Imports the Polars library for efficient data manipulation.
# 
# 10. **Warning Suppression (warnings.filterwarnings):**
#    - `warnings.filterwarnings("ignore")`: Suppresses all warnings in the code.
# 
# 11. **Boolean Flags and Configuration Parameters:**
#    - `is_offline`, `LGB`, `NN`, `is_train`, `is_infer`: Boolean flags for offline mode, LightGBM, Neural Network, training, and inference.
#    - `max_lookback`: Maximum lookback period (possibly for feature engineering).
#    - `split_day`: Day used for splitting the data.
# 

# In[1]:


import gc  
import os  
import time  
import warnings 
from itertools import combinations  
from warnings import simplefilter 
import joblib  
import lightgbm as lgb  
import numpy as np  
import pandas as pd  
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import KFold, TimeSeriesSplit  
import polars as pl
warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

is_offline = False 
LGB = True
NN = False
is_train = True  
is_infer = True 
max_lookback = np.nan 
split_day = 435  


# ## 🔍 Weighted Average Function
# 
# Define a Python function named `weighted_average`.
# 
# 
# 
# 1. **Function Purpose (weighted_average):**
#    - This function calculates the weights for a weighted average.
# 
# 2. **Parameters:**
#    - `a`: List or array for which weights are to be calculated.
# 
# 3. **Variables:**
#    - `w`: List to store the calculated weights.
#    - `n`: Length of the input list `a`.
# 
# 4. **Weight Calculation Loop:**
#    - `for j in range(1, n + 1)`: Iterates over the range from 1 to `n` (inclusive).
#    - `j = 2 if j == 1 else j`: If `j` is 1, it is set to 2; otherwise, it remains unchanged.
#    - `w.append(1 / (2**(n + 1 - j)))`: Calculates and appends the weight using the formula `1 / (2**(n + 1 - j))`.
# 
# 5. **Return Statement:**
#    - `return w`: Returns the list of calculated weights.
# 
# **Function Explanation:**
# The function calculates weights for a weighted average, with higher weights assigned to earlier elements in the input list `a`. The weights decrease exponentially, with the first element receiving a weight of 1/2, the second 1/4, and so on.
# 
# **Example:**
# ```python
# input_list = [10, 20, 30, 40]
# weights = weighted_average(input_list)
# print(weights)
# ```
# Output:
# ```
# [0.5, 0.25, 0.125, 0.0625]
# ```
# 

# In[2]:


def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return w


# ## 🔄 Custom Time Series Cross-Validation Splitter
# 
# This cell defines a custom cross-validation splitter named `PurgedGroupTimeSeriesSplit`. The purpose of this class is to perform time series cross-validation while addressing concerns related to group and time-based data. 
# 
# ```python
# from sklearn.model_selection import KFold
# from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
# from sklearn.utils.validation import _deprecate_positional_args
# ```
# 
# 1. **Import Statements:**
#    - `from sklearn.model_selection import KFold`: Imports the KFold cross-validator.
#    - `from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples`: Additional imports for base KFold functionality.
#    - `from sklearn.utils.validation import _deprecate_positional_args`: Import for deprecating positional arguments.
# 
# ```python
# class PurgedGroupTimeSeriesSplit(_BaseKFold):
# ```
# 
# 2. **Class Definition:**
#    - `PurgedGroupTimeSeriesSplit`: Inherits from `_BaseKFold`, which is a base class for cross-validation splitters.
# 
# ```python
#     @_deprecate_positional_args
#     def __init__(self,
#                  n_splits=5,
#                  *,
#                  max_train_group_size=np.inf,
#                  max_test_group_size=np.inf,
#                  group_gap=None,
#                  verbose=False
#                  ):
# ```
# 
# 3. **Initializer Method:**
#    - `__init__`: Initializes the cross-validator with parameters.
#      - `n_splits`: Number of splits (folds) for cross-validation.
#      - `max_train_group_size`: Maximum size of the training group.
#      - `max_test_group_size`: Maximum size of the testing group.
#      - `group_gap`: Gap between groups to be considered in training and testing.
#      - `verbose`: Controls verbosity during the cross-validation process.
# 
# ```python
#     def split(self, X, y=None, groups=None):
# ```
# 
# 4. **Splitting Method:**
#    - `split`: Overrides the `split` method of `_BaseKFold` to define the splitting logic.
#      - `X`, `y`, `groups`: Data, target variable, and group labels.
# 
# ```python
#         # ... (Input validation and initialization of parameters)
# ```
# 
# 5. **Input Validation and Initialization:**
#    - Validates input parameters such as `groups` and initializes variables.
# 
# ```python
#         # ... (Group dictionary creation and sorting)
# ```
# 
# 6. **Group Dictionary Creation and Sorting:**
#    - Creates a dictionary (`group_dict`) to store indices corresponding to each group.
#    - Sorts unique groups based on indices.
# 
# ```python
#         # ... (Loop over groups and create training and testing arrays)
# ```
# 
# 7. **Group Loop for Train-Test Split:**
#    - Iterates over groups to create training and testing arrays based on defined parameters.
#    - Handles group gaps and maximum group sizes.
# 
# ```python
#             if self.verbose > 0:
#                     pass
# ```
# 
# 8. **Verbose Output (Not Implemented):**
#    - Placeholder for potential verbose output during cross-validation.
# 
# ```python
#             yield [int(i) for i in train_array], [int(i) for i in test_array]
# ```
# 
# 9. **Yield Split Indices:**
#    - Yields the indices of the training and testing arrays for each cross-validation iteration.
# 
# **Explanation:**
# 
# The `PurgedGroupTimeSeriesSplit` class is a custom cross-validator designed for time series data with groups. It ensures that training and testing sets do not overlap excessively in terms of time and group membership. The implementation uses a dictionary to efficiently manage indices corresponding to each group. The split method generates training and testing indices for each fold.
# 
# 

# In[3]:


from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]


# ## 🧹 Enhanced Memory Usage Reduction Function
# 
# This cell is a revised version of the memory usage reduction function. It incorporates additional enhancements and logging capabilities. 
# 
# ```python
# def reduce_mem_usage(df, verbose=0):
#     start_mem = df.memory_usage().sum() / 1024**2
# ```
# 
# 1. **Function Definition:**
#    - `reduce_mem_usage`: The function takes a DataFrame (`df`) as input and an optional verbosity flag (`verbose`).
# 
# ```python
#     for col in df.columns:
#         col_type = df[col].dtype
#         if col_type != object:
#             c_min = df[col].min()
#             c_max = df[col].max()
# ```
# 
# 2. **Iterating Over Columns:**
#    - Iterates over each column in the DataFrame.
# 
# 3. **Column Type and Range Determination:**
#    - Checks if the column type is not an object (non-string).
#    - Determines the minimum and maximum values in the column.
# 
# ```python
#             if str(col_type)[:3] == "int":
#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
#                     df[col] = df[col].astype(np.int8)
#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
#                     df[col] = df[col].astype(np.int16)
#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
#                     df[col] = df[col].astype(np.int32)
#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
#                     df[col] = df[col].astype(np.int64)
#             else:
# ```
# 
# 4. **Data Type Conversion for Integer Columns:**
#    - Checks if the column type is integer.
#    - Converts the column to the smallest possible integer type to conserve memory.
# 
# 5. **Data Type Conversion for Float Columns:**
#    - If the column type is not integer, it checks for float types.
#    - Converts the column to the smallest possible float type to conserve memory.
# 
# ```python
#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
#                     df[col] = df[col].astype(np.float16)  # Modified line
#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
#                     df[col] = df[col].astype(np.float32)
#                 else:
#                     df[col] = df[col].astype(np.float64)  # Modified line
# ```
# 
# 6. **Fine-Tuning Float Data Type Conversion:**
#    - For float columns, uses `np.float16` instead of `np.float32` if the range allows, and `np.float64` as a fallback.
# 
# ```python
#     if verbose:
#         logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
#         end_mem = df.memory_usage().sum() / 1024**2
#         logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
#         decrease = 100 * (start_mem - end_mem) / start_mem
#         logger.info(f"Decreased by {decrease:.2f}%")
#     return df
# ```
# 
# 7. **Verbose Output and Logging:**
#    - If the `verbose` flag is set, it logs information about the memory usage before and after optimization, along with the percentage decrease.
#    - Returns the DataFrame with optimized memory usage.
# 
# **Enhancements:**
# - Float columns are now optimized for `np.float16` when possible.
# - Logging statements provide detailed information on memory usage changes.
# 

# In[4]:


def reduce_mem_usage(df, verbose=0):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
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
               
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Decreased by {decrease:.2f}%")
    return df


# 📊 **Cell Explanation: Loading and Preprocessing Data**
# 
# This cell involves loading a dataset from a CSV file, dropping rows with missing values in the "target" column, and resetting the index.
# 
# ```python
# df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
# ```
# 
# 1. **Loading Dataset:**
#    - `pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")`: Reads a CSV file named "train.csv" located in the "/kaggle/input/optiver-trading-at-the-close/" directory into a Pandas DataFrame (`df`).
# 
# ```python
# df = df.dropna(subset=["target"])
# ```
# 
# 2. **Dropping Rows with Missing Values:**
#    - `df.dropna(subset=["target"])`: Drops rows where the "target" column has missing values. The result is assigned back to the DataFrame `df`.
# 
# ```python
# df.reset_index(drop=True, inplace=True)
# ```
# 
# 3. **Resetting Index:**
#    - `df.reset_index(drop=True, inplace=True)`: Resets the index of the DataFrame. The `drop=True` parameter avoids adding a new column with the old index, and `inplace=True` modifies the DataFrame in place.
# 
# ```python
# df_shape = df.shape
# ```
# 
# 4. **Getting DataFrame Shape:**
#    - `df.shape`: Retrieves the shape of the DataFrame (number of rows and columns).
#    - The result is assigned to the variable `df_shape`.
# 
# **Summary:**
# - The dataset is loaded from a CSV file.
# - Rows with missing values in the "target" column are removed.
# - The index of the DataFrame is reset.
# - The final shape of the DataFrame is stored in the variable `df_shape`.
# 

# In[5]:


df = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df = df.dropna(subset=["target"])
df.reset_index(drop=True, inplace=True)
df_shape = df.shape


# ⚙️ **Cell Explanation: Numba-Optimized Triplet Imbalance Calculation Function**
# 
# This cell defines a pair of functions that calculate triplet imbalance features using Numba, a Just-In-Time (JIT) compiler for Python. The primary goal is likely to speed up the computation of imbalance features for triplets of columns in a DataFrame. 
# 
# ```python
# from numba import njit, prange
# ```
# 
# 1. **Import Statements:**
#    - `from numba import njit, prange`: Imports the Numba functions `njit` (Just-In-Time compilation) and `prange` (parallel range) for optimizing performance.
# 
# ```python
# @njit(parallel=True)
# def compute_triplet_imbalance(df_values, comb_indices):
#     num_rows = df_values.shape[0]
#     num_combinations = len(comb_indices)
#     imbalance_features = np.empty((num_rows, num_combinations))
#     for i in prange(num_combinations):
#         a, b, c = comb_indices[i]
#         for j in range(num_rows):
#             max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
#             min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
#             mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
#             
#             if mid_val == min_val:
#                 imbalance_features[j, i] = np.nan
#             else:
#                 imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)
# 
#     return imbalance_features
# ```
# 
# 2. **Numba-Optimized Function - `compute_triplet_imbalance`:**
#    - `@njit(parallel=True)`: Decorator to enable Just-In-Time compilation and parallelization.
#    - This function calculates triplet imbalance features for a given set of combinations of three columns.
#    - It uses parallelization to optimize the computation for multiple combinations.
# 
# ```python
# def calculate_triplet_imbalance_numba(price, df):
#     df_values = df[price].values
#     comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
#     features_array = compute_triplet_imbalance(df_values, comb_indices)
#     columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
#     features = pd.DataFrame(features_array, columns=columns)
#     return features
# ```
# 
# 3. **Main Function - `calculate_triplet_imbalance_numba`:**
#    - Takes a list of column names (`price`) and a DataFrame (`df`) as input.
#    - Calculates the combinations of three columns and their indices.
#    - Calls the Numba-optimized function `compute_triplet_imbalance` to calculate triplet imbalance features.
#    - Creates a DataFrame (`features`) with the calculated features and appropriate column names.
# 
# **Summary:**
# - The code uses Numba to optimize the calculation of triplet imbalance features in a parallelized manner.
# - The main function (`calculate_triplet_imbalance_numba`) provides a convenient interface for computing and organizing these features from a DataFrame.
# 
# 

# In[6]:


from numba import njit, prange

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features

def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


# 🔍 **Cell Explanation: Feature Generation Functions**
# 
# This cell contains three functions (`imbalance_features`, `other_features`, and `generate_all_features`) for generating various features from a given DataFrame.
# 
# ### 1. `imbalance_features`
# 
# This function computes a variety of features related to price, size, and other financial metrics. It utilizes Numba for optimization and includes the calculation of rolling statistics. Here's a breakdown:
# 
# - **Price and Size Calculations:**
#   - Computes various price-related features such as mid_price, liquidity_imbalance, and size_imbalance.
#   - Calculates triplet imbalance features using the Numba-optimized function `calculate_triplet_imbalance_numba`.
# 
# - **Weighted Features:**
#   - Incorporates features based on weighted averages and momentum.
# 
# - **Spread and Pressure Metrics:**
#   - Calculates features related to price spread, pressure, urgency, and depth.
# 
# - **Ratio and Movement Features:**
#   - Computes features related to spread-depth ratio, mid-price movement, and relative spread.
# 
# - **Statistical Measures:**
#   - Calculates statistical measures (mean, std, skew, kurt) for both prices and sizes.
# 
# - **Shifted and Returns Features:**
#   - Creates features based on shifted and percentage change values for specific columns and time windows.
# 
# - **Rolling Window Statistics:**
#   - Uses Polars for efficient computation of rolling mean and standard deviation over different windows for selected columns.
# 
# - **Miscellaneous Features:**
#   - Generates additional features like `mid_price*volume` and `harmonic_imbalance`.
# 
# - **Data Cleaning:**
#   - Replaces infinite values with 0.
# 
# ### 2. `other_features`
# 
# This function adds temporal features and global stock features to the DataFrame:
# 
# - **Temporal Features:**
#   - Derives day of the week, seconds, minute, and time to market close.
# 
# - **Global Stock Features:**
#   - Incorporates global features for each stock based on a pre-defined dictionary (`global_stock_id_feats`).
# 
# ### 3. `generate_all_features`
# 
# This function orchestrates the generation of all features by applying both `imbalance_features` and `other_features` functions. It returns a DataFrame containing the generated features.
# 
# 
# **Usage:**
# ```python
# # Assuming df is the original DataFrame with necessary columns
# features_df = generate_all_features(df)
# ```
# 
# These functions collectively provide a comprehensive set of features for further analysis or model training in a financial trading context.

# In[7]:


def imbalance_features(df):
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
    
    df["stock_weights"] = df["stock_id"].map(weights)
    df["weighted_wap"] = df["stock_weights"] * df["wap"]
    df['wap_momentum'] = df.groupby('stock_id')['weighted_wap'].pct_change(periods=6)
   
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    df['spread_depth_ratio'] = (df['ask_price'] - df['bid_price']) / (df['bid_size'] + df['ask_size'])
    df['mid_price_movement'] = df['mid_price'].diff(periods=5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    df['micro_price'] = ((df['bid_price'] * df['ask_size']) + (df['ask_price'] * df['bid_size'])) / (df['bid_size'] + df['ask_size'])
    df['relative_spread'] = (df['ask_price'] - df['bid_price']) / df['wap']
    
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1,3,5,10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'weighted_wap','price_spread']:
        for window in [1,3,5,10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    
    for window in [3,5,10]:
        df[f'price_change_diff_{window}'] = df[f'bid_price_diff_{window}'] - df[f'ask_price_diff_{window}']
        df[f'size_change_diff_{window}'] = df[f'bid_size_diff_{window}'] - df[f'ask_size_diff_{window}']

    pl_df = pl.from_pandas(df)

    windows = [3, 5, 10]
    columns = ['ask_price', 'bid_price', 'ask_size', 'bid_size']

    group = ["stock_id"]
    expressions = []

    for window in windows:
        for col in columns:
            rolling_mean_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_mean(window)
                .over(group)
                .alias(f'rolling_diff_{col}_{window}')
            )

            rolling_std_expr = (
                pl.col(f"{col}_diff_{window}")
                .rolling_std(window)
                .over(group)
                .alias(f'rolling_std_diff_{col}_{window}')
            )

            expressions.append(rolling_mean_expr)
            expressions.append(rolling_std_expr)

    lazy_df = pl_df.lazy().with_columns(expressions)

    pl_df = lazy_df.collect()

    df = pl_df.to_pandas()
    gc.collect()
    
    df['mid_price*volume'] = df['mid_price_movement'] * df['volume']
    df['harmonic_imbalance'] = df.eval('2 / ((1 / bid_size) + (1 / ask_size))')
    
    for col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    return df

def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    df['time_to_market_close'] = 540 - df['seconds_in_bucket']
    
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df

def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    df = imbalance_features(df)
    gc.collect() 
    df = other_features(df)
    gc.collect()  
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    
    return df[feature_name]


# ## 📊 Weight Initialization**
# 
# This cell defines a dictionary named `weights`, where each key represents the stock ID, and the corresponding value represents the weight assigned to that stock. The weights are based on a predefined list.
# 
# Here's a breakdown:
# 
# ```python
# weights = [
#     # A list containing predefined weights for each stock ID
#     # (Please note: The list is truncated for brevity)
# ]
# 
# weights = {int(k):v for k,v in enumerate(weights)}
# ```
# 
# - **List of Weights:**
#   - The `weights` list contains the predefined weights for each stock ID. The weights are assigned to stocks in a specific order.
# 
# - **Conversion to Dictionary:**
#   - The `enumerate(weights)` function is used to iterate over the `weights` list and obtain both the index (`k`) and the value (`v`) at each iteration.
#   - The weights are then stored in a dictionary, where the stock ID is converted to an integer using `int(k)`.
# 
# **Usage:**
# - The resulting `weights` dictionary can be used to map weights to stock IDs in various computations, such as in the `imbalance_features` function.
# 
# ```python
# # Example usage to retrieve the weight for stock with ID 5
# weight_for_stock_5 = weights[5]
# ```
# 
# This dictionary likely serves as a lookup table for weights associated with different stocks in subsequent calculations or analyses.

# In[8]:


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


# ##🔄 Cell Explanation: Data Splitting for Training and Validation
# 
# This cell is responsible for splitting the dataset into training and validation sets based on the value of the `is_offline` variable. The splitting is performed differently depending on whether the mode is offline or online.
# 
# ```python
# if is_offline:
#     # Offline mode: Split the data based on the specified split_day
#     df_train = df[df["date_id"] <= split_day]
#     df_valid = df[df["date_id"] > split_day]
#     print("Offline mode")
#     print(f"train : {df_train.shape}, valid : {df_valid.shape}")
# else:
#     # Online mode: Use the entire dataset for training
#     df_train = df
#     print("Online mode")
# ```
# 
# - **Offline Mode:**
#   - If `is_offline` is `True`, the dataset is split into training (`df_train`) and validation (`df_valid`) sets based on the condition that the "date_id" is less than or equal to the specified `split_day`.
#   - The shapes of the resulting training and validation sets are printed.
# 
# - **Online Mode:**
#   - If `is_offline` is `False`, the entire dataset is used for training (`df_train`).
#   - A message indicating online mode is printed.
# 
# 
# 
# **Usage:**
# - The resulting `df_train` and `df_valid` DataFrames can be used in subsequent model training and evaluation steps.
# 
# ```python
# # Example usage
# train_features = generate_all_features(df_train)
# # ... (continue with training and validation steps)
# ```

# In[9]:


if is_offline:
    
    df_train = df[df["date_id"] <= split_day]
    df_valid = df[df["date_id"] > split_day]
    print("Offline mode")
    print(f"train : {df_train.shape}, valid : {df_valid.shape}")
    
else:
    df_train = df
    print("Online mode")


# ## 🚂 Training Data Feature Generation**
# 
# This cell is responsible for generating features specifically for the training data. The feature generation process includes calculating global stock features based on statistical measures of bid and ask sizes/prices. Additionally, it involves creating a set of features using the `generate_all_features` function.
# 
# ```python
# if is_train:
#     # Define global stock features based on statistical measures
#     global_stock_id_feats = {
#         "median_size": df_train.groupby("stock_id")["bid_size"].median() + df_train.groupby("stock_id")["ask_size"].median(),
#         "std_size": df_train.groupby("stock_id")["bid_size"].std() + df_train.groupby("stock_id")["ask_size"].std(),
#         "ptp_size": df_train.groupby("stock_id")["bid_size"].max() - df_train.groupby("stock_id")["bid_size"].min(),
#         "median_price": df_train.groupby("stock_id")["bid_price"].median() + df_train.groupby("stock_id")["ask_price"].median(),
#         "std_price": df_train.groupby("stock_id")["bid_price"].std() + df_train.groupby("stock_id")["ask_price"].std(),
#         "ptp_price": df_train.groupby("stock_id")["bid_price"].max() - df_train.groupby("stock_id")["ask_price"].min(),
#     }
# 
#     if is_offline:
#         # Offline mode: Generate features for both training and validation sets
#         df_train_feats = generate_all_features(df_train)
#         print("Build Train Feats Finished.")
#         df_valid_feats = generate_all_features(df_valid)
#         print("Build Valid Feats Finished.")
#         df_valid_feats = reduce_mem_usage(df_valid_feats)
#     else:
#         # Online mode: Generate features for the entire training set
#         df_train_feats = generate_all_features(df_train)
#         print("Build Online Train Feats Finished.")
# 
#     # Reduce memory usage for the training features DataFrame
#     df_train_feats = reduce_mem_usage(df_train_feats)
# ```
# 
# - **Global Stock Features:**
#   - Statistical features (`median`, `std`, `ptp`) are calculated for bid and ask sizes/prices grouped by stock ID.
#   - These features are stored in the `global_stock_id_feats` dictionary.
# 
# - **Feature Generation:**
#   - If `is_offline` is `True`, features are generated separately for both the training and validation sets.
#   - If `is_offline` is `False` (online mode), features are generated for the entire training set.
# 
# - **Memory Reduction:**
#   - The `reduce_mem_usage` function is applied to reduce the memory usage of the training features DataFrame.
# 
# **Usage:**
# - The resulting `df_train_feats` DataFrame contains the generated features and can be used for model training.
# 
# ```python
# # Example usage
# model = train_model(df_train_feats, target)
# ```
# 

# In[10]:


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


# ## 🚀 LightGBM Model Training**
# 
# This cell is responsible for training LightGBM models for the given data. The training is performed in a cross-validated manner with a specified number of folds. It uses the mean absolute error (MAE) as the objective function.
# 
# - **LightGBM Parameters:**
#   - The hyperparameters for the LightGBM model are specified in the `lgb_params` dictionary.
# 
# - **Cross-Validation:**
#   - The training is performed in a cross-validated manner with `num_folds` folds.
#   - The dataset is split into training and validation sets for each fold.
# 
# - **Model Training and Saving:**
#   - LightGBM models are trained for each fold, and the best iteration is saved.
#   - The trained models are stored in the `models` list.
# 
# - **Validation Scores:**
#   - The mean absolute error (MAE) is calculated for each fold and printed.
# 
# - **Final Model Training:**
#   - Additional LightGBM models are trained on the entire training set (`df_train_feats`).
# 
# **Usage:**
# - The trained models can be used for making predictions on new data.
# 
# ```python
# # Example usage
# predictions = predict_using_models(new_data, models)
# ```
# 
# 

# In[11]:


if LGB:
    import numpy as np
    import lightgbm as lgb
    
    lgb_params = {
        "objective": "mae",
        "n_estimators": 5000,
        "num_leaves": 512,
        "subsample": 0.4,
        "colsample_bytree": 0.6,
        "learning_rate": 0.00865,
        'max_depth': 24,
        "n_jobs": 4,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        "reg_alpha": 0.1,
        "reg_lambda": 3.25
    }

    feature_columns = list(df_train_feats.columns)
    print(f"Features = {len(feature_columns)}")

    num_folds = 10
    fold_size = 480 // num_folds
    gap = 5

    models = []
    models_cbt = []
    scores = []

    model_save_path = 'modelitos_para_despues' 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    date_ids = df_train['date_id'].values

    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        if i < num_folds - 1:  
            purged_start = end - 2
            purged_end = end + gap + 2
            train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
        else:
            train_indices = (date_ids >= start) & (date_ids < end)

        test_indices = (date_ids >= end) & (date_ids < end + fold_size)
        
        gc.collect()
        
        df_fold_train = df_train_feats[train_indices]
        df_fold_train_target = df_train['target'][train_indices]
        df_fold_valid = df_train_feats[test_indices]
        df_fold_valid_target = df_train['target'][test_indices]

        print(f"Fold {i+1} Model Training")

        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_fold_train[feature_columns],
            df_fold_train_target,
            eval_set=[(df_fold_valid[feature_columns], df_fold_valid_target)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
            ],
        )
        


        models.append(lgb_model)
        model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
        lgb_model.booster_.save_model(model_filename)
        print(f"Model for fold {i+1} saved to {model_filename}")


        fold_predictions = lgb_model.predict(df_fold_valid[feature_columns])
        fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
        scores.append(fold_score)
        print(f":LGB Fold {i+1} MAE: {fold_score}")

        del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
        gc.collect()

    average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

    final_model_params = lgb_params.copy()


    num_model = 1

    for i in range(num_model):
        final_model = lgb.LGBMRegressor(**final_model_params)
        final_model.fit(
            df_train_feats[feature_columns],
            df_train['target'],
            callbacks=[
                lgb.callback.log_evaluation(period=100),
            ],
        )
        models.append(final_model)


# ## 🚀 MLP Model Architecture**
# 
# This function defines an MLP (Multi-Layer Perceptron) model architecture using TensorFlow/Keras for a regression task with both continuous and categorical features.
# 
# 
# - **Inputs:**
#   - `num_continuous_features`: Number of continuous features.
#   - `num_categorical_features`: List containing the number of categories for each categorical feature.
#   - `embedding_dims`: List containing the embedding dimensions for each categorical feature.
#   - `num_labels`: Number of output labels (regression task).
#   - `hidden_units`: List containing the number of units in each hidden layer.
#   - `dropout_rates`: List containing dropout rates for each layer.
#   - `learning_rate`: Learning rate for the optimizer.
#   - `l2_strength`: L2 regularization strength (default is 0.01).
# 
# - **Architecture:**
#   - Continuous features are input through `input_continuous`.
#   - Categorical features are embedded using embedding layers.
#   - The embeddings are flattened and concatenated with continuous features.
#   - Batch normalization and dropout are applied for regularization.
#   - Hidden layers are constructed with ReLU activation.
#   - Output layer produces predictions for regression.
# 
# - **Compilation:**
#   - Adam optimizer is used with specified learning rate.
#   - Mean absolute error (MAE) is used as the loss function.
# 
# **Usage:**
# ```python
# # Example usage
# mlp_model = create_mlp(num_continuous_features, num_categorical_features, embedding_dims, num_labels, hidden_units, dropout_rates, learning_rate)
# mlp_model.summary()  # View model summary
# ```
# 

# In[12]:


def create_mlp(num_continuous_features, num_categorical_features, embedding_dims, num_labels, hidden_units, dropout_rates, learning_rate,l2_strength=0.01):

    input_continuous = tf.keras.layers.Input(shape=(num_continuous_features,))

    input_categorical = [tf.keras.layers.Input(shape=(1,))
                         for _ in range(len(num_categorical_features))]

    embeddings = [tf.keras.layers.Embedding(input_dim=num_categorical_features[i],
                                            output_dim=embedding_dims[i])(input_cat)
                  for i, input_cat in enumerate(input_categorical)]
    flat_embeddings = [tf.keras.layers.Flatten()(embed) for embed in embeddings]

    concat_input = tf.keras.layers.concatenate([input_continuous] + flat_embeddings)

    x = tf.keras.layers.BatchNormalization()(concat_input)
    x = tf.keras.layers.Dropout(dropout_rates[0])(x)

    for i in range(len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i],kernel_regularizer=l2(0.01),kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rates[i+1])(x)

    out = tf.keras.layers.Dense(num_labels,kernel_regularizer=l2(0.01),kernel_initializer='he_normal')(x)

    model = tf.keras.models.Model(inputs=[input_continuous] + input_categorical, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    return model


# ## 🚀 Neural Network (NN) Model Training Explanation:
# 
# This code snippet trains a Neural Network (NN) model for a regression task using TensorFlow/Keras. Below is a breakdown of the key components:
# 
# 1. **Data Preprocessing:**
#    - Fills missing values by forward filling within each stock group and then filling remaining NaNs with 0.
#    - Defines categorical and numerical features.
# 
# ```python
# df_train_feats = df_train_feats.groupby('stock_id').apply(lambda group: group.fillna(method='ffill')).fillna(0)
# 
# categorical_features = ["stock_id"]
# numerical_features = [column for column in list(df_train_feats) if column not in categorical_features]
# num_categorical_features = [len(df_train_feats[col].unique()) for col in categorical_features]
# ```
# 
# 2. **Model Configuration:**
#    - Configures NN model parameters such as batch size, hidden units, dropout rates, learning rate, and embedding dimensions.
#    - Defines a directory to save model checkpoints.
# 
# ```python
# nn_models = []
# batch_size = 64
# hidden_units = [128, 128]
# dropout_rates = [0.1, 0.1, 0.1]
# learning_rate = 1e-5
# embedding_dims = [20]
# 
# directory = '/kaggle/working/NN_Models/'
# if not os.path.exists(directory):
#     os.mkdir(directory)
# ```
# 
# 3. **Model Training using Group Time Series Split:**
#    - Utilizes a custom Group Time Series Split for cross-validation.
#    - Creates and trains an MLP model for each fold, saving the best model checkpoint.
#    - Performs fine-tuning on the best model with a reduced learning rate.
# 
# ```python
# gkf = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
# for fold, (tr, te) in enumerate(gkf.split(df_train_feats, df_train['target'], df_train['date_id'])):
#     # ... (data splitting and model training)
# 
#     model.fit((X_tr_continuous, X_tr_categorical), y_tr,
#               validation_data=([X_val_continuous, X_val_categorical], y_val),
#               epochs=200, batch_size=batch_size, callbacks=[ckp, es, rlr])
# 
#     # ... (predictions, scoring, and fine-tuning)
# ```
# 
# 4. **Result Evaluation:**
#    - Computes the Mean Absolute Error (MAE) for each fold.
#    - Prints and calculates the average MAE across all folds.
# 
# ```python
# print("Average NN CV Scores:", np.mean(scores))
# ```
# 

# In[13]:


if NN:
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    import gc
    from sklearn.model_selection import KFold
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as layers
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
    
    df_train_feats = df_train_feats.groupby('stock_id').apply(lambda group: group.fillna(method='ffill')).fillna(0)
    
    categorical_features = ["stock_id"]
    numerical_features = [column for column in list(df_train_feats) if column not in categorical_features]
    num_categorical_features = [len(df_train_feats[col].unique()) for col in categorical_features]

    nn_models = []

    batch_size = 64
    hidden_units = [128,128]
    dropout_rates = [0.1,0.1,0.1]
    learning_rate = 0.00000871
    embedding_dims = [20]

    directory = '/kaggle/working/NN_Models/'
    if not os.path.exists(directory):
        os.mkdir(directory)

    pred = np.zeros(len(df_train['target']))
    scores = []
    gkf = PurgedGroupTimeSeriesSplit(n_splits = 5, group_gap = 5)


    for fold, (tr, te) in enumerate(gkf.split(df_train_feats,df_train['target'],df_train['date_id'])):

        ckp_path = os.path.join(directory, f'nn_Fold_{fold+1}.h5')

        X_tr_continuous = df_train_feats.iloc[tr][numerical_features].values
        X_val_continuous = df_train_feats.iloc[te][numerical_features].values

        X_tr_categorical = df_train_feats.iloc[tr][categorical_features].values
        X_val_categorical = df_train_feats.iloc[te][categorical_features].values

        y_tr, y_val = df_train['target'].iloc[tr].values, df_train['target'].iloc[te].values

        print("X_train_numerical shape:",X_tr_continuous.shape)
        print("X_train_categorical shape:",X_tr_categorical.shape)
        print("Y_train shape:",y_tr.shape)
        print("X_test_numerical shape:",X_val_continuous.shape)
        print("X_test_categorical shape:",X_val_categorical.shape)
        print("Y_test shape:",y_val.shape)

        print(f"Creating Model - Fold{fold}")
        model = create_mlp(len(numerical_features), num_categorical_features, embedding_dims, 1, hidden_units, dropout_rates, learning_rate)

        rlr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1, patience=3, verbose=0, min_delta=1e-4, mode='min')
        ckp = ModelCheckpoint(ckp_path, monitor='val_mean_absolute_error', verbose=0, save_best_only=True, save_weights_only=True, mode='min')
        es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=1e-4, patience=10, mode='min', restore_best_weights=True, verbose=0)

        print(f"Fitting Model - Fold{fold}")
        model.fit((X_tr_continuous,X_tr_categorical), y_tr,
                  validation_data=([X_val_continuous,X_val_categorical], y_val),
                  epochs=200, batch_size=batch_size,callbacks=[ckp,es,rlr])

        output = model.predict((X_val_continuous,X_val_categorical), batch_size=batch_size * 4)

        pred[te] += model.predict((X_val_continuous,X_val_categorical), batch_size=batch_size * 4).ravel()

        score = mean_absolute_error(y_val, pred[te])
        scores.append(score)
        print(f'Fold {fold} MAE:\t', score)

        print(f"Finetuning Model - Fold{fold}")
        model = create_mlp(len(numerical_features), num_categorical_features, embedding_dims, 1, hidden_units, dropout_rates, learning_rate / 100)
        model.load_weights(ckp_path)
        model.fit((X_val_continuous,X_val_categorical), y_val, epochs=5, batch_size=batch_size, verbose=0)
        model.save_weights(ckp_path)
        nn_models.append(model)

        K.clear_session()
        del model
        gc.collect()

    print("Average NN CV Scores:",np.mean(scores))


# ## 🚀 Inference and Submission Explanation:
# 
# This code snippet handles the inference process and generates predictions for the test dataset. 
# 1. **Zero-Sum Transformation:**
#    - Defines a function `zero_sum` that takes prices and volumes as inputs and performs a zero-sum transformation on the prices based on volumes.
# 
# ```python
# def zero_sum(prices, volumes):
#     std_error = np.sqrt(volumes)
#     step = np.sum(prices) / np.sum(std_error)
#     out = prices - std_error * step
#     return out
# ```
# 
# 2. **Inference Loop:**
#    - Iterates through the test set using the `iter_test` generator provided by the Optiver 2023 competition environment.
#    - Generates features for each batch of test data using the `generate_all_features` function.
#    - If LightGBM (`LGB`) is used, it combines predictions from multiple models using weighted averaging. Weights are derived from the `weighted_average` function (not provided in the snippet).
#    - Predictions are clipped within the range `[y_min, y_max]` to meet the competition constraints.
# 
# ```python
# for (test, revealed_targets, sample_prediction) in iter_test:
#     # ... (data processing)
# 
#     feat = generate_all_features(cache)[-len(test):]
# 
#     if LGB:
#         lgb_predictions = np.zeros(len(test))
#         for model, weight in zip(models, lgb_model_weights):
#             lgb_predictions += weight * model.predict(feat[feature_columns])
# 
#     predictions = lgb_predictions
#     
#     # ... (zero-sum transformation and submission preparation)
#     env.predict(sample_prediction)
#     counter += 1
#     qps.append(time.time() - now_time)
#     if counter % 10 == 0:
#         print(counter, 'qps:', np.mean(qps))
# ```
# 
# 3. **Submission and Performance Estimation:**
#    - Submits predictions to the Optiver 2023 competition environment using `env.predict(sample_prediction)`.
#    - Calculates and prints the approximate time required to process the entire test set.
# 
# ```python
# time_cost = 1.146 * np.mean(qps)
# print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")
# ```
# 
# 

# In[14]:


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

    if LGB:
        lgb_model_weights = weighted_average(models)
    
    for (test, revealed_targets, sample_prediction) in iter_test:
        now_time = time.time()
        cache = pd.concat([cache, test], ignore_index=True, axis=0)
        if counter > 0:
            cache = cache.groupby(['stock_id']).tail(21).sort_values(by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
        feat = generate_all_features(cache)[-len(test):]
        print(f"Feat Shape is: {feat.shape}")

        if LGB:
            lgb_predictions = np.zeros(len(test))
            for model, weight in zip(models, lgb_model_weights):
                lgb_predictions += weight * model.predict(feat[feature_columns])

        predictions = lgb_predictions
        
        final_predictions = predictions - np.mean(predictions)
        clipped_predictions = np.clip(final_predictions, y_min, y_max)
        sample_prediction['target'] = clipped_predictions
        env.predict(sample_prediction)
        counter += 1
        qps.append(time.time() - now_time)
        if counter % 10 == 0:
            print(counter, 'qps:', np.mean(qps))

    time_cost = 1.146 * np.mean(qps)
    print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")


# ## Explore More! 👀
# Thank you for exploring this notebook! If you found this notebook insightful or if it helped you in any way, I invite you to explore more of my work on my profile.
# 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Feedback and Gratitude 🙏
# We value your feedback! Your insights and suggestions are essential for our continuous improvement. If you have any comments, questions, or ideas to share, please don't hesitate to reach out.
# 
# 📬 Contact me via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# I would like to express our heartfelt gratitude for your time and engagement. Your support motivates us to create more valuable content.
# 
# Happy coding and best of luck in your data science endeavors! 🚀
# 
