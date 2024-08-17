#!/usr/bin/env python
# coding: utf-8

# # Explained Baseline Solution 💨
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Google - Fast or Slow? Predict AI Model Runtime! This notebook is designed to help you participate in the competition and to Detect sleep onset and wake from wrist-worn accelerometer data.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of Bhukya Satheesh
# , available at [this Kaggle project](https://www.kaggle.com/code/satheeshbhukya1/google-fast-or-slow/notebook). I extend my gratitude to Bhukya Satheesh
#  for sharing their insights and code.
# 
# 🌟 Explore my profile and other public projects, and don't forget to share your feedback! 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍
# 
# ## Purpose 🎯
# The primary purpose of this notebook is to:
# - Load and preprocess the competition data 📁
# - Engineer relevant features for model training 🏋️‍♂️
# - Train predictive models to make target variable predictions 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook is structured as follows:
# 1. **Data Preparation**: In this section, we load and preprocess the competition data.
# 2. **Feature Engineering**: We generate and select relevant features for model training.
# 3. **Model Training**: We train machine learning models on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 
# 
# ## How to Use 🛠️
# To use this notebook effectively, please follow these steps:
# 1. Ensure you have the competition data and environment set up.
# 2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
# 
# **Note**: Make sure to replace any placeholder paths or configurations with your specific information.
# 
# ## Acknowledgments 🙏
# We acknowledge theChild Mind Institute organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈

# ## 📚 Importing necessary libraries 📊
# 

# In[1]:


# 📚 Importing necessary libraries 📊
import numpy as np              # NumPy for numerical operations
import pandas as pd             # Pandas for data manipulation
import plotly.express as px     # Plotly Express for interactive plotting
import matplotlib.pyplot as plt # Matplotlib for basic plotting
import seaborn as sns           # Seaborn for statistical data visualization
import random                   # Random for generating random numbers
import os                       # OS for interacting with the operating system
import gc                       # Garbage collector for memory management
from copy import deepcopy      # Deepcopy for creating deep copies of objects
from functools import partial  # Partial function application for function manipulation
from itertools import combinations  # Combinations for creating combinations of elements
from itertools import groupby  # Groupby for grouping elements in an iterable
from tqdm import tqdm          # tqdm for progress bars
import polars as pl            # Polars for data manipulation
import datetime                # Datetime for date and time operations

# 🧱 Importing specific functions and classes 🧱
from sklearn.model_selection import train_test_split  # Splitting data into training and testing sets
from sklearn.model_selection import StratifiedKFold, KFold  # Cross-validation techniques
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss  # Evaluation metrics
from sklearn.model_selection import cross_validate  # Cross-validation scoring
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, precision_score, average_precision_score  # Metrics and displays
import optuna  # Library for hyperparameter tuning
import xgboost as xgb  # XGBoost for gradient boosting
import lightgbm as lgb  # LightGBM for gradient boosting
from sklearn.linear_model import LogisticRegression  # Logistic Regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Random Forest and Gradient Boosting
from sklearn.pipeline import Pipeline  # Pipeline for building a sequence of data transformations
from catboost import Pool  # CatBoost for gradient boosting

# ⚙️ Importing a custom metric function ⚙️
from metric import score  # Importing a custom event detection AP score function

# 📋 Define column names and tolerances for the score function 📋
column_names = {
    'series_id_column_name': 'series_id',
    'time_column_name': 'step',
    'event_column_name': 'event',
    'score_column_name': 'score',
}

tolerances = {
    'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360], 
    'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
}

# 📊 Setting display options for Pandas DataFrames 📊
pd.set_option('display.max_columns', None)  # Show all columns

# 🚫 Suppressing warnings 🚫
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ## 📂 Importing and transforming data 🔄
# 

# **Explaination**
# 
# This code will do data transformations and loading data from input directory.
# 
# 1. `dt_transforms = [...]`: This defines a list called `dt_transforms` that will contain a series of transformations to be applied to a timestamp column in the data.
# 
# 2. `pl.col('timestamp').str.to_datetime()`: This line converts a column named 'timestamp' to datetime format using the `str.to_datetime()` method.
# 
# 3. `(pl.col('timestamp').str.to_datetime().dt.year() - 2000).cast(pl.UInt8).alias('year')`: This line extracts the year from the 'timestamp' column, subtracts 2000 from it, casts the result as an 8-bit unsigned integer, and assigns it an alias 'year'.
# 
# 4. `pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month')`: Similar to line 3, this line extracts the month from the 'timestamp' column, casts it as an 8-bit unsigned integer, and assigns it an alias 'month'.
# 
# 5. `pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day')`: This line extracts the day from the 'timestamp' column, casts it as an 8-bit unsigned integer, and assigns it an alias 'day'.
# 
# 6. `pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')`: Similar to lines 3 and 4, this line extracts the hour from the 'timestamp' column, casts it as an 8-bit unsigned integer, and assigns it an alias 'hour'.
# 
# 7. `data_transforms = [...]`: This defines another list called `data_transforms` that will contain transformations for columns other than the timestamp.
# 
# 8. `pl.col('anglez').cast(pl.Int16)`: This line casts the column 'anglez' to a 16-bit signed integer.
# 
# 9. `(pl.col('enmo') * 1000).cast(pl.UInt16)`: This line multiplies the 'enmo' column by 1000 and casts the result as a 16-bit unsigned integer.
# 
# 10. `train_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(dt_transforms + data_transforms)`: This line reads a parquet file ('train_series.parquet'), applies the transformations defined in `dt_transforms` and `data_transforms`, and assigns the result to the `train_series` variable.
# 
# 11. `train_events = pl.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(dt_transforms)`: This line reads a CSV file ('train_events.csv'), applies the transformations defined in `dt_transforms`, and assigns the result to the `train_events` variable.
# 
# 12. `test_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(dt_transforms + data_transforms)`: Similar to line 10, this line reads a parquet file ('test_series.parquet'), applies the transformations defined in `dt_transforms` and `data_transforms`, and assigns the result to the `test_series` variable.
# 
# 13. `series_ids = train_events['series_id'].unique(maintain_order=True).to_list()`: This line extracts unique 'series_id' values from the `train_events` DataFrame while maintaining the original order and converts them to a Python list.
# 
# 14. `onset_counts = ...`: These lines calculate the counts of 'onset' events and 'wakeup' events for each 'series_id' and store them in `onset_counts` and `wakeup_counts` DataFrames.
# 
# 15. `counts = pl.DataFrame(...)`: This line creates a DataFrame called `counts` containing 'series_id', 'onset_counts', and 'wakeup_counts' columns by combining the results of the previous step.
# 
# 16. `count_mismatches = counts.filter(...)`: This line filters `counts` to find series where the 'onset_counts' do not match 'wakeup_counts'.
# 
# 17. `train_series = train_series.filter(...)` and `train_events = train_events.filter(...)`: These lines filter the `train_series` and `train_events` DataFrames to exclude series with count mismatches.
# 
# 18. `series_ids = train_events.drop_nulls()['series_id'].unique(maintain_order=True).to_list()`: This line updates the `series_ids` list by removing rows with null values in 'series_id' and extracting unique values while maintaining the original order.
# 
# This code loads and transforms data from different sources, applies various data transformations, and filters out series with count mismatches in 'onset' and 'wakeup' events.

# In[2]:


# Column transformations for timestamp
dt_transforms = [
    pl.col('timestamp').str.to_datetime(),  # Convert timestamp to datetime
    (pl.col('timestamp').str.to_datetime().dt.year() - 2000).cast(pl.UInt8).alias('year'),  # Extract and cast year
    pl.col('timestamp').str.to_datetime().dt.month().cast(pl.UInt8).alias('month'),  # Extract and cast month
    pl.col('timestamp').str.to_datetime().dt.day().cast(pl.UInt8).alias('day'),  # Extract and cast day
    pl.col('timestamp').str.to_datetime().dt.hour().cast(pl.UInt8).alias('hour')  # Extract and cast hour
]

# Column transformations for data
data_transforms = [
    pl.col('anglez').cast(pl.Int16),  # Casting 'anglez' to 16-bit integer
    (pl.col('enmo') * 1000).cast(pl.UInt16)  # Convert 'enmo' to 16-bit unsigned integer
]

# Loading and transforming training series data
train_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/train_series.parquet').with_columns(
    dt_transforms + data_transforms
)

# Loading and transforming training events data
train_events = pl.read_csv('/kaggle/input/child-mind-institute-detect-sleep-states/train_events.csv').with_columns(
    dt_transforms
)

# Loading and transforming test series data
test_series = pl.scan_parquet('/kaggle/input/child-mind-institute-detect-sleep-states/test_series.parquet').with_columns(
    dt_transforms + data_transforms
)

# Getting unique series IDs for convenience
series_ids = train_events['series_id'].unique(maintain_order=True).to_list()

# Removing series with mismatched event counts (onset vs. wakeup)
onset_counts = train_events.filter(pl.col('event') == 'onset').group_by('series_id').count().sort('series_id')['count']
wakeup_counts = train_events.filter(pl.col('event') == 'wakeup').group_by('series_id').count().sort('series_id')['count']

counts = pl.DataFrame({'series_id': sorted(series_ids), 'onset_counts': onset_counts, 'wakeup_counts': wakeup_counts})
count_mismatches = counts.filter(counts['onset_counts'] != counts['wakeup_counts'])

# Filtering out series with count mismatches
train_series = train_series.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))
train_events = train_events.filter(~pl.col('series_id').is_in(count_mismatches['series_id']))

# Updating the list of series IDs, excluding series with no non-null values
series_ids = train_events.drop_nulls()['series_id'].unique(maintain_order=True).to_list()


# ## 🧮 Creating Features 📈
# 

# **Explaination**
# This code is focused on generating and adding various features to two DataFrames (`train_series` and `test_series`) based on different time windows and data transformations.
# 
# 1. `features, feature_cols = [pl.col('hour')], ['hour']`: This line initializes two lists, `features` and `feature_cols`. It starts with one feature, which is the 'hour' column, and one feature column name 'hour'.
# 
# 2. `for mins in [5, 30, 60*2, 60*8]:`: This is the beginning of a loop that iterates over different time window durations specified in minutes (5, 30, 120, and 480).
# 
# 3. `features += [...]`: This line appends new features to the `features` list. For each time window duration (`mins`), it calculates the following features for the 'enmo' column:
#    - Rolling mean with a window of 12 times the specified minutes (`12 * mins`) using `pl.col('enmo').rolling_mean(...)`.
#    - Rolling maximum with the same window using `pl.col('enmo').rolling_max(...)`.
#    It then casts these features to 16-bit unsigned integers and assigns aliases with the format 'enmo_Xm_mean' and 'enmo_Xm_max', where X is the time window duration in minutes.
# 
# 4. `feature_cols += [...]`: This line appends the corresponding feature column names to the `feature_cols` list, which are 'enmo_Xm_mean' and 'enmo_Xm_max' for each time window duration.
# 
# 5. The same steps are repeated for the 'anglez' column, resulting in features named 'anglez_Xm_mean' and 'anglez_Xm_max' for each time window duration.
# 
# 6. `id_cols = ['series_id', 'step', 'timestamp']`: This line defines a list of columns to keep in the final DataFrame. These columns are 'series_id', 'step', and 'timestamp'.
# 
# 7. `train_series = train_series.with_columns(features).select(id_cols + feature_cols)`: This line adds the calculated features to the `train_series` DataFrame using the `with_columns` method and then selects only the columns specified in `id_cols` and `feature_cols`. This effectively updates `train_series` with the newly generated features.
# 
# 8. `test_series = test_series.with_columns(features).select(id_cols + feature_cols)`: Similar to the previous line, this line adds the same calculated features to the `test_series` DataFrame and selects the specified columns.
# 
# This code generates features based on rolling statistics (mean and max) for both the 'enmo' and 'anglez' columns, with different time window durations. It then updates the `train_series` and `test_series` DataFrames with these calculated features while keeping the specified identifier and feature columns. This process is common in feature engineering for machine learning tasks, where additional features are created from existing data to improve model performance.

# In[3]:


# Initializing features and feature column names
features, feature_cols = [pl.col('hour')], ['hour']

# Generating features for different time windows
for mins in [5, 30, 60*2, 60*8]:
    # Enmo rolling mean and max
    features += [
        pl.col('enmo').rolling_mean(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_mean'),
        pl.col('enmo').rolling_max(12 * mins, center=True, min_periods=1).abs().cast(pl.UInt16).alias(f'enmo_{mins}m_max')
    ]

    feature_cols += [
        f'enmo_{mins}m_mean', f'enmo_{mins}m_max'
    ]

    # Anglez and Enmo first variations
    for var in ['enmo', 'anglez']:
        features += [
            (pl.col(var).diff().abs().rolling_mean(12 * mins, center=True, min_periods=1) * 10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_mean'),
            (pl.col(var).diff().abs().rolling_max(12 * mins, center=True, min_periods=1) * 10).abs().cast(pl.UInt32).alias(f'{var}_1v_{mins}m_max')
        ]

        feature_cols += [
            f'{var}_1v_{mins}m_mean', f'{var}_1v_{mins}m_max'
        ]

# Defining columns to keep
id_cols = ['series_id', 'step', 'timestamp']

# Adding calculated features to the training and testing series
train_series = train_series.with_columns(
    features
).select(id_cols + feature_cols)

test_series = test_series.with_columns(
    features
).select(id_cols + feature_cols)


# ## 📊 Creating Training Dataset Function 📉
# 
# 

# **Explaination** :
# 
# This Python function, `make_train_dataset`, is designed to create a training dataset from raw data, which includes both feature data (`train_data`) and event information (`train_events`). It returns feature data `X` and target labels `y` for training our machine learning model.:
# 
# 1. `series_ids = train_data['series_id'].unique(maintain_order=True).to_list()`: This line extracts unique 'series_id' values from the `train_data` DataFrame while maintaining the original order and converts them to a Python list. These unique series identifiers will be used to process data for each individual series.
# 
# 2. `X, y = pl.DataFrame(), pl.DataFrame()`: This line initializes two empty DataFrames, `X` for features and `y` for target labels.
# 
# 3. `for idx in tqdm(series_ids):`: This loop iterates through each unique 'series_id' in the `series_ids` list, using `tqdm` to display a progress bar.
# 
# 4. `sample = train_data.filter(...)`: Within the loop, this line filters the `train_data` DataFrame to obtain the data for the current 'series_id'. It then normalizes the selected columns (except 'hour') by dividing each column by its standard deviation and casting the result as a 32-bit floating-point number (Float32).
# 
# 5. `events = train_events.filter(...)`: This line filters the `train_events` DataFrame to obtain the events data for the current 'series_id'.
# 
# 6. `if drop_nulls: ...`: If the `drop_nulls` parameter is set to `True`, this section removes data points in the `sample` DataFrame that correspond to dates where no events were recorded. It does this by filtering the `sample` DataFrame based on matching dates between `sample` and `events` using the 'timestamp' column.
# 
# 7. `X = X.vstack(...)`: This line vertically stacks the current 'sample' DataFrame onto the existing 'X' DataFrame, combining the features for multiple series.
# 
# 8. `onsets = events.filter(...)`: This line extracts the 'step' values corresponding to 'onset' events from the 'events' DataFrame and stores them in the `onsets` list.
# 
# 9. `wakeups = events.filter(...)`: Similar to the previous line, this extracts the 'step' values corresponding to 'wakeup' events and stores them in the `wakeups` list.
# 
# 10. `y = y.vstack(...)`: This line constructs target labels ('asleep') based on the 'onsets' and 'wakeups' values. It checks for each data point in 'sample' if it falls within any sleep interval defined by 'onset' and 'wakeup' events, and casts the result as a Boolean value. The resulting labels are stacked vertically onto the 'y' DataFrame.
# 
# 11. `y = y.to_numpy().ravel()`: After processing all series, this line converts the 'y' DataFrame to a NumPy array and flattens it to a 1D array. This is typically done to match the format expected by many machine learning models.
# 
# 12. Finally, the function returns `X` (the feature matrix) and `y` (the target labels) as the output.
# 
# This function takes raw feature data and event information, processes them for each series, and creates a dataset suitable for training machine learning models. It provides a way to drop null data points if desired and generates target labels based on sleep event information.

# In[4]:


def make_train_dataset(train_data, train_events, drop_nulls=False):
    """
    Create a training dataset from raw data.

    Args:
        train_data (pl.DataFrame): Raw training data with features.
        train_events (pl.DataFrame): Training events data with event information.
        drop_nulls (bool): Whether to drop null data points where no events were recorded.

    Returns:
        X (pl.DataFrame): Features for training.
        y (numpy.ndarray): Target labels for training.
    """
    
    series_ids = train_data['series_id'].unique(maintain_order=True).to_list()
    X, y = pl.DataFrame(), pl.DataFrame()
    
    for idx in tqdm(series_ids):
        # Normalizing sample features
        sample = train_data.filter(pl.col('series_id') == idx).with_columns(
            [(pl.col(col) / pl.col(col).std()).cast(pl.Float32) for col in feature_cols if col != 'hour']
        )
        
        events = train_events.filter(pl.col('series_id') == idx)
        
        if drop_nulls:
            # Removing datapoints on dates where no data was recorded
            sample = sample.filter(
                pl.col('timestamp').dt.date().is_in(events['timestamp'].dt.date())
            )
        
        X = X.vstack(sample[id_cols + feature_cols])

        onsets = events.filter((pl.col('event') == 'onset') & (pl.col('step') != None))['step'].to_list()
        wakeups = events.filter((pl.col('event') == 'wakeup') & (pl.col('step') != None))['step'].to_list()

        # NOTE: This will break if there are event series without any recorded onsets or wakeups
        y = y.vstack(sample.with_columns(
            sum([(onset <= pl.col('step')) & (pl.col('step') <= wakeup) for onset, wakeup in zip(onsets, wakeups)]).cast(pl.Boolean).alias('asleep')
            ).select('asleep')
        )
    
    y = y.to_numpy().ravel()
    
    return X, y


# ## 🕒 Get Events Function 🌙

# **Explaination**:
# 
# This Python function, `get_events`, is designed to generate sleep event predictions and format them as a submission DataFrame. It takes a time series dataset (`series`) with features and a trained classifier (`classifier`) for predicting sleep events. 
# 
# 1. `series_ids = series['series_id'].unique(maintain_order=True).to_list()`: This line extracts unique 'series_id' values from the `series` DataFrame while maintaining the original order and converts them to a Python list. These unique series identifiers will be used to process data for each individual series.
# 
# 2. `events = pl.DataFrame(...)`: This line initializes an empty DataFrame called `events` with a specific schema containing columns: 'series_id' (string), 'step' (integer), 'event' (string), and 'score' (float). This DataFrame will store sleep event predictions.
# 
# 3. `for idx in tqdm(series_ids):`: This loop iterates through each unique 'series_id' in the `series_ids` list, using `tqdm` to display a progress bar.
# 
# 4. `scale_cols = [...]`: This line identifies the columns in the `feature_cols` list (excluding 'hour') that have a standard deviation not equal to zero. These columns will be used for feature scaling later.
# 
# 5. `X = series.filter(...)`: Within the loop, this line filters the `series` DataFrame to obtain data for the current 'series_id'. It selects the columns specified in `id_cols` and `feature_cols`. The selected columns are then normalized by dividing them by their standard deviation and casting the result as a 32-bit floating-point number (Float32).
# 
# 6. `preds, probs = ...`: This line applies the trained classifier (`classifier`) to predict sleep events and obtain corresponding probabilities for each data point in the 'X' DataFrame.
# 
# 7. `X = X.with_columns(...)`: This code adds two new columns to the 'X' DataFrame:
#    - 'prediction': It contains the integer predictions (0 or 1) for sleep events.
#    - 'probability': It contains the predicted probabilities of being in a sleep state.
# 
# 8. `pred_onsets = X.filter(...)`, `pred_wakeups = X.filter(...)`: These lines extract predicted 'onset' and 'wakeup' time steps based on changes in the 'prediction' column. These steps are converted to Python lists.
# 
# 9. `if len(pred_onsets) > 0: ...`: This conditional block checks if there are any predicted sleep events ('onset' and 'wakeup' pairs). If there are, it proceeds to process and filter these events.
# 
# 10. Inside the conditional block, it ensures that predicted sleep periods have a minimum duration of 30 minutes (12 * 30 time steps).
# 
# 11. It calculates a 'score' for each sleep period by taking the mean probability over that period.
# 
# 12. It then adds sleep events to the `events` DataFrame, including 'onset' and 'wakeup' events, along with their corresponding 'score'.
# 
# 13. The loop continues to process each series, accumulating sleep events in the `events` DataFrame.
# 
# 14. After processing all series, it converts the `events` DataFrame to a Pandas DataFrame, resets the index, and renames the index column to 'row_id'.
# 
# 15. Finally, the function returns the formatted sleep events as a Pandas DataFrame.
# 
# This function takes a time series dataset and a trained classifier, predicts sleep events, filters and processes them, and returns the results as a Pandas DataFrame suitable for submission or further analysis.

# In[5]:


def get_events(series, classifier):
    """
    Takes a time series and a classifier and returns a formatted submission dataframe.

    Args:
        series (pl.DataFrame): Time series data with features.
        classifier: A trained classifier for predicting sleep events.

    Returns:
        events (pd.DataFrame): Formatted submission dataframe with sleep events.
    """
    
    series_ids = series['series_id'].unique(maintain_order=True).to_list()
    events = pl.DataFrame(schema={'series_id': str, 'step': int, 'event': str, 'score': float})

    for idx in tqdm(series_ids):
        # Collecting sample and normalizing features
        scale_cols = [col for col in feature_cols if (col != 'hour') & (series[col].std() != 0)]
        X = series.filter(pl.col('series_id') == idx).select(id_cols + feature_cols).with_columns(
            [(pl.col(col) / series[col].std()).cast(pl.Float32) for col in scale_cols]
        )

        # Applying classifier to get predictions and scores
        preds, probs = classifier.predict(X[feature_cols]), classifier.predict_proba(X[feature_cols])[:, 1]

        # NOTE: Considered using rolling max to get sleep periods excluding <30 min interruptions,
        # but ended up decreasing performance
        X = X.with_columns(
            pl.lit(preds).cast(pl.Int8).alias('prediction'),
            pl.lit(probs).alias('probability')
        )
        
        # Getting predicted onset and wakeup time steps
        pred_onsets = X.filter(X['prediction'].diff() > 0)['step'].to_list()
        pred_wakeups = X.filter(X['prediction'].diff() < 0)['step'].to_list()
        
        if len(pred_onsets) > 0:
            # Ensuring all predicted sleep periods begin and end
            if min(pred_wakeups) < min(pred_onsets):
                pred_wakeups = pred_wakeups[1:]

            if max(pred_onsets) > max(pred_wakeups):
                pred_onsets = pred_onsets[:-1]

            # Keeping sleep periods longer than 30 minutes
            sleep_periods = [(onset, wakeup) for onset, wakeup in zip(pred_onsets, pred_wakeups) if wakeup - onset >= 12 * 30]

            for onset, wakeup in sleep_periods:
                # Scoring using mean probability over period
                score = X.filter((pl.col('step') >= onset) & (pl.col('step') <= wakeup))['probability'].mean()

                # Adding sleep event to dataframe
                events = events.vstack(pl.DataFrame().with_columns(
                    pl.Series([idx, idx]).alias('series_id'),
                    pl.Series([onset, wakeup]).alias('step'),
                    pl.Series(['onset', 'wakeup']).alias('event'),
                    pl.Series([score, score]).alias('score')
                ))

    # Adding row id column and converting to a pandas DataFrame
    events = events.to_pandas().reset_index().rename(columns={'index': 'row_id'})

    return events


# ## 📊 Collecting and Sampling Data 📊

# **Explaination**:
# 
# This line of code is selecting and collecting a subset of data from the `train_series` DataFrame for further processing.
# 
# 1. `train_data = ...`: This part of the code initializes a variable named `train_data` to store the selected data points.
# 
# 2. `train_series.filter(...)`: Here, the `train_series` DataFrame is filtered using the `filter` method. The condition being applied is `pl.col('series_id').is_in(series_ids)`. This condition filters the rows in `train_series` to include only those where the 'series_id' column matches any of the values in the `series_ids` list.
# 
# 3. `.collect()`: After filtering, the `collect` method is used to retrieve all the rows that satisfy the condition and create a new DataFrame containing these rows. This step essentially materializes the filtered DataFrame.
# 
# 4. `.sample(int(1e6))`: Finally, the `sample` method is applied to the collected DataFrame. It randomly selects a specified number of samples from the DataFrame. In this case, it's selecting 1 million samples (`int(1e6)` represents 1 million). The purpose of taking a random sample is often to reduce the size of the data for quicker experimentation or analysis when working with a large dataset.
# 
# So, the overall purpose of this line of code is to filter the `train_series` DataFrame to include only rows where the 'series_id' matches any of the values in the `series_ids` list, and then randomly sample 1 million rows from this filtered data. This can be useful when you want to work with a manageable subset of data for model training or analysis.

# In[6]:


# We will collect datapoints and take 1 million samples
train_data = train_series.filter(pl.col('series_id').is_in(series_ids)).collect().sample(int(1e6))


# ## 🚂 Creating Training Dataset 🚂
# 
# 

# **Explaination**:
# This code is using a custom function called `make_train_dataset` to generate a training dataset from two data sources: `train_data` and `train_events`. It then assigns the resulting features and labels to `X_train` and `y_train`, respectively.
# 
# 1. `X_train, y_train = ...`: This line initializes two variables, `X_train` and `y_train`, which will store the features and target labels for the training dataset, respectively.
# 
# 2. `make_train_dataset(train_data, train_events)`: This is a function call to `make_train_dataset` with two arguments: `train_data` and `train_events`. It's invoking the function and passing these two data sources to it.
# 
# 3. `make_train_dataset` is presumably a custom function defined elsewhere in the code (not shown here). This function is designed to create a training dataset from raw data. Based on the function signature and comments provided earlier, it takes the following arguments:
#    - `train_data`: A DataFrame containing raw training data with features.
#    - `train_events`: A DataFrame containing training events data with event information.
#    - `drop_nulls`: An optional boolean parameter that determines whether to drop null data points where no events were recorded (this parameter is not explicitly passed in the code snippet you provided).
# 
# 4. The function returns two values: `X` and `y`, where:
#    - `X` is a DataFrame containing features for training.
#    - `y` is a NumPy array containing the target labels for training.
# 
# 5. The line `X_train, y_train = make_train_dataset(train_data, train_events)` assigns the returned values from the `make_train_dataset` function to `X_train` and `y_train`, respectively. This essentially populates `X_train` with the features and `y_train` with the corresponding target labels for training.
# 
# In summary, this line of code is invoking a custom function, `make_train_dataset`, to generate a training dataset from `train_data` and `train_events`, and then assigns the resulting features and labels to `X_train` and `y_train`. This is a common step in preparing data for machine learning, where the data is split into features (X) and labels (y) for model training.

# In[7]:


# Generating training dataset using the 'make_train_dataset' function
X_train, y_train = make_train_dataset(train_data, train_events)


# ## 🧹 Recovering Memory 🧹

# **Explaination**:
# 
# 
# This code is responsible for removing the `train_data` variable from memory and then explicitly triggering garbage collection to free up any associated memory resources.:
# 
# 1. `del train_data`: The `del` statement is used in Python to delete a reference to an object. In this case, it is deleting the reference to the `train_data` variable. This means that the variable `train_data` will no longer exist in the current scope, and any memory associated with it will become eligible for garbage collection.
# 
# 2. `gc.collect()`: The `gc` module in Python provides an interface to the garbage collection mechanism. Calling `gc.collect()` explicitly triggers garbage collection. Garbage collection is the process by which Python automatically reclaims memory that is no longer being used by the program, such as objects that have no remaining references. By invoking `gc.collect()`, the program is requesting that the Python garbage collector run immediately to reclaim any memory that can be freed.
# 
# In summary, this code snippet is cleaning up memory by removing the reference to the `train_data` variable and then explicitly requesting garbage collection to release any memory resources associated with it. This can be useful when working with large datasets to ensure that memory is managed efficiently and that resources are released when they are no longer needed.

# In[8]:


# Removing the 'train_data' variable and triggering garbage collection to free up memory
del train_data
gc.collect()


# # 🧮 LightGBM Classifier Training 🚀
# 
# 

# **Explaination**:
# 
# This code segment is configuring, creating, and training a LightGBM (Light Gradient Boosting Machine) classifier for a machine learning task. 
# 
# 1. `lgb_opt`: This is a dictionary containing hyperparameters for the LightGBM classifier. The hyperparameters specified are:
#    - `'num_leaves'`: This parameter controls the number of leaves (terminal nodes) in each tree in the gradient boosting model. A higher value can make the model more expressive, but it can also lead to overfitting if set too high.
#    - `'learning_rate'`: This parameter determines the step size at each iteration during the gradient boosting process. A smaller learning rate often requires more iterations but can result in a better-performing model.
#    - `'random_state'`: This parameter sets the random seed for reproducibility. It ensures that the same results are obtained when the model is trained with the same data and hyperparameters.
# 
# 2. `my_classifier = lgb.LGBMClassifier(**lgb_opt)`: This line creates an instance of the LightGBM classifier with the specified hyperparameters (`lgb_opt`). The double asterisks (`**`) are used to unpack the dictionary `lgb_opt` and pass its contents as keyword arguments to the `LGBMClassifier` constructor. This initializes the classifier with the specified hyperparameters.
# 
# 3. `my_classifier.fit(X_train[feature_cols], y_train)`: This line trains the LightGBM classifier (`my_classifier`) on the training dataset. Here's what each part does:
#    - `X_train[feature_cols]`: This is the feature matrix used for training. It selects only the columns specified in `feature_cols`, which typically represent the features used for prediction.
#    - `y_train`: This is the target labels or ground truth values corresponding to the training dataset.
# 
# The `fit` method is called to train the classifier using the specified features and target labels. During training, the LightGBM algorithm will iteratively build an ensemble of decision trees, optimizing their structure to minimize a specified loss function. The trained model (`my_classifier`) can then be used for making predictions on new data.
# 
# In summary, this code segment sets hyperparameters for a LightGBM classifier, creates an instance of the classifier with those hyperparameters, and trains it on the provided training dataset (`X_train` and `y_train`). The resulting model can be used for making predictions on new data.

# In[9]:


# Hyperparameters for LightGBM
lgb_opt = {
    'num_leaves': 204,
    'learning_rate': 0.07649523437092402,
    'random_state': 42
}

# Creating a LightGBM classifier with the specified hyperparameters
my_classifier = lgb.LGBMClassifier(**lgb_opt)

# Training the classifier on the training dataset
my_classifier.fit(X_train[feature_cols], y_train)


# ## 📝 Getting Event Predictions and Saving Submission 📄
# 
# 

# **Explaination**:
# 
# This code snippet is responsible for generating sleep event predictions for the test set using a trained LightGBM classifier (`my_classifier`) and saving the predictions to a CSV file.
# 
# 1. `submission = get_events(test_series.collect(), my_classifier)`: This line calls a function called `get_events` to obtain sleep event predictions for the test set. Here's what each part of the code does:
#    - `test_series.collect()`: The `collect` method is used to gather all the data points from the `test_series` DataFrame. This step effectively retrieves all the test data for processing.
#    - `my_classifier`: This is the trained LightGBM classifier that will be used to predict sleep events based on the test data.
# 
#    The `get_events` function presumably takes this collected test data and the classifier as input and returns a formatted DataFrame containing sleep event predictions.
# 
# 2. `submission.to_csv('submission.csv', index=False)`: This line saves the DataFrame named `submission` to a CSV file named 'submission.csv'. Here's what each part of the code does:
#    - `submission`: This is the DataFrame containing the sleep event predictions generated using the test data.
#    - `.to_csv(...)`: This is a method call on the DataFrame that exports the data to a CSV file. The parameters are as follows:
#       - `'submission.csv'`: This specifies the name of the CSV file to which the DataFrame will be saved.
#       - `index=False`: This parameter indicates that the DataFrame's index (row numbers) should not be included in the CSV file. When `index` is set to `False`, the CSV file will not have an extra column for row numbers.
# 
# In summary, this code segment applies a trained LightGBM classifier to the test data to predict sleep events, collects these predictions into a DataFrame named `submission`, and then saves this DataFrame to a CSV file named 'submission.csv'. The resulting CSV file can be used for submission in a machine learning competition or for further analysis.

# In[10]:


# Getting event predictions for the test set using the trained classifier
submission = get_events(test_series.collect(), my_classifier)

# Saving the submission dataframe to a CSV file
submission.to_csv('submission.csv', index=False)


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
