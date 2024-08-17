#!/usr/bin/env python
# coding: utf-8

# # Explained Critical point RNN 
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the child mind institute Predict AI Model Runtime! This notebook is designed to help you participate in the competition and to Detect sleep onset and wake from wrist-worn accelerometer data.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of werus23, available at [this Kaggle project](https://www.kaggle.com/code/werus23/sleep-critical-point-infer/notebook). I extend my gratitude to werus23 for sharing their insights and code.
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
# We acknowledge The Child Mind Institute organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 

# ## Import necessary libraries and modules 📚

# In[1]:


import pandas as pd           # Import and alias pandas (data manipulation) 🐼
import numpy as np            # Import and alias numpy (numerical operations) 🧮
import gc                    # Import garbage collection for memory management 🗑️
import time                  # Import time module for time-related operations ⏰
import json                  # Import JSON handling for data 📝
from datetime import datetime  # Import datetime for date and time manipulation 📅
import matplotlib.pyplot as plt  # Import and alias matplotlib for data visualization 📊
import os                    # Import os for operating system functions 📂
import joblib                # Import joblib for job (de)serialization 🧰
import random                # Import random for generating random numbers 🎲
import math                  # Import math module for mathematical functions ➗
from tqdm.auto import tqdm   # Import tqdm for progress bars 📊

# Data science and modeling libraries 📊🧪
from scipy.interpolate import interp1d  # Interpolation functions for data 📈
from math import pi, sqrt, exp        # Math constants and functions 🔵📏√
import sklearn, sklearn.model_selection  # Import scikit-learn for ML tasks 🧪📊
import torch                 # Import PyTorch for deep learning 🔥
from torch import nn, Tensor  # Import neural network components 🧯
import torch.nn.functional as F  # Import PyTorch's functional API 🧯📊
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler  # Data loading utilities 🧮📊
from sklearn.metrics import average_precision_score  # Import a metric for evaluation 🧪📈
from timm.scheduler import CosineLRScheduler  # Scheduler for learning rate adjustment 📅⏰

# Set the style for matplotlib plots 📊📈
plt.style.use("ggplot")

# Import libraries for working with Parquet files 📄
from pyarrow.parquet import ParquetFile  # PyArrow Parquet file handling 📄
import pyarrow as pa          # PyArrow for data serialization 📄
import ctypes                 # Python library for calling C functions 📄

# Set the number of threads for interoperation and processing 🔢
torch.set_num_interop_threads(4)  # Set the number of interop threads 🔢
torch.set_num_threads(4)     # Set the number of CPU threads 🔢

# Determine if CUDA (GPU) is available, and set the device accordingly 💻
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set device to 'cuda' if GPU is available


# ## Define paths to data files 📂

# **Cell 2**
# 
# **Explaination**:
# 
# This code defines a set of classes and functions for handling data loading, cleaning, and memory optimization. Let's break down each part:
# 
# 1. `PATHS` class:
#    - It defines paths to various data files, including submission data, training events data, training series data (in CSV format), and test series data (in Parquet format).
# 
# 2. `CFG` class:
#    - It defines configuration settings, such as `DEMO_MODE`, which can be set to `True` or `False` to enable or disable demo mode.
# 
# 3. `data_reader` class:
#    - It is responsible for handling data loading and processing.
#    - `__init__` method:
#      - Initializes the class with mappings for data loading and sets the demo mode.
#    - `verify` method:
#      - Verifies whether a given data name is valid. If it's not a valid data name, it prints an error message.
#    - `cleaning` method:
#      - Cleans the data by removing rows with missing timestamps.
#    - `reduce_memory_usage` method:
#      - Reduces memory usage by modifying data types (e.g., changing data types to use less memory).
#    - `load_data` method:
#      - Loads data from the provided data name, either from a CSV or Parquet file.
#      - Depending on the demo mode setting, it loads either a limited number of rows or the entire dataset.
#      - If the data has timestamps, it calls the `cleaning` method to remove rows with missing timestamps.
#      - After loading the data, it calls the `reduce_memory_usage` method to optimize memory usage.
# 
# 
# 

# In[2]:


class PATHS:
    MAIN_DIR = "/kaggle/input/child-mind-institute-detect-sleep-states/"
    # CSV FILES : 
    SUBMISSION = MAIN_DIR + "sample_submission.csv"
    TRAIN_EVENTS = MAIN_DIR + "train_events.csv"
    # PARQUET FILES:
    TRAIN_SERIES = MAIN_DIR + "train_series.parquet"
    TEST_SERIES = MAIN_DIR + "test_series.parquet"
class CFG:
    DEMO_MODE = True
class data_reader:
    def __init__(self, demo_mode):
        super().__init__()
        # MAPPING FOR DATA LOADING :
        self.names_mapping = {
            "submission" : {"path" : PATHS.SUBMISSION, "is_parquet" : False, "has_timestamp" : False}, 
            "train_events" : {"path" : PATHS.TRAIN_EVENTS, "is_parquet" : False, "has_timestamp" : True},
            "train_series" : {"path" : PATHS.TRAIN_SERIES, "is_parquet" : True, "has_timestamp" : True},
            "test_series" : {"path" : PATHS.TEST_SERIES, "is_parquet" : True, "has_timestamp" : True}
        }
        self.valid_names = ["submission", "train_events", "train_series", "test_series"]
        self.demo_mode = demo_mode
    
    def verify(self, data_name):
        "function for data name verification"
        if data_name not in self.valid_names:
            print("PLEASE ENTER A VALID DATASET NAME, VALID NAMES ARE : ", valid_names)
        return
    
    def cleaning(self, data):
        "cleaning function : drop na values"
        before_cleaning = len(data)
        print("Number of missing timestamps : ", len(data[data["timestamp"].isna()]))
        data = data.dropna(subset=["timestamp"])
        after_cleaning = len(data)
        print("Percentage of removed rows : {:.1f}%".format(100 * (before_cleaning - after_cleaning) / before_cleaning) )
#         print(data.isna().any())
#         data = data.bfill()
        return data
    
    @staticmethod
    def reduce_memory_usage(data):
        "iterate through all the columns of a dataframe and modify the data type to reduce memory usage."
        start_mem = data.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        for col in data.columns:
            col_type = data[col].dtype    
            if col_type != object:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
            else:
                data[col] = data[col].astype('category')

        end_mem = data.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return data
    
    def load_data(self, data_name):
        "function for data loading"
        self.verify(data_name)
        data_props = self.names_mapping[data_name]
        if data_props["is_parquet"]:
            if self.demo_mode:
                pf = ParquetFile(data_props["path"]) 
                demo_rows = next(pf.iter_batches(batch_size=20_000)) 
                data = pa.Table.from_batches([demo_rows]).to_pandas()
            else:
                data = pd.read_parquet(data_props["path"])
        else:
            if self.demo_mode:
                data = pd.read_csv(data_props["path"], nrows=20_000)
            else:
                data = pd.read_csv(data_props["path"])
                
        gc.collect()
        if data_props["has_timestamp"]:
            print('cleaning')
            data = self.cleaning(data)
            gc.collect()
        data = self.reduce_memory_usage(data)
        return data


# ## Initialize data reader and load test_series data 🆔

# **Cell 3**
# 
# **Explaination**:
# 
# 
# 
# 1. `reader = data_reader(demo_mode=False)`: This line creates an instance of the `data_reader` class with `demo_mode` set to `False`. The `data_reader` class is assumed to be a custom class responsible for data loading and management.
# 
# 2. `test_series = reader.load_data(data_name="test_series")`: It uses the `reader` object to load data with the name "test_series." This data is stored in the variable `test_series`.
# 
# 3. `ids = test_series.series_id.unique()`: It extracts unique series IDs from the `test_series` data using the `unique()` method. The resulting unique series IDs are stored in the `ids` variable.
# 
# 4. `gc.collect()`: This line triggers the Python garbage collector to free up memory by collecting and cleaning up any unreferenced objects, helping to manage memory usage efficiently.
# 
# This code is for data processing pipeline where data is loaded, specific information is extracted (unique series IDs), and memory management is handled by garbage collection.

# In[3]:


# Create a data_reader object with demo_mode set to False 🧑‍💻
reader = data_reader(demo_mode=False)

# Load the 'test_series' data using the data_reader 📥
test_series = reader.load_data(data_name="test_series")

# Get unique series IDs from the 'test_series' data 🆔
ids = test_series.series_id.unique()

# Perform garbage collection to free up memory 🗑️
gc.collect()


# **Cell 4**
# 
# **Explaination**:
# 
# The provided code defines two classes, `ResidualBiGRU` and `MultiResidualBiGRU`, which are components for a neural network model:
# 
# 1. `ResidualBiGRU` class:
# 
#    This class defines a single bidirectional GRU layer with residual connections.
# 
#    - `__init__` method:
#      - It initializes the class with parameters for the hidden size of the GRU, the number of layers, and whether it should be bidirectional.
#      - It creates a bidirectional GRU layer with the specified configuration.
#      - It defines fully connected layers (`fc1` and `fc2`) and LayerNorms (`ln1` and `ln2`) for residual connections.
#      
#    - `forward` method:
#      - It takes an input `x` and an optional hidden state `h` as arguments.
#      - It passes the input through the bidirectional GRU layer.
#      - It applies fully connected layers and LayerNorm for residual connections, using ReLU activation.
#      - It implements a skip connection (residual connection) by adding the original input `x` to the processed output.
#      - It returns the output and the new hidden state.
# 
# 2. `MultiResidualBiGRU` class:
# 
#    This class creates a neural network model with multiple layers of `ResidualBiGRU` modules.
# 
#    - `__init__` method:
#      - It initializes the class with parameters for the input size, hidden size, output size, the number of layers, and whether the GRU layers should be bidirectional.
#      - It defines an initial fully connected layer (`fc_in`) and LayerNorm for the input data.
#      - It creates a list of `ResidualBiGRU` layers, where each layer has a hidden size and is either unidirectional or bidirectional.
#      - It defines an output fully connected layer (`fc_out`).
# 
#    - `forward` method:
#      - It takes an input `x` and an optional list of hidden states `h`.
#      - If no hidden states are provided, it initializes them as `None`.
#      - It applies the initial fully connected layer and LayerNorm to the input data.
#      - It iterates through the list of `ResidualBiGRU` layers, passing the input and the corresponding hidden state through each layer and collecting the new hidden states.
#      - It applies the output fully connected layer to the final output.
#      - It returns the final output and a list of hidden states.
# 
# These classes are designed for building a deep neural network with residual connections using GRU layers. The `MultiResidualBiGRU` class stacks multiple `ResidualBiGRU` layers to create a deep architecture for sequence processing tasks.

# In[4]:


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)

        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)

        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)

        # skip connection
        res = res + x

        return res, new_h

class MultiResidualBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers, bidir=True):
        super(MultiResidualBiGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_layers = n_layers

        self.fc_in = nn.Linear(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(hidden_size, n_layers=1, bidir=bidir)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        # if we are at the beginning of a sequence (no hidden state)
        if h is None:
            # (re)initialize the hidden state
            h = [None for _ in range(self.n_layers)]

        x = self.fc_in(x)
        x = self.ln(x)
        x = nn.functional.relu(x)

        new_h = []
        for i, res_bigru in enumerate(self.res_bigrus):
            x, new_hi = res_bigru(x, h[i])
            new_h.append(new_hi)

        x = self.fc_out(x)
#         x = F.normalize(x,dim=0)
        return x, new_h  # log probabilities + hidden state


# **Cell 5**
# 
# **Explaination**:
# 
# 
# ```python
# # SleepDataset class to create a custom PyTorch dataset 🛏️
# class SleepDataset(Dataset):
#     def __init(
#         self,
#         series_ids,
#         series,
#     ):
#         series_ids = series_ids
#         series = series.reset_index()
#         self.data = []
# 
#         # Loop through series_ids and prepare the data for each visualization
#         for viz_id in tqdm(series_ids):
#             self.data.append(series.loc[(series.series_id==viz_id)].copy().reset_index())
# ```
# - This code defines a custom PyTorch dataset called `SleepDataset` by inheriting from the `Dataset` class.
# - The `__init__` method initializes the dataset. It takes two arguments: `series_ids` and `series`, which represent the series IDs and the series data, respectively.
# - The `series_ids` argument is assigned to a local variable `series_ids`, but it's not used in the constructor.
# - The `series` data is reset to have a new index using `reset_index()`.
# - An empty list `self.data` is created to store the data for each visualization.
# 
# ```python
#     # Function to downsample sequences and generate features 📉
#     def downsample_seq_generate_features(self, feat, downsample_factor):
#         if len(feat) % 12 != 0:
#             feat = np.concatenate([feat, np.zeros(12 - ((len(feat)) % 12)) + feat[-1]])
#         feat = np reshape(feat, (-1, 12))
#         feat_mean = np.mean(feat, 1)
#         feat_std = np.std(feat, 1)
#         feat_median = np.median(feat, 1)
#         feat_max = np.max(feat, 1)
#         feat_min = np.min(feat, 1)
# 
#         return np.dstack([feat_mean, feat_std, feat_median, feat_max, feat_min])[0]
# ```
# - The `downsample_seq_generate_features` method takes a sequence of features `feat` and a `downsample_factor` as arguments. It downsamples the sequence and generates statistical features.
# - If the length of the input sequence is not a multiple of 12, it pads the sequence with zeros to make its length a multiple of 12.
# - The padded sequence is then reshaped into a 2D array where each row has 12 values.
# - Statistical features (mean, standard deviation, median, maximum, and minimum) are calculated along each row.
# - The features are stacked into a 3D array and returned.
# 
# ```python
#     def __len__(self):
#         return len(self.data)
# 
#     def __getitem__(self, index):
#         X = self.data[index][['anglez', 'enmo']].values.astype(np.float32)
# 
#         # Generate features by downsampling sequences
#         X = np.concatenate([self.downsample_seq_generate_features(X[:, i], 12) for i in range(X shape[1])], -1)
#         X = torch.from_numpy(X)
# 
#         return X
# ```
# - The `__len__` method returns the length of the dataset, which is the number of visualizations stored in `self.data`.
# - The `__getitem__` method retrieves an item from the dataset at the specified `index`.
# - It extracts features 'anglez' and 'enmo' from the data at the given index, converts them to a NumPy array of type `float32`, and stores it in the variable `X`.
# - The `downsample_seq_generate_features` function is called to generate features by downsampling the sequences in `X`.
# - The features are concatenated horizontally, and a PyTorch tensor is created from the result. This tensor is returned as the item from the dataset.
# 
# ```python
# # Create a test dataset using the SleepDataset class 🛏️
# test_ds = SleepDataset(test_series.series_id.unique(), test_series)
# 
# # Cleanup to save memory
# del test_series
# gc.collect()
# ```
# - The `test_ds` variable is assigned a new instance of the `SleepDataset` class, which is initialized with unique series IDs from the `test_series` data.
# - After creating the dataset, there's some cleanup to release memory. The `test_series` variable is deleted, and `gc.collect()` is used to trigger garbage collection, freeing up any unreferenced memory.
# 
# This code defines a custom PyTorch dataset for sleep data and initializes a test dataset with it. The dataset processes the data to generate features for use.
# 

# In[5]:


class SleepDataset(Dataset):
    def __init__(
        self,
        series_ids,
        series,
    ):
        series_ids = series_ids
        series = series.reset_index()
        self.data = []
        
        for viz_id in tqdm(series_ids):
            self.data.append(series.loc[(series.series_id==viz_id)].copy().reset_index())
            
    def downsample_seq_generate_features(self,feat, downsample_factor):
        
        if len(feat)%12!=0:
            feat = np.concatenate([feat,np.zeros(12-((len(feat))%12))+feat[-1]])
        feat = np.reshape(feat, (-1,12))
        feat_mean = np.mean(feat,1)
        feat_std = np.std(feat,1)
        feat_median = np.median(feat,1)
        feat_max = np.max(feat,1)
        feat_min = np.min(feat,1)

        return np.dstack([feat_mean,feat_std,feat_median,feat_max,feat_min])[0]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index][['anglez','enmo']].values.astype(np.float32)
        X = np.concatenate([self.downsample_seq_generate_features(X[:,i],12) for i in range(X.shape[1])],-1)
        X = torch.from_numpy(X)
        return X
test_ds = SleepDataset(test_series.series_id.unique(),test_series)
del test_series
gc.collect()


# **Cell 6**
# 
# **Explaination**:
# 
# Certainly, let's explain these two variable assignments:
# 
# 1. `max_chunk_size = 24 * 60 * 100`: This line sets the variable `max_chunk_size` to a value. It's calculated as follows:
#    - `24` represents the number of hours in a day.
#    - `60` represents the number of minutes in an hour.
#    - `100` appears to be a scaling factor.
#    
#    So, `max_chunk_size` is set to a value that represents the maximum chunk size in minutes. It's equivalent to 144,000 minutes, which is 100 days (24 hours * 60 minutes * 100). This value is used in the code to process data in chunks, with each chunk covering a maximum of 100 days of data.
# 
# 2. `min_interval = 30`: This line sets the variable `min_interval` to a value of `30`. This variable represents the minimum interval in minutes. It's used in the code to define a rolling window within which the maximum scores are calculated. The rolling window will have a width of 30 minutes. If you're working with time-series data, this parameter controls how closely the algorithm looks for maximum scores within a certain time interval.
# 

# In[6]:


max_chunk_size = 24 * 60 * 100  # Maximum chunk size in minutes 🕒
min_interval = 30  # Minimum interval in minutes 🕒


# **Cell 7**
# 
# **Explaination**:
# 
# 
# ```python
# # Load the pre-trained MultiResidualBiGRU model for evaluation 🛏️
# model = MultiResidualBiGRU(input_size=10, hidden_size=64, out_size=2, n_layers=5).to(device).eval()
# model.load_state_dict(torch.load(f'/kaggle/input/sleep-critical-point-train/model_best.pth', map_location=device))
# ```
# 1. This code loads a pre-trained model called `MultiResidualBiGRU` for sleep data analysis. It sets the model to evaluation mode and loads the model's state dictionary from a saved file on the device.
# 
# ```python
# # Create an empty DataFrame for the submission data 📊
# submission = pd.DataFrame()
# ```
# 2. An empty DataFrame named `submission` is created to store the results of the analysis.
# 
# ```python
# # Loop through the test dataset
# for i in range(len(test_ds)):
# ```
# 3. This code starts a loop that iterates over the test dataset (`test_ds`).
# 
# ```python
#     X = test_ds[i].half()
#     seq_len = X.shape[0]
#     h = None
#     pred = torch.zeros((len(X), 2)).half()
# ```
# 4. Inside the loop, it extracts a sample from the test dataset (`test_ds`) and converts it to half-precision (float16). It also initializes some variables (`seq_len`, `h`, and `pred`).
# 
# ```python
#     # Process data in chunks
#     for j in range(0, seq_len, max_chunk_size):
# ```
# 5. Another nested loop is started to process the data in chunks. It iterates over the sequence data in `max_chunk_size` increments.
# 
# ```python
#         y_pred, h = model(X[j: j + max_chunk_size].float(), h)
#         h = [hi.detach() for hi in h]
#         pred[j: j + max_chunk_size] = y_pred.detach()
#         del y_pred
#         gc.collect()
# ```
# 6. Within the nested loop, it feeds a chunk of data to the model and collects predictions (`y_pred`). It also updates the hidden state (`h`). Predictions are stored in the `pred` variable.
# 
# ```python
#     del h, X
#     gc.collect()
#     pred = pred.numpy()
# ```
# 7. After processing the chunk, it cleans up some variables and converts the `pred` tensor to a NumPy array.
# 
# ```python
#     series_id = ids[i]
# ```
# 8. It extracts the `series_id` associated with the current test sample.
# 
# ```python
#     # Calculate the number of days
#     days = len(pred) / (17280 / 12)
# ```
# 9. It calculates the number of days based on the length of the prediction data.
# 
# ```python
#     # Initialize arrays to store scores
#     scores0, scores1 = np.zeros(len(pred), dtype=np.float16), np.zeros(len(pred), dtype=np.float16)
# ```
# 10. Two arrays, `scores0` and `scores1`, are initialized to store scores. They are NumPy arrays with a data type of float16.
# 
# ```python
#     # Find the maximum scores in a rolling window
#     for index in range(len(pred)):
# ```
# 11. A loop is started to iterate over the predictions and calculate maximum scores within a rolling window.
# 
# ```python
#         if pred[index, 0] == max(pred[max(0, index - min_interval):index + min_interval, 0]):
#             scores0[index] = max(pred[max(0, index - min_interval):index + min_interval, 0])
#         if pred[index, 1] == max(pred[max(0, index - min_interval):index + min_interval, 1]):
#             scores1[index] = max(pred[max(0, index - min_interval):index + min_interval, 1])
# ```
# 12. Within the loop, it compares each prediction value to the maximum within a specified interval and updates the scores arrays accordingly.
# 
# ```python
#     # Identify candidates for "onset" and "wakeup" events
#     candidates_onset = np.argsort(scores0)[-max(1, round(days)):]
#     candidates_wakeup = np.argsort(scores1)[-max(1, round(days)):]
# ```
# 13. Candidates for "onset" and "wakeup" events are identified based on the highest scores.
# 
# ```python
#     # Extract the corresponding steps for "onset" and "wakeup" events
#     onset = test_ds.data[i][['step']].iloc[np.clip(candidates_onset * 12, 0, len(test_ds.data[i]) - 1)].astype(np.int32)
#     onset['event'] = 'onset'
#     onset['series_id'] = series_id
#     onset['score'] = scores0[candidates_onset]
#     
#     wakeup = test_ds.data[i][['step']].iloc[np.clip(candidates_wakeup * 12, 0, len(test_ds.data[i]) - 1)].astype(np.int32)
#     wakeup['event'] = 'wakeup'
#     wakeup['series_id'] = series_id
#     wakeup['score'] = scores1[candidates_wakeup]
# ```
# 14. Steps corresponding to the "onset" and "wakeup" events are extracted and stored in the `onset` and `wakeup` DataFrames.
# 
# ```python
#     # Concatenate event data to the submission DataFrame
#     submission = pd.concat([submission, onset, wakeup], axis=0)
#     del onset, wakeup, candidates_onset, candidates_wakeup, scores0, scores1, pred, series_id
#     gc.collect()
# ```
# 15. Event data is concatenated to the `submission` DataFrame, and some variables are cleaned up.
# 
# ```python
# # Sort and reset the index of the submission DataFrame
# submission = submission.sort_values(['series_id', 'step']).reset_index(drop=True)
# submission['row_id'] = submission.index.astype(int)
# submission['score'] = submission['score'].fillna(submission['score'].mean())
# submission = submission[['row_id', 'series_id', 'step', 'event', 'score']]
# ```
# 16. The `submission` DataFrame is sorted, and its index is reset. Additional columns ('row_id' and 'score') are added, and missing scores are filled with the mean score.
# 
# ```python
# # Save the submission data to a CSV file
# submission.to_csv('submission.csv', index=False)
# ```
# 17. The final submission data is saved to a CSV file named 'submission.csv' without including the index column.
# 
# 
# 

# In[7]:


model = MultiResidualBiGRU(input_size=10,hidden_size=64,out_size=2,n_layers=5).to(device).eval()
model.load_state_dict(torch.load(f'/kaggle/input/sleep-critical-point-train/model_best.pth',map_location=device))
submission = pd.DataFrame()
for i in range(len(test_ds)):
    X = test_ds[i].half()
    
    seq_len = X.shape[0]
    h = None
    pred = torch.zeros((len(X),2)).half()
    for j in range(0, seq_len, max_chunk_size):
        y_pred, h = model(X[j: j + max_chunk_size].float(), h)
        h = [hi.detach() for hi in h]
        pred[j : j + max_chunk_size] = y_pred.detach()
        del y_pred;gc.collect()
    del h,X;gc.collect()
    pred = pred.numpy()
    
    series_id = ids[i]
    
    days = len(pred)/(17280/12)
    scores0,scores1 = np.zeros(len(pred),dtype=np.float16),np.zeros(len(pred),dtype=np.float16)
    for index in range(len(pred)):
        if pred[index,0]==max(pred[max(0,index-min_interval):index+min_interval,0]):
            scores0[index] = max(pred[max(0,index-min_interval):index+min_interval,0])
        if pred[index,1]==max(pred[max(0,index-min_interval):index+min_interval,1]):
            scores1[index] = max(pred[max(0,index-min_interval):index+min_interval,1])
    candidates_onset = np.argsort(scores0)[-max(1,round(days)):]
    candidates_wakeup = np.argsort(scores1)[-max(1,round(days)):]
    
    onset = test_ds.data[i][['step']].iloc[np.clip(candidates_onset*12,0,len(test_ds.data[i])-1)].astype(np.int32)
    onset['event'] = 'onset'
    onset['series_id'] = series_id
    onset['score']= scores0[candidates_onset]
    wakeup = test_ds.data[i][['step']].iloc[np.clip(candidates_wakeup*12,0,len(test_ds.data[i])-1)].astype(np.int32)
    wakeup['event'] = 'wakeup'
    wakeup['series_id'] = series_id
    wakeup['score']= scores1[candidates_wakeup]
    submission = pd.concat([submission,onset,wakeup],axis=0)
    del onset,wakeup,candidates_onset,candidates_wakeup,scores0,scores1,pred,series_id,
    gc.collect()
submission = submission.sort_values(['series_id','step']).reset_index(drop=True)
submission['row_id'] = submission.index.astype(int)
submission['score'] = submission['score'].fillna(submission['score'].mean())
submission = submission[['row_id','series_id','step','event','score']]
submission.to_csv('submission.csv',index=False)


# In[8]:


submission


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
# 
