#!/usr/bin/env python
# coding: utf-8

# <h1><center> Foursquare Location Matching </center></h1>
# <h2><center> XGBoost </center></h2>
# <h2><center> Sugata Ghosh </center></h2>

# ### Competition: [Foursquare - Location Matching](https://www.kaggle.com/competitions/foursquare-location-matching)
# 
# ### Notebook on Exploratory Data Analysis: [Foursquare Location Matching - EDA](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-eda)
# 
# ### Notebook on Baseline Modeling: [Foursquare Location Matching - Baseline Modeling](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-baseline-modeling)

# The present notebook implements the [XGBoost](https://en.wikipedia.org/wiki/XGBoost) classifier to classify whether two given [points of interest](https://en.wikipedia.org/wiki/Point_of_interest) (POI) match or not. The algorithm has been optimized in terms of hyperparameters to achieve maximal performance. Note that the first four sections (*Introduction*, *Train-Test Split*, *Data Preprocessing* and *Feature Engineering*) are more or less same as in [Foursquare Location Matching - Baseline Modeling](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-baseline-modeling). The focus of the notebook is the fifth section (*Baseline XGBoost*), where we first implement the baseline XGBoost classifier, and the sixth section (*Hyperparameter Tuning*), where we employ [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) on the algorithm.

# ### Contents
# 
# - [Introduction](#1.-Introduction)
# - [Train-Test Split](#2.-Train-Test-Split)
# - [Data Preprocessing](#3.-Data-Preprocessing)
# - [Feature Engineering](#4.-Feature-Engineering)
# - [Baseline XGBoost](#5.-Baseline-XGBoost)
# - [Hyperparameter Tuning](#6.-Hyperparameter-Tuning)
# - [Acknowledgements](#Acknowledgements)
# - [References](#References)

# ### Importing libraries

# In[1]:


# File system manangement
import time, psutil, os, gc

# Mathematical functions
import math
from math import cos, asin, sqrt, pi

# Data manipulation
import numpy as np
import pandas as pd

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# Train-test split and k-fold cross validation
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

# Classifier
from xgboost import XGBClassifier

# Others
import operator as op
from functools import reduce, lru_cache


# ### Runtime and memory usage

# In[2]:


# Recording the starting time, complemented with a stopping time check in the end to compute process runtime
start = time.time()

# Class representing the OS process and having memory_info() method to compute process memory usage
process = psutil.Process(os.getpid())


# # 1. Introduction
# 
# - [Point of Interest](#Point-of-Interest)
# - [The Problem of POI Matching](#The-Problem-of-POI-Matching)
# - [About Foursquare](#About-Foursquare)
# - [Data](#Data)
# - [Project Objective](#Project-Objective)
# - [Evaluation Metric](#Evaluation-Metric)

# **Note:** The introduction section from the [Foursquare Location Matching - EDA](https://www.kaggle.com/code/sugataghosh/foursquare-location-matching-eda) notebook is reproduced here to make the present notebook self-contained, so that one can understand the problem, as well as the objective of the competition, without requiring to go back to the EDA notebook.

# ## Point of Interest
# 
# A point of interest (POI) is a specific point location that someone may find useful or interesting. An example is a point on the Earth representing the location of the Eiffel Tower, or a point on Mars representing the location of its highest mountain, [Olympus Mons](https://en.wikipedia.org/wiki/Olympus_Mons). Most consumers use the term when referring to hotels, campsites, fuel stations or any other categories used in modern automotive navigation systems. Users of a mobile device can be provided with geolocation and time aware POI service that recommends geolocations nearby and with a temporal relevance (e.g. POI to special services in a ski resort are available only in winter). The notion of POI is widely used in cartography, especially in electronic variants including GIS, and GPS navigation software.

# ## The Problem of POI Matching
# 
# It is useful to combine POI data obtained from multiple sources for effective reusability. One issue in merging such data is that different dataset may have variations in POI name, address, and other identifying information for the same POI. It is thus important to identify observations which refer to the same POI. The process of POI matching involves finding POI pairs that refer to the same real-world entity, which is the core issue in geospatial data integration and is perhaps the most technically difficult part of multi-source POI fusion. The raw location data can contain noise, unstructured information, and incomplete or inaccurate attributes, which makes the task even more difficult. Nonetheless, to maintain the highest level of accuracy, the data must be matched and duplicate POIs must be identified and merged with timely updates from multiple sources. A combination of machine-learning algorithms and rigorous human validation methods are optimal for effective de-duplication of such data.

# ## About Foursquare
# 
# [Foursquare Labs Inc.](https://foursquare.com/), commonly known as Foursquare, is an American location technology company and data cloud platform. The company's location platform is the foundation of several business and consumer products, including the [Foursquare City Guide](https://en.wikipedia.org/wiki/Foursquare_City_Guide) and [Foursquare Swarm](https://en.wikipedia.org/wiki/Foursquare_Swarm) apps. Foursquare's products include Pilgrim SDK, Places, Visits, Attribution, Audience, Proximity, and Unfolded Studio. It is one of the leading independent providers of global POI data and is dedicated to building meaningful bridges between digital spaces and physical places. Trusted by leading enterprises like Apple, Microsoft, Samsung, and Uber, Foursquare's tech stack harnesses the power of places and movement to improve customer experiences and drive better business outcomes.

# ## Data
# 
# **Source:** https://www.kaggle.com/competitions/foursquare-location-matching/data
# 
# The data considered in the competition comprises over one-and-a-half million place entries for hundreds of thousands of commercial Points-of-Interest (POIs) around the globe. Though the data entries may represent or resemble entries for real places, they may be contaminated with artificial information or additional noise.
# 
# The training data comprises eleven attribute fields for over one million place entries, together with:
# - `id` : A unique identifier for each entry.
# - `point_of_interest` : An identifier for the POI the entry represents. There may be one or many entries describing the same POI. Two entries *match* when they describe a common POI.

# In[3]:


# Loading the training data
data_train = pd.read_csv('../input/foursquare-location-matching/train.csv')
print(pd.Series({"Memory usage": "{:.2f} MB".format(data_train.memory_usage().sum()/(1024*1024)),
                 "Dataset shape": "{}".format(data_train.shape)}).to_string())
print(" ")
data_train.head()


# In[4]:


# A typical observation from the training set
data_train.iloc[0]


# The pairs data is a pregenerated set of pairs of place entries from the training data designed to improve detection of matches. It includes:
# - `match` : Boolean variables denoting whether or not the pair of entries describes a common POI.

# In[5]:


# Loading pregenerated set of pairs of place entries from the training data
data_pairs = pd.read_csv('../input/foursquare-location-matching/pairs.csv')
print(pd.Series({"Memory usage": "{:.2f} MB".format(data_pairs.memory_usage().sum()/(1024*1024)),
                 "Dataset shape": "{}".format(data_pairs.shape)}).to_string())
print(" ")
data_pairs.head()


# In[6]:


# A typical observation from the pregenerated set of pairs
data_pairs.iloc[0]


# The test data comprises a set of place entries with their recorded attribute fields, similar to the training set. The POIs in the test data are distinct from the POIs in the training data.

# In[7]:


# Loading the test data
data_test = pd.read_csv('../input/foursquare-location-matching/test.csv')
print(pd.Series({"Memory usage": "{:.5f} MB".format(data_test.memory_usage().sum()/(1024*1024)),
                 "Dataset shape": "{}".format(data_test.shape)}).to_string())
print(" ")
data_test.head()


# In[8]:


# A typical observation from the test set
data_test.iloc[0]


# ## Project Objective
# 
# The goal of the project is to match POIs together. Using the provided dataset of over one-and-a-half million places entries, heavily altered to include noise, duplications, extraneous, or incorrect information, the objective is to produce an algorithm that predicts which place entries represent the same POI. Each place entry in the data includes useful attributes like name, street address, and coordinates. Efficient and successful matching of POIs will make it easier to identify where new stores or businesses would benefit people the most.

# ## Evaluation Metric
# 
# **[Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index).** Also known as *Jaccard similarity coefficient*, it is a statistic used for gauging the similarity and diversity of sample sets. It was developed by [Grove Karl Gilbert](https://en.wikipedia.org/wiki/Grove_Karl_Gilbert) in 1884 as his *ratio of verification (v)* and now is frequently referred to as the *Critical Success Index* in meteorology. It was later developed independently by [Paul Jaccard](https://en.wikipedia.org/wiki/Paul_Jaccard), originally giving the French name *coefficient de communauté* and independently formulated again by T. T. Tanimoto. Thus, the *Tanimoto index* or *Tanimoto coefficient* are also used in some fields. However, they are identical in generally taking the ratio of Intersection over Union. The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets:
# 
# $$ J(A, B) := \frac{\left\vert A \cap B \right\vert}{\left\vert A \cup B \right\vert} = \frac{\left\vert A \cap B \right\vert}{\left\vert A \right\vert + \left\vert B \right\vert - \left\vert A \cap B \right\vert}. $$
# 
# Note that by design, $0\leq J\left(A, B\right)\leq 1$. If $A$ and $B$ are both empty, define $J(A, B) = 1$. The Jaccard coefficient is widely used in computer science, ecology, genomics, and other sciences, where binary or binarized data are used. Both the exact solution and approximation methods are available for hypothesis testing with the Jaccard coefficient. See [this paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3118-5) ([arxiv version](https://arxiv.org/abs/1903.11372)) for details.
# 
# Let us assume that for a specific `id` $a$, our algorithm produces three matches $a$, $b$ and $c$ whereas the true matches are $a$, $b$, $d$ and $e$. Then the Jaccard index for the prediction on this particular `id` will be
# 
# $$ \frac{\left\vert \left\{a, b, c\right\} \cap \left\{a, b, d, e\right\} \right\vert}{\left\vert \left\{a, b, c\right\} \cup \left\{a, b, d, e\right\} \right\vert} = \frac{\left\vert \left\{a, b\right\} \right\vert}{\left\vert \left\{a, b, c, d, e\right\} \right\vert} = \frac{2}{5}. $$
# 
# Thus, while correct matching predictions are rewarded, incorrect matching predictions are penalised by equal measure. The evaluation metric is simply the mean of Jaccard indices for each of the test observations, i.e. if the test data comprises $n_{\text{test}}$ observations and $J_i$ denotes the Jaccard index corresponding to the $i$th test observation, $i = 1,2,\cdots,n_{\text{test}}$, then the final metric by which a model will be evaluated is:
# 
# $$ \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} J_i. $$

# **Note:** In this notebook, we shall rely solely on the provided pairs set `data_pairs` to build the baseline models.

# # 2. Train-Test Split

# We split `data_pairs` into two parts:
# - `data_pairs_train`: The portion of data that we use to *train* the models
# - `data_pairs_test`: The portion of data that we use to *test* or evaluate the models
# 
# We shall keep `data_test`, which has only five observations, separate from the modeling procedure.

# In[9]:


# Splitting data_train
data_pairs_train, data_pairs_test = train_test_split(data_pairs, test_size = 0.2, random_state = 40)

labels = ['Train','Test']
values = [len(data_pairs_train), len(data_pairs_test)]
fig_data = [go.Pie(values = values, labels = labels, hole = 0.0, textinfo = 'label+percent')]
fig_title = dict(text = "Train-test split", x = 0.5, y = 0.95)
fig = go.Figure(data = fig_data)
fig.update_layout(height = 500, width = 800, showlegend = False, title = fig_title)
fig.show()


# In[10]:


# Donutplots of the 'match' column
fig = make_subplots(rows = 1, cols = 2, specs = [[{'type': 'domain'}, {'type': 'domain'}]])
x_val_train = data_pairs_train['match'].value_counts(sort = False).index.tolist()
y_val_train = data_pairs_train['match'].value_counts(sort = False).tolist()
x_val_test = data_pairs_test['match'].value_counts(sort = False).index.tolist()
y_val_test = data_pairs_test['match'].value_counts(sort = False).tolist()
fig.add_trace(go.Pie(values = y_val_train, labels = x_val_train, hole = 0.5, textinfo = 'label+percent', title = "Train"), row = 1, col = 1)
fig.add_trace(go.Pie(values = y_val_test, labels = x_val_test, hole = 0.5, textinfo = 'label+percent', title = "Test"), row = 1, col = 2)
fig.update_layout(height = 500, width = 800, showlegend = False, xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1), title = dict(text = "Frequency comparison of 'match'", x = 0.5, y = 0.95)) 
fig.show()


# We observe that in both the `data_pairs_train` and the `data_pairs_test`, the count of pairs that match is higher than the count of pairs that do not match. However, the imbalance is not too big.

# # 3. Data Preprocessing
# 
# - [Decoding States Abbreviations](#Decoding-States-Abbreviations)
# - [Conversion to Lowercase](#Conversion-to-Lowercase)
# - [Missing Data Imputation](#Missing-Data-Imputation)

# ## Decoding States Abbreviations

# In[11]:


# Dictionary of US states abbreviations and names
url_abbrev_to_name = "https://raw.githubusercontent.com/sugatagh/Foursquare-Location-Matching/main/JSON/US_states_abbrev_to_name.json"
dict_abbrev_to_name = pd.read_json(url_abbrev_to_name, typ = 'series')
dict_abbrev_to_name


# In[12]:


# Converting US states abbreviations to names
data_pairs_train.replace({'state_1': dict_abbrev_to_name}, inplace = True)
data_pairs_train.replace({'state_2': dict_abbrev_to_name}, inplace = True)
data_pairs_test.replace({'state_1': dict_abbrev_to_name}, inplace = True)
data_pairs_test.replace({'state_2': dict_abbrev_to_name}, inplace = True)


# ## Conversion to Lowercase

# We convert all alphabetical characters in the relevant object-type columns to lowercase so that the models do not differentiate identical words due to [case sensitivity](https://en.wikipedia.org/wiki/Case_sensitivity).

# In[13]:


# Converting to lowercase
def convert_to_lowercase_skipna(x):
    """
    Converts a given string to lowercase
    Arg:
      x (string/NaN): input string (possibly NaN due to missing value)
    Returns:
      y (string/NaN): x.lower() if x is not NaN, x otherwise
    """
    if str(x) == 'nan':
        y = x
    else:
        y = x.lower()
    return y

text = "Mobile Phone Shops"
print("Input: {}".format(text))
print("Output: {}".format(convert_to_lowercase_skipna(text)))


# Before applying this conversion, we have to ensure that the argument is of `string` type. However, we do not want to convert the `nan` values to the string `'nan'`, as the converted string will no longer be identified as a missing value, and hence will go unaltered in the *missing value imputation* step.

# In[14]:


# Converting to string, unless the argument is nan
def convert_to_string_skipna(x):
    """
    Converts an input to string if it is not NaN, otherwise it is left unaltered
    Arg:
      x (any python data type): input to be converted (possibly NaN due to missing value)
    Returns:
      y (str/NaN): str(x) if x is not NaN, x otherwise
    """
    if str(x) == 'nan':
        y = x
    else:
        y = str(x)
    return y

number = 40
print("Input: {}".format(number))
print("Output:")
convert_to_string_skipna(number)


# We convert the columns `name`, `address`, `city`, `state` and `categories`. We leave the column `url` as it is, for its case sensitivity.

# In[15]:


# Applying the conversion to object columns
cols_lower = ['name', 'address', 'city', 'state', 'categories']
cols_lower_pairs = ['name_1', 'address_1', 'city_1', 'state_1', 'categories_1',
                    'name_2', 'address_2', 'city_2', 'state_2', 'categories_2']
for col in cols_lower:
    data_test[col] = data_test[col].apply(convert_to_string_skipna).apply(convert_to_lowercase_skipna)
for col in cols_lower_pairs:
    data_pairs_train[col] = data_pairs_train[col].apply(convert_to_string_skipna).apply(convert_to_lowercase_skipna)
    data_pairs_test[col] = data_pairs_test[col].apply(convert_to_string_skipna).apply(convert_to_lowercase_skipna)


# ## Missing Data Imputation

# In[16]:


# Columns with missing values in the training set with respective proportion of missing values
(data_pairs_train.isna().sum()[data_pairs_train.isna().sum() != 0] / len(data_pairs_train)).sort_values(ascending = False)


# In[17]:


# Missing values in the training set
plt.figure(figsize = (9, 6))
df_temp = data_pairs_train.isna().sum() * 100 / len(data_pairs_train)
s = sns.barplot(x = df_temp.values, y = df_temp.index)
s.set_xlim(0, 100)
s.set_xlabel("% of missing values", fontsize = 14)
s.set_ylabel("column", fontsize = 14)
plt.axvline(x = 30)
plt.axvline(x = 50)
plt.tight_layout()
plt.show()


# Seven columns `zip_1`, `url_1`, `phone_1`, `address_2`, `zip_2`, `url_2` and `phone_2` have over $30\%$ values missing. We shall drop these columns from the subsequent analysis.

# #### Dropping columns with more than 30% missing values

# Even though `address_1` has less than $30\%$ missing values, `address_2` has a very high proportion of it. For a comparison purpose, `address_1` cannot contribute anything alone because *one hand cannot clap*. Thus we drop it alongside all the columns with more than $30\%$ missing values.

# In[18]:


# Dropping columns with more than 30% missing values in the training set
cols_drop = ['address', 'zip', 'url', 'phone']
cols_drop_pairs = ['address_1', 'zip_1', 'url_1', 'phone_1', 'address_2', 'zip_2', 'url_2', 'phone_2']
data_test.drop(cols_drop, axis = 1, inplace = True)
data_pairs_train.drop(cols_drop_pairs, axis = 1, inplace = True)
data_pairs_test.drop(cols_drop_pairs, axis = 1, inplace = True)


# #### Imputing missing values with 'unknown'

# In[19]:


# Imputing missing names with 'unknown'
cols_unknown = ['name', 'city', 'state', 'country']
for col in cols_unknown:
    data_test[col].fillna('unknown', inplace = True)
cols_unknown_pairs = ['name_1', 'city_1', 'state_1', 'country_1', 'name_2', 'city_2', 'state_2', 'country_2']
for col in cols_unknown_pairs:
    data_pairs_train[col].fillna('unknown', inplace = True)
    data_pairs_test[col].fillna('unknown', inplace = True)


# #### Mode imputation for categories

# In[20]:


# Mode imputation for categories
data_test['categories'].fillna(data_test['categories'].mode()[0], inplace = True)
data_pairs_train['categories_1'].fillna(data_pairs_train['categories_1'].mode()[0], inplace = True)
data_pairs_train['categories_2'].fillna(data_pairs_train['categories_2'].mode()[0], inplace = True)
data_pairs_test['categories_1'].fillna(data_pairs_test['categories_1'].mode()[0], inplace = True)
data_pairs_test['categories_2'].fillna(data_pairs_test['categories_2'].mode()[0], inplace = True)


# In[21]:


# Count of missing values in the 'data_pairs_train'
data_pairs_train.isna().sum()


# In[22]:


# Count of missing values in the 'data_pairs_test'
data_pairs_test.isna().sum()


# In[23]:


# Count of missing values in the true test set
data_test.isna().sum()


# # 4. Feature Engineering
# 
# - [Distance between Locations](#Distance-between-Locations)
# - [Features Based on Largest Common Substring](#Features-Based-on-Largest-Common-Substring)
# - [Matching of Countries](#Matching-of-Countries)
# - [The Target Variable](#The-Target-Variable)

# Each row in `data_pairs_train` or `data_pairs_test` consists of two training observations and a `match` variable which indicates whether the two observations match or not. As we have seen in the previous section, an observation in the pairs set can be represented as:
# 
# $$ \left(a_1,a_2,\ldots,a_n; b_1,b_2,\ldots,b_n; y\right), $$
# 
# where $a_1,a_2,\ldots,a_n$ is the observed features of the first observation, $b_1,b_2,\ldots,b_n$ is the same for the second observation and $y$ is the `match` variable. Now, suppose that $\left(c_1,c_2,\ldots,c_n\right)$ and $\left(d_1,d_2,\ldots,d_n\right)$ are two observations from the `test set`. To predict whether these two observations refer to the same POI or not, we have to predict the corresponding `match` variable $y$. The general idea in this section is to construct a vector of features $\left(z_1,z_2,\ldots,z_m\right)$, with $m \leq n$, which captures the key information about $y$, contained in $\left(c_1,c_2,\ldots,c_n; d_1,d_2,\ldots,d_n\right)$. The goal of the current section is to produce a function that takes in a pair of observations, in the form of $\left(a_1,a_2,\ldots,a_n; b_1,b_2,\ldots,b_n\right)$, as an input and produce $\left(z_1,z_2,\ldots,z_m\right)$ as an output, i.e.
# 
# $$ \left(a_1,a_2,\ldots,a_n; b_1,b_2,\ldots,b_n\right) \mapsto \left(z_1,z_2,\ldots,z_m\right). $$

# In[24]:


# Typical observation from the 'data_pairs_train'
data_pairs_train.iloc[0]


# It contains a pair of observations. The *id* of the two observations are expectedly different. The *name* is slightly different, as are the *latitude* and *longitude*. *Country* and *category* are identical. Some of the attributes are missing in one of the observations, while some are missing in both. The target variable here is `match`, which is a Boolean variable taking the value `True` if the two observations refer to the same POI and `False` otherwise. We observe that the number of features can be greatly reduced if we focus on the information that are relevant in predicting `match`, and discard the rest. We initiate a dataframe to extract and store these relevant information out of the attributes in `data_pairs`.

# In[25]:


# Dataframe initialization for new features
df_train = pd.DataFrame()
df_test = pd.DataFrame()


# ## Distance between Locations

# The information that is relevant in predicting `match`, contained in `latitude_1`, `longitude_1`, `latitude_2` and `longitude_2`, can be encapsulted into a single variable, which is the distance `dist_loc` between the two locations (`latitude_1`, `longitude_1`) and (`latitude_2`, `longitude_2`), given by the haversine formula. Let $\left(\phi_1, \lambda_1\right)$ and $\left(\phi_2, \lambda_2\right)$ be the coordinates of two POIs. Then the distance between the two POIs is
# 
# $$ d = 2r \arcsin\left(\sqrt{\text{hav}\left(\phi_1 - \phi_2\right) + \text{cos}\left(\phi_1\right) \text{cos}\left(\phi_2\right) \text{hav}\left(\lambda_1 - \lambda_2\right)}\right), $$
# 
# where
# 
# $$ \text{hav}\left(\theta\right) = \sin^2\left(\frac{\theta}{2}\right) = \frac{1-\cos{\theta}}{2}. $$

# In[26]:


# Haversine formula
def dist(lat1, lon1, lat2, lon2):
    """
    Computes the distance (over the surface) between two coordinates (lat1, lon1) and (lat2, lon2)
    Args:
      lat1 (float): latitude of first location
      lon1 (float): longitude of first location
      lat2 (float): latitude of second location
      lon2 (float): longitude of second location
    Returns:
      d (float): distance between the two locations in km
    """
    r = 6371
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat1 - lat2, lon1 - lon2
    h = ((1 - cos(dlat)) / 2) + (cos(lat1) * cos(lat2) * ((1 - cos(dlon)) / 2))
    d = 2 * r * asin(sqrt(h))
    return d

lat1, lon1, lat2, lon2 = 5.012169, 100.535805, 40.434209, -80.564160
print(f"Input: ({lat1}, {lon1}) and ({lat2}, {lon2})")
print(f"Output: {dist(lat1, lon1, lat2, lon2)} km")


# In[27]:


# Distance between locations
dist_loc_train = [dist(data_pairs_train['latitude_1'][i], data_pairs_train['longitude_1'][i], data_pairs_train['latitude_2'][i], data_pairs_train['longitude_2'][i]) for i in data_pairs_train.index]
dist_loc_test = [dist(data_pairs_test['latitude_1'][i], data_pairs_test['longitude_1'][i], data_pairs_test['latitude_2'][i], data_pairs_test['longitude_2'][i]) for i in data_pairs_test.index]
df_train['dist_loc'] = dist_loc_train
df_test['dist_loc'] = dist_loc_test


# In[28]:


# Histogram of 'dist_loc' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'dist_loc', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'dist_loc', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In both `df_train` and `df_test`, `dist_loc` is concentrated near $0$. It is likely that in both `df_train` and `df_test`, `dist_loc` is positively skewed to an extreme degree. To elaborate, [Skewness](https://en.wikipedia.org/wiki/Skewness) quantifies the asymmetry of a distribution about its mean. It is given by
# 
# $$ g_1 := \frac{\frac{1}{n}\sum_{i=1}^n\left(x_i-\bar{x}\right)^3}{\left[\frac{1}{n}\sum_{i=1}^n\left(x_i-\bar{x}\right)^2\right]^{3/2}}, $$
# 
# where $\bar{x}$ is the mean of the observations, given by $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$. The measure $g_1$ can be negative, zero, positive. A value close to $0$ suggests that the distribution is more or less symmetric. However, as it deviates from $0$, it becomes more and more skewed (either positively or negatively). A positive skewness indicates that the distribution is concentrated towards the left side, with the longer tail being on the right side. A negative skewness indicates that the distribution is concentrated towards the right side, with the longer tail being on the left side. We back up the observation about skewness of `dist_loc` in `df_train` and `df_test` by computing the corresponding $g_1$ values.

# In[29]:


# Skewness of 'dist_loc'
print(pd.Series({"Skewness of 'dist_loc' in 'df_train'": df_train['dist_loc'].skew(),
                 "Skewness of 'dist_loc' in 'df_test'": df_test['dist_loc'].skew()}).to_string())


# To deal with the extreme skewness, we have applied the following transformation:
# 
# $$ x \mapsto log(x+\epsilon), $$
# 
# where $\epsilon$ is a very small positive real number. Here we have taken $\epsilon = 0.00000001$. The reason behind making this small shift to the data is that the log function maps $0$ to $-\infty$. The shift keeps the transformed data finite, and keeping $\epsilon$ small ensures that the data points which were originally $0$, stands out from the rest in the transformed setup. Visualizations of the distribution of both the original feature and the transformed feature have been shown.

# In[30]:


# Applying the transformation
epsilon = 0.00000001
df_train['dist_loc_transformed'] = df_train['dist_loc'].apply(lambda x: np.log(x + epsilon))
df_test['dist_loc_transformed'] = df_test['dist_loc'].apply(lambda x: np.log(x + epsilon))


# In[31]:


# Deleting old features
df_train.drop('dist_loc', axis = 1, inplace = True)
df_test.drop('dist_loc', axis = 1, inplace = True)


# In[32]:


# Histogram of 'dist_loc_transformed' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'dist_loc_transformed', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'dist_loc_transformed', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In[33]:


# MinMax normalization of 'dist_loc_transformed'
loc_train, loc_test = df_train['dist_loc_transformed'], df_test['dist_loc_transformed']
df_train['dist_loc_transformed'] = (loc_train - loc_train.min()) / (loc_train.max() - loc_train.min())
df_test['dist_loc_transformed'] = (loc_test - loc_test.min()) / (loc_test.max() - loc_test.min())


# ## Features Based on Largest Common Substring

# We observe that, in `data_pairs_train` and `data_pairs_test`, the name for the same POI varies in different records for variety of reasons. For example, some records may contain shortened versions of the names. Similar phenomenon is observed in the city, state and categories attribute. For these reasons, we create similarity features based on these attributes using the length of the largest common substring of two strings, relative to the length of the shorter string. Note that we have to convert some of the input data to string format before feeding it to the function that computes this quantity.

# In[34]:


# Relative length proportion of largest common substring of two strings
def lcss(str1, str2):
    """
    Computes length proportion of largest common substring of two strings, relative to the length of the shorter string
    Args:
      str1 (string): a general string
      str2 (string): a general string
    Returns:
      prop (float): length of the largest common substring of two strings, scaled by length of the shorter string
    """
    n1, n2 = len(str1), len(str2)
    lc = [[0 for k in range(n2 + 1)] for l in range(n1 + 1)]
    out = 0
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            if (i == 0 or j == 0):
                lc[i][j] = 0
            elif (str1[i-1] == str2[j-1]):
                lc[i][j] = lc[i-1][j-1] + 1
                out = max(out, lc[i][j])
            else:
                lc[i][j] = 0
    prop = out / min(n1, n2)
    return prop

str1, str2 = '118th street bus stop', '118th street beach'
print(f"Input: '{str1}' and '{str2}'")
print(f"Output: {lcss(str1, str2)}")


# In[35]:


# Relative length of largest common substring between names, cities, states and categories
lcss_name_train, lcss_name_test = [], []
lcss_city_train, lcss_city_test = [], []
lcss_state_train, lcss_state_test = [], []
lcss_categories_train, lcss_categories_test = [], []
for i in data_pairs_train.index:
    lcss_name_train.append(lcss(data_pairs_train['name_1'][i], data_pairs_train['name_2'][i]))
    lcss_city_train.append(lcss(data_pairs_train['city_1'][i], data_pairs_train['city_2'][i]))
    lcss_state_train.append(lcss(data_pairs_train['state_1'][i], data_pairs_train['state_2'][i]))
    lcss_categories_train.append(lcss(data_pairs_train['categories_1'][i], data_pairs_train['categories_2'][i]))
for i in data_pairs_test.index:
    lcss_name_test.append(lcss(data_pairs_test['name_1'][i], data_pairs_test['name_2'][i]))
    lcss_city_test.append(lcss(data_pairs_test['city_1'][i], data_pairs_test['city_2'][i]))
    lcss_state_test.append(lcss(data_pairs_test['state_1'][i], data_pairs_test['state_2'][i]))
    lcss_categories_test.append(lcss(data_pairs_test['categories_1'][i], data_pairs_test['categories_2'][i]))
df_train['lcss_name'], df_test['lcss_name'] = lcss_name_train, lcss_name_test
df_train['lcss_city'], df_test['lcss_city'] = lcss_city_train, lcss_city_test
df_train['lcss_state'], df_test['lcss_state'] = lcss_state_train, lcss_state_test
df_train['lcss_categories'], df_test['lcss_categories'] = lcss_categories_train, lcss_categories_test


# In[36]:


# Histogram of 'lcss_name' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'lcss_name', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'lcss_name', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In[37]:


# Histogram of 'lcss_city' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'lcss_city', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'lcss_city', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In[38]:


# Histogram of 'lcss_state' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'lcss_state', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'lcss_state', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# In[39]:


# Histogram of 'lcss_categories' for 'df_train' and 'df_test'
fig, ax = plt.subplots(1, 2, figsize = (15, 6), sharey = True)
sns.histplot(data = df_train, x = 'lcss_categories', bins = 30, ax = ax[0])
sns.histplot(data = df_test, x = 'lcss_categories', bins = 30, ax = ax[1])
ax[0].set_title("df_train", fontsize = 14)
ax[1].set_title("df_test", fontsize = 14)
ax[1].set_ylabel("")
plt.tight_layout()
plt.show()


# ## Matching of Countries

# In[40]:


# Matching of countries
df_train['match_country'] = [float(data_pairs_train['country_1'][i] == data_pairs_train['country_2'][i]) for i in data_pairs_train.index]
df_test['match_country'] = [float(data_pairs_test['country_1'][i] == data_pairs_test['country_2'][i]) for i in data_pairs_test.index]


# In[41]:


# Donutplots of the 'match_country' column
fig = make_subplots(rows = 1, cols = 2, specs = [[{'type': 'domain'}, {'type': 'domain'}]])
x_val_train = df_train['match_country'].value_counts(sort = False).index.tolist()
y_val_train = df_train['match_country'].value_counts(sort = False).tolist()
x_val_test = df_test['match_country'].value_counts(sort = False).index.tolist()
y_val_test = df_test['match_country'].value_counts(sort = False).tolist()
fig.add_trace(go.Pie(values = y_val_train, labels = x_val_train, hole = 0.5, textinfo = 'label+percent', title = "Train"), row = 1, col = 1)
fig.add_trace(go.Pie(values = y_val_test, labels = x_val_test, hole = 0.5, textinfo = 'label+percent', title = "Test"), row = 1, col = 2)
fig.update_layout(height = 500, width = 800, showlegend = False, xaxis = dict(tickmode = 'linear', tick0 = 0, dtick = 1), title = dict(text = "Frequency comparison of 'match_country'", x = 0.5, y = 0.95)) 
fig.show()


# ## The Target Variable

# In[42]:


# 'match' column
df_train['match'] = [float(data_pairs_train['match'][i]) for i in data_pairs_train.index]
df_test['match'] = [float(data_pairs_test['match'][i]) for i in data_pairs_test.index]


# From this point on, we shall refer to `df_train` as the *effective training set*, or just `training set` and `df_test` as the *effective test set*, or just `test set`. We shall refer to the five-observations-strong `data_test` as the *true test set*.

# In[43]:


# Effective training set
print(pd.Series({"Memory usage": "{:.2f} MB".format(df_train.memory_usage().sum()/(1024*1024)),
                 "Dataframe shape": "{}".format(df_train.shape)}).to_string())
print(" ")
df_train.head()


# In[44]:


# Effective test set
print(pd.Series({"Memory usage": "{:.2f} MB".format(df_test.memory_usage().sum()/(1024*1024)),
                 "Dataframe shape": "{}".format(df_test.shape)}).to_string())
print(" ")
df_test.head()


# # 5. Baseline XGBoost

# Each row in `df_train` or `df_test` are of the form $\left(z_1,z_2,\ldots,z_m\right)$, which is essentially a vector representing the differences between two POI observations. The task of modeling is to predict the `match` variable $y$ based on $\left(z_1,z_2,\ldots,z_m\right)$, i.e.
# 
# $$ \left(z_1,z_2,\ldots,z_m\right) \mapsto \hat{y} \,\,(\text{estimation of } y). $$
# 
# The plan is to train a baseline XGBoost classifier on `df_train`, and evaluate it on `df_test`. First, we separate out the target variable from the feature variables.

# In[45]:


# Feature-target split
X_train, y_train = df_train.drop('match', axis = 1), df_train['match']
X_test, y_test = df_test.drop('match', axis = 1), df_test['match']


# Next, we construct functions to compute and display the *confusion matrix*, given the true labels and the predicted labels of the target.

# In[46]:


# Function to compute confusion matrix
def conf_mat(y_test, y_pred):
    """
    Computes confusion matrix
    Args:
      y_test (list/array/tuple/set/series): true binary (0/1) labels
      y_pred (list/array/tuple/set/series): predicted binary (0/1) labels
    Returns:
      confusion_mat (array): A 2D array representing a 2x2 confusion matrix
    """
    y_test, y_pred = list(y_test), list(y_pred)
    count, labels, confusion_mat = len(y_test), [0, 1], np.zeros(shape = (2, 2), dtype = int)
    for i in range(2):
        for j in range(2):
            confusion_mat[i][j] = len([k for k in range(count) if y_test[k] == labels[i] and y_pred[k] == labels[j]])
    return confusion_mat


# In[47]:


# Function to print confusion matrix
def conf_mat_heatmap(y_test, y_pred):
    """
    Prints confusion matrix
    Args:
      y_test (list/array/tuple/set/series): true binary (0/1) labels
      y_pred (list/array/tuple/set/series): predicted binary (0/1) labels
    Returns:
      Nothing, prints a heatmap representing a 2x2 confusion matrix
    """
    confusion_mat = conf_mat(y_test, y_pred)
    labels, confusion_mat_df = [0, 1], pd.DataFrame(confusion_mat, range(2), range(2))
    plt.figure(figsize = (6, 4.75))
    # sns.set(font_scale = 1.4) # label size
    sns.heatmap(confusion_mat_df, annot = True, annot_kws = {"size": 16}, fmt = 'd')
    plt.xticks([0.5, 1.5], labels, rotation = 'horizontal')
    plt.yticks([0.5, 1.5], labels, rotation = 'horizontal')
    plt.xlabel("Predicted label", fontsize = 14)
    plt.ylabel("True label", fontsize = 14)
    plt.title("Confusion Matrix", fontsize = 14)
    plt.grid(False)
    plt.show()


# We shall use the *accuracy* measure as an evaluation metric to assess the models. The measure is given by
# 
# $$ \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Number of total predictions}}. $$

# In[48]:


# Function to compute accuracy
def accuracy(y_test, y_pred):
    """
    Computes accuracy, given true and predicted binary (0/1) labels
    Args:
      y_test (list/array/tuple/set/series): true binary (0/1) labels
      y_pred (list/array/tuple/set/series): predicted binary (0/1) labels
    Returns:
      acc (float): accuracy obtained from y_test and y_pred
    """
    confusion_mat = conf_mat(y_test, y_pred)
    num = confusion_mat[0, 0] + confusion_mat[1, 1]
    denom = num + confusion_mat[0, 1] + confusion_mat[1, 0]
    acc = num / denom
    return acc


# Now, we consider the XGBoost classifier, fit the model on the training set, predict and evaluate on both the training set and the test set, and report the respective accuracy scores. Furthermore, we print the confusion matrix for the predictions on observations in the test set.

# In[50]:


# Model
xgb = XGBClassifier()
print(f"Model: {xgb}")

# Fitting the model on the training set
xgb.fit(X_train, y_train)

# Prediction and evaluation on the training set
y_train_pred = xgb.predict(X_train)
score_train = accuracy(y_train, y_train_pred)
print(f"Training accuracy: {score_train}")

# Prediction and evaluation on the test set
y_test_pred = xgb.predict(X_test)
score_test = accuracy(y_test, y_test_pred)
print(f"Test accuracy: {score_test}")

# Confusion matrix for prediction on the test set
conf_mat_heatmap(y_test, y_test_pred)


# # 6. Hyperparameter Tuning

# In[55]:


get_ipython().run_cell_magic('time', '', "# Hyperparameter tuning for XGBoost\ncv = KFold(n_splits = 6, shuffle = True, random_state = 40).split(X = X_train, y = y_train)\nxgb = XGBClassifier()\nparams = {\n    'subsample': [0.5, 1],\n    'max_depth': [6],\n    'learning_rate': [0.01, 0.2, 0.3],\n    'gamma': [0, 0.5, 1, 2],\n    'reg_alpha': [0, 0.5, 1]\n}\ngsearch_xgb = GridSearchCV(estimator = xgb, param_grid = params, scoring = 'accuracy', n_jobs = -1, cv = cv, verbose = 1)\ngsearch_xgb_fit = gsearch_xgb.fit(X = X_train, y = y_train)\n")


# In[56]:


# Best parameters and score
print(f"Best parameters: {gsearch_xgb.best_params_}")
print(f"Best cross-validation accuracy: {gsearch_xgb.best_score_}")


# In[57]:


# Model
xgb_best = gsearch_xgb.best_estimator_
print(f"Model: {xgb_best}")

# Fitting the model on the training set
xgb_best.fit(X_train, y_train)

# Prediction and evaluation on the training set
y_train_pred_tuning = xgb_best.predict(X_train)
score_train = accuracy(y_train, y_train_pred_tuning)
print(f"Training accuracy: {score_train}")

# Prediction and evaluation on the test set
y_test_pred_tuning = xgb_best.predict(X_test)
score_test = accuracy(y_test, y_test_pred_tuning)
print(f"Test accuracy: {score_test}")

# Confusion matrix for prediction on the test set
conf_mat_heatmap(y_test, y_test_pred_tuning)


# - Hyperparameter tuning very slightly increases the test accuracy
# - It improves performance of the model over the positive class (observations with true label $1$)
# - On the flip side, it slightly worsens performance of the model over the negative class (observations with true label $0$)

# # Acknowledgements
# 
# - [Foursquare - Location Matching](https://www.kaggle.com/competitions/foursquare-location-matching) competition

# # References
# 
# - [Case sensitivity](https://en.wikipedia.org/wiki/Case_sensitivity)
# - [Correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
# - [Foursquare City Guide](https://en.wikipedia.org/wiki/Foursquare_City_Guide)
# - [Foursquare Labs Inc.](https://foursquare.com/)
# - [Foursquare Swarm](https://en.wikipedia.org/wiki/Foursquare_Swarm)
# - [Grove Karl Gilbert](https://en.wikipedia.org/wiki/Grove_Karl_Gilbert)
# - [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula)
# - [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
# - [ISO 3166-1 alpha-2](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
# - [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
# - [Jaccard/Tanimoto similarity test and estimation methods for biological presence-absence data](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3118-5)
# - [Jaccard/Tanimoto similarity test and estimation methods (arxiv version)](https://arxiv.org/abs/1903.11372)
# - [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
# - [List of U.S. state and territory abbreviations](https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations)
# - [Olympus Mons](https://en.wikipedia.org/wiki/Olympus_Mons)
# - [Paul Jaccard](https://en.wikipedia.org/wiki/Paul_Jaccard)
# - [Point of Interest](https://en.wikipedia.org/wiki/Point_of_interest)
# - [Skewness](https://en.wikipedia.org/wiki/Skewness)
# - [Vladimir Levenshtein](https://en.wikipedia.org/wiki/Vladimir_Levenshtein)
# - [XGBoost](https://en.wikipedia.org/wiki/XGBoost)

# In[58]:


# Runtime and memory usage
stop = time.time()
print(pd.Series({"Process runtime": "{:.2f} seconds".format(float(stop - start)),
                 "Process memory usage": "{:.2f} MB".format(float(process.memory_info()[0]/(1024*1024)))}).to_string())

