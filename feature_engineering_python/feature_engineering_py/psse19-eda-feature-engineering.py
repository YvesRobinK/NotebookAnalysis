#!/usr/bin/env python
# coding: utf-8

# # Sales Forecasting EDA and Feature Engineering

# ## Imports

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e19/train.csv')
train.head()


# In[3]:


test = pd.read_csv('/kaggle/input/playground-series-s3e19/test.csv')
test.head()


# ## Exploratory Data Analysis

# In[4]:


train.info()


# In[5]:


# Check data statistics
train.describe()


# In[6]:


# Check for missing values
train.isnull().sum()


# In[7]:


# Visualize the data
# Plot distribution of num_sold
sns.histplot(train['num_sold'])
plt.title('Distribution of Number of Sales')
plt.show()


# In[8]:


# Plot sales over time
train['date'] = pd.to_datetime(train['date'])
train['year'] = train['date'].dt.year
train['month'] = train['date'].dt.month

sales_over_time = train.groupby(['year', 'month'])['num_sold'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=sales_over_time, x='month', y='num_sold', hue='year')
plt.title('Sales Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Sales')
plt.show()


# In[9]:


# Convert the 'date' column to datetime format if it's not already in that format
train['date'] = pd.to_datetime(train['date'])


# In[10]:


# Extract week, month, and quarter from the 'date' column
train['week'] = train['date'].dt.week
train['month'] = train['date'].dt.month
train['quarter'] = train['date'].dt.quarter
train['day_of_week'] = train['date'].dt.dayofweek


# In[11]:


# Create separate subplots for week, month, quarter, and day of the week boxplots
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 18))

# Plot boxplots for sales per week
sns.boxplot(data=train, x='week', y='num_sold', ax=axes[0])
axes[0].set_title('Sales Distribution per Week')
axes[0].set_xlabel('Week')
axes[0].set_ylabel('Number of Sales')

# Plot boxplots for sales per month
sns.boxplot(data=train, x='month', y='num_sold', ax=axes[1])
axes[1].set_title('Sales Distribution per Month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of Sales')

# Plot boxplots for sales per quarter
sns.boxplot(data=train, x='quarter', y='num_sold', ax=axes[2])
axes[2].set_title('Sales Distribution per Quarter')
axes[2].set_xlabel('Quarter')
axes[2].set_ylabel('Number of Sales')

# Plot boxplots for sales per day of the week
sns.boxplot(data=train, x='day_of_week', y='num_sold', ax=axes[3])
axes[3].set_title('Sales Distribution per Day of the Week')
axes[3].set_xlabel('Day of the Week')
axes[3].set_ylabel('Number of Sales')

# Adjust the layout
plt.tight_layout()

# Display the plots
plt.show()


# In[12]:


import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming 'date' is in datetime format and 'num_sold' is the target variable
date = train['date']
num_sold = train['num_sold']

# Perform seasonal decomposition
decomposition = seasonal_decompose(num_sold, period=365)  # Assuming a seasonal period of 12 months (yearly seasonality)
seasonal = decomposition.seasonal



# In[13]:


import statsmodels.api as sm

# Run seasonal decompose
decomp = sm.tsa.seasonal_decompose(train['num_sold'], period=365) 

print(decomp.seasonal.head()) # checking the seasonal component
_ = decomp.plot()



# In[14]:


# Plot the seasonal component
plt.figure(figsize=(10, 6))
plt.plot(date, seasonal)
plt.title('Seasonality of Sales')
plt.xlabel('Date')
plt.ylabel('Seasonal Component')
plt.show()


# ## Feature Engineering

# ### Features from the 'product' column

# In[15]:


# Extract new features from the 'product' column
train['product_length'] = train['product'].apply(lambda x: len(x))  # Length of the product name
train['num_words'] = train['product'].apply(lambda x: len(x.split()))  # Number of words in the product name
train['contains_llm'] = train['product'].apply(lambda x: 'LLMs' in x)  # Boolean flag indicating if 'LLMs' is present in the product name
train['contains_kaggle'] = train['product'].apply(lambda x: 'Kaggle' in x)  # Boolean flag indicating if 'LLMs' is present in the product name
train['contains_win'] = train['product'].apply(lambda x: 'Win' in x)  # Boolean flag indicating if 'LLMs' is present in the product name

# Display the updated DataFrame
train.head()


# ### One-hot encoding on categorical features

# In[16]:


# Perform one-hot encoding for 'country' and 'store'
train_encoded = pd.get_dummies(train, columns=['country', 'store'])

# Display the updated DataFrame
train_encoded.head()


# ### Weekday vs Weekend feature

# In[17]:


# Assuming 'date' is in datetime format
train['weekday'] = train['date'].dt.weekday  # Get the weekday as an integer (Monday=0, Sunday=6)

# Create a feature indicating whether it's a weekend or weekday
train['is_weekend'] = train['weekday'].isin([5, 6]).astype(int)  # 5 and 6 represent Saturday and Sunday

# Display the updated DataFrame
train.head()


# ### Local Holiday feature

# In[18]:


import holidays

# Create a function to check if a given date is a holiday in a specific country
def is_holiday(date, country):
    # Initialize the holiday object for the specified country
    holiday_obj = holidays.CountryHoliday(country)

    # Check if the date is a holiday
    if date in holiday_obj:
        return 1  # It's a holiday
    else:
        return 0  # It's not a holiday


# In[19]:


# Create the 'is_holiday' feature based on the 'date' and 'country' columns
train['is_holiday'] = train.apply(lambda row: is_holiday(row['date'], row['country']), axis=1)

# Display the updated DataFrame
train.head()


# ### Lag Factors

# In[20]:


train['num_sold_lag_1'] = train['num_sold'].shift(1)  # Lagged value for the previous day
train['num_sold_lag_7'] = train['num_sold'].shift(7)  # Lagged value for the previous week


# In[21]:


train['num_sold_roll_mean'] = train['num_sold'].rolling(window=7).mean()  # Rolling mean over 7 days
train['num_sold_roll_std'] = train['num_sold'].rolling(window=7).std()  # Rolling standard deviation over 7 days


# In[22]:


train['num_sold_lag_same_week'] = train.groupby(train['date'].dt.week)['num_sold'].shift()  # Lagged value for the same week in previous years


# In[23]:


from sklearn.model_selection import train_test_split

# Convert the 'date' column to datetime format if it's not already in that format
train['date'] = pd.to_datetime(train['date'])

# Set the splitting point as January 1, 2020
split_date = pd.to_datetime('2020-01-01')

# Split the data into training and testing sets
train_data = train[train['date'] < split_date]
test_data = train[train['date'] >= split_date]

# Display the shapes of the training and testing data
print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)


# ## SARIMAX Model

# In[24]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

# Assuming 'num_sold' is the target variable
target = 'num_sold'

# Create the SARIMAX model
model = SARIMAX(train_data[target], order=(1, 0, 1), seasonal_order=(1, 1, 1, 12))

# Fit the model to the training data
model_fit = model.fit()

# Make predictions for the test data
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1])

# Display the predicted values
print(predictions)


# In[25]:


import numpy as np

def smape(actual, predicted):
    """
    Calculate SMAPE (Symmetric Mean Absolute Percentage Error) between actual and predicted values.
    SMAPE = 0 when actual and predicted values are both 0.
    """
    denominator = (np.abs(actual) + np.abs(predicted)).clip(min=0.01)
    diff = np.abs(actual - predicted)
    return 200 * np.mean(diff / denominator)

# Assuming 'actual_values' and 'predicted_values' are NumPy arrays or Pandas Series
actual_values = test_data[target].values
predicted_values = predictions.values

# Calculate SMAPE
smape_score = smape(actual_values, predicted_values)

# Display the SMAPE score
print("SMAPE score:", smape_score)


# # ... Work in progress...

# In[26]:


# # Plot the predicted values and actual values
# plt.figure(figsize=(10, 6))
# plt.plot(test['date'], predictions, label='Predicted')
# plt.plot(test['date'], test[target], alpha=0.5, label='Actual')
# plt.title('Predicted vs Actual')
# plt.xlabel('Date')
# plt.ylabel('Number of Sales')
# plt.legend()
# plt.show()


# In[27]:


# # Plot the predicted values and actual values
# plt.figure(figsize=(10, 6))
# plt.plot(test['date'], predictions, color='red', marker='o', label='Predicted')  # Update color and marker
# plt.plot(test['date'], test[target], alpha=0.5, label='Actual')
# plt.title('Predicted vs Actual')
# plt.xlabel('Date')
# plt.ylabel('Number of Sales')

# # Adjust the y-axis limits
# plt.ylim(bottom=0, top=max(test[target].max(), predictions.max()) * 1.1)

# plt.legend()
# plt.show()


# In[ ]:




