#!/usr/bin/env python
# coding: utf-8

# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background: #0077cc; color: #fff;">Table of Contents</div></h1>
# 
# * [Problem Statement](#ps)
# * [About Dataset](#ad)
# * [EDA](#eda)
#     - [Train](#train)
#     - [Gas Price](#gp)
#     - [Client](#cid)
#     - [Electricity Price](#ep)
#     - [Forecast Weather](#fw)
#     - [Historical Weather](#hw)
# * [Feature Engineering](#fe)
# * [Model Training](#mt)

# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Problem Statement</div></h1>
# 
# <a id='ps'></a>

# The problem described in this competition revolves around forecasting the amount of electricity produced and consumed by energy customers in Estonia who have installed solar panels. The goal is to predict the consumption or production amount for specific segments, defined by factors such as county, business type, and product type. This prediction task is based on various factors, including weather data, energy prices, and records of installed photovoltaic capacity.
# 
# Key Components of the Problem:
# 
# 1. **Target Variable:**
#    - The target variable represents the consumption or production amount for a specific segment, determined by the combination of county, business type, and product type.
#    - The target values are associated with hourly periods, and the datetime column indicates the start of each 1-hour period.
# 
# 2. **Features:**
#    - The features include information about the prosumers (consumers and producers), such as county, business type, and product type.
#    - Weather-related features are available, including temperature, dewpoint, cloud cover, wind components, solar radiation, and precipitation. These features are provided as both historical and forecasted data.
#    - Gas prices and electricity prices are also included as additional features.
# 
# 3. **Data Blocks:**
#    - The concept of data blocks is introduced, where rows sharing the same data_block_id are available at the same forecast time. This is essential for understanding the temporal alignment of the data.
# 
# 4. **Time Series Aspect:**
#    - The competition involves time series forecasting, where the goal is to predict future electricity consumption or production based on historical data.
#    - The private leaderboard is determined using real data gathered after the submission period closes, highlighting the importance of accurate predictions for unseen future time periods.
# 
# 5. **Prediction Unit:**
#    - The prediction unit is defined by the combination of county, business type, and product type. New prediction units may appear or disappear in the test set.
# 
# 6. **Evaluation Metric:**
#    - The competition likely uses a specific evaluation metric (e.g., Root Mean Squared Error - RMSE) to assess the accuracy of predictions.
# 
# 7. **Data Sources:**
#    - Various datasets are provided, including information on prosumers, energy prices, and weather forecasts. These datasets need to be integrated and utilized effectively for accurate predictions.
# 
# In summary, the problem requires participants to develop predictive models that can accurately forecast the electricity consumption or production for specific segments in Estonia, leveraging a diverse set of features including weather data, energy prices, and historical consumption patterns. The temporal nature of the data introduces challenges associated with time series forecasting, and effective feature engineering and model selection are crucial for success in the competition.

# <a id='ad'></a>
# 
# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">About Dataset</div></h1>

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <h2>Dataset Overview </h2>
# 
# This dataset provides information on electricity consumption and production by Estonian energy customers with installed solar panels. The goal is to forecast the amount of electricity generated and consumed based on various factors such as weather data, energy prices, and installed photovoltaic capacity.
# 
# <h2> Files </h2>
# 
# 1. **train.csv**
#    - `county`: ID code for the county.
#    - `is_business`: Boolean for business status.
#    - `product_type`: Contract type ID (0: Combined, 1: Fixed, 2: General service, 3: Spot).
#    - `target`: Consumption or production amount for the segment.
#    - `is_consumption`: Boolean for consumption or production.
#    - `datetime`: Start of the 1-hour period.
#    - `data_block_id`: Identifies rows available at the same forecast time.
# 
# 2. **gas_prices.csv**
#    - `origin_date`: Date for day-ahead prices.
#    - `forecast_date`: Date for relevant forecast prices.
#    - `lowest/highest_price_per_mwh`: Lowest/highest price of natural gas in Euros per MWh.
#    - `data_block_id`: ID for data blocks.
# 
# 3. **client.csv**
#    - `product_type, county, is_business`: ID codes.
#    - `eic_count`: Aggregated number of consumption points.
#    - `installed_capacity`: Installed photovoltaic capacity (kW).
#    - `date, data_block_id`.
# 
# 4. **electricity_prices.csv**
#    - `origin_date, forecast_date`: Dates for electricity prices.
#    - `euros_per_mwh`: Electricity price in Euros per MWh.
#    - `data_block_id`.
# 
# 5. **forecast_weather.csv**
#    - Weather forecasts from the European Centre for Medium-Range Weather Forecasts.
#    - Various weather parameters such as temperature, cloud cover, wind components.
#    - `data_block_id, forecast_datetime`.
# 
# 6. **historical_weather.csv**
#    - Historic weather data.
#    - Measured parameters include temperature, dewpoint, rain, wind speed.
#    - `data_block_id`.
# 
# ## Key Information
# 
# - **Time Series API**: All datasets follow the same time convention in EET/EEST. Forecasting is based on a 1-hour period.
# - **Prediction Unit**: Rows are identified by a combination of county, business status, and product type.
# - **Evaluation**: The private leaderboard uses real data gathered after the submission period closes.
# 

# <a id='eda'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">EDA</div></h1>

# In[2]:


import os
import pandas as pd

directory = "/kaggle/input/predict-energy-behavior-of-prosumers/"

# Load Dataframes
df_train = pd.read_csv(os.path.join(directory, "train.csv"))
df_gas_price = pd.read_csv(os.path.join(directory, "gas_prices.csv"))
df_client = pd.read_csv(os.path.join(directory, "client.csv"))
df_electricity_prices = pd.read_csv(os.path.join(directory, "electricity_prices.csv"))
df_forecast_weather = pd.read_csv(os.path.join(directory, "forecast_weather.csv"))
df_historical_weather = pd.read_csv(os.path.join(directory, "historical_weather.csv"))


# <a id='train'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Train</div><h1>

# In[3]:


import json
# sample view of dataset
with open(os.path.join(directory, "county_id_to_name_map.json"), "r") as j:
    county_json = json.loads(j.read())


# In[4]:


def map_county(id_):
    return county_json[id_]


# In[5]:


df_train["county"] = df_train["county"].apply(lambda x : map_county(str(x)))


# In[6]:


df_train.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[7]:


# Display basic information about the DataFrame
print("DataFrame Info:")
print(df_train.info())


# In[8]:


# Display summary statistics
print("\nSummary Statistics:")
df_train.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[9]:


# Check for missing values
print("\nMissing Values:")
print(df_train.isnull().sum())


# In[10]:


import plotly.express as px
missing_dict = eval(df_train.isnull().sum().to_json())
missing_df = pd.DataFrame({"Attributes": missing_dict.keys(), "Count":missing_dict.values()})
px.bar(missing_df, x="Attributes", y="Count", title="Missing Value Distribution")


# In[11]:


df_train.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[12]:


import plotly.express as px
import pandas as pd

dtype_count = df_train.dtypes.value_counts()
dict_ = eval(dtype_count.to_json())
dict_df = pd.DataFrame({"Data Type": dict_.keys(), "Count":dict_.values()})

px.pie(dict_df, names="Data Type", values="Count", title = "Data Type Distribution")


# In[13]:


county_dict = df_train["county"].value_counts().to_dict()
county_df = pd.DataFrame({"County":county_dict.keys(), "Count":county_dict.values()})
print(county_df)
px.bar(county_df, x="County", y="Count", title="County Distribution", color="County")


# In[14]:


prod_dict = df_train["product_type"].value_counts().to_dict()
prod_df = pd.DataFrame({"Product Type":prod_dict.keys(), "Count":prod_dict.values()})
px.pie(prod_df, names="Product Type", values="Count", title="Product Type Distribution", color="Product Type")


# In[15]:


is_business_dict = df_train["is_business"].value_counts().to_dict()
is_business_df = pd.DataFrame({"Business":is_business_dict.keys(), "Count":is_business_dict.values()})
px.bar(is_business_df, x="Business", y="Count", title="Business Distribution", color="Business")


# <a id='gp'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Gas Price</div></h1>

# In[16]:


# dataframe overview
df_gas_price.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[17]:


# dataframe overview
df_gas_price.info()


# In[18]:


# dataframe overview
df_gas_price.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[19]:


# dataframe overview for missing values
df_gas_price.isnull().sum()


# In[20]:


fig = px.line(df_gas_price, x="forecast_date", y=["highest_price_per_mwh", "lowest_price_per_mwh"], title="Forecast Date based Gas Price Comparison")
fig.show()


# In[21]:


fig = px.line(df_gas_price, x="origin_date", y=["highest_price_per_mwh", "lowest_price_per_mwh"], title="Oridin Date based Gas Price Comparison")
fig.show()


# In[22]:


fig = px.line(df_gas_price, x="data_block_id", y=["highest_price_per_mwh", "lowest_price_per_mwh"], title="Data Block Id based Gas Price Comparison")
fig.show()


# <a id='cid'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Client</div></h1>

# In[23]:


df_client.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[24]:


df_client.info()


# In[25]:


df_client.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[26]:


df_client.isnull().sum()


# In[27]:


df_aggregated = df_client.groupby('date')['eic_count'].sum().reset_index()
fig = px.line(df_aggregated, x="date", y="eic_count", title="Aggregated EIC Distribution Based On Date")
fig.show()


# In[28]:


df_aggregated = df_client.groupby('date')['installed_capacity'].sum().reset_index()
fig = px.line(df_aggregated, x="date", y="installed_capacity", title="Aggregated Installed Capacity Distribution Based On Date")
fig.show()


# In[29]:


df_aggregated = df_client.groupby('is_business')['eic_count'].sum().reset_index()
fig = px.bar(df_aggregated, x="is_business", y="eic_count", title="Aggregated EIC Distribution Based On Business")
fig.show()


# <a id='ep'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Electricity Price</div></h1>

# In[30]:


df_electricity_prices.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[31]:


df_electricity_prices.info()


# In[32]:


df_electricity_prices.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[33]:


df_electricity_prices.isnull().sum()


# In[34]:


px.line(df_electricity_prices, x="forecast_date", y="euros_per_mwh", title = "Electricity Price Distribution Based on Forecast Date")


# In[35]:


px.line(df_electricity_prices, x="origin_date", y="euros_per_mwh", title = "Electricity Price Distribution Based on Forecast Date")


# In[36]:


df_aggregated = df_electricity_prices.groupby('data_block_id')['euros_per_mwh'].sum().reset_index()
fig = px.bar(df_aggregated, x="data_block_id", y="euros_per_mwh", title="Aggregated Price Distribution Based On Block Id")
fig.show()


# <a id='fw'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Forecast Weather</div></h1>

# In[37]:


df_forecast_weather.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[38]:


df_forecast_weather.info()


# In[39]:


df_forecast_weather.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[40]:


df_forecast_weather.isnull().sum()


# In[41]:


import folium

# Create a folium map centered at the mean latitude and longitude
records = 1000
map_center = [df_forecast_weather.head(records)['latitude'].mean(), df_forecast_weather.head(records)['longitude'].mean()]
weather_map = folium.Map(location=map_center, zoom_start=7)

# Add markers for each data point
for index, row in df_forecast_weather.head(records).iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"Temperature: {row['temperature']}°C, Cloud Cover: {row['cloudcover_total']}%"
                 ).add_to(weather_map)

# Display the map
weather_map


# <a id='hw'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 0px 0; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Historical Weather</div></h1>

# In[42]:


df_historical_weather.head().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[43]:


df_historical_weather.info()


# In[44]:


df_historical_weather.describe().style.set_table_styles([
        {'selector': 'thead', 'props': [('background-color', 'lightgrey')]},
        {'selector': 'tr:hover', 'props': [('background-color', 'rgba(173, 216, 230, 0.5)')]}
    ])


# In[45]:


df_historical_weather.isnull().sum()


# In[46]:


import folium

# Create a folium map centered at the mean latitude and longitude
records = 1000
map_center = [df_historical_weather.head(records)['latitude'].mean(), df_historical_weather.head(records)['longitude'].mean()]
weather_map = folium.Map(location=map_center, zoom_start=7)

# Add markers for each data point
for index, row in df_historical_weather.head(records).iterrows():
    folium.Marker([row['latitude'], row['longitude']],
                  popup=f"Temperature: {row['temperature']}°C, Cloud Cover: {row['cloudcover_total']}%"
                 ).add_to(weather_map)

# Display the map
weather_map


# <a id='fe'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Feature Engineering</div></h1>

# ## Designing an Approach
# 
# Let's formulate the process where missing values are filled using a Random Forest model before joining to a third DataFrame.
# 
# ### Joining DataFrames with Missing Value Imputation:
# 
# #### Mathematical Formulation:
# 
# Let's consider three DataFrames: `df1`, `df2`, and `df3`. We want to join `df1` and `df2` based on a common key, fill missing values in the joined DataFrame using a Random Forest model, and then join the result with `df3`.
# 
# 1. **Join `df1` and `df2` based on a common key:**
# 
#    $$df_{\text{joined}} = df1 \, \text{join} \, df2 \, \text{on} \, \text{'key'}$$
# 
# 2. **Fill missing values in** $$df_{\text{joined}}$$ **using a Random Forest model:**
# 
#    $$df_{\text{filled}} = \text{FillMissingValues}(df_{\text{joined}})$$
# 
#    The `FillMissingValues` operation involves training a Random Forest model on features with existing values and target on the existing target values. This trained model is then used to predict missing values in the features.
# 
# 3. **Join the filled DataFrame with `df3`:**
# 
#    $$df_{\text{final}} = df_{\text{filled}} \, \text{join} \, df3 \, \text{on} \, \text{'key'}$$
#    
# 4. **Process this untill all are merged :**
# 
#     Assuming you have a list of DataFrames `[df1, df2, df3, ...]`
#     
#    $$(\text{dataframes} = [df_1, df_2, \ldots, df_n])$$
# 
#    Initialize df_final as the first DataFrame in the list
#    
#    $$(df_{\text{final}} = \text{dataframes}[0])$$
# 
#    Iterate over the remaining DataFrames in the list
#    
#    $$\text{for } i \text{ in range(1, len(dataframes)}:$$
#    $$(df_{\text{joined}} = df_{\text{final}}\, \text{join} \, \text{dataframes}[i] \, \text{on} \, \text{'key'})$$
#    $$(df_{\text{filled}} = \text{FillMissingValues}(df_{\text{joined}}))$$
#    $$(df_{\text{final}} = df_{\text{filled}})$$
# 
# ### Filling Missing Values using Random Forest:
# 
# #### Mathematical Formulation:
# 
# Suppose you have a DataFrame $$(df_{\text{filled}})$$ with missing values in the column `'missing_column'`. You want to fill these missing values using a Random Forest model:
# 
# 1. **Identify the features with existing values $$((X_{\text{train}}))$$ and the target values $$((y_{\text{train}}))$$ for the missing column:**
# 
#    $$X_{\text{train}} = df_{\text{filled}}['features\_with\_values']$$
# 
#    $$y_{\text{train}} = df_{\text{filled}}['missing\_column']$$
# 
# 2. **Train a Random Forest model on $$(X_{\text{train}})$$ and $$(y_{\text{train}})$$:**
# 
#    $$\text{RF_model} = \text{TrainRandomForest}(X_{\text{train}}, y_{\text{train}})$$
# 
# 3. **Identify the features with missing values $$((X_{\text{to_fill}}))$$:**
# 
#    $$X_{\text{to_fill}} = df_{\text{filled}}['features\_to\_fill']$$
# 
# 4. **Predict missing values using the trained Random Forest model:**
# 
#    $$\hat{y}_{\text{filled}} = \text{RF_model.predict}(X_{\text{to_fill}})$$
# 
#    Here, $$\hat{y}_{\text{filled}}$$ represents the predicted missing values.
# 
# 5. **Update the DataFrame with the predicted values:**
# 
#    $$df_{\text{filled}}['missing\_column'] = \hat{y}_{\text{filled}}$$
# 

# ### Merging and Grouping data for generating features

# In[47]:


# Load Dataframes
df_train = pd.read_csv(os.path.join(directory, "train.csv")).copy()
df_gas_prices = pd.read_csv(os.path.join(directory, "gas_prices.csv")).copy()
df_client = pd.read_csv(os.path.join(directory, "client.csv")).copy()
df_electricity_prices = pd.read_csv(os.path.join(directory, "electricity_prices.csv")).copy()
df_forecast_weather = pd.read_csv(os.path.join(directory, "forecast_weather.csv")).copy()
df_historical_weather = pd.read_csv(os.path.join(directory, "historical_weather.csv")).copy()

# Convert date columns to datetime format and extract the date part
df_train['datetime'] = pd.to_datetime(df_train['datetime']).dt.date
df_gas_prices['origin_date'] = pd.to_datetime(df_gas_prices['origin_date']).dt.date
df_gas_prices['forecast_date'] = pd.to_datetime(df_gas_prices['forecast_date']).dt.date
df_client['date'] = pd.to_datetime(df_client['date']).dt.date
df_electricity_prices['origin_date'] = pd.to_datetime(df_electricity_prices['origin_date']).dt.date
df_electricity_prices['forecast_date'] = pd.to_datetime(df_electricity_prices['forecast_date']).dt.date
df_forecast_weather['origin_datetime'] = pd.to_datetime(df_forecast_weather['origin_datetime']).dt.date
df_historical_weather['datetime'] = pd.to_datetime(df_historical_weather['datetime']).dt.date

# Group by multiple columns and aggregate mean for each dataframe
grouped_train = df_train.groupby(['county', 'is_business', 'product_type', 'is_consumption', 'datetime', 'data_block_id', 'prediction_unit_id']).agg({'target': 'mean'}).reset_index()
grouped_gas_price = df_gas_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'lowest_price_per_mwh': 'mean', 'highest_price_per_mwh':'mean'}).reset_index()
grouped_client = df_client.groupby(['product_type', 'county', 'is_business', 'date', 'data_block_id']).agg({'eic_count':'mean', 'installed_capacity':'mean'}).reset_index()
grouped_electricity_prices = df_electricity_prices.groupby(['forecast_date', 'origin_date', 'data_block_id']).agg({'euros_per_mwh':'mean'}).reset_index()
grouped_forecast_weather = df_forecast_weather.groupby(['latitude', 'longitude', 'origin_datetime', 'data_block_id']).agg({'temperature': 'mean','dewpoint': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','cloudcover_total': 'mean','10_metre_u_wind_component': 'mean','10_metre_v_wind_component': 'mean','direct_solar_radiation': 'mean','surface_solar_radiation_downwards': 'mean','snowfall': 'mean','total_precipitation': 'mean'}).reset_index()
grouped_historical_weather = df_historical_weather.groupby(['latitude', 'longitude', 'datetime', 'data_block_id']).agg({'rain':'mean', 'surface_pressure': 'mean','cloudcover_total': 'mean','cloudcover_high': 'mean','cloudcover_low': 'mean','cloudcover_mid': 'mean','windspeed_10m': 'mean','winddirection_10m': 'mean','shortwave_radiation': 'mean','direct_solar_radiation': 'mean','diffuse_radiation': 'mean','snowfall': 'mean'}).reset_index()


# In[48]:


merged_data = pd.merge(grouped_train, grouped_gas_price, on='data_block_id', how='inner')
merged_data = pd.merge(merged_data, grouped_client, on='data_block_id', how='inner')
merged_data = pd.merge(merged_data, grouped_electricity_prices, on='data_block_id', how='inner')


# In[49]:


merged_data = merged_data[['county_x', 'is_business_x', 'product_type_x', 'is_consumption',
       'datetime', 'data_block_id', 'prediction_unit_id', 'target',
       'forecast_date_x','lowest_price_per_mwh',
       'highest_price_per_mwh', 
       'date', 'eic_count', 'installed_capacity',  'euros_per_mwh']].drop_duplicates()


# In[50]:


merged_data = merged_data.groupby(["county_x",
                    "is_business_x",
                    "product_type_x",
                    "is_consumption",
                    "datetime",
                    "data_block_id",
                   "prediction_unit_id",
                   "forecast_date_x",
                   "date"]).agg({"target":"mean",
                                "lowest_price_per_mwh":"mean",
                                "highest_price_per_mwh":"mean",
                               "eic_count":"mean",
                               "installed_capacity":"mean",
                               "euros_per_mwh":"mean"}).reset_index()


# In[51]:


merged_data = pd.merge(merged_data, grouped_forecast_weather, on='data_block_id', how='inner')
merged_data["datetime"] = pd.to_datetime(merged_data["datetime"])


# In[52]:


# date features

# Extract date features
merged_data['year'] = merged_data['datetime'].dt.year
merged_data['month'] = merged_data['datetime'].dt.month
merged_data['day'] = merged_data['datetime'].dt.day
merged_data['day_of_week'] = merged_data['datetime'].dt.dayofweek
merged_data['week_of_year'] = merged_data['datetime'].dt.isocalendar().week

# Calculate time-based features
reference_date = pd.to_datetime('2021-01-01')
merged_data['days_since_reference'] = (merged_data['datetime'] - reference_date).dt.days

# Create binary variable for weekends
merged_data['is_weekend'] = merged_data['datetime'].dt.weekday.isin([5, 6]).astype(int)


# In[53]:


merged_data = merged_data.drop(["datetime", "date", "forecast_date_x","origin_datetime"], axis=1)


# In[54]:


merged_data


# In[55]:


merged_data.info()


# In[56]:


merged_data.columns


# <a id='mt'></a>
# <h1><div style="text-align: center; padding: 10px; border-radius: 100px 80px; overflow: hidden; font-family: 'New Times Roman', serif; font-size: 1em; font-weight: bold; background:#0077cc; color: #fff;">Model  Training </div></h1>

# In[57]:


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error

# Additional libraries for explainability:
from shap import Explainer
from lime import lime_tabular
import shap


# In[58]:


features = merged_data.drop("target", axis=1)
target = merged_data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)


# In[59]:


# Define XGBoost model parameters (adjust based on your data)
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model (e.g., mean squared error)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean squared error: {mse}")


# ### Feature importance

# In[60]:


plot_importance(model)


# ### Tree Visuals

# In[61]:


from xgboost import plot_tree
import matplotlib.pyplot as plt

# Plot the first tree in the ensemble (num_trees=0)
fig, ax = plt.subplots(figsize=(20, 20))  # Corrected 'size' to 'subplots'
plot_tree(model, num_trees=0, ax=ax, rankdir='LR')  # Left-to-right layout
plt.show()


# ### Model Explainability

# In[62]:


explainer = Explainer(model)
shap_values = explainer([X_test.values[0]])
shap.plots.waterfall(shap_values[0])

