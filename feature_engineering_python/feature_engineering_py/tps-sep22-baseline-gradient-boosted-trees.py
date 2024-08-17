#!/usr/bin/env python
# coding: utf-8

# # TPS-SEP22 Baseline + Gradient Boosted Desicion Trees 🌿
# 
# Hello in this Notebbok, I will go in incremental steps to demonstrate some of my workflows...
# I will build the following models:
# 
# * Baseline Model. (Simple model that predicts that future sell will be same as the average of last months in prev. years)
# * Gradient Boosted Desicion Trees, With Simple CV Strategy (Simple CV strategy, using data from 2017-209 to train and 2020 to validate)
# * Gradient Boosted Desicion Trees, With a K-Fold CV Strategy (Multiple experiments, description below)
#      * Time Series Split
#      * Simple K-Fold Split (So far the best strategy for LB scores)
#      * Blocking Time Series Split
# * Hist Gradient Boosting Regressor
# 
# ---
# 
# ### Below A Quick Table of Contents...
# - 1. Installing Libraries
# - 2. Loading Libraries
# - 3. Configuring Notebook Parameters
# - 4. Auxiliary Functions
# - 5. Loading the Datasets / Pandas
# - 6. Exploring the Loaded Information
# - 7. Feature Engineering
# - 8. Data Pre-Processing
# - 9. Feature & Data Selection
# - 10. Baseline Model (Predicting Same Sales as Previous Year / Month)
# - 11. Basic Cross Validation Strategy
# - 12. XGBoost, Basic Training and CV Strategy
# - 13. CatBoost, Basic Training and CV Strategy
# - 14. Advance Model Training, XGBoost and CatBoost, K-Fold CV Loop
#     - 14.1 Training Parameters
#     - 14.2 Training Function Definition
#     - 14.3 CatBoost Regressor -- Training and Validation
#     - 14.4 XGBoost Regressor -- Training and Validation
#     - 14.5 Hist Gradient Boosting Regressor -- Training and Validation
# 
# ---
# 
# ### Dataset...
# For this challenge, you will be predicting a full year worth of sales for 4 items from two competing stores located in six different countries. This dataset is completely fictional, but contains many effects you see in real-world data, e.g., weekend and holiday effect, seasonality, etc. You are given the challenging task of predicting book sales during the year 2021.
# 
# Good luck!
# 
# **Files**
# * train.csv - the training set, which includes the sales data for each date-country-store-item combination.
# * test.csv - the test set; your task is to predict the corresponding item sales for each date-country-store-item combination. Note the Public leaderboard is scored on the first quarter of the test year, and the Private on the remaining.
# * sample_submission.csv - a sample submission file in the correct format
# 
# --- 
# 
# ### Credits To These Interesing Notebooks & Datasets
# * https://www.kaggle.com/code/samuelcortinhas/tps-sept-22-timeseries-analysis/notebook
# * https://www.kaggle.com/datasets/samuelcortinhas/gdp-of-european-countries
# * https://www.kaggle.com/code/landfallmotto/tps-sep-22-eda-histgradientboosting-6-10/notebook
# * https://www.kaggle.com/code/nischaydnk/tps-sept-leak-free-catboost-baseline/notebook
# * https://www.kaggle.com/code/ameerhamza0311/lightgbm
# * https://www.kaggle.com/code/ehekatlact/tps2209-ridge-lgbm-eda-topdownapproach/notebook?scriptVersionId=104853033
# 
# ---
# 
# ### Updates...
# * **09/03/2022**
#     * ...
#     * ...
# * **09/04/2022**
#     * Implemted GDP Per Capita (No Improvements to the Score)
#     * ...
# * **09/05/2022**
#     * Implemted CCI (No Improvements to the Score)
#     * ...
#     * ...

# # 1. Installing Libraries
# In this section we install all the nesesary libraries that are not part of the current Notebook...

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install holidays\n')


# ---

# # 2. Loading Libraries
# In this section of the Notebook we load all the nesesary libraries for the model, I will only load what I will use to keep the Notebook code more clean and easy to follow...

# In[2]:


get_ipython().run_cell_magic('time', '', '#...\n# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[3]:


get_ipython().run_cell_magic('time', '', '#...\n# Load model libraries...\nfrom xgboost import XGBRegressor # GBDT Library, XGBosst Regressor\nfrom catboost import CatBoostRegressor # GBDT Library, CatBoost Regressor\n\n# Load Sklearn libraries...\n\nfrom sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor\n\n\nfrom sklearn.metrics import mean_squared_error # Load metrics\nfrom sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, GroupKFold, KFold # Load CV strategies\nfrom sklearn.preprocessing import LabelEncoder # Load encoder packages\n\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder # Load Normalization libraries\nfrom sklearn.pipeline import Pipeline # Load sklearn pipelines, in case are needed in the CV loop\nfrom sklearn.compose import ColumnTransformer # Load \n\nimport holidays\nimport matplotlib.pyplot as plt\n')


# ---

# # 3. Configuring Notebook Parameters
# Here, I set up the notebook display settings, as the number of rows, cols and decimals to display, I also like to disable warning messages to make my outputs more clean

# In[4]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook warnings to reduce noice.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration.\n\n# Amount of data we want to load into the model from Pandas.\nDATA_ROWS = None\n\n# Dataframe, the amount of rows and cols to visualize.\nNROWS = 100\nNCOLS = 15\n\n# Main data location base path.\nBASE_PATH = '...'\n")


# In[6]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer and compressed.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# In[7]:


get_ipython().run_cell_magic('time', '', '# Configure notebook training parameters to be the same across models.\nSEED = 42\nREPORT_VERBOSE = 100\nSTOPPING_ROUNDS = 250\n')


# ---

# # 4. Auxiliary Functions

# In[8]:


get_ipython().run_cell_magic('time', '', "#...\ndef SMAPE(y_true, y_pred):\n    '''\n    \n    '''\n    denominator = (y_true + np.abs(y_pred)) / 200.0\n    diff = np.abs(y_true - y_pred) / denominator\n    diff[denominator == 0] = 0.0\n    return np.mean(diff)\n")


# In[9]:


get_ipython().run_cell_magic('time', '', "#...\nclass BlockingTimeSeriesSplit():\n    '''\n    \n    '''\n    \n    def __init__(self, n_splits):\n        self.n_splits = n_splits\n    \n    def get_n_splits(self, X, y, groups):\n        return self.n_splits\n    \n    def split(self, X, y=None, groups=None):\n        n_samples = len(X)\n        k_fold_size = n_samples // self.n_splits\n        indices = np.arange(n_samples)\n\n        margin = 0\n        for i in range(self.n_splits):\n            start = i * k_fold_size\n            stop = start + k_fold_size\n            mid = int(0.5 * (stop - start)) + start\n            yield indices[start: mid], indices[mid + margin: stop]\n")


# ---

# # 5. Loading the Datasets / Pandas

# In[10]:


get_ipython().run_cell_magic('time', '', "# Loading the datsets into Pandas\ntrn_data = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')\ntst_data = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/test.csv')\nsubmission = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/sample_submission.csv')\nGDP = pd.read_csv('../input/gpd-gpd-per-capita-by-country/GPD by Country.csv')\nCCI = pd.read_csv('../input/consumer-confidence-index-cci/consumer_confidence_index_cci.csv')\n")


# ---

# # 6. Exploring the Loaded Information

# In[11]:


get_ipython().run_cell_magic('time', '', '# Review the columns loaded\ntrn_data.columns\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the dataframe\ntrn_data.shape\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Explore more details from the dataframe\ntrn_data.info()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', '#...\ntrn_data.head()\n')


# In[15]:


get_ipython().run_cell_magic('time', '', '# Explore more detail information from the dataframe\ntrn_data.describe()\n')


# In[16]:


get_ipython().run_cell_magic('time', '', '# Review the amount of empty in the dataframe\ntrn_data.isnull().sum()\n')


# In[17]:


get_ipython().run_cell_magic('time', '', "# ....\ndef categorical_info(df, cols = ['country', 'store', 'product']):\n    for col in cols:\n        print(f'{col:8}:{df[col].unique()}')\n    return None\n")


# In[18]:


get_ipython().run_cell_magic('time', '', '# ....\ncategorical_info(trn_data)\n')


# In[19]:


get_ipython().run_cell_magic('time', '', '# ....\ncategorical_info(tst_data)\n')


# In[20]:


get_ipython().run_cell_magic('time', '', "# Create a simple function to evaluate the time-ranges of the information provided.\n# It will help with the train / validation separations\n\ndef evaluate_time(df):\n    min_date = df['date'].min()\n    max_date = df['date'].max()\n    print(f'Min Date: {min_date} /  Max Date: {max_date}')\n    return None\n\nevaluate_time(trn_data)\nevaluate_time(tst_data)\n")


# In[21]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Train\ntrn_data.nunique()\n')


# In[22]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Test\ntst_data.nunique()\n')


# In[23]:


get_ipython().run_cell_magic('time', '', '#...\ntst_data.head()\n')


# In[24]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Test\ntst_data.nunique()\n')


# In[25]:


get_ipython().run_cell_magic('time', '', '#...\nsubmission.info()\n')


# In[26]:


get_ipython().run_cell_magic('time', '', '#...\nsubmission.head()\n')


# In[27]:


trn_data['date'] = pd.to_datetime(trn_data['date'])
tst_data['date'] = pd.to_datetime(tst_data['date'])


# In[28]:


summary_by_product = trn_data.groupby(['date', 'product']).sum().reset_index().pivot(index='date', columns='product', values='num_sold').reset_index()


# In[29]:


summary_by_product


# In[30]:


trn_data['product'].unique()


# In[31]:


products = ['Kaggle Advanced Techniques', 'Kaggle Getting Started','Kaggle Recipe Book', 'Kaggle for Kids: One Smart Goose']
total_sales = summary_by_product[products].sum(axis = 1) # Sum values across the axis...


# In[32]:


for product in products:
    summary_by_product[f'{product}_ratio'] = summary_by_product[product] / total_sales


# In[33]:


plt.figure(figsize=(10,5))
for product in products:
    plt.plot(summary_by_product['date'], summary_by_product[product])
    
plt.title('Total Sales, By Product (All Stores)')
plt.xlabel('Dates')
plt.ylabel('Total Sales All Stores')
plt.show()


# In[34]:


plt.figure(figsize=(10,5))

plt.plot(summary_by_product['date'], summary_by_product['Kaggle Advanced Techniques'])  
plt.title('Total Sales, Kaggle Advanced Techniques')
plt.xlabel('Dates')
plt.ylabel('Total Sales All Stores')
plt.show()


# In[35]:


plt.figure(figsize=(10,5))
for product in products:
    plt.plot(summary_by_product['date'], summary_by_product[f'{product}_ratio'])
plt.title('Sales Ratio, Sales by Product Over Total Sales (All Stores)')
plt.xlabel('Dates')
plt.ylabel('Sales Ratio (Sales by Product / Total Sales)')
plt.show()


# ---

# # 7. Feature Engineering

# ## 7.1. Holidays

# In[36]:


get_ipython().run_cell_magic('time', '', "# Country List:['Belgium' 'France' 'Germany' 'Italy' 'Poland' 'Spain']\n\nyears_list = [2017, 2018, 2019, 2020, 2021]\n\nholiday_BE = holidays.CountryHoliday('BE', years = years_list)\nholiday_FR = holidays.CountryHoliday('FR', years = years_list)\nholiday_DE = holidays.CountryHoliday('DE', years = years_list)\nholiday_IT = holidays.CountryHoliday('IT', years = years_list)\nholiday_PL = holidays.CountryHoliday('PL', years = years_list)\nholiday_ES = holidays.CountryHoliday('ES', years = years_list)\n\nholiday_dict = holiday_BE.copy()\nholiday_dict.update(holiday_FR)\nholiday_dict.update(holiday_DE)\nholiday_dict.update(holiday_IT)\nholiday_dict.update(holiday_PL)\nholiday_dict.update(holiday_ES)\n\ndef map_holydays(df, map_dict = holiday_dict):\n    '''\n    Describe the function...\n    '''\n    df['date'] = pd.to_datetime(df['date']) # Convert the date to datetime.\n    df['holiday_name'] = df['date'].map(holiday_dict)\n    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)\n    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')\n\n    return df\n    \ntrn_data = map_holydays(trn_data, holiday_dict)\ntst_data = map_holydays(tst_data, holiday_dict)\n")


# ## 7.2. Date, Extraction Features

# In[37]:


get_ipython().run_cell_magic('time', '', '# Create some simple features base on the Date field...\n\ndef create_time_features(df: pd.DataFrame) -> pd.DataFrame:\n    """\n    Create features base on the date variable, the idea is to extract as much \n    information from the date componets.\n    Args\n        df: Input data to create the features.\n    Returns\n        df: A DataFrame with the new time base features.\n    """\n    \n    df[\'date\'] = pd.to_datetime(df[\'date\']) # Convert the date to datetime.\n    \n    # Start the creating future process.\n    df[\'year\'] = df[\'date\'].dt.year\n    df[\'quarter\'] = df[\'date\'].dt.quarter\n    df[\'month\'] = df[\'date\'].dt.month\n    df[\'day\'] = df[\'date\'].dt.day\n    df[\'dayofweek\'] = df[\'date\'].dt.dayofweek\n    df[\'dayofmonth\'] = df[\'date\'].dt.days_in_month\n    df[\'dayofyear\'] = df[\'date\'].dt.dayofyear\n    df[\'weekofyear\'] = df[\'date\'].dt.weekofyear\n    df[\'is_weekend\'] = np.where((df[\'dayofweek\'] == 5) | (df[\'dayofweek\'] == 6), 1, 0)\n    \n    return df\n\n# Apply the function \'create_time_features\' to the dataset...\ntrn_data = create_time_features(trn_data)\ntst_data = create_time_features(tst_data)\n')


# ## 7.3. Lag Features

# In[38]:


get_ipython().run_cell_magic('time', '', "#...\ntrn_data['is_train']  = 1\ntst_data['is_train']  = 0\n\nmerge_data = trn_data.append(tst_data)\n\n\ntmp = merge_data.copy(deep = True)\ntmp['prev_year'] = (tmp['year'] + 1).astype('int') # Add one year to match back 1 year... 2017 joins on (2016 + 1)\ntmp['num_sold_lag_1'] = tmp['num_sold']\ntmp = tmp[['country','store', 'product', 'prev_year', 'month', 'day', 'num_sold_lag_1']]\n\nmerge_data = merge_data.merge(tmp, \n                              how = 'left', \n                              left_on  = ['country','store', 'product', 'year', 'month', 'day'],\n                              right_on = ['country','store', 'product', 'prev_year', 'month', 'day']\n                             )\n\nmerge_data = merge_data.drop(columns=['prev_year'])\n\ntrn_data = merge_data[merge_data['is_train'] == 1] \ntst_data = merge_data[merge_data['is_train'] == 0].drop(columns = ['num_sold'])\n")


# ## Mean Encoded Feartures...

# In[39]:


# country :['Belgium' 'France' 'Germany' 'Italy' 'Poland' 'Spain']
# store   :['KaggleMart' 'KaggleRama']
# product :['Kaggle Advanced Techniques' 'Kaggle Getting Started' 'Kaggle Recipe Book' 'Kaggle for Kids: One Smart Goose']


# In[40]:


get_ipython().run_cell_magic('time', '', "#...\ndef calculate_mean_encoded(trn_df, tst_df, groups = ['country'], target = 'num_sold', feature_name = 'mean_enc_country'):\n    '''\n    Describe the function...\n    '''\n    \n    tmp = trn_df.groupby(groups)[target].mean().reset_index()\n    tmp = tmp.rename(columns = {target: feature_name})\n    trn_df = trn_df.merge(tmp, how = 'left', on = groups)\n    tst_df = tst_df.merge(tmp, how = 'left', on = groups)\n    \n    return trn_df, tst_df\n\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['country'], target = 'num_sold', feature_name = 'mean_enc_country')\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['store'], target = 'num_sold', feature_name = 'mean_enc_store')\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['product'], target = 'num_sold', feature_name = 'mean_enc_product')\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['month'], target = 'num_sold', feature_name = 'mean_enc_month')\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['dayofweek'], target = 'num_sold', feature_name = 'mean_enc_dayofweek')\ntrn_data, tst_data = calculate_mean_encoded(trn_data,tst_data, groups = ['country', 'store', 'product'], target = 'num_sold', feature_name = 'mean_enc_csp')\n")


# In[41]:


get_ipython().run_cell_magic('time', '', '#...\ntst_data.head()\n')


# ## 7.5. GPD

# In[42]:


get_ipython().run_cell_magic('time', '', "#...\ntrn_data = trn_data.merge(GDP, how = 'left', left_on = ['country', 'year'], right_on = ['Country Name', 'Year'])\ntst_data = tst_data.merge(GDP, how = 'left', left_on = ['country', 'year'], right_on = ['Country Name', 'Year'])\n")


# ## 7.6. CCI

# In[43]:


print(tst_data['country'].unique())
print(CCI['Location'].unique())

country_location = {'FRA': 'France','POL': 'Poland','ESP': 'Spain','BEL': 'Belgium','ITA': 'Italy','DEU': 'Germany', 'OECD': 'Other'}


# In[44]:


tst_data[['year', 'month']]


# In[45]:


CCI['country'] = CCI['Location'].map(country_location)
CCI['year'] = CCI['Time'].str.split('-', expand = True)[0].astype(int)
CCI['month'] = CCI['Time'].str.split('-', expand = True)[1].astype(int)


# In[46]:


CCI.head()


# In[47]:


get_ipython().run_cell_magic('time', '', "#...\ntrn_data = trn_data.merge(CCI[['country', 'year', 'month', 'Value']], how = 'left', left_on = ['country', 'year', 'month'], right_on =['country', 'year', 'month'])\ntst_data = tst_data.merge(CCI[['country', 'year', 'month', 'Value']], how = 'left', left_on = ['country', 'year', 'month'], right_on =['country', 'year', 'month'])\n")


# In[48]:


trn_data = trn_data.rename(columns = {'Value': 'CCI'})
tst_data = tst_data.rename(columns = {'Value': 'CCI'})


# In[49]:


# Calculate the delta sales between dates...
#groups = ['country','store', 'product']
#trn_data['num_sold_yesterday'] = trn_data.groupby(groups)['num_sold'].shift(1)
#trn_data['num_sold_var'] = (trn_data['num_sold'] - trn_data['num_sold_yesterday']) / trn_data['num_sold_yesterday']


# In[50]:


#trn_data[trn_data['store'] == 'KaggleMart'].head(50)


# ---

# # 8. Data Pre-Processing

# ## 9.1. Remove Or Calibrate Outliers

# In[51]:


get_ipython().run_cell_magic('time', '', "#...\n# The first half of 2020 is excluded because it is an outlier.\n# trn_data = trn_data[~((trn_data['year'] == 2020) & (trn_data['month'] >= 1) & (trn_data['month'] <= 7))]\n")


# In[52]:


get_ipython().run_cell_magic('time', '', "#...\nFACTOR = 1\ntrn_data['num_sold_calibrated'] = trn_data['num_sold']\ntrn_data['num_sold_calibrated'] = np.where(((trn_data['year'] == 2020) & (trn_data['month'] >= 1) & (trn_data['month'] <= 3)), trn_data['num_sold'] * FACTOR, trn_data['num_sold'])\nsummary_by_product_cal = trn_data.groupby(['date', 'product']).sum().reset_index().pivot(index='date', columns='product', values='num_sold_calibrated').reset_index()\n")


# In[53]:


get_ipython().run_cell_magic('time', '', "#...\nplt.figure(figsize=(15,6))\n\nholidays = (trn_data[trn_data['is_holiday'] == 1]['date'].unique())\nholidays = [pd.Timestamp(x) for x in holidays]\n# only one line may be specified; full height\nplt.vlines(x = holidays, ymin=0, ymax=5000, colors='gray', ls=':', lw=1, label='vline_single - partial height')\n\n\nplt.plot(summary_by_product_cal['date'], summary_by_product_cal['Kaggle Advanced Techniques'])\n\nplt.title('Total Sales, Kaggle Advanced Techniques')\nplt.xlabel('Dates')\nplt.ylabel('Total Sales All Stores')\n\n\n#plt.show()\n")


# Fromt the plot above, seems like the amounts of holidays provided by the library, are quite a lot creating noise, I will manually remove some of them in the future using this plat as a guide...

# In[54]:


get_ipython().run_cell_magic('time', '', '# I will hide the data from January 2020 - April 2020 and Try to predict it using an ML Model...\n# Work in Progress...\n')


# ## 8.1. Label Encoding

# In[55]:


get_ipython().run_cell_magic('time', '', "# ...\ndef encode_labels(df, text_features = ['country','store', 'product']):\n    '''\n    Describe the function...\n    '''\n    \n    for categ_col in df[text_features].columns:\n        encoder = LabelEncoder()\n        df[categ_col + '_enc'] = encoder.fit_transform(df[categ_col])\n    return df\n\ntrn_data = encode_labels(trn_data, text_features = ['country','store', 'product'])\ntst_data = encode_labels(tst_data, text_features = ['country','store', 'product'])\n")


# ## 8.2. Target Log Transformation

# In[56]:


get_ipython().run_cell_magic('time', '', "#...\ndef transform_target(df, target = 'num_sold'):\n    '''\n    Apply a log transformation to the target for better optimization \n    during training.\n    '''\n    \n    df[target] = np.log(df[target])\n    return df\n\ntrn_data = transform_target(trn_data, target = 'num_sold')\n")


# ---

# # 9. Feature & Data Selection

# ## 9.2. Select Features to Train the Model

# In[57]:


get_ipython().run_cell_magic('time', '', "# Extract features and avoid certain columns from the dataframe for training purposes...\ntarget = 'num_sold'\navoid = ['row_id', 'date','country', 'store', 'product','is_train', 'num_sold']\nfeatures = [feat for feat in trn_data.columns if feat not in avoid]\n\n# Print a list of all the features created...\nprint(features)\n")


# In[58]:


get_ipython().run_cell_magic('time', '', "#...\nfeatures = [\n            'year',\n            'quarter',\n            'month',\n            'day',\n            'dayofweek',\n            'dayofmonth',\n            'dayofyear',\n            'weekofyear',\n            'is_weekend',\n            'country_enc',\n            'store_enc',\n            'product_enc',\n            'is_holiday',\n            #'mean_enc_country', \n            #'mean_enc_store', \n            #'mean_enc_product',\n            #'mean_enc_month',\n            #'mean_enc_dayofweek',\n            #'mean_enc_csp',\n            #'num_sold_lag_1',\n            #'GDP',\n            'GDP per Capita',\n            'CCI',\n            ]\n")


# ## 9.3. Remove NaNs

# In[59]:


get_ipython().run_cell_magic('time', '', '#...\n# trn_data = trn_data.dropna()\n')


# ---

# # 10. Baseline Model (Predicting Same Sales as Previous Year / Month)
# 
# I will use a simple validation stratefy...
# * Train model in Data from  2017-2019 or maybe only 2019...
# * Validate the model in data from 2020...
# * Generate prediction on data from 2021, and submit to Kaggle...

# In[60]:


get_ipython().run_cell_magic('time', '', "#...\ncutoff_date = 2019\nX_train, X_val = trn_data[trn_data['year'] == cutoff_date], trn_data[trn_data['year'] > cutoff_date]\n")


# In[61]:


get_ipython().run_cell_magic('time', '', "# A simple baseline model, same as last year\nselected_cols = ['country','store', 'product', 'month']\nX_train_summary = X_train.groupby(selected_cols)['num_sold'].mean().reset_index()\nX_train_summary = X_train_summary.rename(columns={'num_sold': 'mean_num_sold'})\n")


# In[62]:


get_ipython().run_cell_magic('time', '', "# ...\nX_val = X_val.merge(X_train_summary[['country','store', 'product', 'month', 'mean_num_sold']], how = 'left', on = ['country','store', 'product', 'month'])\n")


# In[63]:


get_ipython().run_cell_magic('time', '', "# ...\nsmape = SMAPE(X_val['num_sold'], X_val['mean_num_sold'])\nprint(f'Validation SMAPE: {smape}')\n")


# In[64]:


get_ipython().run_cell_magic('time', '', '# Results of the experiments...\n## Validation SMAPE: 31.01254904480124 # Using data from 2017-2019 for training...\n## Validation SMAPE: 31.194432568298055 # Using data from 2017-2019 for training...\n')


# In[65]:


get_ipython().run_cell_magic('time', '', "# ...\ntst_data = tst_data.merge(X_train_summary[['country','store', 'product', 'month', 'mean_num_sold']], how = 'left', on = ['country','store', 'product', 'month'])\n")


# In[66]:


get_ipython().run_cell_magic('time', '', "# Use the created model to predict the sales for 2019...\nsubmission['num_sold'] = tst_data['mean_num_sold']\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_baseline.csv',index = False)\n")


# ---

# # 11. Basic Cross Validation Strategy

# In[67]:


get_ipython().run_cell_magic('time', '', "# Creates the Train and Validation sets to train the model...\n# Define a cutoff date to split the datasets\ncutoff_date = '2020-01-01'\n\n# Split the data into train and validation datasets using timestamp best suited for timeseries...\nX_train = trn_data[trn_data['date'] < cutoff_date][features]\ny_train = trn_data[trn_data['date'] < cutoff_date][target]\n\nX_val = trn_data[trn_data['date'] >= cutoff_date][features]\ny_val = trn_data[trn_data['date'] >= cutoff_date][target]\n")


# # 12. XGBoost, Basic Training and CV Strategy

# ## 12.1. Model Parameters

# In[68]:


get_ipython().run_cell_magic('time', '', "# Defines a really simple XGBoost Regressor...\n\nxgboost_params = {'eta'              : 0.02,\n                  'n_estimators'     : 8096,\n                  'max_depth'        : 8,\n                  'max_leaves'       : 256,\n                  'colsample_bylevel': 0.75,\n                  'colsample_bytree' : 0.75,\n                  'subsample'        : 0.75, # XGBoost would randomly sample 'subsample_value' of the training data prior to growing trees\n                  'min_child_weight' : 512,\n                  'min_split_loss'   : 0.002,\n                  'alpha'            : 0.08,\n                  'lambda'           : 128,\n                  'objective'        : 'reg:squarederror',\n                  'eval_metric'      : 'rmse', # Originally using RMSE, trying new functions...\n                  'tree_method'      : 'hist',\n                  'seed'             :  SEED\n                  }\n")


# ## 12.2. Model Initialization

# In[69]:


get_ipython().run_cell_magic('time', '', '# Create an instance of the XGBRegressor and set the model parameters...\nregressor = XGBRegressor(**xgboost_params)\n')


# ## 12.3. Model Training Stage

# In[70]:


get_ipython().run_cell_magic('time', '', '# Train the XGBRegressor using the train and validation datasets, \n# Utilizes early_stopping_rounds to control overfitting...\nregressor.fit(X_train,\n              y_train,\n              eval_set=[(X_val, y_val)],\n              early_stopping_rounds = STOPPING_ROUNDS,\n              verbose = REPORT_VERBOSE)\n')


# ## 12.4. Feature Importance

# In[71]:


get_ipython().run_cell_magic('time', '', "#...\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(features, regressor.feature_importances_):\n    feats[feature] = importance # add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(10,5))\n")


# ## 12.5 Model Evaluation, RMSE and SMAPE

# In[72]:


get_ipython().run_cell_magic('time', '', "val_pred = regressor.predict(X_val[features])\n# Convert the target back to non-logaritmic.\nval_pred = np.exp(val_pred)\ny_val = np.exp(y_val)\n\nscore = np.sqrt(mean_squared_error(y_val, val_pred))\nprint(f'RMSE: {score} / SMAPE: {SMAPE(y_val, val_pred)}')\n")


# ## 12.6 Submission File Creation

# In[73]:


get_ipython().run_cell_magic('time', '', "#...\n# Use the created model to predict the sales for 2019...\npredictions = regressor.predict(tst_data[features])\nsubmission['num_sold'] = predictions\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_xgboost.csv',index = False)\n")


# ## 12.7 Records From Public Leader Board

# In[74]:


# RMSE: 80.8008406259422 / SMAPE: 28.969361167991547 >>> LB SMAPE: 28.65081
# ...


# ---

# # 13. CatBoost, Basic Training and CV Strategy

# ## 13.1. Model Parameters

# In[75]:


get_ipython().run_cell_magic('time', '', "#...\ncatboost_params = {'iterations'      : 4096,\n                   'learning_rate'   : 0.02,\n                   'depth'           : 6,\n                   'min_data_in_leaf': 2,\n                   'l2_leaf_reg'     : 20.0,\n                   'random_strength' : 2.0,\n                   'bootstrap_type'  :'Bayesian',\n                   'loss_function'   :'MAE',\n                   'eval_metric'     :'SMAPE',\n                   'random_seed'     : SEED\n                  }\n")


# ## 13.2. Model Initialization

# In[76]:


get_ipython().run_cell_magic('time', '', '# Create an instance of the XGBRegressor and set the model parameters...\nregressor = CatBoostRegressor(**catboost_params)\n')


# ## 13.3. Model Training Stage

# In[77]:


get_ipython().run_cell_magic('time', '', '# Train the XGBRegressor using the train and validation datasets, \n# Utilizes early_stopping_rounds to control overfitting...\nregressor.fit(X_train,\n              y_train,\n              eval_set=[(X_val, y_val)],\n              early_stopping_rounds = STOPPING_ROUNDS,\n              verbose = REPORT_VERBOSE)\n')


# ## 13.4. Feature Importance

# In[78]:


get_ipython().run_cell_magic('time', '', "#...\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(features, regressor.feature_importances_):\n    feats[feature] = importance #add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(10,5))\n")


# ## 13.5 Model Evaluation, RMSE and SMAPE

# In[79]:


get_ipython().run_cell_magic('time', '', "val_pred = regressor.predict(X_val[features])\n# Convert the target back to non-logaritmic.\nval_pred = np.exp(val_pred)\n\nscore = np.sqrt(mean_squared_error(y_val, val_pred))\nprint(f'RMSE: {score} / SMAPE: {SMAPE(y_val, val_pred)}')\n")


# ## 13.6 Submission File Creation

# In[80]:


get_ipython().run_cell_magic('time', '', "#...\n# Use the created model to predict the sales for 2019...\npredictions = regressor.predict(tst_data[features])\nsubmission['num_sold'] = predictions\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_catboost.csv',index = False)\n")


# # 14. Advance Model Training, XGBoost and CatBoost, K-Fold CV Loop

# ## 14.1 Training Parameters

# In[81]:


get_ipython().run_cell_magic('time', '', '#...\nN_SPLITS = 5\nEARLY_STOPPING_ROUNDS = 250 # Will stop training if one metric of one validation data doesn’t improve in last round\nVERBOSE = 0 # Controls the level of information, verbosity\n')


# ## 14.2 Training Function Definition

# In[82]:


get_ipython().run_cell_magic('time', '', '#...\n# Cross Validation Loop for the Classifier.\ndef cross_validation_train(train_dataset, labels, test_dataset, model, model_params, n_folds = 5, eval_set_logic = True):\n    """\n    The following function is responsable of training a model in a\n    cross validation loop and generate predictions on the specified test set.\n    The function provides the model feature importance list as other variables.\n\n    Args:\n    train  (Dataframe): ...\n    labels (Series): ...\n    test   (Dataframe): ...\n    model  (Model): ...\n    model_params (dict of str: int): ...\n    n_folds (int): ...\n    eval_set_logic (bool): ...\n\n    Return:\n    regressor (Model): ...\n    feat_import (Dataframe): ...\n    test_pred (Dataframe): ...\n    oof_label (Dataframe): ...\n    oof_pred (Dataframe): ...\n    ...\n\n    """\n    \n    # Creates empty place holders for out of fold and test predictions.\n    oof_pred  = np.zeros(len(train_dataset)) # We are predicting prob. we need more dimensions.\n    oof_label = np.zeros(len(train_dataset))\n    test_pred = np.zeros(len(test_dataset)) # We are predicting prob. we need more dimensions\n    val_indexes_used = [] # Array to store the indexes used\n    \n    \n    # Creates empty place holder for the feature importance.\n    feat_import = np.zeros(len(features))\n    \n    \n    # Creates Stratified Kfold object to be used in the train / validation\n    #Kf = TimeSeriesSplit(n_splits = n_folds)\n    #Kf = BlockingTimeSeriesSplit(n_splits = n_folds)\n    \n    Kf = KFold(n_splits = n_folds, shuffle = True, random_state = SEED)\n    \n    \n    # Start the training and validation loops.\n    for fold, (train_idx, val_idx) in enumerate(Kf.split(train_dataset, labels)):\n        # Creates the index for each fold\n        print(f\'Fold: {fold} >>>\')        \n        train_min_date = trn_data.iloc[train_idx][\'date\'].min()\n        train_max_date = trn_data.iloc[train_idx][\'date\'].max()\n        \n        valid_min_date = trn_data.iloc[val_idx][\'date\'].min()\n        valid_max_date = trn_data.iloc[val_idx][\'date\'].max()\n        \n        # Print the date ranges used in each of the folds\n        print(f\'Train Min / Max Dates: {train_min_date} / {train_max_date}\')\n        print(f\'Valid Min / Max Dates: {valid_min_date} / {valid_max_date}\')\n\n        print(f\'Training on {train_dataset.iloc[train_idx].shape[0]} Records\')\n        print(f\'Validating on {train_dataset.iloc[val_idx].shape[0]} Records\')\n        \n        \n        # Generates, Train and Validation datasets\n        X_trn, y_trn = train_dataset.iloc[train_idx], labels.iloc[train_idx]\n        X_val, y_val = train_dataset.iloc[val_idx], labels.iloc[val_idx]\n        \n        # Generate a copy of the Test Dataset, for predictions\n        X_tst = test_dataset\n        \n        val_indexes_used = np.concatenate((val_indexes_used, val_idx), axis = None)\n        \n    \n        # Scaler Initialization, to standarize the Numerical features\n        #scaler = MinMaxScaler()\n        #X_trn[numeric_cols] = scaler.fit_transform(X_trn[numeric_cols])\n        #X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])\n        \n        #X_tst[numeric_cols] = scaler.transform(X_tst[numeric_cols])\n        \n        \n        # Instanciate a regressor based on the model parameters, and model type\n        regressor = model(**model_params)\n \n        # Allow the training of model without eval set, logic, for example LR model and others\n        if eval_set_logic:\n            regressor.fit(X_trn, \n                          y_trn, \n                          eval_set = [(X_val, y_val)], \n                          early_stopping_rounds = EARLY_STOPPING_ROUNDS, \n                          verbose = VERBOSE\n                         )\n        else:\n             regressor.fit(X_trn, y_trn)\n        \n        \n        # Generate predictions using the trained model...\n        val_pred = regressor.predict(X_val)\n        \n        # Apply exponentail transformation for the target...\n        val_pred = np.exp(val_pred)\n        y_val = np.exp(y_val)\n        \n        # Store the results in the arrays...\n        oof_pred[val_idx]  = val_pred # store the predictions for that fold.\n        oof_label[val_idx] = y_val # store the true labels for that fold.\n        \n        # Calculate the model error based on the selected metric...\n        error =  np.sqrt(mean_squared_error(y_val, val_pred))\n\n        # Print some of the model performance metrics...\n        print(f\'RMSE: {error}\')\n        print(f\'SMAPE: {SMAPE(y_val, val_pred)}\')\n        print("."*50)\n        \n        # Populate the feature importance matrix\n        if eval_set_logic: feat_import += regressor.feature_importances_\n                \n        # Generate predictions for the test set, apply exp. transformation.\n        predictions = regressor.predict(X_tst)\n        predictions = np.exp(predictions)\n        test_pred += (predictions) / n_folds\n     \n    \n    # Calculate the error across all the folds and print the reuslts\n    val_indexes_used = val_indexes_used.astype(int)\n    global_error = np.sqrt(mean_squared_error(oof_label[val_indexes_used], oof_pred[val_indexes_used]))\n    print(\'\')\n    print(f\'RMSE: {global_error}...\')\n    print(f\'SMAPE: {SMAPE(oof_label[val_indexes_used], oof_pred[val_indexes_used])}...\')\n    \n    return (regressor, feat_import, test_pred, oof_label, oof_pred)\n')


# ## 14.3 CatBoost Regressor -- Training and Validation

# In[83]:


get_ipython().run_cell_magic('time', '', "#...\ncatboost_params = {'iterations'      : 8096,\n                   'learning_rate'   : 0.02,\n                   'depth'           : 6,\n                   'min_data_in_leaf': 20,\n                   'l2_leaf_reg'     : 20.0,\n                   'random_strength' : 2.0,\n                   'bootstrap_type'  :'Bayesian',\n                   'loss_function'   :'MAE',\n                   'eval_metric'     :'SMAPE',\n                   'random_seed'     : SEED\n                  }\n")


# In[84]:


get_ipython().run_cell_magic('time', '', '#...\n# Uses the cross_validation_train to build and train the model with XGBoost\ncatboost_results = cross_validation_train(train_dataset  = trn_data[features], \n                                          labels = trn_data[target], \n                                          test_dataset   = tst_data[features], \n                                          model  = CatBoostRegressor, \n                                          model_params = catboost_params,\n                                          n_folds = N_SPLITS,\n                                          eval_set_logic = True)\n\ncbsr, feat_imp, predictions, oof_label, oof_pred = catboost_results\n')


# In[85]:


# RMSE: 16.033314814263452...
# SMAPE: 5.832077034607415... >>> 6.93687...

# RMSE: 12.296340475191332...
# SMAPE: 4.263994361419477... >>> 8.65013, Removed 2020 Outliers...


# In[86]:


get_ipython().run_cell_magic('time', '', "#...\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(features, cbsr.feature_importances_):\n    feats[feature] = importance #add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(12,5))\n")


# In[87]:


get_ipython().run_cell_magic('time', '', "#...\n# Use the created model to predict the sales for 2021...\nsubmission['num_sold'] = predictions\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_catboost_cv.csv',index = False)\n")


# ---

# ## 14.4 XGBoost Regressor -- Training and Validation

# In[88]:


get_ipython().run_cell_magic('time', '', "# Defines a really simple XGBoost Regressor...\n\nxgboost_params = {'eta'              : 0.02,\n                  'n_estimators'     : 8096,\n                  'max_depth'        : 8,\n                  'max_leaves'       : 256,\n                  'colsample_bylevel': 0.75,\n                  'colsample_bytree' : 0.75,\n                  'subsample'        : 0.75, # XGBoost would randomly sample 'subsample_value' of the training data prior to growing trees\n                  'min_child_weight' : 512,\n                  'min_split_loss'   : 0.002,\n                  'alpha'            : 0.08,\n                  'lambda'           : 128,\n                  'objective'        : 'reg:squarederror',\n                  'eval_metric'      : 'rmse', # Originally using RMSE, trying new functions...\n                  'tree_method'      : 'hist',\n                  'seed'             :  SEED\n                  }\n")


# In[89]:


get_ipython().run_cell_magic('time', '', '# Uses the cross_validation_train to build and train the model with XGBoost\nxgboost_results = cross_validation_train(train_dataset  = trn_data[features], \n                                         labels = trn_data[target], \n                                         test_dataset   = tst_data[features], \n                                         model  = XGBRegressor, \n                                         model_params = xgboost_params,\n                                         n_folds = N_SPLITS,\n                                         eval_set_logic = True\n                                        )\n\nxgbr, feat_imp, predictions, oof_label, oof_pred = xgboost_results\n')


# In[90]:


get_ipython().run_cell_magic('time', '', "#...\nfeats = {} # a dict to hold feature_name: feature_importance\nfor feature, importance in zip(features, xgbr.feature_importances_):\n    feats[feature] = importance #add the name/value pair \n\nimportances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})\nimportances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=90, figsize=(12,5))\n")


# In[91]:


get_ipython().run_cell_magic('time', '', "#...\n# Use the created model to predict the sales for 2021...\nsubmission['num_sold'] = predictions\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_xgboost_cv.csv',index = False)\n")


# In[92]:


# RMSE: 14.245131759944085...
# SMAPE: 5.1664391190590555... >>> 6.75250...

# RMSE: 12.829848586156686...
# SMAPE: 4.331430247331949... >>> 6.42100 *** Highest Score from the Notebook...

# RMSE: 14.27075724103846...
# SMAPE: 4.650998617604207... >>> 10.7898, Added 1 Year Lags...

# RMSE: 12.675940515357583...
# SMAPE: 4.330553888253027... >>> 7.70470, Removed Outliers...

# RMSE: 12.632162406765653...
# SMAPE: 4.29380199609298... >>> ???, Simple Mean Encoded...

# RMSE: 12.773269918125056...
# SMAPE: 4.316803221403405... >>> ???, Simple Mean Encoded...

# RMSE: 12.554270275037675...
# SMAPE: 4.27840329573787... >>> 8.61658, GDP No Mean Encoded


# ---

# ## 14.5 Hist Gradient Boosting Regressor -- Training and Validation

# In[93]:


hgbr_params = {'learning_rate': 0.02,
               'max_iter': 8096,
               'max_leaf_nodes': 16,
               'max_depth':6,
               'min_samples_leaf': 20,
               'l2_regularization': 20.0,
               'random_state': SEED,
              }


# In[94]:


get_ipython().run_cell_magic('time', '', '# Uses the cross_validation_train to build and train the model with XGBoost\nhgbr_results = cross_validation_train(train_dataset  = trn_data[features], \n                                         labels = trn_data[target], \n                                         test_dataset   = tst_data[features], \n                                         model  = HistGradientBoostingRegressor, \n                                         model_params = hgbr_params,\n                                         n_folds = N_SPLITS,\n                                         eval_set_logic = False\n                                        )\n\nhgbr, feat_imp, predictions, oof_label, oof_pred = xgboost_results\n')


# In[95]:


get_ipython().run_cell_magic('time', '', "#...\n# Use the created model to predict the sales for 2021...\nsubmission['num_sold'] = predictions\n\n# Creates a submission file for Kaggle...\nsubmission.to_csv('submission_hgbr_cv.csv',index = False)\n")


# In[96]:


# RMSE: 12.684494025660708...
# SMAPE: 4.337062344347393... >>> 6.39665, No Mean Encoded or GDP


# ---

# In[ ]:





# In[ ]:




