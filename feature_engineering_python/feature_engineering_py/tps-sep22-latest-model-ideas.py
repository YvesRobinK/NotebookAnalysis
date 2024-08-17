#!/usr/bin/env python
# coding: utf-8

# # 💡 TPS-SEP22, Popular Model Techniques 
# My initial goal for this Notebook is to implement some of the latest modeling ideas I have observed over the last few weeks in Kaggle. I will aim to explain them as straightforwardly as possible and have well-documented code.
# As a second step, I will improve upon these concepts with variations based on my knowledge.
# 
# # 🧱 0.0 Work In Progress, Comeback Soon, Thanks !
# 
# ### Updates
# **09/10/2022**:
# * Building iniatial notebook structure...
#     
# **09/15/2022**:
# * Getting back on track to complete the Notebook...
# * There is a lot of new Notebook I need to process...
# 
# **09/18/2022**:
# * Added some feature engineering to the Notebook...
# 
# **09/21/2022**
# * Working on more data aggregattion function...
# * Constructing a baseline model...
# 
# **09/23/2022**
# * Continue working on the baseline model...
# 
# **09/27/2022**
# * Working on the Notebook structure...
# * Finalizing the end to end model...
# 
# **09/30/2022**
# * ...
# * ...
# * ...
# 

# # 📚 1.0 Importing Libraries...

# In[1]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso


# # ⚙️ 2.0 Configuring the Notebook...

# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook warnings to reduce noice.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration.\n\n# Amount of data we want to load into the model from Pandas.\nDATA_ROWS = None\n\n# Dataframe, the amount of rows and cols to visualize.\nNROWS = 100\nNCOLS = 25\n\n# Main data location base path.\nBASE_PATH = '...'\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer and compressed.\npd.options.display.float_format = '{:,.4f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# In[6]:


get_ipython().run_cell_magic('time', '', '# Configure notebook training parameters to be the same across models.\nSEED = 70\nREPORT_VERBOSE = 150\nSTOPPING_ROUNDS = 250\n')


# # 🚀 3.0 Defining Auxiliart Functions...

# In[7]:


get_ipython().run_cell_magic('time', '', "#...\ndef SMAPE(y_true, y_pred):\n    '''\n    \n    '''\n    denominator = (y_true + np.abs(y_pred)) / 200.0\n    diff = np.abs(y_true - y_pred) / denominator\n    diff[denominator == 0] = 0.0\n    return np.mean(diff)\n")


# # 💾 4.0 Loading the Dataset... 

# In[8]:


get_ipython().run_cell_magic('time', '', "# Loading the datasets into Pandas\ntrn_data = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/train.csv')\ntst_data = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/test.csv')\nsubmission = pd.read_csv('/kaggle/input/tabular-playground-series-sep-2022/sample_submission.csv')\n")


# # 🧭 5.0 Exploring the Loaded Dataset...

# In[9]:


get_ipython().run_cell_magic('time', '', '# Explore more details from the dataframe\ntrn_data.info()\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Review the first 5 rows of the dataset\ntrn_data.head()\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '# Explore more detail information from the dataframe\ntrn_data.describe()\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Review the amount of empty in the dataframe\ntrn_data.isnull().sum()\n')


# In[13]:


get_ipython().run_cell_magic('time', '', "# Create a function to understand categorical variables\ndef categorical_info(df, cols = ['country', 'store', 'product']):\n    for col in cols:\n        print(f'{col:8}:{df[col].unique()}')\n    return None\n")


# In[14]:


get_ipython().run_cell_magic('time', '', '# Utilization of the categorical info function on the train dataset\ncategorical_info(trn_data)\n')


# In[15]:


get_ipython().run_cell_magic('time', '', '# Utilization of the categorical info function on the train dataset\ncategorical_info(tst_data)\n')


# In[16]:


get_ipython().run_cell_magic('time', '', "# Create a simple function to evaluate the time-ranges of the information provided.\n# It will help with the train / validation separations\n\ndef evaluate_time(df):\n    min_date = df['date'].min()\n    max_date = df['date'].max()\n    print(f'Min Date: {min_date} /  Max Date: {max_date}')\n    return None\n\nevaluate_time(trn_data)\nevaluate_time(tst_data)\n")


# In[17]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Train\ntrn_data.nunique()\n')


# In[18]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Train\ntst_data.nunique()\n')


# # 💡 6.0 Feature Engineering...

# In[19]:


get_ipython().run_cell_magic('time', '', '# Create some simple features base on the Date field...\n\ndef create_time_features(df: pd.DataFrame) -> pd.DataFrame:\n    """\n    Create features base on the date variable, the idea is to extract as much \n    information from the date componets.\n    Args\n        df: Input data to create the features.\n    Returns\n        df: A DataFrame with the new time base features.\n    """\n    \n    df[\'date\'] = pd.to_datetime(df[\'date\']) # Convert the date to datetime.\n    \n    # Start the creating future process.\n    df[\'year\'] = df[\'date\'].dt.year\n    df[\'quarter\'] = df[\'date\'].dt.quarter\n    df[\'month\'] = df[\'date\'].dt.month\n    df[\'day\'] = df[\'date\'].dt.day\n    df[\'dayofweek\'] = df[\'date\'].dt.dayofweek\n    df[\'dayofmonth\'] = df[\'date\'].dt.days_in_month\n    df[\'dayofyear\'] = df[\'date\'].dt.dayofyear\n    df[\'weekofyear\'] = df[\'date\'].dt.weekofyear\n    df[\'is_weekend\'] = np.where((df[\'dayofweek\'] == 5) | (df[\'dayofweek\'] == 6), 1, 0)\n    \n    return df\n\n# Apply the function \'create_time_features\' to the dataset...\ntrn_data = create_time_features(trn_data)\ntst_data = create_time_features(tst_data)\n')


# In[20]:


get_ipython().run_cell_magic('time', '', '#...\ntrn_data.head()\n')


# In[21]:


get_ipython().run_cell_magic('time', '', '#...\ntst_data.head()\n')


# # ♟️ 7.0 Model Development & Strategy...
# Multiple Notebooks that are providing good performance against the leaderboard are using a simple level of modeling by working on aggregated levels of the data to predict sales from there specific ratios are used to split the data by store, country and product
# 
# In this section I will replicate some this strategy step by step, from there I will build on top of it new strategies.
# 
# **Most Popular Steps:**
# * ...
# * ...
# * ...
# 
# **New Ideas:**
# * ...
# * ...
# * ...

# ## 7.1 Total Sales by Date and Product

# In[22]:


get_ipython().run_cell_magic('time', '', "# Creates groups for ratio calculations...\n# Group by date and product\nsales_by_date_product = trn_data.groupby(['date', 'product']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')).reset_index()\nsales_by_date_product\n")


# ## 7.2 Total Sales by Month, Country, Store and Product

# In[23]:


get_ipython().run_cell_magic('time', '', "# Group by date (month), country, store and product\ntrn_data_month = trn_data.groupby(['month', 'country', 'store', 'product']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')).reset_index()\ntrn_data_month\n")


# ## 7.3 Sales Ratio by Store

# In[24]:


get_ipython().run_cell_magic('time', '', "# Calculate the num_sold ratios by store\nstore_ratios = trn_data.groupby(['store']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')) / trn_data['num_sold'].sum()\nstore_ratios = store_ratios.reset_index()\nstore_ratios\n")


# ## 7.4 Sales Ratio by Product and Store

# In[25]:


get_ipython().run_cell_magic('time', '', "# Calculate the num_sold ratios by product and store\nproduct_store_ratios = trn_data.groupby(['product', 'store']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')) / trn_data.groupby(['product']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum'))\nproduct_store_ratios = product_store_ratios.reset_index()\nproduct_store_ratios\n")


# ## 7.5 Sales Ratio by Product and Country

# In[26]:


get_ipython().run_cell_magic('time', '', "# Calculate the num_sold ratios by product and country\nproduct_country_ratios = trn_data.groupby(['product', 'country']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')) / trn_data.groupby(['product']).agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum'))\nproduct_country_ratios = product_country_ratios.reset_index()\nproduct_country_ratios\n")


# ## 7.6 ...

# In[27]:


get_ipython().run_cell_magic('time', '', "# Utilize the pivot functionality and stack to built a ratio of sales by product for each date\nsales_by_date_product_pivot = sales_by_date_product.pivot(index = 'date', columns = 'product', values = 'total_sold')\nsales_by_date_product_pivot_ratio = sales_by_date_product_pivot.apply(lambda x: x/x.sum(), axis = 1)\nsales_by_date_product_pivot_ratio = sales_by_date_product_pivot_ratio.stack().rename('ratios').reset_index()\nsales_by_date_product_pivot_ratio\n")


# ## 7.6 Totals Sales by Date

# In[28]:


get_ipython().run_cell_magic('time', '', "trn_data_by_date = trn_data.groupby('date').agg(total_sold = pd.NamedAgg(column = 'num_sold', aggfunc = 'sum')).reset_index()\ntrn_data_by_date.head()\n")


# In[29]:


get_ipython().run_cell_magic('time', '', 'trn_data_by_date = create_time_features(trn_data_by_date)\n')


# In[30]:


trn_data_by_date['month_sine'] = np.sin(trn_data_by_date['month'] * (2 * np.pi / 12))


# In[31]:


important_dates = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                   12,16,17, 124, 125, 126, 127, 140, 141,142, 
                   167, 168, 169, 170, 171, 173, 174, 175, 176, 
                   177, 178, 179, 180, 181, 203, 230, 231, 232, 
                   233, 234, 282, 289, 290, 307, 308, 309, 310, 
                   311, 312, 313, 317, 318, 319, 320, 360, 361, 
                   362, 363, 364, 365]

trn_data_by_date['important_dates'] = trn_data_by_date['dayofyear'].apply(lambda x: x if x in important_dates else 0)


# In[32]:


get_ipython().run_cell_magic('time', '', "# Remove outlier data points from the dataset\ntrn_data_by_date = trn_data_by_date.loc[~((trn_data_by_date['date'] >= '2020-03-01') & (trn_data_by_date['date'] < '2020-06-01'))]\n")


# In[33]:


trn_data_by_date = pd.get_dummies(trn_data_by_date, columns = ['important_dates','dayofweek'], drop_first = True)


# In[34]:


trn_data_by_date.head()


# ## 7.7 Pre-Processing the Test Dataset

# In[35]:


get_ipython().run_cell_magic('time', '', 'tst_data.head()\ntst_data_by_date = tst_data\n')


# In[36]:


get_ipython().run_cell_magic('time', '', "# Create time features...\ntst_data_by_date = tst_data_by_date[['date','year', 'month', 'dayofyear', 'dayofweek']]\n\n# Create the sine of the month feature...\ntst_data_by_date['month_sine'] = np.sin(tst_data_by_date['month'] * (2 * np.pi / 12))\n\n# Flag Important dates as a feature...\ntst_data_by_date['important_dates'] = tst_data_by_date['dayofyear'].apply(lambda x: x if x in important_dates else 0)\n\n# Creating One-Hot encoded variables...\ntst_data_by_date = pd.get_dummies(tst_data_by_date, columns = ['important_dates', 'dayofweek'], drop_first = True)\n")


# ## 7.7 Training a Model to Predict Total Sales by Date

# In[37]:


get_ipython().run_cell_magic('time', '', '## Training a model using...\nfrom sklearn.model_selection import GroupKFold, GridSearchCV\n')


# In[38]:


get_ipython().run_cell_magic('time', '', "ignore = ['month', 'quarter', 'day', 'dayofmonth', 'dayofyear', 'is_weekend', 'weekofyear', 'total_sold', 'date']\nfeatures = [feat for feat in trn_data_by_date if feat not in ignore]\n\nlabel = ['total_sold']\n\nX = trn_data_by_date[features]\ny = trn_data_by_date[label]\n")


# In[39]:


get_ipython().run_cell_magic('time', '', "N_SPLITS = 4\n\nk_fold = GroupKFold(n_splits = N_SPLITS)\nscores = []\ntest_predictions = []\n\nfor fold, (trn_index, val_index) in enumerate(k_fold.split(X, groups = X['year'])):\n    # Print the current fold\n    print(f'Fold: {fold} ...') \n    \n    # Generates the train and validation sets, based on the kfold index\n    X_trn, y_trn = X.iloc[trn_index], y.iloc[trn_index]\n    X_val, y_val = X.iloc[val_index], y.iloc[val_index]\n    \n    # Generates the model predictions using a Lasso Regression\n    model = Lasso(tol = 0.00001, max_iter = 10_000, alpha = 0.1)\n    model = make_pipeline(StandardScaler(), model)\n    \n    # Fit a model to the train datasets\n    model.fit(X_trn, y_trn)\n    \n    # Score the model on the validation datset\n    score =  model.score(X_val, y_val)\n    scores.append(score)\n    \n    # Generated predictions using the test dataset\n    predictions = model.predict(tst_data_by_date[features])\n    test_predictions.append(predictions)\n    \n    # Print the model perfomance results\n    if score < 0:\n        print(f'  >>> Model Score: {score:.3f}')\n    else:\n        print(f'  >>> Model Score:  {score:.3f}')\n        \n    print('')\n    \nprint('')    \nprint(f'Overall Model >>> Score: {np.mean(scores):.3f}', '\\n')\n")


# In[40]:


get_ipython().run_cell_magic('time', '', '# Exploring the test dataset\ntst_data.head()\n')


# In[41]:


get_ipython().run_cell_magic('time', '', '# Exploring the submission dataset\nsubmission.head()\n')


# ## 7.9 Disaggregating the Predictions

# In[42]:


get_ipython().run_cell_magic('time', '', "# Disaggregate the predictions, sales by date into, sales...\n# Isolate the information from 2019, not sure of the reason at this point...\nISOLATE_YEAR = 2019\nproduct_ratio_filtered = sales_by_date_product_pivot_ratio[sales_by_date_product_pivot_ratio['date'].dt.year == ISOLATE_YEAR]\n\n# Create a month and day field...\nproduct_ratio_filtered['month-day'] = product_ratio_filtered['date'].dt.strftime('%m-%d')\ntst_data['month-day'] = tst_data['date'].dt.strftime('%m-%d')\n\n# Merge back the product ratios into the test dataset...\ntst_data = pd.merge(tst_data, product_ratio_filtered[['month-day', 'product', 'ratios']], how = 'left', on = ['month-day', 'product'])\n")


# In[43]:


tst_data


# ---

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




