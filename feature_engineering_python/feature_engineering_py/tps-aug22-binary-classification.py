#!/usr/bin/env python
# coding: utf-8

# # TPS-AUG22 Binary Classification 👾⚡
# ### In this Notebook I will start with a Desicion Tree and then move to a Gradient Boosted Desicion Tree.
# A Simple model to get started into the TPS-AUG22 competition, probably after a few updates the model got a little more complex
# 
# ### My strategy will be the following >>>  
# Some of the code is Inactive, remove magic cell **%%script false --no-raise-error**
# 
# <img src='https://media.springernature.com/relative-r300-703_m1050/springer-static/image/art%3A10.1038%2Fs41591-020-01197-2/MediaObjects/41591_2020_1197_Figa_HTML.png?as=webp' width = 550>
# 
# 
# * **Simple Data Analysis**
#     * A quick overview of the dataset, Head, Info, Describe, NaN Analysis, Target Distribution and Many More...
# * **A Quick Feature Engineering**
#     * Extracting codes from attribute_0, attribute_1
#     * Splitting the loading feature
#     * Aggregated features across columns
# * **Simple Pre-Processing**
#     * Fill missing values, Mean and Mode
#     * Standarization **(Used inside CV loop)**
# * **Advance Feature Engineering**
#     * Using K-means to create clusters
# * **Feature Selection Using Recursive Feature Elimination**
#     * Using Sklearn Recursive Feature Elimination (RFE)
#     * Using Boruta Recursive Feature Elimination (RFE)
#     * Selecting optimal features from RFE
# * **Model Development**
#     * Constructing of a Random Forest Classifier
#     * Constructing of a XGBoost Classifier
#     * Constructing of a Logistic Regression Classifier
# * **A Simple Cross Validation Loop**
#     * 80/20 Split
# * **Creating a Submission File**
#     * Exporting inference to CSV
# * **Training a Model in a CV loop**
#     * K-fold Cross validation loop using group kfold
#     * Logistic Regression Classifier & XGBoost Classifier
# * **Creating a Submission File**
#     * Exporting inference to CSV
# * **Hyper-Param Optimization OPTUNA**
#     * Optimiza the best model, LR
# ---
# ### Credits and Notebook Used As Inspiration
# * https://www.kaggle.com/code/desalegngeb/tps08-logisticregression-and-some-fe
# * https://www.kaggle.com/code/thedevastator/tps-aug-simple-baseline/notebook?scriptVersionId=102551969
# * https://www.kaggle.com/code/pourchot/update-for-keras-optuna-for-lr
# 
# ---
# 
# ### Data Descripton
# 
# **Overview**
# 
# This data represents the results of a large product testing study. For each product_code you are given a number of product attributes (fixed for the code) as well as a number of measurement values for each individual product, representing various lab testing methods. Each product is used in a simulated real-world environment experiment, and and absorbs a certain amount of fluid (loading) to see whether or not it fails.
# 
# Your task is to use the data to predict individual product failures of new codes with their individual lab test results.
# 
# **Files**
# * **train.csv** - the training data, which includes the target failure
# * **test.csv** - the test set; your task is to predict the likelihood each id will experience a failure
# * **sample_submission.csv** - a sample submission file in the correct format
# ---
# 
# ### Notebook Updates
# **07/31/2022 ...**
# 
# * Develop the first iteration of the Notebook...
# * Created the Random Forest Classifier...
# * Implemented a Simple XGBoost Classifier...
# 
# **08/01/2022 ...**
# * Structuring the Notebook better...
# * Attemp to improve the score by adding better imputation
# * Attemp to improve the model by removing product_code
# 
# **08/02/2022 ...**
# * Modified some of the Normalization techniques
# * Implemented Cross-Validation loop for the model
# 
# **08/05/2022 ...**
# * Review the notebook code, because low performance
# * Implemented a logistic regression model on a cross validation loop
# * implemented k-means for feature engineering
# 
# **08/06/2022 ...**
# * Refactored and documented all the code
# * Added Optuna hyper-Param Optimization
# * ...
# 
# **08/07/2022 ...**
# * Added Mean Encoded
# * Added disable some of the code using **%%script false --no-raise-error**
# * ...

# # 1.0 Importing Python Libraries...

# In[1]:


get_ipython().run_cell_magic('capture', '', '!git clone https://github.com/analokmaus/kuma_utils.git\n')


# In[2]:


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


# In[3]:


get_ipython().run_cell_magic('time', '', '# Importing all the Nesesary Libraries\nfrom sklearn.impute import KNNImputer, SimpleImputer\nfrom sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler\nfrom sklearn.preprocessing import LabelEncoder\n\nfrom sklearn.cluster import KMeans\nfrom scipy.spatial.distance import cdist\nimport matplotlib.pyplot as plt\n\nfrom sklearn.model_selection import train_test_split\n\n\n# Evaluate RFE for classification\nfrom numpy import mean\nfrom numpy import std\n\nfrom sklearn.datasets import make_classification\nfrom sklearn.model_selection import cross_val_score\nfrom sklearn.model_selection import RepeatedStratifiedKFold\nfrom sklearn.feature_selection import RFECV\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.pipeline import Pipeline\n\n# Import the nesesary libraries\nfrom boruta import BorutaPy\n\n# Baseline Model Using Random Forest Classifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score, roc_auc_score, f1_score\n\nfrom xgboost import XGBClassifier\nfrom lightgbm import LGBMClassifier\n\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, roc_auc_score, f1_score\n\n# ...\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score, roc_auc_score, f1_score\nfrom sklearn.model_selection import KFold, GroupKFold, StratifiedKFold\n\nfrom imblearn.over_sampling import SMOTE\n')


# In[4]:


get_ipython().run_cell_magic('time', '', 'import os\nimport sys\nsys.path.append("kuma_utils/")\nfrom kuma_utils.preprocessing.imputer import LGBMImputer\n')


# ---

# # 2.0 Configuring the Notebook...

# In[5]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings To Reduce Noice.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[6]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 100\nNCOLS = 15\n\n# Main data location path...\nBASE_PATH = '...'\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# ---

# # 3.0 Loading Dataset into A Pandas DataFrame...

# In[8]:


get_ipython().run_cell_magic('time', '', "# Loading the datsets into Pandas\ntrn_data = pd.read_csv('/kaggle/input/tabular-playground-series-aug-2022/train.csv')\ntst_data = pd.read_csv('/kaggle/input/tabular-playground-series-aug-2022/test.csv')\nsubmission = pd.read_csv('/kaggle/input/tabular-playground-series-aug-2022/sample_submission.csv')\n")


# ___

# # 4.0 Exploring the Dataset...

# In[9]:


get_ipython().run_cell_magic('time', '', '# Review the columns loaded\ntrn_data.columns\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the dataframe\ntrn_data.shape\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '# Explore more details from the dataframe\ntrn_data.info()\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'trn_data.head().T # Transposing the head to visualize more of the dataset\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Explore more detail information from the dataframe\ntrn_data.describe()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', "# Target distribution\ntrn_data['failure'].value_counts()\n")


# In[15]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Train\ntrn_data.nunique()\n')


# In[16]:


get_ipython().run_cell_magic('time', '', '# Review the unique the quantity of values -- Test\ntst_data.nunique()\n')


# In[17]:


get_ipython().run_cell_magic('time', '', "# Review the product codes for the test dataset\ntst_data['product_code'].sample(5)\n")


# In[18]:


get_ipython().run_cell_magic('time', '', '# Review the amount of empty in the dataframe\ntrn_data.isnull().sum()\n')


# In[19]:


get_ipython().run_cell_magic('time', '', "# Try to find more information about the empty values\ntrn_data[trn_data['loading'].isnull()]\n")


# In[20]:


get_ipython().run_cell_magic('time', '', "# ....\nprint(trn_data['attribute_0'].unique())\nprint(trn_data['attribute_1'].unique())\nprint(trn_data['attribute_2'].unique())\nprint(trn_data['attribute_3'].unique())\n")


# In[21]:


get_ipython().run_cell_magic('time', '', "# ....\nprint(tst_data['attribute_0'].unique())\nprint(tst_data['attribute_1'].unique())\nprint(tst_data['attribute_2'].unique())\nprint(tst_data['attribute_3'].unique())\n")


# In[22]:


get_ipython().run_cell_magic('time', '', "# ...\nsummary = trn_data.groupby(['attribute_0', 'attribute_1', 'attribute_2'])['attribute_3'].mean().reset_index()\nsummary.head(100)\n")


# In[23]:


get_ipython().run_cell_magic('time', '', "# ...\nsummary = tst_data.groupby(['attribute_0', 'attribute_1', 'attribute_2'])['attribute_3'].mean().reset_index()\nsummary.head(100)\n")


# In[24]:


get_ipython().run_cell_magic('time', '', "# Trying to understand product codes by aggregation\nsummary_by_product_code = (trn_data\n                           .groupby('product_code')[['loading', 'failure']]\n                           .mean()\n                           .reset_index())\n\nsummary_by_product_code\n")


# ---

# In[ ]:





# # 5.0 Feature Engineering

# # 5.1 Creating Numerical and Categorical Features Lists

# In[25]:


get_ipython().run_cell_magic('time', '', "# Identifying Categorical Faetures...\nlimit = 50 # Anything with more than 50 unique values is considered a category...\ncat_feat = [feat for feat in trn_data.columns if trn_data[feat].nunique() < limit]\nnum_feat = [feat for feat in trn_data.columns if feat not in cat_feat]\n\nprint(f'Categorical Features: {cat_feat}')\nprint(f'Numerical Features: {num_feat[:10]} ...')\n")


# 

# In[26]:


get_ipython().run_cell_magic('time', '', "# Identifying numeric columns\nmeasure_cols = [col for col in trn_data.columns if 'meas' in col] + ['loading']\nmeasure_cols\n")


# # 5.1 Counting Missing

# In[27]:


def count_missing(df):
    missing_list = [feat for feat in df.columns if df[feat].isnull().sum() > 0]
    for col in missing_list:
        df[col + '_missing'] = np.where(df[col].isna() == True, 1, 0)
    
    
    df['missing_data'] = df.isnull().sum(axis = 1)
    return df

trn_data = count_missing(trn_data)
tst_data = count_missing(tst_data)


# # 5.2 Filling Missing Values

# ## 5.2.1 Fill Values Using Mean and Mode

# In[28]:


get_ipython().run_cell_magic('time', '', "# Creating a fill missing values function \ndef fill_missing(df, cols = ['loading']):\n    '''\n    \n    '''\n    \n    numerics = ['int16', 'int32', 'int64','float16','float32', 'float64']\n    for col in df.select_dtypes(include=numerics):\n        df[col] = df[col].fillna(value = df[col].mean())\n    \n    for col in df.select_dtypes(exclude=numerics):\n        df[col] = df[col].fillna(value = df[col].mode())\n    return df\n\n#trn_data = fill_missing(trn_data)\n#tst_data = fill_missing(tst_data)\n")


# ## 5.2.2 Fill Values Using KNNImputer, SimpleImputer

# In[29]:


get_ipython().run_cell_magic('time', '', '# Create a fill missing values function using the Simple Imputer\ndef imputer_numeric(df, cols, group_code = \'product_code\'):\n    \'\'\'\n    \n    \'\'\'\n    product_list = list(df[group_code].unique())\n    result_df = pd.DataFrame()\n    for product in product_list:\n        tmp = df[df[group_code] == product]\n        print(f\'Imputing for Product: {product}...\')\n        imputer = SimpleImputer(strategy = "mean")\n        imputer.fit(tmp[cols])\n\n        tmp[cols] = imputer.transform(tmp[cols])\n        result_df = result_df.append(tmp)\n        \n    print(\'...........\', \'\\n\')\n    return result_df\n\n#trn_data = imputer_numeric(trn_data, measure_cols, group_code = \'product_code\')\n#tst_data = imputer_numeric(tst_data, measure_cols, group_code = \'product_code\')\n')


# ## 5.2.3 Fill Values Using LGBM Imputer

# In[30]:


get_ipython().run_cell_magic('time', '', "# Create a fill missing values function using the Simple Imputer\ndef lgbm_imputer_numeric(df, cols, group_code = 'product_code'):\n    '''\n    \n    '''\n    product_list = list(df[group_code].unique())\n    result_df = pd.DataFrame()\n    \n    object_cols = ['attribute_0', 'attribute_1']\n    lgbm_imputer = LGBMImputer(cat_features = object_cols, n_iter = 50)\n    \n    for product in product_list:\n        tmp = df[df[group_code] == product]\n        print(f'Imputing for Product: {product}...')\n        lgbm_imputer.fit_transform(tmp[cols])\n        result_df = result_df.append(tmp)\n        \n    print('...........', '\\n')\n    return result_df\n\ntrn_data = imputer_numeric(trn_data, measure_cols, group_code = 'product_code')\ntst_data = imputer_numeric(tst_data, measure_cols, group_code = 'product_code')\n")


# In[31]:


#....


# In[32]:


get_ipython().run_cell_magic('time', '', '#...\ntrn_data.isnull().sum()\n')


# # 5.3 Feature Engineering -- Extracting Numeric Code from Attributes

# In[33]:


get_ipython().run_cell_magic('time', '', "# Extracting numeric component from the attribute variables\ndef extract_num_code(df, feat = ['attribute_0', 'attribute_1']):\n    for col in df[feat].columns:\n        df[col] = df[col].str.split('_', 1).str[1].astype('int')\n    return df\n\ntrn_data = extract_num_code(trn_data, feat = ['attribute_0', 'attribute_1'])\ntst_data = extract_num_code(tst_data, feat = ['attribute_0', 'attribute_1'])\n")


# In[34]:


trn_data


# In[35]:


tst_data


# # 5.4 Feature Engineering -- Loading, Integer and Decimal Components

# In[36]:


get_ipython().run_cell_magic('time', '', "#...\ndef split_loading(df):\n    df['loading_one'] = df['loading'].astype(int)\n    df['loading_two'] = df['loading'] - df['loading'].astype(int)\n    return df\n\n#trn_data = split_loading(trn_data)\n#tst_data = split_loading(tst_data)\n")


# # 5.5 Feature Engineering -- Aggregated Variables Across Columns

# In[37]:


get_ipython().run_cell_magic('time', '', "# ...\ncols = ['measurement_0',\n        'measurement_1',\n        'measurement_2',\n        'measurement_3',\n        'measurement_4',\n        'measurement_5',\n        'measurement_6',\n        'measurement_7',\n        'measurement_8',\n        'measurement_9',\n        'measurement_10',\n        'measurement_11',\n        'measurement_12',\n        'measurement_13',\n        'measurement_14',\n        'measurement_15',\n        'measurement_16',\n        'measurement_17',\n       ]\n\ncols = [f'measurement_{i}' for i in range(3, 17)]\ndef aggregate_cols(df, cols):\n    '''\n    '''\n    df['avg_meas'] = df[cols].mean(axis = 1)\n    df['min_meas'] = df[cols].min(axis = 1)\n    df['max_meas'] = df[cols].max(axis = 1)\n    df['std_meas'] = df[cols].std(axis = 1)\n    df['mad_meas'] = df[cols].mad(axis = 1)\n    df['var_meas'] = df[cols].var(axis = 1)\n    df['sum_meas'] = df[cols].sum(axis = 1)\n    df['pro_meas'] = df[cols].prod(axis = 1)\n    return df\n\ntrn_data = aggregate_cols(trn_data, cols)\ntst_data = aggregate_cols(tst_data, cols)\n")


# # 5.6 Feature Engineering -- Attribute 2, 3 Features

# In[38]:


get_ipython().run_cell_magic('time', '', "# ...\ndef attribute_functions(df, cols = ['attribute_2', 'attribute_3']):\n    df['attribute_diff'] = np.abs(df['attribute_2'] - df['attribute_3'])\n    df['attribute_sum'] = df['attribute_2'] + df['attribute_3']\n    df['attribute_ratio'] = df['attribute_2'] / df['attribute_3']\n    df['attribute_prod'] = df['attribute_2'] * df['attribute_3']\n    df['meas_ratio'] = df['measurement_17'] / df['attribute_2']\n    return df\n\ntrn_data = attribute_functions(trn_data, cols = ['attribute_2', 'attribute_3'])\ntst_data = attribute_functions(tst_data, cols = ['attribute_2', 'attribute_3'])\n")


# # 5.7 Feature Engineering -- Creating a Rank Variable from Loading

# In[39]:


get_ipython().run_cell_magic('time', '', "# ...\ntrn_data['rank_load'] = trn_data['loading'].rank()\ntst_data['rank_load'] = tst_data['loading'].rank()\n")


# # 5.8 Mean Encoded Features

# In[40]:


get_ipython().run_cell_magic('time', '', "# ...\ndef mean_encode(trn_df,tst_df, encode_groups = ['attribute_0'], target = 'failure', new_name = 'mean_encoded_att_0'):\n    aggregated_results = trn_df.groupby(encode_groups)[target].mean().reset_index()\n    aggregated_results = aggregated_results.rename(columns = {'failure': new_name})\n    \n    trn_df = trn_df.merge(aggregated_results, how = 'left', on = encode_groups)\n    tst_df = tst_df.merge(aggregated_results, how = 'left', on = encode_groups)\n    \n    return trn_df, tst_df\n\ntrn_data, tst_data = mean_encode(trn_data,tst_data, encode_groups = ['attribute_0'], target = 'failure', new_name = 'mean_encoded_att_0')\ntrn_data, tst_data = mean_encode(trn_data,tst_data, encode_groups = ['attribute_1'], target = 'failure', new_name = 'mean_encoded_att_1')\ntrn_data, tst_data = mean_encode(trn_data,tst_data, encode_groups = ['attribute_2'], target = 'failure', new_name = 'mean_encoded_att_2')\ntrn_data, tst_data = mean_encode(trn_data,tst_data, encode_groups = ['attribute_3'], target = 'failure', new_name = 'mean_encoded_att_3')\n")


# In[41]:


get_ipython().run_cell_magic('time', '', '# ...\ntrn_data = trn_data.fillna(0)\ntst_data = tst_data.fillna(0)\n')


# In[42]:


get_ipython().run_cell_magic('time', '', "tst_data[['attribute_1', 'mean_encoded_att_1']].sample(15)\n")


# ---

# # 6.0 Pre-Processing Dataset for Model Training

# # 6.1 Feature Standarization

# In[43]:


get_ipython().run_cell_magic('time', '', "# ...\ndef scale_variables(df, numeric_feat = ['loading']):\n    '''\n    \n    '''\n    scaler = PowerTransformer()\n    df[numeric_feat] = scaler.fit_transform(df[numeric_feat])\n    return df\n\n#trn_data = scale_variables(trn_data, measure_cols)\n#tst_data = scale_variables(tst_data, measure_cols)\n")


# # 6.2 Label Encoding Categorical Features

# In[44]:


get_ipython().run_cell_magic('time', '', "# ...\ndef encode_labels(df, text_features = ['product_code']):\n    for categ_col in df[text_features].columns:\n        encoder = LabelEncoder()\n        df[categ_col + '_enc'] = encoder.fit_transform(df[categ_col])\n    return df\n\ntrn_data = encode_labels(trn_data, text_features = ['product_code', 'attribute_0', 'attribute_1', 'attribute_2', 'attribute_3'])\ntst_data = encode_labels(tst_data, text_features = ['product_code', 'attribute_0', 'attribute_1', 'attribute_2', 'attribute_3'])\n")


# In[45]:


def one_hot(train_df, test_df, encoded_columns = ['attribute_0', 'attribute_1']):
    '''
    
    '''
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['failure'] = -999
    
    full_data = train_df.append(test_df)
    full_data = pd.get_dummies(full_data, columns=encoded_columns)
    
    tmp_trn = full_data[full_data['is_train'] == 1].drop('is_train', axis = 1)
    tmp_tst = full_data[full_data['is_train'] == 0].drop(['is_train', 'failure'], axis = 1)
    return tmp_trn, tmp_tst

trn_data, tst_data = one_hot(trn_data, tst_data, encoded_columns = ['attribute_0', 'attribute_1'])


# In[46]:


trn_data


# In[47]:


tst_data


# # 6.3 Selection of the Final Features for Training Steps

# In[48]:


get_ipython().run_cell_magic('time', '', "# ...\ntarget = ['failure']\n\nignore = ['id', \n          #'loading',\n          'loading_one', \n          'loading_two',\n          'product_code',\n         ] + target\n\nfeatures = [feat for feat in trn_data.columns if feat not in ignore]\nprint(f'All Features: {features} ...')\n\n# Identifying numeric columns\nmeasure_cols = ([col for col in trn_data.columns if 'meas' in col] \n                + ['loading']\n               )\nfeatures\n")


# In[49]:


get_ipython().run_cell_magic('time', '', '# ...\ntrn_data[features].info()\n')


# ---

# # 7.0 Advance Feature Engineering, Using K-means

# # 7.1 Identify the Optimal Number of Clusters

# In[50]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Placeholder, Describe Code...\n\ndistortions = []\ninertias = []\n\nmapping1 = {}\nmapping2 = {}\n\nmax_centroids = 15\n\nK = range(1, max_centroids)\n\n\nfor k in K:\n    # Building and fitting the model\n    kmeanModel = KMeans(n_clusters = k).fit(trn_data[features])\n  \n    distortions.append(sum(np.min(cdist(trn_data[features], kmeanModel.cluster_centers_,\n                                        'euclidean'), axis=1)) / trn_data[features].shape[0])\n    inertias.append(kmeanModel.inertia_)\n  \n    mapping1[k] = sum(np.min(cdist(trn_data[features], kmeanModel.cluster_centers_,\n                                   'euclidean'), axis=1)) / trn_data[features].shape[0]\n    mapping2[k] = kmeanModel.inertia_\n")


# In[51]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Placeholder, Describe Code...\nfor key, val in mapping1.items():\n    print(f'{key} : {val}')\n")


# # 7.2 Plot the Cluster Analysis to Identify Inflection Point

# In[52]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Placeholder, Describe Code...\nplt.plot(K, distortions, 'bx-')\nplt.xlabel('Values of K')\nplt.ylabel('Distortion')\nplt.title('The Elbow Method using Distortion')\nplt.show()\n")


# # 7.3 Train a K-Means Model using the Optimal Number of Clusters

# In[53]:


features


# In[54]:


#%%script false --no-raise-error
#%%time
# ...
kmeans = KMeans(n_clusters = 10)

trn_data['cluster'] = kmeans.fit_predict(trn_data[features])
tst_data['cluster'] = kmeans.predict(tst_data[features])


# In[55]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nfeatures = [\n             'loading',\n             'attribute_2',\n             'attribute_3',\n             'measurement_0',\n             'measurement_1',\n             'measurement_2',\n             'measurement_3',\n             'measurement_4',\n             'measurement_5',\n             'measurement_6',\n             'measurement_7',\n             'measurement_8',\n             'measurement_9',\n             'measurement_10',\n             'measurement_11',\n             'measurement_12',\n             'measurement_13',\n             'measurement_14',\n             'measurement_15',\n             'measurement_16',\n             'measurement_17',\n             'loading_missing',\n             'measurement_3_missing',\n             'measurement_4_missing',\n             'measurement_5_missing',\n             'measurement_6_missing',\n             'measurement_7_missing',\n             'measurement_8_missing',\n             'measurement_9_missing',\n             'measurement_10_missing',\n             'measurement_11_missing',\n             'measurement_12_missing',\n             'measurement_13_missing',\n             'measurement_14_missing',\n             'measurement_15_missing',\n             'measurement_16_missing',\n             'measurement_17_missing',\n             'missing_data',\n             'avg_meas',\n             'min_meas',\n             'max_meas',\n             'std_meas',\n             'mad_meas',\n             'var_meas',\n             'sum_meas',\n             'pro_meas',\n             'attribute_diff',\n             'attribute_sum',\n             'attribute_ratio',\n             'attribute_prod',\n             'rank_load',\n             'mean_encoded_att_0',\n             'mean_encoded_att_1',\n             'mean_encoded_att_2',\n             'mean_encoded_att_3',\n             'attribute_0_5',\n             'attribute_0_7',\n             'attribute_1_5',\n             'attribute_1_6',\n             'attribute_1_7',\n             'attribute_1_8',\n             'cluster'\n            ]\n")


# # 8.0 Creating a Train and Validation Datasets, Using a 80/20 Rule

# In[56]:


get_ipython().run_cell_magic('time', '', '# ...\nSEED = 78\n\nx_train, x_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target], test_size = 0.10, random_state = SEED)\n')


# ---

# # 8.0 Recursive Feature Elimination, RFE for Classification

# # 8.1 Defining Some Auxiliary Functions

# In[57]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Defines a scoring function for model evaluation.\ndef score_model(label, pred, pred_probability):\n    '''\n    Calculates AUC and ACC scores based on the predictions and predicted probability\n    \n    '''\n    \n    acc_score = accuracy_score(label, pred)\n    auc_score = roc_auc_score(label, pred_probability)\n    \n    return print(f'ACC: {acc_score: .4f} | AUC: {auc_score: .4f}')\n")


# # 8.2 Recursive Feature Elimination, Using Sklearn

# In[58]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Define the method\nlog_reg = LogisticRegression(penalty = 'elasticnet', l1_ratio = 0.8, C = 0.007, tol = 1e-2, solver = 'saga', max_iter = 1000, random_state = SEED)\nrfe = RFECV(estimator = log_reg, cv = 10, scoring = 'roc_auc', verbose = 0)\n\n# Fit the model\nrfe.fit(x_train, y_train)\n")


# In[59]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Report performance\npred = rfe.predict(x_valid)\npred_proba = rfe.predict_proba(x_valid)[:,1]\nscore_model(y_valid, pred, pred_proba)\n')


# In[60]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Report optimal features selected\nrfe_results = pd.DataFrame({'feature_name': features, 'selected': rfe.support_, 'ranking': rfe.ranking_})\nrfe_results = rfe_results.sort_values(['ranking', 'selected'], ascending=True)\nrfe_results.head(15)\n\noptimal_features = list(rfe_results[rfe_results['selected'] == True]['feature_name'])\nprint(optimal_features)\n")


# In[61]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Report optimal features selected\nrfe_results.head(15)\n')


# # 8.3 Recursive Feature Elimination, Using Boruta

# ## 8.3.1 Creating a Baseline Model to Compare the Value of RFE 

# In[62]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Train a baseline model with the same parameters for comparison after feature selection\nrf_all_features = RandomForestClassifier(random_state = SEED, n_estimators = 1000, max_depth = 16)\nrf_all_features.fit(x_train, y_train) \n\ny_pred = rf_all_features.predict(x_valid)\ny_pred_prob = rf_all_features.predict_proba(x_valid)[:, 1]\n\nscore_model(y_valid, y_pred, y_pred_prob)\n')


# ## 8.3.2 Training and RFE Model

# In[63]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Define the method\nrfc = RandomForestClassifier(random_state = 1, n_estimators = 1000, max_depth = 16)\nboruta_selector = BorutaPy(rfc, n_estimators = 'auto', verbose = 0, random_state = 1)\n\n# Fit the model\nboruta_selector.fit(x_train.values, y_train.values)  \n")


# ## 8.3.3 Selecting Optimal Features from RFE

# In[64]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# ...\nprint("Ranking: ",boruta_selector.ranking_)          \nprint("No. of significant features: ", boruta_selector.n_features_)\n')


# In[65]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nselected_rf_features = pd.DataFrame({'Feature':list(x_train.columns),'Ranking':boruta_selector.ranking_})\nselected_rf_features.sort_values(by='Ranking').head()\n")


# ## 8.3.4 Training a new ML Model using the Optimal Features from RFE

# In[66]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Train a new model only on the selected features\nx_important_train = boruta_selector.transform(np.array(x_train))\nx_important_valid = boruta_selector.transform(np.array(x_valid))\n')


# In[67]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Report performance\nrf_boruta = RandomForestClassifier(random_state = SEED, n_estimators = 5000, max_depth = 16)\nrf_boruta.fit(x_important_train, y_train)\n\ny_pred = rf_boruta.predict(x_important_valid)\ny_pred_prob = rf_boruta.predict_proba(x_important_valid)[:, 1]\n\nscore_model(y_valid, y_pred, y_pred_prob)\n')


# ---

# # 9.0 Machine Learning Model Implementation; Simple 80/20 Validation
# ...

# # 9.1 Defining Some Auxiliary Functions

# ## 9.1.1 Model Performance Evaluation

# In[68]:


get_ipython().run_cell_magic('time', '', "# Defines a scoring function for model evaluation.\ndef score_model(label, pred, pred_probability):\n    '''\n    Calculates AUC and ACC scores based on the predictions and predicted probability\n    \n    '''\n    \n    acc_score = accuracy_score(label, pred)\n    auc_score = roc_auc_score(label, pred_probability)\n    \n    return print(f'ACC: {acc_score: .4f} | AUC: {auc_score: .4f}')\n")


# ## 9.1.1 Machine Learning Explainability Function

# In[69]:


get_ipython().run_cell_magic('time', '', "# ...\ndef plot_feature_importance(importance, names, model_type, max_features = 10):\n    '''\n    \n    '''\n    \n    # Create arrays from feature importance and feature names\n    feature_importance = np.array(importance)\n    feature_names = np.array(names)\n\n    # Create a DataFrame using a Dictionary\n    data={'feature_names':feature_names,'feature_importance':feature_importance}\n    fi_df = pd.DataFrame(data)\n\n    # Sort the DataFrame in order decreasing feature importance\n    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)\n    fi_df = fi_df.head(max_features)\n\n    # Define size of bar plot\n    plt.figure(figsize=(8,6))\n    \n    # Plot Searborn bar chart\n    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])\n    \n    # Add chart labels\n    plt.title(model_type + 'FEATURE IMPORTANCE')\n    plt.xlabel('FEATURE IMPORTANCE')\n    plt.ylabel('FEATURE NAMES')\n    \n    return None\n")


# ---

# # 9.2 Random Forest Classifier
# Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees.

# ## 9.2.1 Model Parameter Configuration

# In[70]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nrf_params = {'n_estimators' : 2048,\n             'max_depth' : 32,\n             'random_state' : SEED,\n             'n_jobs' : -1\n            }\n")


# ## 9.2.2 Model Instantiation and Training

# In[71]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# ...\ncls = RandomForestClassifier(**rf_params)\ncls.fit(x_train, y_train)\n\ny_pred = cls.predict(x_valid)\ny_pred_prob = cls.predict_proba(x_valid)[:, 1]\ny_test_pred_prob = cls.predict_proba(tst_data[features])[:, 1]\n\nscore_model(y_valid, y_pred, y_pred_prob)\n')


# ## 9.2.3 Model Results and Submission

# In[72]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nsubmission['failure'] = y_test_pred_prob\nsubmission.to_csv('rf_submission.csv', index = False)\nsubmission.head()\n")


# ---

# # 9.3 Extreme Gradient Boosted Tree Classifier (XGBOOST)
# XGBoost, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

# ## 9.3.1 Model Parameter Configuration

# In[73]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nxgb_params = {'n_estimators': 4096,\n              'max_depth': 7,\n              'learning_rate': 0.15,\n              'subsample': 0.95,\n              'colsample_bytree': 0.60,\n              'reg_lambda': 1.50,\n              'reg_alpha': 6.10,\n              'gamma': 1.40,\n              'random_state': SEED,\n              'objective': 'binary:logistic',\n              'tree_method': 'gpu_hist',\n             }\n")


# ## 9.3.2 Model Instantiation and Training

# In[74]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nxgb = XGBClassifier(**xgb_params)\nxgb.fit(x_train, y_train, eval_set = [(x_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 50)\n\n\ny_pred = xgb.predict(x_valid)\ny_pred_prob = xgb.predict_proba(x_valid)[:, 1]\ny_test_pred_prob = xgb.predict_proba(tst_data[features])[:, 1]\n\nscore_model(y_valid, y_pred, y_pred_prob)\n")


# ## 9.3.3 Machine Learning Explainability

# In[75]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nplot_feature_importance(xgb.feature_importances_, x_train.columns,'XG BOOST ', max_features = 10)\n")


# ## 9.3.4 Model Results and Submission

# In[76]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nsubmission['failure'] = y_test_pred_prob\nsubmission.to_csv('xgb_submission.csv', index = False)\nsubmission.head()\n")


# ---

# # 9.4 Logistic Regression 

# ## 9.4.1 Model Parameter Configuration

# In[77]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# Define the LR model parameters\nlr_params = {'penalty' : 'elasticnet',\n             'l1_ratio' : 0.8,\n             'C' : 0.007,\n             'tol' : 1e-2,\n             'solver' : 'saga',\n             'max_iter' : 1000,\n             'random_state' : SEED\n            }\n")


# ## 9.4.2 Model Instantiation and Training

# In[78]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', '%%time\n# Baseline Model Using Random Forest Classifier\nlr = LogisticRegression(**lr_params)\nlr.fit(x_train, y_train)\n\ny_pred = lr.predict(x_valid)\ny_pred_prob = lr.predict_proba(x_valid)[:,1]\ny_test_pred_prob = lr.predict_proba(tst_data[features])[:,1]\n\nscore_model(y_valid, y_pred, y_pred_prob)\n')


# ## 9.3.3 Model Results and Submission

# In[79]:


get_ipython().run_cell_magic('script', 'false --no-raise-error', "%%time\n# ...\nsubmission['failure'] = y_test_pred_prob\nsubmission.to_csv('lr_submission.csv', index = False)\nsubmission.head()\n")


# ---

# # 10.0 Cross Validations Loop Models

# In[80]:


get_ipython().run_cell_magic('time', '', "# ...\ntrn_data['loading_square'] = trn_data['loading'] ** 2\ntst_data['loading_square'] = tst_data['loading'] ** 2\n\ntrn_data['loading_cube'] = trn_data['loading'] ** 3\ntst_data['loading_cube'] = tst_data['loading'] ** 3\n\ntrn_data['loading_exp4'] = trn_data['loading'] ** 4\ntst_data['loading_exp4'] = tst_data['loading'] ** 4\n\ntrn_data['loading_exp5'] = trn_data['loading'] ** 5\ntst_data['loading_exp5'] = tst_data['loading'] ** 5\n\ntrn_data['loading_sqrt'] = trn_data['loading'] ** 0.5\ntst_data['loading_sqrt'] = tst_data['loading'] ** 0.5\n\ntrn_data['loading_logn'] = np.log(trn_data['loading']) \ntst_data['loading_logn'] = np.log(tst_data['loading'])\n")


# In[81]:


get_ipython().run_cell_magic('time', '', "# ...\nfeatures = [\n            'loading',\n            #'loading_square',\n            #'loading_cube',\n            #'loading_exp4',\n            #'loading_exp5',\n            #'loading_sqrt',\n            #'loading_logn',\n            #'attribute_0',\n            #'attribute_1',\n            #'attribute_2',\n            #'attribute_3',\n            'measurement_0',\n            'measurement_1',\n            'measurement_2',\n            #'measurement_3',\n            #'measurement_4',\n            #'measurement_5',\n            #'measurement_6',\n            #'measurement_7',\n            #'measurement_8',\n            #'measurement_9',\n            #'measurement_10',\n            #'measurement_11',\n            #'measurement_13',\n            #'measurement_14',\n            #'measurement_15',\n            #'measurement_16',\n            'measurement_17',\n            #'loading_missing',\n            'measurement_3_missing',\n            #'measurement_4_missing',\n            'measurement_5_missing',\n            #'measurement_6_missing',\n            #'measurement_7_missing',\n            #'measurement_8_missing',\n            #'measurement_9_missing',\n            #'measurement_10_missing',\n            #'measurement_11_missing',\n            #'measurement_12_missing',\n            #'measurement_13_missing',\n            #'measurement_14_missing',\n            #'measurement_16_missing',\n            #'measurement_17_missing',\n            #'missing_data',\n            'avg_meas',\n            #'min_meas',\n            #'max_meas',\n            #'std_meas',\n            #'mad_meas',\n            #'var_meas',\n            #'sum_meas',\n            #'pro_meas',\n            #'meas_ratio',\n            #'attribute_diff',\n            #'attribute_sum',\n            #'attribute_ratio',\n            'attribute_prod',\n            #'rank_load',\n            #'attribute_0_5',\n            #'attribute_0_7',\n            #'attribute_1_5',\n            #'attribute_1_6',\n            #'attribute_1_7',\n            #'attribute_1_8',\n            #'attribute_0_enc', ***\n            #'attribute_1_enc',\n            #'attribute_2_enc',\n            #'attribute_3_enc',\n            #'product_code_enc',\n            #'mean_encoded_att_0', ***\n            #'mean_encoded_att_1',\n            #'mean_encoded_att_2',\n            #'mean_encoded_att_3',\n            'cluster',\n            ]\n")


# In[82]:


# Identifying numeric columns
measure_cols = [col for col in features if ('meas' in col or 'mean' in col or 'load' in col or 'prod' in col or 'ratio' in col) and 'product_code_enc' not in col]


# In[83]:


measure_cols


# In[84]:


features


# # 10.1 Train Multiple Models, Defining Model Parameters

# In[85]:


get_ipython().run_cell_magic('time', '', '# Define the LR model parameters\n\n#lr_params = {\'penalty\' : \'l2\', \'C\' : 9.33871677744771e-05, \'tol\' : 0.0009623735504317714, \'solver\' : \'saga\', \'max_iter\' : 638, \'l1_ratio\': 0.01859239186414026, \'fit_intercept\': True, \'random_state\' : SEED}\n\n#lr_params = {\'penalty\' : \'elasticnet\',\'l1_ratio\' : 0.8,\'C\' : 0.007,\'tol\' : 1e-2,\'solver\' : \'saga\',\'max_iter\' : 1000,\'random_state\' : SEED}\n\n#lr_params = {\'solver\': \'saga\', \'class_weight\': None, \'warm_start\': False, \'max_iter\': 280, \'fit_intercept\': True, \'tol\': 0.0009295509208633467, \'C\': 9.863985380751957e-05, \'l1_ratio\': 0.022532816207275097, \'random_state\' : SEED}\n\nlr_params = {\'penalty\' : \'elasticnet\',\n             \'l1_ratio\' : 0.8,\n             \'C\' : 0.007,\n             \'tol\' : 1e-2,\n             \'solver\' : \'saga\',\n             \'max_iter\' : 1000,\n             \'random_state\' : SEED\n            }\n\n#lr_params = {\'solver\': \'saga\', \'class_weight\': None, \'warm_start\': False, \'max_iter\': 280, \'fit_intercept\': True, \'tol\': 0.0009295509208633467, \'C\': 9.863985380751957e-05, \'l1_ratio\': 0.022532816207275097, \'random_state\' : SEED}\n\n#lr_params = {"max_iter": 200, "C": 0.0001, "penalty": "l2", "solver": "newton-cg"}\n\n#lr_params = {\'solver\': \'sag\', \'class_weight\': None, \'warm_start\': False, \'max_iter\': 894, \'fit_intercept\': True, \'tol\': 0.07181796363886127, \'C\': 9.093745343170404e-05}\n\nlgb_params = {\'n_estimators\': 8192,\n              \'random_state\': SEED,\n              \'n_jobs\': -1, \n              \'lambda_l2\': 2, \n              \'metric\': "auc", \n              \'max_depth\': -1, \n              \'num_leaves\': 100, \n              \'boosting\': \'gbdt\', \n              \'bagging_freq\': 10, \n              \'learning_rate\': 0.01, \n              \'objective\': \'binary\', \n              \'min_data_in_leaf\': 40, \n              \'num_boost_round\': 70, \n              \'feature_fraction\': 0.90, \n              \'bagging_fraction\': 0.90}\n\nxgb_params = {\'n_estimators\': 8192,\n              \'max_depth\': 16,\n              \'learning_rate\': 0.10,\n              \'subsample\': 0.95,\n              \'colsample_bytree\': 0.90,\n              \'reg_lambda\': 1.50,\n              \'reg_alpha\': 6.10,\n              \'gamma\': 1.40,\n              \'random_state\': SEED,\n              \'objective\': \'binary:logistic\',\n              \'tree_method\': \'gpu_hist\',\n             }\n\nrf_params = {\'n_estimators\' : 2048,\n             \'max_depth\' : 32,\n             \'random_state\' : SEED,\n             \'n_jobs\' : -1\n            }\n')


# In[86]:


#oversample = SMOTE()
#trn_data[features], trn_data[target] = oversample.fit_resample(trn_data[features], trn_data[target])


# # 10.2 Train Multiple Models, Based on the Selection, Using CV Loop

# In[87]:


get_ipython().run_cell_magic('time', '', "# Create empty lists to store model information\nacc_score_list   = []\nauc_score_list   = []\npredictions  = []\noof_pred = np.zeros(trn_data.shape[0])\noof_pred_probability = np.zeros(trn_data.shape[0])\noof_target = np.zeros((trn_data.shape[0],1))\n\nSPLITS = 5\n\n# Define kfolds for training purposes\ngkf = GroupKFold(n_splits = SPLITS)\nskf = StratifiedKFold(n_splits = SPLITS, shuffle = True, random_state = SEED)\nkf = KFold(n_splits = SPLITS, shuffle = True, random_state = SEED)\n\n# Start the CV loop\nprint(f'********** Initializating CV Loop **********', '\\n')\n#for fold, (trn_idx, val_idx) in enumerate(gkf.split(trn_data[features], trn_data[target], trn_data['product_code'])):\nfor fold, (trn_idx, val_idx) in enumerate(skf.split(trn_data[features], trn_data[target])):  \n#for fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data[features], trn_data[target])):  \n    \n    print(f'Training Fold {fold} ...')\n    \n    # ---- Creates train and validation datasets ----\n    x_train, x_valid = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_valid = trn_data.iloc[trn_idx][target], trn_data.iloc[val_idx][target]\n    \n    # ---- Scales the features, for model training inside CV loop ----    \n    # scaler = StandardScaler()\n    # scaler = PowerTransformer()\n    # scaler = MinMaxScaler()\n    # scaler = RobustScaler()\n    \n    scaler = StandardScaler()\n    x_train[measure_cols] = scaler.fit_transform(x_train[measure_cols])\n    x_valid[measure_cols] = scaler.transform(x_valid[measure_cols])\n    \n    tst_data_scaled = tst_data[features].copy(deep = True)\n    tst_data_scaled[measure_cols] = scaler.transform(tst_data[measure_cols])\n    \n    # ---- Train the selected machine learning model ----\n    \n    # LGBM (Uncomment to use, and Comment the XGBoost Part... LGBM Takes forever)...\n    #model = LGBMClassifier(**lgb_params)\n    #model.fit(x_train, y_train, eval_set = [(x_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 512, verbose = 0)\n    # ......................\n    \n    # XGBoost...\n    #model = XGBClassifier(**xgb_params)\n    #model.fit(x_train, y_train, eval_set = [(x_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 128, verbose = 0)\n    # ......................\n    \n    # LR Model...\n    model = LogisticRegression(**lr_params)\n    model.fit(x_train, y_train)\n    # ......................\n    \n    # ---- Evaluates model performance ----\n    \n    y_pred = model.predict(x_valid.values)\n    y_pred_prob = model.predict_proba(x_valid.values)[:,1]\n    \n    oof_pred[val_idx] = y_pred\n    oof_pred_probability[val_idx] = y_pred_prob\n    oof_target[val_idx] = y_valid\n    \n    # ---- Log performance results ----\n    \n    score_model(y_valid, y_pred, y_pred_prob)\n    print((''))\n    \n    # ---- Generate model predicion on the test set for submission ----\n    \n    tst_pred = model.predict_proba(tst_data_scaled.values)[:,1]\n    predictions.append(tst_pred)\n\n# --- Evaluate and report model performance outside the CV loop ----\nprint('****************************************************')\nscore_model(oof_target, oof_pred, oof_pred_probability)\nprint('****************************************************')\n")


# In[88]:


# ACC:  0.7874 | AUC:  0.5876


# In[89]:


# ACC:  0.7874 | AUC:  0.5831


# In[90]:


get_ipython().run_cell_magic('time', '', "# ...\nsubmission['failure'] = np.mean(predictions, axis = 0)\nsubmission.to_csv('lr_cv_submission.csv', index = False)\nsubmission.head()\n")


# In[91]:


# ACC:  0.7873 | AUC:  0.5902 >>> Best Model...


# ---

# # 11.0 Optuna Hyper-Param Optimization

# In[92]:


# import optuna

# def objective(trial):
#     param = {
#     'random_state':1,
#     'solver' :trial.suggest_categorical('solver', ["liblinear", "newton-cg",'lbfgs','sag','saga']),
#     'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
#     'warm_start' : trial.suggest_categorical('warm_start' , [False, True]),
#     'max_iter' : trial.suggest_int('max_iter', 20, 2000),
#     'fit_intercept':trial.suggest_categorical('fit_intercept', [False, True]),
#     'tol':trial.suggest_uniform('tol', 9e-4, 1e-1),
#     'C' : trial.suggest_uniform("C", 9e-5, 1e-3)}
    
#     if param['solver'] == 'liblinear': param['penalty'] =  trial.suggest_categorical('penalty', ['l1', 'l2'])
#     if param['solver'] == 'newton-cg': param['penalty'] = 'l2'
#     if param['solver'] == 'lbfgs': param['penalty'] = 'l2'  
#     if param['solver'] == 'sag': param['penalty'] = 'l2'
    
#     if param['solver'] == 'saga':
#         param['penalty'] = 'elasticnet'
#         param['l1_ratio'] = trial.suggest_uniform('l1_ratio', 0.01, 0.99)
   
#     x_train, x_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target], test_size = 0.05, random_state = SEED)
    
#     scaler = StandardScaler()
#     x_train[measure_cols] = scaler.fit_transform(x_train[measure_cols])
#     x_valid[measure_cols] = scaler.transform(x_valid[measure_cols])
    
#     lr = LogisticRegression(**param)
#     lr.fit(x_train, y_train)
#     score = roc_auc_score(y_valid, lr.predict_proba(x_valid)[:,1])
#     return np.round(score, 5)
    
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# study = optuna.create_study(direction = 'maximize')
# study.optimize(objective, n_trials = 1000, show_progress_bar = True)

# print('Number of finished trials:', len(study.trials))
# print('Best trial auc:',study.best_trial.values)
# print('Best trial parameters:', study.best_trial.params)


# ---
