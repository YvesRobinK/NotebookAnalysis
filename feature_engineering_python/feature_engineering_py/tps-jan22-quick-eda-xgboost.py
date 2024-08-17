#!/usr/bin/env python
# coding: utf-8

# # 🎁 TPS-JAN22, Quick EDA + XGBoost
# The following model is a simple implementation using XGBoost. The objective is to provide a simple framework and foundation as a baseline for more sophisticated implementations.
# The objective of this competition is the following.
# 
# 1. [Loading Python Libraries.](#1)
# 2. [Loading CSV and Creating Dataframes.](#2)
# 3. [Exploring the Dataframes, (Size, Stats, Nulls and Others).](#3)
# 4. [Feature Engineering.](#4)
# 5. [Processing the Datasets for Training.](#5)
# 6. [Creates a Simple Train / Validation Strategy](#6)
# 7. [Train a Simple Model (XGBoost Regressor)](#7)
# 8. [Train a Simple Model (XGBoost Regressor) using a CV Loop](#8)
# 9. [Model Inference (Submission to Kaggle)](#9)
# 
# 
# **Data Description** </br>
# For this challenge, you will be predicting a full year worth of sales for three items at two stores located in three different countries. This dataset is completely fictional, but contains many effects you see in real-world data, e.g., weekend and holiday effect, seasonality, etc. The dataset is small enough to allow you to try numerous different modeling approaches.
# 
# Good luck!
# 
# 
# 
# **Objective** </br>
# Using 2015 - 2018, predict the sales by date, country, store, and product for 2019.
# 
# **Strategy** </br>
# Because we are dealing with a time series type of estimation, we need to hide future information from the model; in this simple approach we will use as validation all the data from 2018, so we will train the model with data from 2015-2017
# 
# **Update 12/31/2021**
# * Developed a simple Notebook, Quick EDA + Simple Feature Engineering.
# * Cross-Validation strategy based on a fixed date.
# 
# **Update 01/01/2021**
# * Added Cross-Validation loop to the model.
# * Added new features, to identify weekends.
# * Added a proper table of contents.
# * Added features based on Holidays for each of the countries.
# 
# **Update 01/02/2021**
# * Improved the CV training function to calculate the SMOTE properly.
# 
# **Ideas that I want to implement**
# * New features based on trends.
# 
# 

# ---

# In[1]:


get_ipython().system('pip install holidays')


# <a name="1"></a>
# # Loading Python Libraries. 

# In[2]:


get_ipython().run_cell_magic('time', '', '#This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[3]:


get_ipython().run_cell_magic('time', '', '# Import LGBM Regressor Model...\n\nfrom xgboost import XGBRegressor\nfrom sklearn.metrics import mean_squared_error\nfrom sklearn.model_selection import StratifiedKFold, TimeSeriesSplit\n\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\n\nimport holidays\n')


# In[4]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', 15) \npd.set_option('display.max_rows', 50)\n")


# In[6]:


get_ipython().run_cell_magic('time', '', '# Define some of the notebook parameters for future experiment replication.\nSEED   = 42\n')


# ---

# # Loading CSV and Creating Dataframes. <a name="2"></a>

# In[7]:


get_ipython().run_cell_magic('time', '', "# Define the datasets locations...\n\nTRN_PATH = '/kaggle/input/tabular-playground-series-jan-2022/train.csv'\nTST_PATH = '/kaggle/input/tabular-playground-series-jan-2022/test.csv'\nSUB_PATH = '/kaggle/input/tabular-playground-series-jan-2022/sample_submission.csv'\n")


# In[8]:


get_ipython().run_cell_magic('time', '', '# Read the datasets and create dataframes...\n\ntrain_df = pd.read_csv(TRN_PATH)\ntest_df = pd.read_csv(TST_PATH)\nsubmission_df = pd.read_csv(SUB_PATH)\n')


# ---

# # Exploring the Dataframes, (Size, Stats, Nulls and Others) <a name="3"></a>

# In[9]:


get_ipython().run_cell_magic('time', '', '# Explore the size of the dataset loaded...\n\ntrain_df.info()\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Explore the first 5 rows to have an idea what we are dealing with...\n\ntrain_df.head()\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '# Explore the size of the dataset loaded...\n\ntest_df.info()\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Explore the first 5 rows to have an idea what we are dealing with, in this case the Test Set...\n\ntest_df.head()\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Review some statistical information for the numeric variables...\n\ntrain_df.describe()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', "# Review some information for the categorical variables...\n\ncountry_list = train_df['country'].unique()\nstore_list = train_df['store'].unique()\nproduct_list = train_df['product'].unique()\n\nprint(f'Country List:{country_list}')\nprint(f'Store List:{store_list}')\nprint(f'Product List:{product_list}')\n")


# In[15]:


get_ipython().run_cell_magic('time', '', '# Review if there is missing information in the dataset...\n\ntrain_df.isnull().sum()\n')


# In[16]:


# Create a simple function to evaluate the time-ranges of the information provided.
# It will help with the train / validation separations

def evaluate_time(df):
    min_date = df['date'].min()
    max_date = df['date'].max()
    print(f'Min Date: {min_date} /  Max Date: {max_date}')
    return None

evaluate_time(train_df)
evaluate_time(test_df)


# ___

# # Feature Engineering <a name="4"></a>

# In[17]:


TARGET = 'num_sold'


# In[18]:


# Country List:['Finland' 'Norway' 'Sweden']
holiday_FI = holidays.CountryHoliday('FI', years=[2015, 2016, 2017, 2018, 2019])
holiday_NO = holidays.CountryHoliday('NO', years=[2015, 2016, 2017, 2018, 2019])
holiday_SE = holidays.CountryHoliday('SE', years=[2015, 2016, 2017, 2018, 2019])

holiday_dict = holiday_FI.copy()
holiday_dict.update(holiday_NO)
holiday_dict.update(holiday_SE)

train_df['date'] = pd.to_datetime(train_df['date']) # Convert the date to datetime.
train_df['holiday_name'] = train_df['date'].map(holiday_dict)
train_df['is_holiday'] = np.where(train_df['holiday_name'].notnull(), 1, 0)
train_df['holiday_name'] = train_df['holiday_name'].fillna('Not Holiday')

test_df['date'] = pd.to_datetime(test_df['date']) # Convert the date to datetime.
test_df['holiday_name'] = test_df['date'].map(holiday_dict)
test_df['is_holiday'] = np.where(test_df['holiday_name'].notnull(), 1, 0)
test_df['holiday_name'] = test_df['holiday_name'].fillna('Not Holiday')


# In[19]:


train_df.sample(10)


# In[20]:


def add_holydays(df):
    """
    Flag the dataframe with a is_holyday field = 1 if the date is on the
    dictionary of holydays loaded.
    Args
        df
    Returs
        df
    """
    new_years_eve = ['12/31/2015','12/31/2016','12/31/2017','12/31/2018','12/31/2019']
    chrismas_day = ['12/24/2015','12/24/2016','12/24/2017','12/24/2018','12/24/2019']


# In[21]:


# Create some simple features base on the Date field...

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features base on the date variable, the idea is to extract as much 
    information from the date componets.
    Args
        df: Input data to create the features.
    Returns
        df: A DataFrame with the new time base features.
    """
    
    df['date'] = pd.to_datetime(df['date']) # Convert the date to datetime.
    
    # Start the creating future process.
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofmonth'] = df['date'].dt.days_in_month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.weekofyear
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = np.where((df['weekday'] == 5) | (df['weekday'] == 6), 1, 0)
    
    return df


# In[22]:


# Apply the function 'create_time_features' to the dataset...
train_df = create_time_features(train_df)
test_df = create_time_features(test_df)


# ___

# # Processing the Datasets for Training <a name="5"></a>

# In[23]:


# Convert the Categorical variables to one-hoe encoded features...
# It will help in the training process

CATEGORICAL = ['country', 'store', 'product', 'holiday_name']
def create_one_hot(df, categ_colums = CATEGORICAL):
    """
    Creates one_hot encoded fields for the specified categorical columns...
    Args
        df
        categ_colums
    Returns
        df
    """
    df = pd.get_dummies(df, columns=CATEGORICAL)
    return df


def encode_categ_features(df, categ_colums = CATEGORICAL):
    """
    Use the label encoder to encode categorical features...
    Args
        df
        categ_colums
    Returns
        df
    """
    le = LabelEncoder()
    for col in categ_colums:
        df['enc_'+col] = le.fit_transform(df[col])
    return df

train_df = encode_categ_features(train_df)
test_df = encode_categ_features(test_df)


# In[24]:


def transform_target(df, taget = TARGET):
    """
    Apply a log transformation to the target for better optimization 
    during training.
    """
    df[TARGET] = np.log(df[TARGET])
    return df

train_df = transform_target(train_df, TARGET)


# In[25]:


train_df.head()


# In[26]:


# Extract features and avoid certain columns from the dataframe for training purposes...
avoid = ['row_id', 'date', 'num_sold']
FEATURES = [feat for feat in train_df.columns if feat not in avoid]

# Print a list of all the features created...
print(FEATURES)


# In[27]:


# Selecting Features....
print(FEATURES)


# In[28]:


FEATURES = [
            #'country',
            #'store',
            #'product',
            #'holiday_name',
            #'is_holiday',
            'year',
            #'quarter',
            'month',
            'day',
            'dayofweek',
            #'dayofmonth',
            #'dayofyear',
            #'weekofyear',
            #'weekday',
            'is_weekend',
            'enc_country',
            'enc_store',
            'enc_product',
            #'enc_holiday_name'
            ]


# ___

# # Creates a Simple Train / Validation Strategy <a name="6"></a>

# In[29]:


# Creates the Train and Validation sets to train the model...
# Define a cutoff date to split the datasets
CUTOFF_DATE = '2018-01-01'

# Split the data into train and validation datasets using timestamp best suited for timeseries...
X_train = train_df[train_df['date'] < CUTOFF_DATE][FEATURES]
y_train = train_df[train_df['date'] < CUTOFF_DATE][TARGET]

X_val = train_df[train_df['date'] >= CUTOFF_DATE][FEATURES]
y_val = train_df[train_df['date'] >= CUTOFF_DATE][TARGET]


# In[30]:


def SMAPE(y_true, y_pred):
    denominator = (y_true + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)


# ---

# # Train a Simple Model (XGBoost Regressor) <a name="7"></a>

# In[31]:


# Defines a really simple XGBoost Regressor...

xgboost_params = {'eta'              : 0.1,
                  'n_estimators'     : 16384,
                  'max_depth'        : 8,
                  'max_leaves'       : 256,
                  'colsample_bylevel': 0.75,
                  'colsample_bytree' : 0.75,
                  'subsample'        : 0.75, # XGBoost would randomly sample 'subsample_value' of the training data prior to growing trees
                  'min_child_weight' : 512,
                  'min_split_loss'   : 0.002,
                  'alpha'            : 0.08,
                  'lambda'           : 128,
                  'objective'        : 'reg:squarederror',
                  'eval_metric'      : 'rmse', # Originally using RMSE, trying new functions...
                  'tree_method'      : 'gpu_hist',
                  'seed'             : SEED
                  }

# Create an instance of the XGBRegressor and set the model parameters...
regressor = XGBRegressor(**xgboost_params)

# Train the XGBRegressor using the train and validation datasets, 
# Utilizes early_stopping_rounds to control overfitting...
regressor.fit(X_train,
              y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds = 250,
              verbose = 500)


# In[32]:


val_pred = regressor.predict(X_val[FEATURES])
# Convert the target back to non-logaritmic.
val_pred = np.exp(val_pred)
y_val = np.exp(y_val)

score = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'RMSE: {score} / SMAPE: {SMAPE(y_val, val_pred)}')


# ### Model Results vs. Features Used in the Traininf and Validation...
# 1. Plain features, nothing added to the model. Removed Id, Datetime and Target </br>
# RMSE: 141.17269369190075 / SMAPE: 17.040551866223385
# 
# 2. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday' </br>
# RMSE: 66.89475324109723 / SMAPE: 9.30006322183181
# 
# 3. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday', 'quarter' </br>
# RMSE: 67.4018691784641 / SMAPE: 9.343389593022566
# 
# 4. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday', 'quarter' </br>
# Added new Features,'is_holiday'</br>
# RMSE: 66.59882566819414 / SMAPE: 9.477461518875648
# 
# 5. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday', 'quarter' </br>
# Added new Features,'is_holiday', 'is_weekend'</br>
# RMSE: 66.27489712300181 / SMAPE: 9.370856195608114
# 
# 6. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday', 'quarter' </br>
# Added new Features,'is_holiday', 'is_weekend','enc_holiday_name' </br>
# RMSE: 65.93668135230337 / SMAPE: 9.428644170683123
# 
# 7. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofmonth', 'dayofyear', 'weekofyear', 'weekday'</br>
# Added new Features,'is_weekend' </br>
# RMSE: 66.73112188359103 / SMAPE: 9.29087254951728
# 
# 8. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'weekday'</br>
# Added new Features,'is_weekend' </br>
# RMSE: 66.1329325693428 / SMAPE: 9.290678813131464
# 
# 9. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'weekofyear', 'weekday'</br>
# Added new Features,'is_weekend' </br>
# RMSE: 66.13737123847237 / SMAPE: 9.256808780901792
# 
# 10. Added Datetime features,'year', 'month', 'day', 'dayofweek', 'weekday'</br>
# Added new Features,'is_weekend' </br>
# RMSE: 65.40444050929132 / SMAPE: 9.045220024208168
# 
# 11. Added Datetime features,'year', 'month', 'day', 'dayofweek'</br>
# Added new Features,'is_weekend' </br>
# RMSE: 65.20180075198031 / SMAPE: 9.049180607434174
# 
# 12. Added Datetime features,'year', 'month', 'day', 'dayofweek'</br>
# Using RMSE and Log of the Target... </br>
# Added new Features,'is_weekend' </br>
# RMSE: 63.544532908755954 / SMAPE: 8.460984766381136

# In[33]:


feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(FEATURES, regressor.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=45, figsize=(10,5))


# ---

# # Train a Simple Model (XGBoost Regressor) using a CV Loop. <a name="8"></a>

# In[34]:


get_ipython().run_cell_magic('time', '', 'N_SPLITS = 3\nEARLY_STOPPING_ROUNDS = 150 # Will stop training if one metric of one validation data doesn’t improve in last round\nVERBOSE = 0 # Controls the level of information, verbosity\n')


# In[35]:


get_ipython().run_cell_magic('time', '', "# Define a Pipeline to process the data for the Model.\ntransformer = Pipeline(steps=[('scaler',StandardScaler()), ('min_max', MinMaxScaler(feature_range=(0, 1)))])\npreprocessor = ColumnTransformer(transformers=[('first', transformer, FEATURES)])       \n")


# In[36]:


get_ipython().run_cell_magic('time', '', '# Cross Validation Loop for the Classifier.\ndef cross_validation_train(train, labels, test, model, model_params, n_folds = 5):\n    """\n    The following function is responsable of training a model in a\n    cross validation loop and generate predictions on the specified test set.\n    The function provides the model feature importance list as other variables.\n\n    Args:\n    train  (Dataframe): ...\n    labels (Series): ...\n    test   (Dataframe): ...\n    model  (Model): ...\n    model_params (dict of str: int): ...\n\n    Return:\n    classifier  (Model): ...\n    feat_import (Dataframe): ...\n    test_pred   (Dataframe): ...\n    ...\n\n    """\n    # Creates empty place holders for out of fold and test predictions.\n    oof_pred  = np.zeros(len(train)) # We are predicting prob. we need more dimensions.\n    oof_label = np.zeros(len(train))\n    test_pred = np.zeros(len(test)) # We are predicting prob. we need more dimensions\n    val_indexes_used = []\n    \n    # Creates empty place holder for the feature importance.\n    feat_import = np.zeros(len(FEATURES))\n    \n    # Creates Stratified Kfold object to be used in the train / validation\n    # phase of the model.\n    Kf = TimeSeriesSplit(n_splits = n_folds)\n    \n    # Start the training and validation loops.\n    for fold, (train_idx, val_idx) in enumerate(Kf.split(train)):\n        # Creates the index for each fold\n        print(f\'Fold: {fold}\')        \n        train_min_date = train_df.iloc[train_idx][\'date\'].min()\n        train_max_date = train_df.iloc[train_idx][\'date\'].max()\n        \n        valid_min_date = train_df.iloc[val_idx][\'date\'].min()\n        valid_max_date = train_df.iloc[val_idx][\'date\'].max()\n        \n        print(f\'Train Min / Max Dates: {train_min_date} / {train_max_date}\')\n        print(f\'Valid Min / Max Dates: {valid_min_date} / {valid_max_date}\')\n\n        print(f\'Training on {train_df.iloc[train_idx].shape[0]} Records\')\n        print(f\'Validating on {train_df.iloc[val_idx].shape[0]} Records\')\n        \n        # Generates the Fold. Train and Validation datasets\n        X_trn, y_trn = train.iloc[train_idx], labels.iloc[train_idx]\n        X_val, y_val = train.iloc[val_idx], labels.iloc[val_idx]\n        \n        val_indexes_used = np.concatenate((val_indexes_used, val_idx), axis=None)\n        \n        # Instanciate a classifier based on the model parameters\n        regressor = model(**model_params)\n \n        regressor.fit(X_trn, \n                      y_trn, \n                      eval_set = [(X_val, y_val)], \n                      early_stopping_rounds = EARLY_STOPPING_ROUNDS, \n                      verbose = VERBOSE)\n        \n        # Generate predictions using the trained model\n        val_pred = regressor.predict(X_val)\n        oof_pred[val_idx]  = val_pred # store the predictions for that fold.\n        oof_label[val_idx] = y_val # store the true labels for that fold.\n\n        # Calculate the model error based on the selected metric\n        error =  np.sqrt(mean_squared_error(y_val, val_pred))\n\n        # Print some of the model performance metrics\n        print(f\'RMSE: {error}\')\n        print(f\'SMAPE: {SMAPE(y_val, val_pred)}\')\n        print("."*50)\n\n        # Populate the feature importance matrix\n        feat_import += regressor.feature_importances_\n\n        # Generate predictions for the test set\n        test_pred += (regressor.predict(test)) / n_folds\n                        \n    # Calculate the error across all the folds and print the reuslts\n    val_indexes_used = val_indexes_used.astype(int)\n    global_error = np.sqrt(mean_squared_error(labels.iloc[val_indexes_used], oof_pred[val_indexes_used]))\n    \n    print(\'\')\n    print(f\'RMSE: {global_error}...\')\n    print(f\'SMAPE: {SMAPE(labels.iloc[val_indexes_used], oof_pred[val_indexes_used])}...\')\n    \n    return regressor, feat_import, test_pred, oof_label, oof_pred\n')


# In[37]:


get_ipython().run_cell_magic('time', '', '# Uses the cross_validation_train to build and train the model with XGBoost\nxgbr, feat_imp, predictions, oof_label, oof_pred = cross_validation_train(train  = train_df[FEATURES], \n                                                                          labels = train_df[TARGET], \n                                                                          test   = test_df[FEATURES], \n                                                                          model  = XGBRegressor, \n                                                                          model_params = xgboost_params,\n                                                                          n_folds = N_SPLITS\n                                                                          )\n')


# In[38]:


train_df.shape


# In[39]:


oof_label.shape


# In[40]:


oof_pred.shape


# In[41]:


feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(FEATURES, xgbr.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance', ascending=False).plot(kind='bar', rot=45, figsize=(12,5))


# # Model Inference (Submission to Kaggle) <a name="1"></a>

# In[42]:


# Use the created model to predict the sales for 2019...
pred = regressor.predict(test_df[FEATURES])
pred = np.exp(pred)
submission_df['num_sold'] = pred
submission_df.head(10)


# In[43]:


# Creates a submission file for Kaggle...
submission_df.to_csv('submission.csv',index=False)


# In[44]:


# Use the created model to predict the sales for 2019...
pred = regressor.predict(test_df[FEATURES])
submission_df['num_sold'] = predictions
submission_df.head(10)


# # Results and Ideas...

# In[ ]:





# In[ ]:





# In[ ]:




