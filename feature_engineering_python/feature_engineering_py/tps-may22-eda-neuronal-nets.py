#!/usr/bin/env python
# coding: utf-8

# # Predicting States of Manufacturing Control Data; Using Neuronal Nets ⚙️

# **Note: Use GPU for Training...**
# 
# **Objective:** Build a powerfull NN Model that can provide a good estimation.
# 
# **Strategy:** I think I will follow this strategy:
# 
# **Level 1 Getting Started**
# 
# * Quick EDA to identify potential opportunities.
# * Simple pre-processing step to encode categorical features.
# * A basic CV strategy using 90% for TRaining and 10% for Testing.
# * Looking at the feature importances.
# * Creating a submission file.
# * Submit the file to Kaggle.
# 
# **Level 2 Feature Engineering**
# 
# * Feature engineering using text information. (Massive boost in the score)
# * Cross validation loop.
# 
# **Level 3 Model Optimization**
# * Work in Progress...
# 
# ---
# **Other Similar Implementations**
# I been working on other architechtures at the same time, to see what works more effiently
# 
# XGBoost and LGBM Models
# 
# https://www.kaggle.com/code/cv13j0/tps-may22-eda-gbdt
# 
# ---
# 
# 
# 
# 
# **Data Description**
# 
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. 
# The data has various feature interactions that may be important in determining the machine state.
# 
# Good luck!
# 
# **Files**
# * train.csv - the training data, which includes normalized continuous data and categorical data
# * test.csv - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
# * sample_submission.csv - a sample submission file in the correct format
# 
# ---
# **Notebooks Ideas and Credits**
# 
# I took ideas or inspiration from the following notebooks, if you enjoy my work, please take a look to the notebooks that inspire my work.
# 
# TPSMAY22 Gradient-Boosting Quickstart: https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart/notebook

# ---

# # 1. Loading the Requiered Libraries

# In[1]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[2]:


get_ipython().run_cell_magic('time', '', 'import datetime\n')


# ---
# 

# # 2. Setting the Notebook

# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 50\nNCOLS = 15\n# Main data location path...\nBASE_PATH = '...'\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.5f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# ---

# # 3. Loading the Information (CSV) Into A Dataframe

# In[6]:


get_ipython().run_cell_magic('time', '', "# Load the CSV information into a Pandas DataFrame...\ntrn_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')\ntst_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')\n\nsub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')\n")


# ---

# # 4. Exploring the Information Available

# ## 4.1. Analysing the Trian Dataset

# In[7]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape\n')


# In[8]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\ntrn_data.info()\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '# Calculates the total number of missing values...\ntrn_data.isnull().sum().sum()\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Display the number of missing values by variable...\ntrn_data.isnull().sum()\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\ntrn_data.nunique()\n')


# In[14]:


# Display the number of unique values for each variable, sorted by quantity...
trn_data.nunique().sort_values(ascending = True)


# In[15]:


get_ipython().run_cell_magic('time', '', "# Check some of the categorical variables\ncateg_cols = ['f_29','f_30','f_13', 'f_18','f_17','f_14','f_11','f_10','f_09','f_15','f_07','f_12','f_16','f_08','f_27']\ntrn_data[categ_cols].sample(5)\n")


# In[16]:


get_ipython().run_cell_magic('time', '', '# Generate a quick correlation matrix to understand the dataset better\ncorrelation = trn_data.corr()\n')


# In[17]:


get_ipython().run_cell_magic('time', '', '# Diplay the correlation matrix\ncorrelation\n')


# In[18]:


get_ipython().run_cell_magic('time', '', "# Check the most correlated variables to the target\ncorrelation['target'].sort_values(ascending = False)[:5]\n")


# In[19]:


get_ipython().run_cell_magic('time', '', "# Check the least correlated variables to the target\ncorrelation['target'].sort_values(ascending = True)[:5]\n")


# ---

# ## 4.2. Analysing the Trian Labels Dataset

# In[20]:


get_ipython().run_cell_magic('time', '', "# Check how well balanced is the dataset\ntrn_data['target'].value_counts()\n")


# In[21]:


get_ipython().run_cell_magic('time', '', "# Check some statistics on the target variable\ntrn_data['target'].describe()\n")


# ---

# # 5. Feature Engineering

# In[22]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()\n')


# In[23]:


trn_data.shape


# In[24]:


get_ipython().run_cell_magic('time', '', "continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']\n\ndef remove_outliers(df, continuous_feat):\n    for col in df[continuous_feat].columns:\n        #df = df[(df[col] > df[col].quantile(0.01)) & (df[col] < df[col].quantile(0.99))]\n        df[col + '_outlier'] = np.where((df[col] > df[col].quantile(0.01)) & (df[col] < df[col].quantile(0.99)),0,1)\n    \n    \n    outliers = [col for col in df.columns if '_outlier' in col]\n    df['total_outliers']  = df[outliers].sum(axis=1)\n    df = df.drop(columns = outliers)\n    return df\n\ntrn_data = remove_outliers(trn_data, continuous_feat)\ntst_data = remove_outliers(tst_data, continuous_feat)\n")


# In[25]:


trn_data.shape


# In[26]:


trn_data.sample(10)


# In[27]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()\n')


# ## 5.1 Text Base Features

# In[28]:


get_ipython().run_cell_magic('time', '', '# The idea is to create a simple funtion to count the amount of letters on feature 27.\n# feature 27 seems quite important \n\ndef count_sequence(df, field):\n    \'\'\'\n    For each letter of the provided suquence it return new feature with the number of occurences.\n    \'\'\'\n    alphabet = [\'A\',\'B\',\'C\',\'D\',\'E\',\'F\',\'G\',\'H\',\'I\',\'J\',\'K\',\'L\',\'M\',\'N\',\'O\',\'P\',\'Q\',\'R\',\'S\',\'T\',\'U\',\'V\',\'W\',\'X\',\'Y\',\'Z\']    \n    \n    for letter in alphabet:\n        df[letter + \'_count\'] = df[field].str.count(letter)\n    \n    df["unique_characters"] = df[\'f_27\'].apply(lambda s: len(set(s)))\n    return df\n')


# In[29]:


get_ipython().run_cell_magic('time', '', "# Utilizes the new created funtions to generate more features.\n#trn_data = count_sequence(trn_data, 'f_27')\n#tst_data = count_sequence(tst_data, 'f_27')\n")


# In[30]:


get_ipython().run_cell_magic('time', '', 'def count_chars(df, field):\n    \'\'\'\n    Describe something...\n    \'\'\'\n    \n    for i in range(10):\n        df[f\'ch_{i}\'] = df[field].str.get(i).apply(ord) - ord(\'A\')\n        \n    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))\n    return df\n')


# In[31]:


get_ipython().run_cell_magic('time', '', "# Utilizes the new created funtions to generate more features.\ntrn_data = count_chars(trn_data, 'f_27')\ntst_data = count_chars(tst_data, 'f_27')\n")


# ## 5.2 Statistical Features

# In[32]:


get_ipython().run_cell_magic('time', '', "continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28']\n\ndef stat_features(df, cols = continuous_feat):\n    '''\n    Calculate aggregated features across the selected continuous columns\n    \n    '''\n    # Base statistical features.\n    df['f_sum']  = df[continuous_feat].sum(axis=1)\n    df['f_min']  = df[continuous_feat].min(axis=1)\n    df['f_max']  = df[continuous_feat].max(axis=1)\n    df['f_std']  = df[continuous_feat].std(axis=1)    \n    df['f_mad']  = df[continuous_feat].mad(axis=1)\n    df['f_mean'] = df[continuous_feat].mean(axis=1)\n    df['f_kurt'] = df[continuous_feat].kurt(axis=1)\n\n    # Extra statistical features.\n    df['f_prod'] = df[continuous_feat].prod(axis=1)\n    df['f_range'] = df[continuous_feat].max(axis=1) - df[continuous_feat].min(axis=1)\n    df['f_count_pos']  = df[df[continuous_feat].gt(0)].count(axis=1)\n    df['f_count_neg']  = df[df[continuous_feat].lt(0)].count(axis=1)\n\n    return df\n")


# In[33]:


get_ipython().run_cell_magic('time', '', 'trn_data = stat_features(trn_data, continuous_feat)\ntst_data = stat_features(tst_data, continuous_feat)\n')


# In[34]:


get_ipython().run_cell_magic('time', '', 'trn_data.head()\n')


# 

# In[35]:


get_ipython().run_cell_magic('time', '', "def calculate_feat_int(df):\n    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)\n    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)\n    i_00_01_26 = df.f_00 + df.f_01 + df.f_26\n    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)\n    return df\n\ntrn_data = calculate_feat_int(trn_data)\ntst_data = calculate_feat_int(tst_data)\n")


# ---

# # 7. Pre-Processing Labels

# ## 7.1 Label Encoding

# In[36]:


get_ipython().run_cell_magic('time', '', "# Define a label encoding function\nfrom sklearn.preprocessing import LabelEncoder\nencoder = LabelEncoder()\ndef encode_features(df, cols = ['f_27']):\n    '''\n    Apply one-hot encode to the selected columns, return a df\n    '''\n    for col in cols:\n        df[col + '_enc'] = encoder.fit_transform(df[col])\n    return df\n\ntrn_data = encode_features(trn_data)\ntst_data = encode_features(tst_data)\n")


# In[37]:


get_ipython().run_cell_magic('time', '', '# Check the results of the transformation\ntrn_data.sample(10)\n')


# ## 7.2 - One-Hot Encode

# In[38]:


# We will process to One-Hot encode all this variables...
# f_29           2
# f_30           3
# f_13          13
# f_18          14
# f_17          14
# f_14          14
# f_11          14
# f_10          15
# f_09          15
# f_15          15
# f_07          16
# f_12          16
# f_16          16
# f_08          16


# In[39]:


get_ipython().run_cell_magic('time', '', "def one_hot_encoder(df_trn, df_tst, var_list):\n    '''\n    Apply one-hot encode to the selected columns, and merged train and test \n    dataframes to avoid issues with missing categories.\n    \n    '''\n    df_trn['is_train'] = 1\n    df_tst['is_train'] = 0\n\n    combined = df_trn.append(df_tst)\n    combined = pd.get_dummies(combined, columns = var_list)\n    return combined[combined['is_train'] == 1], combined[combined['is_train'] == 0]\n")


# In[40]:


get_ipython().run_cell_magic('time', '', "#trn_data, tst_data = one_hot_encoder(trn_data,tst_data, [\n                                                         #'f_29',\n                                                         #'f_30',\n                                                         #'f_13',\n                                                         #'f_18',\n                                                         #'f_17',\n                                                         #'f_14',\n                                                         #'f_11',\n                                                         #'f_10',\n                                                         #'f_09',\n                                                         #'f_15',\n                                                         #'f_07',\n                                                         #'f_12',\n                                                         #'f_16',\n                                                         #'f_08'\n                                                         #])\n")


# ---

# # 8. Feature Selection for Baseline Model

# In[41]:


get_ipython().run_cell_magic('time', '', "# Define what will be used in the training stage\nignore = ['id', \n          'f_27', \n          'f_27_enc', \n          'is_train', \n          'target'] # f_27 has been label encoded...\n\nfeatures = [feat for feat in trn_data.columns if feat not in ignore]\ntarget_feature = 'target'\n")


# ---

# # 9. Creating a Simple Train / Test Split Strategy

# In[42]:


get_ipython().run_cell_magic('time', '', '# Creates a simple train split breakdown for baseline model\nfrom sklearn.model_selection import train_test_split\ntest_size_pct = 0.20\nX_train, X_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target_feature], test_size = test_size_pct, random_state = 42)\n')


# ---

# # 10. Building a Baseline NN Model, Simple Split

# In[43]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping\nfrom tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout\n\nfrom sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nimport random\n')


# In[44]:


get_ipython().run_cell_magic('time', '', 'np.random.seed(1)\nrandom.seed(1)\ntf.random.set_seed(1)\n')


# In[45]:


get_ipython().run_cell_magic('time', '', "\n\n\ndef nn_model():\n    \n    '''\n    '''\n    \n    L2 = 65e-6\n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x = Dense(512, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(L2), \n              activation = activation_func)(inputs)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(512, \n          #use_bias  = True, \n          kernel_regularizer = tf.keras.regularizers.l2(L2), \n          activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(64, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(L2), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(64, \n          #use_bias  = True, \n          kernel_regularizer = tf.keras.regularizers.l2(L2), \n          activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(16, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(L2), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(1 , \n              #use_bias  = True, \n              #kernel_regularizer = tf.keras.regularizers.l2(L2),\n              activation = 'sigmoid')(x)\n    \n    model = Model(inputs, x)\n    \n    return model\n")


# In[46]:


get_ipython().run_cell_magic('time', '', 'architecture = nn_model()\narchitecture.summary()\n')


# In[47]:


get_ipython().run_cell_magic('time', '', "# Defining model parameters...\nBATCH_SIZE         = 2048\nEPOCHS             = 200 \nEPOCHS_COSINEDECAY = 150 \nDIAGRAMS           = True\nUSE_PLATEAU        = False\nINFERENCE          = False\nVERBOSE            = 0 \nTARGET             = 'target'\n")


# In[48]:


get_ipython().run_cell_magic('time', '', '# Defining model training function...\ndef fit_model(X_train, y_train, X_val, y_val, run = 0):\n   \'\'\'\n   \'\'\'\n   lr_start = 0.01\n   start_time = datetime.datetime.now()\n   \n   scaler = StandardScaler()\n   #scaler = RobustScaler()\n   #scaler = MinMaxScaler()\n\n   X_train = scaler.fit_transform(X_train)\n\n   epochs = EPOCHS    \n   lr = ReduceLROnPlateau(monitor = \'val_loss\', factor = 0.7, patience = 4, verbose = VERBOSE)\n   es = EarlyStopping(monitor = \'val_loss\',patience = 12, verbose = 1, mode = \'min\', restore_best_weights = True)\n   tm = tf.keras.callbacks.TerminateOnNaN()\n   callbacks = [lr, es, tm]\n   \n   # Cosine Learning Rate Decay\n   if USE_PLATEAU == False:\n       epochs = EPOCHS_COSINEDECAY\n       lr_end = 0.0002\n\n       def cosine_decay(epoch):\n           if epochs > 1:\n               w = (1 + math.cos(epoch / (epochs - 1) * math.pi)) / 2\n           else:\n               w = 1\n           return w * lr_start + (1 - w) * lr_end\n       \n       lr = LearningRateScheduler(cosine_decay, verbose = 0)\n       callbacks = [lr, tm]\n       \n   model = nn_model()\n   optimizer_func = tf.keras.optimizers.Adam(learning_rate = lr_start)\n   loss_func = tf.keras.losses.BinaryCrossentropy()\n   model.compile(optimizer = optimizer_func, loss = loss_func)\n   \n   X_val = scaler.transform(X_val)\n   validation_data = (X_val, y_val)\n   \n   history = model.fit(X_train, \n                       y_train, \n                       validation_data = validation_data, \n                       epochs          = epochs,\n                       verbose         = VERBOSE,\n                       batch_size      = BATCH_SIZE,\n                       shuffle         = True,\n                       callbacks       = callbacks\n                      )\n   \n   history_list.append(history.history)\n   print(f\'Training loss:{history_list[-1]["loss"][-1]:.5f}\')\n   callbacks, es, lr, tm, history = None, None, None, None, None\n   \n   \n   y_val_pred = model.predict(X_val, batch_size = BATCH_SIZE, verbose = VERBOSE)\n   score = roc_auc_score(y_val, y_val_pred)\n   print(f\'Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}\'\n         f\'| AUC: {score:.5f}\')\n   \n   score_list.append(score)\n   \n   tst_data_scaled = scaler.transform(tst_data[features])\n   tst_pred = model.predict(tst_data_scaled)\n   predictions.append(tst_pred)\n   \n   return model\n')


# In[49]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score, roc_curve\nimport math\n\n# Create empty lists to store NN information...\nhistory_list = []\nscore_list   = []\npredictions  = []\n\n# Define kfolds for training purposes...\nkf = KFold(n_splits = 5)\n\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data)):\n    X_train, X_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_val = trn_data.iloc[trn_idx][TARGET], trn_data.iloc[val_idx][TARGET]\n    \n    fit_model(X_train, y_train, X_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(score_list):.5f}')\n")


# In[50]:


# OOF AUC: 0.99658... 10 Folds, Batch Normalization, Using One-Hot Features, Epochs = 150, [64,64,64,16,1]...
# OOF AUC: 0.99653... 05 Folds, No Batch Normalization, Using Partial One-Hot Features, Epochs = 150, [64,64,64,16,1]...
# OOF AUC: 0.99757... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 150, [64,64,64,16,1]...
# OOF AUC: 0.99766... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 200, [64,64,64,16,1]...
# OOF AUC: 0.99771... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,16,1]...
# OOF AUC: 0.99759... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [96,64,64,16,1]...
# OOF AUC: 0.99772... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,32,16,1]...
# OOF AUC: 0.99772... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,32,16,1]...
# OOF AUC: 0.99769... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,16,1], Stat Features = Yes
# OOF AUC: 0.99769... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [256,64,64,16,1], Stat Features = Yes
# OOF AUC: 0.99774... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,64,16,1], Stat Features = Yes, StandarScaler ****
# OOF AUC: 0.99769... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,64,16,1], Stat Features = Yes, RobustScaler
# OOF AUC: ???????... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,64,16,1], Stat Features = Yes, MinMaxScaler Terminated...

# OOF AUC: 0.99771... 05 Folds, No Batch Normalization, No One-Hot Features, Epochs = 300, [64,64,64,64,16,1], Stat Features = Yes + Extra, StandarScaler ****

# OOF AUC: 0.99767...
# OOF AUC: 0.99784...
# OOF AUC: 0.99763 >> 0.99816...

# OOF AUC: 0.99794 V0, Plain features... (40e-6)
# OOF AUC: 0.99793 V1, Added MAD...

# OOF AUC: 0.99797 ...  45e-6
# OOF AUC: 0.99798 ...  50e-6
# OOF AUC: 0.99799 ...  60e-6

# OOF AUC: 0.99803
# OOF AUC: 0.99798 ...


# ---

# # 11. Undertanding Model Behavior, Feature Importance

# In[51]:


# Work in Progress...


# ---

# # 12. Baseline Model Submission File Generation

# In[52]:


get_ipython().run_cell_magic('time', '', '# Review the format of the submission file\nsub.head()\n')


# In[53]:


get_ipython().run_cell_magic('time', '', "# Populated the prediction on the submission dataset and creates an output file\nsub['target'] = np.array(predictions).mean(axis = 0)\nsub.to_csv('my_submission_052222_v5.csv', index = False)\n")


# In[54]:


get_ipython().run_cell_magic('time', '', '%%script false --no-raise-error\n# Create submission\nprint(f"{len(features)} features")\n\npred_list = []\nfor seed in range(10):\n    model = fit_model(X_tr, y_tr, run = seed)\n    model.fit(X_tr.values, y_tr)\n    pred_list.append(scipy.stats.rankdata(model.predict(tst_data[features].values, batch_size = BATCH_SIZE)))\n    print(f"{seed:2}", pred_list[-1])\nprint()\n\nsubmission = tst_data[[\'id\']].copy()\nsubmission[TARGET] = np.array(pred_list).mean(axis = 0)\n\nsubmission.to_csv(\'submission_nn_05222022_v0.csv\', index = False)\n')


# In[55]:


get_ipython().run_cell_magic('time', '', '# Review the submission file as a final step to upload to Kaggle.\nsub.head()\n')


# ---
