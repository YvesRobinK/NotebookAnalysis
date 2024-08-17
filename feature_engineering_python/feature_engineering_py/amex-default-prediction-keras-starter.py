#!/usr/bin/env python
# coding: utf-8

# # American Express - Default Prediction 
# ## Predict If A Customer Will Default in the Future ...
# The objective of this competition is to predict the probability that a customer does not pay back their credit card balance amount in the future based on their monthly customer profile. The target binary variable is calculated by observing 18 months performance window after the latest credit card statement, and if the customer does not pay due amount in 120 days after their latest statement date it is considered a default event.
# 
# <img style="float: center;" src="https://img.freepik.com/free-vector/brain-with-digital-circuit-programmer-with-laptop-machine-learning-artificial-intelligence-digital-brain-artificial-thinking-process-concept-vector-isolated-illustration_335657-2246.jpg?w=2000" width = '550'>
# <a href='https://www.freepik.com/vectors/machine-learning'>Machine learning vector created by vectorjuice - www.freepik.com</a>
# 
# #### Data Description
# The dataset contains aggregated profile features for each customer at each statement date. Features are anonymized and normalized, and fall into the following general categories:
# 
# * D_* = Delinquency variables
# * S_* = Spend variables
# * P_* = Payment variables
# * B_* = Balance variables
# * R_* = Risk variables
# 
# With the following features being categorical:
# 
# **['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']**
# 
# Your task is to predict, for each customer_ID, the probability of a future payment default (target = 1).
# 
# Note that the negative class has been subsampled for this dataset at 5%, and thus receives a 20x weighting in the scoring metric.
# 
# **Files**
# * train_data.csv - training data with multiple statement dates per customer_ID
# * train_labels.csv - target label for each customer_ID
# * test_data.csv - corresponding test data; your objective is to predict the target label for each customer_ID
# * sample_submission.csv - a sample submission file in the correct format
# 
# ---
# 
# ## My Strategy, or How I Will Aproach this Competition...
# We have data from many Customers and there is many points of information by for each of the customers, the target labels are only one per customer id so aggregation will be requiered, from here there is quie a lot of possibilities, this is what I will folow in this Notebook...
# 
# #### Loading the Datasets
# The datasets is massive so I will rely on other Kaggles optimized datasets stored in a feather format to make my life easier in this competition.
# 
# #### Quick EDA
# The typical analysis that I always like to complete to undertstand the dataset better...
# * Information of the datasets, size and others.
# * Simple visualization of the first few records.
# * Data statistical analalysis using describe.
# * Visualization of the number of NaNs.
# * Understanding the amount of unique records.
# 
# #### Exploring the Target Variable
# Nothing in particular dataset seems to be quite inbalanced so I will get back to this part later...
# 
# #### Structuring the Datasets
# Here is where everything happens, because we have time-base data o multiple points per customer we are trying to aggregate the information in certain way that's practical:
# * Statistical aggregation for numeric features
# * Only keep the last know record for analysis
# * Statictical aggregation for categorical features
# 
# #### Feature Engineering
# At this point the only thing that I can consider some type of feature will be the aggregation of the datasets, as I mentioned in the previous point
# * Statistical aggregation
# * Only keep the last know record for analysis
# 
# #### Label Encoding
# Because there is quite a lot of categorical variables and this is a NN model I will use the following encoding technique:
# * OneHot encoder, only train in the train dataset and applyed on test
# 
# #### Fill NaNs**
# At this point just to get started, I will fill everything with ceros, probably not a good idea.
# * Fill NaNs with 0
# 
# #### Model Development and Training
# I'm going to go first with an NN in the last few competitions the NN models have been working quite well also we have so much data.
# * Simple NN tested, layer after later.
# * I also tested a more complex NN, that I learned from Ambross with Skip conections.
# 
# #### Predictions and Submission
# No much details here, just the simple average of all the predictions across multiple folds.
# * Average predictions across 5 folds
# 
# ---
# 
# ## Updates
# #### 05/28/2022
# * Build the initial model using Neuronal Nets and simple agg strategy (Last data point).
# * Evaluated the model and uploaded for Ranking.
# 
# #### 05/29/2022
# * Improve model architecture.
# * Really dive deep into Feature Engineering (Not much here, memory is a big challenge)
# 
# #### 05/30/2022
# * ...
# 
# ---
# 
# ## Resources, Inspiration
# I have taken Ideas or learned quite a lot from the Notebooks below, please check also if you like my work.
# 
# * https://www.kaggle.com/code/ambrosm/amex-keras-quickstart-1-training/notebook
# * ...
# * ...
# * ...

# ---

# # 1.0 Loading Model Libraries...

# In[1]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[2]:


get_ipython().run_cell_magic('time', '', 'import datetime # ...\n')


# ---

# # 2.0 Setting the Notebook Parameters and Default Configuration...

# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 50\nNCOLS = 15\n# Main data location path...\nBASE_PATH = '...'\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.5f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# ---

# # 3.0 Loading the Dataset Information (Using Feather)...

# In[6]:


get_ipython().run_cell_magic('time', '', "# Load the CSV information into a Pandas DataFrame...\ntrn_data = pd.read_feather('../input/parquet-files-amexdefault-prediction/train_data.ftr')\ntrn_lbls = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv').set_index('customer_ID')\n\ntst_data = pd.read_feather('../input/parquet-files-amexdefault-prediction/test_data.ftr')\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "sub = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')\n")


# ---

# # 4.0 Exploring the Dataset, Quick EDA...

# In[8]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\ntrn_data.info()\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()\n')


# In[11]:


get_ipython().run_cell_magic('time', '', "# Display the Min Date...\ntrn_data['S_2'].min()\n")


# In[12]:


get_ipython().run_cell_magic('time', '', "# Display the Max Date...\ntrn_data['S_2'].max()\n")


# In[13]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', '# Calculates the total number of missing values...\ntrn_data.isnull().sum().sum()\n')


# In[15]:


get_ipython().run_cell_magic('time', '', '# Display the number of missing values by variable...\ntrn_data.isnull().sum()\n')


# In[16]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\ntrn_data.nunique()\n')


# In[17]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable, sorted by quantity...\ntrn_data.nunique().sort_values(ascending = True)\n')


# ---

# # 5.0 Understanding the Target Variable...

# In[18]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_lbls.shape\n')


# In[19]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\ntrn_lbls.info()\n')


# In[20]:


get_ipython().run_cell_magic('time', '', "# Check how well balanced is the dataset\ntrn_lbls['target'].value_counts()\n")


# In[21]:


get_ipython().run_cell_magic('time', '', "# Check some statistics on the target variable\ntrn_lbls['target'].describe()\n")


# ---

# # 6.0 Structuring Data for the Model (Aggreations and More)

# ## 6.1 Training Dataset...

# In[22]:


get_ipython().run_cell_magic('time', '', '# We have 458913 customers. and we have 458913 train labels...\n')


# In[23]:


get_ipython().run_cell_magic('time', '', "# Calculates the amount of information by costumer or records available...\ntrn_num_statements = trn_data.groupby('customer_ID').size().sort_index()\n")


# In[24]:


get_ipython().run_cell_magic('time', '', '# Review some of the information created...\ntrn_num_statements\n')


# In[25]:


get_ipython().run_cell_magic('time', '', "# Create a new dataset based on aggregated information\ntrn_agg_data = (trn_data\n                .groupby('customer_ID')\n                .tail(1)\n                .set_index('customer_ID', drop=True)\n                .sort_index()\n                .drop(['S_2'], axis='columns'))\n\n# Merge the labels from the labels dataframe\ntrn_agg_data['target'] = trn_lbls.target\ntrn_agg_data['num_statements'] = trn_num_statements\n\ntrn_agg_data.reset_index(inplace = True, drop = True) # forget the customer_IDs\n")


# In[26]:


get_ipython().run_cell_magic('time', '', 'trn_agg_data.head()\n')


# ---

# ## 6.2 Test Dataset...

# In[27]:


get_ipython().run_cell_magic('time', '', "# Calculates the amount of information by costumer or records available...\ntst_num_statements = tst_data.groupby('customer_ID').size().sort_index()\n")


# In[28]:


get_ipython().run_cell_magic('time', '', "# Create a new dataset based on aggregated information\ntst_agg_data = (tst_data\n                .groupby('customer_ID')\n                .tail(1)\n                .set_index('customer_ID', drop=True)\n                .sort_index()\n                .drop(['S_2'], axis='columns'))\n\n# Merge the labels from the labels dataframe\ntst_agg_data['num_statements'] = tst_num_statements\n\ntst_agg_data.reset_index(inplace = True, drop = True) # forget the customer_IDs\n")


# In[29]:


get_ipython().run_cell_magic('time', '', 'tst_agg_data.head()\n')


# ---

# # 7.0 Label / One-Hot Encoding the Categorical Variables...

# ## 7.1 One Hot Encoding Configuration...

# In[30]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, OrdinalEncoder\n')


# In[31]:


get_ipython().run_cell_magic('time', '', "# One-hot Encoding Configuration\ncat_features = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']\n\n#trn_agg_data[cat_features] = trn_agg_data[cat_features].astype(object)\ntrn_not_cat_features = [f for f in trn_agg_data.columns if f not in cat_features]\ntst_not_cat_features = [f for f in tst_agg_data.columns if f not in cat_features]\n")


# In[32]:


get_ipython().run_cell_magic('time', '', 'trn_agg_data[cat_features].head()\n')


# In[33]:


get_ipython().run_cell_magic('time', '', "#encoder = OneHotEncoder(drop = 'first', sparse = False, dtype = np.float32, handle_unknown = 'ignore')\nencoder = OrdinalEncoder()\ntrn_encoded_features = encoder.fit_transform(trn_agg_data[cat_features])\n#feat_names = list(encoder.get_feature_names())\n")


# ## 7.2 Train Dataset One Hot Encoding...

# In[34]:


get_ipython().run_cell_magic('time', '', '# One-hot Encoding\ntrn_encoded_features = pd.DataFrame(trn_encoded_features)\n#trn_encoded_features.columns = feat_names\n')


# In[35]:


get_ipython().run_cell_magic('time', '', 'trn_agg_data = pd.concat([trn_agg_data[trn_not_cat_features], trn_encoded_features], axis = 1)\ntrn_agg_data.head(5)\n')


# ---

# ## 7.3 Test Dataset One-Hot Encoding...

# In[36]:


get_ipython().run_cell_magic('time', '', 'tst_agg_data[cat_features].head()\n')


# In[37]:


get_ipython().run_cell_magic('time', '', '# One-hot Encoding\ntst_encoded_features = encoder.transform(tst_agg_data[cat_features])\ntst_encoded_features = pd.DataFrame(tst_encoded_features)\n#tst_encoded_features.columns = feat_names\n')


# In[38]:


get_ipython().run_cell_magic('time', '', 'tst_agg_data = pd.concat([tst_agg_data[tst_not_cat_features], tst_encoded_features], axis = 1)\ntst_agg_data.head()\n')


# ---

# # 8.0 Pre-Processing the Data, Fill NaNs for model functionality...

# In[39]:


get_ipython().run_cell_magic('time', '', '# Impute missing values\ntrn_agg_data.fillna(value = 0, inplace = True)\ntst_agg_data.fillna(value = 0, inplace = True)\n')


# ---

# # 9.0 Feature Selection for Baseline Model...

# In[40]:


get_ipython().run_cell_magic('time', '', "features = [f for f in trn_agg_data.columns if f != 'target' and f != 'customer_ID']\n")


# ---

# # 10.0 NN Development

# In[41]:


get_ipython().run_cell_magic('time', '', '# Release some memory by deleting the original DataFrames...\nimport gc\ndel trn_data, tst_data\ngc.collect()\n')


# ## 10.1 Loading Specific Model Libraries...

# In[42]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping\nfrom tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout, Concatenate\nfrom tensorflow.keras.utils import plot_model\nfrom sklearn.metrics import log_loss\n\nfrom sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nimport random\n')


# ---

# ## 10.2 Amex Metric, Function...

# In[43]:


get_ipython().run_cell_magic('time', '', '# From https://www.kaggle.com/code/inversion/amex-competition-metric-python\n\ndef amex_metric(y_true, y_pred, return_components=False) -> float:\n    """Amex metric for ndarrays"""\n    \n    def top_four_percent_captured(df) -> float:\n        """Corresponds to the recall for a threshold of 4 %"""\n        \n        df[\'weight\'] = df[\'target\'].apply(lambda x: 20 if x==0 else 1)\n        four_pct_cutoff = int(0.04 * df[\'weight\'].sum())\n        df[\'weight_cumsum\'] = df[\'weight\'].cumsum()\n        df_cutoff = df.loc[df[\'weight_cumsum\'] <= four_pct_cutoff]\n        return (df_cutoff[\'target\'] == 1).sum() / (df[\'target\'] == 1).sum()\n    \n    \n    def weighted_gini(df) -> float:\n        df[\'weight\'] = df[\'target\'].apply(lambda x: 20 if x==0 else 1)\n        df[\'random\'] = (df[\'weight\'] / df[\'weight\'].sum()).cumsum()\n        total_pos = (df[\'target\'] * df[\'weight\']).sum()\n        df[\'cum_pos_found\'] = (df[\'target\'] * df[\'weight\']).cumsum()\n        df[\'lorentz\'] = df[\'cum_pos_found\'] / total_pos\n        df[\'gini\'] = (df[\'lorentz\'] - df[\'random\']) * df[\'weight\']\n        return df[\'gini\'].sum()\n\n    \n    def normalized_weighted_gini(df) -> float:\n        """Corresponds to 2 * AUC - 1"""\n        \n        df2 = pd.DataFrame({\'target\': df.target, \'prediction\': df.target})\n        df2.sort_values(\'prediction\', ascending=False, inplace=True)\n        return weighted_gini(df) / weighted_gini(df2)\n\n    \n    df = pd.DataFrame({\'target\': y_true.ravel(), \'prediction\': y_pred.ravel()})\n    df.sort_values(\'prediction\', ascending=False, inplace=True)\n    g = normalized_weighted_gini(df)\n    d = top_four_percent_captured(df)\n\n    if return_components: return g, d, 0.5 * (g + d)\n    return 0.5 * (g + d)\n')


# ---

# ## 10.3 Defining the NN Model Architecture...

# ## 10.3.1 Architecture 01, Simple NN

# In[44]:


get_ipython().run_cell_magic('time', '', "def nn_model():\n    '''\n    '''\n    regularization = 4e-4\n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x = Dense(256, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(regularization), \n              activation = activation_func)(inputs)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(64, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(regularization), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(64, \n          #use_bias  = True, \n          kernel_regularizer = tf.keras.regularizers.l2(regularization), \n          activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(32, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(regularization), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(1, \n              #use_bias  = True, \n              #kernel_regularizer = tf.keras.regularizers.l2(regularization),\n              activation = 'sigmoid')(x)\n    \n    model = Model(inputs, x)\n    \n    return model\n")


# ---

# ## 10.3.2 Architecture 02, Concatenated NN

# In[45]:


get_ipython().run_cell_magic('time', '', "def nn_model():\n    regularization = 4e-4\n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n\n    x0 = Dense(256,\n               kernel_regularizer = tf.keras.regularizers.l2(regularization), \n               activation = activation_func)(inputs)\n    x1 = Dense(128,\n               kernel_regularizer = tf.keras.regularizers.l2(regularization),\n               activation = activation_func)(x0)\n    x1 = Dense(64,\n               kernel_regularizer = tf.keras.regularizers.l2(regularization),\n               activation = activation_func)(x1)\n    x1 = Dense(32,\n           kernel_regularizer = tf.keras.regularizers.l2(regularization),\n           activation = activation_func)(x1)\n    \n    x1 = Concatenate()([x1, x0])\n    x1 = Dropout(0.1)(x1)\n    \n    x1 = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(regularization),activation=activation_func,)(x1)\n    \n    x1 = Dense(1, \n              #kernel_regularizer=tf.keras.regularizers.l2(regularization),\n              activation='sigmoid')(x1)\n    \n    model = Model(inputs, x1)\n    \n    return model\n    \n")


# ---

# ## 10.4 Visualizing the Model Structure...

# In[46]:


get_ipython().run_cell_magic('time', '', 'architecture = nn_model()\narchitecture.summary()\n')


# In[47]:


get_ipython().run_cell_magic('time', '', 'plot_model(nn_model(), show_layer_names = False, show_shapes = True, dpi = 60)\n')


# ---

# ## 10.5 Defining Model Training Parameters...

# In[48]:


get_ipython().run_cell_magic('time', '', "# Defining model parameters...\nBATCH_SIZE         = 2048\nEPOCHS             = 192 \nEPOCHS_COSINEDECAY = 192 \nDIAGRAMS           = True\nUSE_PLATEAU        = False\nINFERENCE          = False\nVERBOSE            = 0 \nTARGET             = 'target'\n")


# ---

# ## 10.6 Defining the Model Training Configuration...

# In[49]:


get_ipython().run_cell_magic('time', '', '# Defining model training function...\ndef fit_model(X_train, y_train, X_val, y_val, run = 0):\n   \'\'\'\n   \'\'\'\n   lr_start = 0.01\n   start_time = datetime.datetime.now()\n   \n   scaler = StandardScaler()\n   X_train = scaler.fit_transform(X_train)\n\n   epochs = EPOCHS    \n   lr = ReduceLROnPlateau(monitor = \'val_loss\', factor = 0.7, patience = 4, verbose = VERBOSE)\n   es = EarlyStopping(monitor = \'val_loss\',patience = 12, verbose = 1, mode = \'min\', restore_best_weights = True)\n   tm = tf.keras.callbacks.TerminateOnNaN()\n   callbacks = [lr, es, tm]\n   \n   # Cosine Learning Rate Decay\n   if USE_PLATEAU == False:\n       epochs = EPOCHS_COSINEDECAY\n       lr_end = 0.0002\n\n       def cosine_decay(epoch):\n           if epochs > 1:\n               w = (1 + math.cos(epoch / (epochs - 1) * math.pi)) / 2\n           else:\n               w = 1\n           return w * lr_start + (1 - w) * lr_end\n       \n       lr = LearningRateScheduler(cosine_decay, verbose = 0)\n       callbacks = [lr, tm]\n   \n   # Model Initialization...\n   model = nn_model()\n   optimizer_func = tf.keras.optimizers.Adam(learning_rate = lr_start)\n   loss_func = tf.keras.losses.BinaryCrossentropy()\n   model.compile(optimizer = optimizer_func, loss = loss_func)\n   \n   \n   X_val = scaler.transform(X_val)\n   validation_data = (X_val, y_val)\n   \n   history = model.fit(X_train, \n                       y_train, \n                       validation_data = validation_data, \n                       epochs          = epochs,\n                       verbose         = VERBOSE,\n                       batch_size      = BATCH_SIZE,\n                       shuffle         = True,\n                       callbacks       = callbacks\n                      )\n   \n   history_list.append(history.history)\n   \n   print(f\'Training Loss: {history_list[-1]["loss"][-1]:.5f}, Validation Loss: {history_list[-1]["val_loss"][-1]:.5f}\')\n   callbacks, es, lr, tm, history = None, None, None, None, None\n   \n   \n   y_val_pred = model.predict(X_val, batch_size = BATCH_SIZE, verbose = VERBOSE).ravel()\n   amex_score = amex_metric(y_val.values, y_val_pred, return_components = False)\n   \n   print(f\'Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}\'\n         f\'| Amex Score: {amex_score:.5f}\')\n   \n   print(\'\')\n   \n   score_list.append(amex_score)\n   \n   tst_data_scaled = scaler.transform(tst_agg_data[features])\n   tst_pred = model.predict(tst_data_scaled)\n   predictions.append(tst_pred)\n   \n   return model\n')


# ---

# ## 10.7 Creating a Model Training Loop and Cross Validating in 5 Folds... 

# In[50]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score, roc_curve\nimport math\n\n# Create empty lists to store NN information...\nhistory_list = []\nscore_list   = []\npredictions  = []\n\n# Define kfolds for training purposes...\nkf = KFold(n_splits = 5)\n\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_agg_data)):\n    X_train, X_val = trn_agg_data.iloc[trn_idx][features], trn_agg_data.iloc[val_idx][features]\n    y_train, y_val = trn_agg_data.iloc[trn_idx][TARGET], trn_agg_data.iloc[val_idx][TARGET]\n    \n    fit_model(X_train, y_train, X_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(score_list):.5f}')\n")


# ---

# # 11.0 Model Prediction and Submissions

# In[51]:


get_ipython().run_cell_magic('time', '', 'sub.head()\n')


# In[52]:


get_ipython().run_cell_magic('time', '', "sub['prediction'] = np.array(predictions).mean(axis = 0)\n")


# In[53]:


get_ipython().run_cell_magic('time', '', "sub.to_csv('my_submission.csv', index = False)\n")


# In[54]:


get_ipython().run_cell_magic('time', '', 'sub.head()\n')


# ---
