#!/usr/bin/env python
# coding: utf-8

# # 🌵 A Simple Keras Model...
# I will build a simple neuronal network with minimun amount of data to use the model for feature engineering development and improvement in performance...
# 
# ### Notebook Ideas...
# * Start simple and add complexity...
# * Keep well documented code...
# * Complete a simple end to end model...
# * Submit predictions and continue building improvements...

# ---

# ## 1.0 Loading Libraries...

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


# In[2]:


# Loading other important libraries
import matplotlib.pyplot as plt # Import visualization library
import seaborn as sns # Import visualization library


# ---

# ## 2.0 Notebook Configurations...

# In[3]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration.\n\n# Amount of data we want to load into the model from Pandas.\nDATA_ROWS = None\n\n# Memory and replicability\nDATA_PCT = 0.35 # Load only 50% of the dataset to avoid memory issues...\nSEED = 777 # This will be the seed utilized across the notebook\n\n# Dataframe, the amount of rows and cols to visualize.\nNROWS = 20\nNCOLS = 15\n\n# Main data location base path.\nBASE_PATH = '...'\n\n# Model development parameters\nTEST_PCT = 0.20\nDATA_PCT = 0.20\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer and compressed.\npd.options.display.float_format = '{:,.3f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# ---

# ## 3.0 Loading the Datasets

# In[6]:


get_ipython().run_cell_magic('time', '', "# Load the Test Datset utilizing the provided dtype arrays for memory eficiency\n\ndtypes_df = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/test_dtypes.csv')\ndtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "TRN_PATH = '/kaggle/input/tabular-playground-series-oct-2022/train_1.csv'\ntrn_data = pd.read_csv(TRN_PATH, dtype = dtypes)\n")


# In[8]:


get_ipython().run_cell_magic('time', '', "TST_PATH = '/kaggle/input/tabular-playground-series-oct-2022/test.csv'\ntst_data = pd.read_csv(TST_PATH, dtype = dtypes)\n")


# In[9]:


get_ipython().run_cell_magic('time', '', "# Create\nSUB_PATH = '/kaggle/input/tabular-playground-series-oct-2022/sample_submission.csv'\nsubmission = pd.read_csv(SUB_PATH)\n")


# ---

# ## 4.0 Exploring the Datasets...

# In[10]:


get_ipython().run_cell_magic('time', '', '# Display some basic dataset information, the most important is the memory usage...\n\ntrn_data.info(verbose=False)\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '# Display information of the first 5 rows in the datset...\n\ntrn_data.head()\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Display advanced statistics of the dataset...\n\ntrn_data.describe() \n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Display the number of NaNs in each column of the dataset...\n\ntrn_data.isnull().sum()\n')


# ---

# ## 5.0 Feature Engineering...
# Hello, at this point I don't have any feature engineering built in the Notebbok, first I like to try how the feature provided will perform... 

# In[14]:


# ...
# ...


# ---

# ## 6.0 Preparing the Datasets for the Model...

# In[15]:


get_ipython().run_cell_magic('time', '', "# The type of ML model I'm currenthly using can deal with NaNs\n\nDEFAULT_VALUE = -1000\ntrn_data = trn_data.fillna(value = DEFAULT_VALUE)\ntst_data = tst_data.fillna(value = DEFAULT_VALUE)\n")


# In[16]:


get_ipython().run_cell_magic('time', '', "# ...\n\n# List of featuare that needs to be avoided...\nskip = ['game_num', \n        'event_id', \n        'event_time', \n        'player_scoring_next', \n        'team_scoring_next', \n        'team_A_scoring_within_10sec',\n        'team_B_scoring_within_10sec'\n       ]\n\n# Using a list comprehension we generate a list of features to train the model...\nfeatures = [feat for feat in trn_data.columns if feat not in skip]\n\nlabel_a = 'team_A_scoring_within_10sec'\nlabel_b = 'team_B_scoring_within_10sec'\n")


# In[17]:


get_ipython().run_cell_magic('time', '', '# Scale the train and test datasets to improve the learning capabilities of the model.\n# Modified to be during the training stage.\n\n# scaler = StandardScaler()\n# trn_data[features] = scaler.fit_transform(trn_data[features])\n')


# ---

# ## 7.0 Baseline Modeling, Using a Keras Classifier...

# ### 7.1 Importing Tenforflow Libraries

# In[18]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping\nfrom tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout, Concatenate\n')


# In[19]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nfrom sklearn.model_selection import KFold, StratifiedKFold, GroupKFold\nfrom sklearn.metrics import roc_auc_score, roc_curve, accuracy_score\n')


# In[20]:


get_ipython().run_cell_magic('time', '', 'import datetime\nimport random\nimport math\n')


# ### 7.2 Defining Model Architecture

# In[21]:


get_ipython().run_cell_magic('time', '', "def nn_model_one():\n    \n    '''\n    Function to define the Neuronal Network architecture...\n    '''\n    \n    L2 = 65e-6\n    dropout_value = 0.05\n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x0 = Dense(256, kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(inputs)\n    x0 = BatchNormalization()(x0)\n    x0 = Dropout(dropout_value)(x0)\n    \n    x1 = Dense(256, kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x0)\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(64,  kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x1)\n    x1 = Concatenate()([x1, x0])\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(32, kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x1)\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(1,  \n               #kernel_regularizer = tf.keras.regularizers.l2(4e-4), \n               activation = 'sigmoid')(x1)\n    \n    model = Model(inputs, x1)\n    \n    return model\n")


# ### 7.3 Visualizing Model Architecture

# In[22]:


get_ipython().run_cell_magic('time', '', 'architecture = nn_model_one()\narchitecture.summary()\n')


# ### 7.4 Defining Model Parameters

# In[23]:


get_ipython().run_cell_magic('time', '', '# Defining model parameters...\nBATCH_SIZE         = 256\n\nEPOCHS             = 5\nEPOCHS_COSINEDECAY = 5\n\nDIAGRAMS           = True\nUSE_PLATEAU        = True\nINFERENCE          = False\nVERBOSE            = 1 \n\nTARGET             = label_a\n')


# ### 7.5 Defining Model Train Function

# In[24]:


get_ipython().run_cell_magic('time', '', '# Defining model training function...\n\ndef fit_model(X_train, y_train, X_val, y_val, run = 0):\n    \'\'\'\n    This function train an NN model...\n    \'\'\'\n    \n    lr_start = 0.2 # Initial value for learning rate...\n\n    start_time = datetime.datetime.now()\n    scaler = StandardScaler()\n    \n    X_train[features] = scaler.fit_transform(X_train[features])\n\n    epochs = EPOCHS    \n    lr = ReduceLROnPlateau(monitor = \'val_loss\', factor = 0.1, patience = 8, verbose = VERBOSE)\n    es = EarlyStopping(monitor = \'val_loss\',patience = 16, verbose = 1, mode = \'min\', restore_best_weights = True)\n    tm = tf.keras.callbacks.TerminateOnNaN()\n    callbacks = [lr, es, tm]\n    \n    # Cosine Learning Rate Decay\n    if USE_PLATEAU == False:\n        epochs = EPOCHS_COSINEDECAY\n        lr_end = 0.0002\n\n        def cosine_decay(epoch):\n            if epochs > 1:\n                w = (1 + math.cos(epoch / (epochs - 1) * math.pi)) / 2\n            else:\n                w = 1\n            return w * lr_start + (1 - w) * lr_end\n        \n        lr = LearningRateScheduler(cosine_decay, verbose = 0)\n        callbacks = [lr, tm]\n        \n    model = nn_model_one()\n    \n    optimizer_func = tf.keras.optimizers.Adam(learning_rate = lr_start)\n    loss_func = tf.keras.losses.BinaryCrossentropy()\n    model.compile(optimizer = optimizer_func, loss = loss_func)\n    \n    X_val[features] = scaler.transform(X_val[features])\n    validation_data = (X_val, y_val)\n    \n    history = model.fit(X_train, \n                        y_train, \n                        validation_data = validation_data, \n                        epochs          = epochs,\n                        verbose         = VERBOSE,\n                        batch_size      = BATCH_SIZE,\n                        shuffle         = True,\n                        callbacks       = callbacks\n                       )\n    \n    history_list.append(history.history)\n    print(f\'Training loss:{history_list[-1]["loss"][-1]:.3f}\')\n    callbacks, es, lr, tm, history = None, None, None, None, None\n    \n    \n    y_val_pred_proba = model.predict(X_val, batch_size = BATCH_SIZE, verbose = VERBOSE)\n    y_val_pred = [1 if x > 0.5 else 0 for x in y_val_pred_proba]\n    \n    acc_score = accuracy_score(y_val, y_val_pred)\n    auc_score = roc_auc_score(y_val, y_val_pred_proba)\n    \n    print(f\'Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}\'\n          f\' | ACC: {acc_score:.5f} | AUC: {auc_score:.5f}\')\n    \n    auc_score_list.append(auc_score)\n    \n    tst_data_scaled = tst_data.copy(deep = True)\n    tst_data_scaled[features] = scaler.transform(tst_data_scaled[features])\n    tst_pred = model.predict(tst_data_scaled[features])\n    predictions.append(tst_pred)\n    \n    return model\n')


# ### 7.6 Defining a Cross Validation Function & Training the Model

# In[25]:


get_ipython().run_cell_magic('time', '', "# Create empty lists to store NN information...\nTARGET = label_a\n\nhistory_list = []\nauc_score_list = []\npredictions  = []\n\n# Define kfolds for training purposes...\n#kf = KFold(n_splits = 5)\nkf = StratifiedKFold(n_splits = 5)\n\n#kf = GroupKFold(n_splits = 5)\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data, trn_data[TARGET])):\n    x_train, x_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_val = trn_data.iloc[trn_idx][TARGET], trn_data.iloc[val_idx][TARGET]\n    \n    fit_model(x_train, y_train, x_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(auc_score_list):.5f}')\n")


# In[26]:


submission['team_A_scoring_within_10sec'] = np.mean(predictions, axis = 0)


# In[27]:


get_ipython().run_cell_magic('time', '', "# Create empty lists to store NN information...\nTARGET = label_b\n\nhistory_list = []\nauc_score_list = []\npredictions  = []\n\n# Define kfolds for training purposes...\n#kf = KFold(n_splits = 5)\nkf = StratifiedKFold(n_splits = 5)\n\n#kf = GroupKFold(n_splits = 5)\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data, trn_data[TARGET])):\n    x_train, x_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_val = trn_data.iloc[trn_idx][TARGET], trn_data.iloc[val_idx][TARGET]\n    \n    fit_model(x_train, y_train, x_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(auc_score_list):.5f}')\n")


# In[28]:


submission['team_B_scoring_within_10sec'] = np.mean(predictions, axis = 0)


# ---

# ## 8.0 Creating Submission Files.

# In[29]:


get_ipython().run_cell_magic('time', '', "# Creates a kaggle submission file...\nsubmission.to_csv('submission_10292022.csv', index = False)\n")


# ---

# In[ ]:




