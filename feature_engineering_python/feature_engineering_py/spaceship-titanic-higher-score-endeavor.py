#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic Higher Score Endeavor 👽
# The goal of this Notebook is simple. Improve my score from my previous work...
# Because I will attemp multiple iterations and I will loose track of my changes I will use Neptune.AI to keep track of all the experiments...
# 
# https://www.kaggle.com/code/cv13j0/spaceship-titanic-nn-model-feature-eng/edit
# 

# In[1]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[2]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import LabelEncoder \n')


# In[3]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping\nfrom tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout\n\nfrom sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nimport random\n\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score, roc_curve, accuracy_score\nimport datetime\nimport math\n')


# In[4]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[5]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 100\nNCOLS = 8\n# Main data location path...\nBASE_PATH = '...'\n")


# In[6]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "trn_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')\ntst_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')\n\nsub = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')\n")


# In[8]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '# Display the information from the dataset...\ntrn_data.info()\n')


# In[11]:


get_ipython().run_cell_magic('time', '', "# Filling NaNs Based on Feature Engineering...\ndef fill_nans_by_age(df, age_limit = 13):\n    df['RoomService'] = np.where(df['Age'] < age_limit, 0, df['RoomService'])\n    df['FoodCourt'] = np.where(df['Age'] < age_limit, 0, df['FoodCourt'])\n    df['ShoppingMall'] = np.where(df['Age'] < age_limit, 0, df['ShoppingMall'])\n    df['Spa'] = np.where(df['Age'] < age_limit, 0, df['Spa'])\n    df['VRDeck'] = np.where(df['Age'] < age_limit, 0, df['VRDeck'])\n    \n    return df\n")


# In[12]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_nans_by_age(trn_data)\ntst_data =  fill_nans_by_age(tst_data)\n')


# In[13]:


get_ipython().run_cell_magic('time', '', "# Filling NaNs Based on Feature Engineering...\ndef fill_nans_by_cryo(df, age_limit = 13):\n    df['RoomService'] = np.where(df['CryoSleep'] == True, 0, df['RoomService'])\n    df['FoodCourt'] = np.where(df['CryoSleep'] == True, 0, df['FoodCourt'])\n    df['ShoppingMall'] = np.where(df['CryoSleep'] == True, 0, df['ShoppingMall'])\n    df['Spa'] = np.where(df['CryoSleep'] == True, 0, df['Spa'])\n    df['VRDeck'] = np.where(df['CryoSleep'] == True, 0, df['VRDeck'])\n    \n    return df\n")


# In[14]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_nans_by_cryo(trn_data)\ntst_data =  fill_nans_by_cryo(tst_data)\n')


# In[15]:


get_ipython().run_cell_magic('time', '', "def age_groups(df, age_limit = 13):\n    df['AgeGroup'] = np.where(df['Age'] < age_limit, 0, 1)\n    return df\n")


# In[16]:


get_ipython().run_cell_magic('time', '', 'trn_data =  age_groups(trn_data)\ntst_data =  age_groups(tst_data)\n')


# In[17]:


get_ipython().run_cell_magic('time', '', "def fill_missing(df):\n    '''\n    Fill NaNs values or with mean or most commond value...\n    \n    '''\n    \n    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n    \n    numeric_tmp = df.select_dtypes(include = numerics)\n    categ_tmp = df.select_dtypes(exclude = numerics)\n\n    for col in numeric_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mean())\n        \n    for col in categ_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mode()[0])\n        \n    print('...')\n    \n    return df\n")


# In[18]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_missing(trn_data)\ntst_data =  fill_missing(tst_data)\n')


# In[19]:


get_ipython().run_cell_magic('time', '', "def total_billed(df):\n    '''\n    Calculates total amount billed in the trip to the passenger... \n    Args:\n    Returns:\n    \n    '''\n    \n    df['Total_Billed'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']\n    return df\n")


# In[20]:


get_ipython().run_cell_magic('time', '', 'trn_data = total_billed(trn_data)\ntst_data = total_billed(tst_data)\n')


# In[21]:


get_ipython().run_cell_magic('time', '', "def cabin_separation(df):\n    '''\n    Split the Cabin name into Deck, Number and Side\n    \n    '''\n    \n    df['CabinDeck'] = df['Cabin'].str.split('/', expand=True)[0]\n    df['CabinNum']  = df['Cabin'].str.split('/', expand=True)[1]\n    df['CabinSide'] = df['Cabin'].str.split('/', expand=True)[2]\n    \n    df.drop(columns = ['Cabin'], inplace = True)\n    return df\n")


# In[22]:


get_ipython().run_cell_magic('time', '', 'trn_data = cabin_separation(trn_data)\ntst_data = cabin_separation(tst_data)\n')


# In[23]:


get_ipython().run_cell_magic('time', '', "def name_ext(df):\n    '''\n    Split the Name of the passenger into First and Family...\n    \n    '''\n    \n    df['FirstName'] = df['Name'].str.split(' ', expand=True)[0]\n    df['FamilyName'] = df['Name'].str.split(' ', expand=True)[1]\n    df.drop(columns = ['Name'], inplace = True)\n    return df\n")


# In[24]:


get_ipython().run_cell_magic('time', '', 'trn_data = name_ext(trn_data)\ntst_data = name_ext(tst_data)\n')


# In[25]:


def extract_group(df):
    '''
    '''
    df['TravelGroup'] =  df['PassengerId'].str.split('_', expand = True)[0]
    return df


# In[26]:


get_ipython().run_cell_magic('time', '', 'trn_data = extract_group(trn_data)\ntst_data = extract_group(tst_data)\n')


# In[27]:


get_ipython().run_cell_magic('time', '', "# A list of the original variables from the dataset\nnumerical_features = [\n                      'Age', \n                      'RoomService', \n                      'FoodCourt', \n                      'ShoppingMall', \n                      'Spa', \n                      'VRDeck', \n                      'Total_Billed'\n                     ]\n\ncategorical_features = [\n                        #'Name',\n                        'FirstName',\n                        'FamilyName',\n                        'CabinNum',\n                        #'TravelGroup',\n                       ]\n\n\ncategorical_features_onehot = [\n                               'HomePlanet',\n                               'CryoSleep',\n                               #'Cabin',\n                               'CabinDeck',\n                               'CabinSide',\n                               'Destination',\n                               'VIP',\n                               ]\n\ntarget_feature = 'Transported'\n")


# In[28]:


get_ipython().run_cell_magic('time', '', "\ndef encode_categorical(train_df, test_df, categ_feat = categorical_features):\n    '''\n    \n    '''\n    encoder_dict = {}\n    \n    concat_data = pd.concat([trn_data[categ_feat], tst_data[categ_feat]])\n    \n    for col in concat_data.columns:\n        print('Encoding: ', col, '...')\n        encoder = LabelEncoder()\n        encoder.fit(concat_data[col])\n        encoder_dict[col] = encoder\n\n        train_df[col + '_Enc'] = encoder.transform(train_df[col])\n        test_df[col + '_Enc'] = encoder.transform(test_df[col])\n    \n    train_df = train_df.drop(columns = categ_feat, axis = 1)\n    test_df = test_df.drop(columns = categ_feat, axis = 1)\n\n    return train_df, test_df\n")


# In[29]:


get_ipython().run_cell_magic('time', '', 'trn_data, tst_data = encode_categorical(trn_data, tst_data, categorical_features)\n')


# In[30]:


def one_hot(df, one_hot_categ):
    for col in one_hot_categ:
        tmp = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, tmp], axis = 1)
    df = df.drop(columns = one_hot_categ)
    return df


# In[31]:


trn_data = one_hot(trn_data, categorical_features_onehot) 
tst_data = one_hot(tst_data, categorical_features_onehot) 


# In[32]:


get_ipython().run_cell_magic('time', '', "remove = ['PassengerId', \n          'Route', \n          'FirstName_Enc', \n          'CabinNum_Enc', \n          'Transported',\n          'Cabin',\n          #'IsKid', \n          #'IsAdult', \n          #'IsOlder'\n         ]\nfeatures = [feat for feat in trn_data.columns if feat not in remove]\n")


# In[33]:


get_ipython().run_cell_magic('time', '', 'features\n')


# In[34]:


get_ipython().run_cell_magic('time', '', "drop_out_pct = 0.45\nl2 = 80e-6\n\ndef nn_model():\n    '''\n    '''\n    \n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x = Dense(2048, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(l2), \n              activation = activation_func)(inputs)\n    \n    x = Dropout(drop_out_pct)(x)\n    x = BatchNormalization()(x)\n    \n    \n    x = Dense(512, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(l2), \n              activation = activation_func)(x)\n    \n    x = Dropout(drop_out_pct)(x)\n    x = BatchNormalization()(x)\n\n    x = Dense(16, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(l2), \n              activation = activation_func)(x)\n\n    x = Dropout(drop_out_pct)(x)\n    x = BatchNormalization()(x)\n\n\n    x = Dense(1 , \n              #use_bias  = True, \n              #kernel_regularizer = tf.keras.regularizers.l2(30e-6),\n              activation = 'sigmoid')(x)\n    \n    model = Model(inputs, x)\n    \n    return model\n")


# In[35]:


get_ipython().run_cell_magic('time', '', 'architecture = nn_model()\narchitecture.summary()\n')


# In[36]:


get_ipython().run_cell_magic('time', '', "# Defining model parameters...\nBATCH_SIZE         = 128\nEPOCHS             = 512 \nEPOCHS_COSINEDECAY = 512 \nDIAGRAMS           = True\nUSE_PLATEAU        = True\nINFERENCE          = False\nVERBOSE            = 0 \nTARGET             = 'Transported'\n")


# In[37]:


get_ipython().run_cell_magic('time', '', '# Defining model training function...\ndef fit_model(X_train, y_train, X_val, y_val, run = 0):\n    \'\'\'\n    \'\'\'\n    lr_start = 0.1\n    start_time = datetime.datetime.now()\n    \n    scaler = StandardScaler()\n\n    X_train = scaler.fit_transform(X_train)\n\n    epochs = EPOCHS    \n    lr = ReduceLROnPlateau(monitor = \'val_loss\', factor = 0.7, patience = 4, verbose = VERBOSE)\n    es = EarlyStopping(monitor = \'val_loss\',patience = 48, verbose = 1, mode = \'min\', restore_best_weights = True)\n    tm = tf.keras.callbacks.TerminateOnNaN()\n    callbacks = [lr, es, tm]\n    \n    # Cosine Learning Rate Decay\n    if USE_PLATEAU == False:\n        epochs = EPOCHS_COSINEDECAY\n        lr_end = 0.0002\n\n        def cosine_decay(epoch):\n            if epochs > 1:\n                w = (1 + math.cos(epoch / (epochs - 1) * math.pi)) / 2\n            else:\n                w = 1\n            return w * lr_start + (1 - w) * lr_end\n        \n        lr = LearningRateScheduler(cosine_decay, verbose = 0)\n        callbacks = [lr, tm]\n        \n    model = nn_model()\n    \n    optimizer_func = tf.keras.optimizers.Adam(learning_rate = lr_start)\n    loss_func = tf.keras.losses.BinaryCrossentropy()\n    model.compile(optimizer = optimizer_func, loss = loss_func)\n    \n    X_val = scaler.transform(X_val)\n    validation_data = (X_val, y_val)\n    \n    history = model.fit(X_train, \n                        y_train, \n                        validation_data = validation_data, \n                        epochs          = epochs,\n                        verbose         = VERBOSE,\n                        batch_size      = BATCH_SIZE,\n                        shuffle         = True,\n                        callbacks       = callbacks\n                       )\n    \n    history_list.append(history.history)\n    print(f\'Training loss:{history_list[-1]["loss"][-1]:.3f}\')\n    callbacks, es, lr, tm, history = None, None, None, None, None\n    \n    \n    y_val_pred = model.predict(X_val, batch_size = BATCH_SIZE, verbose = VERBOSE)\n    y_val_pred = [1 if x > 0.5 else 0 for x in y_val_pred]\n    \n    score = accuracy_score(y_val, y_val_pred)\n    print(f\'Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}\'\n          f\'| ACC: {score:.5f}\')\n    \n    score_list.append(score)\n    \n    tst_data_scaled = scaler.transform(tst_data[features])\n    tst_pred = model.predict(tst_data_scaled)\n    predictions.append(tst_pred)\n    \n    return model\n')


# In[38]:


get_ipython().run_cell_magic('time', '', "# Create empty lists to store NN information...\nhistory_list = []\nscore_list   = []\npredictions  = []\n\n# Define kfolds for training purposes...\nkf = KFold(n_splits = 5)\n\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data)):\n    X_train, X_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_val = trn_data.iloc[trn_idx][TARGET], trn_data.iloc[val_idx][TARGET]\n    \n    fit_model(X_train, y_train, X_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(score_list):.5f}')\n")


# In[39]:


get_ipython().run_cell_magic('time', '', "# Populated the prediction on the submission dataset and creates an output file\nsub['Transported'] = np.array(predictions).mean(axis = 0)\nsub['Transported'] = np.where(sub['Transported'] > 0.5, True, False)\nsub.to_csv('my_submission_051822.csv', index = False)\n")


# In[40]:


sub.head()


# In[41]:


# OOF AUC: 0.79950 Plain Model Version 1.01 >>> LB Score AUC: 0.81108, Using CPU
# OOF AUC: 0.79904 Plain Model Version 1.01 >>> LB Score AUC: 0.81108, Using GPU
# OOF AUC: 0.80019 Plain Model Version 1.01 >>> LB Score AUC: 0.81014, Using GPU, Earlier Stopping Patience: 48


# In[ ]:




