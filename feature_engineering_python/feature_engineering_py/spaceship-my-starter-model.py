#!/usr/bin/env python
# coding: utf-8

# # 🌌 Spaceship My Starter Model
# 
# Hello a Simple Starter Model, **Stay Tune for More Updates...**

# ### File and Data Field Descriptions
# 
# **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * Destination - The planet the passenger will be debarking to.
# * Age - The age of the passenger.
# * VIP - Whether the passenger has paid for special VIP service during the voyage.
# * RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * Name - The first and last names of the passenger.
# * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# 
# **test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
# 
# **sample_submission.csv** - A submission file in the correct format.
# 
# * PassengerId - Id for each passenger in the test set.
# * Transported - The target. For each passenger, predict either True or False.

# # Loading Libraries...

# In[1]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# # Seeting Notebook Parameters...

# In[2]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[3]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 50\nNCOLS = 15\n# Main data location path...\nBASE_PATH = '...'\n")


# In[4]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# # Loading Information from CSV...

# In[5]:


get_ipython().run_cell_magic('time', '', "trn_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')\ntst_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')\n\nsub = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')\n")


# # Exploring the Information Available...

# In[6]:


get_ipython().run_cell_magic('time', '', 'trn_data.info()\n')


# In[7]:


get_ipython().run_cell_magic('time', '', 'trn_data.head()\n')


# In[8]:


get_ipython().run_cell_magic('time', '', 'trn_data.describe()\n')


# In[9]:


get_ipython().run_cell_magic('time', '', "def describe_categ(df):\n    for col in df.columns:\n        unique_samples = list(df[col].unique())\n        unique_values = df[col].nunique()\n\n        print(f' {col}: {unique_values} Unique Values,  Data Sample >> {unique_samples[:5]}')\n    print(' ...')\n    return None\n")


# In[10]:


get_ipython().run_cell_magic('time', '', 'describe_categ(trn_data)\n')


# In[11]:


get_ipython().run_cell_magic('time', '', 'describe_categ(tst_data)\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'trn_data.isnull().sum()\n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'tst_data.head()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', 'tst_data.isnull().sum()\n')


# In[15]:


get_ipython().run_cell_magic('time', '', 'sub.sample(10)\n')


# # Exploring the Target Variable...

# In[16]:


get_ipython().run_cell_magic('time', '', "def analyse_categ_target(df, target = 'Transported'):\n    \n    transported = df[df[target] == True].shape[0]\n    not_transported = df[df[target] == False].shape[0]\n    total = transported + not_transported\n    \n    print(f'Transported     : {transported / total:.2f} %')\n    print(f'Not Transported : {not_transported / total:.2f} %')\n    print(f'Total Passengers: {total}')\n    print('...')\n")


# In[17]:


get_ipython().run_cell_magic('time', '', 'analyse_categ_target(trn_data)\n')


# In[18]:


get_ipython().run_cell_magic('time', '', "trn_passenger_ids = set(trn_data['PassengerId'].unique())\ntst_passenger_ids = set(tst_data['PassengerId'].unique())\nintersection = trn_passenger_ids.intersection(tst_passenger_ids)\nprint('Overlapped Passengers:', len(intersection))\n")


# # Feature Engineering...

# In[19]:


get_ipython().run_cell_magic('time', '', "def fill_missing(df):\n    '''\n    Fill nan values or missing data with mean or most commond value...\n    \n    '''\n    \n    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n    numeric_tmp = df.select_dtypes(include = numerics)\n    categ_tmp = df.select_dtypes(exclude = numerics)\n\n    for col in numeric_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mean())\n        \n    for col in categ_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mode()[0])\n        \n    print('...')\n    \n    return df\n")


# In[20]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_missing(trn_data)\ntst_data =  fill_missing(tst_data)\n')


# In[21]:


get_ipython().run_cell_magic('time', '', "def total_billed(df):\n    '''\n    Calculates total amount billed in the trip to the passenger... \n    Args:\n    Returns:\n    \n    '''\n    \n    df['Total_Billed'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']\n    return df\n")


# In[22]:


get_ipython().run_cell_magic('time', '', 'trn_data = total_billed(trn_data)\ntst_data = total_billed(tst_data)\n')


# In[23]:


get_ipython().run_cell_magic('time', '', "def name_ext(df):\n    '''\n    Split the Name of the passenger into First and Family...\n    \n    '''\n    \n    df['FirstName'] = df['Name'].str.split(' ', expand=True)[0]\n    df['FamilyName'] = df['Name'].str.split(' ', expand=True)[1]\n    df.drop(columns = ['Name'], inplace = True)\n    return df\n")


# In[24]:


get_ipython().run_cell_magic('time', '', 'trn_data = name_ext(trn_data)\ntst_data = name_ext(tst_data)\n')


# In[25]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_data.groupby('FamilyName')['PassengerId'].count().reset_index()\ntst_relatives = tst_data.groupby('FamilyName')['PassengerId'].count().reset_index()\n")


# In[26]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_relatives.rename(columns = {'PassengerId': 'NumRelatives'})\ntst_relatives = tst_relatives.rename(columns = {'PassengerId': 'NumRelatives'})\n")


# In[27]:


get_ipython().run_cell_magic('time', '', "trn_data = trn_data.merge(trn_relatives, how = 'left', on = ['FamilyName'])\ntst_data = tst_data.merge(tst_relatives, how = 'left', on = ['FamilyName'])\n")


# In[28]:


get_ipython().run_cell_magic('time', '', "def cabin_separation(df):\n    '''\n    Split the Cabin name into Deck, Number and Side\n    \n    '''\n    \n    df['CabinDeck'] = df['Cabin'].str.split('/', expand=True)[0]\n    df['CabinNum'] = df['Cabin'].str.split('/', expand=True)[1]\n    df['CabinSide'] = df['Cabin'].str.split('/', expand=True)[2]\n    df.drop(columns = ['Cabin'], inplace = True)\n    return df\n")


# In[29]:


get_ipython().run_cell_magic('time', '', 'trn_data = cabin_separation(trn_data)\ntst_data = cabin_separation(tst_data)\n')


# In[30]:


get_ipython().run_cell_magic('time', '', "def route(df):\n    '''\n    Calculate a combination of origin and destinations, creates a new feature for training.\n    Args:\n    Returns:\n    '''\n    \n    df['Route'] = df['HomePlanet'] + df['Destination']\n    return df\n")


# In[31]:


get_ipython().run_cell_magic('time', '', 'trn_data = route(trn_data)\ntst_data = route(tst_data)\n')


# In[32]:


def age_groups(df):
    '''
    
    '''
    df['IsKid'] = np.where(df['Age'] <= 10, 1, 0)
    df['IsAdult'] = np.where(df['Age'] > 10, 1, 0)
    df['IsOlder'] = np.where(df['Age'] >= 65, 1, 0)
    return df


# In[33]:


get_ipython().run_cell_magic('time', '', 'trn_data = age_groups(trn_data)\ntst_data = age_groups(tst_data)\n')


# In[34]:


def extract_group(df):
    '''
    '''
    df['TravelGroup'] =  df['PassengerId'].str.split('_', expand = True)[0]
    return df


# In[35]:


get_ipython().run_cell_magic('time', '', 'trn_data = extract_group(trn_data)\ntst_data = extract_group(tst_data)\n')


# In[36]:


get_ipython().run_cell_magic('time', '', 'trn_data.head()\n')


# # Pre-Processing for Training

# In[37]:


get_ipython().run_cell_magic('time', '', "# A list of the original variables from the dataset\nnumerical_features = ['Age', \n                      'RoomService', \n                      'FoodCourt', \n                      'ShoppingMall', \n                      'Spa', \n                      'VRDeck', \n                      'Total_Billed'\n                     ]\n\ncategorical_features = ['FirstName',\n                        'FamilyName',\n                        'CabinNum',\n                        'TravelGroup',]\n\n\ncategorical_features_onehot = ['HomePlanet',\n                               'CryoSleep',\n                               'CabinDeck',\n                               'CabinSide',\n                               'Destination',\n                               'VIP',]\n\ntarget_feature = 'Transported'\n")


# In[38]:


get_ipython().run_cell_magic('time', '', "from sklearn.preprocessing import LabelEncoder \n\ndef encode_categorical(train_df, test_df, categ_feat = categorical_features):\n    '''\n    \n    '''\n    encoder_dict = {}\n    \n    concat_data = pd.concat([trn_data[categ_feat], tst_data[categ_feat]])\n    \n    for col in concat_data.columns:\n        print('Encoding: ', col, '...')\n        encoder = LabelEncoder()\n        encoder.fit(concat_data[col])\n        encoder_dict[col] = encoder\n\n        train_df[col + '_Enc'] = encoder.transform(train_df[col])\n        test_df[col + '_Enc'] = encoder.transform(test_df[col])\n    \n    train_df = train_df.drop(columns = categ_feat, axis = 1)\n    test_df = test_df.drop(columns = categ_feat, axis = 1)\n\n    return train_df, test_df\n")


# In[39]:


get_ipython().run_cell_magic('time', '', 'trn_data, tst_data = encode_categorical(trn_data, tst_data, categorical_features)\n')


# In[40]:


def one_hot(df, one_hot_categ):
    for col in one_hot_categ:
        tmp = pd.get_dummies(df[col], prefix = col)
        df = pd.concat([df, tmp], axis = 1)
    df = df.drop(columns = one_hot_categ)
    return df


# In[41]:


trn_data = one_hot(trn_data, categorical_features_onehot) 
tst_data = one_hot(tst_data, categorical_features_onehot) 


# In[42]:


trn_data.info(verbose=True)


# # Simple CV Sttrategy 80/20 Split

# In[43]:


get_ipython().run_cell_magic('time', '', 'trn_data.columns\n')


# In[44]:


get_ipython().run_cell_magic('time', '', "remove = ['PassengerId', \n          'Route', \n          'FirstName_Enc', \n          'CabinNum_Enc', \n          'Transported', \n          #'IsKid', \n          #'IsAdult', \n          #'IsOlder'\n         ]\nfeatures = [feat for feat in trn_data.columns if feat not in remove]\n")


# In[45]:


get_ipython().run_cell_magic('time', '', 'features\n')


# In[46]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import train_test_split\ntest_size_pct = 0.01\nX_train, X_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target_feature], test_size = test_size_pct, random_state = 42)\n')


# # Training a ML Classifier

# In[47]:


get_ipython().run_cell_magic('time', '', 'X_train.shape\n')


# In[48]:


get_ipython().run_cell_magic('time', '', 'from xgboost  import XGBClassifier\nfrom catboost import CatBoostClassifier\nfrom lightgbm import LGBMClassifier\n')


# In[49]:


get_ipython().run_cell_magic('time', '', "param = {'learning_rate': 0.05,\n         'n_estimators': 1024,\n         'n_jobs': -1,\n         'random_state': 42,\n         'objective': 'binary:logistic',\n        }\n")


# In[50]:


get_ipython().run_cell_magic('time', '', "cls = XGBClassifier(**param)\ncls.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['logloss'], early_stopping_rounds = 128, verbose = False)\n")


# In[51]:


get_ipython().run_cell_magic('time', '', "from sklearn.metrics import accuracy_score\n\nval_preds = cls.predict(X_valid[features])\nval_preds = val_preds.astype('bool')\naccuracy = accuracy_score(val_preds, y_valid)\n")


# In[52]:


get_ipython().run_cell_magic('time', '', "print(f'Mean accuracy score: {accuracy}')\n")


# In[53]:


# Mean accuracy score: 0.7586206896551724
# Mean accuracy score: 0.7586206896551724
# Mean accuracy score: 0.7471264367816092
# Mean accuracy score: 0.7816091954022989
# Mean accuracy score: 0.7827586206896552
# Mean accuracy score: 0.7908045977011494 (One Hot Encode...)
# Mean accuracy score: 0.7862068965517242
# Mean accuracy score: 0.7954022988505747
# Mean accuracy score: 0.7701149425287356 (Best Model)


# In[54]:


get_ipython().run_cell_magic('time', '', "import matplotlib.pyplot as plt\n\ndef feature_importance(clf):\n    importances = clf.feature_importances_\n    i = np.argsort(importances)\n    features = X_train.columns\n    plt.title('Feature Importance')\n    plt.barh(range(len(i)), importances[i], align='center')\n    plt.yticks(range(len(i)), [features[x] for x in i])\n    plt.xlabel('Scale')\n    plt.show()\n")


# In[55]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize = (10,7))\nfeature_importance(cls)\n')


# In[56]:


get_ipython().run_cell_magic('time', '', 'preds = cls.predict(tst_data[features])\n')


# In[57]:


get_ipython().run_cell_magic('time', '', "sub['Transported'] = preds\nsub.to_csv('submission_simple_split_03272022.csv', index = False)\n")


# # Time for Optuna

# In[58]:


get_ipython().run_cell_magic('time', '', 'import optuna\n')


# In[59]:


get_ipython().run_cell_magic('time', '', 'X_train, X_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target_feature])\n\ndef objective(trial):\n    n_estimators = trial.suggest_int("n_estimators", 8, 2048)\n    max_depth = trial.suggest_int("max_depth", 2, 16)\n    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)\n    subsample = trial.suggest_float("subsample", 0.5, 1)\n    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1)\n    reg_lambda = trial.suggest_float("reg_lambda", 1, 20)\n    reg_alpha = trial.suggest_float("reg_alpha", 0, 20)\n    gamma = trial.suggest_float("gamma", 0, 20)\n    min_child_weight  = trial.suggest_int("min_child_weight", 0, 128)\n    \n    clf = XGBClassifier(n_estimators  = n_estimators,\n                       learning_rate = learning_rate,\n                       max_depth = max_depth,\n                       subsample = subsample,\n                       colsample_bytree = colsample_bytree,\n                       reg_lambda = reg_lambda,\n                       reg_alpha = reg_alpha,\n                       gamma = gamma,\n                       min_child_weight = min_child_weight,\n                       random_state  = 69,\n                       objective = \'binary:logistic\',\n                       tree_method = \'gpu_hist\',\n                      )\n    \n    clf.fit(X_train, y_train)\n    \n    valid_pred = clf.predict(X_valid)\n    score = accuracy_score(y_valid, valid_pred)\n    \n    return score\n')


# In[60]:


get_ipython().run_cell_magic('time', '', '#study = optuna.create_study(direction = "maximize")\n#study.optimize(objective, n_trials = 100)\n')


# In[61]:


get_ipython().run_cell_magic('time', '', '#parameters = study.best_params\n#parameters\n')


# # Training a ML Classifier Using a N Fold CV Loop

# In[62]:


get_ipython().run_cell_magic('time', '', 'import optuna\nfrom sklearn.ensemble import ExtraTreesClassifier\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import MinMaxScaler\n')


# In[63]:


get_ipython().run_cell_magic('time', '', 'N_SPLITS = 20\nfolds = StratifiedKFold(n_splits = N_SPLITS, shuffle = True)\n')


# In[64]:


get_ipython().run_cell_magic('time', '', "optuna_params = {'n_estimators': 474,\n 'max_depth': 12,\n 'learning_rate': 0.17092496820170439,\n 'subsample': 0.8681931753955343,\n 'colsample_bytree': 0.6753406152924646,\n 'reg_lambda': 8.439432864212677,\n 'reg_alpha': 1.6521594249189673,\n 'gamma': 9.986385923158347,\n 'min_child_weight': 11,\n 'random_state': 69,\n 'objective': 'binary:logistic',\n 'tree_method':'gpu_hist',}\n")


# In[65]:


get_ipython().run_cell_magic('time', '', '\nscores  = []\ny_probs = []\n\nfor fold, (trn_id, val_id) in enumerate(folds.split(trn_data[features], trn_data[target_feature])):  \n    X_train, y_train = trn_data[features].iloc[trn_id], trn_data[target_feature].iloc[trn_id]\n    X_valid, y_valid = trn_data[features].iloc[val_id], trn_data[target_feature].iloc[val_id]\n    \n    #scaler = MinMaxScaler()\n    #X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])\n    #X_valid[numerical_features] = scaler.transform(X_valid[numerical_features])\n        \n    model = XGBClassifier(**optuna_params)\n    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = [\'logloss\'], early_stopping_rounds = 50, verbose = False)\n    \n    valid_pred = model.predict(X_valid)\n    valid_score = accuracy_score(y_valid, valid_pred)\n    \n    print("Fold:", fold, "Accuracy:", valid_score)\n    scores.append(valid_score)\n    #tst_data[numerical_features] = scaler.transform(tst_data[numerical_features])\n    y_probs.append(model.predict_proba(tst_data[features]))\n')


# In[66]:


get_ipython().run_cell_magic('time', '', 'print("Mean accuracy score:", np.array(scores).mean())\n')


# In[67]:


# Mean accuracy score: 0.8035192541977858
# Mean accuracy score: 0.8046655013507072
# Mean accuracy score: 0.8084726415594046
# Mean accuracy score: 0.8043222628317178
# Mean accuracy score: 0.8043269446979618
# Mean accuracy score: 0.8050191570881226
# Mean accuracy score: 0.7999563006515177
# Mean accuracy score: 0.8024831823719477
# Mean accuracy score: 0.7991493193495418
# Mean accuracy score: 0.8023674453096034 ... Best 


# In[68]:


get_ipython().run_cell_magic('time', '', "y_prob = sum(y_probs) / len(y_probs)\ny_prob_results = np.argmax(y_prob, axis = 1)\ny_prob_results = y_prob_results.astype('bool')\n\nsub['Transported'] = y_prob_results\nsub.to_csv('submission_twenty_fold_loop_03272022.csv', index = False)\n")


# In[ ]:




