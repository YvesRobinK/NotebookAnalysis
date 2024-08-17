#!/usr/bin/env python
# coding: utf-8

# # Spaceship Titanic NN Model 🌠
# 
# Hello Kaggle; 
# My goal is to provide the best single **Neuronal Network Model** for the **Space Ship Titanic** competition...
# I follow this typical strategy in majority of my Notebook and works quite well for me so I want it to share it.
# 
# ## Strategy
# 
# * Installing Libraries.
# * Importing Requiered Libraries
# * Setting Notebook Parameters
# * Loading the Datasets from CSV to Pandas Dataset
# * Exploring the Datasets
# * Creating Additional Features
# * Pre-Processing
# * Feature Selection
# * Advance Feature Creation (Clustering)
# * Selecting Features for Model Developmetn
# * Developing a DNN Model
# * Training a Model Utilizing a CV Loop
# * Generating Finala Predictions
# 
# 
# ## File and Data Field Descriptions
# 
# train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# * **PassengerId** - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# * **HomePlanet** - The planet the passenger departed from, typically their planet of permanent residence.
# * **CryoSleep** - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# * **Cabin** - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# * **Destination** - The planet the passenger will be debarking to.
# * **Age** - The age of the passenger.
# * **VIP** - Whether the passenger has paid for special VIP service during the voyage.
# * **RoomService**, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# * **Name** - The first and last names of the passenger.
# * **Transported** - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
# 
# test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.
# 
# sample_submission.csv - A submission file in the correct format.
# * **PassengerId** - Id for each passenger in the test set.
# * **Transported** - The target. For each passenger, predict either True or False.
# 
# 
# ## Baseline Model
# Hello, here is the location of a basaline model for the competition if you are interested in the modeling component...
# 
# https://www.kaggle.com/code/cv13j0/spaceship-my-starter-model
# 
# 
# **Last Updates**
# 
# Worked on the notebook structure and format.

# # 1.0 Installing Required Libraries

# In[1]:


get_ipython().run_cell_magic('capture', '', '!git clone https://github.com/analokmaus/kuma_utils.git\n')


# In[2]:


get_ipython().run_cell_magic('time', '', 'import os\nimport sys\nsys.path.append("kuma_utils/")\nfrom kuma_utils.preprocessing.imputer import LGBMImputer\n')


# In[3]:


get_ipython().run_cell_magic('capture', '', '# Install Gokinjo...\n!pip install gokinjo\n')


# ---

# # 2.0 Importing Libraries for the Model

# In[4]:


get_ipython().run_cell_magic('time', '', '# This Python 3 environment comes with many helpful analytics libraries installed\n# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n# For example, here\'s several helpful packages to load\n\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\n# Input data files are available in the read-only "../input/" directory\n# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n\nimport os\nfor dirname, _, filenames in os.walk(\'/kaggle/input\'):\n    for filename in filenames:\n        print(os.path.join(dirname, filename))\n\n# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" \n# You can also write temporary files to /kaggle/temp/, but they won\'t be saved outside of the current session\n')


# In[5]:


get_ipython().run_cell_magic('capture', '', 'from sklearn.preprocessing import LabelEncoder\nfrom gokinjo import knn_kfold_extract\nfrom gokinjo import knn_extract\n')


# In[6]:


get_ipython().run_cell_magic('time', '', 'from sklearn.preprocessing import LabelEncoder \n')


# ---

# # 3.0 Seeting Notebook Parameters...

# In[7]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')\n")


# In[8]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 100\nNCOLS = 8\n# Main data location path...\nBASE_PATH = '...'\n")


# In[9]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)\n")


# ---

# # 4.0 Loading Information from CSV...

# In[10]:


get_ipython().run_cell_magic('time', '', "trn_data = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')\ntst_data = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')\n\nsub = pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')\n")


# ---

# # 5.0 Exploring the Information Available

# In[11]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '# Display the information from the dataset...\ntrn_data.info()\n')


# In[14]:


get_ipython().run_cell_magic('time', '', '# Checking for empty or NaN values in the dataset by variable...\ntrn_data.isnull().sum()\n')


# ---

# # 6.0 Feature Engineering...

# ## 6.1 Filling NaNs by Using EDA Insights (Age)

# In[15]:


get_ipython().run_cell_magic('time', '', "# Filling NaNs Based on Feature Engineering...\ndef fill_nans_by_age(df, age_limit = 13):\n    df['RoomService'] = np.where(df['Age'] <= age_limit, 0, df['RoomService'])\n    df['FoodCourt'] = np.where(df['Age'] <= age_limit, 0, df['FoodCourt'])\n    df['ShoppingMall'] = np.where(df['Age'] <= age_limit, 0, df['ShoppingMall'])\n    df['Spa'] = np.where(df['Age'] <= age_limit, 0, df['Spa'])\n    df['VRDeck'] = np.where(df['Age'] <= age_limit, 0, df['VRDeck'])\n    \n    return df\n")


# In[16]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_nans_by_age(trn_data)\ntst_data =  fill_nans_by_age(tst_data)\n')


# ---

# ## 6.2 Filling NaNs by Using EDA Insights (CryoSleep)

# In[17]:


get_ipython().run_cell_magic('time', '', "# Filling NaNs Based on Feature Engineering...\ndef fill_nans_by_cryo(df, age_limit = 13):\n    df['RoomService'] = np.where(df['CryoSleep'] == True, 0, df['RoomService'])\n    df['FoodCourt'] = np.where(df['CryoSleep'] == True, 0, df['FoodCourt'])\n    df['ShoppingMall'] = np.where(df['CryoSleep'] == True, 0, df['ShoppingMall'])\n    df['Spa'] = np.where(df['CryoSleep'] == True, 0, df['Spa'])\n    df['VRDeck'] = np.where(df['CryoSleep'] == True, 0, df['VRDeck'])\n    \n    return df\n")


# In[18]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_nans_by_cryo(trn_data)\ntst_data =  fill_nans_by_cryo(tst_data)\n')


# ---

# ## 6.3 Creating Age Groups Using EDA Insights (Age)

# In[19]:


get_ipython().run_cell_magic('time', '', "def age_groups(df, age_limit = 13):\n    df['AgeGroup'] = np.where(df['Age'] < age_limit, 0, 1)\n    return df\n")


# In[20]:


get_ipython().run_cell_magic('time', '', 'trn_data =  age_groups(trn_data)\ntst_data =  age_groups(tst_data)\n')


# ---

# ## 6.7 Extracting Deck, Cabin Number and Side

# In[21]:


get_ipython().run_cell_magic('time', '', "def cabin_separation(df):\n    '''\n    Split the Cabin name into Deck, Number and Side\n    \n    '''\n    \n    df['CabinDeck'] = df['Cabin'].str.split('/', expand=True)[0]\n    df['CabinNum']  = df['Cabin'].str.split('/', expand=True)[1]\n    df['CabinSide'] = df['Cabin'].str.split('/', expand=True)[2]\n    \n    df.drop(columns = ['Cabin'], inplace = True)\n    return df\n")


# In[22]:


get_ipython().run_cell_magic('time', '', 'trn_data = cabin_separation(trn_data)\ntst_data = cabin_separation(tst_data)\n')


# ---

# ## 6.8 Extracting Family Name and Name

# In[23]:


get_ipython().run_cell_magic('time', '', "def name_ext(df):\n    '''\n    Split the Name of the passenger into First and Family...\n    \n    '''\n    \n    df['FirstName'] = df['Name'].str.split(' ', expand=True)[0]\n    df['FamilyName'] = df['Name'].str.split(' ', expand=True)[1]\n    df.drop(columns = ['Name'], inplace = True)\n    return df\n")


# In[24]:


get_ipython().run_cell_magic('time', '', 'trn_data = name_ext(trn_data)\ntst_data = name_ext(tst_data)\n')


# ---

# ## 6.9 Creating Age Groups, Based on EDA

# In[25]:


get_ipython().run_cell_magic('time', '', "def age_groups(df, age_limit = 13):\n    df['AgeGroup'] = np.where(df['Age'] < age_limit, 0, 1)\n    return df\n")


# In[26]:


get_ipython().run_cell_magic('time', '', 'trn_data =  age_groups(trn_data)\ntst_data =  age_groups(tst_data)\n')


# ---

# ## 6.10 Extracting Group

# In[27]:


def extract_group(df):
    '''
    '''
    df['TravelGroup'] =  df['PassengerId'].str.split('_', expand = True)[0]
    df['TravelGroupPos'] =  df['PassengerId'].str.split('_', expand = True)[1]
    return df


# In[28]:


get_ipython().run_cell_magic('time', '', 'trn_data = extract_group(trn_data)\ntst_data = extract_group(tst_data)\n')


# ## 6.11 Imputing Using LightGBM

# In[29]:


trn_data.columns


# In[30]:


cols = [
        #'HomePlanet',
        #'CryoSleep',
        #'Destination',
        'Age',
        #'VIP',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        #'AgeGroup',
        #'CabinDeck',
        #'CabinNum',
        #'CabinSide',
        #'FirstName',
        #'FamilyName',
        #'TravelGroup'
       ]

object_cols = [
               #'HomePlanet',
               #'CryoSleep',
               #'Destination',
               #'VIP',
               #'CabinDeck',
               #'CabinNum',
               #'CabinSide',
               #'FirstName',
               #'FamilyName',
               #'TravelGroup'
              ]


# In[31]:


trn_data[cols].isnull().sum()


# In[32]:


trn_data['VIP']


# In[33]:


get_ipython().run_cell_magic('time', '', "object_cols = ['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']\n\n# Create a fill missing values function using the Simple Imputer\ndef ml_imputer(df, cols, object_cols):\n    '''\n    \n    '''\n    tmp = df.copy(deep = True)\n    lgbm_imputer = LGBMImputer(cat_features = object_cols, n_iter = 50)\n    tmp[cols] = lgbm_imputer.fit_transform(tmp[cols])\n    print('...........', '\\n')\n    return tmp\n")


# In[34]:


get_ipython().run_cell_magic('time', '', 'trn_data = ml_imputer(trn_data, cols, object_cols)\ntst_data = ml_imputer(tst_data, cols, object_cols)\n')


# In[35]:


trn_data.isnull().sum()


# ---

# ## 6.4 Filling NaNs by Mode and Mean (Using Groups)

# In[36]:


trn_data[trn_data['TravelGroup'] == '2926']


# In[37]:


people_per_group = trn_data.groupby('TravelGroup')['CabinDeck'].nunique().reset_index()
people_per_group.sort_values('CabinDeck', ascending=False)


# In[38]:


people_per_group.max()


# In[39]:


get_ipython().run_cell_magic('time', '', "def fill_missing(df, group = 'TravelGroup',):\n    '''\n    Fill NaNs values or with mean or most commond value...\n    \n    '''\n    \n    tmp = df.copy()\n    tmp = tpm[tmp[group].ising()]\n    \n    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n    \n    numeric_tmp = df.select_dtypes(include = numerics)\n    categ_tmp = df.select_dtypes(exclude = numerics)\n    \n\n    for col in numeric_tmp.columns:\n        print(col)\n        #df[col] = df[col].fillna(value = df[col].mean())\n        df[col] = df[col].fillna(df.groupby(group)[col].transform('mean'))\n        \n    for col in categ_tmp.columns:\n        print(col)\n        #df[col] = df[col].fillna(value = df[col].mode()[0])\n        #df[col] = df[col].fillna(df.groupby(group)[col].transform(lambda S: S.mode()[0]))\n        df[col] = (df.groupby(group)[col].transform(lambda x: x.value_counts().idxmax()))\n    print('...')\n    \n    return df\n")


# In[40]:


get_ipython().run_cell_magic('time', '', '#trn_data =  fill_missing(trn_data)\n#tst_data =  fill_missing(tst_data)\n')


# ## 6.4 Filling NaNs by Mode and Mean

# In[41]:


get_ipython().run_cell_magic('time', '', "def fill_missing(df):\n    '''\n    Fill NaNs values or with mean or most commond value...\n    \n    '''\n    \n    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']\n    \n    numeric_tmp = df.select_dtypes(include = numerics)\n    categ_tmp = df.select_dtypes(exclude = numerics)\n\n    for col in numeric_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mean())\n        \n    for col in categ_tmp.columns:\n        print(col)\n        df[col] = df[col].fillna(value = df[col].mode()[0])\n        \n    print('...')\n    \n    return df\n")


# In[42]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_missing(trn_data)\ntst_data =  fill_missing(tst_data)\n')


# In[43]:


trn_data.isnull().sum()


# ## 6.5 Calculating Total Expended in the Ship

# In[44]:


get_ipython().run_cell_magic('time', '', "def total_billed(df):\n    '''\n    Calculates total amount billed in the trip to the passenger... \n    Args:\n    Returns:\n    \n    '''\n    \n    df['TotalBilled'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']\n    return df\n")


# In[45]:


get_ipython().run_cell_magic('time', '', 'trn_data = total_billed(trn_data)\ntst_data = total_billed(tst_data)\n')


# ## 6.6 Filling NaNs by Using EDA Insights (TotalBilled)

# In[46]:


get_ipython().run_cell_magic('time', '', "def fill_nans_by_totalspend(df):\n    df['CryoSleep'] = np.where(df['TotalBilled'] > 0, False, df['CryoSleep'])\n    return df\n")


# In[47]:


get_ipython().run_cell_magic('time', '', 'trn_data =  fill_nans_by_totalspend(trn_data)\ntst_data =  fill_nans_by_totalspend(tst_data)\n')


# ---

# ## 6.11 Calculating Aggregated Features, Based on Cabin Deck

# In[48]:


get_ipython().run_cell_magic('time', '', "Weltiest_Deck = trn_data.groupby('CabinDeck').aggregate({'TotalBilled': 'sum', 'Transported': 'sum', 'CryoSleep': 'sum', 'PassengerId': 'size'}).reset_index()\nWeltiest_Deck['AvgSpended'] = Weltiest_Deck['TotalBilled'] / Weltiest_Deck['PassengerId']\nWeltiest_Deck['TransportedPercentage'] = Weltiest_Deck['Transported'] / Weltiest_Deck['PassengerId']\nWeltiest_Deck['CryoSleepPercentage'] = Weltiest_Deck['CryoSleep'] / Weltiest_Deck['PassengerId']\nWeltiest_Deck = Weltiest_Deck.sort_values('AvgSpended', ascending = False)\nWeltiest_Deck.head(10)\n")


# In[49]:


get_ipython().run_cell_magic('time', '', "trn_data = trn_data.merge(Weltiest_Deck[['CabinDeck', 'TransportedPercentage', 'AvgSpended']], how = 'left', on = ['CabinDeck'])\ntst_data = tst_data.merge(Weltiest_Deck[['CabinDeck', 'TransportedPercentage', 'AvgSpended']], how = 'left', on = ['CabinDeck'])\n")


# ---

# ## 6.12 Calulating the Number of Relatives, Using Family Name

# In[50]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_data.groupby('FamilyName')['PassengerId'].count().reset_index()\ntst_relatives = tst_data.groupby('FamilyName')['PassengerId'].count().reset_index()\n")


# In[51]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_relatives.rename(columns = {'PassengerId': 'NumRelatives'})\ntst_relatives = tst_relatives.rename(columns = {'PassengerId': 'NumRelatives'})\n")


# In[52]:


get_ipython().run_cell_magic('time', '', "trn_data = trn_data.merge(trn_relatives[['FamilyName', 'NumRelatives']], how = 'left', on = ['FamilyName'])\ntst_data = tst_data.merge(tst_relatives[['FamilyName', 'NumRelatives']], how = 'left', on = ['FamilyName'])\n")


# ---

# ## 6.13 Calulating the Number of People Traveling Together, Using Traveling Group**

# In[53]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_data.groupby('TravelGroup')['PassengerId'].count().reset_index()\ntst_relatives = tst_data.groupby('TravelGroup')['PassengerId'].count().reset_index()\n")


# In[54]:


get_ipython().run_cell_magic('time', '', "trn_relatives = trn_relatives.rename(columns = {'PassengerId': 'GroupSize'})\ntst_relatives = tst_relatives.rename(columns = {'PassengerId': 'GroupSize'})\n")


# In[55]:


get_ipython().run_cell_magic('time', '', "trn_data = trn_data.merge(trn_relatives[['TravelGroup', 'GroupSize']], how = 'left', on = ['TravelGroup'])\ntst_data = tst_data.merge(tst_relatives[['TravelGroup', 'GroupSize']], how = 'left', on = ['TravelGroup'])\n")


# ---

# # 7.0 Pre-Processing for Training

# ## 7.1 Separating the Fields by Type

# In[56]:


trn_data.columns


# In[57]:


get_ipython().run_cell_magic('time', '', "# A list of the original variables from the dataset\nnumerical_features = [\n                      'Age', \n                      'RoomService', \n                      'FoodCourt', \n                      'ShoppingMall', \n                      'Spa', \n                      'VRDeck', \n                      'TotalBilled',\n                      'TravelGroupPos'\n                     ]\n\ncategorical_features = [\n                        #'Name',\n                        'FirstName',\n                        'FamilyName',\n                        'CabinNum',\n                        'TravelGroup',\n                        'AgeGroup'\n                       ]\n\n\ncategorical_features_onehot = [\n                               'HomePlanet',\n                               'CryoSleep',\n                               #'Cabin',\n                               'CabinDeck',\n                               'CabinSide',\n                               'Destination',\n                               'VIP',\n                               #'AgeGroup'\n                               ]\n\ntarget_feature = 'Transported'\n")


# ---

# ## 7.2 Encoding Categorical Variables

# In[58]:


get_ipython().run_cell_magic('time', '', "\ndef encode_categorical(train_df, test_df, categ_feat = categorical_features):\n    '''\n    \n    '''\n    encoder_dict = {}\n    \n    concat_data = pd.concat([trn_data[categ_feat], tst_data[categ_feat]])\n    \n    for col in concat_data.columns:\n        print('Encoding: ', col, '...')\n        encoder = LabelEncoder()\n        encoder.fit(concat_data[col])\n        encoder_dict[col] = encoder\n\n        train_df[col + '_Enc'] = encoder.transform(train_df[col])\n        test_df[col + '_Enc'] = encoder.transform(test_df[col])\n    \n    train_df = train_df.drop(columns = categ_feat, axis = 1)\n    test_df = test_df.drop(columns = categ_feat, axis = 1)\n\n    return train_df, test_df\n")


# In[59]:


get_ipython().run_cell_magic('time', '', 'trn_data, tst_data = encode_categorical(trn_data, tst_data, categorical_features)\n')


# ---

# ## 7.3 One Hot Encoding Categorical Variables

# In[60]:


get_ipython().run_cell_magic('time', '', 'def one_hot(df, one_hot_categ):\n    for col in one_hot_categ:\n        tmp = pd.get_dummies(df[col], prefix = col)\n        df = pd.concat([df, tmp], axis = 1)\n    df = df.drop(columns = one_hot_categ)\n    return df\n')


# In[61]:


get_ipython().run_cell_magic('time', '', 'trn_data = one_hot(trn_data, categorical_features_onehot) \ntst_data = one_hot(tst_data, categorical_features_onehot) \n')


# ---

# # 8.0 Feature Selection for Baseline Model

# In[62]:


get_ipython().run_cell_magic('time', '', "remove = ['PassengerId', \n          'Route', \n          'FirstName_Enc', \n          #'CabinNum_Enc', \n          'Transported',\n          'Cabin',\n          'TransportedPercentage',\n          #'IsKid', \n          #'IsAdult', \n          #'IsOlder'\n          #'RoomService',\n          #'FoodCourt',\n          #'ShoppingMall',\n          #'Spa',\n          #'VRDeck',\n         ]\nfeatures = [feat for feat in trn_data.columns if feat not in remove]\n")


# In[63]:


get_ipython().run_cell_magic('time', '', 'features\n')


# ---

# # 9.0 Advance Feature Engineering, KNN

# In[64]:


get_ipython().run_cell_magic('time', '', "# Convert X and y to Numpy arrays as library requirements\nX_array = trn_data[features].to_numpy()\ny_array = trn_data['Transported'].to_numpy()\nX_test_array = tst_data[features].to_numpy()\n")


# In[65]:


K = 2


# In[66]:


get_ipython().run_cell_magic('time', '', "# It Takes almost  35min 21s for K = 2 and 50_000 rows...\n# It Takes almost  17min 36s for K = 1 and 50_000 rows...\nKNN_trn_features = knn_kfold_extract(X_array, y_array, k = K, normalize = 'standard')\n")


# In[67]:


get_ipython().run_cell_magic('time', '', 'KNN_trn_features\n')


# In[68]:


get_ipython().run_cell_magic('time', '', "knn_cols = ['KNN_K1_01',\n            'KNN_K1_02',\n            'KNN_K2_01',\n            'KNN_K2_02']\n\nKNN_feat = pd.DataFrame(KNN_trn_features, columns = knn_cols)\nKNN_feat = pd.DataFrame(KNN_trn_features, columns = knn_cols).set_index(trn_data.index)\n")


# In[69]:


get_ipython().run_cell_magic('time', '', 'trn_data = pd.concat([trn_data, KNN_feat], axis = 1)\ntrn_data.head()\n')


# In[70]:


get_ipython().run_cell_magic('time', '', "KNN_tst_features = knn_extract(X_array, y_array, X_test_array, k = K, normalize = 'standard')\nKNN_feat = pd.DataFrame(KNN_tst_features, columns = knn_cols).set_index(tst_data.index)\n\ntst_data = pd.concat([tst_data, KNN_feat], axis = 1)\ntst_data.head()\n")


# ---

# # 10.0 Selection of Features for Training Stage

# In[71]:


get_ipython().run_cell_magic('time', '', "remove = ['PassengerId', \n          'Route', \n          'FirstName_Enc', \n          'CabinNum_Enc', \n          'Transported',\n          'Cabin',\n          'TransportedPercentage',\n          #'IsKid', \n          #'IsAdult', \n          #'IsOlder'\n          #'RoomService',\n          #'FoodCourt',\n          #'ShoppingMall',\n          #'Spa',\n          #'VRDeck',\n          'KNN_K2_02',\n          'KNN_K2_01',\n         ]\nfeatures = [feat for feat in trn_data.columns if feat not in remove]\n")


# In[72]:


get_ipython().run_cell_magic('time', '', 'features\n')


# In[73]:


features = ['Age',
            'RoomService',
            'FoodCourt',
            'ShoppingMall',
            'Spa',
            'VRDeck',
            'TravelGroupPos',
            'TotalBilled',
            #'AvgSpended',
            #'NumRelatives',
            #'GroupSize',
            'FamilyName_Enc',
            'TravelGroup_Enc',
            #'AgeGroup_Enc',
            'HomePlanet_Earth',
            'HomePlanet_Europa',
            'HomePlanet_Mars',
            'CryoSleep_False',
            'CryoSleep_True',
            'CabinDeck_A',
            'CabinDeck_B',
            'CabinDeck_C',
            'CabinDeck_D',
            'CabinDeck_E',
            'CabinDeck_F',
            'CabinDeck_G',
            'CabinDeck_T',
            'CabinSide_P',
            'CabinSide_S',
            'Destination_55 Cancri e',
            'Destination_PSO J318.5-22',
            'Destination_TRAPPIST-1e',
            'VIP_False',
            'VIP_True',
            #'AgeGroup_0',
            #'AgeGroup_1',
            #'AgeGroup_0',
            #'AgeGroup_1',
            'KNN_K1_01',
            #'KNN_K1_02'
]


# ---

# # 11.0 Building a Neuronal Network Model...

# ## 11.1 Importing all the required libraries...

# In[74]:


get_ipython().run_cell_magic('time', '', 'import tensorflow as tf\nfrom tensorflow.keras.models import Model\nfrom tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping\nfrom tensorflow.keras.layers import Dense, Input, InputLayer, Add, BatchNormalization, Dropout, Concatenate\n\nfrom sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler\nimport random\n\nfrom sklearn.model_selection import KFold, StratifiedKFold \nfrom sklearn.metrics import roc_auc_score, roc_curve, accuracy_score\nimport datetime\nimport math\n')


# ---

# ## 11.2 Defining the Model Architecture...

# In[75]:


get_ipython().run_cell_magic('time', '', "def nn_model_one():\n    '''\n    '''\n    \n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x = Dense(1024, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(inputs)\n    \n    x = BatchNormalization()(x)\n    \n    x = Dense(256, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(128, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n\n    x = Dense(1 , \n              #use_bias  = True, \n              #kernel_regularizer = tf.keras.regularizers.l2(30e-6),\n              activation = 'sigmoid')(x)\n    \n    model = Model(inputs, x)\n    \n    return model\n")


# In[76]:


get_ipython().run_cell_magic('time', '', "def nn_model_two():\n    '''\n    '''\n\n    dropout_value = 0.025\n    \n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x = Dense(1024, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(inputs)\n    \n    x = BatchNormalization()(x)\n    x = Dropout(dropout_value)(x)\n    \n    x = Dense(256, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    x = Dropout(dropout_value)(x)\n\n    x = Dense(128, \n              #use_bias  = True, \n              kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n              activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    x = Dropout(dropout_value)(x)\n    \n    x = Dense(8, \n          #use_bias  = True, \n          kernel_regularizer = tf.keras.regularizers.l2(30e-6), \n          activation = activation_func)(x)\n    \n    x = BatchNormalization()(x)\n    x = Dropout(dropout_value)(x)\n    \n    x = Dense(1 , \n              #use_bias  = True, \n              #kernel_regularizer = tf.keras.regularizers.l2(30e-6),\n              activation = 'sigmoid')(x)\n    \n    model = Model(inputs, x)\n    \n    return model\n")


# In[77]:


get_ipython().run_cell_magic('time', '', "def nn_model_three():\n    \n    '''\n    Function to define the Neuronal Network architecture...\n    '''\n    \n    L2 = 65e-6\n    dropout_value = 0.025\n    activation_func = 'swish'\n    inputs = Input(shape = (len(features)))\n    \n    x0 = Dense(1024, kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(inputs)\n    x0 = BatchNormalization()(x0)\n    x0 = Dropout(dropout_value)(x0)\n    \n    x1 = Dense(1024,  kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x0)\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(64,  kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x1)\n    x1 = Concatenate()([x1, x0])\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(16, kernel_regularizer = tf.keras.regularizers.l2(L2), activation = activation_func)(x1)\n    x1 = BatchNormalization()(x1)\n    x1 = Dropout(dropout_value)(x1)\n    \n    x1 = Dense(1,  \n               #kernel_regularizer = tf.keras.regularizers.l2(4e-4), \n               activation = 'sigmoid')(x1)\n    \n    model = Model(inputs, x1)\n    \n    return model\n")


# ---

# ## 11.3 Visualizing the Architecture Created...

# In[78]:


get_ipython().run_cell_magic('time', '', 'architecture = nn_model_one()\narchitecture.summary()\n')


# ---

# ## 11.4 Defining Model Parameters for Training...

# In[79]:


get_ipython().run_cell_magic('time', '', "# Defining model parameters...\nBATCH_SIZE         = 128\nEPOCHS             = 250 \nEPOCHS_COSINEDECAY = 250\nDIAGRAMS           = True\nUSE_PLATEAU        = True\nINFERENCE          = False\nVERBOSE            = 0 \nTARGET             = 'Transported'\n")


# ---

# ## 11.5 Defining Training Functions for the NN Model

# In[80]:


get_ipython().run_cell_magic('time', '', '# Defining model training function...\ndef fit_model(X_train, y_train, X_val, y_val, run = 0):\n   \'\'\'\n   \'\'\'\n   lr_start = 0.2\n   start_time = datetime.datetime.now()\n   \n   #scaler = StandardScaler()\n   #scaler = RobustScaler()\n   scaler = MinMaxScaler()\n   X_train = scaler.fit_transform(X_train)\n\n   epochs = EPOCHS    \n   lr = ReduceLROnPlateau(monitor = \'val_loss\', factor = 0.1, patience = 8, verbose = VERBOSE)\n   es = EarlyStopping(monitor = \'val_loss\',patience = 16, verbose = 1, mode = \'min\', restore_best_weights = True)\n   tm = tf.keras.callbacks.TerminateOnNaN()\n   callbacks = [lr, es, tm]\n   \n   # Cosine Learning Rate Decay\n   if USE_PLATEAU == False:\n       epochs = EPOCHS_COSINEDECAY\n       lr_end = 0.0002\n\n       def cosine_decay(epoch):\n           if epochs > 1:\n               w = (1 + math.cos(epoch / (epochs - 1) * math.pi)) / 2\n           else:\n               w = 1\n           return w * lr_start + (1 - w) * lr_end\n       \n       lr = LearningRateScheduler(cosine_decay, verbose = 0)\n       callbacks = [lr, tm]\n       \n   model = nn_model_one()\n   \n   optimizer_func = tf.keras.optimizers.Adam(learning_rate = lr_start)\n   loss_func = tf.keras.losses.BinaryCrossentropy()\n   model.compile(optimizer = optimizer_func, loss = loss_func)\n   \n   X_val = scaler.transform(X_val)\n   validation_data = (X_val, y_val)\n   \n   history = model.fit(X_train, \n                       y_train, \n                       validation_data = validation_data, \n                       epochs          = epochs,\n                       verbose         = VERBOSE,\n                       batch_size      = BATCH_SIZE,\n                       shuffle         = True,\n                       callbacks       = callbacks\n                      )\n   \n   history_list.append(history.history)\n   print(f\'Training loss:{history_list[-1]["loss"][-1]:.3f}\')\n   callbacks, es, lr, tm, history = None, None, None, None, None\n   \n   \n   y_val_pred = model.predict(X_val, batch_size = BATCH_SIZE, verbose = VERBOSE)\n   y_val_pred = [1 if x > 0.5 else 0 for x in y_val_pred]\n   \n   score = accuracy_score(y_val, y_val_pred)\n   print(f\'Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}\'\n         f\'| ACC: {score:.5f}\')\n   \n   score_list.append(score)\n   \n   tst_data_scaled = scaler.transform(tst_data[features])\n   tst_pred = model.predict(tst_data_scaled)\n   predictions.append(tst_pred)\n   \n   return model\n')


# ---

# # Training a XGBoost Classifier ...

# # 12.0 Training the NN Model, Using a CV Loop

# In[81]:


get_ipython().run_cell_magic('time', '', "# Create empty lists to store NN information...\nhistory_list = []\nscore_list   = []\npredictions  = []\n\n# Define kfolds for training purposes...\n#kf = KFold(n_splits = 5)\nkf = StratifiedKFold(n_splits = 5, random_state = 15, shuffle = True)\n\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data, trn_data[TARGET])):\n    X_train, X_val = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_val = trn_data.iloc[trn_idx][TARGET], trn_data.iloc[val_idx][TARGET]\n    \n    fit_model(X_train, y_train, X_val, y_val)\n    \nprint(f'OOF AUC: {np.mean(score_list):.5f}')\n")


# In[82]:


# OOF AUC: 0.80375
# OOF AUC: 0.80433
# OOF AUC: 0.80536
# OOF AUC: 0.80778
# OOF AUC: 0.80812 ***
# OOF AUC: 0.79420
# OOF AUC: 0.79133

# OOF AUC: 0.79731
# OOF AUC: 0.79708

# OOF AUC: 0.79973 ***

# OOF AUC: 0.79777
# OOF AUC: 0.80387 ***
# OOF AUC: 0.80191 
# OOF AUC: 0.80364 ...

# OOF AUC: 0.80755 >>> 0.81061
# OOF AUC: 0.80421 >>> 0.80664
# OOF AUC: 0.80778 >>> 0.80664
# OOF AUC: 0.80904 >>>


# ---

# # 13.0 Generating Predictions

# In[83]:


get_ipython().run_cell_magic('time', '', "# Populated the prediction on the submission dataset and creates an output file\nsub['Transported'] = np.array(predictions).mean(axis = 0)\nsub['Transported'] = np.where(sub['Transported'] > 0.5, True, False)\nsub.to_csv('my_submission_051922.csv', index = False)\n")


# In[84]:


get_ipython().run_cell_magic('time', '', 'sub.head()\n')


# ---

# In[ ]:





# In[ ]:





# In[ ]:




