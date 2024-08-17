#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pycaret')


# In[2]:


import pandas as pd
from pycaret.classification import *


# # Feature Engineering
# 
# - Identify groups of passengers based on first 4 digits of `PassengerId`
# - Create group level features
# - Merge group level features with passenger level features

# In[3]:


def create_group_features(df):
    
    '''
    Group level features
    - Number of passengers
    - Number of VIPs passengers
    - Number of passengers in cryosleep
    - Number of unique cabins
    - Number of unique decks
    - Number of unique sides
    - Mean age of passengers in the group
    - mean spend on various expense area
    - mean total spend
    - Number of unique home planets
    
    '''
    
    df = (df.groupby('PassengerGroup', as_index = False)
          .agg({'PassengerNo':'nunique',
                'VIP':lambda x: sum(x == True),
                'CryoSleep': lambda x: sum(x == True),
                'Cabin': 'nunique',
                'Deck': 'nunique',
                'Side': 'nunique',
                'Age': 'mean',
                'RoomService': 'mean',
                'FoodCourt': 'mean',
                'ShoppingMall':'mean',
                'Spa':'mean',
                'VRDeck': 'mean',
                'TotalSpend':'mean',
                'HomePlanet': 'nunique'})
          .rename(columns = {'PassengerNo':'Count'})
         )
    
    df['PctRoomService'] = df['RoomService']/df['TotalSpend']
    df['PctFoodCourt'] = df['FoodCourt']/df['TotalSpend']
    df['PctShoppingMall'] = df['ShoppingMall']/df['TotalSpend']
    df['PctSpa'] = df['Spa']/df['TotalSpend']
    df['PctVRDeck'] = df['VRDeck']/df['TotalSpend']
    
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    
    df.columns = [f'Group{i}' if i not in ['PassengerGroup'] else i for i in df.columns]
    
    
    
    return df


def create_features(df):
    
    bool_type = ['VIP', 'CryoSleep']
    df[bool_type] = df[bool_type].astype(bool)
    
    df['PassengerGroup'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['PassengerNo'] = df['PassengerId'].apply(lambda x: x.split('_')[1])
    df.loc[df['Cabin'].isnull(), 'Cabin'] = 'None/None/None'
    
    fill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['PctRoomService'] = df['RoomService']/df['TotalSpend']
    df['PctFoodCourt'] = df['FoodCourt']/df['TotalSpend']
    df['PctShoppingMall'] = df['ShoppingMall']/df['TotalSpend']
    df['PctSpa'] = df['Spa']/df['TotalSpend']
    df['PctVRDeck'] = df['VRDeck']/df['TotalSpend']
    fill_cols = ['PctRoomService', 'PctFoodCourt', 'PctShoppingMall', 'PctSpa', 'PctVRDeck']
    df[fill_cols] = df[fill_cols].fillna(0)
    
    df['Age'] = df['Age'].fillna(df.groupby('HomePlanet')['Age'].transform('median'))
    df['CryoSleep'] = df['CryoSleep'].fillna(False)
    
    df['Deck'] = df['Cabin'].apply(lambda x: str(x).split('/')[0])
    df['Side'] = df['Cabin'].apply(lambda x: str(x).split('/')[2])
    
    df_group_features = create_group_features(df)    
    
    df = pd.merge(df, df_group_features, on = 'PassengerGroup', how = 'left')
    
    return df


# In[4]:


df_train = pd.read_csv('../input/spaceship-titanic/train.csv')
df_test = pd.read_csv('../input/spaceship-titanic/test.csv')


# In[5]:


train = create_features(df_train)
test = create_features(df_test)


# # Model
# 
# - Pycaret, Catboost

# In[6]:


num_cols = list(train.select_dtypes('float64').columns) + list(train.select_dtypes('int64').columns) 

s = setup(data = train,
          target = 'Transported',
          train_size = 0.99,
          fold_strategy = 'stratifiedkfold',
          fold = 5,
          fold_shuffle = True,
          numeric_features = num_cols,
          ignore_low_variance=True,
          remove_multicollinearity = True,
          normalize = True,
          normalize_method = 'robust',
          data_split_stratify = True,
          
          ignore_features = ['PassengerNo', 'Name', 'PassengerId', 'PassengerGroup', 'Cabin'],
          silent = True)


remove_metric('kappa')
remove_metric('mcc')


# In[7]:


best = compare_models(n_select = 4, include = ['catboost', 'lightgbm'])


# In[8]:


catboost = tune_model(create_model('catboost'), choose_better = True, n_iter = 20)


# In[9]:


df_pred = predict_model(catboost, test)
df_sub = df_pred.loc[:, ['PassengerId', 'Label']].rename(columns = {'Label':'Transported'})
df_sub.to_csv('submission.csv', index = False)

