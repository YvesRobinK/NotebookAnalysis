#!/usr/bin/env python
# coding: utf-8

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


# ![](https://i0.wp.com/www.compoundchem.com/wp-content/uploads/2022/11/Mohs-Hardness-Scale.png?ssl=1)

# # <span style="color: ForestGreen">Table of Contents</span>
# 
# 1. [Import libraries](#1)
# 2. [Open data files](#2)
# 3. [Show first 5 lines of train data](#3)
#     - 3.1 [Compare original train and competition train](#3.1)
#     - 3.2 [Concatenate original train with competition train](#3.2)
#     - 3.3 [3.3 Feature Engineering](#3.3)
# 4. [Shapes of train and test data](#4)
# 5. [Display descriptive statistics of train data](#5)
# 6. [Check the number of gaps for each feature](#6)
# 7. [Data types of training set](#7)
#     - 7.1 [Figure out how much duplicates in data](#7.1)
#     - 7.2 [Remove duplicates from train data](#7.2)
# 8. [Display histograms of distribution](#8)
# 9. [Let's count target of train data](#9)
# 10. [Transorm the data with logarithm](#10)
#     - 10.1 [Pie plot of smoking](#10.1)
#     - 10.2 [Feature importance](#10.2)
# 11. [Build a heat map of correlations](#11)
# 12. [Define base models](#12)
# 13. [Defining the meta-model](#13)
# 14. [Creating and fitting the stacking model](#14)
# 15. [Predict the validation set and calculate Median Absolute Error score](#15)
# 16. [Predict on the test data](#16)
# 17. [Build DataFrame and make first submission](#17)
# 

# <font face="Bahnschrift Condensed" style="font-size: 14pt;">Hardness, or the quantitative value of resistance to permanent or plastic deformation, plays a very crucial role in materials design in many applications, such as ceramic coatings and abrasives. Hardness testing is an especially useful method as it is non-destructive and simple to implement to gauge the plastic properties of a material. In this study, I proposed a machine, or statistical, learning approach to predict hardness in naturally occurring materials, which integrates atomic and electronic features from composition directly across a wide variety of mineral compositions and crystal systems. First, atomic and electronic features from the composition, such as van der Waals and covalent radii as well as the number of valence electrons, were extracted from the composition.
# </font>
# <br><br>
# <font face="Bahnschrift Condensed" style="font-size: 14pt;">In this study, the author trained a set of classifiers to understand whether compositional features can be used to predict the Mohs hardness of minerals of different chemical spaces, crystal structures, and crystal classes. The dataset for training and testing the classification models used in this study originated from experimental Mohs hardness data, their crystal classes, and chemical compositions of naturally occurring minerals reported in the Physical and Optical Properties of Minerals CRC Handbook of Chemistry and Physics and the American Mineralogist Crystal Structure Database. The database is composed of 369 uniquely named minerals. Due to the presence of multiple composition combinations for minerals referred to by the same name, the first step was to perform compositional permutations on these minerals. This produced a database of 622 minerals of unique compositions, comprising 210 monoclinic, 96 rhombohedral, 89 hexagonal, 80 tetragonal, 73 cubic, 50 orthorhombic, 22 triclinic, 1 trigonal, and 1 amorphous structure. An independent dataset was compiled to validate the model performance. The validation dataset contains the composition, crystal structure, and Mohs hardness values of 51 synthetic single crystals reported in the literature. The validation dataset includes 15 monoclinic, 7 tetragonal, 7 hexagonal, 6 orthorhombic, 4 cubic, and 3 rhombohedral crystal structures.
# </font>
# <br><br>
# <font face="Bahnschrift Condensed" style="font-size: 14pt;">In this study, the author constructed a database of compositional feature descriptors that characterize naturally occurring materials obtained directly from the Physical and Optical Properties of Minerals CRC Handbook45. This comprehensive compositional-based dataset allows us to train models that are able to predict hardness across a wide variety of mineral compositions and crystal classes. Each material in both the naturally occurring mineral and artificial single crystal datasets was represented by 11 atomic descriptors. The elemental features are the number of electrons, number of valence electrons, atomic number, Pauling electronegativity of the most common oxidation state, covalent atomic radii, van der Waals radii, and ionization energy of neutral.
# 

# ![](https://i0.wp.com/www.trigonalsystem.com/wp-content/uploads/2021/05/07.jpg?resize=1024%2C606&ssl=1)

# <a id='1'></a>
# # <span style="color: brown">1 - Import libraries</span>

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import median_absolute_error


# <a id='2'></a>
# # <span style="color: brown">2 - Open data files</span>

# In[3]:


original_train = pd.read_csv('/kaggle/input/prediction-of-mohs-hardness-with-machine-learning/jm79zfps6b-1/Mineral_Dataset_Supplementary_Info.csv')

train = pd.read_csv('/kaggle/input/playground-series-s3e25/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e25/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s3e25/sample_submission.csv')


# <a id='3'></a>
# # <span style="color: brown">3 - Show first 5 lines of train data</span>

# In[4]:


# Show all properties on display
pd.set_option('display.max_columns', None)

train.head()


# <a id='3.1'></a>
# ### <span style="color: brown">3.1 Compare original train and competition train</span>

# In[5]:


print(original_train.shape)
original_train.columns.tolist()


# In[6]:


print(train.shape)
train.columns.tolist()


# <a id='3.2'></a>
# ### <span style="color: brown">3.2 Concatenate original train with competition train</span>

# In[7]:


train = pd.concat(objs=[train, original_train]).reset_index(drop=True)
train.shape


# In[8]:


train.columns.tolist()


# In[9]:


train = train.drop(['id', 'Unnamed: 0'], axis=1)
test.drop(columns='id', axis=1, inplace=True)
train.columns.tolist()


# <a id='3.3'></a>
# ### <span style="color: brown">3.3 Feature Engineering</span>

# In[10]:


# Electronegativity Difference
train['n_electronegativity_diff'] = train['el_neg_chi_Average'] - train['atomicweight_Average']
train['n_electronegativity_diff'].replace([np.inf, -np.inf], train['n_electronegativity_diff'].max(), inplace=True)
train['n_electronegativity_diff'].fillna(train['n_electronegativity_diff'].median(), inplace=True)

test['n_electronegativity_diff'] = test['el_neg_chi_Average'] - test['atomicweight_Average']
test['n_electronegativity_diff'].replace([np.inf, -np.inf], test['n_electronegativity_diff'].max(), inplace=True)
test['n_electronegativity_diff'].fillna(test['n_electronegativity_diff'].median(), inplace=True)

# Density Difference
train['Density Difference'] = train['density_Total'] - train['density_Average']
train['Density Difference'].replace([np.inf, -np.inf], train['Density Difference'].max(), inplace=True)
train['Density Difference'].fillna(train['Density Difference'].median(), inplace=True)

test['Density Difference'] = test['density_Total'] - test['density_Average']
test['Density Difference'].replace([np.inf, -np.inf], test['Density Difference'].max(), inplace=True)
test['Density Difference'].fillna(test['Density Difference'].median(), inplace=True)

# Последняя проверка на NaN (можно убрать, если не нужно)
print(train.isnull().sum(), end='\n\n\n')
print(test.isnull().sum())

train.describe()


# <a id='4'></a>
# # <span style="color: brown">4 - Shapes of train and test data</span>

# In[11]:


print(f'Train data: {train.shape}')
print(f'Test data: {test.shape}\n')

train_data_percentage = np.round(train.shape[0] / (train.shape[0] + test.shape[0]), 4)
print(f'Train data consists of {train_data_percentage * 100}% of all observations')
print(f'Test data consists of {(1 - train_data_percentage) * 100}% of all observations')


# <a id='5'></a>
# # <span style="color: brown">5 - Display descriptive statistics of train data</span>

# In[12]:


train.describe().T


# #### <span style="color: ForestGreen">allelectrons_Total, density_Total features here contain several values ​​that are much greater than the 3rd quantile</span>

# <a id='6'></a>
# # <span style="color: brown">6 - Check the number of gaps for each feature</span>

# ![](https://civileblog.com/wp-content/uploads/2016/06/rock-test.jpg)

# In[13]:


print('TRAIN data\n')
print(f'{train.isna().sum()}\n\n\n')

print('TEST data\n')
print(train.isna().sum())


# #### <span style="color: ForestGreen">As we can see, there are no preliminary gaps in the data. However, sometimes it is useful to check the unique values ​​for each characteristic. After all, the gaps could be filled with a '?' and then the .isna() method will not notice them</span>

# <a id='7'></a>
# # <span style="color: brown">7 - Data types of training set</span>

# In[14]:


train.info()


# #### <span style="color: ForestGreen">As You can see, all data types are numeric, so if there are gaps, they were most likely filled with some kind of numeric values ​​like 0, median or average value</span>

# <a id="7.1"></a>
# ### <span style="color: brown">7.1 Figure out how much duplicates in data</span>

# In[15]:


train_duplicates_number = train[train.duplicated()]
                             
print(len(train_duplicates_number))


# #### <span style="color: ForestGreen">There are 23 duplicates in train data. Let's remove them!</span>

# <a id="7.2"></a>
# ### <span style="color: brown">7.2 Remove duplicates from train data</span>

# In[16]:


train = train.drop_duplicates()

# Check whether all duplicates were removed
duplicates = train[train.duplicated()]
len(duplicates)


# <a id='8'></a>
# # <span style="color: brown">8 - Display histograms of distribution</span>

# In[17]:


sns.set(rc={'figure.figsize': (20, 16)})
train.hist(color='Lime');


# <a id='9'></a>
# # <span style="color: brown">9 - Let's count target of train data</span>

# In[18]:


print(f'{train.Hardness.value_counts()}\n\n')
print(train.Hardness.value_counts() / train.shape[0])


# <a id='10'></a>
# # <span style="color: brown">10 - Transorm the data with MinMax</span>

# In[19]:


# Split the train data into X and y
X = train.drop(['Hardness'], axis=1)
y = train.Hardness

for column in X.columns.tolist():
    X[column] = X[column].apply(lambda x: (x - X[column].min()) / (X[column].max() - X[column].min()))

# Transform test data
for column in test.columns.tolist():
    test[column] = test[column].apply(lambda x: (x - test[column].min()) / (test[column].max() - test[column].min()))
    
X.hist(color='LightSeaGreen');


# <a id='10.1'></a>
# ### <span style="color: brown">10.1 - Pie plot of Hardness</span>

# In[20]:


label_counts = y.value_counts()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 15))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Pie plot of Hardness')
plt.show();


# <a id='10.2'></a>
# ### <span style="color: brown">10.2 Feature importance</span>

# In[21]:


get_ipython().run_cell_magic('time', '', "# I figured out best hyperparameters previously\nbest_forest = RandomForestRegressor(\n    random_state=26,\n)\n    \nbest_forest.fit(X, y)\nimportance = best_forest.feature_importances_\n\nfeature_importance = pd.DataFrame(data=importance, index=X.columns, columns=['importance']) \\\n    .sort_values(ascending=True, by='importance')\n\nfeature_importance.plot(kind='barh', figsize=(12, 8), color='SteelBlue');\n")


# <a id='11'></a>
# # <span style="color: brown">11 - Build a heat map of correlations</span>

# In[22]:


correlation = X.corr()
correlation.style.background_gradient(cmap='coolwarm')


# #### <span style="color: ForestGreen">There is huge correlation between 'allelectrons_Average' and 'atomicweight_Average'. Maybe I have to drop second to avoid multicorrelation problem (second's feature importance is lower than first's)</span>

# In[23]:


X = X.drop('atomicweight_Average', axis=1)
test = test.drop('atomicweight_Average', axis=1)


# <a id='12'></a>
# # <span style="color: brown">12 - Define base models</span>

# In[24]:


# Split data into train and val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=27)


# ### 12.1 PyTorch model with 3 layers

# In[25]:


# model = torch.nn.Sequential(
#     OrderedDict([
#         ('Linear_layer_1', torch.nn.Linear(in_features=10, out_features=16)),
#         ('ReLU_activation_1', torch.nn.ReLU()),
#         ('Linear_layer_2', torch.nn.Linear(in_features=16, out_features=32)),
#         ('ReLU_activation_2', torch.nn.ReLU()),
#         ('Linear_layer_3', torch.nn.Linear(in_features=32, out_features=64)),
#         ('ReLU_activation_3', torch.nn.ReLU()),
#         ('Linear_layer_4', torch.nn.Linear(in_features=64, out_features=1)),
#     ])
# )

# model


# In[26]:


# X_np = X.values
# y_np = y.values

# # Convert NumPy arrays to PyTorch tensors
# X = torch.tensor(X_np, dtype=torch.float32, requires_grad=True)
# y = torch.tensor(y_np, dtype=torch.float32, requires_grad=True)


# In[27]:


# import torch
# import torch.nn as nn

# class MedianAbsoluteError(nn.Module):
#     def __init__(self):
#         super(MedianAbsoluteError, self).__init__()

#     def forward(self, y_true, y_pred):
#         errors = torch.abs(y_true - y_pred)
#         median_error = torch.median(errors)
        
#         return median_error

    
# loss_fn = MedianAbsoluteError()


# In[28]:


# %%time
# from IPython.display import clear_output

# num_epochs = 1000

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.001
# )

# losses = []

# for epoch in range(1, num_epochs + 1):
#     optimizer.zero_grad()
    
#     pred = model(X)
#     loss = loss_fn(pred, y)
#     losses.append(loss.item())
    
#     loss.backward()
#     optimizer.step()
    
#     losses.append(loss.item())
    
#     if epoch % 100 == 0:
#         clear_output(True)
#         fig, ax = plt.subplots(figsize=(30, 10))
#         plt.title("Error Plot")
#         plt.plot(losses, ".-")
#         plt.xlabel("Training Iteration")
#         plt.ylabel("Error Value")
#         plt.yscale("log")
#         plt.grid()
#         plt.show()


# In[29]:


# test = torch.tensor(test.values, dtype=torch.float32)

# with torch.no_grad():
#     y_pred_test = model(test)[:, 0]
#     print(y_pred_test)


# ### 12.2 Tree-based models

# In[30]:


# Searching for best parameters of XGBoost

# xgb_regressor = XGBRegressor(random_state=27)

# xgb_parameters = {
#     'n_estimators': range(5, 1001, 10),
#     'learning_rate': [0.001, 0.05, 0.01],
#     'max_depth': range(2, 100, 4),
# }

# xgb_random_search = RandomizedSearchCV(estimator=xgb_regressor, param_distributions=xgb_parameters, n_iter=20, n_jobs=-1, cv=5, verbose=4)
# xgb_random_search.fit(X_val, y_val)
# print(f'Best params: {xgb_random_search.best_params_}')

# {'n_estimators': 215, 'max_depth': 26, 'learning_rate': 0.01}

# {'n_estimators': 245, 'max_depth': 42, 'learning_rate': 0.01}


# In[31]:


# Searching for best parameters of CatBoost
# catboost_regressor = CatBoostRegressor(random_state=27)

# catboost_parameters = {
#     'iterations': range(5, 1001, 10),
#     'learning_rate': [0.001, 0.05, 0.1],
#     'depth': range(2, 16, 2),
# }

# catboost_random_search = RandomizedSearchCV(estimator=catboost_regressor, param_distributions=catboost_parameters, n_jobs=-1, cv=5, verbose=4)
# catboost_random_search.fit(X_val, y_val)
# print(catboost_random_search.best_params_)

# {'learning_rate': 0.1, 'iterations': 345, 'depth': 8}

# {'learning_rate': 0.05, 'iterations': 335, 'depth': 8}


# In[32]:


# Searching for best parameters of LightGBM

# lgbm_regressor = LGBMRegressor(random_state=27)

# lgbm_parameters = {
#     'n_estimators': range(5, 1001, 10),
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': range(2, 100, 2),
# }

# lgbm_random_search = RandomizedSearchCV(estimator=lgbm_regressor, param_distributions=lgbm_parameters, n_jobs=-1, cv=5, verbose=4)
# lgbm_random_search.fit(X_val, y_val)
# print(lgbm_random_search.best_params_)

# {'n_estimators': 25, 'max_depth': 20, 'learning_rate': 0.1}

# {'n_estimators': 445, 'max_depth': 10, 'learning_rate': 0.01}


# In[33]:


# Searching for best parameters of RandomForestRegressor

# random_regressor = RandomForestRegressor(random_state=27)

# param_dist = {
#     'n_estimators': range(5, 1001, 10),
#     'max_features': ['sqrt', 'log2'],
#     'max_depth': [int(x) for x in np.linspace(10, 110, num=21)],
#     'min_samples_split': range(2, 101, 2),
#     'min_samples_leaf': range(2, 101, 2),
#     'bootstrap': [True, False]
# }

# random_search = RandomizedSearchCV(estimator=random_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# random_search.fit(X_val, y_val)
# print(random_search.best_params_)

# {'n_estimators': 135,
#  'min_samples_split': 4,
#  'min_samples_leaf': 6,
#  'max_features': 'log2',
#  'max_depth': 85,
#  'bootstrap': False}

# {'n_estimators': 855, 'min_samples_split': 32, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 75, 'bootstrap': True}


# In[34]:


# Searching for best parameters of HistGradientBoostingRegressor

# HGB_regressor = HistGradientBoostingRegressor(random_state=27)

# param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_iter': range(25, 1001, 25),
#     'max_depth': range(3, 100, 2),
#     'min_samples_leaf': range(2, 101, 2),
# }

# random_search = RandomizedSearchCV(estimator=HGB_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# random_search.fit(X_val, y_val)
# best_params = random_search.best_params_
# print(best_params)

# {'min_samples_leaf': 88,
#  'max_iter': 900,
#  'max_depth': 47,
#  'learning_rate': 0.01}


# In[35]:


# Searching for best parameters of ExtraTreesRegressor

# ETR_regressor = ExtraTreesRegressor(random_state=27)

# param_dist = {
#     'n_estimators': range(5, 1001, 10),
#     'max_depth': range(2, 100, 2),
#     'min_samples_leaf': range(2, 101, 2),
# }

# ETR = RandomizedSearchCV(estimator=ETR_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# ETR.fit(X_val, y_val)
# best_params = ETR.best_params_
# print(best_params)

# {'n_estimators': 235, 'min_samples_leaf': 6, 'max_depth': 86}

# {'n_estimators': 285, 'min_samples_leaf': 4, 'max_depth': 14}


# In[36]:


# Searching for best parameters of AdaBoostRegressor

# AdaBoost_regressor = AdaBoostRegressor(random_state=27)

# param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': range(5, 1001, 10),
# }

# AdaBoost = RandomizedSearchCV(estimator=AdaBoost_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# AdaBoost.fit(X_val, y_val)
# best_params = AdaBoost.best_params_
# print(best_params)

# {'n_estimators': 325, 'learning_rate': 0.01}

# {'n_estimators': 295, 'learning_rate': 0.01}


# In[37]:


# Searching for best parameters of BaggingRegressor

# Bagging_regressor = BaggingRegressor(random_state=27)

# param_dist = {
#     'n_estimators': range(5, 1001, 10)
# }

# bagging = RandomizedSearchCV(estimator=Bagging_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# bagging.fit(X_val, y_val)
# best_params = bagging.best_params_
# print(best_params)

# {'n_estimators': 275}


# In[38]:


# Searching for best parameters of GradientBoostingRegressor

# GB_regressor = GradientBoostingRegressor(random_state=27)

# param_dist = {
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': range(25, 1001, 25),
#     'max_depth': range(2, 100, 2),
#     'min_samples_leaf': range(2, 100, 2)
# }

# gb = RandomizedSearchCV(estimator=GB_regressor, param_distributions=param_dist, n_jobs=-1, cv=5, verbose=4)
# gb.fit(X_val, y_val)
# best_params = gb.best_params_
# print(best_params)

# {'n_estimators': 925, 'min_samples_leaf': 26, 'max_depth': 8, 'learning_rate': 0.01}


# In[39]:


# I'm 27 years old, that's why I use random_state=27
base_models = [
    ('Catboost', CatBoostRegressor(
        iterations=335,
        learning_rate=0.05,
        depth=8,
        random_state=27        
    )),
    ('XGBoost', XGBRegressor(
        n_estimators=245,
        learning_rate=0.01,
        max_depth=42,
        random_state=27
    )),
    ('LightGBM', LGBMRegressor(
        n_estimators=445,
        learning_rate=0.01,
        max_depth=10,
        random_state=27
    )),
    ('RandomForest', RandomForestRegressor(
        n_estimators=855,
        min_samples_split=32,
        min_samples_leaf=2,
        max_features='log2',
        max_depth=75,
        bootstrap=True,
        random_state=27
    )),
    ('HistGradientBoostingRegressor', HistGradientBoostingRegressor(
        min_samples_leaf=88,
        max_iter=900,
        max_depth=47,
        learning_rate=0.01
    )),
#     ('ExtraTreesRegressor', ExtraTreesRegressor(
#         n_estimators=285, 
#         min_samples_leaf=4, 
#         max_depth=14
#     )),
#     ('AdaBoostRegressor', AdaBoostRegressor(
#         n_estimators=295, 
#         learning_rate=0.01
#     )),
#     ('BaggingRegressor', BaggingRegressor(
#         n_estimators=275
#     )),
    ('GradientBoostingRegressor', GradientBoostingRegressor(
        n_estimators=925, 
        min_samples_leaf=26, 
        max_depth=8,
        learning_rate=0.01
    ))
]


# <a id='13'></a>
# # <span style="color: brown">13 - Defining the meta-model</span>

# In[40]:


meta_model = XGBRegressor(
    n_estimators=245,
    learning_rate=0.01,
    max_depth=42,
    random_state=27
)


# ![](https://laboratuar.com/images/iso-11125-2-metalik-puskurtmeli-temizleme-asindiricilari-icin-test-yontemleri---parcacik-boyutu-dagiliminin-belirlenmesi.jpg)

# <a id='14'></a>
# # <span style="color: brown">14 - Creating and fitting the stacking model

# In[41]:


get_ipython().run_cell_magic('time', '', 'stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)\nstacking_model.fit(X, y)\n')


# <a id='15'></a>
# # <span style="color: brown">15 - Predict the validation set and calculate Median Absolute Error score</span>

# In[42]:


y_pred_val = stacking_model.predict(X_val)

medae_val = median_absolute_error(y_val, y_pred_val)
print(f"Validation Median Absolute Error: {medae_val:.4f}")


# <a id='16'></a>
# # <span style="color: brown">16 - Predict on the test data</span>

# In[43]:


y_pred_test = stacking_model.predict(test)
y_pred_test[:10]


# <a id='17'></a>
# # <span style="color: brown">17 - Build DataFrame and make first submission</span>

# In[44]:


submission = pd.DataFrame({
    'id': sample_submission.id,
    'Hardness': y_pred_test
})

submission.to_csv('Kapturov_S3E25_submission.csv', index=False)
submission.head(10)


# ![](https://terracotta.by/upload/iblock/626/2mbuqiuhyl9bywxjpyysgyu35k2r9i1u.jpg)
