#!/usr/bin/env python
# coding: utf-8

# > 

# ### Step 1)  Import relevant libaries

# In[1]:


# Base libraries
import numpy as np
import pandas as pd
import time


# In[2]:


# Visualization
import matplotlib as mpl
import scikitplot as skplt
import seaborn as sns
sns.set()


# In[3]:


# Preprocessing
from sklearn.preprocessing import LabelEncoder
# ML Metrics
from sklearn.metrics import make_scorer, accuracy_score
# ML Model selection
from sklearn.model_selection import train_test_split, GridSearchCV
# CatBoost model
from catboost import CatBoostClassifier, Pool


# ### Step 2)  Extract data from files

# In[4]:


train_csv = pd.read_csv("../input/train.csv")
test_csv = pd.read_csv("../input/test.csv")
y_train = train_csv['Survived'].copy()
X_train = train_csv.drop(['Survived'], axis=1).copy()
X_test = test_csv.copy()


# In[5]:


train_csv.head(5)


# In[6]:


train_csv.info()


# In[7]:


print('Train columns with null values:\n', X_train.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', X_test.isnull().sum())
print("-"*10)


# *Note: For this kernel, I will ignore the data exploration and visualization components and go straight to the feature and model preparing. I recommend viewing the kernels mentioned in the **Credits** section and the bottom. 

# ### Step 3)  Cleaning the data

# In[8]:


def feature_cleaning(actual):
    new = actual.copy()
    #complete missing age with median
    new['Age'].fillna(new['Age'].median(), inplace = True)
    #complete embarked with mode
    new['Embarked'].fillna('S', inplace = True)
    #complete missing fare with median
    new['Fare'].fillna(new['Fare'].median(), inplace = True)
    #drop cabin to ignore it
    new = new.drop(['Cabin'], axis=1)
    
    return new


# In[9]:


X_train_cln, X_test_cln = feature_cleaning(X_train), feature_cleaning(X_test)

# Sanity check
print('Train columns with null values:\n', X_train_cln.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', X_test_cln.isnull().sum())
print("-"*10)


# ### Step 4)  Feature engineering

# In[10]:


# Prepare the data
# 'actual': current dataset, 'new': featured engineered dataset to be returned
def feature_engineering(actual):
    
    # Include all of the features from the actual dataset
    new = actual.copy()
    
    # Engineer new features
    new['FamilySize'] = new['SibSp'] + new['Parch'] + 1
    new['IsAlone'] = np.where(new['FamilySize'] == 1, 1, 0)
    
    def get_title(name):
        return name.split(',')[1].split('.')[0].strip()
    
    new['Title'] = new['Name'].apply(get_title)
    new['Title'] = new['Title'].replace('Mlle', 'Miss')
    new['Title'] = new['Title'].replace('Ms', 'Miss')
    new['Title'] = new['Title'].replace('Mme', 'Mrs')
    
    stat_min = 10
    title_names = (new['Title'].value_counts() < stat_min)
    new['Title'] = new['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
    
    new['FareBin'] = pd.qcut(new['Fare'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    new['AgeBin'] = pd.qcut(new['Age'], 6, labels=[1, 2, 3, 4, 5, 6], duplicates='drop')

    drop_elements = ['PassengerId', 'Name', 'Age', 'Ticket', 'SibSp', 'Parch', 'Fare']
    new = new.drop(drop_elements, axis=1)
    
    return new


# In[11]:


# Feature engineer X_train and X_test data
X_train_eng, X_test_eng = feature_engineering(X_train_cln), feature_engineering(X_test_cln)
X_train_eng.head()


# ### Step 5)  GridSearch CV

# In[12]:


clf = CatBoostClassifier()
params = {'iterations': [500],
          'depth': [4, 5, 6],
          'loss_function': ['Logloss', 'CrossEntropy'],
          'l2_leaf_reg': np.logspace(-20, -19, 3),
          'leaf_estimation_iterations': [10],
#           'eval_metric': ['Accuracy'],
#           'use_best_model': ['True'],
          'logging_level':['Silent'],
          'random_seed': [42]
         }
scorer = make_scorer(accuracy_score)
clf_grid = GridSearchCV(estimator=clf, param_grid=params, scoring=scorer, cv=5)


# In[13]:


X_train_enc = X_train_eng.apply(LabelEncoder().fit_transform)
X_train_enc.head()


# In[14]:


clf_grid.fit(X_train_enc, y_train)
best_param = clf_grid.best_params_
best_param


# In[15]:


best_param


# ### Step 5)  Fit the best model

# In[16]:


# use_best_model params to prevent model overfitting
model = CatBoostClassifier(iterations=1000,
                           loss_function=best_param['loss_function'],
                           depth=best_param['depth'],
                           l2_leaf_reg=best_param['l2_leaf_reg'],
                           eval_metric='Accuracy',
                           leaf_estimation_iterations=10,
                           use_best_model=True,
                           logging_level='Silent',
                           random_seed=42
                          )


# In[17]:


# make the x for train and test (also called validation data)
xtrain, xval, ytrain, yval = train_test_split(X_train_eng, y_train,train_size=0.8,random_state=42)
# sanity check to ensure all features are categories. In our case, yes.
cate_features_index = np.where(X_train_eng.dtypes != float)[0]
cate_features_index
# create a training pool for the model to fit
train_pool = Pool(xtrain, ytrain, cat_features=cate_features_index)


# In[18]:


model.fit(train_pool, eval_set=(xval,yval))


# ### Step 6)  Model checking and submission

# In[19]:


y_train_pred = model.predict(X_train_eng)
skplt.metrics.plot_confusion_matrix(y_train, y_train_pred, normalize=True)


# In[20]:


# Predicting the Test set results
y_pred = model.predict(X_test_eng)
y_pred = list(map(int, y_pred))  # Convert all the y_pred to int in the case y_preds are type floats

submission = pd.DataFrame({
        'PassengerId': test_csv['PassengerId'],
        'Survived': y_pred
    })

submission.to_csv("submission.csv", index=False)


# **Credits:**
# 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
# 
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# 
# https://www.kaggle.com/manrunning/catboost-for-titanic-top-7
# 
# https://www.kaggle.com/miklgr500/catboost-with-gridsearch-cv
