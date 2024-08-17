#!/usr/bin/env python
# coding: utf-8

# ## How to Use PyCaret🥕 with Feature Engineering🛠
# 
# ![image](https://ericonanalytics.com/wp-content/uploads/2021/01/image-13.png)
# 
# I think kaggle, especially this competition(tabular playground series) is a fun playground where you can focus on feature engineering and minimize the effort spent on model tuning.
# 
# And based on that most of the top notebooks are AutoML, this kernel is introduced.
# 
# [PyCaret](https://pycaret.org/) is an open source, low-code machine learning library in Python that allows you to go from preparing your data to deploying your model within minutes in your choice of notebook environment.
# 
# 
# This notebook uses [Tabular-Playground-Series-Apr](https://www.kaggle.com/c/tabular-playground-series-apr-2021) dataset, but you can do the same process on [Titanic dataset](https://www.kaggle.com/c/titanic/data), and it can be easily used with most structured data.
# 

# ## ⚙️ Install PyCaret & Import Libraries
# 
# Kaggle notebooks do not provide pycaret by default. So, you can install it with the following command :
# 
# > `pip install pycaret`

# In[1]:


get_ipython().system('pip install pycaret')


# Call the basic data science library.

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from pycaret.classification import *


# In[3]:


train = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')


# ## 🏭 Feature Engineering
# 
# I recommend that you refer to the following discussion for a basic full framework of feature engineering.
# 
# - [@Chris Deotte](https://www.kaggle.com/cdeotte): [IEEE-CIS Fraud Detection | Feature Engineering Techniques](https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575)
# 
# ### Complex(Text + Num) 
# 
# Along with the general Titanic, this competition helps to improve performance if there is a pre-processing for the Ticket Cabin Name.
# 
# `Ticket`s and `Cabin`s are divided into letters and numbers, and `Name`s are divided into family names and names.
# 
# 
# ### Familiy Size & isAlone
# 
# `FamiliySize` coude be made like this : 
# 
# - `data['FamilySize'] = data['SibSp'] + data['Parch']`
# 
# > The assumption that the survival rate is low can also be included if the number of families to be kept is large.
# 
# ### Sex
# 
# The `Sex` can be labeled as `1,0` or `0,1` depending on the male/female.
# 
# ### Age
# 
# `Age` can usually be filled in for missing values as mean or median. (usual in numeric values)
# 
# However, in [my previous notebook, the EDA](https://www.kaggle.com/subinium/tps-apr-highlighting-the-data) result showed that there was some distribution of Age according to Pclass, so I grouped it into Pclass and used the median value for each group.
# 
# ### Embarked
# 
# For `Embarked`, you can fill in the missing values with the most observations. 
# 
# I have filled this data with X, which means there is no data, assuming that this data is randomly generated

# In[4]:


def converter(x):
    '''
    convert text to 2 values(string part & numeric part)
    '''
    c, n = '', ''
    x = str(x).replace('.', '').replace('/','').replace(' ', '')
    for i in x:
        if i.isnumeric():
            n += i
        else :
            c += i 
    if n != '':
        return c, int(n)
    return c, np.nan

# Feature Engineering based on EDA
def create_extra_features(data):
    data['Ticket_type'] = data['Ticket'].map(lambda x: converter(x)[0])
    data['Ticket_number'] = data['Ticket'].map(lambda x: converter(x)[1])
    data['Cabin_type'] = data['Cabin'].map(lambda x: converter(x)[0])
    data['Cabin_number'] = data['Cabin'].map(lambda x: converter(x)[1])
    data['Name1'] = data['Name'].map(lambda x: x.split(', ')[0])    
    data['Name2'] = data['Name'].map(lambda x: x.split(', ')[1])
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['isAlone'] = data['FamilySize'].apply(lambda x : 1 if x == 1 else 0)
    
    # Sex
    data['Sex'] = data['Sex'].map({'male':0, 'female':1})
    
    # Age
    age_map = train[['Age', 'Pclass']].dropna().groupby('Pclass').median().to_dict()['Age']
    data.loc[train['Age'].isnull(), 'Age'] = data.loc[train['Age'].isnull(), 'Pclass'].map(age_map)

    # Embarked
    data['Embarked'] = data['Embarked'].fillna('X')
    return data

train = create_extra_features(train)
test = create_extra_features(test)


# ### Remove Features
# 
# You can select features to ignore in pycaret, but I just drop them.

# In[5]:


train.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ## 🔧 Use Cateory Encoders
# 
# The generated text data can use several encoders.
# 
# I've seen in previous [Categorical Competitions & Benchmarks](https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark) that the performance varies a lot with this encoding method.
# 
# Among the various methods of target encoding, I tried using `CatBoostEncoder`.

# In[6]:


from category_encoders.cat_boost import CatBoostEncoder

ce = CatBoostEncoder()

column_name = ['Ticket_type', 'Embarked', 'Cabin_type', 'Name1', 'Name2']
train[column_name] = ce.fit_transform(train[column_name], train['Survived'])
test[column_name] = ce.transform(test[column_name])


# Check the dataset for proper input.

# In[7]:


train.head()


# ## 🥕 Setup AutoML Enviroment (PyCaret)
# 
# To use pycaret's automl, you can basically enter the input data, the desired target column name, the number of folds of the cross validation, and the rest of the settings as desired.
# 
# There are many different settings you can put in. **Normalize, remove outliers, etc**. You can add your insights.
# 
# And here too, some missing values can be resolved, and `numeric_inmputation` was used to fill the numerical missing values with the median. (`Fare` feature)

# In[8]:


setup(data = train, 
      target = 'Survived',
      numeric_imputation = 'median',
      fold=5,
      silent = True,
     )


# ## ✅ Benchmark
# 
# Models can be made individually, but they provide benchmarks by default.
# 
# It provides a benchmark by turning all representative models used in machine learning with a small number of iters.
# 
# Each model shows scores of **Accuracy, AUC, Recall, Prec, F1, etc**.
# 
# The distribution of the dataset for this competition is severely unbalanced.
# 
# - [How do I compare the leaderboard and private CV results?](https://www.kaggle.com/c/tabular-playground-series-apr-2021/discussion/231187)
# 
# Looking at what I have discussed with people here, there are many thoughts that the difference between my CV results and the public leaderboard is the result of the distribution of data.
# 
# Therefore, it is recommended to look at AUC or F1 together rather than simply look at Accuracy. (In my case, doing that resulted in better performance)
# 

# In[9]:


best_model = compare_models(sort = 'Accuracy', n_select = 3) # we will use it later


# Of these, the N models that appear at the top can also be extracted. (And can be tuned)
# 
# In general, it is good to select 3 to 5 and proceed with an ensemble such as blending.

# ## 🆕 Create Model
# 
# In general, individual models can be made like this:
# 
# I recommend starting with the fastest, almost best performing LightGBM model on these Tabular datasets.
# 
# Each model name is a code. Each code can be used as it is in the table above.
# 

# In[10]:


lightgbm = create_model('lightgbm')


# ## 🔄 Tune Model
# 
# You can use `tune_model` to tune the performance of your model.
# 
# But here, the number of iters is only 10, so let's increase the performance by increasing the number of times. (I recommend more than 100)
# 
# And the target metric is accuracy, but in this case, too much fit for the train occurs.
# 
# In my case, when I used AUC, LB came out better.

# In[11]:


lightgbm = tune_model(lightgbm
#                      ,num_iter=100
#                      ,optimize='AUC' 
                     )


# ## 📊 Plotting
# 
# 
# It provides a variety of plotting, and you can better understand the results by plotting like this:
# 
# ### Confusion Matrix
# 
# There are two possible errors in binary classification. Let's consider the direction of the model by looking at the False Positive and False Negative.
# 
# 
# And it is good to look at the confusion matrix of several models and consider how to apply it to the ensemble.

# In[12]:


plot_model(lightgbm, plot = 'confusion_matrix')


# ### Validation Metrix Check
# 
# I have an imbalance in the data distribution, but I think that CV will be able to resolve the gap between LB and my results to some extent.
# 
# Let's check the CV's score and check the overfitting.

# In[13]:


plot_model(lightgbm, plot = 'learning')


# In[14]:


plot_model(lightgbm, plot = 'vc')


# ### Feature Importance
# 
# You can select important features to increase the model's efficiency and make a good model.
# 
# The strategy for choosing a good feature among these features is the following notebooks.
# 
# - [@michau96](https://www.kaggle.com/michau96) : [Simple trick to select variables for model 💡](https://www.kaggle.com/michau96/simple-trick-to-select-variables-for-model)

# In[15]:


plot_model(lightgbm, plot = 'feature_all')


# ### ETC
# 
# Other than that, you can look at AUC Curve, Decision Boundary, etc. and provide various visualizations.

# In[16]:


plot_model(lightgbm, plot = 'threshold')


# In[17]:


plot_model(lightgbm, plot = 'auc')


# In[18]:


plot_model(lightgbm, plot = 'boundary')


# ## 🗳️ Blending Model
# 
# **Blending model**s is a method of ensembling which uses consensus among estimators to generate final predictions. 
# 
# 
# The idea behind blending is to combine different machine learning algorithms and use a majority vote or the average predicted probabilities in case of classification to predict the final outcome. 
# 
# You can create models individually and pass them as a list.

# In[19]:


blended = blend_models(estimator_list = best_model, fold = 5, method = 'soft')

# lightgbm = create_model('lightgbm')
# catboost = create_model('catboost')
# blended = blend_models(estimator_list = [lightgbm, catboost], fold = 5, method = 'soft')


# ## 📏 Calibrate Model
# 
# When performing Classification experiments you often want to predict not only the class labels, but also obtain a probability of the prediction. T
# 
# his probability gives you some kind of confidence. 
# 
# Some models can give you poor estimates of the class probabilities. 
# 
# Well calibrated classifiers are probabilistic classifiers for which the probability output can be directly interpreted as a confidence level. 
# 
# Calibrating classification models in PyCaret is as simple as writing calibrate_model. 

# In[20]:


calibrated_blended = calibrate_model(blended)


# ## 🔥 Submit your Result!!
# 
# It can be used to predict on unseen data using `predict_model` function.
# 
# The format for submission is as follows:

# In[21]:


predictions = predict_model(calibrated_blended, data = test)
predictions.head()


# In[22]:


sample_submission['Survived'] = predictions['Label']
sample_submission.to_csv(f'submission.csv',index=False)


# Although not introduced in this notebook, you can also look at interpretation of the model results using SHAP, etc.
# 
# ### If the content is helpful, please upvote. :)
