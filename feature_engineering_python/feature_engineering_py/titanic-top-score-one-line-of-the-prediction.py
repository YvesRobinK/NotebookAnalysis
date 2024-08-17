#!/usr/bin/env python
# coding: utf-8

# # [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

# ![](https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg)

# ## Titanic : one lines of the prediction code for LB = 0.80382

# Now I will give a code with forecasting not in the context of the classes of cabins and ports, but in the context of the surnames of passengers (Thanks to: https://www.kaggle.com/mauricef/titanic). 
# 
# 
# It is interesting that there are obvious data errors in the dataset (for example, with respect to the Andersson (see https://www.kaggle.com/c/titanic/discussion/14904#latest-208058), who are not all the same family, and lived in several different cabins), but if they are corrected, the forecast will worsen!
# 
# After the code, I will show how this one line of prediction code was developed.

# Thanks to:
# 
# * [Automatic selection from 20 classifier models](https://www.kaggle.com/vbmokin/automatic-selection-from-20-classifier-models)
# * [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# * [Three lines of code for Titanic Top 15%](https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-15)
# * [Three lines of code for Titanic Top 20%](https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-20)
# * [Titanic Top 3% : cluster analysis](https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis)
# * [Feature importance - xgb, lgbm, logreg, linreg](https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg)

# In[1]:


# Download data and preparing to prediction (including FE) 
# Thanks to: https://www.kaggle.com/mauricef/titanic
import pandas as pd
import numpy as np 
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
df = pd.concat([traindf, testdf], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]
family = df.groupby(df.LastName).Survived
df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)
df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
df['Alone'] = (df.WomanOrBoyCount == 0)
train_y = df.Survived.loc[traindf.index]
df = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1)

test_x = df.loc[testdf.index]


# In[2]:


# My upgrade - the one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 
test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \
          ((test_x.WomanOrBoySurvived > 0.238) & \
           ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))


# In[3]:


# Saving the result
pd.DataFrame({'Survived': test_x['Survived'].astype(int)}, \
             index=testdf.index).reset_index().to_csv('survived.csv', index=False)


# # Tuning the model

# ### Download data and preparing to prediction (including FE) 

# In[4]:


import pandas as pd
import numpy as np 
import graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Download data and preparing to prediction (including FE) 
traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
df = pd.concat([traindf, testdf], axis=0, sort=False)


# In[6]:


# FE
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]
family = df.groupby(df.LastName).Survived
df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount - df.Survived.fillna(0), axis=0)
df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
df['Alone'] = (df.WomanOrBoyCount == 0)


# In[7]:


df


# In[8]:


train_y = df.Survived.loc[traindf.index]
data = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, \
                  df.Sex.replace({'male': 0, 'female': 1})], axis=1)
train_x, test_x = data.loc[traindf.index], data.loc[testdf.index]
train_x.head(5)


# ### Tuning model

# In[9]:


# Tuning the DecisionTreeClassifier by the GridSearchCV
parameters = {'max_depth' : np.arange(2, 5, dtype=int),
              'min_samples_leaf' :  np.arange(2, 5, dtype=int)}
classifier = DecisionTreeClassifier(random_state=1000)
model = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=2, n_jobs=-1)
model.fit(train_x, train_y)
best_parameters = model.best_params_
print(best_parameters)


# In[10]:


model=DecisionTreeClassifier(max_depth = best_parameters['max_depth'], 
                             random_state = 1000)
model.fit(train_x, train_y)


# ### Plot tree

# In[11]:


# plot tree
dot_data = export_graphviz(model, out_file=None, feature_names=train_x.columns, class_names=['0', '1'], 
                           filled=True, rounded=False,special_characters=True, precision=3) 
graph = graphviz.Source(dot_data)
graph 


# ### Prediction

# In[12]:


# # Prediction by the DecisionTreeClassifier
y_pred = model.predict(test_x).astype(int)
print('Mean =', y_pred.mean(), ' Std =', y_pred.std())


# In[13]:


# The one line of the code for prediction : LB = 0.80382 (Titanic Top 4%) 
test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & \
                       (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) | \
                      ((test_x.WomanOrBoySurvived > 0.238) & \
                       ~((test_x.WomanOrBoySurvived > 0.55) & \
                       (test_x.WomanOrBoySurvived <= 0.633)))).astype(int)

y_pred = test_x['Survived']
print('Mean =', y_pred.mean(), ' Std =', y_pred.std())


# As you can see there is a slight difference in std, possibly related to the fact that the rules on the decision tree are shown with rounding. But this did not affect the accuracy of the solution, at least the first 5 decimal places in the Kaggle leaderboard

# ### Saving the result

# In[14]:


# Saving the result
pd.DataFrame({'Survived': y_pred}, index=testdf.index).reset_index().to_csv('submission.csv', index=False)

