#!/usr/bin/env python
# coding: utf-8

# In[1]:


from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import TargetEncoder

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedStratifiedKFold, cross_val_score, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier, VotingClassifier


pd.options.display.max_rows=200
pd.set_option('mode.chained_assignment', None)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')
test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')


# In[3]:


y_train = train.Survived.copy()
train = train.drop('Survived', axis=1)
X_test = test


# In[4]:


train.info()


# In[5]:


# if you go through the names, you will find out that every name has a designation associated, these designations can
# provide us with an extra categorical feature which we can encode using binary encoding or the mean target encoding!!!
# so dont just drop name column!!

def name_labeling(df):
    for i in ['Mr.', 'Mrs.', 'Miss', 'Master', 'Army', 'Revered/Important', 'rare', 'Doctor']:
        if i == 'Army':
            df.Name[df.Name.str.contains(pat='(Major. |Col. |Capt. )', regex=True)] = 'Army'

        elif i == 'Revered/Important':
            df.Name[df.Name.str.contains(pat='(Rev. |Countess. |Jonkheer. |Sir. |Lady. )', regex=True)] = 'Revered/Important'

        elif i == 'rare':
            df.Name[df.Name.str.contains(pat='(Mme. |Ms. |Mlle. |Don. |Dona. )', regex=True)] = 'rare'

        elif i == 'Doctor':
            df.Name[df.Name.str.contains(pat='Dr. ')] = 'doctor'

        else:
            df.Name[df.Name.str.contains(pat=f'{i}', regex=False)] = i
    return df


# In[6]:


# char = X_train.Ticket[X_train.Ticket.str.contains(pat='([a-zA-Z])')].str.split(n=1, expand=True).iloc[:, 0]
# char.unique()

# using the above code you can first of all find out that what are the tickets which have alphabets in ticket numbers and then
# use those alphabet patterns to separate them out using regex.

def ticket_labeling(df):
    for label, pattern in [('ca', 'C[.]?A[.]?'),('soton', 'SOTON'), ('ston', 'STON'), ('wc', 'W[.]?[/]?C'), 
                           ('sc', 'S[.]?C[.]?'), ('a', 'A[.]?'), ('soc', 'S[.]?O[.]?[/]?C'), ('pp', 'PP'), 
                           ('fc', '(F.C.|F.C.C.)'), ('rest_char', '[A-Z]'), ('small_serial_number', '^\d{3,5}$'), 
                           ('large_serial_number', '^\d{6,7}$')]:
        
        df.Ticket[df.Ticket.str.contains(pattern)] = label
        
    return df


# in **Cabin** feature we have a main cabin type for example 'A', 'B' etc. but the problem is that the feature has a lot of null
# values, we should extract the main cabin name and fill out all the null values as 'NaN' for now, as we will see that there
# is a relation between the Cabin and the Pclass, we can use this relation to impute the missing values in Cabin!!!

# In[7]:


def cabin_labeling(df):
    for i in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        
        df.Cabin[df.Cabin.str.contains(i, na=False)] = i
        
        # temporary nan value imputation for visualization.
        
        df.Cabin.fillna('missing', inplace=True)
        
    return df


# In[8]:


temp = cabin_labeling(train.copy())


# In[9]:


temp = temp.groupby(['Pclass', 'Cabin'])[['Name']].count().rename(columns={'Name':'Passengers'})


# In[10]:


temp = temp.reset_index()
temp_no_missing_value = temp[temp.Cabin != 'missing']
temp_missing_value = temp[temp.Cabin == 'missing']


# In[11]:


plt.figure(figsize=(16, 8))

plt.subplot(121)
sns.set(font_scale=1.5)
sns.barplot(data=temp_no_missing_value, x='Cabin', y='Passengers', hue='Pclass')

plt.subplot(122)
sns.set(font_scale=1.5)
sns.barplot(data=temp_missing_value, x='Cabin', y='Passengers', hue='Pclass')
plt.show()


# It's quite evident that only 1st class passengers were present in Cabin A, B, C and then some 2nd class in Cabin D, hence while imputing values for 1st class we will use this information and impute values using this information. Similarly we can impute the missing values in 3rd class with only Cabin E, F and G. 
# 
# This is one of the good approach to the large missing values present in the Cabin feature, it is only an estimation with the data already present, this might help us use the information present in the 200 non-missing values, in Cabin feature, and improve the accuracy or it might just create unnecessary noise for the model. We can always ask the model if it likes this imputed Cabin feature or not using a Feature Engineering class which allows us to drop and keep the Cabin feature. we can then add feature engineering class and a classification model for example Logistic regression in a pipeline and then grid search the feature engineering class's parameters to find out whether the model found the Cabin feature useful or not.... We will be performing this search in the code below!!!

# *We also have a very interesting relation between the **pclass, sex and age** which can help us better impute the missing age values, this is shown in the visualization below.*

# In[12]:


plt.figure(figsize=(12, 8))
sns.set(font_scale=1.5)
sns.violinplot(data=train, x='Sex', y='Age', hue='Pclass')
plt.show()


# From the above visualization we can clearly see that the age of passengers are dependent upon the gender and the pclass, in 1st class the average age of the passenger is higher than both the 2nd and 3rd class, also in 1st class the average age of women is lower than the average age of men. **We can utilize this knowledge to better impute the missing Age values in dataset!!!**

# ## Missing value imputation

# In[13]:


def combined_labeling(df):
    
    return cabin_labeling(ticket_labeling(name_labeling(df)))


# In[14]:


X_train = combined_labeling(train.copy())
X_test = combined_labeling(test.copy())


# In[15]:


def proportions(df):
    """
    this function returns the proportions of passengers in various cabins on the basis of Pclass
    """
    df = df.groupby(['Pclass', 'Cabin'])['Name'].count().reset_index().rename(columns={'Name':'Passengers'})
    
    # lets find proportion of passengers in different cabins for a given pclass.

    total_passengers_in_cabins = df.Passengers[df.Cabin != 'missing'].sum()  # total passengers in cabins in a pclass.

    cabin_proportions = df['Passengers'][df.Cabin != 'missing'] / total_passengers_in_cabins

    return cabin_proportions

def no_cabins(df):
    """
    This function returns the number of passengers in a given Pclass with out any Cabin feature value
    """
    
    df = df.groupby(['Pclass', 'Cabin'])['Name'].count().reset_index().rename(columns={'Name':'Passengers'})
    
    return df.Passengers[df.Cabin == 'missing']


def cabin_imputer(df, missing_vals, cabin_proportions):
    
    if all(df.Pclass == 1):

        imputation_ndarray = np.random.choice(['A', 'B', 'C', 'D', 'E', 'T'], size=missing_vals[0], p=cabin_proportions[1])
        
        missing_values_index = df[df.Cabin == 'missing'].index
        
        imputation_series = pd.Series(imputation_ndarray, index=missing_values_index)
        
        return imputation_series
    
        
    elif all(df.Pclass == 2):
        
        imputation_ndarray = np.random.choice(['D', 'E', 'F'], size=missing_vals[1], p=cabin_proportions[2])
        
        missing_values_index = df[df.Cabin == 'missing'].index
        
        imputation_series = pd.Series(imputation_ndarray, index=missing_values_index)
        
        return imputation_series
    
        
    elif all(df.Pclass == 3):
        
        imputation_ndarray = np.random.choice(['E', 'F', 'G'], size=missing_vals[2], p=cabin_proportions[3])
        
        missing_values_index = df[df.Cabin == 'missing'].index
        
        imputation_series = pd.Series(imputation_ndarray, index=missing_values_index)
        
        return imputation_series


def imputer(df):
    
    # Using the information revealed by the above violin plot, lets impute the missing age values grouping by the pclass and sex
    # feature as shown below.
    df.Age.fillna(df.groupby(['Pclass', 'Sex']).Age.transform('mean'), inplace=True)
    
    
    
    cabin_proportions = X_train.groupby(['Pclass']).apply(proportions) # cabin proportions with respect to 
                                                                       # Pclass(multi indexed with Pclass as first index)
    missing_vals = list(df.groupby(['Pclass']).apply(no_cabins))  # list containing missing cabin values in each Pclass
    
    
    # we have to keep the proportions of cabins same as that in X_train for X_test as we are imputing cabins using the info
    # from training set, but the missing vals have are different values for X_train and X_test so we have to keep missing vals
    # specific for both the data sets.
    missing_val_imputed_series = df.groupby(['Pclass']).apply(partial(cabin_imputer, 
                                                                      missing_vals=missing_vals, 
                                                                      cabin_proportions=cabin_proportions)) 
    # get the multi indexed series
    
    missing_val_imputed_series = missing_val_imputed_series.reset_index().set_index('PassengerId') 
    # reset index and set PassengerId as index
    
    missing_val_imputed_series = missing_val_imputed_series.drop('Pclass', axis=1).sort_index().iloc[:, 0] 
    # finally we drop the Pclass column and sort the index, the object type is DataFrame, so we have to use .iloc[:, 0]
    # to take out the series.
   
    
    for index, val in zip(missing_val_imputed_series.index, missing_val_imputed_series):
        df.Cabin[index] = val
        
    
    # lets fill out all the other missing values using the most frequent value.
    
    for feature in df.columns:
        df[feature].fillna(df[feature].mode()[0], inplace=True)
        
    # finally there is a 'T' cabin with only one instance in training Cabin feature and no occurence in test set so we should
    # convert it to 'E'.
    
    df.Cabin[df.Cabin == 'T'] = 'E'
    
    # finally lets drop the passengerid as index
    df.reset_index(drop=True, inplace=True)
        
    return df


# In[16]:


X_train_imputed = imputer(X_train.copy())
X_test_imputed = imputer(X_test.copy())


# ## Feature Engineering
# 
# 
# We have imputed our data with reasonable values, now there is the part of feature engineering, we wont go too deep, but there are some questions that i wish to ask my model. 
# 
# * As we know that we have assumed a lot while imputing Cabin feature, is it better to just drop it or we should keep it?
# * is it better to one hot encode Embarked or Target Encode it?
# * is it better to Simplify SibSp and Parch as relatives(1) or no relatives(0) instead of discrete number of relatives and one hot encode it, or is it better to let it stay as it is?
# * Should we drop Ticket?
# * Should we drop Name?
# * How much smoothing should we use for our raget encoded columns?
# * Which is better StandardScaler, MinMaxScaler or RobustScaler?
# 
# To make our model answer these questions, we can put all of it in a feature engineering class and combine this transformation step with our model using a pipeline and cross validate... But beware the dataset is too small and variance is quite high, we might not get a plausible result from crossvalidations, in which case we will just have to check out few of the first combinations on our whole Stacking ensemble and check out the final result.

# In[17]:


class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    def __init__(self, drop_Cabin=False, drop_Name=False, Embarked_target=False, SibSp_Parch_simplify=True, 
                 drop_Ticket=False, scaler='MinMaxScaler', smoothing=10, test=False):
        self.drop_Cabin = drop_Cabin
        self.drop_Name = drop_Name
        self.drop_Ticket = drop_Ticket
        self.Embarked_target = Embarked_target
        self.SibSp_Parch_simplify = SibSp_Parch_simplify
        self.scaler = scaler
        self.smoothing = smoothing
        self.target_encoder = None
        self.test = test

        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X['Pclass'] = X.Pclass.astype(str)
        
        dummies = pd.get_dummies(X.loc[:, ['Pclass', 'Sex']])
        X = pd.concat([X, dummies], axis=1).drop(['Pclass', 'Sex'], axis=1)
        
        
        # next we will target encode the Name, Ticket and Cabin. We want only the training set to fit to the Target encoding
        # class object and not the test set.
        
        if not self.test and len(X) > 400: # assuming that cross validation test set size will never be greater tham 400
            
            # i do not know whether it will be better to target encode 'Embarked' or one hot encode it, so lets test it out!!!
            self.cols = ['Ticket', 'Name', 'Cabin'] if not self.Embarked_target else ['Cabin', 'Name', 'Ticket', 'Embarked']
            
            self.target_encoder = TargetEncoder(cols=self.cols, smoothing=self.smoothing)
            self.target_encoder.fit(X.loc[:, self.cols], y_train.iloc[X.index])
            X.loc[:, self.cols] = self.target_encoder.transform(X.loc[:, self.cols])
            
        else:
            X.loc[:, self.cols] = self.target_encoder.transform(X.loc[:, self.cols])
            
            
            
        if self.drop_Cabin:
            
            X.drop('Cabin', axis=1, inplace=True)
            
        if self.drop_Name:
            
            X.drop('Name', axis=1, inplace=True)
            
        if self.drop_Ticket:
            
            X.drop('Ticket', axis=1, inplace=True)
    
            
        if self.SibSp_Parch_simplify:
            X['SibSp'].replace({0:'No', 1:'Yes', 2:'Yes', 3:'Yes', 4:'Yes', 
                                5:'Yes', 6:'Yes', 7:'Yes', 8:'Yes', 9:'Yes'}, inplace=True)
            X['Parch'].replace({0:'No', 1:'Yes', 2:'Yes', 3:'Yes', 4:'Yes', 
                                5:'Yes', 6:'Yes', 7:'Yes', 8:'Yes', 9:'Yes'}, inplace=True)
            
            dummies = pd.get_dummies(X.loc[:, ['SibSp', 'Parch']])
            X = pd.concat([X, dummies], axis=1).drop(['SibSp', 'Parch'], axis=1)
            
        X = pd.get_dummies(X)
                
        
        features_to_scale = [feature for feature in X.columns if X[feature].max() != 1 and X[feature].min() != 1]

        # we do not want to scale one hot encoded features columns.
        
        if not self.test and len(X) > 400:
            if self.scaler == 'StandardScaler':
                self.scale_transformer = StandardScaler()
            
            elif self.scaler == 'MinMaxScaler':
                self.scale_transformer = MinMaxScaler()
                
            elif self.scaler == 'RobustScaler':
                self.scale_transformer = RobustScaler()
                
            scaled_features = self.scale_transformer.fit_transform(X.loc[:, features_to_scale])
            X.loc[:, features_to_scale] = pd.DataFrame(data=scaled_features, columns=features_to_scale, index=X.index)
            
        else:
            scaled_features = self.scale_transformer.fit_transform(X.loc[:, features_to_scale])
            X.loc[:, features_to_scale] = pd.DataFrame(data=scaled_features, columns=features_to_scale, index=X.index)
        
        return X
            


# In[18]:


def results(cv_results_, n):
    df = pd.DataFrame(cv_results_)[['params', 'mean_test_score']].nlargest(n, columns='mean_test_score')
    for i in range(len(df)):
        print(f'{df.iloc[i, 0]} : {df.iloc[i, 1]}')


# In[19]:


param_grid = {'feature_engineering__drop_Cabin':[True, False],
              'feature_engineering__drop_Ticket':[True, False],
              'feature_engineering__drop_Name':[True, False],
              'feature_engineering__Embarked_target':[True, False], 
              'feature_engineering__SibSp_Parch_simplify':[False],
              'feature_engineering__scaler':['StandardScaler'],
              'feature_engineering__smoothing':[5]}


feature_engineering_pipeline = Pipeline([('feature_engineering', FeatureEngineering()),
                                         ('model', LogisticRegression())])

grid = GridSearchCV(feature_engineering_pipeline, param_grid, 
                    cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42), 
                    scoring='accuracy', verbose=2, n_jobs=-1)


# In[20]:


grid.fit(X_train_imputed.copy(), y_train)


# In[21]:


results(grid.cv_results_, n=5)


# Well, to be honest the above results do not tell us much about which is the best data set, so i just tested out few of the top ones and found the mix below to be the best.

# In[22]:


fe = FeatureEngineering(drop_Cabin=True, drop_Name=False, drop_Ticket=False, Embarked_target=False, SibSp_Parch_simplify=False, 
                        scaler='StandardScaler', smoothing=5)


# In[23]:


X_train_fe = fe.fit_transform(X_train_imputed.copy())


# In[24]:


fe.test = True


# In[25]:


X_test_fe = fe.transform(X_test_imputed.copy())


# ## Hyperparameter Tuning

# From the above feature engineering practice we were able to score accuracy of about 0.772 on the test submission using Logistic Regression, But we can do better than that. In this section we will be finding best hyper parameters for a number of models with different strengths and weaknesses so that we can finally blend the results and achieve higher accuracy. The idea is that every model has different strengths and weaknesses, there will be instances that,  for eg. SVC classifies correctly but LogisticRegression cannot, but LogisticRegression might classify some other instances correctly which SVC cannot, these difference will be learnt by the final **meta model** in stacking and we will be able to achieve higher accuracy.
# 
# We will train the following models as base models for stacking:
# * LogisticRegression
# * SVC
# * KNeighborsClassifier
# * RandomForestClassifier
# * GradientBoostingClassifier
# * XGBCLassifier
# 
# 

# #### Tips:
# 
# * When grid searching hyperparameters you should look at top 10 or so best results to get a better idea, if you find a parameter mix which gives you a decent result(not the best) but it is more regulrized than the top result, choose the regularized option and accept the loss in accuracy(if its not a lot), This will ensure that the variance is low and you will actually have more chance of achieving similar result with first model than the best model as per grid search. 
# 
# * The above practice becomes a cumbersome for tree ensemble methods if you just have to go through the grid search results, simple reason being that there are too many parameters which affect the bias and variance, unlike LogisticRegression or SVC. In this case we will plot the results!!! check out the code below.

# In[26]:


def parameter_plot(model, X, y, n_estimators=[100, 200, 300, 400, 600, 900, 1300, 1700, 2000, 2500], hyper_param=None, **kwargs):
    param_name, param_vals = hyper_param
    param_grid = {'n_estimators':n_estimators,
                  f'{param_name}':param_vals}
    
    grid = GridSearchCV(model(**kwargs), param_grid, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42), 
                        scoring='accuracy', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)['mean_test_score'].values
    results = results.reshape(len(param_vals), len(n_estimators))
    
    plt.figure(figsize=(15, 9))
    for i in range(1, len(param_vals) + 1):
        plt.plot(n_estimators, results[i-1], label=f'{param_name} - {param_vals[i-1]}')
      
    plt.legend()
    plt.show()


# In[27]:


def learning_curve_plotter(Model, X, y, params_1, params_2, step=50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    plt.figure(figsize=(16, 7))
    for i, (name, params) in enumerate([params_1, params_2]):
        train_score = []
        val_score = []
        plt.subplot(1, 2, i+1)
        for j in range(100, len(X_train), step):
            model = Model(**params).fit(X_train[:j], y_train[:j])
            train_score.append(model.score(X_train[:j], y_train[:j]))
            val_score.append(model.score(X_test, y_test))
            
        plt.plot(train_score, 'r-', label='Training accuracy')
        plt.plot(val_score, 'b-', label='Validation accuracy')
        plt.title(f'{name}')
        plt.xlabel('Training set size')
        plt.ylabel('Accuracy')
        plt.legend()
            
    plt.show()


# ### LogisticRegression

# In[28]:


param_grid_logreg = {'penalty':['elasticnet'],
                     'C':[0.03],
                     'l1_ratio':[0.0],
                     'solver':['saga']} # these params are the best params


# In[29]:


grid_logreg = GridSearchCV(LogisticRegression(), param_grid_logreg, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[30]:


grid_logreg.fit(X_train_fe, y_train)


# In[31]:


results(grid_logreg.cv_results_, n=40)


# ### KNeighborsCLassifier

# In[32]:


param_grid_knn = {'n_neighbors':[20],
                     'weights':['uniform'],
                  'algorithm':['ball_tree']}


# In[33]:


grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[34]:


grid_knn.fit(X_train_fe, y_train)


# In[35]:


results(grid_knn.cv_results_, n=60)


# ### SVC

# In[36]:


param_grid_svc = {'C':[0.5],
                  'kernel':['rbf'],
                  'gamma':[0.1]}


# In[37]:


grid_svc = GridSearchCV(SVC(), param_grid_svc, cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[38]:


grid_svc.fit(X_train_fe, y_train)


# In[39]:


results(grid_svc.cv_results_, n=40)


# ### RandomForestClassifier
# 
# 
# Lets first check the max_depth vs n_estimator relationship for the given dataset for our RandomForestClassifier from the plot then we can narrow down our grid search.

# In[40]:


parameter_plot(RandomForestClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 8, 9, None]))


# Looking at the above plot it is clear that max depth of 3 is out of question. for the depths from 5-9 we can see that as we increase the depth the accuracy increase, but overfitting also increases. We should choose a depth of 5 and 9 for testing for 300 to 500 trees. while using grid search we will also explore other parameters such as max_samples, max_features which will help diversify the trees even more. 
# 
# The titanic dataset is quite small so we might not be able to leverage most out of the above practice, but you can imagine if the data is too large, it will be best to first get an idea of how the params affect the accuracy and then select a narrow Randomized search, as grid search will be take too much time.

# In[41]:


param_grid_random = {'n_estimators':[200, 500],
                     'max_depth':[5, 9],
                     'max_samples':[0.5, 0.7],
                     'max_features':[0.5, 0.7], 
                     'min_samples_split':[2, 5, 8]} 


# In[42]:


grid_random = GridSearchCV(RandomForestClassifier(), param_grid_random, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=4)


# In[43]:


grid_random.fit(X_train_fe, y_train)


# In[44]:


results(grid_random.cv_results_, n=40)


# lets choose one model which is more regularized than the top one....
# 
# **model with params with more regularization**
# * {'max_depth': 5, 'max_features': 0.5, 'max_samples': 0.7, 'min_samples_split': 5, 'n_estimators': 500} : 0.8288 :(
# 
# **Best model**
# * {'max_depth': 9, 'max_features': 0.7, 'max_samples': 0.7, 'min_samples_split': 5, 'n_estimators': 500} : 0.837
# 
# 
# *(Grid search values may vary if it runs again)*
# 
# 
# top model has **more trees, more max_samples ratio and more depth**, these parameters if increased, reduces bias.

# In[45]:


params_1 = ('Regularized model', {'max_depth': 5, 'max_features': 0.5, 'max_samples': 0.7, 
                                  'min_samples_split': 5, 'n_estimators': 500})

params_2 = ('Best model', {'max_depth': 9, 'max_features': 0.7, 'max_samples': 0.7, 
                           'min_samples_split': 5, 'n_estimators': 500})

learning_curve_plotter(RandomForestClassifier, X_train_fe, y_train, params_1, params_2, step=30)


# Well its quite clear which model we should be choosing!!!! the top model is horribly overfitting the data which can result in the top model not performing good on the final test set, whereas our regularized option will perform way better as per above results..... **Dont just straight away choose the best grid search params folks!!!!**

# ### GradientBoostingClassifier

# In[46]:


parameter_plot(GradientBoostingClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 6, 9]))


# n_estimators from range 100 - 400 looks good, we will be testing depth 3 with 200-400 n_estimators. We can observe that as the n_estimators increase the model starts to overfit the data and accuracy decreases. 

# Now as we have choosen a max depth, lets see how learning rate affects accuracy with change in estimators

# In[47]:


parameter_plot(GradientBoostingClassifier, X_train_fe, y_train, hyper_param=('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.1]),
              max_depth=3) # we have passed depth - 3 to **kwargs


# From above plot learning rate of 0.04 - 0.06 looks good for n_estimtors between 300 - 500. We can also observe that as the number of trees increase, the model starts to overfit, which is worse for higher learning rates, in contrast learning rate of 0.01 actually takes lead with more estimators which is quite intuitive

# In[48]:


param_grid_gradient = {'max_depth':[3],
                       'n_estimators':[300, 400, 500],
                       'learning_rate':[0.035, 0.055],
                       'subsample':[0.4, 0.6],
                       'max_features':[0.4, 0.6],
                       'min_samples_split':[2, 5, 8, 12]
                      }


# In[49]:


grid_gradient = GridSearchCV(GradientBoostingClassifier(), param_grid_gradient, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[50]:


grid_gradient.fit(X_train_fe, y_train)


# In[51]:


results(grid_gradient.cv_results_, n=25)


# **model with params with more regularization**
# * {'learning_rate': 0.055, 'max_depth': 3, 'max_features': 0.4, 'min_samples_split': 12, 'n_estimators': 400, 'subsample': 0.6} : 0.826629 :(
# 
# **Best model**
# * {'learning_rate': 0.055, 'max_depth': 3, 'max_features': 0.4, 'min_samples_split': 2, 'n_estimators': 400, 'subsample': 0.4} : 0.8355
# 
# 
# *(Grid search values may vary if it runs again)*

# In[52]:


params_1 = ('Regularized model', {'learning_rate': 0.035, 'max_depth': 3, 'max_features': 0.4, 
                                  'min_samples_split': 10, 'n_estimators': 400, 'subsample': 0.6})
params_2 = ('Best model', {'learning_rate': 0.055, 'max_depth': 3, 'max_features': 0.4, 
                           'min_samples_split': 2, 'n_estimators': 400, 'subsample': 0.4})

learning_curve_plotter(GradientBoostingClassifier, X_train_fe, y_train, params_1, params_2, step=100)


# Regularization is not too great, we might as well use the best model

# ### XGBClassifier

# In[53]:


parameter_plot(XGBClassifier, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 6]))


# max depth 6 looks to be quite good

# In[54]:


parameter_plot(XGBClassifier, X_train_fe, y_train, 
               hyper_param=('learning_rate', [0.01, 0.03, 0.05, 0.07, 0.1]), max_depth=6) 


# From the above plots learning rate of 0.01 - 0.03 looks good for estimators between 200 - 500

# In[55]:


param_grid_xgb = {'n_estimators':[300, 450],
                  'learning_rate':[0.02, 0.03],
                  'max_depth':[6],
                  'subsample':[0.5, 0.7],
                  'colsample_bylevel':[0.5, 0.7],
                  'reg_lambda':[1, 5, 15, ]
                 }


# In[56]:


grid_xgb = GridSearchCV(XGBClassifier(), param_grid_xgb, 
                        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='accuracy', verbose=2, n_jobs=-1)


# In[57]:


grid_xgb.fit(X_train_fe, y_train)


# In[58]:


results(grid_xgb.cv_results_, n=30)


# **model with params with more regularization**
# * {'colsample_bylevel': 0.5, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 300, 'reg_lambda': 5, 'subsample': 0.7} : 0.8316 :(
# 
# **Best model**
# * {'colsample_bylevel': 0.5, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 450, 'reg_lambda': 1, 'subsample': 0.5} : 0.8378
# 
# (Grid search values may vary if it runs again)

# In[59]:


params_1 = ('Regularized model', {'colsample_bylevel': 0.5, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 300, 'reg_lambda': 5, 'subsample': 0.7})
params_2 = ('Best model', {'colsample_bylevel': 0.5, 'learning_rate': 0.03, 'max_depth': 6, 'n_estimators': 450, 'reg_lambda': 1, 'subsample': 0.5})

learning_curve_plotter(XGBClassifier, X_train_fe, y_train, params_1, params_2, step=50)


# Funny part is that with just 0.83 accuracy in grid search the learning curve is way better for the regularized model against the best model which scored 0.837 in grid search!!! **even accuracy for regularized model is way better when we look at learning curve validation accucary**  :D
#  
# 
# for all three tree ensembles, we could overfit them to reach accuracy of 0.85 in grid search, but grid searching alone gives such a superficial estimate of the tru accuracy!!!!! 

# #### Lets summarize the way we just grid searched good parameters
# 
# * we first checked out our tree ensemble model's performance on the given dataset and then shortlisted range of max depths and n_estimators which are the mone of the most influential parameters.
# 
# * then we grid searched the shorlisted parameter values and with some other parameters such as max_samples and max_features, these parameters are must to search as they diversify our trees and make them less similar to each other.
# 
# * after we get the results, we then start searching the small parameter space near the optimal parameters obtained from gridsearch, we will usually see increase in accuracy for next few small gridsearchs this way.
# 
# * Check for overfitting and pick that model that has good accuracy with least overfitting...

# ## Stacking

# In[60]:


logreg = LogisticRegression(**{'C': 0.03, 'l1_ratio': 0, 'penalty': 'elasticnet', 'solver': 'saga'})
svc = SVC(**{'C': 0.5, 'gamma': 0.1, 'kernel': 'rbf'})
knn = KNeighborsClassifier(**{'algorithm': 'ball_tree', 'n_neighbors': 20, 'weights': 'uniform'})
rfc = RandomForestClassifier(**{'max_depth': 5, 'max_features': 0.5, 'max_samples': 0.7, 
                                  'min_samples_split': 5, 'n_estimators': 500})
gradient = GradientBoostingClassifier(**{'learning_rate': 0.055, 'max_depth': 3, 'max_features': 0.4, 
                           'min_samples_split': 2, 'n_estimators': 400, 'subsample': 0.4})
xgb = XGBClassifier(**{'colsample_bylevel': 0.5, 'learning_rate': 0.03, 'max_depth': 6, 
                       'n_estimators': 300, 'reg_lambda': 5, 'subsample': 0.7})

estimators = [('logreg', logreg), ('knn', knn), ('svc', svc), ('rfc', rfc), ('gradient', gradient), 
              ('xgb', xgb)]


stack = StackingClassifier(estimators=estimators,
                           cv=10, n_jobs=-1)


# In[61]:


stack.fit(X_train_fe, y_train)


# In[62]:


y_preds = stack.predict(X_test_fe)


# In[63]:


submission = pd.DataFrame({'PassengerId':test.index, 
              'Survived':y_preds})


# In[64]:


submission.to_csv('submission.csv', index=False)


# In[65]:


pd.read_csv('submission.csv')


# **If you like work, an upvote would be appreciated!!!! :D**
# 
# If you have any suggestions please share in the comments!!
# 
# The best i could get with this notebooks was 0.797, which help me push upto 1090 on leaderboard, which is about Top 5-6%
# 
# If you want to squeeze more out of the dataset used, you can try implementing stacking by yourself and then create a dataframe of predictions made by all the models and train a neural network with dropouts on it, finally when testing turn training = True to for dropout layers and make around 50 - 100 predictions, all these predictions will be different as we turned on the training = True, when we take mean out of all these predictions, we can get an even better result, let me now in the comments if you would want me to demonstrate this technique!!
# 
# Cheers and best of luck for future Kaggling!!

# In[ ]:




