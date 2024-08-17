#!/usr/bin/env python
# coding: utf-8

# ## Objectives
# 
# The main objective of this notebook is to present some hyper parameter tuning techniques in an approachable way to beginners, using both GridSearchCV and Optuna library which is becoming quite popular for performing optimization.. we will optimizing XGBRegressor using optuna and will also check out some cool inbuilt plots that library provides to give deeper insights into the parameter selection and optimization process.
# * Optuna
# * GridSearchCV
# 
# 
# This notebook also serves as a beginner tutorial for implementing a feature engineering class to grid search the feature space using a linear model through a pipeline. The best set of features hence found will then be used to search the best parameters,
# validated using learning curve, for base models to make predictions used for trainig meta model in stacking ensemble. We will
# also look at the diminishing returns in the case of gradient boosting algorithms and how to use this knowledge to improve our 
# stacking results. the objectives are laid down below:
# 
# #### I - To create a Feature Engineering class whose parameters can be grid searched to find the features which helps achieve lowest loss, the class has following features:
# 
#  1. add or not to add features which are combined using multiple features.
#  
#  2. add or not to add polynomials of features most correlated with the target, if yes then how many features to select.
#  
#  3. drop or not to drop features with lowest correlation with target, if yes then what should be the threshold.
#  
#  4. remove or not to remove outliers from features, if yes than what should be the threshold.
#  
#  5. to drop or not drop one hot encoded columns which are too sparse, if yes than what should be the threshold.
#  
# #### II - To fine tune base model parameters with help of learning curves.
# 
# #### III - To understand diminishing returns point for gradient boosting algortihms.
# 
# #### IV - Use stacking to combine the predictions of our models.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, train_test_split, KFold
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

pd.options.display.max_rows=200
pd.set_option('mode.chained_assignment', None)

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

import optuna
from functools import partial

plt.style.use('fivethirtyeight')


# In[2]:


train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[4]:


X_train = train.drop('SalePrice', axis=1)
y_train = np.log1p(train.SalePrice)
X_test = test


# In[5]:


def missing_value_imputation(X):
    
    numerical_features = [feature for feature in X.columns if X[feature].dtype !='O']
        
    ordinal_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                        'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu',
                        'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']

    categorical_features = [feature for feature in X.columns if feature not in ordinal_features and 
                           feature not in numerical_features]
    
    # Numerical feature imputation
        
    # we have selected Neighborhood feature as the feature to group by the data set to impute
    # the other numerical feature on the basis of Neighborhood because it is quite common to have similar types of houses
    # within a Neighborhood for eg:- a Neighborhood may have Expensive yet houses with small floor area, but on the other 
    # hand there might be houses in a Neighborhood where houses have lower prices yet they have larger total sq ft. area.
    # hence this feature may help us to capture better median values for the missing value imputation.
    
    for feature in numerical_features:
        X[feature] = X[feature].fillna(X.groupby('Neighborhood')[feature].transform('median'))
        
    
    # Simplifying Categorical features and combining rare types
    
    X['MSZoning'].replace({'RL':'R', 'RM':'R', 'RP':'R', 'RM':'R', 'I':'O', 'A':'O', 'C (all)':'O', 'FV':'O'}, inplace=True)
    
    X['Alley'].replace({np.nan : 'No', 'Grvl':'Yes', 'Pave':'Yes'}, inplace=True)
    
    X['LandContour'].replace({'Lvl':'Lvl', 'HLS':'Slope', 'Bnk':'Slope', 'Low':'Slope'}, inplace=True)
    
    X['Condition1'].where(X['Condition1'] == 'Norm', 'Other', inplace=True)
    
    X['HouseStyle'].where((X['HouseStyle'] == '1Story') | (X['HouseStyle'] == '2Story') | (X['HouseStyle'] == '1.5Fin'), 
                          'rare', inplace=True)
    
    X['RoofStyle'].where((X['RoofStyle'] == 'Gable') | (X['RoofStyle'] == 'Hip'), 'rare', inplace=True)
    
    X['MasVnrType'].where(X['MasVnrType'] == 'None', 'yes', inplace=True)
    
    X['MasVnrType'].replace({np.nan : 'None'}, inplace=True)
    
    X['Exterior1st'].where((X['Exterior1st'] == 'VinylSd') |  (X['Exterior1st'] == 'HdBoard') | 
                           (X['Exterior1st'] == 'MetalSd') | (X['Exterior1st'] == 'Wd Sdng') | 
                           (X['Exterior1st'] == 'Plywood'), 'rare', inplace=True)
    
    X['PavedDrive'].where(X['PavedDrive'] == 'Y', 'N', inplace=True)
    
    X['Fence'].replace({'MnPrv':'Yes', 'GdPrv':'Yes', 'GdWo':'Yes', 'MnWw':'Yes', np.nan:'No'}, inplace=True)
    
    X['SaleType'].where(X['SaleType'] == 'WD', 'other', inplace=True)
    
    X['SaleCondition'].where((X['SaleCondition'] == 'Normal') | (X['SaleCondition'] == 'Partial') | 
                             (X['SaleCondition'] == 'Abnorml'), 'other', inplace=True)
    
    X['Neighborhood'].replace({'Blmngtn':'rare', 'BrDale':'rare', 'Veenker':'rare', 'NPkVill':'rare', 
                               'Blueste':'rare', 'ClearCr':'rare'}, 
                              inplace=True)
    

    # for missing values still left
    
    for feature in categorical_features:
        val = X[feature].value_counts().index[0]
        X[feature] = X[feature].fillna(val)
    

    # Ordinal feature encoding
    
    # you should not use LabelEncoder() as label encoder will not preserve the order!!

    X['LotShape'].replace({'Reg':1, 'IR1':0, 'IR2':0, 'IR3':0}, inplace=True)
    X['LandSlope'].replace({'Gtl':3, 'Mod':2, 'Sev':1}, inplace=True)
    X['ExterQual'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['ExterCond'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['BsmtQual'].replace({np.nan:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['BsmtCond'].replace({np.nan:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['BsmtExposure'].replace({np.nan:0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4}, inplace=True)
    X['BsmtFinType1'].replace({np.nan:0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)
    X['BsmtFinType2'].replace({np.nan:0, 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6}, inplace=True)
    X['HeatingQC'].replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['KitchenQual'].replace({np.nan:3, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['Functional'].replace({np.nan:8, 'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 
                              'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8}, inplace=True)
    X['FireplaceQu'].replace({np.nan:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['GarageFinish'].replace({np.nan:0, 'Unf':1, 'RFn':2, 'Fin':3}, inplace=True)
    X['GarageQual'].replace({np.nan:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['GarageCond'].replace({np.nan:0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}, inplace=True)
    X['PoolQC'].replace({np.nan:0, 'Fa': 1, 'Gd':2, 'Ex':3}, inplace=True)
    
    categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
    
    for feature in categorical_features:
        value_counts = X[feature].value_counts()
        for val in value_counts[value_counts / len(X) < 0.015].index:
            X[feature].loc[X[feature] == val] = 'rare'
            
    # the problem of many rare features in categorical columns is solved only to some extent with the above practice, 
    # if you experiment with different rare feature threshold, you will find that there will be still few categories,
    # which were categorized as 'rare' in training set, but not in test set, hence the one hot encoded columns will 
    # not be same across the final training and test sets.
    
    return X



class FeatureEngineering(BaseEstimator, TransformerMixin):
    
    # you have to inherit from the BaseEstimator and TransformerMixin classes if you want to grid search the class params
    # using grid search through a pipeline.
    
    def __init__(self, combined_features=True, drop_underlying_features=True, drop_features=True, correlation_threshold=0.05, 
                 polynomial_features=True, outlier_threshold=1.5,  empty_column_dropping_threshold=0.99, 
                 final_correlation_threshold=0.05, test=False):
        
        self.combined_features = combined_features
        self.drop_underlying_features = drop_underlying_features
        self.drop_features = drop_features
        self.correlation_threshold = correlation_threshold
        self.polynomial_features = polynomial_features
        self.polynomial_features_list = []
        self.outlier_threshold = outlier_threshold
        self.empty_column_dropping_threshold = empty_column_dropping_threshold
        self.low_correlation_drop_list = []
        self.one_hot_features_to_drop = []
        self.final_low_correlation_drop_list = []
        self.final_correlation_threshold = final_correlation_threshold
        self.test = test
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        ordinal_features = ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu',
                            'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC']
        
        numerical_features = [feature for feature in X.columns if X[feature].dtype !='O' and feature not in ordinal_features]
        
        categorical_features = [feature for feature in X.columns if feature not in ordinal_features and 
                               feature not in numerical_features]
        
        
        # Converting temporal features to more useful values which is year past till present year.
        X['GarageYrBlt'] = 2020 - X['GarageYrBlt']
        X['YrSold'] = 2020 - X['YrSold']
        X['YearBuilt'] = 2020 - X['YearBuilt']
        X['YearRemodAdd'] = 2020 - X['YearRemodAdd']
        

       
        if self.combined_features:
            
            X['OverallGrade'] =  X['OverallQual'] * X['OverallCond']
            X['ExterGrade'] =  X['ExterQual'] * X['ExterCond']
            X['BsmtGrade'] = X['BsmtQual'] * X['BsmtCond']
            X['BsmtFinType'] = X['BsmtFinType1'] + X['BsmtFinType2']
            X['BsmtFinSF'] = X['BsmtFinSF1'] + X['BsmtFinSF2']
            X['FlrSF'] = X['1stFlrSF'] + X['2ndFlrSF']
            X['BsmtBath'] = X['BsmtFullBath'] + X['BsmtHalfBath']
            X['Bath'] = X['FullBath'] + X['HalfBath']
            X['GarageGrade'] = X['GarageQual'] * X['GarageCond']
            X['Porch'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
            
            numerical_features.extend(['OverallGrade', 'ExterGrade', 'BsmtGrade', 'BsmtFinType', 'BsmtFinSF', 'FlrSF',
                                       'BsmtBath', 'Bath', 'GarageGrade', 'Porch'])
            
            
        
        if self.combined_features and self.drop_underlying_features: # do not wish to drop these features if we have not created
                                                                     # any features using them.
            to_drop = [ 'BsmtFinType1','BsmtFinType2', 'BsmtFinSF1', '1stFlrSF', 'BsmtFullBath',
                       'BsmtHalfBath', 'FullBath', 'HalfBath', 'GarageCond', 
                       'OpenPorchSF', 'BsmtFinSF2', '2ndFlrSF', 'EnclosedPorch', 
                       '3SsnPorch', 'ScreenPorch']
              
            
            X.drop(to_drop, axis=1, inplace=True)
        
            
        if self.drop_features:
            
            to_drop = ['Utilities', 'PoolQC', 'MiscFeature', 'Street', 
                       'Condition2', 'MasVnrType', 'LowQualFinSF', 'Alley', 'MiscVal', 'Fence', 
                   'KitchenAbvGr', 'PoolArea', 'Street', 'RoofMatl', 'Exterior2nd', 'Heating']

            X.drop(to_drop, axis=1, inplace=True)
            
        ordinal_features = [feature for feature in X.columns if X[feature].dtype != 'O' and feature not in numerical_features]
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O' and feature not in ordinal_features]
        categorical_features = [feature for feature in X.columns if X[feature].dtype == 'O']
        # we are refreshing our list of features names in ordinal and numerical lists because we have dropped few features
        # from ordinal and numerical feature lists.
        
        
            
        if self.polynomial_features:
            
            if len(X) > 800 and not self.test:
                y = train['SalePrice'].iloc[X.index].to_frame()
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                self.polynomial_features_list = list(saleprice.sort_values(ascending=False).index[1:self.polynomial_features + 2])   
                
                for feature in self.polynomial_features_list:
                    for i in [2, 3]:
                        X[f'{feature} - sq{i}'] = X[feature] ** i
                        
            else: # test sets should add those polynomial features which were found worthy while going through training set.
                for feature in self.polynomial_features_list:
                    for i in [2, 3]:
                        X[f'{feature} - sq{i}'] = X[feature] ** i
                        
                        
        
            
        if self.outlier_threshold:
            
            numerical_features = [feature for feature in X.columns   
                              if feature not in ordinal_features and X[feature].dtype != 'O' ]
            # again we have to refresh the list because we might have generated a lot of numerical features if the execution
            # went through the polynomial feature "if statement" above.
            
            for feature in numerical_features:
                unique_vals = X[feature].nunique()
                # if we look at the data set we can observe that generally numerical features with less than 10 unique values
                # do not have any outliers, what they have is majority of only one value across instances, so it will be better
                # and all the other distinct values are quite close magnitude wise to the value in majority, so i am not
                # considering such features for outlier removal.
                if unique_vals > 10:
                    q1 = np.percentile(X[feature], 25)
                    q3 = np.percentile(X[feature], 75)
                    iqr = q3 - q1 
                    if not iqr:  # will be executed if iqr = 0, to prevent creation of constant data columns
                        iqr = 1  
                    cut_off = iqr * self.outlier_threshold
                    lower, upper = q1 - cut_off, q3 + cut_off
                    X[feature].where(~(X[feature] > upper), upper, inplace=True)
                    X[feature].where(~(X[feature] < lower), lower, inplace=True)
                    
                    
            
        if self.correlation_threshold:
            
            if len(X) > 800 and not self.test:             
                
                # cannot allow test set to pass through this code block, we will drop those columns in test set too which 
                # were found less correlated with target in training set
                
                self.low_correlation_drop_list = []            
                y = train['SalePrice'].iloc[X.index].to_frame() 
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                corr_dict = saleprice.sort_values(ascending=False).to_frame().to_dict()['SalePrice']

                for key, value in corr_dict.items():
                    if value < self.correlation_threshold and value > - self.correlation_threshold:
                        self.low_correlation_drop_list.append(key)
            
                X = X.drop(self.low_correlation_drop_list, axis=1)
            
            else:
                X = X.drop(self.low_correlation_drop_list, axis=1)
                
                
        numerical_features = [feature for feature in X.columns if X[feature].dtype != 'O']  
        # now we do not have to separate ordinal features from numerical features
        # all numeric types go for power transformation!!
        
        categorical_features = [feature for feature in X.columns if feature not in numerical_features]
        
        X.replace({0 : 1e-5}, inplace=True) # values for box-cox should be strictly positive.
        X_index = X.index
        
        num_pipeline = Pipeline([('scale', MinMaxScaler(feature_range=(1e-5, 1))), 
                                 ('power_transform', PowerTransformer(method='box-cox'))])
       
        transformer = ColumnTransformer([('num_pipeline', num_pipeline, numerical_features)])
        
        numerical_features_transformed = transformer.fit_transform(X)
        
        numerical_df = pd.DataFrame(numerical_features_transformed, columns=numerical_features, index=X.index)   
        
        X = pd.concat([numerical_df, X.loc[:, categorical_features]], axis=1)
        
        if not self.test:
            X_total = pd.concat([X, X_train_imputed.loc[:, categorical_features].loc[~(X_train_imputed.index.isin(X.index))]])
            X_total = pd.get_dummies(X_total)
            one_hot_features = [feature for feature in X_total.columns if feature not in numerical_features]
        
        # while cross validation it might be the case that some rare categories in categorical features might get 
        # concentrated in just the training set of the cross validation and test fold might not have even a single occurence
        # this will cause the test and training sets to be of different column sizes after one hot encoding, to circumvent 
        # this issue i had to use only the categorical portion of those rows in main training set which are not present in the
        # cross val training set. This is not the best way to avoid the problem and is somewhat controversial as test should not
        # be exposed to the training set. any suggestions with regards to this problem will be highly appreciated. 
            
            
        else: 
            X = pd.get_dummies(X)
        
        
        if self.empty_column_dropping_threshold:
            if len(X) > 800 and not self.test:
                self.one_hot_features_to_drop = [] 
                for feature in one_hot_features:
                    zero_count = X_total[feature].value_counts()[0] / len(X_total)

                    if zero_count > self.empty_column_dropping_threshold:
                        self.one_hot_features_to_drop.append(feature)
                        X_total.drop(feature, axis=1, inplace=True)
                            
        
                
            elif not self.test: # exclusively for test set of cross validation...
                X_total.drop(self.one_hot_features_to_drop, axis=1, inplace=True)
            
            
                
            else: # for the final test set
                X.drop(self.one_hot_features_to_drop, axis=1, inplace=True)
            
        if not self.test:
            X = X_total.dropna() # drop the rows added for one hot encoding, those rows have nan values in numerical columns 
        
            
        if self.final_correlation_threshold:
            
            # lets just let the model decide if it finds dropping any more features less correlated with target useful or not!!
            
            if len(X) > 800 and not self.test:                  
                                                               
                self.final_low_correlation_drop_list = []           
                y = train['SalePrice'].iloc[X.index].to_frame() 
                saleprice = pd.concat([X, y], axis=1).corr()['SalePrice']
                corr_dict = saleprice.sort_values(ascending=False).to_frame().to_dict()['SalePrice']

                for key, value in corr_dict.items():
                    if value < self.final_correlation_threshold and value > - self.final_correlation_threshold:
                        self.final_low_correlation_drop_list.append(key)
            
                X = X.drop(self.final_low_correlation_drop_list, axis=1)
            
            else:
                X = X.drop(self.final_low_correlation_drop_list, axis=1)
        
        # finally, we can also add more functionality, such as dimensionality reduction using pca, lle.. but for the
        # present dataset these were not useful so i did not add them, but there will be occasions where dimensionality
        # reduction can help, but searching feature space will take a lot more time, if it does, than it would be better to 
        # find the best feature set and then see if reducing dimensions just only reduces training time or if it also helps
        # boost score(which it generally does not!!)

        return X


# ## Feature Space Search

# In[6]:


X_train_imputed = missing_value_imputation(X_train.copy())


# In[7]:


param_grid = {'feature_engineering__combined_features':[True],
              'feature_engineering__drop_features':[True],
              'feature_engineering__drop_underlying_features':[True],
              'feature_engineering__polynomial_features':[9],
              'feature_engineering__correlation_threshold': [0.07],
              'feature_engineering__outlier_threshold':[1.5], 
              'feature_engineering__empty_column_dropping_threshold':[0.99],
              'feature_engineering__final_correlation_threshold':[False]
              }
#these are the parameters that i found to be the best


# In[8]:


linreg = Pipeline([('feature_engineering', FeatureEngineering()),
                   ('linreg', Ridge())])

grid = GridSearchCV(linreg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid.fit(X_train_imputed.copy(), y_train)


# In[9]:


grid.best_params_


# ## Final Data set creation

# In[10]:


fe = FeatureEngineering(**{'combined_features': True,
                             'correlation_threshold': 0.07,
                             'drop_features': True,
                             'drop_underlying_features': True,
                             'empty_column_dropping_threshold': 0.99,
                             'final_correlation_threshold':False,
                             'outlier_threshold': 1.5,
                             'polynomial_features': 9})


# In[11]:


X_train_imputed = missing_value_imputation(X_train.copy())
X_test_imputed = missing_value_imputation(X_test.copy())


# In[12]:


X_train_imputed.shape, X_test_imputed.shape


# In[13]:


X_train_fe = fe.transform(X_train_imputed.copy())


# In[14]:


X_train_fe.shape


# In[15]:


fe.test = True


# In[16]:


X_test_fe = fe.transform(X_test_imputed.copy())


# In[17]:


X_test_fe.shape


# In[18]:


fe.test = False


# In[19]:


corr_sale = pd.concat([X_train_fe, y_train], axis=1).corr()['SalePrice']


# In[20]:


corr_sale.sort_values(ascending=False)


# In[21]:


X_train_fe.columns == X_test_fe.columns


# ## Model Selection

# In[22]:


def results(cv_results_, n):
    df = pd.DataFrame(cv_results_)[['params', 'mean_test_score']].nlargest(n, columns='mean_test_score')
    for i in range(len(df)):
        print(f'{df.iloc[i, 0]} : {df.iloc[i, 1]}')


# In[23]:


def parameter_plot(model, X, y, n_estimators=[100, 200, 300, 400, 600, 900, 1300], hyper_param=None, **kwargs):
    param_name, param_vals = hyper_param
    param_grid = {'n_estimators':n_estimators,
                  f'{param_name}':param_vals}
    
    grid = GridSearchCV(model(**kwargs), param_grid, 
                        cv=RepeatedKFold(n_splits=8, n_repeats=1, random_state=42), 
                        scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    grid.fit(X, y)
    results = pd.DataFrame(grid.cv_results_)['mean_test_score'].values
    results = results.reshape(len(param_vals), len(n_estimators))
    
    plt.figure(figsize=(15, 9))
    for i in range(1, len(param_vals) + 1):
        plt.plot(n_estimators, results[i-1], label=f'{param_name} - {param_vals[i-1]}')
      
    plt.legend()
    plt.show()


# In[24]:


def learning_curve_plotter(Model, X, y, params_1, params_2, step=50):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 7))
    for i, (name, params) in enumerate([params_1, params_2]):
        train_score = []
        val_score = []
        for j in range(100, len(X_train), step):
            model = Model(**params).fit(X_train[:j], y_train[:j])
            y_train_preds = model.predict(X_train[:j])
            y_test_preds = model.predict(X_test)
            train_score.append(np.sqrt(mean_squared_error(y_train[:j], y_train_preds)))
            val_score.append(np.sqrt(mean_squared_error(y_test, y_test_preds)))
            
        ax[i].plot(train_score, 'r-', label='Training error')
        ax[i].plot(val_score, 'b-', label='Validation error')
        ax[i].set_title(f'{name}', fontsize=18)
        ax[i].set_xlabel('Training set size', fontsize=15, labelpad=10)
        ax[i].set_ylabel('Root mean squared log error', fontsize=15, labelpad=10)
        ax[i].legend()
            
    plt.show()


# ### Ridge

# In[25]:


param_grid_ridge = {'alpha':0.1 * np.arange(1, 70)}


# In[26]:


grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)                  


# In[27]:


grid_ridge.fit(X_train_fe, y_train)


# In[28]:


results(grid_ridge.cv_results_, n=40)


# In[29]:


best_params_ridge = {'alpha': 3.8000000000000003}


# ### Lasso

# In[30]:


param_grid_lasso = {'alpha': 0.0001 * np.arange(1, 100)}


# In[31]:


grid_lasso = GridSearchCV(Lasso(), param_grid_lasso, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[32]:


grid_lasso.fit(X_train_fe, y_train)


# In[33]:


results(grid_lasso.cv_results_, n=40)


# In[34]:


best_params_lasso = {'alpha': 0.0002}


# ### Elastic

# In[35]:


param_grid_elastic = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
                      'l1_ratio':0.01 * np.arange(10)}


# In[36]:


grid_elastic = GridSearchCV(ElasticNet(), param_grid_elastic, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[37]:


grid_elastic.fit(X_train_fe, y_train)


# In[38]:


results(grid_elastic.cv_results_, n=40)


# In[39]:


best_params_elastic = {'alpha': 0.001, 'l1_ratio': 0.09} 


# ### SVR

# In[40]:


param_grid_svr = {'kernel': ['rbf'],
                  'degree':[1], 
                  'epsilon':[0.001, 0.01, 0.008, 0.013],
                  'C':[0.1, 0.5, 1,  20, 25],
                  'gamma':[0.0004, 0.0006, 0.0007, 0.0008]}


# In[41]:


grid_svr = GridSearchCV(SVR(), param_grid_svr, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[42]:


grid_svr.fit(X_train_fe, y_train)


# In[43]:


results(grid_svr.cv_results_, n=60)


# In[44]:


regularized_params = ('Regularized Model', {'C': 1, 'degree': 1, 'epsilon': 0.013, 'gamma': 0.0008, 'kernel': 'rbf'} )
best_params = ('Best Model from Grid Search', {'C': 25, 'degree': 1, 'epsilon': 0.008, 'gamma': 0.0004, 'kernel': 'rbf'})
learning_curve_plotter(SVR, X_train_fe, y_train, regularized_params, best_params)


# Best model looks good, the difference between training loss and validation loss for both of them is equal, plus regularized model is underfitting the data, so lets go with The best model!!

# In[45]:


best_params_svr = {'C': 25, 'degree': 1, 'epsilon': 0.008, 'gamma': 0.0004, 'kernel': 'rbf'}


# ### ExtraTreesRegressor

# create a regularized model with params such as max depth, max features and max samples, then start tweeking the estimators.

# In[46]:


parameter_plot(ExtraTreesRegressor, X_train_fe, y_train, hyper_param=('max_depth', [3, 5, 7, 11, 13, 15, None]))


# In[47]:


param_grid_extra = {'max_depth': [8, 11], 
                    'max_samples':[0.6, 0.8, None],
                    'max_features':[0.5, 0.8, None],
                    'n_estimators': [200, 600, 1000]}


# In[48]:


grid_extra = GridSearchCV(ExtraTreesRegressor(), param_grid_extra, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[49]:


grid_extra.fit(X_train_fe, y_train)


# In[50]:


results(grid_extra.cv_results_, n=40)


# In[51]:


regularized_params = ('Regularized Model', {'max_depth': 8, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 200} )
best_params = ('Best Model from Grid Search', {'max_depth': 11, 'max_features': 0.5, 'max_samples': None, 'n_estimators': 1000})
learning_curve_plotter(ExtraTreesRegressor, X_train_fe, y_train, regularized_params, best_params)


# This looks tricky, both models are horribly overfitting the training data, regularized model is a little better but the difference between the too model is not great, we will have to test both of them to find out which one works the best.

# In[52]:


best_params_extra = {'max_depth': 8, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 200}


# ### RandomForestRegressor

# In[53]:


parameter_plot(RandomForestRegressor, X_train_fe, y_train, hyper_param=('max_depth', [3, 5, 7, 11, 13, 15, None]))


# In[54]:


param_grid_random = {'max_depth': [8, 11],
                     'max_samples':[0.6, 0.8, None], 
                     'max_features':[0.5, 0.7, None], 
                    'n_estimators': [200, 500, 1000]}


# In[55]:


grid_random = GridSearchCV(RandomForestRegressor(), param_grid_random, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[56]:


grid_random.fit(X_train_fe, y_train)


# In[57]:


results(grid_random.cv_results_, n=40)


# In[58]:


regularized_params = ('Regularized Model', {'max_depth': 8, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 1000} )
best_params = ('Best Model from Grid Search', {'max_depth': 11, 'max_features': 0.5, 'max_samples': None, 'n_estimators': 500})
learning_curve_plotter(RandomForestRegressor, X_train_fe, y_train, regularized_params, best_params)


# RandomForestRegressor has the same problem with the ExtraTreesRegressor.

# In[59]:


best_params_random = {'max_depth': 8, 'max_features': 0.5, 'max_samples': 0.8, 'n_estimators': 1000}


# ### Finding the diminishing returns point
# 
# For a given data set there will be a point for a given a set of parameters after which gradient boosted algorithms will stop
# experiencing substantial amount of decrease in loss with increase in number of estimators, if you still train the model after 
# that point you may experience some decrease in loss but the model will overfit the training data and you will not achieve
# generalized result in final test set. 
# 
# in the function below we will first grid search the learning rate with default n_estimators as provided in function params.
# then we will use the best learning rate hence found to find the best max_depth param.
# 
# we have already set **subsample = 0.5**, this is because for gradient boosting having subsample from 0.4 - 0.6 is usually is always
# better than no subsampling.
# 
# this will give us a idea about gradient boosting as applied to our present dataset and will allow us to avoid overfitting
# and achieve the best possible loss before we hit the point of diminishing returns

# ### GradientBoostingRegressor

# In[60]:


parameter_plot(GradientBoostingRegressor, X_train_fe, y_train, hyper_param=('max_depth', [3, 4, 5, 6, 7]))


# from the plot above we can see that the higher learning rates reach the point of diminishing returns quickly and after that as observed in the case of 0.03 and 0.05, loss starts to increase. 0.1 is behaving erratically!!
# so the learning rate for our model should be searched in range 0.015 - 0.025 and i would search with trees ranging from 350 - 650

# Once you decide on what learning rate to choose, pass it to the function and it will be get caught by kwargs and passed to model in gridsearch.  

# In[61]:


parameter_plot(GradientBoostingRegressor, X_train_fe, y_train, 
               n_estimators=[100, 250, 400, 600, 1000],
               hyper_param=('learning_rate', [0.01, 0.02, 0.04, 0.07, 0.1]), 
               max_depth=3)


# Grid searching depth 3 and 4 would be best...

# The above exercise was done so that we could understand our gradient boosting models dealing with the data set at hand... now we can narrow down our search with facing least overfitting

# In[62]:


param_grid_gradient = {'n_estimators':[250, 400, 600],
                       'learning_rate':[0.035, 0.05],
                       'subsample':[0.5, 0.7],
                       'max_depth':[3],
                       'max_features':[0.5, 0.7]
                       }


# In[63]:


grid_gradient = GridSearchCV(GradientBoostingRegressor(), param_grid_gradient, 
                        cv=RepeatedKFold(n_splits=10, n_repeats=2, random_state=42),
                              scoring='neg_root_mean_squared_error', verbose=2, n_jobs=-1)


# In[64]:


grid_gradient.fit(X_train_fe, y_train)


# In[65]:


results(grid_gradient.cv_results_, n=40)


# In[66]:


regularized_params = ('Regularized Model', {'learning_rate': 0.035, 'max_depth': 3, 'max_features': 0.5, 
                                            'n_estimators': 400, 'subsample': 0.7} )
best_params = ('Best Model from Grid Search', {'learning_rate': 0.035, 'max_depth': 3, 'max_features': 0.5,
                                               'n_estimators': 600, 'subsample': 0.5})
learning_curve_plotter(GradientBoostingRegressor, X_train_fe, y_train, regularized_params, best_params)


# Again, overfitting is present in both models, but in regularized model it is not as severe as in the Best model.

# In[67]:


best_params_gradient = {'learning_rate': 0.035, 'max_depth': 3, 'max_features': 0.5, 
                                            'n_estimators': 400, 'subsample': 0.7}


# ### XGBRegressor

# To work with Optuna you have to follow the following guideline:
# * Make a objective function, this function will return the value wish to optimize.. in our case it will be the RMSLE loss.
# * create a **study** using **`optuna.create_study(direction='minimize')`**, will use direction=minimize because we wish to minimize the loss which is RMSLE. 
# * Finally enjoy looking at optuna optimizing your loss!!
# 
# Thats is it, optuna makes it so simple to optimize the parameters.

# In[68]:


def objective(trial, X, y):
    
    params = {'n_estimators':2000,
              # trail.suggest_unifrom() allows to pick out any value between the given range, values will be continuous and
              # not just integers.
              'learning_rate':trial.suggest_uniform('learning_rate', 0.005, 0.01),
              
              # trial.suggest_categorical() allows only the passed categorical values to be suggested.
              'subsample':trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
              
              # trial.suggest_int() will suggest integer values within the integer range. 
              'max_depth':trial.suggest_int('max_depth', 3, 11),
              
              'colsample_bylevel':trial.suggest_categorical('colsample_bylevel', [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
              
              # trail.suggest_loguniform() is used when the range of values have different scales.
              'reg_lambda':trial.suggest_loguniform('reg_lambda', 1e-3, 100),
              'reg_alpha':trial.suggest_loguniform('reg_alpha', 1e-3, 100),
              'n_jobs':-1}
    
    model = XGBRegressor(**params)
    
    split = KFold(n_splits=5)
    train_scores = []
    test_scores = []
    for train_idx, val_idx in split.split(X_train_fe):
        X_tr = X_train_fe.iloc[train_idx]
        X_val = X_train_fe.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric=['rmse'],
                  early_stopping_rounds=30, verbose=0,
                  # optuna allows us to pass pruning callback to xgboost callbacks, so any trial which does not seem to be 
                  # better or not qualify a given threshold of loss reduciton after some iterations will get pruned, that is
                  # stopped in between hence saving time, we will see it in action below.
                  callbacks=[optuna.integration.XGBoostPruningCallback(trial, observation_key="validation_0-rmse")]
                 )
    
        train_score = np.round(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))), 4)
        test_score = np.round(np.sqrt(mean_squared_error(y_val, model.predict(X_val))), 4)
        train_scores.append(train_score)
        test_scores.append(test_score)
        
    
    print(f'train score : {train_scores}')
    print(f'test score : {test_scores}')
    train_score = np.round(np.mean(train_scores), 4)
    test_score = np.round(np.mean(test_scores), 4)
    
    print(f'TRAIN RMSE : {train_score} || TEST RMSE : {test_score}')
    
    # you can make this function as bespoke as possible... you can return any kind of modified value using the return function
    # optuna will try to optimize it!!
    
    return test_score


# In[69]:


optimize = partial(objective, X=X_train_fe, y=y_train)

study = optuna.create_study(direction='minimize')
study.optimize(optimize, n_trials=100) # we have passed 300 trials which are like steps optuna can take to get to the 
                                       # optimized value.


# ### Optuna plots

# In[70]:


optuna.visualization.plot_optimization_history(study)


# This plot lets us visualize a summary of the trials which optuna went through.

# In[71]:


optuna.visualization.plot_slice(study)


# Using this plot we can easily visualize which parameters range optuna found useful for optimization.. in case of learning_rate we can see that optuna concentrates more on the right half which is evident from the cluster of dots in the right half of the plot.

# ### Fine tuning 'gamma'

# gamma governs the minimum impurity reduction required to split the node. It is 0 by default which will make the algorithm keep on splitting nodes arbitrarily even with miniscule reduction in impurity. This cause birth of leaves that are fitted on the noise of training data... what we want to do is restrict this and improve generalization. lets fine tune gamma using a plot.

# In[72]:


plt.style.use('fivethirtyeight')
training_scores = []
validation_scores = []
test_score_matrix = []
params = study.best_params
params['n_estimators'] = 10000
params['n_jobs'] = -1
for gamma in 0.001*np.arange(0, 100, 1):
    params['gamma'] = gamma
    model = XGBRegressor(**params)
    train_scores = []
    test_scores = []
    split = KFold(n_splits=5)
    for train_idx, val_idx in split.split(X_train_fe):
        X_tr = X_train_fe.iloc[train_idx]
        X_val = X_train_fe.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  eval_metric=['rmse'],
                  early_stopping_rounds=30, verbose=0)
                  #callbacks=[optuna.integration.XGBoostPruningCallback(trial, observation_key="validation_0-rmse")])
        
        train_score = np.round(np.sqrt(mean_squared_error(y_tr, model.predict(X_tr))), 4)
        test_score = np.round(np.sqrt(mean_squared_error(y_val, model.predict(X_val))), 4)
        train_scores.append(train_score)
        test_scores.append(test_score)
    test_score_matrix.append(test_scores)
 
    print(f'working on gamma : {gamma}')
    print(f'train scores : {train_scores}')
    print(f'test scores : {test_scores}')
    print(f'Mean OOF RMSLE : {np.mean(test_scores)}')
    train_score = np.mean(train_scores)
    test_score = np.mean(test_scores)
    
    training_scores.append(train_score)
    validation_scores.append(test_score)
    
plt.figure(figsize=(15, 8))
plt.plot(training_scores, 'r-', label='Train-Gamma-Effect')
plt.plot(validation_scores, 'b-', label='Test-Gamma-Effect')
plt.legend()
plt.grid()
plt.show()


# We can see that as the gamma value increases the overfitting decreases.... if the dataset had many more instances, we could have seen that increasing gamma initially improves loss and then it agains starts to increase... using gamma is important as it prunes leafs that are causing model to fit on the noise of the training data set.

# In[73]:


params = study.best_params
params['n_estimators'] = 1300
params['gamma'] = 0.05
best_params_xgb = params


# ## Stacking

# In[74]:


def model_evaluation(model):
    cv = RepeatedKFold(n_splits=8, n_repeats=2)
    scores = cross_val_score(model, X_train_fe, y_train, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error')
    return np.sqrt(np.abs(scores))


# In[75]:


lasso = Lasso(**best_params_lasso)
ridge = Ridge(**best_params_ridge)
elastic = ElasticNet(**best_params_elastic)
svr = SVR(**best_params_svr)
extra = ExtraTreesRegressor(**best_params_extra)
random = RandomForestRegressor(**best_params_random)
gradient = GradientBoostingRegressor(**best_params_gradient)
xgb = XGBRegressor(**best_params_xgb)



model_list1 = [('lasso', lasso),   
          ('svr', svr),
          ('extra', extra),
          ('elastic', elastic),
          ('ridge', ridge), 
          ('random', random),
          ('gradient', gradient),
          ('xgb', xgb),
           ]


# In[76]:


stack_reg = StackingRegressor(estimators=model_list1, cv=8, n_jobs=-1)


# In[77]:


models = {'lasso': lasso, 
           'ridge': ridge, 
           'elastic': elastic, 
           'svr': svr, 
           'extra' : extra, 
           'random':random, 
           'gradient': gradient, 
           'xgb': xgb, 
           'stack_reg':stack_reg
         }


scores = []
names = []

for name, model in models.items():
    score = model_evaluation(model)
    scores.append(score)
    names.append(name)
    
plt.figure(figsize=(16, 8))    
plt.boxplot(scores, labels=names, showmeans=True)
plt.show()


# We can appreciate how the stacking has its spread lowest among all the other models!!

# In[78]:


stack_reg.fit(X_train_fe, y_train)


# In[79]:


submission = pd.DataFrame(stack_reg.predict(X_test_fe), columns=['SalePrice'])


# In[80]:


submission = np.expm1(submission)


# In[81]:


submission.insert(loc=0, column='id', value=[i for i in range(1461, 2920)])


# In[82]:


submission.to_csv('submission.csv', index=False)


# In[83]:


pd.read_csv('submission.csv')


# **If you found the kernel useful, upvote!**
# 
# **If you have forked it but not upvoted yet, show support and upvote!! :)**
# 
# **Please leave in the comments any suggestions and constructive criticism!!**

# In[ ]:





# In[ ]:




