#!/usr/bin/env python
# coding: utf-8

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Notebook resume</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# Throughout this notebook I will first make an initial features selection with Shap, permutation feature importance, to see if there is any difference regards to feature selection, and then try to make a feature engineering process and see what results sheds. The Shap  technique helped me in the previous competition. I hope you like it.
#     <br><br>

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Table of contents</div>
# 
# * [1-Libraries and data loading](#section-one)
# * [2-Folds creation](#section-two)
# * [3-Initial feature selection (Baseline)](#section-three)
#     - [3.1-Shap study](#subsection-three-one)
#     - [3.2-Permutation feature importance](#subsection-three-three)
# * [4-Feature engineering](#section-four)
#     - [4.1-PCA decomposition](#subsection-four-one)
#     - [4.2-SVD decomposition](#subsection-four-two)
#     - [4.3-Polynomial features](#subsection-four-three)
#     - [4.4-Stats features](#subsection-four-four)
# * [5-Final feature selection](#section-five)
# * [6-Final thoughts](#section-six)

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 1 | Libraries and data loading</div>

# In[1]:


import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, decomposition, model_selection
from tsfresh.feature_extraction import feature_calculators as fc

import eli5
from eli5.sklearn import PermutationImportance
import shap
import gc

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[4]:


train = pd.read_csv('../input/tabular-playground-series-nov-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-nov-2021/test.csv')


# In[5]:


features = [feature for feature in train.columns if feature not in ('id','kfold', 'target')]


# In[6]:


len(features)


# <a id="section-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 2 | Folds creation</div>

# In[7]:


kf = model_selection.KFold(n_splits=5) 
train['kfold'] = -1
def kfold (df):
    for fold, (train_idx, test_idx) in enumerate(kf.split(X = df)):
        df.loc[test_idx, 'kfold'] = fold
        
    return df


# In[8]:


train = kfold(train)


# <a id="section-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 3 | Initial feature selection with Shap and Permutation (Baseline)</div>
# 
# <a id="subsection-three-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#303030;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%; margin-left:20px"> 3.1 | Shap study</div>

# In[9]:


models = {
          'B-XGB': xgb.XGBClassifier(n_estimators=100, objective = 'reg:squarederror', gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor',),
          }


# In[15]:


def shap_feature_selection(df, subset, features, target, folds, models):
    
    """
    Performs feature selection using SHAP values with cross validation.
    Arguments:
        - df (dataframe): pandas dataframe.
        - subset (float): the fraction of the input DataFrame to use for training and testing. Must be a value between 0 and 1.
        - features (list): a list of column names to use as features for the model.
        - target (str): the name of the column to use as the target variable.
        - folds (int): the number of cross-validation folds to use when evaluating the model.
        - models: a scikit-learn model object to fit the data.
    """
    
    list_shap_values = []
    list_valid_dfs = []
    
    df = df.sample(frac=subset, random_state=42).reset_index()

    for fold in range(folds):
        
        print(f' Fold number = {fold}')
        
        X_train = df[df.kfold != fold].reset_index(drop=True)
        X_valid = df[df.kfold == fold].reset_index(drop=True)

        y_train = X_train[target]
        y_valid = X_valid[target]
               
        X_train = X_train[features]
        X_valid = X_valid[features]

        model = models.fit(X_train, y_train)
            
        explainer = shap.TreeExplainer(model)
        shap_value = explainer.shap_values(X_valid)
        list_shap_values.append(shap_value)
        list_valid_dfs.append(X_valid)
        
    feature_name = X_valid[features].columns
    shap_values = np.concatenate(list_shap_values, axis=0)
    sv = np.abs(shap_values).mean(0)
    
    X_valid_dfs = pd.concat(list_valid_dfs, axis=0)
    
    importance_df = pd.DataFrame({"feature": feature_name, "shap_values": sv})
    
    return importance_df, shap_values, explainer, X_valid_dfs


# In[17]:


importance_df, shap_values, explainer, X_valid = shap_feature_selection(train, 0.5, features, 'target', 5, models['B-XGB'])


# In[18]:


importance_df[importance_df.shap_values > 0.01].sort_values(by='shap_values',ascending=False)


# In[19]:


shaped_features = importance_df[importance_df.shap_values > 0.01]['feature'].to_list()


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# I'm only going to select only the features with a contribution greater than 0.01 to use later.</p>

# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:10px">
# SHAP Summary Plot </p>
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">
#    With this plot we can visualize the overall impact of the features across multiple instances. For that reason the result of the shapley study it's much more reliable to establish what features are the most relevant in comparassion with for example the feature importance of a tree base model.    
#     <br><br>

# In[21]:


shap.summary_plot(shap_values, X_valid)


# <div class="alert alert-info" style="border-radius:5px; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# This graph tells us which are the most important characteristics and their range of effects on the data set. The features are sorted by rank based on their impact on the target value. This technique helped me a lot in the previous competition, on the one hand in the training times of the models, and I also think that to avoid some overfitting by reducing the number of features and making a more generalizable model.</p>

# <a id="subsection-three-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#303030;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%; margin-left:20px"> 3.2 | Permutation feature importance</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:40px">
# This is another technique wich consist in randomly shuffle a single column of the validation data, leaving the target and all other columns in place, and how this shuffle affect the accuracy of predictions in that now-shuffled data. There is a mini Kaggle course where you can learn more about it. I share the link for that down below.
# https://www.kaggle.com/learn/machine-learning-explainability    <br><br>

# In[22]:


def eli5_feature_selection(df, subset, features, target, folds, models, random_state=42):
       
    """
    Performs feature selection using permutation importance with cross validation.
    Arguments:
        - df (dataframe): pandas dataframe.
        - subset (float): the fraction of the input DataFrame to use for training and testing. Must be a value between 0 and 1.
        - features (list): a list of column names to use as features for the model.
        - target (str): the name of the column to use as the target variable.
        - folds (int): the number of cross-validation folds to use when evaluating the model.
        - models: a scikit-learn model object to fit the data.
        - random_state (int): value to use as the random seed for reproducibility (default: 42).
    """
    
    list_results = []
    
    df = df.sample(frac=subset, random_state=random_state).reset_index()

    for fold in range(folds):
        print(f' Fold number = {fold}')
        X_train = df[df.kfold != fold].reset_index(drop=True)
        X_valid = df[df.kfold == fold].reset_index(drop=True)

        y_train = X_train[target]
        y_valid = X_valid[target]

        X_train = X_train[features]
        X_valid = X_valid[features]

        model = models.fit(X_train, y_train)
        
        perm = PermutationImportance(model, random_state=random_state).fit(X_valid, y_valid)
        weights = eli5.show_weights(perm, top=len(X_valid.columns), feature_names=X_valid.columns.tolist())
        result = pd.read_html(weights.data)[0]
        eli_features = pd.DataFrame(result)
        eli_features.Weight = eli_features.Weight.str.split(' ').apply(lambda x: x[0]).astype('float32')
        eli_features.set_index('Feature', inplace=True)
        
        list_results.append(eli_features)

    results = pd.concat(list_results, axis=1)
    results = results.mean(1)
    
    return results


# In[24]:


results = eli5_feature_selection(train, 0.5, features, 'target', 5, models['B-XGB'])


# In[29]:


results[results.sort_values(ascending=False) > 0.01].reset_index().rename(columns={0:'Feature Importance'})


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em">   
# <b> Insights: </b> We can see that although the results are not the same, permutation it's quite more restrictive. Perhaps the difference is due to the fact that permutation only uses 50% of the data.</p>

# <a id="section-four"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 4 | Feature engineering</div>
# 
# <a id="subsection-four-one"></a>
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em;  margin-left:20px">
#    <b> 1 - PCA decomposition:</b>
#     PCA or Principal Components Analysis gives us our ideal set of features. It creates a set of principal components that are rank ordered by variance (the first component has higher variance than the second, and so on), uncorrelated, and low in number (we can throw away the lower ranked components as they contain little signal). In this case I just apply the PCA decomposition to the features selected by Shap, not all the set. To more info about the PCA, you can check the Kaggle course, <a href="https://www.kaggle.com/ryanholbrook/principal-component-analysis">[link]</a>
#     <br><br>

# In[30]:


def decompositions_pca(train, test, folds, features, n_components=2, random_state=42):
    
    """
    Performs PCA decomposition with cross validation.
    Arguments:
        - train (dataframe): train dataframe.
        - test (dataframe): test dataframe.
        - folds (int): the number of cross-validation folds to use when evaluating the model.
        - features (list): a list of column names to use as features for the model.
        - n_components (int): number of PCA components.
        - random_state (int): value to use as the random seed for reproducibility (default: 42).
    """
    
    x_valid_pca = []
    x_test_pca = []
    
    for fold in range(folds):
        X_train = train[train.kfold != fold].reset_index(drop=True)
        X_valid = train[train.kfold == fold].reset_index(drop=True)

        X_train = X_train[features]
        X_valid = X_valid[features]

        X_test = test[features].copy()
        
        scl = preprocessing.StandardScaler()
        X_train = scl.fit_transform(X_train.fillna(0))
        X_valid = scl.transform(X_valid.fillna(0))
        X_test = scl.transform(X_test.fillna(0))

        pca = decomposition.PCA(n_components=n_components, random_state=random_state).fit(X_train)
        X_valid_pca = pca.transform(X_valid)
        X_test_pca = pca.transform(X_test)

        x_valid_pca.append(X_valid_pca)
        x_test_pca.append(X_test_pca)
        
        del X_train, X_valid, X_test
        gc.collect()

    train_pca = np.concatenate(x_valid_pca, axis=0)
    component_names_1 = [f"PC_{i+1}" for i in range(train_pca.shape[1])]
    train_pca = pd.DataFrame(train_pca, columns=component_names_1, dtype='float32')
    
    test_pca = np.mean(x_test_pca, axis=0)
    component_names_2 = [f"PC_{i+1}" for i in range(test_pca.shape[1])]
    test_pca = pd.DataFrame(test_pca, columns=component_names_2, dtype='float32')
        
    gc.collect()
    
    return train_pca, test_pca


# In[31]:


train_pca, test_pca = decompositions_pca(train, test, 5, features)


# In[32]:


train = pd.concat([train, train_pca], axis=1)
test = pd.concat([test, test_pca], axis=1)


# <a id="subsection-four-two"></a>
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">
#    <b> 2 - SVD decomposition:</b>
#     Im going to use the SVD or Singular Value Decomposition for dimensionality reduction and also see if helps to denoise the data. In this case Im going to apply the SVD to the whole dataset. You can check the documentation here <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">[link]</a>, or explanations on TDS here <a href="https://towardsdatascience.com/search?q=svd">[link]</a>.
#     <br><br>

# In[33]:


def decompositions_svd(train, test, folds, features, n_components=2, random_state=42):
    
    """
    Performs SVD decomposition with cross validation.
    Arguments:
        - train (dataframe): train dataframe.
        - test (dataframe): test dataframe.
        - folds (int): the number of cross-validation folds to use when evaluating the model.
        - features (list): a list of column names to use as features for the model.
        - n_components (int): number of SVD components.
        - random_state (int): value to use as the random seed for reproducibility (default: 42).
    """
    
    x_valid_svd = []
    x_test_svd = []
    
    for fold in range(folds):
        X_train = train[train.kfold != fold].reset_index(drop=True)
        X_valid = train[train.kfold == fold].reset_index(drop=True)

        X_train = X_train[features]
        X_valid = X_valid[features]

        X_test = test[features].copy()
        
        scl = preprocessing.StandardScaler()
        X_train = scl.fit_transform(X_train.fillna(0))
        X_valid = scl.transform(X_valid.fillna(0))
        X_test = scl.transform(X_test.fillna(0))

        svd = decomposition.TruncatedSVD(n_components=n_components, random_state=random_state).fit(X_train)
        X_valid_svd = svd.transform(X_valid)
        X_test_svd = svd.transform(X_test)

        x_valid_svd.append(X_valid_svd)
        x_test_svd.append(X_test_svd)
        
        del X_train, X_valid, X_test
        gc.collect()

    train_svd = np.concatenate(x_valid_svd, axis=0)
    component_names_1 = [f"SVD_{i+1}" for i in range(train_svd.shape[1])]
    train_svd = pd.DataFrame(train_svd, columns=component_names_1, index=train.index, dtype='float32')
    
    test_svd = np.mean(x_test_svd, axis=0)
    component_names_2 = [f"SVD_{i+1}" for i in range(test_svd.shape[1])]
    test_svd = pd.DataFrame(test_svd, columns=component_names_2, index=test.index, dtype='float32')
    
    gc.collect()
    
    return train_svd, test_svd


# In[34]:


train_svd, test_svd = decompositions_svd(train, test, 5, features)


# In[35]:


train = pd.concat([train, train_svd], axis=1)
test = pd.concat([test, test_svd], axis=1)


# <a id="subsection-four-three"></a>
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em;  margin-left:20px">
#    <b> 3 - Polynomial features:</b>
#     This process generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2]. You can learn more about it here, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">[link].</a>
#   <br><br>

# In[36]:


poly = preprocessing.PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)
train_poly = poly.fit_transform(train[shaped_features])
test_poly = poly.fit_transform(test[shaped_features])

train_poly_df = pd.DataFrame(train_poly, columns=[f"POLY_{i}" for i in range(train_poly.shape[1])])
test_poly_df = pd.DataFrame(test_poly, columns=[f"POLY_{i}" for i in range(test_poly.shape[1])])


# In[37]:


train = pd.concat([train, train_poly_df], axis=1)
test = pd.concat([test, test_poly_df], axis=1)


# <a id="subsection-four-four"></a>
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em;  margin-left:20px">
# <b>4 - Stats features: </b> 
#     <br><br>

# In[38]:


train['mean'] = train[features].mean(axis=1)
train['median'] = train[features].median(axis=1)
train['std'] = train[features].std(axis=1)
train['var'] = train[features].var(axis=1)
train['kurt'] = train[features].kurtosis(axis=1)


# <a id="section-five"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 5 | Final feature selection</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">
# For the final selection of features I'm going to use only Shap, because the Permutation approch it's quite resource demanding. 
# <br><br>

# In[39]:


features = [feature for feature in train.columns if feature not in ('id','kfold', 'target')]


# In[41]:


importance_df, shap_values, explainer, X_valid = shap_feature_selection(train,1., features, 'target', 5, models['B-XGB'])


# In[45]:


importance_df[importance_df.shap_values > 0].sort_values(by='shap_values',ascending=False).head(30)


# <h3>
# SHAP Summary Plot
# </h3>

# In[43]:


shap.summary_plot(shap_values, X_valid)


# <a id="section-six"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#000000;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 6 | Final thoughts</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">
#     We can se how some of the features we created are useful, because provide some information to predict the target. I will be adding more features as they occur to me. If you have any questions, suggestions, or if I make some mistake, please let me know. Happy Kaggling!
#     <br><br>

# In[ ]:




