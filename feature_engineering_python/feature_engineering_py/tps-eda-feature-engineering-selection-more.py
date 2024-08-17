#!/usr/bin/env python
# coding: utf-8

# # <b>1 <span style='color:#3f4d63'>|</span> Introduction</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>1.1 | Table of contents</b></p>
# </div>
# 
# * **<span style = 'color:red'>Exploratory Data Analysis</span>**
#     * **Basic information**
#     * **Distributions**
#         * Categorical Values
#         * Integers Values
#         * Floating Values
#         * Normal Distribution Test for Floating Values
#         * Distribution of dependent variable
#     * **Correlations**
# * **<span style = 'color:red'>Feature Engineering</span>**
#     * **Handling Categorical Features**
#     * **Handling Missing Values**
#     * **Feature Scaling**
#     * **Outlier Detection**
# * **<span style = 'color:red'>Feature Selection</span>**
#     * **Variance Threshold**
#         * Theory
#         * Code
#     * **Information Gain Method**
#         * Theory
#         * Code
#     * **ExtraTree**
#         * Code
# * **<span style = 'color:red'>Model</span>**
#     * **Preparing Test set**
#     * **GroupK Fold**
#     * **Combining Results**
# * **<span style = 'color:red'>Future Work</span>**

# In[1]:


import pandas as pd 
import numpy as np
from scipy.stats import shapiro
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import gc


# In[2]:


trainDF = pd.read_csv('../input/tabular-playground-series-aug-2022/train.csv')
trainDF.drop('id', axis=1, inplace=True)
testDF = pd.read_csv('../input/tabular-playground-series-aug-2022/test.csv')
testDF.drop('id', axis=1, inplace=True)


# In[3]:


trainPseudoDF = trainDF.copy()


# In[4]:


def infoDF(data):
    floatColCounter = 0
    floatCols = []
    intColCounter = 0
    intCols = []
    stringColCounter = 0
    stringCols = []
    print('No of rows-> {}, No of columns-> {}'.format(data.shape[0], data.shape[1]))
    print('          ------------------------          ')
    for column in data.columns:
        if data[column].dtype == int:
            intColCounter += 1
            print('{} dtype -> integer, % of null values-> {}%, No of distinct values-> {}'.format(column, round((data[column].isnull().sum()/data.shape[0])*100, 2), data[column].nunique()))
            intCols.append(column)
            print('          ------------------------          ')
        elif data[column].dtype == float:
            floatColCounter += 1
            print('{} dtype -> float, % of null values-> {}%, No of distinct values-> {}'.format(column, round((data[column].isnull().sum()/data.shape[0])*100, 2), data[column].nunique()))
            floatCols.append(column)
            print('          ------------------------          ')
        else:
            stringColCounter += 1
            print('{} dtype -> string, % of null values-> {}%, No of distinct values-> {}'.format(column, round((data[column].isnull().sum()/data.shape[0])*100, 2), data[column].nunique()))
            stringCols.append(column)
            print('          ------------------------          ')
            
    print('No of integer column-> {}, No of floating column-> {}, No of string or object columns-> {}'.format(intColCounter, floatColCounter, stringColCounter))
    print('          ------------------------          ')
    print('% of Null/Missing Values in training data-> {}%'.format(round((data.isnull().sum().sum()/(data.shape[0]*trainDF.shape[1]))*100, 2)))
    print('          ------------------------          ')
    return intCols, floatCols, stringCols


# # <b>2 <span style='color:#3f4d63'>|</span> Exploratory Data Analysis</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.1 | Basic information</b></p>
# </div>
# 
# To start with, we're gonna take a brief view on the dataset given in order to get some basic information about it: 
# 
# * We're gonna show some sample rows and columns of the dataframe.
# * Examine which type of features we've given.
# * Find out whether there are missing values or not.
# * No of distinct values of given feature.

# In[5]:


intColumns, floatColumns, stringColumns = infoDF(trainDF)


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>2.2 | Distributions</b></p>
# </div>
# 
# In this first section, we're gonna focus on analysing every feature's distribution and its related statistical information. To start with, let's plot the distributions in order to determine whether features are distributed normally. 
# > We're gonna distinguish between different categorical features and we can se the distribution of all three categorical feature of train set.

# In[6]:


figure = plt.figure(figsize=(16, 3))
colors = ['red', 'blue', 'green']
for i in range(len(stringColumns)):
    plt.subplot(1, 3, i+1)
    sns.histplot(trainDF[stringColumns[i]], shrink=0.8, color=colors[i])
figure.tight_layout(h_pad=1.0, w_pad=0.5)
plt.suptitle('Distribution of String Column Values')
plt.show()


# > Below is the code for the distribution of the integer feautre and and the graph should show un-even spread as we have seen in basic information that all the integers values conatin 4-5(max) distinct values.

# In[7]:


fig=plt.figure(figsize=(16, 5))
for i, f in enumerate(intColumns):
    plt.style.use('ggplot')
    plt.subplot(2, 4, i+1)
    sns.histplot(trainDF[f])
    plt.title('Feature: {}'.format(f))
    plt.xlabel('')
    
fig.suptitle('Integer Feature distributions',  size=20)
fig.tight_layout()  
plt.show()


# > After integer we look at floating value feature which should show different distribution of values integer value feature as there way more distinct feature and its also clear from graph. Most of the feature seen to be normally distributed.

# In[8]:


fig=plt.figure(figsize=(16, 8))
for i, f in enumerate(floatColumns):
    plt.style.use('ggplot')
    plt.subplot(4, 4, i+1)
    sns.histplot(trainDF[f])
    plt.title('Feature: {}'.format(f))
    plt.xlabel('')
    
fig.suptitle('Float Feature distributions',  size=20)
fig.tight_layout()  
plt.show()


# 📌 **Early insights:**
# 
# * It seems that every `float` feature is distributed normally. By contrast, that's not what happens when talking about `ìnt` features. 
# 
# Thus, let's determine it. We can proceed with different methods. Let's show some of them: 
# 
# **Shapiro-Wilk Test** (Code taken from [TPS Jul 22 Advanced. Author: Torch me](https://www.kaggle.com/code/kartushovdanil/tps-jul-22-advanced-2-sol))
# 
# This test is used to test whether a dataset is distributed normally or not. The null hypothesis is that a sample $$x_1\hspace{0.1cm},\hspace{0.1cm}\cdots\hspace{0.1cm},\hspace{0.1cm}x_n$$ comes from a normally distributed population. It was published in 1965 by Samuel Shapiro and Martin Wilk. **It is considered one of the most powerful tests for normality testing.** The test stadistic will be: 
# 
# $$W = \frac{(\sum_{i=1}^{n}a_{i}x_i)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
# 
# where
# 
# * $x_i$ is the number occupying the i-th position in the sample (with the sample ordered from smallest to largest).
# * $\bar{x}$ is the sample mean. 
# * Variables $a_i$ are calculated this way: 
# 
# $$(a_1, ... , a_n) = \frac{m^T V^{-1}}{(m^T V^{-1}V^{-1}m)^{1/2}} \hspace{2cm}m = (m_1 , ... , m_n)$$
# 
# where $m_1 , ... , m_n$ are the mean values of the ordered statistic, of independent and identically distributed random variables, sampled from normal distributions and $V$ denotes the covariance matrix of that order statistic. **The null hypothesis is rejected if W is too small. The value of W can range from 0 to 1.**

# In[9]:


for col in intColumns+floatColumns:
    stat, p_value = shapiro(trainDF[col])  
    alpha = 0.05
    if p_value > alpha: 
        result = colored('Accepted', 'green')  
    else:
        result = colored('Rejected','red')        
    print('Feature: {}\t Hypothesis: {}'.format(col, result))


# > Finally distribution of dependent vairable and from the first look itself, it seem to be an imbalance in dependen feature which wee need to tackle in model creation.

# In[10]:


sns.histplot(trainDF['failure'])


# It's convenient not to use features that are correlated (hence redundant), when trying to make a proper ML application. Thus, in this section, our main aim will be to analyse the different relationships between each of the features. Thus, we'll be able to determine which features are linearly related.
# 
# 📌 Insights:
# 
# It seems that attribute features are lightly correlated with measurement features.
# When talking about float features, if we recap which features are normally distributed and these features seem to  be correlated.
# Most correlated features are attribute_01 and attribute_00.

# In[11]:


plt.figure(figsize=(12,6))
corr = trainPseudoDF.corr()
matrix = np.triu(corr)
sns.heatmap(corr, mask = matrix, center = 0, cmap = 'vlag').set_title('Correlations')


# # <b>3 <span style='color:#3f4d63'>|</span> Feature Engineering</b>

# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.1 | Handling Categorical Values</b></p>
# </div>
# 
# Firsly we will have glimpse on the values of the categorical feature so that we can come up with a strategy of handling categorical values.

# In[12]:


for i in stringColumns:
    print('Unique Values for {} -> {}'.format(i, trainDF[i].unique()))


# * **Proudct Type:** Product type seem to have 5 distinct values in train and 4 distinct values in test thus it not feasable to use any encoder technique to apply on product type as it will have no information of product code importance while predicting for test set.
# * **Attribute 0:** we can simply fetch number from their values by removing material and use it as final feature and alos same for test set feature as thses values contain in both train and test set.
# * **Attribute 0**: we can simply fetch number from their values by removing material and use it as final feature and alos same for test set feature as thses values contain in both train and test set.

# In[13]:


trainPseudoDF['attribute_1'] = trainPseudoDF['attribute_1'].str.split('_', 1).str[1].astype('int')
trainPseudoDF['attribute_0'] = trainPseudoDF['attribute_0'].str.split('_', 1).str[1].astype('int')


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.2 | Handling Missing Values</b></p>
# </div>
# 
# 📌 **Early insights:**
# 
# * All the null values in float value features. 
# * All the floating value feature are normally distributed
# * **Null Values/ Missing Values**: Because of the above insights replacing the missing values with mean seem to be most easy, less computaional and efficient solution.

# In[14]:


for col in floatColumns:
    if trainPseudoDF[col].isnull().sum():
        trainPseudoDF[col].fillna(trainPseudoDF[col].mean(), inplace=True)


# In[15]:


trainPseudoDF[floatColumns].isnull().sum()


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.3 | Feature Scaling</b></p>
# </div>
# 
# 📌 **Early insights:**
# * As we know, there is a drastic change in range of integer, floating values features and in categorical feature thus it semm feasible to bring all feature value in a single range this scaling feature seems to be a logical steps

# In[16]:


scalerModel = StandardScaler().fit(trainPseudoDF.drop(['failure', 'product_code'], axis=1))
scaledPseudoDF = scalerModel.transform(trainPseudoDF.drop(['failure', 'product_code'], axis=1))
scaledPseudoDF = pd.DataFrame(scaledPseudoDF, columns=trainPseudoDF.drop(['failure', 'product_code'], axis=1).columns)
scaledPseudoDF


# In[17]:


scaledPseudoDF['product_code'] = trainPseudoDF['product_code']
scaledPseudoDF['failure'] = trainPseudoDF['failure']


# * Lets see the correlation after scaling

# In[18]:


plt.figure(figsize=(12,6))
corr = scaledPseudoDF.corr()
matrix = np.triu(corr)
sns.heatmap(corr, mask = matrix, center = 0, cmap = 'vlag').set_title('Correlations')


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>3.4 | Handling outliers</b></p>
# </div>

# In[19]:


tmpDF = pd.DataFrame(data = scaledPseudoDF[intColumns].drop('failure', axis=1))
plt.figure(figsize=(16,4)) 
sns.boxplot(x="variable", y="value", data=pd.melt(tmpDF)).set_title('Boxplot of each feature',size=15)
plt.show()


# * Floating value feature seems to have nore outliers compaired to integer value feature let more to find out about outlieries in floating value features.

# In[20]:


tmpDF = pd.DataFrame(data = scaledPseudoDF[floatColumns])
plt.figure(figsize=(16,4)) 
sns.boxplot(x="variable", y="value", data=pd.melt(tmpDF)).set_title('Boxplot of each feature',size=15)
plt.show()


# In[21]:


for col in floatColumns:
    outlierLen = scaledPseudoDF[(scaledPseudoDF[col] <= -3) | (scaledPseudoDF[col] >= 3)].shape[0]
    outliersPercentage = round((outlierLen/len(scaledPseudoDF[col]))*100,2)
    print('% of outliers in col {} -> {}%'.format(col, outliersPercentage))


# 📌 Insights:
# * There is somehwere around 1% of outliers in almost all the floating value features thus we try to cut this % to half so that our model can have some information about the outliers also when predicting value for test set.

# In[22]:


for col in floatColumns:
    indexes = scaledPseudoDF[(scaledPseudoDF[col] <= -3) | (scaledPseudoDF[col] >= 3)].index
    scaledPseudoDF.drop(indexes, inplace=True)


# In[23]:


tmpDF = pd.DataFrame(data = scaledPseudoDF[floatColumns])
plt.figure(figsize=(16,4)) 
sns.boxplot(x="variable", y="value", data=pd.melt(tmpDF)).set_title('Boxplot of each feature',size=15)
plt.show()


# # <b>4 <span style='color:#3f4d63'>|</span> Feature Selection</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.1 | Variance Threshold</b></p>
# </div>
# 
# * Feature selector that removes all low-variance features.
# 
# * This feature selection algorithm looks only at the features (X), not the desired outputs (y).

# In[24]:


selector = VarianceThreshold(threshold=1)
selector.fit_transform(scaledPseudoDF.drop(['failure', 'product_code'], axis=1))
plt.figure(figsize=(15,10))
sns.barplot(x=selector.variances_, y=scaledPseudoDF.drop(['failure', 'product_code'], axis=1).columns,orient='h' ).set_title('Feature selection with VarianceThreshold',size=15);
plt.xlabel('Variance');
plt.axvline(x=.8, color='r', linestyle='--', label='Threshold')
plt.legend()


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.2 | Information Gain</b></p>
# </div>
# 
# * MI Estimate mutual information for a discrete target variable.
# 
# * Mutual information (MI) between two random variables is a non-negative value, which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
# 
# * The function relies on nonparametric methods based on entropy estimation from k-nearest neighbors distances.
# 
# <b>Inshort<b>
# 
# * A quantity called mutual information measures the amount of information one can obtain from one random variable given another.
# 
# * The mutual information between two random variables X and Y can be stated formally as follows:
# 
# <b>I(X ; Y) = H(X) – H(X | Y)<b>
# Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional entropy for X given Y. The result has the units of bits.

# In[25]:


mutual_info = mutual_info_classif(scaledPseudoDF.drop(['failure', 'product_code'], axis=1),scaledPseudoDF['failure'])
mutual_info = pd.Series(mutual_info)
mutual_info.index = scaledPseudoDF.drop(['failure', 'product_code'], axis=1).columns
mutual_info.sort_values(ascending=False)


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>4.3 | Extra Tree</b></p>
# </div>
# 
# * This technique gives you a score for each feature of your data,the higher the score mor relevant it is

# In[26]:


model = ExtraTreesClassifier()
model.fit(scaledPseudoDF.drop(['failure', 'product_code'], axis=1), scaledPseudoDF['failure'])
plt.figure(figsize=(12, 6))
sns.barplot(x=model.feature_importances_, y=scaledPseudoDF.drop(['failure', 'product_code'], axis=1).columns,orient='h' ).set_title('Feature selection with VarianceThreshold',size=15);
plt.show()


# 📌 Insights:
# 
# * Floating values feature seem to be more important compaired to some integer values feature to predict the dependable feature outcome.
# * loading feature seems to be the best feature with highest importance across all features.

# # <b>5 <span style='color:#3f4d63'>|</span> Model</b>
# 
# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.1 | Prepairing Test set</b></p>
# </div>
# 
# * Prepairing test and performing all the transformation we did with train set so we can ready our test set for prediction.

# In[27]:


_,_,_ = infoDF(testDF)


# In[28]:


testPseudoDF = testDF.copy()
for col in floatColumns:
    if testPseudoDF[col].isnull().sum():
        testPseudoDF[col].fillna(testPseudoDF[col].mean(), inplace=True)
testPseudoDF[floatColumns].isnull().sum()


# In[29]:


for i in stringColumns:
    print('Unique Values for {} -> {}'.format(i, testDF[i].unique()))


# In[30]:


testPseudoDF['attribute_1'] = testPseudoDF['attribute_1'].str.split('_', 1).str[1].astype('int')
testPseudoDF['attribute_0'] = testPseudoDF['attribute_0'].str.split('_', 1).str[1].astype('int')


# In[31]:


scalerModel = StandardScaler().fit(testPseudoDF.drop('product_code', axis=1))
scaledtestPseudoDF = scalerModel.transform(testPseudoDF.drop('product_code', axis=1))
scaledtestPseudoDF = pd.DataFrame(scaledtestPseudoDF, columns=testPseudoDF.drop('product_code', axis=1).columns)
scaledtestPseudoDF['product_code'] = testPseudoDF['product_code']
scaledtestPseudoDF


# In[32]:


Xtrain = scaledPseudoDF.drop('failure', axis=1).reset_index(drop=True)
ytrain = scaledPseudoDF['failure'].reset_index(drop=True)
Xtest = scaledtestPseudoDF


# In[33]:


def getScore(model, yval, yvalPred):
    valScore = roc_auc_score(yval, yvalPred)
    print("Model -> {}, Validation Score -> {}".format(model, valScore))


# In[34]:


predictProbs = pd.DataFrame()


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.2 | Group K Fold</b></p>
# </div>
# 
# * We want to create a classifier which predicts correct probabilities for previously unseen products. To validate such a classifier, we have to simulate this situation by splitting the data so that the validation set contains other products than the training set. The correct method is a five-fold cross-validation where every fold uses four products for training and the fifth product for validation (GroupKFold).
# 
# Fold 0: Train on products A, B, C, D; validate on E<br>
# Fold 1: Train on products A, B, C, E; validate on D<br>
# Fold 2: Train on products A, B, D, E; validate on C<br>
# Fold 3: Train on products A, C, D, E; validate on B<br>
# Fold 4: Train on products B, C, D, E; validate on A<br>
# 
# * If you don't split your data with the GroupKFold on products, you'll get a data leak and inflated cross-validation scores. 
# <b> **Idea by @AmbrosM** <b>

# In[35]:


gkf = GroupKFold(n_splits=5)
for fold, (idx_tr, idx_va) in enumerate(gkf.split(Xtrain, ytrain, Xtrain.product_code)):
    print(f"===== fold{fold} =====")
    
    X_train = Xtrain.iloc[idx_tr][Xtest.columns]
    X_valid = Xtrain.iloc[idx_va][Xtest.columns]
    X_test = Xtest.copy()
    y_train = ytrain.iloc[idx_tr]
    y_valid = ytrain.iloc[idx_va]
    
    features = [f for f in X_train.columns if f != 'product_code']
        # Logistic Regression
    
    lrModel = LogisticRegression().fit(X_train[features], y_train)
    yValPred = lrModel.predict_proba(X_valid[features])[:,1]
    getScore('Logistic Regression', y_valid, yValPred)  
    predictProbs[f'LR_{fold}'] = lrModel.predict_proba(X_test[features])[:,1]
    
    del lrModel, yValPred
    gc.collect()
    
        
    # SVM 
    
    svcModel = SVC(probability=True).fit(X_train[features], y_train)
    yValPred = svcModel.predict_proba(X_valid[features])[:,1]
    getScore('SVM', y_valid, yValPred)
    predictProbs[f'SVC_{fold}'] = svcModel.predict_proba(X_test[features])[:,1]
    
    del svcModel, yValPred
    gc.collect()
    
    # K Neighbours 
    
    knnModel = KNeighborsClassifier(n_neighbors=20).fit(X_train[features], y_train)
    yValPred = knnModel.predict_proba(X_valid[features])[:,1]
    getScore('KNN', y_valid, yValPred)
    predictProbs[f'KNN_{fold}'] = knnModel.predict_proba(X_test[features])[:,1]
    
    del knnModel, yValPred
    gc.collect()
    
    
    
    


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>5.3 | Combining Results</b></p>
# </div>
# 
# * Combining result of best fermormed model and submitting values

# In[36]:


requiredPredCols = [col for col in predictProbs if 'LR' in col]
failure = predictProbs[requiredPredCols].sum(axis=1)/5


# In[37]:


submission = pd.read_csv('../input/tabular-playground-series-aug-2022/sample_submission.csv')
submission['failure'] = failure
submission.to_csv('firstSubmission.csv', index=False)
submission


# <div style="color:white;display:fill;
#             background-color:#3f4d6f;font-size:150%;
#             font-family:Nexa;letter-spacing:0.5px">
#     <p style="padding: 8px;color:white;"><b>Future Work</b></p>
# </div>
# 
# 1. Ensembling Model Result
# 2. Votting Classifier

# In[ ]:




