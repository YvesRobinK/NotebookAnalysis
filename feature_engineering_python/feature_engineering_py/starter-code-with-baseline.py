#!/usr/bin/env python
# coding: utf-8

# ## This notebook was presented during the WiDS Datathon 2021 Workshop held on 13th Jan,2021. Sharing the notebook here incase others find it useful too.

# 
# ![1610090459081.jpg](attachment:1610090459081.jpg)
# 
# ##  Objective
# The objective of this competition is to determine whether a patient admitted to an ICU has been diagnosed with a particular type of diabetes - Diabetes Mellitus. 
# 
# 
# ## Dataset
# * The dataset consists of 130,157 patient records along in the TrainingWiDS2021.csv file. This is the data that we will sue to train our model. 
# * The UnlabeledWiDS2021.csv  file consists of the unlabelled data and will be sued for testing purposes
# * SampleSubmissionWiDS2021.csv and the SolutionTemplateWiDS2021.csv provide a template and the submissions should be in this form
# * DataDictionaryWiDS2021.csv contains additional information about the dataset.
# 
# ## Evaluation Metric
# 
# Submissions for the leaderboard will be evaluated on the Area under the Receiver Operating Characteristic (ROC) curve between the predicted and the observed target (diabetes_mellitus_diagnosis).
# 
# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
# 
# ![](https://imgur.com/yNeAG4M.png)
# 
# An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. The following figure shows a typical ROC curve.
# 
# ![roccomp.jpg](attachment:roccomp.jpg)
# 
# source: http://gim.unmc.edu/dxtests/ROC3.htm
# 
# The closer the AUC is to 1, the better. An AUC of 0.9 is much better than an AUC of 0.01.
# For additional reading: http://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/

# ## Importing the necessary libraries

# In[1]:


# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

pd.set_option('display.max_rows', 500)
import gc

# sklearn preprocessing 
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

import lightgbm as lgb

# File system manangement
import os

#eda
get_ipython().system('pip install klib')
import klib

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


seed = 2357
np.random.seed(seed)


# ## Read in Data

# In[2]:


# List files available
print(os.listdir("../input/widsdatathon2021/"))


# In[3]:


# Training data
df_train = pd.read_csv('../input/widsdatathon2021/TrainingWiDS2021.csv')
print('Training data shape: ', df_train.shape)
df_train.head()


# In[4]:


df_train.drop('Unnamed: 0',axis=1,inplace=True)
print('Training data shape: ', df_train.shape)
df_train.head()


# The training data has 130157 observations and 180 variables including the TARGET (the label we want to predict).

# In[5]:


# Testing data features
df_test = pd.read_csv('../input/widsdatathon2021/UnlabeledWiDS2021.csv')
df_test.drop('Unnamed: 0',axis=1,inplace=True)
print('Testing data shape: ', df_test.shape)
df_test.head()


# The test set is considerably smaller and lacks a TARGET column.

# # Exploratory Data Analysis
# 
# ##  Column Types

# In[6]:


df_train.info(verbose=True, null_counts=True)


# In[7]:


# Number of each type of column
df_train.dtypes.value_counts()


# In[8]:


# Number of unique classes in each object column
df_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# ## Missing Values
# 
# Real world data is messy and often contains a lot of missing values. There could be multiple reasons for the missing values:
# 
# ![](https://imgur.com/68u0dD2.png)
# 
# Resource: [A Guide to Handling Missing values in Python](https://www.kaggle.com/parulpandey/a-guide-to-handling-missing-values-in-python)

# In[9]:


# Function to calculate missing values by column# Funct 

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[10]:


# Missing values for training data
missing_values_train = missing_values_table(df_train)
missing_values_train[:20].style.background_gradient(cmap='Greens')


# In[11]:


# Missing values for testing data
missing_values_test = missing_values_table(df_test)
missing_values_test[:20].style.background_gradient(cmap='Greens')


# We can also visualise the missign values instead of looking at the numbers. 

# Depending upon the percentage of missing values, you can decide to drop the columns which have a very high percentage of missing values and see if that improves the results.Let's drop the empty and single valued columns as well as empty and duplicate rows and visualise the clean dataframe. I'll do it for train but the same needs to be done for the test dataframe as well.

# In[12]:


df_cleaned_train = klib.data_cleaning(df_train)
klib.missingval_plot(df_cleaned_train)


# The graph above gives a high-level overview of the missing values in a dataset. It pinpoints which columns and rows to examine in more detail.
# 
# Top portion of the plot shows the aggregate for each column. Summary statistics is displayed on the right most side.
# 
# Bottom portion of the plot shows the missing values (black colors) in the DataFrame.

# ## The Target Column

# In[13]:


df_train['diabetes_mellitus'].value_counts(normalize=True)


# In[14]:


df_train['diabetes_mellitus'].astype(int).plot.hist();


# This is an example of imbalanced class problem i.e we where the total number of a class of data (0) is far less than the total number of another class of data (1). Examples include:
# * Fraud detection.
# * Outlier detection.
# 

# In[15]:


plt.figure(figsize=(10,6))
plt.title("Age vs people suffering from Diabetes")
sns.lineplot(df_train["age"],df_train["diabetes_mellitus"])
plt.xlabel("Age")
plt.ylabel("Number of people Suffering")


# In[16]:


plt.figure(figsize=(10,6))
plt.title("Age vs people suffering from Diabetes")
sns.lineplot(data=df_train, x="age", y="diabetes_mellitus", hue="gender")
plt.xlabel("Age")
plt.ylabel("Number of people Suffering")


# In[17]:


# Ethnicity vs Diabetes 
g = sns.catplot(x="ethnicity", hue="gender", col="diabetes_mellitus",
                data=df_train, kind="count",
                height=5, aspect=1.5);


# ## Correlation Plot
# 
# 

# In[18]:


#Display correlation with a target variable of interest
klib.corr_plot(df_train, target='diabetes_mellitus')


# ## Preprocessing Data
# 
# 

# In[19]:


# combine train and test together to do common feature engineering

train_copy = df_train.copy()
test_copy = df_test.copy()

# set up a flag field to distinguish records from training and testing sets in the combined dataset
train_copy['source'] = 0
test_copy['source'] = 1


all_data = pd.concat([train_copy, test_copy], axis=0, copy=True)
del train_copy
del test_copy
gc.collect()


# ### Dropping unnecessary columns
# There were no recurring patient visits, so the encounter_id would not be relevant to our models

# In[20]:


all_data.drop('encounter_id',axis=1,inplace=True)


# let's check if there is overlap between the hospitals in the labelled dataset and the hospitals in the unlabelled dataset

# In[21]:


df_train['hospital_id'].isin(df_test['hospital_id']).value_counts()


# In[22]:


# Dropping hospital id also
all_data.drop('hospital_id',axis=1,inplace=True)


# ### Encoding Categorical Variables
# 
# Resource: https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/

# Let's first visualise the categorical columns

# In[23]:


klib.cat_plot(df_train, figsize=(50,20))


# In[24]:


categorical_columns = all_data.select_dtypes('object').columns
categorical_columns


# In[25]:


objList = all_data.select_dtypes(include = "object").columns
print (objList)


# Create a label encoder object
le = LabelEncoder()
for feat in objList:
    all_data[feat] = le.fit_transform(all_data[feat].astype(str))

print (all_data.info())



# In[26]:


all_data[categorical_columns].head()


# ### Handling missing values

# In[27]:


all_data.fillna(-9999, inplace=True)
all_data.isnull().sum()


# ### split the all-data DF into training and testing again

# In[28]:


# split the all-data DF into training and testing again
training = all_data[all_data['source']==0]
testing = all_data[all_data['source']==1]

del all_data
gc.collect()


# In[29]:


print(training.shape)
print(testing.shape)


# In[30]:


testing.drop('diabetes_mellitus',axis=1,inplace=True)
print(testing.shape)


# # Baseline
# 
# ### 1. Logistic Regression

# In[31]:


TARGET = 'diabetes_mellitus'
train_labels = training[TARGET]
train = training.drop(columns = [TARGET,'source'])
features = list(train.columns)
test = testing.drop(columns = ['source'])
print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# In[32]:


# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, train_labels)


# In[33]:


# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]


# In[34]:


log_reg_pred 


# The predictions must be in the format shown in the sample_submission.csv file, where there are only two columns: encounter_id and diabetes_mellitus are present. We will create a dataframe in this format from the test set and the predictions called submit.
# 
# 

# In[35]:


# Submission dataframe
submit = df_test[['encounter_id']]
submit['diabetes_mellitus'] = log_reg_pred
submit.to_csv('logreg_baseline.csv',index=False)
submit.head()


# ### 2. Random Forest

# In[36]:


# Make the random forest classifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)


# In[37]:


# Train on the training data
rf.fit(train, train_labels)

# Extract feature importances
feature_importance_values = rf.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = rf.predict_proba(test)[:, 1]


# In[38]:


# Make a submission dataframe
# Submission dataframe
submit = df_test[['encounter_id']]
submit['diabetes_mellitus'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_domain.csv', index = False)


# # Feature Engineering
# 
# Depending on your model, your dataset may contain too many features for it to handle.Hence you can eliminate the ones which donot appear very high on the feature importance leaderboard.

# In[39]:


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


# In[40]:


# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)


# In[41]:


numerical_columns = train.columns[~train.columns.isin(categorical_columns)]


# # Evaluating the model
# 
# We have created our model but how will we know how accurate it is? We do have a Test dataset but since it doesn't have the Target column, everytime we optimize our model, we will have to submit our predictions to public Leaderboard to assess it accuracy.
# 
# ### Creating a Validation set
# Another option would be to create a validation set from the training set. We will hold out a part of the training set during the start of the experiment and use it for evaluating our predictions. We shall use the scikit-learn library's model_selection.train_test_split() function that we can use to split our data

# In[42]:


X = train
y = train_labels

train_X, val_X, train_y, val_y = train_test_split( X, y, test_size=0.20,random_state=0)


# In[43]:


# Logistic Regression

lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions2 = lr.predict_proba(val_X)[:,1]
roc_auc_score(val_y, predictions2)




# In[44]:


# RF

rf.fit(train_X, train_y)
predictions3 = rf.predict_proba(val_X)[:,1]
roc_auc_score(val_y, predictions3)


# In[45]:


# Light GBM

d_train=lgb.Dataset(train_X, label=train_y)

#Specifying the parameter
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='binary' #Binary target feature
params['metric']='binary_logloss' #metric for binary classification
params['max_depth']=10

#train the model 
clf=lgb.train(params,d_train,100) #train the model on 100 epocs

predictions_lgb = clf.predict(val_X)
roc_auc_score(val_y, predictions_lgb)


# # Using cross validation for more robust error measurement
# Using a Validation dataset has a drawback. Firstly, it decreases the training data and secondly since it is tested against a small amount of data, it has high chances of overfitting. To overcome this, there is a technique called cross validation. The most common form of cross validation, and the one we will be using, is called k-fold cross validation. ‘Fold’ refers to each different iteration that we train our model on, and ‘k’ just refers to the number of folds. In the diagram above, we have illustrated k-fold validation where k is 5.
# 
# ![](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)
# 
# 
# source: https://scikit-learn.org/stable/modules/cross_validation.html

# In[46]:


# Log Reg baseline with CV
scores_log = cross_val_score(lr, X, y, cv=10, scoring='roc_auc')
scores_log.sort()
print('Mean Absolute Score %2f' %(scores_log.mean()))


# In[47]:


# RF baseline with CV

scores_rf = cross_val_score(rf, X, y, cv=10, scoring='roc_auc')
scores_rf.sort()
print('Mean Absolute Score %2f' %(scores_rf.mean()))


# ## Cross-Validation for Imbalanced Classification
# A better way of splitting data would be to split it in such a way that maintains the same class distribution in each subset.we can use a version of k-fold cross-validation that preserves the imbalanced class distribution in each fold. It is called stratified k-fold cross-validation and will enforce the class distribution in each split of the data to match the distribution in the complete training dataset.
# 
# Logistic regression is used for modelling. The data set is split using Stratified Kfold. In each split model is created and predicted using that model. The final predicted value is average of all model.

# In[48]:


kf = StratifiedKFold(n_splits=3,shuffle=False,random_state=seed)
pred_test_full = 0
cv_score =[]
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y[train_index],y[test_index]
    
    #model
    lr = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
    #lr = LogisticRegression(C = 0.0001)
    lr.fit(xtr,ytr)
    score = roc_auc_score(yvl,lr.predict_proba(xvl)[:,1])
    print('ROC AUC score:',score)
    cv_score.append(score)    
    pred_test = lr.predict_proba(test)[:,1]
    pred_test_full +=pred_test
    i+=1


# In[49]:


print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))
print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))


# ## Reciever Operating Characteristics

# In[50]:


proba = lr.predict_proba(xvl)[:,1]
frp,trp, threshold = roc_curve(yvl,proba)
roc_auc_ = auc(frp,trp)

plt.figure(figsize=(10,6))
plt.title('Reciever Operating Characteristics')
plt.plot(frp,trp,'r',label = 'AUC = %0.2f' % roc_auc_)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')


# ## Some things that you could try to improve performance:
# 
# * Check for outliers
# * Efficient Missing value handling
# * Hyperparameter tuning
# 
# When we do hyperparameter tuning, it's crucial to not tune the hyperparameters on the testing data. We can only use the testing data a single time when we evaluate the final model that has been tuned on the validation data.
# 
# * Ensembling
# 

# In[ ]:





# In[ ]:




