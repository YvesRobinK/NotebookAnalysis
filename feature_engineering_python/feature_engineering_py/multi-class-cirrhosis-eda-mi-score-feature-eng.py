#!/usr/bin/env python
# coding: utf-8

# <a id='0'></a>
# # Playground Series - Season 3, Episode 26 (+EDA)
# **Our Goal:** For this Episode of the Series, our task is to use a multi-class approach to predict the the outcomes of patients with cirrhosis.

# # Easy Navigation
# 
# - [1- Data Exploration](#1)
# - [2- Explanatory Data Analysis (EDA)](#2)
#     - [2.0- Get the data ready for EDA](#2-0)
#     - [2.1- Categorical Features](#2-1)
#         - [2.1.1- Categorical features Counts/Distributions](#2-1-1)
#         - [2.1.2- Tabular Relationship](#2-1-2)
#         - [2.1.3- Visualization of Tabular Frequency of Features Vs Status](#2-1-3)
#     - [2.2- Numeric Features](#2-2)
# - [3- Feature Engineering](#3)
# - [4- Modeling](#4)
#     - [4.1- Model Construction](#4-1)
#     - [4.2- Model Utilization & Submission](#4-2)

# In[1]:


# import required libraies/dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from warnings import simplefilter
simplefilter('ignore')


# <a id='1'></a>
# # 1- Data Exploration

# **Files**<br>
# - train.csv - the training dataset; Status is the categorical target; C (censored) indicates the patient was alive at N_Days, CL indicates the patient was alive at N_Days due to liver a transplant, and D indicates the patient was deceased at N_Days.
# - test.csv - the test dataset; your objective is to predict the probability of each of the three Status values, e.g., Status_C, Status_CL, Status_D.
# sample_submission.csv - a sample submission file in the correct format

# In[2]:


# load datasets
df_train = pd.read_csv('/kaggle/input/playground-series-s3e26/train.csv', index_col=0)
df_test = pd.read_csv('/kaggle/input/playground-series-s3e26/test.csv', index_col=0)


# In[3]:


df_train.head()


# In[4]:


print(f'The training set contains {df_train.shape[0]} rows and {df_train.shape[1]} columns.')


# In[5]:


# Combine the training and the test sets to do some preprocessing.
df_both = pd.concat([df_train, df_test])


# ---

# <a id='2'></a>
# # 2- Explanatory Data Analysis (EDA)

# **In this section, we will first go through all features one by one, and then we will see the relationships among features themselves and the target variable**

# <a id='2-0'></a>
# ## 2.0- Intro for EDA

# In[6]:


# extract categorical and numerical features
cate_features = [
    'Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
numeric_features = list(set(df_train.columns)  - set(cate_features) - set(['Status']))


# In[7]:


## utils/functions

# funtion to draw a pie plot regarding a features counts
def draw_count_pie(df, feature):
    explode = [0]*df[feature].value_counts().shape[0]
    explode[0] = 0.1
    plt.pie(
        x = df[feature].value_counts(),
        labels=df[feature].value_counts().index,
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        startangle=0
    )
    plt.title(f'{feature} Counts', fontdict={'fontsize': 18})
    
# draw a countplot of a categorical variable along with how it effects the target variable
def draw_cate_vs_target(df, feature):
    plt.grid(True)
    ax = sns.countplot(data=df, x=feature , hue='Status')
    ax.set_title(f'{feature} counts vs Status', fontdict={'fontsize': 18})

# display tabular relationship between a numeric feature and Status
def display_tabular_relationship_cate_target(df, feature):
    display(pd.crosstab(
        index=df['Status'],
        columns=df[feature],
        normalize='columns'
    ))
    
# draw distribution plot for a numeric feature
def draw_numeric_dist(df, feature):
    sns.kdeplot(df[feature]) 
    ax = sns.distplot(df[feature])
    ax.set_title(f'{feature} Distribution', fontdict={'fontsize': 18})

# draw boxen plot for a numeric feature vs Status
def draw_numeric_target_violin(df, feature):
    ax = sns.violinplot(data=df, x='Status', y=feature)
    ax.set_title(f'{feature} VS Status', fontdict={'fontsize': 18})


# In[8]:


# Count the occurrences of each status category
status_counts = df_train['Status'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.set_palette('Set2')
sns.countplot(x='Status', data=df_train, order=status_counts.index)
plt.title('Count of Status', fontsize=18)
plt.xlabel('Status', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Add labels to the bars
for i, count in enumerate(status_counts):
    plt.text(i, count, count, ha='center', va='bottom', fontsize=12)

plt.show()


# <a id='2-1'></a>
# ## 2.1- Categorical Features

# <a id='2-1-1'></a>
# ### 2.1.1- Categorical features Counts/Distributions

# **NOTE:**<br>
# In this section we will visualize how categorical features are distributed. <br>
# I used Pie charts so as to help readers to conveniently identify the followings about categorical features:
# - How many & which values a particular feature contains
# - the most frequent value (a.k.a Mode)
# - the least frequent value
# - the percentage of occurance of each value
# 

# In[9]:


# draw the pie plots for all categorical features counts
plt.figure(figsize=(10, 20))
sns.set_palette('Set2')

for i, feature in enumerate(cate_features):
    plt.subplot(len(cate_features)//2+1, 2, i+1)
    draw_count_pie(df_train, feature)
    
plt.show()    


# Above-displayed figures are quite self-explaining. Hence, I leave it to you to look at them and grasp how particular features are distributed.<br>
# We will use these plots in feature engineering section as well.

# ---

# <a id='2-1-2'></a>
# ### 2.1.2- Cross Tabulation Relationships

# **NOTE:**<br>
# Using cross tabular relationships, we can easily identify how particular features have an impact on each other. In this case, we use it to determine how accurance of a particular value of a feature have effected the target variable.<br>
# The columns in each chart bellow represent the values of a particular feature. The sum of each column is equal to one. Furthermore, Each chart bellow contains two rows. And the intersection of each column and the first row represents the percentage of occurance of the corresponding column which has NOT resulted in an attrition; but the intersection of each column and the second row represents the percentage of occurance of the corresponding column which has resulted in an attrition. 

# In[10]:


# display the cross tabular relationships of categorical features and Status
print('           *************START***********             \n\n\n')
for i, feature in enumerate(cate_features):
    print(f'{i+1}: {feature} and Status')
    display_tabular_relationship_cate_target(df_train, feature)
    print(' '*10+'*'*10+'\n\n')


# <a id='2-1-3'></a>
# ### 2.1.3- Visualization of Tabular Frequency of Features Vs Status

# **NOTE:**<br>
# In this section we visualize how much each categorical feature has an impact on the target variable (Status).

# In[11]:


# draw the counts/distributions of categorical features vs the target variable
plt.figure(figsize=(20, 30))
sns.set_palette('Set1')

for i, feature in enumerate(cate_features):
    plt.subplot(len(cate_features)//2+1, 2, i+1)
    draw_cate_vs_target(df_train, feature)
    
plt.show()    


# Above-displayed figures are quite self-explaining. Hence, I leave it to you to look at them and grasp how the distribution of particular features have an impact on the target variable.<br>
# 

# ---

# <a id='2-2'></a>
# ## 2.2- Numeric features

# **NOTE:**<br>
# In this section we will visualize how each numeric feature is distributed, and how the distribution of each particular feature has an impact on the target variable (Status).

# In[12]:


# draw some plots for all numeric features [+ vs Status]
plt.figure(figsize=(25, 120))
sns.set_palette('Set2')

i = 1
for feature in numeric_features:
    # dist
    plt.subplot(len(numeric_features), 2, i)
    draw_numeric_dist(df_train, feature)
    # box
    plt.subplot(len(numeric_features), 2, i+1)
    draw_numeric_target_violin(df_train, feature)
    i += 2
    
plt.show()    


# Above-displayed figures are quite self-explaining. Hence, I leave it to you to look at them and grasp how particular features are distributed, and how they have an impact on the target variable.<br>
# We will use these plots in feature engineering section as well.

# <a id='3'></a>
# # 3- Feature Engineering

# What is **Feature Engineering?**<br>
# > Feature engineering is a machine learning technique that leverages data to create new variables that aren’t in the training set. It can produce new features for both supervised and unsupervised learning, with the goal of simplifying and speeding up data transformations while also enhancing model accuracy. (https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10)

# In[13]:


# Encode the categorical features into numbers

# Drug
df_both['Drug'] = df_both['Drug'].map({
    'D-penicillamine': 0,
    'Placebo': 1
}).astype('int')

# Sex
df_both['Sex'] = df_both['Sex'].map({
    'M': 0,
    'F': 1,
}).astype('int')

# Ascites
df_both['Ascites'] = df_both['Ascites'].map({
    'N': 0,
    'Y': 1
}).astype('int')

# Hepatomegaly
df_both['Hepatomegaly'] = df_both['Hepatomegaly'].map({
    'N': 0,
    'Y': 1
}).astype('int')

# Spiders
df_both['Spiders'] = df_both['Spiders'].map({
    'N': 0,
    'Y': 1
}).astype('int')

# Edema
df_both['Edema'] = df_both['Edema'].map({
    'N': 0,
    'Y': 1,
    'S': 2
}).astype('int')


# **Create some new features**:<br>
# - The `Bilirubin_to_Albumin_Ratio` feature is calculated as the ratio of `Bilirubin` to `Albumin`. It provides insights into the liver's synthetic and excretory functions.
# 
# - The `Disease_Severity_Score` feature represents the overall severity of cirrhosis by summing up the presence of conditions such as Ascites, Hepatomegaly, and Spiders.
# 
# - The `Liver_Health_Index` feature is calculated as the average of liver function tests such as Bilirubin, SGOT, Alk_Phos, and Prothrombin. It provides an index of overall liver health.
# 
# - The `Metabolic_Index` feature represents the metabolic status of patients with cirrhosis. It is calculated as the average of relevant biomarkers such as Cholesterol, Tryglicerides, and Albumin.

# In[14]:


# Liver function ratios
df_both['Bilirubin_to_Albumin_Ratio'] = df_both['Bilirubin'] / df_both['Albumin']

# Disease severity score
df_both['Disease_Severity_Score'] = df_both['Ascites'] + df_both['Hepatomegaly'] + df_both['Spiders']

# Liver health index
df_both['Liver_Health_Index'] = (df_both['Bilirubin'] + df_both['SGOT'] + df_both['Alk_Phos'] + df_both['Prothrombin']) / 4

# Metabolic index
df_both['Metabolic_Index'] = (df_both['Cholesterol'] + df_both['Tryglicerides'] + df_both['Albumin']) / 3

# N_Months & N_Years
df_both['N_Months'] = df_both['N_Days']/30
df_both['N_Years'] = df_both['N_Days']/365
#Age_Years
df_both['Age_Year'] = df_both['Age']/365
df_both.shape


# In[15]:


df_both.head()


# In[16]:


# re-assign df_train and df_test to reflect the changes we have made.
df_train = df_both.iloc[:df_train.shape[0]]
df_test = df_both.iloc[df_train.shape[0]:].drop('Status', axis=1)


# In[17]:


# Encode the target variable
df_train['Status'] = df_train['Status'].map({
    'D': 0,
    'C': 1,
    'CL': 2
}).astype('int')


# <div style='padding:20px; background:lightblue;color:black;border-radius:10px'>
#     <b>Mutual Information (MI)</b> is a way to how every feature interact with the target variable
# in this case 'price'. Here to interact means how a particular feature changes the target variable.
# The higher the score, the stronger the interaction.<br />
#     <b>Note: Mutual Information only works with numerrical data</b>
#     </div>

# In[18]:


from sklearn.feature_selection import mutual_info_classif

def get_mi_score(X, y):
    mi = mutual_info_classif(X, y, random_state=10)
    mi = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    return mi


# In[19]:


# display Mutual Information scores of the features

# Separate the features from the target variable
X_ = df_train.drop('Status', axis=1)
y_ = df_train['Status'].astype('int')

mi_scores = get_mi_score(X_, y_)
mi_scores


# In[20]:


plt.figure(figsize=(12, 8))
ax = sns.barplot(y=mi_scores.index[1:], x=mi_scores[1:])
ax.set_title('MI scores', fontdict={'fontsize': 16})
plt.show()


# In[21]:


# Determine the number of features to remove
num_features_to_remove = 3

# Remove the top least effective features from h_df
least_effective_features = mi_scores[-num_features_to_remove:].index
df_train.drop(least_effective_features, axis=1, inplace=True)
df_test.drop(least_effective_features, axis=1, inplace=True)

# Display the updated h_df
df_train.columns


# <a id='4'></a>
# # 4- Modeling

# <a id='4-1'></a>
# ## 4.1- Model Construction

# In[22]:


# Separate the features from the target variable
X = df_train.drop('Status', axis=1)
y = df_train['Status'].astype('int')


# In[23]:


# Split the data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=1)


# In[24]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

k = StratifiedKFold(n_splits=10)


# In[25]:


# LGBM Classifier
from lightgbm import LGBMClassifier

lgbm_params = { 'objective': 'multi_logloss', 
                'max_depth': 9, 
                'min_child_samples': 14, 
                'learning_rate': 0.034869481921747415, 
                'n_estimators': 274, 
                'min_child_weight': 9, 
                'colsample_bytree': 0.1702910221565107, 
                'reg_alpha': 0.10626128775335533, 
                'reg_lambda': 0.624196407787772, 
                'random_state': 42}
lgbm_model = LGBMClassifier(**lgbm_params)
score = cross_val_score(lgbm_model, X, y, cv=k)
np.mean(score)


# In[26]:


# XGB classifier
from xgboost import XGBClassifier

xgb_params = {
    'objective': 'multi_logloss', 
    'max_depth': 6, 
    'learning_rate': 0.010009541152584345, 
    'n_estimators': 1878,
    'min_child_weight': 9, 
    'colsample_bytree': 0.3292032860985591, 
    'reg_alpha': 0.10626128775335533, 
    'reg_lambda': 0.624196407787772, 
    'random_state': 42,
    'tree_method': 'hist', 
    'eval_metric': 'mlogloss',
    'subsample': 0.47524425009347593
}
xgb_model = XGBClassifier(**xgb_params)
score = cross_val_score(xgb_model, X, y, cv=k)
np.mean(score)


# In[27]:


# CatBoost classifier
from catboost import CatBoostClassifier

ct_params = {'iterations':470,
              'depth': 20,
              'learning_rate': 0.138112945166,
              'l2_leaf_reg': 4.0368544113430485,
              'random_strength': 0.1279482215776108,
              'max_bin': 238,
              'od_wait': 49,
              'one_hot_max_size': 39,
              'grow_policy': 'Lossguide',
              'bootstrap_type': 'Bernoulli',
              'od_type': 'Iter',    
              'min_data_in_leaf': 11}
ct_model = CatBoostClassifier(**ct_params, silent=True)

scores = cross_val_score(ct_model, X, y, cv=k)
np.mean(scores)


# In[28]:


# Voting classifier
from sklearn.ensemble import VotingClassifier

vc_model = VotingClassifier(
    estimators=[
        ('CatBoost', ct_model), 
        ('XGB', xgb_model),
        ('LGBM', lgbm_model)
    ],
    voting='soft'
)
score = cross_val_score(vc_model, X, y, cv=k)
np.mean(score)


# In[29]:


# fit the voting classifier model
vc_model.fit(X_train, y_train)


# In[30]:


from sklearn.metrics import confusion_matrix

y_pred_vc = vc_model.predict(X_test)

print('LGBM CM: ')
display(confusion_matrix(y_test, y_pred_vc))


# In[31]:


# compute the loggloss on the test set
from sklearn.metrics import log_loss

vc_preds = vc_model.predict_proba(X_test)

# Calculate the log loss
vc_logloss = log_loss(y_test, vc_preds)

print(f"Log Loss: {vc_logloss:.4f}")


# <a id='4-2'></a>
# ## 4.2- Prediction & Submission

# In[32]:


# predict on the test set
# classification
vc_preds = vc_model.predict_proba(df_test)


# In[33]:


# create the submission file
sb_file = pd.read_csv('/kaggle/input/playground-series-s3e26/sample_submission.csv')
sb_file.Status_D = vc_preds[:, 0]
sb_file.Status_C = vc_preds[:, 1]
sb_file.Status_CL = vc_preds[:, 2]
sb_file.head()


# In[34]:


# write to a file
sb_file.to_csv('./submission.csv', index=False)
print('Done...')


# # Thank you :)
# By: [Hikmatullah Mohammadi](https://www.kaggle.com/hikmatullahmohammadi) <br>
# 
# [Go to top](#0)
