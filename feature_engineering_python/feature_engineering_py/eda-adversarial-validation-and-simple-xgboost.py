#!/usr/bin/env python
# coding: utf-8

# # EDA and Adversarial Validation
# 
# In this notebook we'll perform a Simple Exploratory Data Analysis on the train dataset of this competition, look at how similar/dissimilar train and test data are using Adversarial Validation and in the end train a Simple XGBoost Model to make a submission. 
# 
# EDA is a very crucial step in the model building process as it allows us to get to know the data better. It helps in cleaning the data and engineering features for the model.
# 
# Let's Start!!!

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_log_error


#  ### Data description:
# 
# * store_sales(in millions) - Store Sales(in million dollars)
# * unit_sales(in millions) - Unit Sales(in millions) in Stores Quantity
# * Total_children - Total Children in Home
# * avg_cars_at home(approx) - Average Cars at home(approx)
# * Num_children_at_home - Number of Children at Home as per Customer Filled Details
# * Gross_weight - Gross Weight of the Item
# * Recyclable_package - Whether Food Item is in a Recyclable Package or not
# * Low_fat - Whether Food Item is Low Fat or not
# * Units_per_case - Units Available in Each Store Shelves
# * Store_sqft - Store Area In SQFT
# * Coffee_bar - Whether Coffee Bar is available in store or not
# * Video_store - Whether Gaming Store is available or not
# * Salad_bar - Salad Bar available in store or not
# * Prepared_food - Prepared Food available in store or not
# * Florist - Flower shelves available in store or not
# * Cost - Cost of Acquiring Customers in dollars
# 
# #### Our Task is to devise a Machine Learning Model that helps us predict the cost of media campaigns in the food marts on the basis of the features provided.

# ### Evalaution Metric for this Competition:
# 
# #### Root Mean Squared Log Error (RMLSE):
# * The mean_squared_log_error function computes a risk metric corresponding to the expected value of the squared logarithmic (quadratic) error or loss.
# * Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.
# * We can use sklearn's mean_squared_log_error with squared=False for evaluating our models
# * We can also transform the target variable by taking a log transformation to use RMSE as the evaluation metric and in the end transform the predictions of our model back by taking an anti-log. 

# In[2]:


# Import Train Data and View the first five samples
df = pd.read_csv('/kaggle/input/playground-series-s3e11/train.csv')
print('Shape of Train Dataset:', df.shape)
print()
df.head()


# #### Key Insights:
# * In the train dataset, we have 360336 samples and 17 Variables. 
# * Out of these 17 Variables, the id column is just an index column, so we can drop it while training our model and the cost column is the target variable. 
# * So, there are 15 features that we can use to train our model.

# In[3]:


# Import Test Data and View the first five samples
test_df = pd.read_csv('/kaggle/input/playground-series-s3e11/test.csv')
print('Shape of Test Dataset:', test_df.shape)
print()
test_df.head()


# In[4]:


df.dtypes


# In[5]:


df.describe()


# In[6]:


df.isna().sum()


# #### Key Insights:
# There are no Missing Values for any column in the dataset! Now that's a relief!!

# In[7]:


columns = list(df.columns)
columns.remove('id')
print('No. of Columns to Visualize:' , len(columns))


# In[8]:


# Plot Histograms for Numerical variables and Bar Plots for Categorical Variables
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))
ax = ax.ravel()

no_cat = 0
no_num = 0

for i in range(16):
    if df[columns[i]].nunique() <= 10:
        no_cat+=1
        ax[i].bar(df[columns[i]].value_counts().sort_index().index, df[columns[i]].value_counts().sort_index().values, color='lightgreen') 
        ax[i].set_xticks(df[columns[i]].value_counts().sort_index().index)
        ax[i].set_title(f'{columns[i]} barplot: ');
    else:
        no_num+=1
        ax[i].hist(df[columns[i]], bins=50, color = "hotpink")
        ax[i].set_title(f'{columns[i]} distribution: ');

print()
print('No. of Numerical Variables:', no_num)
print('No. of Categorical Variables:', no_cat)
print()


# #### Key Insights:
# We have 5 Numerical(including the target variable) and 11 Categorical Variables in this dataset.

# In[9]:


# BoxPlots for Numerical Variables
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))
ax = ax.ravel()
i=0
colors = ['hotpink','lightgreen', 'hotpink', 'lightgreen', 'hotpink']
for c in range(16):
    if df[columns[c]].nunique() > 10:
        bplot = ax[i].boxplot(df[columns[c]], notch=True, patch_artist=True)
        ax[i].set_title(f'{columns[c]} BoxPlot: ')
        bplot['boxes'][0].set_facecolor(colors[i])
        i+=1


# #### Key Insights:
# There are no outliners in any of the numerical varibles except for stores_sales

# In[10]:


# The Correlation Matrix
plt.figure(figsize=(20,10))
corr_matrix = df.corr()
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="PiYG");


# #### Key Insights:
# * The most correlated variables with the target variable (Cost) are Florist, Video_Store, Salad_Bar and Prepared_Food. 
# * Moreover, salad_bar and prepared_bar have a correlation of 1.

# # Adversarial Validation
# 
# Adversarial Validation is used to check the similarity between the train and test datasets in terms of feature distributions. It's very simple to perform. 
# 
# We train a binary classifier to distinguish between the train and test samples by assigning label 0 to test samples and label 1 to train samples. If the model is able to differentiate, then that means the train and test data are very different from each other and if not, then they are probably similar and the usual validation techniques should work.

# In[11]:


# Most of the data preparation is taken from here:
# https://www.kaggle.com/code/konradb/adversarial-validation-and-other-scary-terms

X_train = df.drop(['id','cost'], axis=1)
y_train = df['cost'].copy()

X_test = test_df.drop(['id'], axis = 1)

# add an identifier and combine
X_train['istrain'] = 1
X_test['istrain'] = 0
X = pd.concat([X_train, X_test], axis = 0)

y = X['istrain'].copy()
X.drop('istrain', axis = 1, inplace = True)


# In[12]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We'll use a XGBoost Classifier
# Simple XGBoost Parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method':'gpu_hist',
    'learning_rate': 0.05, 
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'n_estimators':1000,
    'early_stopping_rounds':10
    }


# In[13]:


clf = XGBClassifier(**xgb_params, seed=42)


# In[14]:


for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        x0, x1 = X.iloc[train_index], X.iloc[test_index]
        y0, y1 = y.iloc[train_index], y.iloc[test_index]
        clf.fit(x0, y0, eval_set=[(x1, y1)],
                verbose=False)
        
        prval = clf.predict_proba(x1)[:,1]
        print(f'Fold {i+1} AUC Score:', roc_auc_score(y1,prval))


# We see that the validation AUC Score for all folds above is VERY close to 0.5, i.e., it is not easy to distinguish between the train and test datasets. This means that these two datasets are statistically very similar and have similar feature distributions. So, relying on your local validation should work very well for this competition.

# # XGBoost Model

# In[15]:


# Log Transforming the Target Variable
df['cost'] = np.log(df['cost'])
df.head()


# In[16]:


# Features used to Train
# Taken from https://www.kaggle.com/competitions/playground-series-s3e11/discussion/396508
FEATURES = [
 'total_children',
 'num_children_at_home',
 'avg_cars_at home(approx).1',
 'low_fat',
 'store_sqft',
 'coffee_bar',
 'video_store',
 'prepared_food',
 'florist'
]


# In[17]:


# Data Preparation
X = df[FEATURES].copy()
y = df['cost'].copy()


# In[18]:


# Simple XGBoost Params
xgb_params = {
    'booster': 'gbtree',
    'objective' : 'reg:squarederror',
    'eval_metric':'rmse', # We use RMSE as we log transformed the target variable
    'learning_rate': 0.1,
    'max_depth': 8,
    'n_estimators': 9999,
    'early_stopping_rounds': 200,
    'tree_method':'gpu_hist',
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'seed': 42
}


# In[19]:


# 5 KFold Training 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_iteration_xgb = []
scores = []
MODELS = []

for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
    
    print('#'*25)
    print('### Fold',i+1)
    print('#'*25)
    
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    
    X_valid = X.iloc[valid_index]
    y_valid = y.iloc[valid_index]
    
    model = XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
             eval_set=[(X_train, y_train), (X_valid, y_valid)],
             verbose=0)

    MODELS.append( model )
    
    fold_score = mean_squared_log_error(np.exp(y_valid), np.exp(model.predict(X_valid)), squared=False)
    print(f'Fold RMSLE Score:', fold_score)
    scores.append(fold_score)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X_train.columns
    fold_importance_df["importance"] = model.feature_importances_
    
    best_iteration_xgb.append(model.best_ntree_limit)
    print('Fold Feature Importance:')
    display(fold_importance_df.sort_values(by='importance', ascending=False).head(10))
    
print()
print(f'Average Vaildation RMSLE Score:', sum(scores)/5)


# ## Submission

# In[20]:


submission = pd.DataFrame(index = test_df.index.unique())
submission['id'] = test_df.id.unique()


# In[21]:


test_df = test_df[FEATURES]


# In[22]:


for i in range(5):
    submission[f'cost{i}'] = MODELS[i].predict(test_df)
submission['cost'] = np.exp((submission.cost0 + submission.cost1 + submission.cost2 + submission.cost3 + submission.cost4) / 5)
submission = submission[['id','cost']]


# In[23]:


submission.head()


# In[24]:


submission.to_csv('submission.csv', index=False)


# You can even try to use the original dataset from which this synthetic data is generated to get more data see if it improves your model. You can find the dataset [here][0].
# 
# I hope this notebook will help you in better understanding this competition and making amazing ML Models.
# #### Happy Modelling!!
# 
# [0]: https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction
