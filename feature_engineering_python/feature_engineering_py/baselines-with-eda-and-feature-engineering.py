#!/usr/bin/env python
# coding: utf-8

# # Baselines with EDA and Feature Engineering
# 
# Below is my exploratory analysis, feature engineering and them some baseline models. 
# 
# Please let me know what you think in the comments and **upvote** if you find anything useful.
# 
# Thanks and enjoy!

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from scipy.stats import norm
import scipy.stats as st

get_ipython().system('pip install sklearn-contrib-py-earth')
from pyearth import Earth

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Load the Data
# 
# Here we will load the data into a pandas dataframe.

# In[2]:


train_df = pd.read_csv('../input/tabular-playground-series-feb-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-feb-2021/test.csv')
display(train_df.head())
train_df.describe()
print(train_df.columns)


# We can see that we have 14 continuous variables and 10 categorical variables.

# In[3]:


cont_FEATURES = ['cont%d' % (i) for i in range(0, 14)]
cat_FEATURES = ['cat%d' % (i) for i in range(0, 10)]


# # Cleaning the Dataset
# 
# Following the steps of the Machine Learning Checklist we will start by cleaning out invalid values and outliers from the dataset.

# ### Invalid Values

# In[4]:


train_df.info()


# Here we can see that there are no *non-null* values so there is nothing to remove here.
# 
# ### Outliers
# 
# #### **Removing outliers is less of a science and more of an art form. So I will leave the choice up to you, but show you how to visualise these points.**
# 
# First we will look at outliers for the *target*.
# 
# We will add noise to the one dimensional features in order to "explode" the points out, helping us see the distributions and potential outliers.
# 
# We will use two methods for finding outliers:
# * The first will consider a point to be an outlier if it is N standard deviations from the mean. N is defined as the threshold.
# * A more complex form of outlier detection is LOF (Local Outlier Factor) which uses a points 20 nearest neighbours to determine if it is in a low density region (and therefore potentially and outlier).

# In[5]:


def plot_outliers(df, feature, threshold=3):
    mean, std = np.mean(df), np.std(df)
    z_score = np.abs((df-mean) / std)
    good = z_score < threshold

    print(f"Rejection {(~good).sum()} points")
    visual_scatter = np.random.normal(size=df.size)
    plt.scatter(df[good], visual_scatter[good], s=2, label="Good", color="#4CAF50")
    plt.scatter(df[~good], visual_scatter[~good], s=8, label="Bad", color="#F44336")
    plt.legend(loc='upper right')
    plt.title(feature)
    plt.show();
    
    return good
    
def plot_lof_outliers(df, feature):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, p=1)
    good = lof.fit_predict(df) > 0.5 # change this value to set the threshold for outliers
    print(f"Rejection {(~good).sum()} points")
    
    visual_scatter = np.random.normal(size=df.size)
    plt.scatter(df[good], visual_scatter[good], s=2, label="Good", color="#4CAF50")
    plt.scatter(df[~good], visual_scatter[~good], s=8, label="Bad", color="#F44336")
    plt.legend(loc='upper right')
    plt.title(feature)
    plt.show();
    
    return good


# ### Target Outliers

# In[6]:


good = plot_outliers(train_df['target'], 'target', threshold=4)


# Above we can see that these points are very reasonable outliers. There is a clear grouping for the target values however these points marked in red fall outside this grouping. I will therefore remove these 24 rejected points.

# In[7]:


train_df = train_df[good]
print('Now train_df has %d rows.' % (train_df.shape[0]))


# Next we will look at the LOF outliers.

# In[8]:


good = plot_lof_outliers(train_df['target'].values.reshape(train_df['target'].shape[0], -1), 'target')


# The above is harder to read as it has picked some points inside grouping. However, since there are only 300 points and I trust the LOF measurement, I am going to remove these points from dataset as well.

# In[9]:


train_df = train_df[good]
print('Now train_df has %d rows.' % (train_df.shape[0]))


# ### Feature Outliers
# 
# First we will look at the threshold outliers.

# In[10]:


for feature in cont_FEATURES:
    plot_outliers(train_df[feature], feature)


# So above we can see that the majority of the features do not contain outliers, however features *cont5* and *cont12* do contain some points that are could be considered as outliers.

# We will now look at the **LOF (Local Outlier Factor)** outliers.

# In[11]:


for feature in cont_FEATURES:
    # There some reshaping done here for syntax sake
    plot_lof_outliers(train_df[feature].values.reshape(train_df[feature].shape[0], -1), feature)


# We can see from the above that there are a small number of reasonable outliers selected here. I am therefore not going to remove any of these points as outliers.

# # Analysing Distributions
# 
# Here we will look at correlations between the features, distributions of the features.
# 
# First let's check that each row has it's own unique id.

# In[12]:


len(set(list(train_df['id'].values)))


# ### Continuous Variables

# In[13]:


for feature in cont_FEATURES:
    sns.violinplot(x=train_df[feature], inner='quartile', bw=0.1)
    plt.title(feature)
    plt.show();


# The above shows us that each feature has a unique distribution which could likely be used to help our models make predictions.
# 
# We can also see that contain features contain points that are most likely outliers (looking at the tails/heads), namely *cont3*, *cont4*, *cont5*, *cont6* and *cont12*.

# ### Categorical Variables
# 
# First let's look at what values the categorical variables can take.

# In[14]:


for cat in cat_FEATURES:
    values = train_df.groupby(cat)['id'].count().reset_index()
    sns.barplot(x=cat, y='id', data=values)
    plt.title(cat)
    plt.show();


# In[15]:


for feature in cat_FEATURES:
    sns.violinplot(x=feature, y='target', data=train_df, inner='quartile');
    plt.title(feature)
    plt.show()


# The takeaway from this is that the categorical variables are not a silver bullet for determining the target. The models will need to receive a combination of these variables in order to make accurate predictions.

# From the above I think we should remove some categories from the dataset since they are so small they serve no purpose. 
# 
# First let's look at the percentage of the rows that are the most common category.

# In[16]:


number_of_rows = train_df.shape[0]
for feature in cat_FEATURES:
    percentage_common_category = train_df.groupby(feature)['id'].count().reset_index()
    print(feature)
    print(percentage_common_category['id'].max() / number_of_rows)


# We can see from above that cat4 and cat6 are over 95% one class. And so they will be of minimal use to use and can be removed from the dataset.

# # Empirical CDFs
# 
# The below graphs show us where the 10th/20th/..../90th percentiles lie for each of the features.

# In[17]:


def plot_cdf(df, feature):
    ps = 100 * st.norm.cdf(np.linspace(-4, 4, 10)) # The last number in this tuple is the number of percentiles
    x_p = np.percentile(df, ps)

    xs = np.sort(df)
    ys = np.linspace(0, 1, len(df))

    plt.plot(xs, ys * 100, label="ECDF")
    plt.plot(x_p, ps, label="Percentiles", marker=".", ms=10)
    plt.legend()
    plt.ylabel("Percentile")
    plt.title(feature)
    plt.show();

for feature in cont_FEATURES:
    plot_cdf(train_df[feature], feature)


# This is perhaps the most revealing visualisations. It shows us that our features (especially '*cont1*' and '*cont5*') have unusual distributions. '*cont1*' appears to turn into an categorical variable when greater than 0.4 and '*cont4*' is a linear distribution once above 0.3. 
# 
# This could suggest that these variables need to split into additional features or have functions applied to their values to create a bigger distinction between very similar values.

# # Correlation
# 
# Here we can look at the correlation between the features and each other (and the target)

# In[18]:


# This plots a 16x16 matrix of correlations between all the features and the target
# Note: I sometimes comment this out because it takes a few minutes to run and doesn't show any useful information.

# pd.plotting.scatter_matrix(train_df, figsize=(10, 10));


# We can see that the above graph is far too busy to show us any useful information. However, at least we know that there isn't any clear correlations between a particular variable and the target.

# In[19]:


fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(train_df.drop(columns=['id']).corr(), annot=True, cmap='viridis', fmt='0.2f', ax=ax)


# Above we can see a cluster of features (cont1, cont5-cont12) that appear to be quite highly correlated together. This suggests that dimensionality reduction techniques could be used to reduce these features to a smaller set.

# # Analyse the Target

# In[20]:


sns.violinplot(x=train_df['target'], inner='quartile', bw=0.1)
plt.title('target')
plt.show();


# This doesn't show us much that is interesting other than the target is grouped around it's mean of 7.5, with some long tails out to either side.

# Finally we will look at the 2D histogram plots for each features vs. the target, this can be a clue of unusual correlations between the target and features. 
# 
# **Note:** There is also code for a KDE plot but these take a long time to run.

# In[21]:


for feature in cont_FEATURES:
    #sns.kdeplot(x=train_df['target'], y=train_df[feature], bins=20, cmap='magma', shade=True) 
    plt.hist2d(x=train_df['target'], y=train_df[feature], bins=20)
    plt.xlabel(feature)
    plt.ylabel('target')
    plt.title(feature)
    plt.show()


# # Feature Engineering

# ## Categorical Variables 

# In[22]:


# Remove cat4 and cat6 since over 95% of instances have the same value
train_df = train_df.drop(columns=['cat4', 'cat6'])

cat_FEATURES.remove('cat4')
cat_FEATURES.remove('cat6')

print(cat_FEATURES)


# #### Ordinal Variables

# In[23]:


# Converting to Ordinal Variables
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

ordinal_cat_FEATURES = ordinal_encoder.fit_transform(train_df[cat_FEATURES])
ordinal_cat_FEATURES


# We will now run these two models against our XGBoostRegressor to get a quick sense of it's performance.

# In[24]:


kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = ordinal_cat_FEATURES[train_index, :]
    test_X = ordinal_cat_FEATURES[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Mean Score: ", np.mean(scores))


# #### Categorical Variables

# In[25]:


# Converting to One-Hot Encoded Variables
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(handle_unknown='ignore') # Ignore categories that we don't see in training
onehot_cat_FEATURES = onehot_encoder.fit_transform(train_df[cat_FEATURES])
onehot_cat_FEATURES
print(onehot_encoder.get_feature_names())


# In[26]:


kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = onehot_cat_FEATURES[train_index, :]
    test_X = onehot_cat_FEATURES[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Mean Score: ", np.mean(scores))


# ## Continuous Variables

# #### PCA

# In[27]:


from sklearn.decomposition import PCA
pca_FEATURES = ['cont1', 'cont5', 'cont6','cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12']
non_pca_FEATURES = [feature for feature in cont_FEATURES if feature not in pca_FEATURES]
# For sake of argument we will reduce the number of variables by 50%
pca = PCA(n_components=5)

pca.fit(train_df[pca_FEATURES])


# In[28]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[pca_FEATURES].iloc[train_index, :]
    test_X = train_df[pca_FEATURES].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Mean Score: ", np.mean(scores))


# In[29]:


# Test the features after applying PCA
kf = KFold(n_splits=5)

scores = []
pca_X = pca.transform(train_df[pca_FEATURES])
for train_index, test_index in kf.split(pca_X):
    
    train_X = pca_X[train_index, :]
    test_X = pca_X[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Mean Score: ", np.mean(scores))


# So we can see by introducing PCA our models performed significantly better.

# #### Transformations

# In[30]:


for feature in non_pca_FEATURES:
    plt.scatter(train_df[feature], train_df['target'], s=2)
    plt.title(feature)
    plt.show()


# In[31]:


# Create the square and cube of the features
sq_features = []
cb_features = []
log_features = []
exp_features = []

for feature in non_pca_FEATURES:
    sq_feature = feature + '_2'
    cb_feature = feature + '_3'
    log_feature = feature + '_log'
    exp_feature = feature + '_exp'
    
    sq_features = sq_features + [sq_feature]
    cb_features = cb_features + [cb_feature]
    log_features = log_features + [log_feature]
    exp_features = exp_features + [exp_feature]
    
    train_df[sq_feature] = train_df[feature]**2
    train_df[cb_feature] = train_df[feature]**3
    train_df[log_feature] = np.log10(train_df[feature])
    train_df[exp_feature] = np.exp(train_df[feature])


# In[32]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[non_pca_FEATURES].iloc[train_index, :]
    test_X = train_df[non_pca_FEATURES].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Continuous Variables")
print("Mean Score: ", np.mean(scores))


# In[33]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[sq_features].iloc[train_index, :]
    test_X = train_df[sq_features].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Squared Variables")
print("Mean Score: ", np.mean(scores))


# In[34]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[cb_features].iloc[train_index, :]
    test_X = train_df[cb_features].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Cubed Variables")
print("Mean Score: ", np.mean(scores))


# In[35]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[log_features].iloc[train_index, :]
    test_X = train_df[log_features].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Log Variables")
print("Mean Score: ", np.mean(scores))


# In[36]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = train_df[exp_features].iloc[train_index, :]
    test_X = train_df[exp_features].iloc[test_index, :]
    
    train_target = train_df['target'].iloc[train_index]
    test_target = train_df['target'].iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("Exponential Variables")
print("Mean Score: ", np.mean(scores))


# **Note**: Here I would also like to apply winsorising, bucketting for one or two variables where required. This will be in a future version of the notebook so watch this space. 

# # Baselines

# So after the above results we will run our baselines with:
# * PCA to replace some of our continuous variables
# * The remaining continious variables will be add with all the above transformations
# * The categorical variables will be added with ordinal encoding

# In[37]:


# Construct our features 

features =  sq_features # + cb_features +  log_features + exp_features + non_pca_FEATURES
X = train_df[features] 

pca_features = ['pca_' + str(i) for i in range(0, pca_X.shape[1])]
X[pca_features] = pd.DataFrame(pca_X, index=X.index)

ordinal_features = ['ordinal_' + str(i) for i in range(0, ordinal_cat_FEATURES.shape[1])]
X[ordinal_features] = pd.DataFrame(ordinal_cat_FEATURES, index=X.index)

y = train_df['target']


# In[38]:


X['target'] = y
X.to_csv('preprocessed_train.csv', index=False)


# In[39]:


# Preprocess the Test Set
sq_features = []
for feature in non_pca_FEATURES:
    sq_feature = feature + '_2'
    sq_features = sq_features + [sq_feature]

    test_df[sq_feature] = test_df[feature]**2

test_pca_X = pca.transform(test_df[pca_FEATURES])
test_ordinal_cats = ordinal_encoder.transform(test_df[cat_FEATURES])

features =  sq_features # + cb_features +  log_features + exp_features + non_pca_FEATURES
test_X = test_df[features] 

pca_features = ['pca_' + str(i) for i in range(0, test_pca_X.shape[1])]
test_X[pca_features] = pd.DataFrame(test_pca_X, index=test_X.index)

ordinal_features = ['ordinal_' + str(i) for i in range(0, ordinal_cat_FEATURES.shape[1])]
test_X[ordinal_features] = pd.DataFrame(test_ordinal_cats, index=test_X.index)

test_X.to_csv('preprocessed_test.csv', index=False)


# #### LGMB Regressor

# In[40]:


# Test the non-PCA features
kf = KFold(n_splits=5)

scores = []
for train_index, test_index in kf.split(train_df):
    
    train_X = X.iloc[train_index, :]
    test_X = X.iloc[test_index, :]
    
    train_target = y.iloc[train_index]
    test_target = y.iloc[test_index]
    
    
    model = LGBMRegressor(random_state=42, objective='regression', metric='rmse')
    model.fit(train_X, train_target, eval_set=[(test_X, test_target)], verbose=False)
    preds = model.predict(test_X)
    score = mean_squared_error(test_target, preds)
    
    scores.append(score)
    print("Score:", score)

print("LGBM Regressor")
print("Mean Score: ", np.mean(scores))


# #### Splines Mars
# 
# This model is great for finding polynomial patterns in continuous features.

# In[41]:


model = Earth(allow_missing=True)
model.fit(X, y)


# In[42]:


preds = model.predict(X)
print("Mean Square Error: ", mean_squared_error(y, preds))


# # Takeaways and Future Work
# 
# Takeaways:
# * None of the features are "silver bullets" for making accurate predictions
# * Outliers exist in the dataset but aren't common
# * Dimensionality reduction performs well for a subset of the features
# * Transformations of the continuous variables has little affect on the score
# 
# Future Work:
# * I'm going to create a notebook where I dig deeper into these models and do some hyperparameter tuning to improve the performance. 
# 

# In[ ]:




