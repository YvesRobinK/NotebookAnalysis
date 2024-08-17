#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:rgba(15, 159, 21, 0.5);">
#     <h1><center>Importing Libraries and Data</center></h1>
# </div>

# In[1]:


import random
random.seed(123)

import pandas as pd
import numpy as np
import datatable as dt
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import shap

from scipy.stats import chi2_contingency
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif,mutual_info_classif,SelectKBest,chi2, SelectFromModel


# In[33]:


# using datatable for faster loading

train = dt.fread(r'../input/tabular-playground-series-nov-2021/train.csv').to_pandas()
test = dt.fread(r'../input/tabular-playground-series-nov-2021/test.csv').to_pandas()


# <div style="background-color:rgba(15, 159, 21, 0.5);">
#     <h1><center>Basic Data Check</center></h1>
# </div>

# In[3]:


print(train.info())
print(test.info())

# The train data has 600k entries, target being boolean and rest 100 continuous variables


# In[4]:


train.describe()

# on first look, f2 seems different compared to others, in terms of range.


# In[5]:


train.head()

# different distributions of columns, it seems. We may need to scale later.


# In[6]:


# no missing values in the datasets

print('missing values in Train data: ',train.isna().sum().sum())
print('missing values in Test data: ',test.isna().sum().sum())


# In[7]:


# checking for duplicates in the data

print('number of duplicates in train: ',len(train.drop_duplicates())-len(train))
print('number of duplicates in test: ',len(test.drop_duplicates())-len(test))


# In[8]:


train.nunique().sort_values(ascending=True)

# no categorical variables other than our target - but might explore binning them later.


# In[9]:


# variables have low correlation with the target (though not as low as we saw in TPS October)

train.corr()['target'].sort_values(ascending=False)


# In[10]:


# checking if the variables are correlated with each other - they are NOT

sns.set(rc = {'figure.figsize':(12,8)})
sns.heatmap(train.corr())


# <div style="background-color:rgba(15, 159, 21, 0.5);">     <h1><center>Basic EDA</center></h1>
# </div>

# In[11]:


# plotting our target variable - very balanced

sns.countplot(train['target'])


# In[12]:


# plotting all the features' distribution
# 3 types - bimodal, spiked, right-skewed
# Maybe scaling needed and outliers need to be dealt with

columns = 10
rows = 10
f=0
fig, ax_array = plt.subplots(rows, columns, squeeze=False)
for i,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        axes.set_title('f'+str(f))
        col = 'f'+str(f)
        sns.set(rc = {'figure.figsize':(14,14)})
        g2 = sns.kdeplot(train[col],ax=axes)
        g2.set(ylabel=None)
        g2.set(xticklabels=[])
        g2.set(yticklabels=[])
        f=f+1
plt.show()


# In[13]:


# seeing which features may have outliers - spiked and skewed ones have plenty

columns = 10
rows = 10
f=0
fig, ax_array = plt.subplots(rows, columns, squeeze=False)
for i,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        axes.set_title('f'+str(f))
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        col = 'f'+str(f)
        sns.set(rc = {'figure.figsize':(14,14)})
        g2 = sns.boxplot(train[col],ax=axes)
        g2.set(ylabel=None)
        g2.set(xticklabels=[])
        g2.set(yticklabels=[])
        f=f+1
plt.show()


# <div style="background-color:rgba(15, 159, 21, 0.5);">     
#     <h1><center>Feature Selection</center></h1>
# </div>

# Credits to the following beautiful notebook by Bex - https://www.kaggle.com/bextuychiev/model-explainability-with-shap-only-guide-u-need/notebook
# 
# I have also used the following one from Luca as reference - https://www.kaggle.com/lucamassaron/feature-selection-by-boruta-shap
# 
# Following is a good notebook on LOFO - https://www.kaggle.com/frankmollard/lofo-importance-correlations-tps-nov-21
# 
# I am doing a simple SelectKBest (30 variables) in this data and then taking common ones from the above methods, to see what variables truly stand out.
# 
# Will update if I add another set and once I try **mutual information with my engineered variables**

# # SelectKBest

# In[14]:


# using selectkbest to get top 30 features - f_classif

X = train.drop(['id','target'],axis=1)
y = train['target']

selector = SelectKBest(score_func=f_classif,k=30)
selector.fit(X,y)

mask = selector.get_support()
new_features = [] # The list of your K best features

for bool, feature in zip(mask, X.columns):
    if bool:
        new_features.append(feature)
        
print(new_features)


# # Absolute Correlation

# In[15]:


# top 30 variables as per absolute correlation

vars = pd.DataFrame(np.abs(train.drop(['id'],axis=1).corr()['target']).sort_values(ascending=False).head(31)).index.to_list()
print(vars)


# # Variance and Coefficient of Variance Check

# In[28]:


plt.figure(figsize=(10,4))
train.var().sort_values(ascending=True).head(10).plot(kind='bar')

# f73 and f21 have very low variance, but f21 seems an important variable


# In[29]:


plt.figure(figsize=(10,4))
train.apply(lambda x: np.std(x)/np.mean(x)).sort_values(ascending=False).head(10).plot(kind='bar',color='g')

# all are spike/skew variables , with high COV compared to others.


# # Selection

# In[22]:


# from borutashap (no feature engineering done)

luca_variables = set(['f1', 'f10', 'f11', 'f14', 'f15', 'f16', 'f17', 'f2', 'f20', 'f21', 'f22','f24',
'f25', 'f26', 'f27', 'f28', 'f3', 'f30', 'f31', 'f32', 'f33', 'f34', 'f36', 'f37', 'f4', 'f40', 'f41',
'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f5','f50', 'f51', 'f53', 'f54', 'f55', 'f57', 
'f58', 'f59', 'f60', 'f61', 'f62', 'f64', 'f66','f67', 'f70', 'f71', 'f76','f77', 'f8', 'f80', 'f81',
'f82', 'f83', 'f87', 'f89', 'f9', 'f90', 'f91', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98'])

# from my selectkbest (top 30)

my_features_f_classif = set(['f3','f8','f10','f17','f21','f22','f24','f25','f26','f27','f34',
               'f40','f41','f43','f44','f47','f50','f54','f55','f57','f60','f66',
               'f71','f80','f81','f82','f91','f96','f97','f98'])

# from lofo notebook (taking top 30)

lofo_features = set(['f34','f55','f8','f43','f91','f71','f80','f27','f50','f41','f97','f66','f57',
                'f22','f25','f96','f81','f82','f21','f24','f26','f54','f98','f40','f60','f3','f17',
                'f95','f5','f45'])

# top 30 as per absolute correlation with target

cor_features = set(['f34', 'f55', 'f43', 'f71', 'f80', 'f91', 'f8', 'f27', 'f97', 'f50', 'f41', 'f57',
                    'f25', 'f22', 'f66', 'f96', 'f81', 'f82', 'f21', 'f40', 'f24', 'f60', 'f98', 'f3',
                    'f54', 'f44', 'f26', 'f47', 'f17', 'f10'])


# In[23]:


# common useful features - 30 in number, as per my 2 methods

useful_features = my_features_f_classif | cor_features
print(useful_features)


# <div style="background-color:rgba(15, 159, 21, 0.5);">     
#     <h1><center>Feature Engineering Ideas</center></h1>
# </div>

# # Creating variables using Row-wise Statistics

# In[30]:


# using columns of original dataset

columns = train.drop(['id','target'],axis=1).columns


# In[31]:


train['mean'] = train[columns].mean(axis=1)
train['sum'] = train[columns].sum(axis=1)
train['min'] = train[columns].min(axis=1)
train['max'] = train[columns].max(axis=1)
train['std'] = train[columns].std(axis=1)
train['var'] = train[columns].var(axis=1)


# In[32]:


# these features are not adding any value it seems, so will drop them for now

np.abs(train.corr()['target']).sort_values(ascending=False).head(10)


# # Creating Clusters

# I have referred to the following notebook by Kaveh - https://www.kaggle.com/kavehshahhosseini/tps-oct-2021-pca-and-kmeans-feature-eng/notebook

# In[34]:


# I may change number of clusters and the features used to get to them later
# Their distribution plots look nice - Gaussian type, but I may be wrong in first impressions :P

n_clusters = 6
cluster_cols = [f"cluster{i+1}" for i in range(n_clusters)]
kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, random_state=42)

X_cd = kmeans.fit_transform(train[useful_features])
X_cd = pd.DataFrame(X_cd, columns=cluster_cols, index=train.index)
train = train.join(X_cd)

fig = plt.figure(figsize = (10,5))
sns.kdeplot(data=train[cluster_cols])

plt.show()


# In[49]:


cluster_cols.append('target')
train[cluster_cols].corr()

# no significant correlation with the target, will check mutual information later


# # PCA Check

# PCA should be used mainly for variables which are strongly correlated. If the relationship is weak between variables, PCA does not work well to reduce data. In general, if most of the correlation coefficients are smaller than 0.3, PCA will not help. It might not help in our case too then.
# 
# As per the plot below, if we consider going ahead with PCA and explain **95% variance**-
# 1. Standard Scaling - we can take **94-95 components**
# 2. Robust Scaling - we can take **42-43 components**
# 3. MinMax Scaling - we can take **44-45 components**
# 
# The current plot is set for MinMax. I think Standard Scaling won't help much here, if we do want to reduce the number of features going into a model.
# 

# In[50]:


from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
sc = RobustScaler() # Scaling Data for PCA

X_scaled = sc.fit_transform(X)
pca = PCA().fit(X_scaled)

plt.rcParams["figure.figsize"] = (20,6)
fig, ax = plt.subplots()
xi = np.arange(1, 101, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 101, step=1))
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.90, '95% cut-off threshold', color = 'red', fontsize=16)
plt.axhline(y=0.99, color='g', linestyle='-')
plt.text(0.5, 1.0, '99% cut-off threshold', color = 'green', fontsize=16)

ax.grid(axis='x')
plt.tight_layout()
plt.show()


# # Binning Continuous Variables

# In[55]:


# As seen earlier, f14 has the least unique values among all.
# Now, it may not be categorical at all, but I wanted to check how it may look binned
# Original and 50 bins look identical...

columns = 4
rows = 1
plt.rcParams["figure.figsize"] = (12,4)
fig, ax_array = plt.subplots(rows, columns, squeeze=False)
data = pd.DataFrame(train['f14'])
data['Binned_10'] = pd.cut(data['f14'],bins=10,labels=False)
data['Binned_20'] = pd.cut(data['f14'],bins=20,labels=False)
data['Binned_50'] = pd.cut(data['f14'],bins=50,labels=False)

ax_array[0][0].set_title('Original')
sns.kdeplot(data['f14'],ax=ax_array[0][0])
ax_array[0][1].set_title('10 Bins')
sns.kdeplot(data['Binned_10'],ax=ax_array[0][1])
ax_array[0][2].set_title('20 Bins')
sns.kdeplot(data['Binned_20'],ax=ax_array[0][2])
ax_array[0][3].set_title('50 Bins')
sns.kdeplot(data['Binned_50'],ax=ax_array[0][3])


# # Transforming the Bimodal and Skewed Variables

# In[56]:


# I am taking f2 (skewed) for experiments
# Log Transformation might work for skewed - BUT HAVE TO THINK OF DEALING WITH NEGATIVE VALUES
# BoxCox and Square Root will fail because data has to be positive
# exp fails because it leads to infinitely high values
# Has outliers, so I can go for Robust Scaling

columns = 4
rows = 1
plt.rcParams["figure.figsize"] = (12,4)
fig, ax_array = plt.subplots(rows, columns, squeeze=False)
data = train['f2']

ax_array[0][0].set_title('Original')
sns.kdeplot(data,ax=ax_array[0][0])
ax_array[0][1].set_title('Logged')
sns.kdeplot(np.log1p(data),ax=ax_array[0][1]) # used 1p to deal with zeroes
ax_array[0][2].set_title('Reciprocal')
sns.kdeplot(1/data,ax=ax_array[0][2])
ax_array[0][3].set_title('Square Root')
sns.kdeplot(data**0.5,ax=ax_array[0][3])


# In[57]:


# I am taking f1 (bimodal) for relevant transformations
# square root looks decent. There are no outliers in these variables, hence no robust scaling needed.

columns = 4
rows = 1
plt.rcParams["figure.figsize"] = (12,4)
fig, ax_array = plt.subplots(rows, columns, squeeze=False)
data = train['f1']

ax_array[0][0].set_title('Original')
sns.kdeplot(data,ax=ax_array[0][0])
ax_array[0][1].set_title('Logged')
sns.kdeplot(np.log1p(data),ax=ax_array[0][1]) # used 1p to deal with zeroes
ax_array[0][2].set_title('Exponential')
sns.kdeplot(np.exp(data),ax=ax_array[0][2])
ax_array[0][3].set_title('Square Root')
sns.kdeplot(data**0.5,ax=ax_array[0][3])


# <div style="background-color:rgba(15, 159, 21, 0.5);">     
#     <h1><center>Summary</center></h1>
# </div>

# I have the following take-aways from this exercise-
# 
# 1. I will have to **check MI scores** to determine whether clustering and PCAs will help or not.
# 2. **Robust Scaling** the skewed variables should be ok.
# 3. Standard or MinMax Scaling or **taking square root** (if no negatives) of the bimodal variables.
# 4. Binning will not help here I think.
# 5. Creating row-wise stats will also not help much. Will wait for MI scores in my first few models.
# 
# Next up-
# 1. Create **basic models with feature engineering** ideas picked from here.
# 2. Have **read about NNs** doing really well in this competition - will learn that.
# 3. MI scores have to be checked once I have the complete cluster and PCA data added to my train.
