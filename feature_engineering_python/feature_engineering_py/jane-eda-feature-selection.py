#!/usr/bin/env python
# coding: utf-8

# <html>
#     <body>
#         <p><font size="6" color="blue">Contents</font></p>
#     </body>
#     
# - [Overview](#1)
# - [Basic Idea](#2)
# - [Target distribution](#3)
# - [Exploratory data analysis](#4)
# - [Random Forest Feature Selection](#5)
# - [Explainability](#6)    

# ###  <p><font size="5" color="blue">Overview</font></p><a id="1" ></a>
# ![](https://media.giphy.com/media/S4178TW2Rm1LW/giphy.gif)
# 
# ##### <p><font size='3' color='blue'>In this challenge, our task is to build a quantitative trading model to maximize returns using market data from a major global stock exchange. Each row in the dataset represents a trading opportunity, for which you will be predicting an action value: 1 to make the trade and 0 to pass on it.</font></p>

# ### <p><font size ='4' color='blue'> Import libraries</font></p>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier




# #### <font size ='4' color='blue'><a> Load data files </a></font>

# In[2]:


get_ipython().run_line_magic('time', '')
train = pd.read_csv("../input/jane-street-market-prediction/train.csv",nrows=1e5)
test = pd.read_csv("../input/jane-street-market-prediction/example_test.csv")


# ## <font size ='5' color='blue'><a> Basic Idea </a></font><a id="2" ></a>

# In[3]:


print(f"Train data contains {train.shape[0]} rows and {train.shape[1]} features")
print(f"Example test data contains {test.shape[0]} rows and {test.shape[1]} features")


# In[4]:


train.head(5)


# #### <font size ='4' color='blue'><a> Missing Values </a></font>

# In[5]:


temp = pd.DataFrame(train.isna().sum().sort_values(ascending=False)*100/train.shape[0],columns=['missing %']).head(20)
temp.style.background_gradient(cmap='Purples')


# In[6]:


train=train[train['weight']!=0]
train['action']=(train['resp']>0)*1
train.action.value_counts()


# #### <font size ='4' color='blue'><a> Target distribution </a></font><a id="3" ></a>

# In[7]:


fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.action.values,ax=ax[0],palette='husl')
sns.violinplot(x=train.action.values, y=train.index.values, ax=ax[1], palette="husl")
sns.stripplot(x=train.action.values, y=train.index.values,
              jitter=True, ax=ax[1], color="black", size=0.5, alpha=0.5)
ax[1].set_xlabel("Target")
ax[1].set_ylabel("Index");
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Counts");


# - The class distribution seems to be almost the same.
# - There is no relation between the target and index value.

# ## <font size ='5' color='blue'><a> Exploratory Data analysis </a></font><a id="4" ></a>
# #### <font size ='4' color='blue'><a> Some Feature distribution </a></font><a id="4" ></a>
# Let's check the distribution of some features for each target

# In[8]:


def plot_features(df1,target='action',features=[]):
    
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(5,5,figsize=(14,14))
    
    
    for feature in features:
        i += 1
        plt.subplot(5,5,i)
        sns.distplot(df1[df1[target]==1][feature].values,label='1')
        sns.distplot(df1[df1[target]==0][feature].values,label='0')
        plt.xlabel(feature, fontsize=9)
        plt.legend()
    
    plt.show();
    


# In[9]:


plot_features(train,features=[f'feature_{i}' for i in range(25)])


# - <p><font size='3' color='red'> Unhide output to see feature distribution</font></p>

# In[10]:


plot_features(train,features=[f'feature_{i}' for i in range(25,50)])


# - <p><font size='3' color='red'> Unhide output to see feature distribution</font></p>

# In[11]:


plot_features(train,features=[f'feature_{i}' for i in range(50,75)])


# #### <font size ='4' color='blue'><a> Weight </a></font>
# Let's check the distribution of values of weight feature

# In[12]:


fig,ax = plt.subplots(1,2,figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Distribution of weight")
sns.distplot(train['weight'],color='blue',kde=True,bins=100)

t0 = train[train['action']==0]
t1 =  train[train['action']==1]
plt.subplot(1,2,2)
sns.distplot(train['weight'],color='blue',kde=True,bins=100)
sns.distplot(t0['weight'],color='blue',kde=True,bins=100,label='action = 0')
sns.distplot(t1['weight'],color='red',kde=True,bins=100,label='action = 1')
plt.legend()




# The distribution of weight is highly left skewed,which indicates that there are many samples with 0 weight which we will have to remove.

# #### <font size ='4' color='blue'><a>How response variables changes with weight factor? </a></font>
# 

# In[13]:


fig,ax = plt.subplots(2,2,figsize=(12,10))
for i,col in enumerate([f'resp_{i}' for i in range(1,5)]):
    plt.subplot(2,2,i+1)
    plt.scatter(train[train.weight!=0].weight,train[train.weight!=0][col])
    plt.ylabel(col)
    plt.xlabel('weight')
plt.show()


# - Most of weights are in range of 0 to 20 and resp variables are in range of -0.05 to 0.05
# - It seems that all of the resp variables follows almost same pattern.
# - The outlier remains almost the same in all cases.

# #### <font size ='4' color='blue'><a> Resp </a></font>
# Let's compare values of resp_1,resp_2,resp_3,resp_4 with resp. There are changes in different time zones.

# In[14]:


def plot_resp():
    fig,ax = plt.subplots(2,2,figsize=(12,10))
    i=1
    for col in ([f'resp_{i}' for i in range(1,5)]):
        
        plt.subplot(2,2,i)
        plt.plot(train.ts_id.values,train.resp.values,label='resp',color='blue')
        plt.plot(train.ts_id.values,train[f'resp_{i}'].values,label=f'resp_{i}',color='red')
        plt.xlabel('ts_id')
        plt.legend()
        
        i+=1
    plt.show()
    
plot_resp()


# In[15]:


plt.figure(figsize=(10,10))
plt.scatter(train.resp.values,train.resp_1.values,color='red',label='resp_1')
plt.scatter(train.resp.values,train.resp_2.values,color='blue',label='resp_2')
plt.scatter(train.resp.values,train.resp_3.values,color='orange',label='resp_3')
plt.scatter(train.resp.values,train.resp_4.values,color='green',label='resp_4')
plt.xlabel("resp")
plt.ylabel('other resp variables')
plt.legend()




# - Most of the values have linear relationship with resp variable.
# 

# #### <font size ='4' color='blue'><a> Cumilative sum of response variable in diff time horizons</a></font>
# 

# In[16]:


plt.figure(figsize=(8,6))
for col in [f'resp_{i}' for i in range(1,5)]:
    plt.plot(train[col].cumsum().values,label=col)   
plt.legend()
plt.title("resp in different time horizons")
plt.show()


# #### <font size ='4' color='blue'><a> feature_0</a></font>
# feature_0 is a catergorical variable,let's check it's distribution.
# 

# In[17]:


sns.countplot(train.feature_0)


# ### <font size ='5' color='blue'><a>Feature Stats</a></font>
# #### <font size ='4' color='blue'><a> Distribution of mean values per row in the train set </a></font>
# 

# In[18]:


features = [col for col in train.columns if 'feature' in col]
t0 = train.loc[train['action'] == 0]
t1 = train.loc[train['action'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ### <font size ='4' color='blue'><a> Distribution of mean values per column in the train set </a></font>

# In[19]:


plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ### <font size ='4' color='blue'><a>Distribution of standard deviation values per row in the train set</a></font>

# In[20]:


features = [col for col in train.columns if 'feature' in col]
t0 = train.loc[train['action'] == 0]
t1 = train.loc[train['action'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per row in the train set")
sns.distplot(t0[features].std(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ### <font size ='4' color='blue'><a> Distribution of std values per column in the train set </a></font>

# In[21]:


plt.figure(figsize=(16,6))
plt.title("Distribution of standard deviation values per column in the train set")
sns.distplot(t0[features].std(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].std(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ### <font size ='4' color='blue'><a> Distribution of min values per row in the train set</a></font>

# In[22]:


t0 = train.loc[train['action'] == 0]
t1 = train.loc[train['action'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of min values per row in the train set")
sns.distplot(t0[features].min(axis=1),color="orange", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].min(axis=1),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ### <font size ='4' color='blue'><a> Distribution of min values per column in the train set</a></font>

# In[23]:


plt.figure(figsize=(16,6))
plt.title("Distribution of min values per column in the train set")
sns.distplot(t0[features].min(axis=0),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].min(axis=0),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()


# ## <font size ='5' color='blue'><a>Are there correlations between features?</a></font>
# Let's check the of there are highly correlated features in our data.

# In[24]:


train_corr = train[features].corr().values.flatten()
train_corr = train_corr[train_corr!=1]
test_corr = test[features].corr().values.flatten()
test_corr = test_corr[test_corr!=1]


plt.figure(figsize=(20,5))
sns.distplot(train_corr, color="Red", label="train")
sns.distplot(test_corr, color="Green", label="test")
plt.xlabel("Correlation values found in train (except 1)")
plt.ylabel("Density")
plt.title("Are there correlations between features?"); 
plt.legend();


# There are some highly correlated features in our data,we should probably remove them in the future.

# ### <font size ='5' color='blue'><a>PCA components of feature varibles</a></font>
# 

# In[25]:


plt.figure(figsize=(8,5))
pca = PCA().fit(train[features].iloc[:,1:].fillna(train.fillna(train.mean())))
plt.plot(np.cumsum(pca.explained_variance_ratio_),linewidth=4)
plt.axhline(y=0.9, color='r', linestyle='-')
plt.xlabel("number of components")
plt.ylabel("sum of explained variance ratio")
plt.show()


# - We only need less than 20 PCA components to explain 90% of varience of features.
# 
# Now,let's check if there exists any clusters

# In[26]:


rb = RobustScaler()
data = rb.fit_transform(train[features].iloc[:,1:].fillna(train[features].fillna(train[features].mean())))
data = PCA(n_components=2).fit_transform(data)
plt.figure(figsize=(7,7))
sns.scatterplot(data[:,0],data[:,1],hue=train['action'])
plt.xlabel('pca comp 1')
plt.ylabel('pca comp 2')


# ## <font size ='5' color='blue'><a>KMeans clustering </a></font><a id="5" ></a>
# Let's first chose number of clusters K by using elbow method

# In[27]:


from sklearn.cluster import KMeans
X_std = train[[f'feature_{i}' for i in range(1,130)]].fillna(train.mean()).values
sse = []
list_k = list(range(1, 10))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(data)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');


# Now,let's cluster and see..

# In[28]:


knn = KMeans(n_clusters=2)
labels=knn.fit_predict(data)
sns.scatterplot(data[:,0],data[:,1],hue=labels)


# ## <font size ='5' color='blue'><a> Random Forest feature importances </a></font><a id="5" ></a>
# Let's build a quick tree based model and see which all are the most important features.

# In[29]:


target='action'
cols_drop = list(np.setdiff1d(train.columns,test.columns))+['ts_id','date']

clf = RandomForestClassifier()
clf.fit(train.drop(cols_drop,axis=1).fillna(-999),train['action'])


# In[30]:


top=20
top_features = np.argsort(clf.feature_importances_)[::-1][:top]
feature_names = train.drop(cols_drop,axis=1).iloc[:,top_features].columns


# In[31]:


plt.figure(figsize=(8,7))
sns.barplot(clf.feature_importances_[top_features],feature_names,color='blue')


# ### <font size ='4' color='blue'><a> Distribution of top features </a></font>
# Let's check the feature value distribution for each target for the topn 8 features.

# In[32]:


top=8
top_features = np.argsort(clf.feature_importances_)[::-1][:top]
top_features = train.drop(cols_drop,axis=1).iloc[:,top_features].columns


# In[33]:


def plot_features(df1,target='action',features=[]):
    
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4,2,figsize=(14,14))
    
    
    for feature in features:
        i += 1
        plt.subplot(4,2,i)
        sns.distplot(df1[df1[target]==1][feature].values,label='1')
        sns.distplot(df1[df1[target]==0][feature].values,label='0')
        plt.xlabel(feature, fontsize=9)
        plt.legend()
    
    plt.show();
    
plot_features(train,features=top_features)


# ## <font size ='4' color='blue'><a>Top feature interactions </a></font><a id="6" ></a>
# 

# In[34]:


sns.pairplot(train[list(feature_names[:10])+['action']],hue='action')


# ## <font size ='4' color='blue'><a>SHAP Exaplainability </a></font><a id="6" ></a>
# Let's take a look at SHAP feature importance values
# 

# In[35]:


import shap


# In[36]:


explainer = shap.TreeExplainer(clf)
X = train.drop(cols_drop,axis=1).fillna(-999).sample(1000)
shap_values = explainer.shap_values(X)


# In[37]:


shap.summary_plot(shap_values, X, plot_type="bar")


# ### <font size ='4' color='blue'><a>SHAP Dependence Plots</a></font>
# SHAP dependence plots show the effect of a single feature across the whole dataset. They plot a feature's value vs. the SHAP value of that feature across many samples. Let's take a look at `feature_35`

# In[38]:


shap.dependence_plot('feature_35', shap_values[1], X, display_features=X.sample(1000))


# ### <p><font size ='4' color="red">Please do an upvote if you liked it :)</font></p>
# ### <font size ='3' color='blue'><a> References. </a></font>
# - https://www.kaggle.com/gpreda/santander-eda-and-prediction
# 

# In[ ]:




