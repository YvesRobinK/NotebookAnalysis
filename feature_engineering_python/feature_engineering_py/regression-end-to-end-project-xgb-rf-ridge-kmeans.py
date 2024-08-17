#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



#matplolib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
df.head()


# In[3]:


df.info()


# In[4]:


# total number of cells with missing values
print("Number of cells with the missing values are {}".format((df.isnull().sum()).sum()))


# In[5]:


# seperating numerical and categorical variabled for EDA and feature engineering

num_cols = [col for col in df.columns if df[col].dtype in ['int64','float64']]

cat_cols = [col for col in df.columns if df[col].dtype == 'object']

y = df['SalePrice']

# numerical and categorical dataframe

num_df = df[num_cols]
cat_df = df[cat_cols]


# In[6]:


# EDA and feature engineering on numerical cols

num_df.dropna(axis=1, inplace=True)

num_df.nunique()


# In[7]:


cols = [col for col in num_df.columns if num_df[col].nunique() > 15 ] 

cols.remove('SalePrice')

conti_fea = num_df[cols]


# # 1. Numerical columns data analysis

# > Below code will give us bivariate analysis of continous features and its releationship with our target
# 
# > we can see that some features are more likely to co-relate than others and we will cross check with mutual info regressor as well

# In[8]:


for idx, col in enumerate(conti_fea.columns):
    plt.figure(idx, figsize=(5,5))
    sns.relplot(x=col, y=y, kind="scatter", data=conti_fea)
    plt.show


# > From above plots we can see that 'Yearbuilt', 'YearRemodadd', 'GrLivArea', 'GarageArea' highly co-relate with our target 'SalePrice'

# In[9]:


cols = [col for col in num_df.columns if num_df[col].nunique() <= 15 ] 

dist_feature = num_df[cols]
dist_feature.head()


# In[10]:


for idx, col in enumerate(dist_feature.columns):
    plt.figure(idx, figsize=(5,5))
    sns.stripplot(x=col , y=y , data=dist_feature)
    plt.show


# > From above plots we can clearly see that 'MSSubclass', 'YrSold', 'MoSold', 'PoolArea' are not informative features and will examine them in mutual info regressor

# In[11]:


# MI score on num_df dataframe
#nti_fea = num_df[cols]
#ist_feature = num_df[cols]

X = num_df.drop('SalePrice', axis=1)

y = df['SalePrice']

#iscrete_features = dist_feature

def mi_score(X,y):
    mi_score = mutual_info_regression(X,y, discrete_features=False, random_state=0)
    mi_score = pd.Series(mi_score, name="MI_SCORE", index=X.columns)
    mi_score = mi_score.sort_values(ascending=False)
    return mi_score

mi_series = mi_score(X,y)

#i_df = pd.DataFrame(mi_series, columns="MI_SCORE", index = mi_series.index)

pd.DataFrame(mi_series)


# > From above plots and MI_score we will consider features having score above **0.15** to keep most informative features in our model training dataset.
# 
# > We will do feature engineering and outliers handling on these features

# In[12]:


# plootting MI scores of numrical data analysis
data=pd.DataFrame(mi_series)

fig, ax = plt.subplots(figsize=(6,15))

ax.set_title("MI SCORES OF NUMERICAL FEATURES")
sns.barplot(data=data, y=data.index, x='MI_SCORE', ax=ax)
ax.set_ylabel("Numrerical features")

plt.show


# # 2. Categorical columns data analysis

# > We will do bivariate analysis of categorical features to see how they explains our target variable

# In[13]:


cat_df = df[cat_cols]
cat_df.head()


# In[14]:


#cat_df.isnull().sum()

# remove features wiht most number of data missing

features_rem = [col for col in cat_df.columns if cat_df[col].isnull().sum() > 500]

cat_df = cat_df.drop(features_rem, axis=1)

cat_df.fillna(method='bfill', axis=1, inplace=True)


# In[15]:


# plotting categorical features to find releationship with our target variable

y = df['SalePrice']

col_15 = [col for col in cat_df.columns if cat_df[col].nunique() >= 15]

for idx, col in enumerate(col_15):
    plt.figure(idx, figsize=(8,8))
    sns.stripplot(x=col, y=y, data=cat_df[col_15])
    plt.xticks(rotation=60)
    plt.show


# > From these plots we can say that above features contains no information to explain our target variable as we have more uncertainity 

# In[16]:


y = df['SalePrice']

cols_ = [col for col in cat_df.columns if cat_df[col].nunique() < 15]

for idx, col in enumerate(cols_):
    plt.figure(idx, figsize=(5,5))
    sns.stripplot(x=col, y=y, data=cat_df[cols_])
    plt.xticks(rotation=60)
    plt.show


# > Some of the features like 'Street', 'BldgType', 'PavedDrive', 'Garagecond', 'HeatingQC' are more informative and explains target variable with less antropy than others

# In[17]:


# Mutual info regressio on categorical features
y = df['SalePrice']
X1 = cat_df[cols_]

for colname in X1.select_dtypes('object'):
    X1[colname],_= X1[colname].factorize()

miscore = mutual_info_regression(X1, y, random_state=0)
miscore = pd.Series(miscore, name="MISCORE", index = X1.columns)
miscore = miscore.sort_values(ascending=False)

pd.DataFrame(miscore)


# In[18]:


# plootting MI scores of numrical data analysis
data = pd.DataFrame(miscore)

fig, ax = plt.subplots(figsize=(6,15))

ax.set_title("MI SCORES OF CATEGORICAL FEATURES")
sns.barplot(data=data, y=data.index, x='MISCORE', ax=ax)
ax.set_ylabel("Categorical features")
ax.set_xlim(0,0.5)

plt.show


# > We will keep features having MI score of **0.15** and above in our model training 

# In[19]:


# corelation matrix for numerical continues features
plt.figure(figsize=(10,10))
sns.heatmap(data=conti_fea.corr(), annot=True, vmin=-1, vmax=1)
plt.show


# In[20]:


# features transformations and  new features addition

# ratio of ground live area and frist floor area

num_df['grlivperfirstflr'] = num_df['GrLivArea']/num_df['1stFlrSF']

# total number of porch types

num_df['Porchtypes'] = num_df[['ScreenPorch','3SsnPorch',
                               'EnclosedPorch','OpenPorchSF','WoodDeckSF']].gt(0).sum(axis=1)

# we can see from above numerical plots that houses with 2nd floor highly correlate with price of the house
# but are not having all house with 2nd floor so will tell model that what houses are having 2nd floor

num_df['2nd_flr'] = np.where(num_df['2ndFlrSF'].map(lambda x: x>0), 1 ,0) 

# total number of bathrooms 

num_df['Totalbath'] = (num_df['HalfBath'] + 
                       num_df['FullBath'] + 
                       num_df['BsmtHalfBath'] +   
                       num_df['BsmtFullBath'])

# mean saleprice surrounded by each Neighborhood 

num_df['MeanpriceNBH'] = df.groupby('Neighborhood').SalePrice.transform('mean')


# In[21]:


#num_df.head()


# In[22]:


data=pd.DataFrame(mi_series)
arr = np.array(data.index)

feature1 = []

for fea in arr:
    if data.loc[fea,'MI_SCORE'] > 0.15:
        feature1.append(fea)
    else:
        pass

feature1


# In[23]:


#X1.head()


# In[24]:


data2 = pd.DataFrame(miscore)
arr2 = np.array(data2.index)

feature2 = []

for fea in arr2:
    if data2.loc[fea,'MISCORE'] > 0.15:
        feature2.append(fea)
    else:
        pass
    
feature2


# In[25]:


# combining all transformd data and creating one single dataframe for model traning


# In[26]:


feature3 = ['grlivperfirstflr','Porchtypes','2nd_flr','Totalbath','MeanpriceNBH']


# # 3. K-means clustering

# In[27]:


# we will cluster houses based on area ocuupied by various floors and basement

feat = ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','2ndFlrSF']

def clusters(df):
    X = df.copy()
    X_scaled = (X - X.mean(axis=0))/X.std(axis=0)
    kmeans = KMeans(n_clusters=15, n_init=10, max_iter=400, random_state=0)
    kmeans = kmeans.fit_predict(X_scaled)
    X['clusters'] = np.array(kmeans)
    return X

clustering_df = clusters(num_df[feat])

clustering_df


# In[28]:


y = df['SalePrice']
feat = ['GrLivArea','GarageArea','TotalBsmtSF','1stFlrSF','2ndFlrSF']
# plotting clusters and its significance on our target

for idx,col in enumerate(feat):
    plt.figure(idx, figsize=(6,6))
    sns.relplot(x=col,
            y=y, 
            hue='clusters',
            kind='scatter',
            palette='deep',
            data=clustering_df)
    plt.show()


# In[29]:


data2 = clustering_df.copy()
data2['Neighbor'] = df['Neighborhood']
data2['Salep'] = df['SalePrice']

# plotting using facet grid categorising via neighborhood

g = sns.FacetGrid(data2, col='Neighbor',height= 5,
                aspect= 1,
                col_wrap=3,
                palette='deep')
g.map_dataframe(sns.scatterplot,
                x='GrLivArea', 
                y='Salep', 
                hue='clusters')


# # 4. Training dataset preparation

# In[30]:


# ccategorical dataframe and numerical dataframe

d1 = X1[feature2]
d = num_df[feature1+feature3]


# In[31]:


# normalising data frame
d2 = d - d.mean(axis=0)/d.std(axis=0)

# concating dataframes d1 and d2

X = d2.join(d1, how='left')
X = round(X,1)
X = X.astype('int')
y = df['SalePrice']

# splitting train and validation dataset

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8, 
                                                      test_size=0.2,random_state=0)


# # 5. Hyperparamters tuning and models training

# In[32]:


def results(results):
    print('Best Params {}\n'.format(results.best_params_))
    print('Best Estimator {}\n'.format(results.best_estimator_))
    
    meanscore = results.cv_results_['mean_test_score']
    stdscore  = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean, std, param in zip(meanscore,stdscore,params):
        print('{} (+/-{}) for {}'.format(round(mean),round(std),param))


# In[33]:


#mlp = MLPRegressor()

#params = {'hidden_layer_sizes':[(100,),(150,2)],
#          'activation':['identity','tanh','relu'],
 #         'solver':['lbfgs','sgd','adam'],
 #         'learning_rate':['constant','invscaling','adaptive'],
  #        'max_iter':[500,1000]
  #        }

#cv1 = GridSearchCV(mlp, params, cv=5, scoring='neg_mean_absolute_error')
#cv1.fit(X_train,y_train)

#results(cv1)


# In[34]:


# we will train first model via stochastic gradient descent regressor

sgd = SGDRegressor()
params = {'loss': ['squared_loss','huber'],
          'max_iter': [500,1000,1500],
          'learning_rate': ['constant', 'optimal', 'invscaling'],
          'penalty': ['l2','l1','elasticnet']
          }

cv = GridSearchCV(sgd, params, cv=5, scoring='neg_mean_absolute_error')
cv.fit(X_train, y_train)

results(cv)


# In[35]:


# using random forest regressor

rfr = RandomForestRegressor()

params = {'n_estimators':[100,200,500],
          'max_depth':[5,8,10,15,20],
          'max_leaf_nodes':[50,100,150]
         }

cv1 = GridSearchCV(rfr, params, cv=5, scoring = 'neg_mean_absolute_error')
cv1.fit(X_train,y_train)

results(cv1)


# In[36]:


# using XGB algoritham

xgb = XGBRegressor()

params = {'n_estimators':[50,100,500,1000],
          'max_depth':[3,4,5],
          'learning_rate':[0.01,0.5,0.8]}

cv2 = GridSearchCV(xgb, params, cv=5, scoring='neg_mean_absolute_error')
cv2.fit(X_train,y_train)

results(cv2)


# In[37]:


# using svr algoritham 

#svr = SVR()

#params = {'kernel':['rbf','poly','linear'],
#          'degree':[3,4,5],
 #         'C':[0.5,1,5]}

#cv3 = GridSearchCV(svr, params, cv=5, scoring='neg_mean_absolute_error')
#cv3.fit(X_train,y_train)

#results(cv3)


# In[38]:


# using linear regression

lr = LinearRegression()

lr.fit(X_train,y_train)

preds = lr.predict(X_valid)

mae = mean_absolute_error(y_valid,preds)

print("Mean absolute error: ", mae)
print(lr.coef_)


# In[39]:


# ridge regression with l2 regularization

rdg = Ridge()

params = {'alpha': [0.0001,0.001,0.01,0.5,1,1.5]}

cv3 = GridSearchCV(rdg, params, cv=5, scoring='neg_mean_absolute_error')
cv3.fit(X_train,y_train)

results(cv3)


# > From above models our best models turn out to be randomforest and xtreame gradient boosting.
# 
# > We will test our models on validation dataset.
# 
# > We can notice that our tree based models are performing better than linear models. this is because our dataset contains lot of outliars that we haven't removed, instead we tried to normalize and cluster data.
# 
# > Due to cluatering dataset, and having multiple categorical features our dataset is prepared for tree based model than linear models.

# # 6. Evaluting results on validation dataset

# In[40]:


# evaluting validation model

pre1 = cv1.predict(X_valid)
pre2 = cv2.predict(X_valid)

# mae
mae1 = mean_absolute_error(y_valid,pre1)
mae2 = mean_absolute_error(y_valid,pre2)

print("MAE of RandomForest model:",mae1)

print("MAE of XGB model: ",mae2)

