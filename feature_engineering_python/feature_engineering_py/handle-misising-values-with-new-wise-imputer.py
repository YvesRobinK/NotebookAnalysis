#!/usr/bin/env python
# coding: utf-8

# # Introduction to Imputers
# As you know, one of the most usual challenge in the field of data science is preparing the datasets to ones that are usable by machine learning algorithms. For this, we almost always need a dataset that doesn't contain missing values and therefore when our dataset contains them, we need to fill them by right values as more as accuarate is possible.
# 
# Data scientists know this challenge and therefore, creat different nice tools for dealing with it. They provide great Imputers that fill those empty values by different approaches. Some of them are:
# 
# **1. SimpleImputer:** That fill missing values in the data with different statistical properties like mean, median and most frequent.
# 
# **2. KNNImputer:** That complete missing values using k-Nearest Neighbors.
# 
# **3. IterativeImputer:** That estimates each feature from all the others.

# # Wise-Imputer
# 
# Actually I wrote this Imputer which I think act a litte better than above imputers. Its nature is very similar to the iterative imputer but has some differences:
# 
# 1. The IterativeImputer has a hyperparameter(n_nearest_features) that is the number of other features to use to estimate the missing values of each feature column. Its default uses all other columns to estimate the target one, which is not good enough beacuase uncorrelated features will weak the prediction and also slow down the process.
# Besides, setting "n_nearest_features" to a constant, assumes all features are the same in their relational nature; all of them correlate to exact number of the others.
# 2. The "estimator" hyperparameter of the IterativeImputer is applied for both categorical and numerical columns.
# 3. In the IterativeImputer we need to convert categorical values to numeric and it is not automated. 
# 
# The **Wise-Imputer** tries to handle these issues by setting absolute correlation coefficient and seperate esimator for classification and reggression respectively instead of n_nearest_features and just one estimator, and also factorize the categorical values to numeric.

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[2]:


train = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')
submission = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
concat = pd.concat([train, test], sort=False)


# In[3]:


def info(data):
    for i in range(data.shape[1]):

        print('====',data.columns[i],'====')
        print(data[data.columns[i]].value_counts())
        print('Number of unique values:',data[data.columns[i]].nunique())
        print('Number of nan values:',data[data.columns[i]].isnull().sum())


# In[4]:


info(concat)


# In[5]:


#Detecting numerical and categorical features

def seperate_cat_num_cols(data):
    Categorical_col = []
    Numerical_col = []
    
    for i in range(data.shape[1]):
        if data[data.columns[i]].isnull().values.any()==True:
            if (data[data.columns[i]].dtypes == 'O'):
                Categorical_col.append(data.columns[i])
            else:
                Numerical_col.append(data.columns[i])
    return Numerical_col,Categorical_col


# In[6]:


def prepare_data_for_corr(data): 
    
    Numerical_col = seperate_cat_num_cols(data)[0]
    Categorical_col = seperate_cat_num_cols(data)[1]

    cat_col_index,num_col_index = [],[]            
    for i in Categorical_col:
        cat_col_index.append(data.columns.tolist().index(i))
    for i in Numerical_col:
        num_col_index.append(data.columns.tolist().index(i))

    #Factorize categorical feature
    for i in range(len(Categorical_col)):
        data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]

    data = data.replace(-1, np.nan)

    num_data_for_corr = data.copy()
    imputer1 = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer1 = imputer1.fit(num_data_for_corr.values[:,num_col_index])
    
    num_data_for_corr = imputer1.transform(num_data_for_corr.values[:,num_col_index])
    num_data_for_corr = pd.DataFrame(num_data_for_corr,columns = Numerical_col)

    Cat_data_for_corr = data.copy()
    imputer2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputer2 = imputer2.fit(Cat_data_for_corr.values[:,cat_col_index])
    Cat_data_for_corr = imputer2.transform(Cat_data_for_corr.values[:,cat_col_index])
    Cat_data_for_corr = pd.DataFrame(Cat_data_for_corr,columns=Categorical_col)

    data_for_corr = pd.concat([num_data_for_corr, Cat_data_for_corr], axis=1)
    data_for_corr = data_for_corr.astype(float)

    return data_for_corr


# In[7]:


def most_correlated_columns(data,corr_coef):
    data = prepare_data_for_corr(data)
    corr_table = data.corr('pearson')
    corr_table = pd.DataFrame(corr_table)
    corr_table = corr_table.rename_axis().reset_index()

    correlated_features = {}
    for i in range(1,corr_table.shape[0]+1):
        a=[]
        for j in range(corr_table.shape[0]):

            if abs(corr_table[corr_table.columns[i]][j]) > corr_coef:

                a.append(corr_table['index'][j])
        
        correlated_features[corr_table.columns[i]] = a
    return correlated_features


# In[8]:


def fill_nan_numeric_cols(data,regression_estimator,S):
    
    data_org = data.copy()
    Numerical_col = seperate_cat_num_cols(data)[0]
    Categorical_col = seperate_cat_num_cols(data)[1]
    correlated_features = S
    
    for i in range(len(Categorical_col)):
        data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]

    data = data.replace(-1, np.nan)
    
    for i in Numerical_col:
        
        columns = []
        m = correlated_features[i]
        m.remove(i)
        
        num_data = data[m].copy()
        
        label = data[i]
        """need_scale = []  #If you need to scale the numerical columns you can uncomment this and work a little on it. 
        for j in correlated_features[i]:
            if j in Numerical_col:
                need_scale.append(j)"""
        
        #Scaler = scaler    
        #num_data[need_scale] = Scaler.fit_transform(num_data[need_scale])
        

        #print(data)
        nan=[]
        fill = [] 
        for k in range(data.shape[0]):
            
            if data[i].isnull()[k] ==True: 
                nan.append(k)
            else:
                fill.append(k)
        
        #Fill nan in num_data with SimpleImputer
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer = imputer.fit(num_data.values)
        transform = imputer.transform(num_data.values)
        num_data = pd.DataFrame(transform)
        num_data[i] = label
        
        #Train regression model
        
        X = num_data.values[fill,:-1]
        Y = num_data.values[fill,-1]
    
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
        reg = regression_estimator
        reg.fit(X_train, y_train)  
        predicted = reg.predict(num_data.values[nan,:-1])
        
        k = 0
        for t in nan:
            data[i][t] = predicted[k]
            k = k+1

    data.drop(Categorical_col,axis=1,inplace=True)
    
    for i in Categorical_col:
        data[i] = data_org[i]
        
    return data


# In[9]:


def fill_nan_categoric_cols(data,classifiation_estimator,S):
    
    Numerical_col = seperate_cat_num_cols(data)[0]
    Categorical_col = seperate_cat_num_cols(data)[1]
    correlated_features = S
    
    for i in range(len(Categorical_col)):
        data[Categorical_col[i]] = pd.factorize(data[Categorical_col[i]])[0]
        
    data = data.replace(-1, np.nan)
    data_org = data.copy()
   
    for i in Categorical_col:
        columns = []
        m = correlated_features[i]
        m.remove(i)
       
        cat_data = data[m].copy()
        label = data[i]
        
        nan=[]
        fill = [] 
        for k in range(data.shape[0]):
        
            if data[i].isnull()[k] ==True:
                nan.append(k)
            else:
                fill.append(k)
        
        imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imputer = imputer.fit(cat_data.values)
        transform = imputer.transform(cat_data.values)
        cat_data = pd.DataFrame(transform)
        cat_data[i] = label
        
        X = cat_data.values[fill,:-1]
        Y = cat_data.values[fill,-1]

        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
        classify = classifiation_estimator
        classify.fit(X_train, y_train)  
        predicted = classify.predict(cat_data.values[nan,:-1])
        
        
      #Fill nan for i column with predicted valus
        
        k = 0
        for t in nan:
            data[i][t] = predicted[k]
            k = k+1

    return data 


# In[10]:


def Imputer(train,test,label,corr_coef,regression_estimator,classifiation_estimator):
    
    All = pd.concat([train, test], sort=False)
    Label = All[label]
    Label = Label.reset_index()
    All.drop([label],axis=1,inplace=True)
    All = All.reset_index()
    All.drop(['index'],axis=1,inplace=True)
    All_org = All.copy()
    s = most_correlated_columns(All,corr_coef)
    numeric = fill_nan_numeric_cols(All_org,regression_estimator(),s)
    categoric = fill_nan_categoric_cols(numeric,classifiation_estimator(),s)
    categoric[label] = Label[label]

    return categoric


# **Attention**
# In the Spaceship-titanic dataset, we remove the Cabin, PassengerId and the Name columns, beacuse they need more work to convert to the usable categorial features.

# In[11]:


train.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
test.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)


# In[12]:


final = Imputer(train,test,'Transported',0.1,GradientBoostingRegressor,ExtraTreesClassifier)
final


# In[13]:


info(final)


# # Predict Transported
# As we see above all missing values filled, and now we apply the LightGBM model to predict the Transported status.

# In[14]:


final.Transported = final.Transported.replace({True: 1, False: 0})


# In[15]:


train = final.head(train.shape[0])

X = train.values[:,:-1]
Y = train.values[:,-1]

label_encoded_y = LabelEncoder().fit_transform(Y)

kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

lgb = LGBMClassifier()

results = cross_val_score(lgb, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std() )


# In[16]:


test = final.tail(test.shape[0])
test.values[:,:-1]
lgb.fit(X, label_encoded_y)
predict = lgb.predict(test.values[:,:-1])
predict


# # Iterative Imputer

# In[17]:


Categorical_col = seperate_cat_num_cols(concat)[1]
for i in range(len(Categorical_col)):
        concat[Categorical_col[i]] = pd.factorize(concat[Categorical_col[i]])[0]

concat = concat.replace(-1, np.nan)
concat.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
concat = concat.reset_index()
concat.drop(['index'],axis=1,inplace=True)
concat


# In[18]:


label = concat['Transported']
#label = label.reset_index()
X = concat.values[:,:-1]
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(X)
X = imp_mean.transform(X)
data = pd.DataFrame(X)
data['Transported'] = label
data


# In[19]:


train = data.head(train.shape[0])

X = train.values[:,:-1]
Y = train.values[:,-1]

label_encoded_y = LabelEncoder().fit_transform(Y)

kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

lgb = LGBMClassifier()

results = cross_val_score(lgb, X, label_encoded_y, cv=kfold)
print(results.mean() ,results.std() )


# # Iterative Imputer VS Wise Imputer
# As we saw above wise Imputer act a little bit better than Iterative Imputer. I also submit both of them and again wise Imputer got better accuracy than Iterative Imputer, but the difference was not very high.
# It should be noted that for comparing them fairly, both of them need more tuning.

# ## At Last
# The Wise_Imputer need more manipulation and I am sure it will get very better perfomance.
# I am very thankful for the time you dedicated to this notebook and hope to know your opinion about my Wise-Imputer.
