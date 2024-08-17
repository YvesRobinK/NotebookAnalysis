#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='darkgrid')

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor

from warnings import filterwarnings
filterwarnings('ignore')


# # Read Data

# In[2]:


df_test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print("Train shape:",df_train.shape)
print("Test Shape:",df_test.shape)


# # Separate features and target

# In[3]:


X_trainfull=df_train.drop(["SalePrice"], axis=1)
y=df_train.SalePrice


# # Distribution of Saleprice

# In[4]:


plt.figure(figsize=(8,4))
plt.title("Distribution of Sales Price (y)")
sns.distplot(y)
plt.show()


# It can be observed from above that y is right-skewed, log transform can be applied to make it normal distribution.

# In[5]:


y=np.log1p(y)

plt.figure(figsize=(8,4))
plt.title("Distribution of log Sales Price (y)")
sns.distplot(y)
plt.xlabel("Log of Sales Price")
plt.show()


# # Percentage of null valued features in Train data

# In[6]:


d_temp=X_trainfull.isna().sum().sort_values(ascending=False)
d_temp=d_temp[d_temp>0]
d_temp=d_temp/df_train.shape[0]*100

plt.figure(figsize=(8,5))
plt.title("Features Vs Percentage Of Null Values")
sns.barplot(y=d_temp.index,x=d_temp, orient='h')
plt.xlim(0,100)
plt.xlabel("Null Values (%)")
plt.show()


# # Drop features where more than 20% records are null

# In[7]:


na_index=(d_temp[d_temp>20]).index
X_trainfull.drop(na_index, axis=1, inplace=True)


# Drop na>20% fields

# # Split Categorical and Numeric Features

# In[8]:


num_cols=X_trainfull.corrwith(y).abs().sort_values(ascending=False).index
X_num=X_trainfull[num_cols]
X_cat=X_trainfull.drop(num_cols,axis=1)


# # NUMERICAL FEATURES: FEATURE SELECTION AND ENGINEERING

# # View sample data

# In[9]:


X_num.sample(5)


# # Identify Features Highly correlated with target

# In[10]:


high_corr_num=X_num.corrwith(y)[X_num.corrwith(y).abs()>0.5].index
X_num=X_num[high_corr_num]


# # Heat-map of highly correlated Features

# In[11]:


plt.figure(figsize=(10,6))
sns.heatmap(X_num.corr(), annot=True, cmap='coolwarm')
plt.show()

print("Correlation of Each feature with target")
X_num.corrwith(y)


# # Remove multi-colinear features 

# In[12]:


X_num=X_num[high_corr_num]
X_num.drop(['TotRmsAbvGrd','GarageArea','1stFlrSF','GarageYrBlt'],axis=1, inplace=True)


# # Handling Null values

# In[13]:


#function to handle NA
def handle_na(df, func):
    """
    Input dataframe and function 
    Returns dataframe after filling NA values
    eg: df=handle_na(df, 'mean')
    """
    na_cols=df.columns[df.isna().sum()>0]
    for col in na_cols:
        if func=='mean':
            df[col]=df[col].fillna(df[col].mean())
        if func=='mode':
            df[col]=df[col].fillna(df[col].mode()[0])
    return df


# In[14]:


X_num=handle_na(X_num, 'mean')


# # Scale values

# In[15]:


# Function to scale df 
def scale_df(df):
    """
    Input: data frame
    Output: Returns minmax scaled Dataframe 
    eg: df=scale_df(df)
    """
    scaler=MinMaxScaler()
    for col in df.columns:
        df[col]=scaler.fit_transform(np.array(df[col]).reshape(-1,1))
    return df


# In[16]:


X_num=scale_df(X_num)


# ## Model Testing : Only Numerical Features

# In[17]:


X_train, X_val, y_train, y_val=train_test_split(X_num,y, test_size=0.2)
model=LinearRegression()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[18]:


model=SVR()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[19]:


model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[20]:


model=XGBRegressor(learning_rate=0.1)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# Observation:
# * Linear Regression and SVM models show similar performance moderately good score in both training and validation data
# * Random forest model is overfitting
# * XGB Regressor seems to be the best suited model

# # List the Numerical features required

# In[21]:


num_features=X_num.columns


# # CATEGORICAL DATA FEATURE SELECTION AND ENGINEERING

# # Explore Data

# In[22]:


X_cat.sample(5)


# In[23]:


X_cat.describe()


# In[24]:


for feature in X_cat.columns:
    print(
        f"{feature} :{len(X_cat[feature].unique())}: {X_cat[feature].unique()}"
    )


# # Drop features with more than 30 null values

# In[25]:


cat_na=X_cat.isna().sum().sort_values(ascending=False)
cat_na=cat_na[cat_na>30]
X_cat.drop(cat_na.index, axis=1, inplace=True)


# # EDA: Relation between each feature and saleprice

# In[26]:


for feature in X_cat.columns:
    plt.figure(figsize=(4,6))
    plt.title(f"{str(feature)} vs log Sale Price")
    sns.boxplot(X_cat[feature],y)
    plt.show()


# # Handling Null Values

# In[27]:


X_cat=handle_na(X_cat, 'mode')


# # Label encode features

# In[28]:


le=LabelEncoder()
X_cat_le=pd.DataFrame()
for col in X_cat.columns:
    X_cat_le[col] = le.fit_transform(X_cat[col])


# # Split into Train and validation set

# In[29]:


Xc_train, Xc_test, yc_train,yc_test=train_test_split(X_cat_le,y, test_size=0.2)


# # Fit and Evaluate Random Forest Model

# In[30]:


model=RandomForestRegressor()
model.fit(Xc_train,yc_train)


# In[31]:


print(f"Train score : {model.score(Xc_train,yc_train)}")
print(f"Test score : {model.score(Xc_test,yc_test)}")


# # Feature importance from RF Model

# In[32]:


feat_imp=pd.DataFrame({"Feature":Xc_train.columns,"imp":model.feature_importances_})
feat_imp=feat_imp.sort_values('imp', ascending=False)

plt.figure(figsize=(10,4))
plt.title("Feature Importance", fontsize=16)
sns.barplot('Feature', 'imp', data=feat_imp)
plt.xticks(rotation=80)
plt.show()


# # Calculate Training and Validation Accuracy for different number of features

# In[33]:


feat=[]
score_train=[]
score_test=[]
for i in range(29):
    imp_ft=feat_imp.head(i+1).Feature.unique()

    X_cat_imp=pd.DataFrame()
    for col in imp_ft:
        X_cat_imp[col] = le.fit_transform(X_cat[col])

    Xc_train, Xc_test, yc_train,yc_test=train_test_split(X_cat_imp,y, test_size=0.2)

    model=RandomForestRegressor(n_estimators=100)
    model.fit(Xc_train,yc_train)
    feat.append(i+1)
    score_train.append(model.score(Xc_train,yc_train))
    score_test.append(model.score(Xc_test,yc_test))
    
acc_feat_df=pd.DataFrame({"Feature":feat,"TrainAcc":score_train,"ValAcc":score_test})


# # Plot Number of Features vs Model Performance

# In[34]:


plt.figure(figsize=(10,5))
sns.lineplot('Feature', 'TrainAcc', data=acc_feat_df, label="Training Accuracy")
sns.lineplot('Feature', 'ValAcc', data=acc_feat_df, label="Validation Accuracy")
plt.xlabel("Number of Features")
plt.ylabel("R2 Score")
plt.xticks(rotation=80)
plt.xlim(1,29)
plt.show()


# Observation:
# * We can observe significant increase in train and validation accuracy with increase in features intitially.
# * After around 10 features, no significant improvement can be observed in either train or validation accuracy.
# * This is known as Curse of Dimensionality.
# * We can select the ideal number of features depending 
# * I am selecting top 17 features for training

# # List of selected Categorical Features 

# In[35]:


cat_features=list(feat_imp.iloc[:17,0])


# # Model Testing Only catagorical Featues

# In[36]:


# Selecting only important features
X_cat=X_cat[cat_features]
# OHE features
X_cat=pd.get_dummies(X_cat)
# Scaling the data
X_cat=scale_df(X_cat)


# In[37]:


X_train, X_val, y_train, y_val=train_test_split(X_cat,y, test_size=0.2)


# In[38]:


model=LinearRegression()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[39]:


model=SVR()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[40]:


model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# In[41]:


model=XGBRegressor(learning_rate=0.1)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")


# Observation:
# * Linear Regression preforms very poorly in validation.
# * Other three models have similar accuracy in validation, eventhough Random Forest model is overfitting.

# # FEATURE ENGINEERING IN COMBINED TRAIN AND TEST DATA

# In[42]:


#Combine train and test data
Xtt=pd.concat([X_trainfull,df_test])

#Split into Numeric and categoric features
Xtt_num= Xtt[num_features]
Xtt_cat= Xtt[cat_features]

#Handling null values
Xtt_cat=handle_na(Xtt_cat, 'mode')
Xtt_num=handle_na(Xtt_num,'mean')

#OHE Categoric features
Xtt_cat=pd.get_dummies(Xtt_cat,drop_first=True)

#Combine Numeric and Categorical features
Xtt=pd.concat([Xtt_num,Xtt_cat], axis=1)

#Scale Features
Xtt=scale_df(Xtt)

#Training and Testing Features after Feature Engineering
X=Xtt.iloc[:df_train.shape[0],:]
X_test=Xtt.iloc[df_train.shape[0]:,:]

#Training and Validation features and target
X_train, X_val, y_train, y_val=train_test_split(X,y, test_size=0.2)


# # Training, Evaluation and Prediction

# In[43]:


model=LinearRegression()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")
y_LR=model.predict(X_test)


# In[44]:


model=SVR()
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")
y_SVR=model.predict(X_test)


# In[45]:


model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")
y_RF=model.predict(X_test)


# In[46]:


model=XGBRegressor(learning_rate=0.1)
model.fit(X_train,y_train)
print(f"Train score : {model.score(X_train,y_train)}")
print(f"Validation score : {model.score(X_val,y_val)}")
y_XGB=model.predict(X_test)


# Observation:
# * Performance of Linear Regression is very poor in validation data
# * Accuracy of SVM model is reasonable
# * RF model is overfitting, still gives validation accuracy better than SVM model
# * XGBoost model gives the best result in validation data. 

# # Prepare Submission file

# In[47]:


sub = pd.DataFrame()
sub["Id"] = df_test.Id
sub["SalePrice"] = np.expm1(y_XGB)
sub.to_csv("submission.csv", index=False)


# # UPVOTE THE KERNEL IF YOU FIND IT HELPFUL 
