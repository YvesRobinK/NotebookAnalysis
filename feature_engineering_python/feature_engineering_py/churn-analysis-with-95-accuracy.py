#!/usr/bin/env python
# coding: utf-8

# ![](https://i.imgur.com/kwgnmDy.png)

# **Thanks to @ayushv322 for letting me do work on his Notebook**

# In[1]:


#importing libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,PowerTransformer

#Modelling
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

#Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report

import os


# In[2]:


df = pd.read_csv('../input/telecom-churn-case-study-hackathon-38/train (1).csv') #getting the training dataset


# In[3]:


df.head() #upto 5 rows


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# 

# In[6]:


percent_null = df.isnull().sum().sum() / np.product(df.shape) * 100
percent_null


# In[7]:


colls = []
for col in df.columns:
    null_col = df[col].isnull().sum() / df.shape[0] * 100
    if null_col>50:
        colls+=[col]
    print("{} : {:.2f}".format(col,null_col))


# In[8]:


colls


# In[9]:


df = df.drop(colls,axis=1)


# In[10]:


percent_null = df.isnull().sum().sum() / np.product(df.shape) * 100
percent_null


# In[11]:


# df = df.fillna(0)


# In[12]:


df.isnull().sum().sum()


# In[13]:


colls = []
for col in df.columns:
    if df[col].dtype == object:
        colls+=[col]
        print(col)


# In[14]:


df[colls]


# In[15]:


df = df.drop(colls,axis=1)


# In[16]:


df.info()


# In[17]:


colls = []
for col in df.columns:
    if df[col].nunique() ==1:
        colls+=[col]
    print('{} : {}'.format(col,df[col].nunique()))


# In[18]:


df[colls]


# In[19]:


df = df.drop(colls,axis=1)


# In[20]:


df.duplicated().sum()


# In[21]:


colls = []
for col in df.columns:
    if df[col].nunique() <20:
        colls+=[col]
        print('{} : {}'.format(col,df[col].nunique()))


# In[22]:


def category_counts(col):
    plt.figure(figsize=(12,8));
    sns.countplot(x=df[col],palette='RdYlGn');
    plt.xlabel(col);
    plt.ylabel('Counts');
    plt.title(f"{col} Value Counts");
    plt.show()


# In[23]:


for col in colls:
    category_counts(col)


# In[24]:


df.head()


# In[25]:


# count_class_0, count_class_1 = df["churn_probability"].value_counts()
# # Divide by class
# df_class_0 = df[df['churn_probability'] == 0]
# df_class_1 = df[df['churn_probability'] == 1]


# In[26]:


# # Undersample 0-class and concat the DataFrames of both class
# df_class_0_under = df_class_0.sample(count_class_1)
# df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

# print('Random under-sampling:')
# print(df_test_under.churn_probability.value_counts())


# In[27]:


# from tensorflow_addons import losses
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.metrics import confusion_matrix , classification_report


# In[28]:


# import tensorflow as tf
# from tensorflow import keras


# model = keras.Sequential([
#     keras.layers.Dense(124, input_shape=(124,), activation='relu'),
#     keras.layers.Dense(61, activation='relu'),
#     keras.layers.Dense(30, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])

# # opt = keras.optimizers.Adam(learning_rate=0.01)

# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit(X_train, y_train, epochs=100)


# In[29]:


# model.evaluate(X_test, y_test)


# In[30]:


# yp = model.predict(final_normalized_x_test)
# yp


# In[31]:


# yp1 = model.predict(X_test)
# y_pred1 = []
# for element in yp1:
#     if element >= 0.5:
#         y_pred1.append(1)
#     else:
#         y_pred1.append(0)


# In[32]:


# y_pred = []
# for element in yp:
#     if element >= 0.5:
#         y_pred.append(1)
#     else:
#         y_pred.append(0)


# In[33]:


# from sklearn.metrics import confusion_matrix , classification_report

# print(classification_report(y_test,y_pred1))


# In[34]:


# confusion_matrix(y_test, y_pred1)


# In[35]:


# X = df_test_under.drop(['churn_probability',"id"],axis='columns')
# y = df_test_under['churn_probability']

X = df.drop(['churn_probability','id'],axis=1)
y=df['churn_probability']


# In[36]:


get_ipython().run_cell_magic('capture', '', '!git clone https://github.com/analokmaus/kuma_utils.git\n')


# In[37]:


from kuma_utils.preprocessing.imputer import LGBMImputer
lgbm_imtr = LGBMImputer(n_iter=100, verbose=True)
X = lgbm_imtr.fit_transform(X)


# In[38]:


columns=X.columns.tolist()


# In[39]:


# from imblearn.over_sampling import SMOTE

# smote = SMOTE(sampling_strategy='minority')
# X, y = smote.fit_resample(X, y)

# y.value_counts()


# In[40]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=1)


# In[41]:


scaler = StandardScaler()


# In[42]:


normalized_x_train = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)


# In[43]:


normalized_x_train.head()


# In[44]:


from lightgbm import LGBMClassifier, plot_importance 
lgbm_model = LGBMClassifier()
lgbm_model.fit(normalized_x_train, y_train)
y_pred = lgbm_model.predict(X_test)        
plot_importance(lgbm_model, figsize=(12, 25));


# In[45]:


# features=pd.DataFrame(sorted(zip(lgbm_model.feature_importances_,X.columns)), columns=['Value','Feature'])
# features


# In[46]:


# features[features["Value"]>95]["Feature"].to_list()


# In[47]:


normalized_x_test = pd.DataFrame(scaler.transform(X_test),columns = X_test.columns)


# In[48]:


def confusion_matrix_plot(matrix=None,classes=None,name='Logistic Regression'):
    plt.figure(figsize=(12,10))
    cmap = "YlGnBu"
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, fmt='g', ax=ax, cmap=cmap);  #annot=True to annotate cells, ftm='g' to disable scientific notation
    plt.savefig('/kaggle/working/img1.png')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels',fontsize = 15,fontweight = 3);
    ax.set_ylabel('True labels',fontsize = 15,fontweight = 3); 
    ax.set_title('Confusion Matrix of '+name,fontsize = 25,fontweight = 5); 
    ax.xaxis.set_ticklabels(classes); 
    ax.yaxis.set_ticklabels(classes[::-1]);
    plt.show()


# In[49]:


def cal_score(x_test=None,y_test=None,model=None,name=None):
    predictions = model.predict(x_test)
    labels=y_test
    matrix = confusion_matrix(predictions, labels)
    print(matrix)
    print('\n')

    f1 = f1_score(predictions, labels, average='weighted')
    print(f'F1 Score: {f1}')
    print('\n')
    classes=[False,True]
    print(classification_report(predictions, labels, labels=classes))
    
    confusion_matrix_plot(matrix = matrix,classes = classes,name = name)


# In[50]:


# lr = LogisticRegression(solver='newton-cg', class_weight='balanced')
# lr.fit(normalized_x_train,y_train)
# cal_score(x_test=normalized_x_test,y_test=y_test,model=lr,name='Logistic Regression')


# In[51]:


# dr = DecisionTreeClassifier()
# dr.fit(normalized_x_train,y_train)
# cal_score(x_test=normalized_x_test,y_test=y_test,model=dr,name='Decision Tree')


# In[52]:


#  from sklearn.svm import SVC

# from sklearn.ensemble import GradientBoostingClassifier
# rf = GradientBoostingClassifier()
# rf.fit(normalized_x_train,y_train)
# cal_score(x_test=normalized_x_test,y_test=y_test,model=rf,name='Random Forest')


# In[53]:


# y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy',weights)


# In[54]:


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
rf = LGBMClassifier(boosting_type='gbdt', objective='binary', num_leaves=20,
                    learning_rate=0.1, max_depth=20, reg_lambda=0.2)
rf.fit(normalized_x_train,y_train)
cal_score(x_test=normalized_x_test,y_test=y_test,model=rf,name='Lightgbclassifier')


# In[55]:


# pd.DataFrame({
#     'Feature':rf.feature_names_in_,
#     'Importance':rf.feature_importances_}).sort_values(by='Importance',ascending=False)


# In[56]:


test_df = pd.read_csv('../input/telecom-churn-case-study-hackathon-38/test (1).csv')


# In[57]:


sol_df = pd.read_csv('../input/telecom-churn-case-study-hackathon-38/solution (1).csv')
dict_df = pd.read_csv('../input/telecom-churn-case-study-hackathon-38/data_dictionary.csv')


# In[58]:


test_df.head()


# In[59]:


test_df.isnull().sum()


# In[60]:


a = test_df['id']


# In[61]:


test_df = test_df[X.columns]


# In[62]:


test_df.isnull().sum()


# In[63]:


percent_null = test_df.isnull().sum().sum() / np.product(test_df.shape) * 100
percent_null


# In[64]:


for col in test_df.columns:
    null_col = test_df[col].isnull().sum() / test_df.shape[0] * 100
    print("{} : {:.2f}".format(col,null_col))


# In[65]:


test_df['jun_vbc_3g'].mode()[0]


# In[66]:


# for col in test_df.columns:
#     null_col = test_df[col].isnull().sum() / test_df.shape[0] * 100
#     if null_col > 0:
#         test_df[col] = test_df[col].fillna(test_df[col].mode()[0])


# In[67]:


test_df[columns] = lgbm_imtr.fit_transform(test_df[columns])


# In[68]:


test_df.isnull().sum().sum()


# In[69]:


final_normalized_x_test = pd.DataFrame(scaler.transform(test_df),columns = test_df.columns)


# In[70]:


probabilities = rf.predict(final_normalized_x_test)


# In[71]:


probabilities.shape


# In[72]:


len(a)


# In[73]:


sol_df.head()


# In[74]:


submission = pd.DataFrame({'id':a,'churn_probability':probabilities})


# In[75]:


submission.to_csv('Submission1.csv',index=False)

