#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-success" style = "border-radius: 20px;text-align: center;" role="alert">
#     Don't forget to upvote if you liked the notebook
# </div>

# ## Table of Contents
# 
# - [1 - Importing the libraries](#1)
# - [2 - Importing the Datasets](#2)
# - [3 - percentage of Women and Men](#3)
# - [4 - Describing Data](#4)
# - [5 - Relations between dependent variable and independent variables](#5)
# - [6 - Taking Ground Truth from Train Data](#6)
# - [7 - Merging Train Data and Test Data](#7)
# - [8 - Feature Engineering](#8)
# - [9 - Checking Nan values](#9)
# - [10 - Now fix these values](#10)
# - [11 - Feature Selection](#11)
# - [12 - Taking care of missing data](#12)
# - [13 - Encoding Categorical Data](#13)
# - [14 - Getting back Train data and Test data](#14)
# - [15 - Splitting the Data](#15)
# - [16 - Feature Scaling](#16)
# - [17 - Applying Kernel PCA](#17)
# - [18 - K-Fold Cross Validation](#18)
# - [19 - Hyperparameter Tuning](#19)
# - [20 - Confusion Matrix](#20)
# - [21 - Checking the Overfitting](#21)
# - [22 - Logistic Regression](#22)
# - [23 - Ridge Classifier](#23)
# - [24 - LGBM Classifier](#24)
# - [25 - Naive_Bayes](#25)
# - [26 - XGBoost Classifier](#26)
# - [27 - Random Forest](#27)
# - [28 - k_Nearest_Neighbors](#28)
# - [29 - Kernel SVM](#29)
# - [30 - Decision Tree](#30)
# - [31 - Best Accuracy](#31)

# <a name='1'></a>
# # 1- Importing the libraries

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a name='2'></a>
# # 2- Importing the Datasets

# In[2]:


train_data=pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
train_data.shape


# In[3]:


test_data_org=test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
print(test_data.head())
test_data.shape


# <a name='3'></a>
# # 3- percentage of Women and Men

# In[4]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# In[5]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# <a name='4'></a>
# # 4- Describing Data

# In[6]:


print(train_data.info())


# In[7]:


print(test_data.info())


# In[8]:


print(train_data.describe())


# In[9]:


print(test_data.describe())


# <a name='5'></a>
# # 5- Relations between dependent variable and independent variables

# In[10]:


plt.figure(figsize=(10,6))


# In[11]:


plt.title("Males and Females who survived")
sns.barplot(x=train_data['Sex'],
             y=train_data['Survived'])


# In[12]:


plt.title("Age of people who survived or did not survive")
sns.swarmplot(x=train_data['Survived'], y=train_data['Age'])


# In[13]:


plt.title("People's Fair who survived or did not survive")
sns.swarmplot(x=train_data['Survived'], y=train_data['Fare'])


# In[14]:


plt.title("Survived and not survived people with their Fare and Age")
sns.scatterplot(x=train_data['Age'], y=train_data['Fare'], hue=train_data['Survived'])


# In[15]:


train_data.corr().round(2)


# In[16]:


train_data.plot(kind = 'scatter',
  x = 'Parch',
  y = 'Survived',
  figsize=(8,6))
plt.show()


# In[17]:


train_data.plot(kind = 'scatter',
  x = 'Fare',
  y = 'Survived',
  figsize=(8,6))
plt.show()


# <a name='6'></a>
# # 6- Taking Ground Truth from Train Data

# In[18]:


y=train_data['Survived'].values
y.shape


# <a name='7'></a>
# # 7- Merging Train Data and Test Data

# In[19]:


train_data=train_data.drop(['Survived'], axis=1)
concated_data=pd.concat([train_data,test_data],ignore_index=True)
print(concated_data)


# <a name='8'></a>
# # 8- Feature Engineering

# In[20]:


concated_data['Relatives'] = concated_data['SibSp'] + concated_data['Parch']
concated_data.loc[concated_data['Relatives'] > 0, 'Alone'] = 0
concated_data.loc[concated_data['Relatives'] == 0, 'Alone'] = 1


# <a name='9'></a>
# # 9- Checking Nan values

# In[21]:


#Thanks to this notebook (https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial)
for col in concated_data.columns.tolist():          
    print('{} column missing values: {}'.format(col, concated_data[col].isnull().sum()))


# In[22]:


# Thanks to this notebook (https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial)
concated_data[concated_data['Embarked'].isnull()]


# <a name='10'></a>
# # 10- Now fix these values

# In[23]:


concated_data['Embarked'] = concated_data['Embarked'].fillna('S')
print("row 61: ",concated_data.iloc[61,10])
print("row 829: ",concated_data.iloc[829,10])


# <a name='11'></a>
# # 11- Feature Selection

# In[24]:


X=concated_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Relatives']].values
print(X[0:5,:])


# <a name='12'></a>
# # 12- Taking care of missing data

# In[25]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:,2].reshape(-1,1))
X[:,2]=(imputer.transform(X[:,2].reshape(-1,1))).reshape(-1,)

imputer.fit(X[:,5].reshape(-1,1))
X[:,5]=(imputer.transform(X[:,5].reshape(-1,1))).reshape(-1,)

print(X[:6])


# <a name='13'></a>
# # 13- Encoding Categorical Data

# In[26]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[6])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X[:6,:])


# In[27]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,4]=le.fit_transform(X[:,4])
print(X[:6,:])
print("")


# <a name='14'></a>
# # 14- Getting back Train data and Test data

# In[28]:


test_data=X[891:]
X=X[:891]
print(X.shape,test_data.shape)


# In[29]:


print(test_data[:6])


# <a name='15'></a>
# # 15- Splitting the Data

# In[30]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=1)


# In[31]:


print(x_train[:6])


# In[32]:


print(x_test[:6])


# In[33]:


print(y_train[:6])


# In[34]:


print(y_test[:6])


# In[35]:


print(x_train.shape,x_test.shape, y_train.shape, y_test.shape)


# <a name='16'></a>
# # 16- Feature Scaling

# In[36]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]=sc.transform(x_test[:,3:])
test_data[:,3:]=sc.transform(test_data[:,3:])


# In[37]:


print(x_train[:6])


# In[38]:


print(x_test[:6])


# <a name='17'></a>
# # 17- Applying Kernel PCA

# In[39]:


from sklearn.decomposition import PCA
pca=PCA(n_components=6)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
test_data=pca.transform(test_data)


# <a name='18'></a>
# # 18- K-Fold Cross Validation

# In[40]:


# Used to make sure that We do not get lucky on easy examples in the training set and measure the real Accuracy
def K_Fold_CV(model):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X=x_train, y=y_train, cv=10, n_jobs=1)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
    return (np.mean(scores))


# <a name='19'></a>
# # 19- Hyperparameter Tuning

# In[41]:


# Used to find the best hyperparameters for a given model
def best_param(model,param_grid):
    gs=GridSearchCV(model,param_grid,cv=10)
    gs.fit(x_train,y_train)
    print("best params: ",gs.best_params_)


# <a name='20'></a>
# # 20- Confusion Matrix

# In[42]:


best_Acc={
    
    }
def Confusion_Matrix(y_pred,name):
    cm=confusion_matrix(y_test,y_pred)
    print(cm)
    print("")
    print("Sum of Wrong predictions",cm[0,1]+cm[1,0])
    print("Accuracy of the model: ",accuracy_score(y_test,y_pred))
    best_Acc[name]=accuracy_score(y_test,y_pred)
    


# <a name='21'></a>
# # 21- Checking the Overfitting

# In[43]:


def check_Overfitting(yhat_test,model):
    yhat_train=model.predict(x_train)
    return accuracy_score(y_train,yhat_train),accuracy_score(y_test,yhat_test)


# <a name='22'></a>
# # 22- Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()

best_K_FCV={
    "Logistic Regression":K_Fold_CV(classifier)
}

classifier.fit(x_train,y_train)


# In[45]:


y_pred=classifier.predict(x_test)
Overfitting={
    "train_Log,test_log":check_Overfitting(y_pred,classifier)
}
y_pred_TD_LogR=classifier.predict(test_data)


# In[46]:


print((np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))[:20])


# In[47]:


Confusion_Matrix(y_pred,"Logistic Regression")


# <a name='23'></a>
# # 23- Ridge Classifier

# In[48]:


from sklearn.linear_model import RidgeClassifier
param_grid={
    'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'max_iter':[25,50,75,100],
    'tol':[0.0001,0.00005,0.001,0.002,0.005,0.007,0.009]
}

best_param(RidgeClassifier(random_state=1),param_grid)


# In[49]:


classifier_RID=RidgeClassifier(alpha=0.1,max_iter=25,tol=0.0001,random_state=1)

best_K_FCV['Ridge']=K_Fold_CV(classifier_RID)

classifier_RID.fit(x_train,y_train)


# In[50]:


y_pred_RID=classifier_RID.predict(x_test)
Overfitting["train_RID,test_RID"]=check_Overfitting(y_pred_RID,classifier_RID)
y_pred_TD_RID=classifier_RID.predict(test_data)
print((np.concatenate((y_pred_RID.reshape(len(y_pred_RID),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[51]:


Confusion_Matrix(y_pred_RID,"Ridge")


# <a name='24'></a>
# # 24- LGBM Classifier

# In[52]:


from lightgbm import LGBMClassifier
classifier_LGBM = LGBMClassifier(num_leaves=20,max_depth=10,learning_rate=0.09,reg_lambda=10,random_state=1)

best_K_FCV['LGBM']=K_Fold_CV(classifier_LGBM)

classifier_LGBM.fit(x_train, y_train)


# In[53]:


y_pred_LGBM=classifier_LGBM.predict(x_test)
Overfitting["train_LGBM,test_LGBM"]=check_Overfitting(y_pred_LGBM,classifier_LGBM)
y_pred_TD_LGBM=classifier_LGBM.predict(test_data)
print((np.concatenate((y_pred_LGBM.reshape(len(y_pred_LGBM),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[54]:


Confusion_Matrix(y_pred_LGBM,"LGBM")


# <a name='25'></a>
# # 25- Naive_Bayes

# In[55]:


from sklearn.naive_bayes import GaussianNB
classifier_NB=GaussianNB()

best_K_FCV['Naive Bayes']=K_Fold_CV(classifier_NB)

classifier_NB.fit(x_train, y_train)


# In[56]:


y_pred_NB=classifier_NB.predict(x_test)
Overfitting["train_NB,test_NB"]=check_Overfitting(y_pred_NB,classifier_NB)
y_pred_TD_NB=classifier_NB.predict(test_data)


# In[57]:


print((np.concatenate((y_pred_NB.reshape(len(y_pred_NB),1),y_test.reshape(len(y_test),1)),1))[:20])


# In[58]:


Confusion_Matrix(y_pred_NB,"Naive Bayes")


# <a name='26'></a>
# # 26- XGBoost Classifier

# In[59]:


from xgboost import XGBClassifier
classifier_XGB=XGBClassifier(max_depth=1,min_child_weight=4,gamma=2.5)

best_K_FCV['XGBoost Classifier']=K_Fold_CV(classifier_XGB)

classifier_XGB.fit(x_train,y_train)


# In[60]:


y_pred_XGB=classifier_XGB.predict(x_test)
Overfitting["train_XGB,test_XGB"]=check_Overfitting(y_pred_XGB,classifier_XGB)
y_pred_TD_XGB=classifier_XGB.predict(test_data)


# In[61]:


print((np.concatenate((y_pred_XGB.reshape(len(y_pred_XGB),1),y_test.reshape(len(y_test),1)),1))[:20])


# In[62]:


Confusion_Matrix(y_pred_XGB,"XGBoost Classifier")


# <a name='27'></a>
# # 27- Random Forest

# In[63]:


from sklearn.ensemble import RandomForestClassifier
# it may take some time to find best parameters
param_grid={
    'n_estimators':[10,25,50,75,100],
    'max_features':[3,4,5,6],
    'max_depth':[5,7,10,12],
    'min_samples_leaf':[1,2,4,6]
}
best_param(RandomForestClassifier(random_state=1),param_grid)


# In[64]:


classifier_RF=RandomForestClassifier(criterion='entropy',max_depth=2,max_features=3,min_samples_leaf=5,n_estimators=128)

best_K_FCV['Random Forest']=K_Fold_CV(classifier_RF)

classifier_RF.fit(x_train,y_train)


# In[65]:


y_pred_RF=classifier_RF.predict(x_test)
Overfitting["train_RF,test_RF"]=check_Overfitting(y_pred_RF,classifier_RF)
y_pred_TD_RF=classifier_RF.predict(test_data)
print((np.concatenate((y_pred_RF.reshape(len(y_pred_RF),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[66]:


Confusion_Matrix(y_pred_RF,"Random Forest")


# <a name='28'></a>
# # 28- k_Nearest_Neighbors

# In[67]:


from sklearn.neighbors import KNeighborsClassifier
# finiding the best parameters using the GridSearchCV
n_neighbors=list(range(1,101))
param_grid={
    'n_neighbors':n_neighbors
}
best_param(KNeighborsClassifier(),param_grid)


# In[68]:


classifier_KNN=KNeighborsClassifier(n_neighbors=33)

best_K_FCV['K_Nearest_Neighbors']=K_Fold_CV(classifier_KNN)

classifier_KNN.fit(x_train,y_train)


# In[69]:


y_pred_KNN = classifier_KNN.predict(x_test)
Overfitting["train_KNN,test_KNN"]=check_Overfitting(y_pred_KNN,classifier_KNN)
y_pred_TD_KNN=classifier_KNN.predict(test_data)
print((np.concatenate((y_pred_KNN.reshape(len(y_pred_KNN),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[70]:


Confusion_Matrix(y_pred_KNN,"K_Nearest_Neighbors")


# <a name='29'></a>
# # 29- Kernel SVM

# In[71]:


from sklearn.svm import SVC
# finiding the best parameters using GridSearchCV
param_grid=[{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

best_param(SVC(random_state=1),param_grid)


# In[72]:


classifier_SVM=SVC(C=0.75,gamma=0.1,kernel='rbf')

best_K_FCV['Kernel SVM']=K_Fold_CV(classifier_SVM)

classifier_SVM.fit(x_train,y_train)


# In[73]:


y_pred_SVM = classifier_SVM.predict(x_test)
Overfitting["train_SVM,test_SVM"]=check_Overfitting(y_pred_SVM,classifier_SVM)
y_pred_TD_SVM=classifier_SVM.predict(test_data)
print((np.concatenate((y_pred_SVM.reshape(len(y_pred_SVM),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[74]:


concatenated_array=(np.concatenate((y_pred_SVM.reshape(len(y_pred_SVM),1), y_test.reshape(len(y_test),1)),1))
print(concatenated_array[(concatenated_array[:,0]!=concatenated_array[:,1])])


# In[75]:


Confusion_Matrix(y_pred_SVM,"Kernel SVM")


# <a name='30'></a>
# # 30- Decision Tree

# In[76]:


from sklearn.tree import DecisionTreeClassifier
# finiding the best parameters using GridSearchCV
param_grid={
    'max_depth':[5,15,25],
    'min_samples_leaf':[1,3,5],
    'max_leaf_nodes':[10,20,35,50]
}
best_param(DecisionTreeClassifier(criterion='entropy'),param_grid)


# In[77]:


classifier_DT=DecisionTreeClassifier(criterion='entropy',max_depth=1,max_leaf_nodes=20,min_samples_leaf=3)

best_K_FCV['Decision Tree']=K_Fold_CV(classifier_DT)

classifier_DT.fit(x_train,y_train)


# In[78]:


y_pred_DT = classifier_DT.predict(x_test)
Overfitting["train_DT,test_DT"]=check_Overfitting(y_pred_DT,classifier_DT)
y_pred_TD_DT=classifier_DT.predict(test_data)
print((np.concatenate((y_pred_DT.reshape(len(y_pred_DT),1), y_test.reshape(len(y_test),1)),1))[:20])


# In[79]:


Confusion_Matrix(y_pred_DT,"Decision Tree")


# <a name='31'></a>
# # 31- Best Accuracy

# In[80]:


print(best_Acc)
print("Best Accuracy: ",max(best_Acc.values()))


# In[81]:


print(best_K_FCV)
print("Best CV Accuracy: ",max(best_K_FCV.values()))


# In[82]:


print(Overfitting)


# In[83]:


submission = pd.DataFrame({
         "PassengerId": test_data_org["PassengerId"],
         "Survived": y_pred_TD_SVM
     })
submission.to_csv('submission.csv', index=False)

