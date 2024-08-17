#!/usr/bin/env python
# coding: utf-8

# Nest stpes - 
# 
# Impute Fare and Embarked missing values based on other variables

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


sns.set(rc={'figure.figsize':(5,5)})


# In[2]:


train=pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
val=pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')


# ***Combining training and validation data for feature engineering. This is usefull at the later stage when predicting on validation set***

# In[3]:


train['dataset']= "Train"
val['dataset']="Test"


data= pd.concat([train,val],axis=0)


# In[4]:


print(data.shape)
print(data.columns)

print(data.isna().agg('sum'))


# ***Variables contains missing values, We use different methodology for each variable based on best estimation possible ***

# Lets see the distribution of survival based on variable 'Sex'

# In[5]:


sns.countplot(x='Sex',hue='Survived',data=train)


# Distribution of 'Age'

# In[6]:


plt.hist(x=data.Age, bins=10)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Frequency')


# We Can check the ditribution and ratio of survived passenger based on first 2 letters of 'Ticket' variable

# ***TICKET***

# In[7]:


#Creating a variable for ticket with first 2 letters
data['Ticket']=data['Ticket'].fillna('NA')
data['Ticket_1']=data['Ticket'].str.slice(stop=2)



#Plotting the overall distribution
sns.set(rc={'figure.figsize':(11.7,20)})

sns.countplot(y='Ticket_1',data=data, order = data['Ticket_1'].value_counts().index)

sns.set(rc={'figure.figsize':(5,5)})


#Plotting survival rate based on ticket
p=data[['Ticket_1','Survived']].groupby(['Ticket_1']).agg(np.mean)

p=p.reset_index()
p=p.sort_values(['Survived'],ascending=False)
p.plot.bar(x='Ticket_1',y='Survived',figsize=(20,10))




# **It can be seen in the charts above Tickets with PP and PC have higher chance of survival, SO we can built a new binary variable for them. We do the same for lower survival rate as well**

# In[8]:


#Varaible for high survival
conditions = [
    (data['Ticket_1'] == 'PC') | (data['Ticket_1'] == 'PP')]
choices = [1]

data['ticket_pp']=np.select(conditions, choices,default=0)

#Varaible for lower survival
conditions = [
    (data['Ticket_1'] == 'ST') | (data['Ticket_1'] == 'A.')| (data['Ticket_1'] == 'SO')| (data['Ticket_1'] == 'CA')]
choices = [1]

data['ticket_S']=np.select(conditions, choices,default=0)


# In[9]:


data['Cabin_new']=data.Cabin.str.slice(stop=1)
data['Cabin_new']=data['Cabin_new'].fillna('NA')

sns.countplot(x='Cabin_new',hue='Survived',data=data)


# In[10]:


data.Fare.hist(by=[data.Survived])


# In[11]:


sns.countplot(x='Parch',hue='Survived',data=train)

#Passengers with 0 ,4,5,6 Parch are less likely to survive than 1,2,3


# In[12]:


sns.countplot(x='SibSp',hue='Survived',data=train)


# In[13]:


data['Family']=data['SibSp']+data['Parch']+1


sns.countplot(x='Family',hue='Survived',data=data)


# In[14]:


conditions = [
    (data['Family'] == 1),
    (data['Family'] == 2),
    (data['Family'] == 3),
    (data['Family'] == 4),
    (data['Family'] == 5),]
choices = ['1','2','3','4','5']

data['FamSize']=np.select(conditions, choices,default='6')

data['FamSize']=pd.to_numeric(data['FamSize'])
data.FamSize.value_counts()


# **AGE**

# In[15]:


sns.boxplot(x="Pclass",y="Age",data=data)


# In[16]:


sns.boxplot(x="Sex",y="Age",data=data)


# In[17]:


sns.boxplot(x="SibSp",y="Age",data=data)


# In[18]:


sns.boxplot(x="Parch",y="Age",data=data)


# **Average Age varies widely within Sibsp, Pclass, Parch, We can use median ages within these categories to use as proxy data for missing age values**

# In[19]:


age_avg=pd.DataFrame(data[['Age','Pclass','Parch','SibSp']].groupby(['Pclass','Parch','SibSp']).median())

age_avg.reset_index(inplace=True)
age_avg['Age_avg']=age_avg['Age']
age_avg=age_avg.drop('Age',axis=1)
m=np.median(age_avg['Age_avg'].dropna())
age_avg=age_avg.fillna(m)


# In[20]:


data=data.merge(age_avg,on=['Pclass','Parch','SibSp'],how='left')


# In[21]:


data['Age']=data['Age'].fillna(data['Age_avg'])


# In[22]:


sns.countplot(x='Embarked',hue='Survived',data=train)


# In[23]:


print(data['Embarked'].isna().agg('sum'))

data['Embarked']=data['Embarked'].fillna('S')
print(data['Embarked'].isna().agg('sum'))


print(data['Fare'].isna().agg('sum'))
data['Fare']=data['Fare'].fillna(np.mean(data['Fare']))
print(data['Fare'].isna().agg('sum'))


# **Using lable encoding for variable "Sex"**

# In[24]:


conditions = [
    (data['Sex'] == 'male') ]
choices = [1]

data['Sex']=np.select(conditions, choices,default=0)

data.Sex.hist()


# In[25]:


data_final=pd.get_dummies(data,columns=['Pclass','Cabin_new','Embarked'],prefix=['Pclass','Cabin','Embark'])

data_final=data_final.drop(['PassengerId','Ticket_1','Name','Cabin','Ticket','Age_avg'],axis=1)


data_final.head()

data_final.dtypes


# In[26]:


Train_final = data_final[data_final['dataset']=='Train']
Val_final=data_final[data_final['dataset']=='Test']

Train_final=Train_final.drop('dataset',axis=1)
Val_final=Val_final.drop(['dataset','Survived'],axis=1)


# In[27]:


y=Train_final['Survived']
X=Train_final.drop('Survived', axis =1)


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10, test_size=0.2)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

params={'max_depth' : [6,8,10], 
       'n_estimators': [50, 100], 
        'min_samples_split': [2, 3], 
        'min_samples_leaf': [1, 3], 
        'bootstrap': [False]
       }

rf=RandomForestClassifier()

rf_b=GridSearchCV(rf,param_grid=params,cv=3) 

rf_b.fit(X_train,y_train)

print(rf_b.best_score_)

print(rf_b.score(X_test,y_test))

print(rf_b.best_params_)

y_rf_pred=rf_b.predict(X_test)

print(confusion_matrix(y_test,y_rf_pred))

y_rf_prob_pred=rf_b.predict_proba(X_test)

print(roc_auc_score(y_test,y_rf_prob_pred[:,1]))


# d={'Feature':np.array(X_train.columns),'Importance':rf_b.best_estimator_.feature_importances_}
# Features=pd.DataFrame(d)
# Features.sort_values('Importance', inplace=True,ascending=False)

# In[30]:


d={'Feature':np.array(X_train.columns),'Importance':rf_b.best_estimator_.feature_importances_} 
Features=pd.DataFrame(d) 
Features.sort_values('Importance', inplace=True,ascending=False)


# In[31]:


sns.barplot(y='Feature',x='Importance', data=Features,palette="Blues_d")


# In[32]:


from sklearn.ensemble import GradientBoostingClassifier

params={'n_estimators':[200,400], 
#        'learning_rate': [ 0.1,0.2,0.4],
#        'max_features': [2,3,4,7],
        'max_depth': [2,4]}

gb=GradientBoostingClassifier(random_state=42)

gb_b=GridSearchCV(gb,param_grid=params,cv=3)
gb_b.fit(X_train,y_train)

print(gb_b.best_score_)

print(gb_b.score(X_test,y_test))

print(gb_b.best_params_)

y_gb_pred=gb_b.predict(X_test)
print(confusion_matrix(y_test,y_gb_pred))

y_gb_prob_pred=gb_b.predict_proba(X_test)

print(roc_auc_score(y_test,y_gb_prob_pred[:,1]))


# In[33]:


d={'Feature':np.array(X_train.columns),'Importance':gb_b.best_estimator_.feature_importances_}
Features_gb=pd.DataFrame(d)
Features_gb.sort_values('Importance', inplace=True,ascending=False)


# In[34]:


sns.barplot(y='Feature',x='Importance', data=Features_gb,palette="Blues_d")


# In[35]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

xgb = XGBClassifier(use_label_encoder=False,verbosity=1)


parameters = {'eval_metric':["error"],
              'objective':['binary:logistic'],
              'learning_rate': [0.1,0.2], 
              'max_depth': [1,2,5],
              'booster' : ['gbtree'],
              'n_estimators': [200],
              'seed': [1337]}


xgb_b = GridSearchCV(xgb, parameters, 
                   cv=3, 
                   scoring='accuracy',
                    refit=True)

xgb_b.fit(X_train,y_train)

print(xgb_b.best_params_)

print(xgb_b.score(X_test,y_test))


# In[36]:


import lightgbm as lgb

X_lgb_train,X_lgb_eval,y_lgb_train,y_lgb_eval = train_test_split(X_train,y_train,random_state=10, test_size=0.2)


params = {
        "objective" : "binary",
        "eval_metric" : "error",
        "learning_rate" : 0.004,
        "bagging_fraction" : 1,
        "feature_fraction" : 0.9,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
lgtrain = lgb.Dataset(X_lgb_train, label=y_lgb_train)
lgval = lgb.Dataset(X_lgb_eval, label=y_lgb_eval)


model_lgb = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                       early_stopping_rounds=100,
                      verbose_eval=150
                      )
preds=model_lgb.predict(X_test)  
preds=[round(value) for value in preds]

print(accuracy_score(y_test,preds))


# ***Using an ensemble for all classifiers to get the accuracy***

# In[37]:


ensemble=VotingClassifier(estimators=[ ('XGBoost', xgb_b.best_estimator_), ('Random Forest', rf_b.best_estimator_), ('Gradient boosting', gb_b.best_estimator_)], voting='soft', weights=[1,1,1]).fit(X_train,y_train) 
print('The accuracy for Ensemble is:',ensemble.score(X_test,y_test))


# In[38]:


print(Val_final['Fare'].isnull().agg('sum'))

Val_final['Fare']=Val_final['Fare'].fillna(np.mean(Val_final['Fare']))

print(Val_final['Fare'].isnull().agg('sum'))


# In[39]:


pred=gb_b.predict(Val_final)

#pred2=pred.astype('int64')
pred2=[round(value) for value in pred]

submission = pd.DataFrame({'PassengerId': val['PassengerId'],'Survived': pred2})

submission.head()


# In[40]:


submission.to_csv("./Submission.csv", index=False)

