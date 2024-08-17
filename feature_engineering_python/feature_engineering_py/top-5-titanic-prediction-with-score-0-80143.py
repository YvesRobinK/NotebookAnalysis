#!/usr/bin/env python
# coding: utf-8

# # Necessary Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chi2_contingency
import category_encoders as ce
from sklearn.model_selection import GridSearchCV


# # Reading Train and Test Data

# In[192]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')


# In[253]:


train.head(5)


# In[6]:


# Printing number of columns and rows in the dataset
print("There are {} number of rows and {} number of columns in training data".format(train.shape[0],train.shape[1]))
print("There are {} number of rows and {} number of columns in testing data".format(test.shape[0],test.shape[1]))


# In[7]:


# Checking for data imbalanceness if any
sns.countplot(y=train["Survived"])


# ### From the above statistics, it appears that the dataset is partially imbalanced.

# In[8]:


# Checking the type of columns in dataset
train.info()


# In[9]:


# Describing columns statistics
train.describe()


# # Exploratory Data Analysis

# # Handling Missing Values
# We will first check which all columns have the missing values with the help of Visualization.

# In[10]:


train.isnull().sum()


# In[196]:


test.isnull().sum()


# In[197]:


plt.figure(figsize=(12,9))
plt.subplot(1,2,1)
plt.title("Training Data")
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="viridis")

plt.subplot(1,2,2)
plt.title("Testing Data")
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap="viridis")


# **From the above visualization, it appears that the Age,Cabin and Embarked columns contain null values in training data while Age,Cabin and Fare columns contain null values in testing data. The Cabin column contains maximum null values in both the dataset.**

# In[12]:


# Null features
null_features = [feature for feature in train.columns if train[feature].isnull().sum()>=1]
for features in null_features:
    print(features,np.round(train[features].isnull().mean(),4),'%missing values')
    


# In[13]:


# Explore Numerical Variables
numerical_features = [feature for feature in train.columns if train[feature].dtypes!='O']
print("The number of numerical features in the dataset are {}.".format(len(numerical_features)))
train[numerical_features].head()


# In[14]:


# Discrete variables in data
discrete_features = [feature for feature in numerical_features if len(train[feature].unique())<25 and feature not in ['PassengerId','Survived']]
print("The number of discrete variables is: {}".format(len(discrete_features)))
train[discrete_features].head()


# In[15]:


# Understanding the relationship between discrete and target variables
for feature in discrete_features:
    data = train.copy()
    data.groupby(feature)['Survived'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Survived')
    plt.show()


# In[16]:


# Continuous Distribution 
continuous_features = [feature for feature in numerical_features if feature not in discrete_features and feature not in ['PassengerId','Survived']]
print("The number of continuous features are : {}".format(len(continuous_features)))
train[continuous_features].head()


# In[17]:


# Relationship between continous and target variable
for feature in continuous_features:
    data = train.copy()
    data[feature].hist(bins=50)
    plt.xlabel(feature)
    plt.ylabel('Survived')
    plt.show()


# In[18]:


# Detecting Outliers

for feature in numerical_features:
    data = train.copy()
    data.boxplot(column=feature)
    plt.xlabel(feature)
    plt.title(feature)
    plt.show()


# In[19]:


for feature in numerical_features:
    data = train.copy()
    sns.scatterplot(x=data[feature],y=data['Survived'])
    plt.xlabel(feature)
    plt.title(feature)
    plt.show()


# **From box plot and scatterplot, it appears that some of the columns contains outliers but we will leave them as it is since they are acceptable**

# In[20]:


# Computing the correlation of variables
plt.figure(figsize=(20,12))
sns.heatmap(train[numerical_features].corr(),annot=True)


# **From above correlation graph, it seems like there is no correlation amongst and with the target variables**

# In[21]:


# Categorical Features
categorical_feature = [feature for feature in train.columns if feature not in numerical_features]
print("There are {} number of categorical features".format(len(categorical_feature)))
train[categorical_feature].head()


# In[22]:


# Determining the cadinality of categorical data
for feature in categorical_feature:
    print("The feature is {} and its cardinality is {}".format(feature,len(train[feature].unique())))


# **It appears like columns Name,Ticket and Cabin has higher cardinality**

# In[23]:


# Visualizing relationsip between categorical and target values
for feature in ['Sex','Embarked']:
    data = train.copy()
    data.groupby(feature)['Survived'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Survived')
    plt.title(feature)
    plt.show()


# **Observations from above Graphs:**
# **1. Sex: It appears that on an average number of females survived > no. of males.**
# **2. Embarked: It appears the number of person belonging to category 'C' survived the most while in 'S' survived the least.** 

# In[24]:


dummy = train.copy()
plt.figure(figsize=(12,7))
sns.boxplot(x=dummy["Pclass"],y=dummy["Age"])


# **It appears that the median age of people in class 1 is around 37. While it is 29 for class 2 and it is 22 for class 3. We can consider this info. while imputing the missing values in Age Feature**

# In[25]:


# Inspecting Name feature
train["Name"].tail()


# In[46]:


import re
dummy_train = train.copy()
dummy_test = test.copy()
dummy_train["Title"] = dummy_train.Name.apply(lambda x:re.search(' ([A-Z][a-z]+)\. ',x).group(1))
dummy_test["Title"] = dummy_test.Name.apply(lambda x:re.search(' ([A-Z][a-z]+)\. ',x).group(1))
plt.figure(figsize=(10,9))
plt.subplot(1,2,1)
sns.countplot(x="Title",data = dummy_train)
plt.xticks(rotation=45) 

plt.subplot(1,2,2)
sns.countplot(x="Title",data = dummy_test)
plt.xticks(rotation=45)                                              


# In[47]:


dummy_train['Title'] = dummy_train['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
dummy_train['Title'] = dummy_train['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

dummy_test['Title'] = dummy_test['Title'].replace({'Ms':'Miss'})
dummy_test['Title'] = dummy_test['Title'].replace(['Dona', 'Rev', 'Dr',
                                            'Col'],'Special')
plt.subplot(1,2,1)
sns.countplot(x='Title', data=dummy_train);
plt.xticks(rotation=45);

plt.subplot(1,2,2)
sns.countplot(x='Title', data=dummy_test);
plt.xticks(rotation=45);


# ### Data Cleaning
# 

# In[3]:


#Listing Null values in Numerical data
numerical_with_nan = [feature for feature in train.columns if train[feature].isnull().sum()>=1 and train[feature].dtypes!='O' ]
for feature in numerical_with_nan:
    print(feature,np.round(train[feature].isnull().mean(),4),'%missing values')


# In[4]:


# Filling the missing values with median
def impute_numerical(n_feature):
    
        median_value_train = train[n_feature].median()
        median_value_test = test[n_feature].median()
        train[feature+'_nan'] = np.where(train[n_feature].isnull(),1,0)
        train[feature] = train[n_feature].fillna(median_value_train)
        test[feature+'_nan'] = np.where(test[n_feature].isnull(),1,0)
        test[feature] = test[n_feature].fillna(median_value_test)
        return train,test
    
train,test = impute_numerical(numerical_with_nan)
train[numerical_with_nan].isnull().sum()


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


#Listing Null values in Categorical data
categorical_with_nan = [feature for feature in train.columns if train[feature].isnull().sum()>=1 and train[feature].dtypes=='O' ]
for feature in categorical_with_nan:
    print(feature,np.round(train[feature].isnull().mean(),4),'%missing values')


# **We will only handle missing values in Embarked column not the Cabin columns since, it contains lots of missing values**

# In[8]:


# Filling the embarked column with mode count.
def replace_cat_feature(train,test,feature):
    train_data = train.copy()
    test_data = test.copy()
    train_data[feature] = np.where(train_data[feature].isnull(),train_data[feature].mode(),train_data[feature])
    test_data[feature] =  np.where(test_data[feature].isnull(),test_data[feature].mode(),test_data[feature])
    return train_data,test_data

train,test = replace_cat_feature(train,test,['Embarked'])
train[categorical_with_nan].isnull().sum()


# In[9]:


train.head()


# In[10]:


train.isnull().sum()  


# In[11]:


test.isnull().sum()


# In[13]:


# Filling missing values in Fare column in test data
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# In[14]:


test.isnull().sum()


# #### Performing Feature Engineering

# In[15]:


test_data = test.copy()


# In[16]:


# Feature Engineering numerical variables: [Pclass	Age	SibSp	Parch	Fare]

def eng_age(train_data,testing_data):
    train_data['Age_cat'] = pd.qcut(train_data.Age,  q=4, labels=False)
    testing_data['Age_cat'] = pd.qcut(testing_data.Age, q=4, labels=False)
    return train_data,testing_data


def eng_fare(train_data,testing_data):
    train_data["Fare_cat"] = pd.qcut(train_data["Fare"], q=4, labels=False)
    testing_data["Fare_cat"] = pd.qcut(testing_data["Fare"], q=4, labels=False)
    return train_data,testing_data

def eng_family(train_data,testing_data):
    train_data["Family_size"] = train_data["Parch"] + train_data["SibSp"]
    testing_data["Family_size"] = testing_data["Parch"] + testing_data["SibSp"]
    return train_data,testing_data


def feature_eng_numerical(train,test_data):
    train_data = train.copy()
    testing_data = test_data.copy()
    
    train_data,testing_data = eng_family(train_data,testing_data)
    
    train_data,testing_data = eng_age(train_data,testing_data)
    
    train_data,testing_data = eng_fare(train_data,testing_data)
    
    return train_data,testing_data

train,test_data = feature_eng_numerical(train,test_data) 


# In[17]:


# Dropping columns which contribute less.['PassengerId','Cabin','Ticket']

train = train.drop(columns=['PassengerId','Ticket','Cabin'],axis=1)
test_data = test_data.drop(columns=['PassengerId','Ticket','Cabin'],axis=1)


# In[18]:


train.head()


# In[19]:


test_data.head()


# In[20]:


# Feature Engineering Categorical Variable: 'Embarked'
import re

def eng_categorical(train,test_data):
    train_data = train.copy()
    testing_data = test_data.copy()
    
    
    train_data['Embarked_min'] = np.where(train['Embarked']=='Q',1,0)
    train_data['Embarked_max'] = np.where(train['Embarked']=='S',1,0)
    testing_data['Embarked_min'] = np.where(testing_data['Embarked']=='Q',1,0)
    testing_data['Embarked_max'] = np.where(testing_data['Embarked']=='S',1,0)
    
    train_data["Title"] = train_data.Name.apply(lambda x:re.search(' ([A-Z][a-z]+)\. ',x).group(1))
    testing_data["Title"] = testing_data.Name.apply(lambda x:re.search(' ([A-Z][a-z]+)\. ',x).group(1))
    train_data['Title'] = train_data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
    train_data['Title'] = train_data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
    testing_data['Title'] = testing_data['Title'].replace({'Ms':'Miss'})
    testing_data['Title'] = testing_data['Title'].replace(['Dona', 'Rev', 'Dr',
                                                'Col'],'Special')
    return train_data,testing_data

train,test_data = eng_categorical(train,test_data)
train.head()


# In[21]:


train = train.drop(columns=["Name","Age","Fare"],axis=1)
test_data = test_data.drop(columns=["Name","Age","Fare"],axis=1)


# In[22]:


train.head()


# In[23]:


# Encoding Columns with category

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#cols = ['Cabin']
cols = ['Sex','Embarked','Title']

def encoding_columns(train,test_data,cols):
    encoded_train = train.copy()
    encoded_test =  test_data.copy()
    for feature in cols:
        encoded_train[feature] = le.fit_transform(train[feature])
        encoded_test[feature] =  le.fit_transform(test_data[feature])
    return encoded_train,encoded_test


train_encoded,test_encoded = encoding_columns(train,test_data,cols)
train_encoded.head()


# In[24]:


test_encoded.head()


# In[25]:


# Spitting the independent and dependent variables
y_train_splitted = train_encoded[["Survived"]]
x_train = train_encoded.drop(columns=["Survived"],axis=1)


# ### Feature Scaling

# In[27]:


# Standardizing the variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_norm = pd.DataFrame(sc.fit_transform(x_train))
X_train_norm.columns = x_train.columns


X_test_norm = pd.DataFrame(sc.fit_transform(test_encoded))
X_test_norm.columns = test_encoded.columns

X_train_norm.head()


# In[28]:


X_test_norm.head()


# In[30]:


X_train_norm.to_csv("training.csv",index=False)
X_test_norm.to_csv("testing.csv",index=False)


# In[31]:


X_train_normed = pd.read_csv("training.csv")
X_test_normed = pd.read_csv("testing.csv")


# ### Feature Selection

# In[32]:


# We will perform feature selection using SHAP Values
# Using Extra Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X_train_normed,y_train_splitted.values.ravel())


# In[187]:


feat_importances=pd.Series(model.feature_importances_,index=X_train_normed.columns)
feat_importances.nlargest(23).plot(kind="barh")


# In[91]:


'''
X_train_norm_select = X_train_normed
X_train_norm = X_train_norm_select[['Sex','Fare_cat','Age_cat',"Title",'Pclass',
                                    "Family_size","SibSp","Parch","Age_nan","Embarked_max"
                                ]]



# In[372]:


'''
X_test_normed = X_test_normed[['Sex_female','Fare','Age',"Title_Mr",'Pclass',"Sex_male","Ticket_Freq_Count",
                                    "Has_Cabin","Family_size","Title_Mrs","Title_Miss","SibSp","Parch"
                                ]]
                               #'SibSp','Agenan','Embarked','Embarked_max','Parch']]
                               #,'Parch','Mean_Fare',
                                  # 'Age>18','Agenan','Embarked','Embarked_max','Embarked_min']]
                                  '''


# In[34]:


# Spitting the data
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test = train_test_split(X_train_normed,y_train_splitted,test_size=0.1,stratify=y_train_splitted,random_state=0)


# In[35]:


x_train,x_valid,y_train,y_valid = train_test_split(X_train,Y_train,test_size=0.1,stratify=Y_train,random_state=0)


# # Modelling
# Now, we will perform training of data using various Classification Models.

# In[36]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# ## Logistic Regression

# In[102]:


model_1=LogisticRegression(max_iter=500,random_state=0)
model_1.fit(x_train,y_train.values.ravel())
pred = model_1.predict(x_valid)
score_1 = accuracy_score(y_valid,pred)
score_1


# In[103]:


predictions_1 = model_1.predict(x_test)
score_1 = accuracy_score(y_test,predictions_1)
score_1


# In[39]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions_1,target_names=['0','1']))


# ## KNN

# In[97]:


model_2 = KNeighborsClassifier()
model_2.fit(x_train,y_train.values.ravel())
pred = model_2.predict(x_valid)
score_2 = accuracy_score(y_valid,pred)
score_2


# In[98]:


predictions_2 = model_2.predict(x_test)
print(classification_report(y_test,predictions_2,target_names=['0','1']))


# ## SVM

# In[42]:


model_3=SVC(random_state=0)
model_3.fit(x_train,y_train.values.ravel())
pred = model_3.predict(x_valid)
score_3 = accuracy_score(y_valid,pred)
score_3


# In[43]:


predictions_3 = model_3.predict(x_test)
print(classification_report(y_test,predictions_3,target_names=['0','1']))


# ## Naive-Bayes

# In[44]:


model_4=GaussianNB()
model_4.fit(x_train,y_train.values.ravel())
pred = model_4.predict(x_valid)
score_4 = accuracy_score(y_valid,pred)
score_4


# In[45]:


predictions_4 = model_4.predict(x_test)
print(classification_report(y_test,predictions_4,target_names=['0','1']))


# ## Decision Tree

# In[46]:


model_5=DecisionTreeClassifier(random_state=0)
model_5.fit(x_train,y_train.values.ravel())
pred = model_5.predict(x_valid)
score_5 = accuracy_score(y_valid,pred)
score_5


# In[47]:


predictions_5 = model_5.predict(x_test)
print(classification_report(y_test,predictions_5,target_names=['0','1']))


# ## Random Forest (Untuned)

# In[48]:


model_6=RandomForestClassifier(random_state=0)
model_6.fit(x_train,y_train.values.ravel())
pred = model_6.predict(x_valid)
score_6 = accuracy_score(y_valid,pred)
score_6


# In[49]:


predictions_6 = model_6.predict(x_test)
print(classification_report(y_test,predictions_6,target_names=['0','1']))


# ## XGBOOST (Untuned)

# In[50]:


from xgboost import XGBClassifier
model_7 = XGBClassifier()
model_7.fit(x_train,y_train.values.ravel())
pred = model_7.predict(x_valid)
score_7 = accuracy_score(y_valid,pred)
score_7


# In[51]:


predictions_7 = model_7.predict(x_test)
print(classification_report(y_test,predictions_7,target_names=['0','1']))


# ## CatBoost (Untuned)

# In[52]:


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(verbose=2,iterations=500,od_type='Iter')
cat_model.fit(x_train,y_train,eval_set=(x_valid,y_valid))
print(cat_model.best_score_)


# In[53]:


predictions_8 = cat_model.predict(x_test)
print(classification_report(y_test,predictions_8,target_names=['0','1']))


# ## Hyper-parameter Tuning Random Forest Model

# In[55]:


from sklearn.model_selection import RandomizedSearchCV
param_grid={'max_depth':range(3,6),'n_estimators':range(400,700,100),"max_features": range(3,6)}
grid_search = RandomizedSearchCV(RandomForestClassifier(),param_grid,verbose=1,cv=10,n_jobs=-1)
grid_search.fit(x_train,y_train.values.ravel())


# In[56]:


grid_search.best_estimator_


# In[57]:


grid_search_pred = grid_search.predict(x_valid)
score = accuracy_score(y_valid,grid_search_pred)
score


# In[58]:


grid_search_predictions = grid_search.predict(x_test)
print(classification_report(y_test,grid_search_predictions,target_names=['0','1']))


# ## Hyper-parameter Tuning SVM Model

# In[59]:


param_grid={'C': [100],
              'gamma': [0.01,0.001,0.0001],
              'kernel': ['rbf','poly']}
grid_search_1 = GridSearchCV(SVC(),param_grid,verbose=1,cv=10,n_jobs=-1)
grid_search_1.fit(x_train,y_train.values.ravel())


# In[60]:


grid_search_1.best_estimator_


# In[61]:


grid_search_pred_1 = grid_search_1.predict(x_valid)
score = accuracy_score(y_valid,grid_search_pred_1)
score


# In[62]:


grid_search_predictions_1 = grid_search_1.predict(x_test)
print(classification_report(y_test,grid_search_predictions_1,target_names=['0','1']))


# ## Hyper-Parameter Tuning XGBOOST

# In[63]:


from sklearn.model_selection import RandomizedSearchCV
param_grid_xg={"learning_rate" : [0.05] ,
 "max_depth"        : [ 1],
 "min_child_weight" : [ 1],
 "gamma"            : [ 0.0],
 "colsample_bytree" : [ 0.1,0.3],
 "n_estimators"     : [300,400]}
grid_search_xg = RandomizedSearchCV(XGBClassifier(),param_grid_xg,verbose=1,cv=10,n_jobs=-1)
grid_search_xg.fit(x_train,y_train.values.ravel())


# In[64]:


grid_search_xg.best_estimator_


# In[65]:


grid_search_pred_xg = grid_search_xg.predict(x_valid)
score = accuracy_score(y_valid,grid_search_pred_xg)
score


# In[66]:


grid_search_predictions_xg = grid_search_xg.predict(x_test)
print(classification_report(y_test,grid_search_predictions_xg,target_names=['0','1']))


# ## Hyper-tuning CatBoost Classifier

# In[175]:


param_grid_cat = {'iterations': range(10,100,40),
                 'depth': range(1, 8),
                 'learning_rate': [0.03,0.001,0.01,0.1,0.2,0.3],
                 
                 'bagging_temperature': [0.0,0.2,0.4,0.6,0.8,1.0],
                 'border_count': range(1, 255),
                 'l2_leaf_reg': range(2, 30),
                 'scale_pos_weight': [0.01,0.1,0.3,0.5,0.7,0.9,1.0]}


# In[184]:


grid_search_cat = RandomizedSearchCV(CatBoostClassifier(verbose=2,od_type='Iter'),param_grid_cat,verbose=1,cv=10,n_jobs=-1)
grid_search_cat.fit(x_train,y_train.values.ravel())


# In[185]:


grid_search_predictions_cat = grid_search_cat.predict(x_test)
print(classification_report(y_test,grid_search_predictions_cat,target_names=['0','1']))


# ## Using Neural Networks

# In[167]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras import regularizers

model_neural = Sequential()

model_neural.add(Dense(100, activation='relu', input_shape=(12,),kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model_neural.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

model_neural.add(Dense(1, activation='sigmoid'))

model_neural.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.00001),
              metrics=['accuracy'])
                   
model_neural.fit(x_train, y_train,epochs=150, batch_size=1, verbose=1,validation_data=(x_valid,y_valid))


# In[168]:


# Making predictions using test data
predict_neural = model_neural.predict_classes(x_test)
print(classification_report(y_test,predict_neural,target_names=['0','1']))


# #### **Final Model:** Although, most of the models implemented above output accuracy > 80% , but Neural Network is chosen as the final model since, it performed less overfitting on test data.

# # Predictions on Test Data

# In[169]:


final_predictions =  pd.DataFrame(model_neural.predict_classes(X_test_normed))
final_predictions.columns = ["Survived"]
final_predictions = pd.concat([test["PassengerId"],final_predictions],axis=1)
final_predictions.head()


# In[186]:


final_predictions.head()

