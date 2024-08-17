#!/usr/bin/env python
# coding: utf-8

# # Getting Started with Titanic - Machine Learning from Disaster

# # <img src="https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg">

# 1. # Importing Python Libraries 📕 📗 📘 📙

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# <div class="alert alert-block alert-danger">  
#     <h1><strong>Loading training data</strong></h1>
#     <i></i>
# </div>

# In[2]:


train_data = pd.read_csv("../input/titanic/train.csv")


# # Exploratory data analysis of train data

# # Five top records of data

# In[3]:


train_data.head()


# # Five last records of data

# In[4]:


train_data.tail()


# # Coloumns/features in data

# In[5]:


train_data.columns


# # Length of data

# In[6]:


print('lenght of data is', len(train_data))


# # Shape of data

# In[7]:


train_data.shape


# # Data information

# In[8]:


train_data.info()


# # Data types of all coloumns

# In[9]:


train_data.dtypes


# # Checking missing Values

# In[10]:


train_data[train_data.isnull().any(axis=1)].head()


# # Count of missing values

# In[11]:


np.sum(train_data.isnull().any(axis=1))


# # Is there any missing values?

# In[12]:


train_data.isnull().values.any()


# # Counts of missing values in each column

# In[13]:


train_data.isnull().sum()


# # Looking at the train data missing values.

# In[14]:


NANColumns=[]
i=-1
for a in train_data.isnull().sum():
    i+=1
    if a!=0:
        print(train_data.columns[i],a)
        NANColumns.append(train_data.columns[i])


# # Frequency Distribution of pclass

# In[15]:


carrier_count = train_data["Pclass"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of pclass')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('pclass', fontsize=12)
plt.show()


# In[16]:


train_data["Pclass"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of survived

# In[17]:


carrier_count = train_data["Survived"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of survived    ')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('survived    ', fontsize=12)
plt.show()


# In[18]:


train_data["Survived"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of sex

# In[19]:


train_data["Sex"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of top 10 age

# In[20]:


train_data["Age"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of embarked

# In[21]:


train_data["Embarked"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # All features of train data distrubution 

# In[22]:


train_data.hist(figsize=(15,12),bins = 20, color="#107009AA")
plt.title("Features Distribution")
plt.show()


# <div class="alert alert-block alert-danger">  
#     <h1><strong>Loading testing data</strong></h1>
#     <i></i>
# </div>

# In[23]:


test_data = pd.read_csv("../input/titanic/test.csv")
ids_test_data = test_data['PassengerId'].values


# # Exploratory data analysis of test data

# # Five top records of data

# In[24]:


test_data.head()


# # Five last records of data

# In[25]:


test_data.tail()


# # Coloumns/features in data

# In[26]:


test_data.columns


# # Length of data

# In[27]:


print('lenght of data is', len(test_data))


# # Shape of data

# In[28]:


test_data.shape


# # Data information

# In[29]:


test_data.info()


# # Data types of all coloumns

# In[30]:


test_data.dtypes


# # Checking missing Values

# In[31]:


test_data[test_data.isnull().any(axis=1)].head()


# # Count of missing values

# In[32]:


np.sum(test_data.isnull().any(axis=1))


# # Is there any missing values?

# In[33]:


test_data.isnull().values.any()


# # Counts of missing values in each column

# In[34]:


test_data.isnull().sum()


# # Looking at the test data missing values.

# In[35]:


NANColumns=[]
i=-1
for a in test_data.isnull().sum():
    i+=1
    if a!=0:
        print(test_data.columns[i],a)
        NANColumns.append(test_data.columns[i])


# # Frequency Distribution of pclass

# In[36]:


carrier_count = test_data["Pclass"].value_counts()
sns.set(style="darkgrid")
sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
plt.title('Frequency Distribution of pclass')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('pclass', fontsize=12)
plt.show()


# In[37]:


test_data["Pclass"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of sex

# In[38]:


test_data["Sex"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of top 10 age

# In[39]:


test_data["Age"].value_counts().head(10).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # Frequency Distribution of embarked

# In[40]:


test_data["Embarked"].value_counts().head(7).plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# # All features of test data distrubution 

# In[41]:


test_data.hist(figsize=(15,12),bins = 20, color="#107009AA")
plt.title("Features Distribution")
plt.show()


# # Looking at correlated features with Survived 

# In[42]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# <div class="alert alert-block alert-danger">  
# <h2><center><strong>As we can see from the graphs, features has good correlation with Pclass</strong></center></h2>
#         
# </div>

# # Correlation Survived with Pclass

# In[43]:


train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# - We can see that the correlation of pclass with survived is more than 0.5 among Pclass=1 so we are going to add this feature in training

# # Correlation Survived with SEX

# In[44]:


train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# - We can see that the correlation of Sex with survived is more than 0.5 among Sex=female so we are going to add this feature in training

# # Correlation Survived with SibSp

# In[45]:


train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# - We can see that the siblling with 1 is high correlated with survival but others are lower and zero

# # Correlation Survived with Parch

# In[46]:


train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# - We can see that the Parch with 1 and 2 is high correlated with survival but others are lower and zero

# # Age plot

# In[47]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# - As we can see that most of old age peoples not survived

# # Pclass plot

# In[48]:


grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# - Pclass=3 had most passengers, however most did not survive.
# - Infant passengers in Pclass=2 and Pclass=3 mostly survived. 
# - Most passengers in Pclass=1 survived. 
# - Pclass varies in terms of Age distribution of passengers.

# # Embarked plot

# In[49]:


grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# - Higher fare paying passengers had better survival.
# - Port of embarkation correlates with survival rates. 

# <div class="alert alert-block alert-info">  
# <h2><center><strong>Features engineering and preparation</strong></center></h2>
#         
# </div>

# ## Extract the Survived out from the train data

# In[50]:


y = train_data["Survived"]


# ## Combining the train and test dataset

# In[51]:


all_data = pd.concat([train_data,test_data],axis=0).reset_index(drop=True)


# ## Drop the Survived & PassengerId  columns

# In[52]:


all_data = all_data.drop(["Survived","PassengerId"],axis=1)


# ## A function for checking the missing values

# In[53]:


def missing_value(df):
    number = df.isnull().sum().sort_values(ascending=False)
    number = number[number > 0]
    percentage = df.isnull().sum() *100 / df.shape[0]
    percentage = percentage[percentage > 0].sort_values(ascending=False)
    return  pd.concat([number,percentage],keys=["Total","Percentage"],axis=1)
missing_value(all_data)


# ## Imputing the Missing Values of all data

# ### int = numrical features 
# ### object = categorical features 

# In[54]:


## Imputing the missing values with the Mode because mode fill the values with the most accuring values and best for the categorical features
all_data["Cabin"] = all_data["Cabin"].transform(lambda x: x.fillna(x.mode()[0]))


# In[55]:


## Imputing the missing values with the Mode because mode fill the values with the most accuring values and best for the categorical features
all_data["Embarked"] = all_data["Embarked"].transform(lambda x: x.fillna(x.mode()[0]))


# In[56]:


#Mapping the Age into 5 groups from 0 to 4
all_data['Age']=all_data.loc[ all_data['Age'] <= 16, 'Age'] = 0
all_data['Age']=all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 32), 'Age'] = 1
all_data['Age']=all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 2
all_data['Age']=all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 3
all_data['Age']=all_data.loc[ all_data['Age'] > 64, 'Age'] = 4 


# In[57]:


#Mapping the Fare into 5 groups from 0 to 4
all_data['Fare']=all_data.loc[ all_data['Fare'] <= 7.91, 'Fare'] = 0
all_data['Fare']=all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
all_data['Fare']=all_data.loc[(all_data['Fare'] > 14.454) & (all_data['Fare'] <= 31), 'Fare']   = 2
all_data['Fare']=all_data.loc[ all_data['Fare'] > 31, 'Fare'] = 3
all_data['Fare']=all_data['Fare'] = all_data['Fare'].astype(int)


# In[58]:


#Checking missing values now
missing_value(all_data)


# ## Coverting the categorical features into numeric form by applying the get_dummies function

# In[59]:


all_data = pd.get_dummies(all_data).reset_index(drop=True)


# ## Now splitting the data for training and testing with same index ID's

# In[60]:


n = len(y)
train_data = all_data[:n]
test_data = all_data[n:]


# <div class="alert alert-block alert-info">  
# <h2><center><strong> Building the models for training and testing</strong></center></h2>
#         
# </div>

# <div class="alert alert-block alert-danger">  
# <h2><center><strong> Applying Cross Vaildation on each algorithm</strong></center></h2>
#         
# </div>

# In[61]:


X = np.array(train_data)
y = np.array(y)


# # Random Forest Machine Algorithm

# In[62]:


rf = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2)
kf = KFold(n_splits=5)
outcomes1 = []
ClassR=0
ConM=0
fold = 0
i=0
conf_matrix_list_of_arrays = []
for train_index, test_index in kf.split(X,y):
    i=i+1
    print("KFold Split:",i)
    print('\n')
    fold += 1
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    print('Running time of algorithm')
    get_ipython().run_line_magic('time', 'rf.fit(Xtrain, ytrain)')
    predictions = rf.predict(Xtest)
    accuracy = accuracy_score(ytest, predictions)
    outcomes1.append(accuracy)
    print("Accuracy of KFold ",i, "is: ",accuracy)
    print('\n')
    print("Classification Report of KFold ",i," is following:")
    print('\n')
    CR=classification_report(ytest, predictions)
    print(CR)
    print('\n')
    print("Confusion Matrix of KFold ",i," is following:")
    print('\n')
    CM=confusion_matrix(ytest, predictions)
    conf_matrix_list_of_arrays.append(CM)
    print(CM)
    print('\n')
    print('\n')

print('\n')
print('Average Confusion Matrix')
aa = np.mean(conf_matrix_list_of_arrays, axis=0)

aaa = np.ceil(aa)

b=pd.DataFrame(aaa)
b=b.astype(int)
labels =['Not Survived','Survived']

c=np.array(b)

fig, ax = plot_confusion_matrix(conf_mat=c,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()
print('\n')
print('\n')
mean_outcome1 = np.mean(outcomes1)
print("Total Average Accuracy of Random Forest Classifier is : {0}".format(mean_outcome1)) 


# # KNN Machine Algorithm

# In[63]:


rf = KNeighborsClassifier(n_neighbors=2)
kf = KFold(n_splits=5)
outcomes2 = []
ClassR=0
ConM=0
fold = 0
i=0
conf_matrix_list_of_arrays = []
for train_index, test_index in kf.split(X,y):
    i=i+1
    print("KFold Split:",i)
    print('\n')
    fold += 1
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    print('Running time of algorithm')
    get_ipython().run_line_magic('time', 'rf.fit(Xtrain, ytrain)')
    predictions = rf.predict(Xtest)
    accuracy = accuracy_score(ytest, predictions)
    outcomes2.append(accuracy)
    print("Accuracy of KFold ",i, "is: ",accuracy)
    print('\n')
    print("Classification Report of KFold ",i," is following:")
    print('\n')
    CR=classification_report(ytest, predictions)
    print(CR)
    print('\n')
    print("Confusion Matrix of KFold ",i," is following:")
    print('\n')
    CM=confusion_matrix(ytest, predictions)
    conf_matrix_list_of_arrays.append(CM)
    print(CM)
    print('\n')
    print('\n')

print('\n')
print('Average Confusion Matrix')
aa = np.mean(conf_matrix_list_of_arrays, axis=0)

aaa = np.ceil(aa)

b=pd.DataFrame(aaa)
b=b.astype(int)
labels =['Not Survived','Survived']

c=np.array(b)

fig, ax = plot_confusion_matrix(conf_mat=c,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()
print('\n')
print('\n')
mean_outcome2 = np.mean(outcomes2)
print("Total Average Accuracy of KNN Classifier is : {0}".format(mean_outcome2)) 


# # Decision Trees Machine Algorithm

# In[64]:


rf = DecisionTreeClassifier(random_state=10)
kf = KFold(n_splits=5)
outcomes3 = []
ClassR=0
ConM=0
fold = 0
i=0
conf_matrix_list_of_arrays = []
for train_index, test_index in kf.split(X,y):
    i=i+1
    print("KFold Split:",i)
    print('\n')
    fold += 1
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    print('Running time of algorithm')
    get_ipython().run_line_magic('time', 'rf.fit(Xtrain, ytrain)')
    predictions = rf.predict(Xtest)
    accuracy = accuracy_score(ytest, predictions)
    outcomes3.append(accuracy)
    print("Accuracy of KFold ",i, "is: ",accuracy)
    print('\n')
    print("Classification Report of KFold ",i," is following:")
    print('\n')
    CR=classification_report(ytest, predictions)
    print(CR)
    print('\n')
    print("Confusion Matrix of KFold ",i," is following:")
    print('\n')
    CM=confusion_matrix(ytest, predictions)
    conf_matrix_list_of_arrays.append(CM)
    print(CM)
    print('\n')
    print('\n')

print('\n')
print('Average Confusion Matrix')
aa = np.mean(conf_matrix_list_of_arrays, axis=0)

aaa = np.ceil(aa)

b=pd.DataFrame(aaa)
b=b.astype(int)
labels =['Not Survived','Survived']

c=np.array(b)

fig, ax = plot_confusion_matrix(conf_mat=c,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()
print('\n')
print('\n')
mean_outcome3 = np.mean(outcomes3)
print("Total Average Accuracy of Decision Trees Classifier is : {0}".format(mean_outcome3)) 


# # Comparison of all algorithms Results

# In[65]:


a=pd.DataFrame()
a['outcomes1']=outcomes1
a['outcomes2']=outcomes2
a['outcomes3']=outcomes3

plt.figure(figsize=(25, 10))
plt.subplot(1,1,1)
plt.plot(a.outcomes1.values,color='blue',label='Random Forest')
plt.plot(a.outcomes2.values,color='green',label='KNN')
plt.plot(a.outcomes3.values,color='red',label='Decision Trees')
plt.title('Algorithms Comparison')
plt.xlabel('Number of time')
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()


# In[66]:


a=a.rename(columns={'outcomes1':'Random Forest', 'outcomes2':'KNN','outcomes3':'Decision Tree'})
a.plot(kind='bar',figsize=(25, 10))


# # Comparison of all algorithms Results

# In[67]:


a


# <div class="alert alert-block alert-danger">  
# <h2><center><strong> Best Model is Random Forest as we can see that it performed well on cross validation</strong></center></h2>
#         
# </div>

# In[68]:


final_model = RandomForestClassifier(min_samples_leaf=1, min_samples_split=2)
final_model = final_model.fit(X,y)


# <div class="alert alert-block alert-success">  
# <h1><center><strong> Submitting the classifications on test data</strong></center></h1>
#         
# </div>

# In[69]:


submission_results = pd.read_csv("../input/titanic/gender_submission.csv")
submission_results.iloc[:,1] = np.floor(np.expm1(final_model.predict(test_data)))
submission_results.to_csv('submission_results', index=False)


# # <img src="https://thumbs.dreamstime.com/t/bright-colorful-thank-you-banner-vector-overlapping-letters-118244535.jpg">
