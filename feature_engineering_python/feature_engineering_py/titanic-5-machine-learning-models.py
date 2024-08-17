#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Packages and Models for ML

# In[1]:


import numpy as np # Linear Algebra

import pandas as pd # Dataset related Filtering

import seaborn as sns # Beautiful Graphs
sns.set_style('dark') # Set graph styles to 'dark'

import matplotlib.pyplot as plt # Normal ploating graphs
# show graphs in this notebook only 
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px # For interactive plots


# ignore  the warning
import warnings  
warnings.filterwarnings('ignore') 



# # 1. Data Collection

# * First of all We need to data to train our model to predict better result.
# 
# So, We need the data for train our model

# > ###  Dataset Informations:
# 
# 1. survival - Survival (0 = No; 1 = Yes)
# - class - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - name - Name
# - sex - Sex
# - age - Age
# - sibsp - Number of Siblings/Spouses Aboard
# - parch - Number of Parents/Children Aboard
# - ticket - Ticket Number
# - fare - Passenger Fare
# - cabin - Cabin
# - embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# In[2]:


# Read Train.csv File 
trainDF = pd.read_csv('./../input/titanic/train.csv')

# show first five rows from training dataset
trainDF.head()


# In[3]:


# Read Test.csv File 
testDF = pd.read_csv('./../input/titanic/test.csv')

#  show 5 rows
testDF.head()


# In[4]:


# Print 5 Rows
testDF.head() # train() for last 5 rows


# In[5]:


def show_shape(train, test):
    """ 
    display the shape of train and test DF 
    
    """   
    print(" Shape of Training DF", train.shape)
    print("")
    print(" Shape of Testing DF", test.shape)


# In[6]:


#  to know shape of the training and testing data
show_shape(trainDF, testDF)


# In[7]:


# Create an function to display the information of our train and test dataset. Function can be called multiple time in this notebook.
def show_info(train, test):
    """ 
    display the Information of train and test DF 
    
    """
    
    print("Information of Training DF"+ "-"*10)
    print(train.info())
    print("")
    print("")
    print("")
    print("Information of Testing DF"+ "-"*10)
    print(test.info())


# In[8]:


show_info(trainDF, testDF)


# # 2. Feature Engineering

# * Remove Object type of feature from train and test datasets.
# 
# Here we have some columns to remove from dataset 
# 

# In[9]:


removedFeatures = ['Name', 'Ticket', 'Cabin']

trainDF = trainDF.drop(removedFeatures, axis=1) # remove from train DF
testDF = testDF.drop(removedFeatures, axis=1) # remove from test DF

trainDF.head()


# In[10]:


# Age Feature

trainDF['Age'] = trainDF['Age'].fillna(trainDF['Age'].mean()) # fill for train DF
testDF['Age'] = testDF['Age'].fillna(testDF['Age'].mean()) # fill for test DF


# In[11]:


trainDF['Embarked'].value_counts() # Group Wise count records


# In[12]:


# Fill to Embarked column NA with S
 
trainDF['Embarked'] = trainDF['Embarked'].fillna('S') # for train DF only


# In[13]:


# show info of train and test data set by calling function

show_info(trainDF, testDF)


# # 3. Visualization
# 
# 

# In[14]:


# Show servived graph
 

# Plot Counts for Each survived groupby counts
fig = px.bar(trainDF.Survived.value_counts())

fig.show()


# In[15]:


# Plot Counts for Each survived groupby counts
fig = px.bar(trainDF.groupby(['Survived']).count())

fig.show()


# In[16]:


fig = px.histogram(trainDF, x='Survived', y='Pclass', color='Pclass');
fig.show()


# In[17]:


sns.catplot(x="Pclass", col="Survived", data=trainDF, kind="count");

plt.show()


# In[18]:


fig = px.histogram(trainDF, x='Pclass', y= 'Survived', color='Pclass', )
fig.show()


# In[19]:


#  Pclass wise survived graph 


plt.figure(figsize=(10, 7))

sns.barplot(x= 'Pclass', y='Survived', data=trainDF)
plt.title("Pclass wise survived ")
plt.show()


# In[20]:


# Gender wise Survived graph

fig = px.bar(trainDF, x='Sex', y='Survived', color='Sex')
fig.show()
 


# In[21]:


# Parch and Survived Bar graph
 
plt.figure(figsize=(10, 7))

sns.barplot(x = 'Parch', y= 'Survived', data= trainDF)
plt.title("Parch and Survived Graph")

plt.show()


# In[22]:


# Embarked and Survived bar Graph
plt.figure(figsize=(10, 7))

sns.barplot(x= 'Embarked', y = 'Survived', data= trainDF)
plt.title("Embarked and Survived Graph")

plt.show()


# In[23]:


plt.figure(figsize=(10, 5))
sns.distplot(trainDF.Fare)
plt.title('Distribution of Fares')
plt.show()


# In[24]:


# heatmap show
plt.figure(figsize=(10, 7))
sns.heatmap(trainDF.corr(), cmap='Greens', linewidths=1, annot=True, fmt='.1f')

fig=plt.gcf()
plt.show()


# In[25]:


# show the info
show_info(trainDF, testDF)


# In[26]:


# Fill na with median for Fare feature

testDF["Fare"] = testDF["Fare"].fillna(testDF["Fare"].mean()) # for test DF only


# In[27]:


# Convert sex object values to numeric male=1 and female=0, for both train and test DF

trainDF['Sex'] = trainDF['Sex'].replace({'male': 0, 'female': 1})
testDF['Sex'] = testDF['Sex'].replace({'male': 0, 'female': 1})
 


# In[28]:


# count values for Embarked
print(testDF['Embarked'].value_counts())
print(trainDF['Embarked'].value_counts())


# In[29]:


#  Now, Replace with alphabets to Numbers, for both train and test DF

trainDF['Embarked'] = trainDF['Embarked'].replace({'C': 1, 'S':2, 'Q': 3})
testDF['Embarked'] = testDF['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3})


# * Checking train and test dataset frame with first five rows from both datasets

# In[30]:


print(trainDF.head())
print(testDF.head())


# # 4. Model Prediction
# 

# In[31]:


# Load Accuracy
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


# In[32]:


# Set Prediction value

X_train = trainDF.drop(['PassengerId', 'Survived'], axis=1)
y_train = trainDF['Survived']
X_test = testDF.drop(['PassengerId'], axis=1)



# * Showing Train and test shape of the dataset.

# In[33]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)


# In[34]:


# Load Model
from sklearn.tree import DecisionTreeClassifier


# In[35]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[36]:


# To predict our model

pred = model.predict(X_test)
pred.shape


# In[37]:


# show prediction

accu = model.score(X_train, y_train) # model accuracy
print( "Model Prediction Score", (accu * 100).round(2))


# # 5. Model Submission

# In[38]:


dict = {
    'PassengerId' : testDF['PassengerId'],
    'Survived' : pred
}

new_submission = pd.DataFrame(dict, )
new_submission.shape


# - 
# Other Machine learning scores calculating.

# In[39]:


# Import other Models Classes

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier


# In[40]:


def model_wise_predict(models):
    """ 
    Model Predictions
    
    """
    ans_score = []
    for mdl, filename in models:
        m = mdl
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        m_accuracy = m.score(X_train, y_train)
        ans_score.append((m_accuracy*100).round(2))
        
        dict = {
            'PassengerId' : testDF['PassengerId'],
            'Survived' : pred
        }
        new_submission = pd.DataFrame(dict, )
        
#         Uncomment this line if you want to generate all the csv file for all of the models.
#         new_submission.to_csv(filename, index=False)
        
        
    return ans_score


# In[41]:


#  Using DecisionTreeClassifier Model

#  make list of Models
models = [
    (RandomForestClassifier(n_estimators=300, max_depth=20, random_state=5), 'DTC_submission.csv'),
    (RandomForestClassifier(), 'RFC_submission.csv'),
    (LogisticRegression(), 'LR_submission.csv'),
    (LinearSVC(), 'SVC_submission.csv'),
    (GaussianNB(), 'GNB_submission.csv'),
    (SGDClassifier(), 'SGD_submission.csv'),
    (KNeighborsClassifier(), 'KNC_submission.csv')
]

data = model_wise_predict(models)
print("scores are", data)


# In[42]:


list_model_name = [
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'LogisticRegression', 
    'LinearSVC',
    'GaussianNB',
    'SGDClassifier', 
    'KNeighborsClassifier'
]


# In[43]:


print(X_train, y_train)


# In[44]:


# Customize Model
# TEST

# HYPER TUNNING -------------------------------------------------------------- Start

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

rfc_model = RandomForestClassifier(random_state=45)


rfc_params_grid = {
#     'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
    'n_estimators':[100,250,500,750,1000,1250,1500,1750],
    'max_depth': np.random.randint(1, (len(X_train.columns)*.85),20),
    'max_features': np.random.randint(1, len(X_train.columns),20),
    'min_samples_split':[2,4,6,8,10,20,40,60,100], 
    'min_samples_leaf':[1,3,5,7,9],
    'criterion': ["gini", "entropy"]
}

# gscv_random_classifier = GridSearchCV(estimator = rfc_model, param_grid = rfc_params_grid, cv = 5 , n_jobs = -1, verbose = 5)
gscv_random_classifier = RandomizedSearchCV(rfc_model, rfc_params_grid, cv = 5, n_jobs = -1, verbose = 5)

gscv_random_classifier.fit(X_train, y_train)

pred = gscv_random_classifier.predict(X_test)

print("--------------- START ---------------")

# print(accuracy_score(y_test, pred))
print(gscv_random_classifier.best_estimator_)
print(gscv_random_classifier.best_score_)
print(gscv_random_classifier.best_params_)
bestEstimator = gscv_random_classifier.best_estimator_
bestParams = gscv_random_classifier.best_params_

print("--------------- OVER ---------------")

# HYPER TUNNING -------------------------------------------------------------- END


# 
# 
# > OUTPUT
# 
# 
# ```
# UPDATED Tunning
# 
# --------------- START ---------------
# RandomForestClassifier(max_depth=7, max_features=5, min_samples_leaf=3,
#                        min_samples_split=4, n_estimators=600, random_state=35)
# 0.8361747536250078
# {'criterion': 'gini', 'max_depth': 7, 'max_features': 5, 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 600}
# --------------- OVER ---------------
# 
# 
# --------------- START ---------------
# RandomForestClassifier(criterion='entropy', max_depth=7, max_features=5,
#                        min_samples_leaf=3, min_samples_split=9,
#                        n_estimators=1400, random_state=35)
# 0.8339275626137719
# {'criterion': 'entropy', 'max_depth': 7, 'max_features': 5, 'min_samples_leaf': 3, 'min_samples_split': 9, 'n_estimators': 1400}
# --------------- OVER ---------------
# 
# 
# 
# --------------- START ---------------
# 
# RandomForestClassifier(max_depth=7, min_samples_split=6, n_estimators=850,
#                        random_state=42)
# 0.8271985437197916
# {'criterion': 'gini', 'max_depth': 7, 'max_features': 'auto', 'min_samples_leaf': 1, 'min_samples_split': 6, 'n_estimators': 850}
# --------------- OVER ---------------
# 
# 
# NEW --> Using RandomizedSearchCV
# 
# --------------- START ---------------
# RandomForestClassifier(criterion='entropy', max_depth=7, min_samples_split=4,
#                        n_estimators=750, random_state=42)
# 0.8193396522503296
# {'n_estimators': 750, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 7, 'criterion': 'entropy'}
# --------------- OVER ---------------
# 
# OLD --> Using GridSearchCV
# --------------- START ---------------
# RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=600,
#                        random_state=45)
# 0.8249513527085558
# {'criterion': 'entropy', 'max_depth': 7, 'max_features': 'auto', 'n_estimators': 600}
# --------------- OVER ---------------
# ```
# 
# 

# In[45]:


#  DOWNLOAD SUBMISSION

# Submission FILE EXPORTING  -------------------------------------------------------------- Start
# m = RandomForestClassifier(criterion='entropy', max_depth=7, min_samples_split=4,
#                        n_estimators=750, random_state=42)

m = RandomForestClassifier(criterion='entropy', max_depth=3, max_features=6,
                       min_samples_split=4, n_estimators=1250, random_state=35)

m.fit(X_train, y_train)
pred = m.predict(X_test)

print("Acc: ", m.score(X_train, y_train))

dict = {
    'PassengerId' : testDF['PassengerId'],
    'Survived' : pred
}

new_submission = pd.DataFrame(dict, )
new_submission.to_csv('Random-Forest-GSCV-Hyper Tunning.csv', index=False)

# Submission FILE EXPORTING  -------------------------------------------------------------- END


# 0.78468
# 
# 
# 97.98
# 
# 
# Acc:  0.8799102132435466
# 
# Acc:  0.8832772166105499
# 
# 

# # Prediction Dashboard

# In[46]:


modelDF = pd.DataFrame({"Model_Name" : list_model_name, "Pred_Score": data})
modelDF.sort_values(by='Pred_Score', ascending=False)
modelDF


# --- 
# ---
# 
# <div class="text-center">
#     <h1>That's it Guys,</h1>
#     <h1>🙏</h1>
#     
#         
#         I Hope you guys you like and enjoy it, and learn something interesting things from this notebook, 
#         
#         Even I learn a lots of  things while I'm creating this notebook
#     
#         Keep Learning,
#         Regards,
#         Vikas Ukani.
#     
# </div>
# 
# ---
# ---
# 
# <img src="https://static.wixstatic.com/media/3592ed_5453a1ea302b4c4588413007ac4fcb93~mv2.gif" align="center" alt="Thank You" style="min-height:20%; max-height:20%" width="90%" />
# 
# 
