#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Problem Statement
# Titanic is the largest travel agent in the world. Incidentally, titanic having a bad luck. Their ship will sink. Titanic only have limited man power to save the passenger. They need to prioritize passenger saving plan based on their survival rate.
# 
# It is your job to predict if a passenger survived the sinking of the Titanic or not.
# For each in the test set, you must predict a 0 or 1 value for the variable. The target feature for this case is `Survived` feature.
# 
# ## Data description
# ```
# survival 	Survival 	0 = No, 1 = Yes
# pclass 	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd
# sex 	Sex
# Age 	Age in years
# sibsp 	# of siblings / spouses aboard the Titanic
# parch 	# of parents / children aboard the Titanic
# ticket 	Ticket number
# fare 	Passenger fare
# cabin 	Cabin number
# embarked 	Port of Embarkation 	C = Cherbourg, Q = Queenstown, S = Southampton
# ```
# 
# ```
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# ```
# 

# In[2]:


# load the dataset
df = pd.read_csv('/kaggle/input/titanic/train.csv')


# # Exploratory Data Analytics
# You can explore the data through:
# 1. Data descrioption
# 2. Data information
# 3. Data snapshot
# 4. Nullability, check for the null column
# 5. Data distribution for each feature, relative to the target

# In[3]:


# Print the data description
df.describe()


# In[4]:


# Print the data information
df.info()


# In[5]:


# Get the data snapshot (head)
df.head()


# In[6]:


# Check for the nullability
df.isna().sum()


# In[7]:


# Get the data distribution, you can use histogram plot
df.Survived.value_counts().plot.bar()


# In[8]:


df.groupby(['Survived', 'Sex']).PassengerId.count().unstack().plot.bar()


# # Feature Engineering
# You have to handle several things:
# 1. Non numeric data
# 2. Null data
# 3. Related column

# In[9]:


# handle the null data, drop soalnya nullnya banyak bgt
df = df.drop(columns='Cabin')


# In[10]:


df.columns
df = df.drop(columns=['PassengerId', 'Name'])


# In[11]:


# Encode all non numeric column
from sklearn.preprocessing import LabelEncoder

for i in df.columns:
    if (df[i].dtype == df['Sex'].dtype):
        le = LabelEncoder()
        df[i+'_encoded'] = le.fit_transform(df[i])


# In[12]:


df.columns


# In[13]:


# Check for the correlation, protip: visualize using seaborn sns heatmap
sns.heatmap(df[['Survived', 'Pclass','Age', 'SibSp', 'Parch', 'Sex_encoded',
       'Ticket_encoded', 'Embarked_encoded']].corr(), annot=True)


# In[14]:


# drop the unnecessary column
df = df[['Survived', 'Pclass', 'Sex_encoded',
       'Ticket_encoded', 'Embarked_encoded']]


# # Modeling and Evaluation
# You have to split the data, use them as train and validation data. Validate your model prediciton and choose for the best model based on your experiment. If you wanna get a greater insight, you can use kfolds, but it is optional. Here is some model that you can use:
# 1. SVC
# 2. Logistic Regression
# 3. ANN
# 4. etc: choose wisely
# 
# You can use several metric to evaluate the model:
# 1. Accuracy
# 2. F1-Score
# 3. Precision
# 4. Recall
# Protip: use confussion matrix and classification report

# In[15]:


# split the data set, train and val
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[16]:


# train 1st model and evaluate, train as you want, make more model, more feature engineering combination,
# this is up to you. Choose the best design experiment results
# Create an SVM classifier
from sklearn.svm import SVC
svm_classifier = SVC()

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Generate a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[17]:


# train 1st model and evaluate, train as you want, make more model, more feature engineering combination,
# this is up to you. Choose the best design experiment results
# Create an Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Train the classifier
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Generate a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # Summarize:
# Write down your summary here. Write the lesson learned.
