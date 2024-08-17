#!/usr/bin/env python
# coding: utf-8

# ## Import Modules & Data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[3]:


# Load the dataset
df = pd.read_csv("/kaggle/input/titanic/train.csv")


# ## Initial Exploratory Data Analysis

# In[4]:


# View the first few rows of the dataset
df.head()



# In[5]:


df.tail()


# In[6]:


df.shape


# In[7]:


# Get information about the dataset
df.info()


# In[8]:


# Generate summary statistics
df.describe()


# ## Visualization

# In[9]:


# Histogram of passenger ages
plt.hist(df["Age"].dropna(), bins=20, edgecolor="k")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Passenger Age Distribution")
plt.show()


# In[10]:


# Bar chart of survival counts based on gender
survived_by_gender = df.groupby("Sex")["Survived"].value_counts().unstack()
survived_by_gender.plot(kind="bar", stacked=True)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Survival Counts by Gender")
plt.legend(["Did not survive", "Survived"])
plt.show()


# ## Missing Data

# In[11]:


# Calculate percentage of missing values
missing_percentage = df.isnull().sum() / len(df) * 100

# Print columns with missing values and their corresponding percentages
print(missing_percentage[missing_percentage > 0])


# Since the 'Cabin' column has over 77% missing values, it seems more reasonable to drop this collumn alltogether. For 'Age' and 'Embarked' we can impute a value, e.g. the mean or median (to lessen impact of outliers).

# In[12]:


df.drop(['Cabin'], axis=1, inplace=True)


# In[13]:


imputer = SimpleImputer(strategy="median")  # Use median for numerical features
df["Age"] = imputer.fit_transform(df[["Age"]]).ravel()  # Fill missing Age values with median


# In[14]:


df.head()


# ## Remove Unique Values Columns

# In[15]:


# Check for columns with unique values
columns_with_unique_values = []
for column in df.columns:
    if df[column].nunique() == len(df):
        columns_with_unique_values.append(column)

# Print the columns with unique values
print("Columns with unique values:", columns_with_unique_values)


# In[16]:


df.drop(['PassengerId'], axis=1, inplace=True) # We'll remove 'Name' later after we've extracted some feature information.


# In[17]:


# Calculate the percentage of unique values in the 'Ticket' column
unique_tickets = df['Ticket'].nunique()
total_tickets = df['Ticket'].count()
percentage_unique_tickets = (unique_tickets / total_tickets) * 100

# Print the percentage of unique tickets
print("Percentage of unique tickets: {:.2f}%".format(percentage_unique_tickets))


# Most of the values in the 'Ticket' column is unique as well. Let's remove it.

# In[18]:


df.drop(['Ticket'], axis=1, inplace=True)


# ## Correlation Analysis

# In[19]:


# Create a correlation matrix
correlation_matrix = df.corr()

# Generate a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# ## Survival Analysis

# In[20]:


# Calculate survival rate by passenger class
survival_by_class = df.groupby("Pclass")["Survived"].mean()

# Bar chart of survival rate by passenger class
sns.barplot(x="Pclass", y="Survived", data=df)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Passenger Class")
plt.show()


# In[21]:


# Box plot of age distribution by survival
sns.boxplot(x="Survived", y="Age", data=df)
plt.xlabel("Survived")
plt.ylabel("Age")
plt.title("Age Distribution by Survival")
plt.show()


# ## Feature Engineering

# In[22]:


# Create a new feature for family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Calculate survival rate by family size
survival_by_family_size = df.groupby("FamilySize")["Survived"].mean()

# Line plot of survival rate by family size
plt.plot(survival_by_family_size.index, survival_by_family_size.values, marker="o")
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Family Size")
plt.show()


# In[23]:


# # Extract the titles from the 'Name' column using regular expressions
# df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Create bins for age groups
bins = [0, 12, 18, 30, 50, 200]  # Adjust the age ranges as desired
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

# Create bins for fare groups
bins = [0, 10, 30, 1000]  # Adjust the fare ranges as desired
labels = ['Low', 'Medium', 'High']
df['FareGroup'] = pd.cut(df['Fare'], bins=bins, labels=labels)

df.drop(['Name'], axis=1, inplace=True)


# In[24]:


# Countplot of embarkation point and survival
sns.countplot(x="Embarked", hue="Survived", data=df)
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.title("Passenger Count by Embarked and Survival")
plt.show()


# ## Encode Categorical Data

# In[25]:


encoder = OneHotEncoder(drop="first")  # Drop first category to avoid multicollinearity
encoded_features = pd.DataFrame(encoder.fit_transform(df[[
    "Sex", 
    "Embarked", 
#     "Title", 
    "AgeGroup", 
    "FareGroup"
]]).toarray(),
                                columns=encoder.get_feature_names_out([
                                    "Sex", 
                                    "Embarked", 
#                                     "Title", 
                                    "AgeGroup", 
                                    "FareGroup"
                                ]))
df_encoded = pd.concat([df, encoded_features], axis=1)

df_encoded.drop([
    'Sex', 
    'Embarked',
#     'Title', 
    "AgeGroup", 
    "FareGroup"
], axis=1, inplace=True)


# In[26]:


df_encoded.head()


# In[27]:


df_encoded.info()


# ## Models 

# In[28]:


# Separate the features (X) and target variable (y)
X = df_encoded.drop("Survived", axis=1)
y = df_encoded["Survived"]


# ### Logistic Regression

# In[29]:


# Instantiate the model
lr_model = LogisticRegression()

# Perform 5-fold cross-validation
scores = cross_val_score(lr_model, X, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())


# ### Multiple Models

# In[30]:


# Instantiate the models
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier(n_estimators=1000)
gradient_boosting = GradientBoostingClassifier()
svm = SVC(kernel='rbf', C=1000) # Can be tuned...
neural_network = MLPClassifier()
xgboost_model = xgb.XGBClassifier()
catboost_model = CatBoostClassifier(verbose=False)
lightgbm_model = lgb.LGBMClassifier()


# In[31]:


# Perform cross-validation for each model
models = [
    ('Random Forest', random_forest),
    ('Gradient Boosting', gradient_boosting),
    ('Support Vector Machines', svm),
    ('Neural Network', neural_network),
    ('XGBoost', xgboost_model),
    ('CatBoost', catboost_model),
    ('LightGBM', lightgbm_model)
]



# In[32]:


model_scores = {}

for name, model in models:
    scores = cross_val_score(model, X, y, cv=5)
    model_scores[name] = scores.mean()
    print(f"{name}:")
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())
    print()

# Find the best and worst models based on cross-validation scores
best_model = max(model_scores, key=model_scores.get)
worst_model = min(model_scores, key=model_scores.get)

print("Best Model:", best_model)
print("Cross-validation score:", model_scores[best_model])
print()

print("Worst Model:", worst_model)
print("Cross-validation score:", model_scores[worst_model])
print()


# In[33]:


# Create an ensemble of all models
ensemble = VotingClassifier(models)
ensemble_scores = cross_val_score(ensemble, X, y, cv=5)
ensemble_score_mean = ensemble_scores.mean()

print("Ensemble of all models:")
print("Cross-validation scores:", ensemble_scores)
print("Mean cross-validation score:", ensemble_score_mean)


# ## Submit on Test

# ### Preprocess Test Data

# In[34]:


# Load the test dataset
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.info()


# In[35]:


imputer = SimpleImputer(strategy="median")  # Use median for numerical features
test_df["Age"] = imputer.fit_transform(test_df[["Age"]]).ravel()  # Fill missing Age values with median


# In[36]:


# # Extract the titles from the 'Name' column using regular expressions
# test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Create bins for age groups
bins = [0, 12, 18, 30, 50, 200]  # Adjust the age ranges as desired
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

# Create bins for fare groups
bins = [0, 10, 30, 1000]  # Adjust the fare ranges as desired
labels = ['Low', 'Medium', 'High']
test_df['FareGroup'] = pd.cut(test_df['Fare'], bins=bins, labels=labels)

test_df.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1


# In[ ]:





# In[37]:


# Calculate percentage of missing values
missing_percentage = test_df.isnull().sum() / len(test_df) * 100

# Print columns with missing values and their corresponding percentages
print(missing_percentage[missing_percentage > 0])


# 'Fare' has missing data in the Test set, while it didn't in Train. Let's impute using the average.

# In[38]:


imputer = SimpleImputer(strategy="mean")  # Use mean for numerical features
test_df["Fare"] = imputer.fit_transform(test_df[["Fare"]]).ravel()  # Fill missing Age values with mean
test_df.info()


# In[39]:


encoder = OneHotEncoder(drop="first")  # Drop first category to avoid multicollinearity
encoded_features = pd.DataFrame(encoder.fit_transform(test_df[[
    "Sex", 
    "Embarked", 
#     "Title", 
    "AgeGroup", 
    "FareGroup"
]]).toarray(),
                                columns=encoder.get_feature_names_out([
                                    "Sex", 
                                    "Embarked", 
#                                     "Title", 
                                    "AgeGroup", 
                                    "FareGroup"
                                ]))
test_df_encoded = pd.concat([test_df, encoded_features], axis=1)

test_df_encoded.drop([
    'Sex', 
    'Embarked',
#     'Title', 
    "AgeGroup", 
    "FareGroup"
], axis=1, inplace=True)




# 'Embarked' didn't have any missing values in Test, while it did in Train. The encoder will therefore not add an 'Embarked_nan' column. Since both our Train and Test df have to match when predicting on Test, let's add the column manually and fill it with 0s.
# 

# In[40]:


# # Add 'Embarked_nan' column filled with 0s 
# test_df_encoded['Embarked_nan'] = 0

test_df_encoded.insert(test_df_encoded.columns.get_loc('Embarked_S') + 1, 'Embarked_nan', 0)


# In[41]:


test_df_encoded.head()


# In[42]:


# Separate the features (X_test) and 'PassengerId'
X_test = test_df_encoded.drop("PassengerId", axis=1)


# ### Fit & Predict

# In[43]:


# Instantiate the ensemble model with the best models from the previous step
ensemble = VotingClassifier(models)

# Fit the ensemble model on the entire training data
ensemble.fit(X, y)


# In[44]:


# Make predictions on the test data
predictions = ensemble.predict(X_test)


# ## Submission

# In[45]:


# Create a DataFrame with 'PassengerId' and 'Survived' columns
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
submission_df.head()


# In[46]:


# Save the predictions to a CSV file
submission_df.to_csv("submission.csv", index=False)

