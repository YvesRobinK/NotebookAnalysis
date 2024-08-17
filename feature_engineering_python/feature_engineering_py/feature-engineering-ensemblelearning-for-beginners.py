#!/usr/bin/env python
# coding: utf-8

# ## About this version
# 
# The following version applied RandomForest Classification to replace missing values for various features, dummification, and improved the result obtained from the previous version using Ensemble Learning, where I applied different classification techniques, such as RandomForest, GradientBoosting, ADABoosting and Logistic Regression.
# 
# The code was not reduced even where possible, so everyone may understand the steps and thinking process, beggining with the feature engineering till the application of the Ensemble Learning.
# 
# I hope you enjoy the code, and please comment if you find any way I could improve it. I will be glad to hear it and up to sugestions. Thank you!

# In[1]:


# Importing libraries

import numpy as np
import pandas as pd
import warnings #Used in order to ignore the warnings messages showed through the notebook
warnings.filterwarnings('ignore') #Setting to ignore the warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Loading the train and test file to a dataframe using Pandas

train_set = pd.read_csv('../input/spaceship-titanic/train.csv')
test_set = pd.read_csv('../input/spaceship-titanic/test.csv')


# ## Describing the variables
# 
# PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# 
# HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# 
# CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. 
# Passengers in cryosleep are confined to their cabins.
# 
# Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# 
# Destination - The planet the passenger will be debarking to.
# 
# Age - The age of the passenger.
# 
# VIP - Whether the passenger has paid for special VIP service during the voyage.
# 
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# 
# Name - The first and last names of the passenger.
# 
# Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# In[3]:


# Checking the data from the first lines

train_set.head()


# In[4]:


# Checking the statistics from the numerical features

train_set.describe() 


# In[5]:


# Checking dtype and missing values for each feature in the train set

train_set.info()


# In[6]:


# Checking dtype and missing values for each feature in the test set

test_set.info()


# ## Concatenating the data before the transformation
# 
# Since we will execute transformations, some that envolves the values both from the train and test data, it is easier to concatenate them and separate after the steps below.

# In[7]:


# Appending the dataframes using pd.concat

df = pd.concat([train_set, test_set])


# In[8]:


# Checking the data

df.info()


# ## Dealing with missing values
# 
# Missing values should be dealt when working with prediction algorithms, in this version I will use classification techniques to replace missing values.

# In[9]:


# First let's drop some columns that will not be used now.


columns_d = df[['Cabin', 'Name', 'Transported']]
target = df[['Transported']]

df = df.drop(columns_d, axis=1)


# In[10]:


# Counting rows with missing values

df.isna().any(axis=1).sum()


# In[11]:


# Printing the percentage of rows that has at least one column with missing value.

print('The total of rows with missing values is: ', round((df.isna().any(axis=1).sum()/len(df.index)*100),2),'%')


# In[12]:


# We will train a model for every column using the features with the best correlations as possible
# For now we should drop the rows with missing values

df_new = df.dropna()


# In[13]:


# Counting rows with missing values

df_new.isna().any(axis=1).sum()


# ## Approaching missing values
# 
# Lets take a deep look into data, and instead of replacing missing values with 'Unknown' as the first version, I will search for correlation between features and input what is the most probable value using classification techniques for binary values (in most cases).

# ### Gettting dummies for categorical variables
# 
# When working with categorical values that are not presented as ordinal, we should transform it into what is called Dummies. So in this case, categorical values with more than 2 categories will be transformed into columns, and the row that contains the dummy from that column will be replaced by 1, while the others will be zero!

# In[14]:


# We should get dummies for 'HomePlanet' and 'Destination'

dummies1 = pd.get_dummies(df_new['HomePlanet'])
dummies2 = pd.get_dummies(df_new['Destination'])


# In[15]:


# Concatenate the new dummies columns and dropping the original column.

df_new = pd.concat([df_new.drop('HomePlanet', axis = 1), dummies1], axis = 1)
df_new = pd.concat([df_new.drop('Destination', axis = 1), dummies2], axis = 1)


# In[16]:


# Checking the new columns! As described, both HomePlanet and Destination were transformed into 3 dummies.
# The rows that before denoted a categorical string value, now are represented by 1 just in the column that 
# the value is True, and 0 in the column that the value is False.

df_new.head()


# ### Casting int into the object dtype
# 
# Since CryoSleep and VIP are True/False values as object dtypes, we should cast/transform them into int values, this way we can find the correlation with other features later.

# In[17]:


# Casting 'CryoSleep' and 'VIP' as int, so we can get the correlation

df_new['CryoSleep'] = df_new['CryoSleep'].astype(int)
df_new['VIP'] = df_new['VIP'].astype(int)


# ## Checking correlation of the features

# In[18]:


# Creating the corr object that get the values of correlation of features from df_new

corr = df_new.corr()


# In[19]:


# Using seaborn to show the correlation map

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize = (15,10))         # Sample figsize in inches
sns.heatmap(corr, cmap = "Blues", annot = True, linewidths = 0.5, ax = ax)


# ## Working out the CryoSleep missing values

# In[20]:


# For CryoSleep, we can use RoomService, FoodCourt, ShoppingMall, Spa and VRDeck as predictors.
# Once the passenger in CryoSleep are confined to the Cabin, does make sense it will not be spending
# any money in the luxury amenities.

# The opposite is not totally true though, it is possible that some passengers that not spend money,
# are not in CryoSleep. 

cryo_feat = df_new[['PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
cryo_label = df_new['CryoSleep']


# In[21]:


# Let's start using the standard hyperparameters of RandomForestClassifier and evaluate with Cross Validation Score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier(random_state = 42)
rfc_scores = cross_val_score(rfc, cryo_feat, cryo_label, cv = 3)


# In[22]:


# Printing the mean accuracy between de 3 folds

print("The Mean Accuracy is %0.3f (+/- %0.3f)" % (rfc_scores.mean().mean(), rfc_scores.mean().std() * 2))


# ## Tuning in hyperparameters for CryoSleep prediciton

# In[23]:


# Now let's use the entire data, split it in train, validation and test set to tune in hyperparameters

# Importing train_test_split

from sklearn.model_selection import train_test_split


# In[24]:


# X and y are the cryo_feat and cryo_label, respectively

X_train, X_test, y_train, y_test = train_test_split(cryo_feat, cryo_label, test_size = 0.3, random_state = 42)


# In[25]:


# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# ## Using GridSearchCV
# 
# When it comes to machine learning, GridSearchCV is a popular method for exhaustively exploring a range of hyperparameters associated with a given model. In this context, hyperparameters are those adjustable settings that dictate a model's structure and behavior. 
# 
# Using it we can get the best hyperparameters to the RandomForest to predict the missing values in this case.

# In[26]:


from sklearn.model_selection import GridSearchCV

# Define a range of values for the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# In[27]:


# Create a GridSearchCV object

grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 3)


# In[28]:


# Fit the GridSearchCV object to the training data

grid_search.fit(X_train, y_train)


# In[29]:


# Importing the accuracy_score and f1_score to evaluate

from sklearn.metrics import accuracy_score, f1_score


# In[30]:


# Get the best hyperparameters and evaluate the model on the validation set

best_rfc = grid_search.best_estimator_
y_pred_val = best_rfc.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)


# In[31]:


# Printing the results obtained

print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[32]:


# Evaluate the performance of the selected model on the test set

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[33]:


# evaluate the model using cross-validation into the entire data
best_rfc_scores = cross_val_score(best_rfc, cryo_feat, cryo_label, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_rfc_scores.mean(), best_rfc_scores.std() * 2))


# ## Using the tuned model to predict CryoSleep missing values
# 
# Since it improved the accuracy, let's use this parameters to predict the missing values previously excluded.

# In[34]:


# Get only the rows where column 'CryoSleep' is missing

cryo_miss = df[df['CryoSleep'].isnull()]


# In[35]:


# Getting the features of the missing values

m_cryo_feat = cryo_miss[['PassengerId', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]


# In[36]:


# We can have some problems with missing values in other columns other than 'CryoSleep'
# In this case, it seems reasonable inputing the average of the other values spent in ammenities
# as the missing value in a specific column.

# Compute the average of the columns and store it in 'avg'

avg = m_cryo_feat[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(axis = 1)


# In[37]:


# Fill missing values in the columns with the average

# Loop through columns and replace missing values with the average 'avg'

for col in m_cryo_feat.columns:
    m_cryo_feat[col].fillna(avg, inplace = True)


# In[38]:


# Predicting the labels

cryo_pred = best_rfc.predict(m_cryo_feat)


# In[39]:


# Creating a dataframe with 'PassengerId' and 'CryoSleep'
 
cryo_replace = pd.DataFrame(zip(m_cryo_feat['PassengerId'], cryo_pred),
                                columns=['PassengerId', 'CryoSleep'])


# In[40]:


# Create a dictionary mapping ID to Value

id_value_dict = dict(zip(cryo_replace['PassengerId'].values, cryo_replace['CryoSleep'].values))


# In[41]:


# Loop through the PassengerId's and update the corresponding values to the missing values

for id, value in id_value_dict.items():
    df.loc[df['PassengerId'] == id, 'CryoSleep'] = value


# ## Repeat the process of dropping rows with missing values, and checking correlation
# 
# We will repeat this process iteratively for the missing values remaining.

# In[42]:


# We will train a model for every column using the features with the best correlations as possible
# For now we should drop the rows with missing values

df_new = df.dropna()


# In[43]:


# We should get dummies for 'HomePlanet' and 'Destination'

dummies1 = pd.get_dummies(df_new['HomePlanet'])
dummies2 = pd.get_dummies(df_new['Destination'])


# In[44]:


# Concatenate the new dummies columns and dropping the original column.

df_new = pd.concat([df_new.drop('HomePlanet', axis = 1), dummies1], axis = 1)
df_new = pd.concat([df_new.drop('Destination', axis = 1), dummies2], axis = 1)


# In[45]:


# Casting 'CryoSleep' and 'VIP' as int, so we can get the correlation

df_new['CryoSleep'] = df_new['CryoSleep'].astype(int)
df_new['VIP'] = df_new['VIP'].astype(int)


# In[46]:


# Getting corr again

corr = df_new.corr()


# In[47]:


# Using seaborn to show the correlation map

fig, ax = plt.subplots(figsize = (15,10))         # Sample figsize in inches
sns.heatmap(corr, cmap = "Blues", annot = True, linewidths = 0.5, ax = ax)


# ## Working out the HomePlanet missing values

# In[48]:


# For HomePlanet dummies, we can use Age, FoodCourt, Spa, VRDeck and Destination dummies as predictors
# as they show that their correlation can cause a higher impact than the others features.

HP_feat = df_new[['PassengerId', 'Age', 'FoodCourt', 'Spa', 'VRDeck', '55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']]
HP_label = df_new[['Earth', 'Europa', 'Mars']]


# In[49]:


# Let's start using the standard hyperparameters of RandomForestClassifier and evaluate with Cross Validation Score

rfc = RandomForestClassifier(random_state = 42)
rfc_scores = cross_val_score(rfc, HP_feat, HP_label, cv = 3)


# In[50]:


# Print the mean accuracy

print("The Mean Accuracy is %0.3f (+/- %0.3f)" % (rfc_scores.mean().mean(), rfc_scores.mean().std() * 2))


# ## Tuning hyperparameters to HomePlanet

# In[51]:


# X and y are the HP_feat and HP_label, respectively

X_train, X_test, y_train, y_test = train_test_split(HP_feat, HP_label, test_size = 0.3, random_state = 42)


# In[52]:


# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# In[53]:


# Define a range of values for the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# In[54]:


# Create a GridSearchCV object

grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 3)


# In[55]:


# Fit the GridSearchCV object to the training data

grid_search.fit(X_train, y_train)


# In[56]:


# Get the best hyperparameters and evaluate the model on the validation set

best_rfc = grid_search.best_estimator_
y_pred_val = best_rfc.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, average = 'micro')


# In[57]:


print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[58]:


# Evaluate the performance of the selected model on the test set

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[59]:


# evaluate the model using cross-validation into the entire data
best_rfc_scores = cross_val_score(best_rfc, HP_feat, HP_label, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_rfc_scores.mean(), best_rfc_scores.std() * 2))


# ## Using the chosen model to predict HomePlanet missing values
# 
# Let's use the chosen model to predict the missing values previously excluded. Let's get rows where 'HomePlanet' is missing, then we will work with it.

# In[60]:


# Get only the rows where column 'Destination' is missing

HP_miss = df[df['HomePlanet'].isnull()]


# In[61]:


# Getting the features of the missing values

m_HP_feat = HP_miss[['PassengerId','Age', 'FoodCourt', 'Spa', 'VRDeck', 'Destination']]


# In[62]:


# We need to use get_dummies to get the same columns as we trained the model

dummies = pd.get_dummies(m_HP_feat['Destination'])


# In[63]:


# Concatenate as columns

m_HP_feat = pd.concat([m_HP_feat.drop('Destination', axis=1), dummies], axis=1)


# In[64]:


# First let's impute the average age inplace as missing values in 'Age'

m_HP_feat['Age'] = m_HP_feat['Age'].fillna(m_HP_feat['Age'].mean())


# In[65]:


# We can have some problems with missing values in the ammenities columns
# In this case, it seems reasonable inputing the average of the values spent in ammenities
# as the missing value in the FoodCourt, Spa and VRDeck

# Compute the average of the columns and store it in 'avg'

avg = m_HP_feat[['FoodCourt', 'Spa', 'VRDeck']].mean(axis = 1)


# In[66]:


# Fill missing values in the columns with the average

# Loop through columns and replace missing values with the average 'avg'

for col in m_HP_feat.columns:
    m_HP_feat[col].fillna(avg, inplace = True)


# In[67]:


# Predicting the labels

HP_pred = best_rfc.predict(m_HP_feat)


# In[68]:


# Define list of string values
string_values = 'Earth', 'Europa', 'Mars'


# In[69]:


# Use argmax() to find index of column with highest value for each row

index_array = np.argmax(HP_pred, axis=1)


# In[70]:


# Map indices to corresponding string values

string_array = np.array([string_values[i] for i in index_array])


# In[71]:


# Reshape string values into new array with single column
HP_pred = string_array.reshape(-1, 1)


# In[72]:


# Creating a dataframe with 'PassengerId' and 'Destination'
 
HP_rep = pd.DataFrame(zip(m_HP_feat['PassengerId'], HP_pred),
                                columns=['PassengerId', 'HomePlanet'])


# In[73]:


# Create a dictionary mapping ID to Value

id_value_dict = dict(zip(HP_rep['PassengerId'].values, HP_rep['HomePlanet'].values))


# In[74]:


# Loop through the PassengerId's and update the corresponding values to the missing values

for id, value in id_value_dict.items():
    df.loc[df['PassengerId'] == id, 'HomePlanet'] = value


# ## Repeat the process of dropping rows with missing values, and checking correlation

# In[75]:


# We will train a model for every column using the features with the best correlations as possible
# For now we should drop the rows with missing values

df_new = df.dropna()


# In[76]:


# We should get dummies for 'HomePlanet' and 'Destination'

dummies1 = pd.get_dummies(df_new['HomePlanet'])
dummies2 = pd.get_dummies(df_new['Destination'])


# In[77]:


# Concatenating and drop columns

df_new = pd.concat([df_new.drop('HomePlanet', axis = 1), dummies1], axis = 1)
df_new = pd.concat([df_new.drop('Destination', axis = 1), dummies2], axis = 1)


# In[78]:


# Casting 'CryoSleep' and 'VIP' as int, so we can get the correlation

df_new['CryoSleep'] = df_new['CryoSleep'].astype(int)
df_new['VIP'] = df_new['VIP'].astype(int)


# In[79]:


corr = df_new.corr()


# In[80]:


# Using seaborn to show the correlation map

fig, ax = plt.subplots(figsize = (15,10))         # Sample figsize in inches
sns.heatmap(corr, cmap = "Blues", annot = True, linewidths = 0.5, ax = ax)


# ## Working out the Destination missing values

# In[81]:


# For Destination, we can use FoodCourt, Spa, VRDeck and HomePlanet dummies as predictors
# as they show that their correlation can cause a higher impact than the others features.

dest_feat = df_new[['PassengerId', 'FoodCourt', 'VRDeck', 'Earth', 'Europa', 'Mars']]
dest_label = df_new[['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']]


# In[82]:


# Let's start using the standard hyperparameters of RandomForestClassifier and evaluate with Cross Validation Score

rfc = RandomForestClassifier(random_state = 42)
rfc_scores = cross_val_score(rfc, dest_feat, dest_label, cv = 3)


# In[83]:


print("The Mean Accuracy is %0.3f (+/- %0.3f)" % (rfc_scores.mean().mean(), rfc_scores.mean().std() * 2))


# In[84]:


# X and y are the cryo_feat and cryo_label, respectively

X_train, X_test, y_train, y_test = train_test_split(dest_feat, dest_label, test_size = 0.3, random_state = 42)


# In[85]:


# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# ## Tuning in hyperparameters to predict Destination

# In[86]:


# Define a range of values for the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# In[87]:


# Create a GridSearchCV object
grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 3)


# In[88]:


# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


# In[89]:


# Get the best hyperparameters and evaluate the model on the validation set

best_rfc = grid_search.best_estimator_
y_pred_val = best_rfc.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, average = 'micro')


# In[90]:


print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[91]:


# Evaluate the performance of the selected model on the test set

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[92]:


# evaluate the model using cross-validation into the entire data
best_rfc_scores = cross_val_score(best_rfc, dest_feat, dest_label, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_rfc_scores.mean(), best_rfc_scores.std() * 2))


# ## Using the tuned model to predict Destination values
# 
# Let's use this parameters to predict the missing values previously excluded. In this case, since Destination transformed into 3 dummies, let's get rows where 'Destination' is missing, then we will work with it.

# In[93]:


# Get only the rows where column 'Destination' is missing

dest_miss = df[df['Destination'].isnull()]


# In[94]:


# Getting the features of the missing values

m_dest_feat = dest_miss[['PassengerId', 'FoodCourt', 'VRDeck', 'HomePlanet']]


# In[95]:


# We need to use get_dummies to get the same columns as we trained the model

dummies = pd.get_dummies(m_dest_feat['HomePlanet'])


# In[96]:


# Concatenate as columns

m_dest_feat = pd.concat([m_dest_feat.drop('HomePlanet', axis = 1), dummies], axis = 1)


# In[97]:


# We can have some problems with missing values in the ammenities columns
# In this case, it seems reasonable inputing the average of the values spent in ammenities
# as the missing value in the FoodCourt, Spa and VRDeck

# Compute the average of the columns and store it in 'avg'

avg = m_dest_feat[['FoodCourt', 'VRDeck']].mean(axis = 1)


# In[98]:


# Fill missing values in the columns with the average

# Loop through columns and replace missing values with the average 'avg'

for col in m_dest_feat.columns:
    m_dest_feat[col].fillna(avg, inplace = True)


# In[99]:


# Predicting the labels

dest_pred = best_rfc.predict(m_dest_feat)


# In[100]:


# Define list of string values
string_values = '55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e'


# In[101]:


# Use argmax() to find index of column with highest value for each row

index_array = np.argmax(dest_pred, axis=1)


# In[102]:


# Map indices to corresponding string values

string_array = np.array([string_values[i] for i in index_array])


# In[103]:


# Reshape string values into new array with single column
dest_pred = string_array.reshape(-1, 1)


# In[104]:


# Creating a dataframe with 'PassengerId' and 'Destination'
 
dest_rep = pd.DataFrame(zip(m_dest_feat['PassengerId'], dest_pred),
                                columns=['PassengerId', 'Destination'])


# In[105]:


# Create a dictionary mapping ID to Value

id_value_dict = dict(zip(dest_rep['PassengerId'].values, dest_rep['Destination'].values))


# In[106]:


# Loop through the PassengerId's and update the corresponding values to the missing values

for id, value in id_value_dict.items():
    df.loc[df['PassengerId'] == id, 'Destination'] = value


# ## Repeat the process of dropping rows with missing values, and checking correlation

# In[107]:


# We will train a model for every column using the features with the best correlations as possible
# For now we should drop the rows with missing values

df_new = df.dropna()


# In[108]:


# We should get dummies for 'HomePlanet' and 'Destination'

dummies1 = pd.get_dummies(df_new['HomePlanet'])
dummies2 = pd.get_dummies(df_new['Destination'])


# In[109]:


df_new = pd.concat([df_new.drop('HomePlanet', axis = 1), dummies1], axis = 1)
df_new = pd.concat([df_new.drop('Destination', axis = 1), dummies2], axis = 1)


# In[110]:


# Casting 'CryoSleep' and 'VIP' as int, so we can get the correlation

df_new['CryoSleep'] = df_new['CryoSleep'].astype(int)
df_new['VIP'] = df_new['VIP'].astype(int)


# In[111]:


corr = df_new.corr()


# In[112]:


# Using seaborn to show the correlation map

fig, ax = plt.subplots(figsize = (15,10))         # Sample figsize in inches
sns.heatmap(corr, cmap = "Blues", annot = True, linewidths = 0.5, ax = ax)


# ## Working out the VIP missing values

# In[113]:


# For VIP, we can use FoodCourt, VRDeck and HomePlanet dummies as predictors
# as they show that their correlation can cause a higher impact than the others features.

vip_feat = df_new[['PassengerId', 'FoodCourt', 'VRDeck', 'Earth', 'Europa', 'Mars']]
vip_label = df_new[['VIP']]


# In[114]:


# Let's start using the standard hyperparameters of RandomForestClassifier and evaluate with Cross Validation Score

rfc = RandomForestClassifier(random_state = 42)
rfc_scores = cross_val_score(rfc, vip_feat, vip_label, cv = 3)


# In[115]:


print("The Mean Accuracy is %0.3f (+/- %0.3f)" % (rfc_scores.mean().mean(), rfc_scores.mean().std() * 2))


# In[116]:


# X and y are the cryo_feat and cryo_label, respectively

X_train, X_test, y_train, y_test = train_test_split(vip_feat, vip_label, test_size = 0.3, random_state = 42)


# In[117]:


# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# ## Tuning in hyperparameters to predict VIP

# In[118]:


# Define a range of values for the hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# In[119]:


# Create a GridSearchCV object
grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 3)


# In[120]:


# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)


# In[121]:


# Get the best hyperparameters and evaluate the model on the validation set

best_rfc = grid_search.best_estimator_
y_pred_val = best_rfc.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)


# In[122]:


print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[123]:


# Evaluate the performance of the selected model on the test set

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[124]:


# evaluate the model using cross-validation into the entire data
best_rfc_scores = cross_val_score(best_rfc, vip_feat, vip_label, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_rfc_scores.mean(), best_rfc_scores.std() * 2))


# ## Using the model to predict VIP values
# 
# Let's use the model to predict the missing values previously excluded. In this case, let's get rows where 'VIP' is missing, then we will work with it.

# In[125]:


# Get only the rows where column 'Destination' is missing

vip_miss = df[df['VIP'].isnull()]


# In[126]:


# Getting the features of the missing values

m_vip_feat = vip_miss[['PassengerId', 'FoodCourt', 'VRDeck', 'HomePlanet']]


# In[127]:


# We need to use get_dummies to get the same columns as we trained the model

dummies = pd.get_dummies(m_vip_feat['HomePlanet'])


# In[128]:


# Concatenate as columns

m_vip_feat = pd.concat([m_vip_feat.drop('HomePlanet', axis = 1), dummies], axis = 1)


# In[129]:


# We can have some problems with missing values in the ammenities columns
# In this case, it seems reasonable inputing the average of the values spent in ammenities
# as the missing value in the FoodCourt, Spa and VRDeck

# Compute the average of the columns and store it in 'avg'

avg = m_vip_feat[['FoodCourt', 'VRDeck']].mean(axis = 1)


# In[130]:


# Fill missing values in the columns with the average

# Loop through columns and replace missing values with the average 'avg'

for col in m_vip_feat.columns:
    m_vip_feat[col].fillna(avg, inplace = True)


# In[131]:


# Predicting the labels

vip_pred = best_rfc.predict(m_vip_feat)


# In[132]:


# Creating a dataframe with 'PassengerId' and 'Destination'
 
vip_rep = pd.DataFrame(zip(m_vip_feat['PassengerId'], vip_pred),
                                columns=['PassengerId', 'VIP'])


# In[133]:


# Create a dictionary mapping ID to Value

id_value_dict = dict(zip(vip_rep['PassengerId'].values, vip_rep['VIP'].values))


# In[134]:


# Loop through the PassengerId's and update the corresponding values to the missing values

for id, value in id_value_dict.items():
    df.loc[df['PassengerId'] == id, 'VIP'] = value


# ## Replacing Age and ammenities missing values
# 
# In this case we will not be using Regression algorithms to predict values, let's replacing using the technique known as mean imputation. Maybe we could use some regression techinques in the next version.

# ### Age

# In[135]:


# First let's check how many values are missing in Age

df['Age'].isna().sum()


# In[136]:


# Checking statistics from Age

df['Age'].describe()


# In[137]:


# It does seem reasonable replacing Age missing values with mean without losing too much in prediction
# Let's replace it using the replace()

df['Age'].fillna(df['Age'].mean(), inplace=True)


# ### RoomService, FoodCourt, ShoppingMall, Spa, VRDeck

# In[138]:


# In this case, it seems reasonable inputing the average of the other values spent in ammenities
# as the missing value in a specific column.

# Compute the average of the columns and store it in 'avg'

col = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']]
avg = col.mean(axis = 1)


# In[139]:


# Fill missing values in the columns with the average

# Loop through columns and replace missing values with the average 'avg'

for col in df.columns:
    df[col].fillna(avg, inplace = True)


# ## Working with the final dataframe!

# ### Inserting dummies

# In[140]:


# We should get dummies for 'HomePlanet' and 'Destination' first

dummies1 = pd.get_dummies(df['HomePlanet'])
dummies2 = pd.get_dummies(df['Destination'])


# In[141]:


df = pd.concat([df.drop('HomePlanet', axis = 1), dummies1], axis = 1)
df = pd.concat([df.drop('Destination', axis = 1), dummies2], axis = 1)


# ### Casting features into int dtype

# In[142]:


# Casting 'CryoSleep' and 'VIP' as int, so we can get the correlation

df['CryoSleep'] = df['CryoSleep'].astype(int)
df['VIP'] = df['VIP'].astype(int)


# ## Separating the train set from the test set

# In[143]:


# Getting the 8693 rows to the train_df and the remaining to the test_df

train_df = df.iloc[:8693, :]
test_df = df.iloc[8693:, :]


# In[144]:


# Checking the train_df

train_df.info()


# In[145]:


# Checking the test_df

test_df.info()


# In[146]:


# Retrieving the target to concatenate back to the dataframe

target = target.iloc[:8693, :]


# ### Inserting the labels in the train set back

# In[147]:


# Concatenate the 'Transported' label as column

train_df = pd.concat([train_df, target], axis=1)


# In[148]:


train_df.head()


# ### Transforming the target to int

# In[149]:


train_df['Transported'] = train_df['Transported'].astype('int64')


# In[150]:


train_df.info()


# ## Checking the correlation between features and target

# In[151]:


# Compute the standard correlation coefficient (also called Pearson’s r) 
# between every pair of attributes using the corr() method:

corr = train_df.corr()


# In[152]:


# Using seaborn to show the correlation map

fig, ax = plt.subplots(figsize = (15,10))         # Sample figsize in inches
sns.heatmap(corr, cmap = "Blues", annot = True, linewidths = 0.5, ax = ax)


# ## Getting the best RandomForest Classifier

# In[153]:


# Let's get the features and targets from the train_df, after we will split into train, validation and test sets!

features = train_df.drop(['Transported'], axis = 1)
target = train_df[['Transported']]


# In[154]:


# Splitting the features and targets into train and test, using 70% and 30% respectively.

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)


# In[155]:


# Split the training set into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# In[156]:


# Fit the model to the training data

rfc.fit(X_train, y_train)


# In[157]:


# Predict the target variable for the validation data

y_pred_val = rfc.predict(X_val)


# In[158]:


# Evaluate the performance on the validation set

accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)


# In[159]:


# Predict the target variable for the test data

y_pred_test = rfc.predict(X_test)


# In[160]:


# Evaluate the performance on the test set

accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)


# In[161]:


# Print the scores for validation and test sets

print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)
print("Test Accuracy:", accuracy_test)
print("Test F1 Score:", f1_test)


# ### Tuning in the hyperparameters to the RandomForestClassifier

# In[162]:


# Define a range of values for the hyperparameters

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}


# In[163]:


# Create a GridSearchCV object

grid_search = GridSearchCV(rfc, param_grid = param_grid, cv = 3)


# In[164]:


# Fit the GridSearchCV object to the training data

grid_search.fit(X_train, y_train)


# In[165]:


# Get the best hyperparameters and evaluate the model on the validation set

best_rfc = grid_search.best_estimator_
y_pred_val = best_rfc.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)


# In[166]:


# Printing the results obtained with the best parameters

print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[167]:


# Evaluate the performance of the selected model on the test set

y_pred = best_rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[168]:


# evaluate the model using cross-validation into the entire data
best_rfc_scores = cross_val_score(best_rfc, features, target, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_rfc_scores.mean(), best_rfc_scores.std() * 2))


# ## Getting the best GradientBoostingClassifier

# In[169]:


# Importing the GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier


# In[170]:


# Define the Gradient Boosting Classifier and set the hyperparameters

gb_clf = GradientBoostingClassifier(random_state = 42)


# In[171]:


# Define the hyperparameters to be tuned

param_grid = {'learning_rate': [0.15, 0.1, 0.05, 0.01],
              'max_depth': [2, 4, 6],
              'n_estimators': [50, 100, 200]}


# In[172]:


# Use GridSearchCV to find the best hyperparameters

grid_search = GridSearchCV(gb_clf, param_grid, cv = 3)
grid_search.fit(X_train, y_train)


# In[173]:


# Get the best hyperparameters and evaluate the model on the validation set

best_gb_clf = grid_search.best_estimator_
y_pred_val = best_gb_clf.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val)


# In[174]:


# Printing the results using the best parameters

print("Best Hyperparameters:", grid_search.best_params_)
print("Validation Accuracy:", accuracy_val)
print("Validation F1 Score:", f1_val)


# In[175]:


# Evaluate the performance of the selected model on the test set

y_pred = best_gb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Test set accuracy:", accuracy)


# In[176]:


# evaluate the model using cross-validation into the entire data
best_gb_clf_scores = cross_val_score(best_gb_clf, features, target, cv = 3)

# print the mean and standard deviation of the scores
print("The Mean Accuracy is: %0.3f (+/- %0.3f)" % (best_gb_clf_scores.mean(), best_gb_clf_scores.std() * 2))


# # ADABoosting

# In[177]:


from sklearn.ensemble import AdaBoostClassifier


# In[178]:


# Define the Adaboost classifier
ada_clf = AdaBoostClassifier(random_state = 42)


# In[179]:


ada_clf.fit(X_train, y_train)


# # Logistic Regression

# In[180]:


from sklearn.linear_model import LogisticRegression


# In[181]:


# Define the parameter grid to search
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}


# In[182]:


# Create a logistic regression model
logreg = LogisticRegression()


# In[183]:


# Create a GridSearchCV object
grid_search = GridSearchCV(logreg, param_grid, cv = 3)


# In[184]:


# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)


# In[185]:


# Create a final model using the best parameters
best_logr = grid_search.best_estimator_


# In[186]:


# Print the best parameters and score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.4f}".format(grid_search.best_score_))


# # Applying Ensemble Learning
# 
# Ensemble learning is a technique in machine learning where multiple models are combined to improve the accuracy and robustness of predictions.
# 
# In ensemble learning, multiple models are trained on the same dataset but with different algorithms or hyperparameters. These models are then combined to make a final prediction that is more accurate and reliable than any individual model.
# 
# Each model in the ensemble contributes its own set of strengths and weaknesses to produce a more accurate prediction. By combining these models, ensemble learning can help overcome the limitations of individual models and provide better results.

# # Ensemble learning

# 

# In[187]:


from sklearn.ensemble import VotingClassifier


# In[188]:


# Train the classifiers

best_rfc.fit(X_train, y_train)
best_gb_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)
best_logr.fit(X_train, y_train)


# In[189]:


# Make predictions on the test set

best_rfc_pred = best_rfc.predict(X_test)
best_gb_clf_pred = best_gb_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)
best_logr_pred = best_logr.predict(X_test)


# In[190]:


# Combine the predictions using voting

voting_clf = VotingClassifier(estimators=[('rfc', best_rfc), ('gbc', best_gb_clf), 
                                          ('ada', ada_clf), ('log', best_logr)], voting ='hard')

voting_clf.fit(X_train, y_train)
ensemble_pred = voting_clf.predict(X_test)


# In[191]:


# Evaluate the accuracy of each classifier and the ensemble

best_rfc_accuracy = accuracy_score(y_test, best_rfc_pred)
best_gb_clf_accuracy = accuracy_score(y_test, best_gb_clf_pred)
ada_clf_accuracy = accuracy_score(y_test, ada_pred)
best_logr_accuracy = accuracy_score(y_test, best_logr_pred)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)


# In[192]:


print("Random Forest Classifier accuracy:", best_rfc_accuracy)
print("Gradient Boosting Classifier accuracy:", best_gb_clf_accuracy)
print("AdaBoost Classifier accuracy:", ada_clf_accuracy)
print("Logistic Classifier accuracy:", best_logr_accuracy)
print("Ensemble accuracy:", ensemble_accuracy)


# ## Getting the final prediction

# In[193]:


# Predicting the labels

final_predictions = voting_clf.predict(test_df)


# In[194]:


final_predictions


# In[195]:


# Creating a dataframe with 'PassengerId' and 'Transported'
 
final_submission = pd.DataFrame(zip(test_df['PassengerId'], final_predictions),
                                columns=['PassengerId', 'Transported'])


# In[196]:


# Replacing values to boolean as requested as format to submission

final_submission.replace({0:False, 1:True}, inplace = True)


# In[197]:


# Checking the final dataframe

final_submission


# In[198]:


# Getting the dataframe as a .csv file for the submission!

final_submission.to_csv('submission.csv', index = False)


# ## Extra
# 
# Adjustments using feature engineering and more advanced machine learning techniques will be applied to the next version to improve the predictions!

# # Thank you!
