#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd #Data processing
import numpy as np #Linear Algebra
#Graphs ans plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer # Impute NaN values
import re #Regular expressions
from sklearn.model_selection import train_test_split #Split data set into training and validation subsets
from sklearn.preprocessing import OneHotEncoder #Encoder for data
from sklearn.svm import SVC #Model for predictions
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix #Model evaluation


# # Reading and general info

# In[2]:


#read csv's
df_train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
df_train.head()


# In[3]:


#Display columns data type and non null rows
df_train.info()
print()
df_test.info()


# In[4]:


df_train.describe()


# As we can see, most of the columns have some null values that we need to take care of. 
# 
# Also, it seems that the luxury related columns have a highly lefted distribution so that most of the values are 0.

# # Visualization

# ## Histograms

# In[5]:


df_train.hist(figsize=(16,20), bins=30)


# As suspected, the vast mayority of samples are 0 in the RoomService, FoodCourt, ShooppingMall, Spa and VRDeck columns with some extreme outliers in each one. 

# ## Count plots

# Now, let's plot the distribution in the categorical columns

# In[6]:


for column in ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Transported']:
    plt.figure(figsize=(16,8))
    plt.grid()
    sns.countplot(x=df_train[column])
    plt.title(column)


# ## Variables correlation

# In order to gain more insights about the data, plot the number of observation in some column and group it with other column

# ### HomePlanet

# #### CryoSleep

# In[7]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['CryoSleep'], hue=df_train['HomePlanet'])


# #### Destination

# In[8]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Destination'], hue=df_train['HomePlanet'])


# #### VIP

# In[9]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['VIP'], hue=df_train['HomePlanet'])


# #### Age

# In[10]:


plt.figure(figsize=(16,8))
plt.grid()
sns.histplot(data=df_train, x='Age', hue='HomePlanet', multiple='stack')


# #### Transported

# In[11]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Transported'], hue=df_train['HomePlanet'])


# ### Cryosleep

# #### Destination

# In[12]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Destination'], hue=df_train['CryoSleep'])


# #### VIP

# In[13]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['VIP'], hue=df_train['CryoSleep'])


# #### Age

# In[14]:


plt.figure(figsize=(16,8))
plt.grid()
sns.histplot(data=df_train, x='Age', hue='CryoSleep', multiple='stack')


# #### Transported

# In[15]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Transported'], hue=df_train['CryoSleep'])


# ### Destination

# #### VIP

# In[16]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['VIP'], hue=df_train['Destination'])


# #### Age

# In[17]:


plt.figure(figsize=(16,8))
plt.grid()
sns.histplot(data=df_train, x='Age', hue='Destination', multiple='stack')


# #### Transported

# In[18]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Transported'], hue=df_train['Destination'])


# ### VIP

# #### Age

# In[19]:


plt.figure(figsize=(16,8))
plt.grid()
sns.histplot(data=df_train, x='Age', hue='VIP', multiple='stack')


# #### Transported

# In[20]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(x=df_train['Transported'], hue=df_train['VIP'])


# ### Age

# #### Transported

# In[21]:


plt.figure(figsize=(16,8))
plt.grid()
sns.histplot(data=df_train, x='Age', hue='Transported', multiple='stack')


# # Feature Engineering

# In[22]:


#Classify people by age
def process_age(row):
    if row < 6:
        return 'EarlyChild'
    elif row < 12:
        return 'Childhood'
    elif row < 18:
        return 'Teen'
    elif row < 35:
        return 'YoungAdult'
    elif row < 60:
        return 'Adult'
    else:
        return 'Elder'

#Codify the boolean values to 1, 0
def process_bin(row):
    if row == True:
        return 1
    else:
        return 0
# Returns 1 is the amount in the row is greater than 0
def process_num(row):
    if row > 0:
        return 1
    else:
        return 0

# Collects the Deck, Number and Side of each cabins and returns them in a list
def process_cabin(row):
    pattern = r'([A-Z])+/(\d+)/(P|S)'
    result = re.search(pattern, str(row))
    return [result[1], result[2], result[3]]

#Classifies the cabin number
def process_num_cabin(row):
    row = float(row)
    if row < 201:
        return '1-200'
    elif row < 401:
        return '201-400'
    elif row < 601:
        return '401-600'
    elif row < 801:
        return '601-800'
    elif row < 1001:
        return '801-1000'
    elif row < 1201:
        return '1001-1200'
    elif row < 1401:
        return '1201-1400'
    elif row < 1601:
        return '1401-1600'
    elif row < 1801:
        return '1600-1800'
    else:
        return '>1800'


# In[23]:


#Replane NaN values using the most frequent value
imputer = SimpleImputer(strategy='most_frequent')

X = pd.concat([df_train.select_dtypes(['int', 'float']), 
               pd.DataFrame(imputer.fit_transform(df_train.select_dtypes('object')), columns=imputer.get_feature_names_out())],axis=1)

X_test = pd.concat([df_test.select_dtypes(['int', 'float']), 
               pd.DataFrame(imputer.transform(df_test.select_dtypes('object')), columns=imputer.get_feature_names_out())],axis=1)
X


# In[24]:


#Process Age column
X['Age'] = X['Age'].apply(process_age)
X_test['Age'] = X_test['Age'].apply(process_age)

#Process CryoSleep column
X['CryoSleep'] = X['CryoSleep'].apply(process_bin)
X_test['CryoSleep'] = X_test['CryoSleep'].apply(process_bin)

#Process VIP column
X['VIP'] = X['VIP'].apply(process_bin)
X_test['VIP'] = X_test['VIP'].apply(process_bin)

#Process Lux columns    
for i in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    X[i] = X[i].apply(process_num)
    X_test[i] = X_test[i].apply(process_num)

#Process Cabin column
cabin = X['Cabin'].apply(process_cabin)
decks = [deck[0] for deck in cabin]
nums = [num[1] for num in cabin]
sides = [side[2] for side in cabin]
X['Deck'] = decks
X['Num'] = nums
X['Num'] = X['Num'].apply(process_num_cabin)
X['Side'] = sides

cabin = X_test['Cabin'].apply(process_cabin)
decks = [deck[0] for deck in cabin]
nums = [num[1] for num in cabin]
sides = [side[2] for side in cabin]
X_test['Deck'] = decks
X_test['Num'] = nums
X_test['Num'] = X_test['Num'].apply(process_num_cabin)
X_test['Side'] = sides

#Create column that indicates if at least one of the luxury services was used
X['Lux'] = X[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(axis=1).apply(process_num)
X_test['Lux'] = X_test[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].mean(axis=1).apply(process_num)

#Save Ids
PId =X_test.pop('PassengerId')

#Drop unnecesary columns
X_test.drop(['Cabin', 'Name'], axis=1, inplace=True)
X.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

#Save Transported column
Y = df_train['Transported'].apply(process_bin)


# In[25]:


X


# ## Visualization with new variables

# ### Lux

# In[26]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Lux', hue=Y)


# ### RoomService

# In[27]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='RoomService', hue=Y)


# ### FoodCourt

# In[28]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='FoodCourt', hue=Y)


# ### ShoppingMall

# In[29]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='ShoppingMall', hue=Y)


# ### Spa

# In[30]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Spa', hue=Y)


# ### VRDeck

# In[31]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='VRDeck', hue=Y)


# ### Deck

# In[32]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Deck', hue=Y)


# ### Num

# In[33]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Num', hue=Y)


# ### Side

# In[34]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Side', hue=Y)


# ### Age

# In[35]:


plt.figure(figsize=(16,8))
plt.grid()
sns.countplot(data=X, x='Age', hue=Y)


# ## Encoding 

# In[36]:


#Apply OneHot encoding to the new categorical values
encoder = OneHotEncoder(sparse=False)
X_cat = pd.DataFrame(encoder.fit_transform(X[['Age', 'HomePlanet', 'Destination', 'Deck', 'Num', 'Side']]), columns=encoder.get_feature_names_out())
X_cat_test = pd.DataFrame(encoder.transform(X_test[['Age', 'HomePlanet', 'Destination', 'Deck', 'Num', 'Side']]), columns=encoder.get_feature_names_out())
X_cat


# In[37]:


#Concatenate the result dfs
X = pd.concat([X_cat, X.drop(['Age', 'HomePlanet', 'Destination', 'Deck', 'Num', 'Side'],axis=1)],axis=1)
X_test = pd.concat([X_cat_test, X_test.drop(['Age', 'HomePlanet', 'Destination', 'Deck', 'Num', 'Side'],axis=1)],axis=1)
X


# In[38]:


X_test


# ## Correlation

# As a result of the encoding we have 40 columns, and as we saw in the plots not all of them are as useful. 
# 
# Therefore we calculte the correlation with the Transported column and select the ones with with the higher value

# In[39]:


X_ = X.copy()
X_['Y'] = Y
#Columns with an absolute correlation greater than 0.1 with the response column(Transported), expect from that column itself
X_.corr().loc[(abs(X_.corr()['Y']) > 0.1) & (X_.corr()['Y']!= 1) ]['Y'].sort_values(ascending=False)


# In[40]:


#Select those columns
X_red = X_[X_.corr().loc[(abs(X_.corr()['Y']) > 0.1) & (X_.corr()['Y']!= 1) ]['Y'].index].copy()
X_test_red = X_test[X_red.columns]
X_red


# Side_P and Side_S columns are mutually exclusive, this means that none of the rows have the same value in both columns and have the same correlation with different sign.
# 
# Having the two of them it's reduntant, so it's better to drop one. 

# In[41]:


X.drop(['Side_P'], inplace=True, axis=1)
X_red.drop(['Side_P'], inplace=True, axis=1)
X_test.drop(['Side_P'], inplace=True, axis=1)
X_test_red.drop(['Side_P'], inplace=True, axis=1)


# In[42]:


X_red


# # Model training

# In[43]:


def predict(X_train,y_train,X_val, y_val,model):
    #Predicts and contrast different scores with the training and validation set
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    print("---Training set---\n")
    print("\t-Precision Score: {:.4f}\n".format(precision_score(y_train,y_train_pred)))
    print("\t-Accuracy Score: {:.4f}\n".format(accuracy_score(y_train,y_train_pred)))
    print("\t-F1 Score: {:.4f}\n".format(f1_score(y_train,y_train_pred)))
    print("\t-Recall Score: {:.4f}\n".format(recall_score(y_train, y_train_pred)))
    print("---Validation set---\n")
    print("\t-Precision Score: {:.4f}\n".format(precision_score(y_val,y_val_pred)))
    print("\t-Accuracy Score: {:.4f}\n".format(accuracy_score(y_val,y_val_pred)))
    print("\t-F1 Score: {:.4f}\n".format(f1_score(y_val,y_val_pred)))
    print("\t-Recall Score: {:.4f}\n".format(recall_score(y_val, y_val_pred)))


# In[44]:


#Split datasets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, random_state=42, test_size=0.2)
X_red_train, X_red_val, Y_train, Y_val = train_test_split(X_red, Y, random_state=42, test_size=0.2)


# In[45]:


X_red_train


# ## Model 1

# In[46]:


#Model using the whole dataframe
model1 = SVC(random_state=42)
model1.fit(X_train, Y_train)
predict(X_train, Y_train, X_val, Y_val, model1)


# In[47]:


#Confusion matrix with the training data
cf_train = pd.DataFrame(confusion_matrix(Y_train, model1.predict(X_train)), index = ['Real 0', 'Real 1'], columns= ['Pred 0', 'Pred 1'])
plt.figure(figsize=(15,10))
sns.heatmap(cf_train, annot=True)


# In[48]:


#Confusion matrix with the validation data
cf_val = pd.DataFrame(confusion_matrix(Y_val, model1.predict(X_val)), index = ['Real 1', 'Real 0'], columns= ['Pred 1', 'Pred 0'])
plt.figure(figsize=(15,10))
sns.heatmap(cf_val, annot=True)


# ## Model 2

# In[49]:


#Model using the first reduced dataset
model2 = SVC(random_state=42)
model2.fit(X_red_train, Y_train)
predict(X_red_train, Y_train, X_red_val, Y_val, model2)


# In[50]:


#Matriz de confusión en datos de entrenamiento
cf_train = pd.DataFrame(confusion_matrix(Y_train, model2.predict(X_red_train)), index = ['Real 0', 'Real 1'], columns= ['Pred 0', 'Pred 1'])
plt.figure(figsize=(15,10))
sns.heatmap(cf_train, annot=True)


# In[51]:


cf_val = pd.DataFrame(confusion_matrix(Y_val, model2.predict(X_red_val)), index = ['Real 1', 'Real 0'], columns= ['Pred 1', 'Pred 0'])
plt.figure(figsize=(15,10))
sns.heatmap(cf_val, annot=True)


# # Submission

# In[52]:


submission = model2.predict(X_test_red)
submission = pd.DataFrame(submission, columns=['Transported'])
submission['PassengerID'] = PId
submission = submission[['PassengerID', 'Transported']]
submission['Transported'] = submission['Transported'].apply(bool)
submission.to_csv('/kaggle/working/submission.csv', index=False)


# In[ ]:




