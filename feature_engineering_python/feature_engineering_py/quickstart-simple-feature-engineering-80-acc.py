#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Read the Dataset
train_path = '/kaggle/input/spaceship-titanic/train.csv'
test_path = '/kaggle/input/spaceship-titanic/test.csv'

dftrain = pd.read_csv(train_path)
dftest = pd.read_csv(test_path)


# In[3]:


# Get column and row data for training and testing
print('train data have '+str(len(dftrain.index))+' rows')
print('train data have '+str(len(dftrain.columns))+' column')
print('test data have '+str(len(dftest.index))+' rows')
print('test data have '+str(len(dftest.columns))+' column')


# In[4]:


dftrain.head(3)


# In[5]:


dftrain.info()


# In[6]:


dftest.info()


# In[7]:


# Find missing values
print('Training missing values : ')
for column in dftrain.columns:
    if dftrain[column].count() < dftrain.shape[0]:
        print( 'Column {} has {} missing values'.format(column, dftrain.shape[0]-dftrain[column].count()) )
print('-'*40)
print('Test missing values : ')
for column in dftest.columns:
    if dftest[column].count() < dftest.shape[0]:
        print( 'Column {} has {} missing values'.format(column, dftest.shape[0]-dftest[column].count()) )


# In[8]:


dftrain_clean = dftrain.dropna()
dftest_clean = dftest.dropna()
print('training rows with no missing data : {}'.format(len(dftrain_clean)))
print('test rows with no missing data : {}'.format(len(dftest_clean)))


# In[9]:


dfcombined = pd.concat([dftrain_clean.drop(columns='Transported'), dftest_clean], sort=True).reset_index(drop=True)
print(len(dfcombined))


# In[10]:


# Identify object and numerical columns
obj_cols = dfcombined.describe(include='object').columns.tolist()
num_cols = dfcombined.describe(include='number').columns.tolist()
print('There are {} object columns'.format(len(obj_cols)))
print(obj_cols)
print('There are {} numerical columns'.format(len(num_cols)))
print(num_cols)


# In[11]:


dfcombined.describe(include='object')


# In[12]:


dfcombined.describe(include='number')


# ## Analysis

# In[13]:


age_bins = [-1, 5, 20, 30, 40, 50, 60, 100]
dftrain_clean['age_category'] = pd.cut(dftrain_clean['Age'],
                                       bins=age_bins,
                                       labels=['Infant','Teen','20s', '30s', '40s', '50s', 'Elderly'])


# In[14]:


sns.barplot(x=dftrain_clean['age_category'],y=dftrain_clean['Transported'], palette='deep')


# In[15]:


dftrain_clean['cabin_deck'] = dftrain_clean['Cabin'].astype(str).str.split('/').apply(lambda x : x[0] if len(x)==3 else 'Unlisted')
dftrain_clean['cabin_num'] = dftrain_clean['Cabin'].astype(str).str.split('/').apply(lambda x : x[1] if len(x)==3 else 'Unlisted')
dftrain_clean['cabin_side'] = dftrain_clean['Cabin'].astype(str).str.split('/').apply(lambda x : x[2] if len(x)==3 else 'Unlisted')


# In[16]:


grid = sns.FacetGrid(dftrain_clean ,col='cabin_deck')
grid.map(sns.countplot, 'age_category', palette='deep')
grid.add_legend()


# In[17]:


cabin_bins = [-1, 300, 600, 900, 1200, 1500, 1800, 2100]
dftrain_clean['cabin_category'] = pd.cut(dftrain_clean['cabin_num'].astype(int),
                                         bins=cabin_bins,
                                         labels=['range1', 'range2', 'range3', 'range4', 'range5', 'range6', 'range7'])


# In[18]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
sns.countplot(x=dftrain_clean['cabin_deck'],hue=dftrain_clean['Transported'],ax=axes[0], palette='deep')
sns.countplot(x=dftrain_clean['cabin_category'],hue=dftrain_clean['Transported'],ax=axes[1], palette='deep')
sns.countplot(x=dftrain_clean['cabin_side'],hue=dftrain_clean['Transported'],ax=axes[2], palette='deep')


# In[19]:


grid = sns.FacetGrid(dftrain_clean, col='cabin_deck')
grid.map(sns.countplot, 'age_category', palette='deep')
grid.add_legend()


# In[20]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.countplot(x=dftrain_clean['CryoSleep'],hue=dftrain_clean['Transported'],ax=axes[0], palette='deep')
sns.barplot(x=dftrain_clean['CryoSleep'],y=dftrain_clean['Transported'],ax=axes[1], palette='deep')


# In[21]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.countplot(x=dftrain_clean['VIP'],hue=dftrain_clean['Transported'],ax=axes[0], palette='deep')
sns.barplot(x=dftrain_clean['VIP'],y=dftrain_clean['Transported'],ax=axes[1], palette='deep')


# In[22]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.countplot(x=dftrain_clean['Destination'],hue=dftrain_clean['Transported'],ax=axes[0], palette='deep')
sns.barplot(x=dftrain_clean['Destination'],y=dftrain_clean['Transported'],ax=axes[1], palette='deep')


# In[23]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.countplot(x=dftrain_clean['HomePlanet'],hue=dftrain_clean['Transported'],ax=axes[0], palette='deep')
sns.barplot(x=dftrain_clean['HomePlanet'],y=dftrain_clean['Transported'],ax=axes[1], palette='deep')


# In[24]:


dftrain_clean['PassengerGroup'] = dftrain_clean['PassengerId'].apply(lambda x: x[:4]).astype(int)


# In[25]:


sns.histplot(dftrain_clean['PassengerGroup'])


# In[26]:


dftrain_clean['GroupCount'] = dftrain_clean.groupby('PassengerGroup')['PassengerGroup'].transform('count')
sns.countplot(dftrain_clean['GroupCount'].astype(int))


# In[27]:


dftrain_clean['first_name'] = dftrain['Name'].astype(str).str.split(' ').apply(lambda x: x[0] if len(x) == 2 else 'unlisted')
dftrain_clean['last_name'] = dftrain['Name'].astype(str).str.split(' ').apply(lambda x: x[1] if len(x) == 2 else 'unlisted')


# In[28]:


dftrain_clean['Family_Count'] = dftrain_clean.groupby('last_name')['last_name'].transform('count') 
sns.countplot(dftrain_clean['Family_Count'].astype(int))


# In[29]:


fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4))
sns.histplot(dftrain_clean['RoomService'],ax=axes[0],bins=20)
sns.histplot(dftrain_clean['FoodCourt'],ax=axes[1],bins=20)
sns.histplot(dftrain_clean['ShoppingMall'],ax=axes[2],bins=20)
sns.histplot(dftrain_clean['Spa'],ax=axes[3],bins=20)
sns.histplot(dftrain_clean['VRDeck'],ax=axes[4],bins=20)


# In[30]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
dftrain_clean['TotalSpent'] = dftrain_clean['RoomService'] + dftrain_clean['FoodCourt'] + dftrain_clean['ShoppingMall'] + dftrain_clean['Spa'] + dftrain_clean['VRDeck']
dftrain_clean['AverageSpent'] = dftrain_clean['TotalSpent']/5
dftrain_clean['MostSpent'] = pd.DataFrame([dftrain_clean['RoomService'], dftrain_clean['FoodCourt'], dftrain_clean['ShoppingMall'], dftrain_clean['Spa'], dftrain_clean['VRDeck']]).transpose().idxmax(axis=1)
sns.histplot(x=dftrain_clean['TotalSpent'],ax=axes[0],bins=20)
sns.histplot(x=dftrain_clean['AverageSpent'],ax=axes[1],bins=20)
sns.countplot(x=dftrain_clean['MostSpent'],ax=axes[2])


# ## Feature Engineering

# In[31]:


combined = pd.concat([dftrain.drop(columns='Transported'), dftest], sort=True).reset_index(drop=True)


# In[32]:


age_bins = [-1, 5, 20, 30, 40, 50, 60, 100]
cabin_bins = [-1, 300, 600, 900, 1200, 1500, 1800, 2100]

vip_mode = combined['VIP'].mode()[0]
cryo_mode = combined['CryoSleep'].mode()[0]
age_median = combined['Age'].median()
rs_median = combined['RoomService'].median()
fc_median = combined['FoodCourt'].median()
sm_median = combined['ShoppingMall'].median()
spa_median = combined['Spa'].median()
vr_median = combined['VRDeck'].median()

for df in [dftrain,dftest]:
    df['HomePlanet'] = df['HomePlanet'].fillna('Unknown_Home')
    df['Destination'] = df['Destination'].fillna('Unknown_Dest')
    df['VIP'] = df['VIP'].fillna(vip_mode)
    df['CryoSleep'] = df['CryoSleep'].fillna(cryo_mode)
    df['Cabin'] = df['Cabin'].fillna('Unknown_Deck/0/Unknown_Side')
    df['Age'] = df['Age'].fillna(age_median)
    df['RoomService'] = df['RoomService'].fillna(rs_median)
    df['FoodCourt'] = df['FoodCourt'].fillna(fc_median)
    df['ShoppingMall'] = df['ShoppingMall'].fillna(sm_median)
    df['Spa'] = df['Spa'].fillna(spa_median)
    df['VRDeck'] = df['VRDeck'].fillna(vr_median)
    df['TotalSpent'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
    df['AverageSpent'] = df['TotalSpent']/5
    df['MostSpent'] = pd.DataFrame([df['RoomService'], df['FoodCourt'], df['ShoppingMall'], df['Spa'], df['VRDeck']]).transpose().idxmax(axis=1)
    df['Age_Category'] = pd.cut(df['Age'],
                                bins=age_bins,
                                labels=['Infant','Teen','20s', '30s', '40s', '50s', 'Elderly'])
    df['Cabin_Deck'] = df['Cabin'].astype(str).str.split('/').apply(lambda x : x[0] if len(x)==3 else 'Unlisted')
    df['Cabin_Num'] = df['Cabin'].astype(str).str.split('/').apply(lambda x : x[1] if len(x)==3 else 'Unlisted')
    df['Cabin_Side'] = df['Cabin'].astype(str).str.split('/').apply(lambda x : x[2] if len(x)==3 else 'Unlisted')
    df['Cabin_Category'] = pd.cut(df['Cabin_Num'].astype(int),
                                  bins=cabin_bins,
                                  labels=['range1', 'range2', 'range3', 'range4', 'range5', 'range6', 'range7'])
    df['Passenger_Group'] = df['PassengerId'].apply(lambda x: x[:4]).astype(int)
    df['Group_Count'] = df.groupby('Passenger_Group')['Passenger_Group'].transform('count')
    df['Name'] = df['Name'].fillna('Unlisted_FN Unlisted_LN')
    df['First_Name'] = dftrain['Name'].astype(str).str.split(' ').apply(lambda x: x[0] if len(x) == 2 else 'unlisted')
    df['Last_Name'] = dftrain['Name'].astype(str).str.split(' ').apply(lambda x: x[1] if len(x) == 2 else 'unlisted')
    df['Family_Count'] = df.groupby('Last_Name')['Last_Name'].transform('count')


# In[33]:


dftrain.head(3)


# In[34]:


dftrain.columns


# In[35]:


for df in [dftrain,dftest] :
    df['Age_Category'] = df['Age_Category'].map({'Infant':0,'Teen':1,'20s':2,'30s':3,'40s':4,'50s':5,'Elderly':6})
    df['Cabin_Category'] = df['Cabin_Category'].map({'range1':0,'range2':1,'range3':2,'range4':3,'range5':4,'range6':5,'range7':6})


# In[36]:


Test_Index = dftest['PassengerId']


# In[37]:


dftrain = dftrain.drop(columns=['PassengerId','Cabin','Age','Name','Cabin_Num','Passenger_Group','First_Name','Last_Name'])
dftest = dftest.drop(columns=['PassengerId','Cabin','Age','Name','Cabin_Num','Passenger_Group','First_Name','Last_Name'])


# In[38]:


dftrain = pd.get_dummies(dftrain, columns=['HomePlanet','Destination','MostSpent','Cabin_Deck','Cabin_Side']).astype(float)
dftest = pd.get_dummies(dftest, columns=['HomePlanet','Destination','MostSpent','Cabin_Deck','Cabin_Side']).astype(float)


# In[39]:


from sklearn.feature_selection import f_classif
X = dftrain.drop(columns='Transported')
y = dftrain['Transported']
f_stats, p_val = f_classif(X,y)
data = pd.DataFrame(data=p_val.round(2),columns=['p_val'],index=X.columns).sort_values(by='p_val',ascending=False).T
data


# In[40]:


cols = [c for c in X if data[c].mean() < 0.5 and c in list(dftest.columns)]
dftrain = dftrain[cols+['Transported']]
dftest = dftest[cols]


# In[41]:


fig, ax = plt.subplots(figsize=(20, 16))
sns.heatmap(round(dftrain.corr(),2),vmax=1,vmin=-1, annot=True, cmap='coolwarm')


# ## Create a prediction Model

# In[42]:


# machine learning packages
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[43]:


# Split into training and testing data
# We do not use validation because of the small amount of data
X_train = dftrain.drop("Transported", axis=1).astype(float)
Y_train = dftrain["Transported"].astype(int)
X_test  = dftest.copy().astype(float)
X_train.shape, Y_train.shape, X_test.shape


# In[44]:


# Normalize both train and test input
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[45]:


x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, train_size=0.7, random_state=1)


# In[46]:


logreg = LogisticRegression(random_state=42).fit(x_train,y_train)

logreg_train = logreg.score(x_train,y_train)
logreg_valid = logreg.score(x_valid,y_valid)
print('Train accuracy = {}'.format(logreg_train))
print('Test accuracy = {}'.format(logreg_valid))


# In[47]:


gbc = GradientBoostingClassifier(random_state=42).fit(x_train,y_train)

gbc_train = gbc.score(x_train,y_train)
gbc_valid = gbc.score(x_valid,y_valid)
print('Train accuracy = {}'.format(gbc_train))
print('Test accuracy = {}'.format(gbc_valid))


# In[48]:


rf = RandomForestClassifier(random_state=42).fit(x_train,y_train)

rf_train = rf.score(x_train,y_train)
rf_valid = rf.score(x_valid,y_valid)
print('Train accuracy = {}'.format(rf_train))
print('Test accuracy = {}'.format(rf_valid))


# In[49]:


dt = DecisionTreeClassifier(random_state=42).fit(x_train,y_train)

dt_train = dt.score(x_train,y_train)
dt_valid = dt.score(x_valid,y_valid)
print('Train accuracy = {}'.format(dt_train))
print('Test accuracy = {}'.format(dt_valid))


# In[50]:


svc = SVC(random_state=42).fit(x_train,y_train)

svc_train = svc.score(x_train,y_train)
svc_valid = svc.score(x_valid,y_valid)
print('Train accuracy = {}'.format(svc_train))
print('Test accuracy = {}'.format(svc_valid))


# In[51]:


models = pd.DataFrame({
    'Model': ['Logistic Regression (Base)', 'Gradient Boosting Classifier','SVC',
             'Decision Tree Classifier', 'Random Forest Classifier'],
    'Train': [logreg_train, gbc_train, svc_train ,dt_train, rf_train],
    'Test': [logreg_valid, gbc_valid, svc_valid ,dt_valid, rf_valid]})
models.sort_values(by='Test', ascending=False)


# In[52]:


highest_score = models.sort_values(by='Test', ascending=False).reset_index().iloc[0]['Model']
highest_score


# In[53]:


if highest_score == 'Random Forest Classifier' :
    Submission_Model = rf.fit(X_train,Y_train)
elif highest_score == 'Gradient Boosting Classifier' :
    Submission_Model = gbc.fit(X_train,Y_train)
elif highest_score == 'Logistic Regression (Base)' :
    Submission_Model = logreg.fit(X_train,Y_train)
elif highest_score == 'SVC' :
    Submission_Model = svc.fit(X_train,Y_train)
elif highest_score == 'Decision Tree Classifier' :
    Submission_Model = dt.fit(X_train,Y_train)


# In[54]:


print(classification_report(Y_train,Submission_Model.predict(X_train)))


# In[55]:


cm = confusion_matrix(Y_train,Submission_Model.predict(X_train))

sns.heatmap(cm, annot = True)


# In[56]:


submission = pd.DataFrame({
        "PassengerId": Test_Index,
        "Transported": Submission_Model.predict(X_test).astype(bool)
    })
submission.to_csv('submission.csv', index=False)

