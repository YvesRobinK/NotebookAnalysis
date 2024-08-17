#!/usr/bin/env python
# coding: utf-8

# # **Importing Packages**

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier

sns.set(color_codes=True)
sns.set_style('darkgrid')
plt.style.use('ggplot')


# # **Loading and Preview the Dataset:**

# In[2]:


train=pd.read_csv('/kaggle/input/spaceship-titanic/train.csv') ## Training Dataset
test=pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')   ## Testing Dataset
train.head(10)


# First of all, we need to meet with our target variable: "Transported", and know how many passengers were transported (True) vs. passengeres were not transported (False), also let's descover if there is duplicated values so we can drop them from the beginning.

# In[3]:


print('No. of duplicated values at the the whole dataset is: ',train.duplicated().sum())
train['Transported'].value_counts()


# The ratio is equal (Transported vs. not Transported), also no duplicate values at the whole of dataset.
# 
# Then, we need to know more about the data features that can be used finally for model building..
# 
# So I will define a defination that will extract each of (Feature name, Value Counts, no. unique values, data type, and no. of null values) in order to get more details about the features.
# 
# After that I will take each feature and perform the following on it:
# 
#  1. Find the relationship between the feature and the target variable.
#   
#  2. Fill missing values if existing at this feature.
#  
#  3. Apply feature engineering / selection if applicable.

# In[4]:


def info(dataset):
    
    """ This defination is to print most valuable information
        about dataset columns.
        Input: dataset
        Output: dataset columns information
    """
    for column in dataset.columns:
        print('==========%s =========='%column)
        print('Type is: ',dataset[column].dtype)
        print(dataset[column].value_counts())
        print('Number of unique values: ',dataset[column].nunique())
        print('Number of null values: ',dataset[column].isna().sum())

info(train)


# # **EDA: Exploratory Data Analysis**

# **"PassengerId" Column**
# 
# Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# 
# 1. I will extract group_id , member_id from each passenger_id
# 
# 2. Then creat 2 new columns, one specifying whether the passenger was traveling alone or in a group of more than 1 member, and th another counting the members at each group.
# 
# 3. Plot the distribution of the two new created columns to get insights.

# In[5]:


train[['Group','Member']]=train['PassengerId'].str.split('_',expand=True) # split PassengerId to Group, Member

x=train.groupby('Group')['Member'].count().sort_values(ascending=False) # Count members at each group

train['is_alone']=train['Group'].apply(lambda y: y not in set(x[x>1].index)) # create new column for groups have more than 1 member

train['No_members_in_group']=0   ## Creat new column for number of members in each group
for i in x.items():
    train.loc[train['Group']==i[0],'No_members_in_group']=i[1]
    
plt.figure(figsize=[14,6])

plt.subplot(1,2,1)
sns.countplot(data=train,x='No_members_in_group',hue='Transported')
plt.title('No. of members in a group vs. Transported')

plt.subplot(1,2,2)
sns.countplot(data=train,x='is_alone',hue='Transported') 
plt.title('Is Alone vs. Transported')

plt.show()


# **Conclusions (1):**
# 
# - Most of the passengers are travelling alone.
# 
# - No clear correlation between "is_alone", "no. of memebrs in a group" and the "Transported" passengers, although there is a slight larger prospect to not be transported in case of the passenger is not travelling alone.

# **"Neme" Column:**
# 
# "Name": The first and last names of the passenger.
# 
# Now let's explore "Name" column if we can find insights.
# 
# We can get the family members by extracting the surname from "Name" column, and grouping by surname so we can explore the same characteristics of the family members.

# In[6]:


train['surname']=train['Name'].str.split(' ',expand=True)[1]
df_surname_grouped=train.groupby(['surname','Transported'])['PassengerId'].count()
df_surname_grouped=df_surname_grouped.reset_index()
df_surname_grouped.rename({'surname':'Family Name','PassengerId':'Count of Passengers'},axis=1,inplace=True)

""" 
    Because of number of families is very large, I will take samples from families to explore
    the distribution of transported vs. Not Transported number of members in family samples.
    
"""

plt.figure(figsize=[16,16])
j=1
for i in np.random.randint(0,df_surname_grouped.shape[0],9):
    plt.subplot(3,3,j)
    if df_surname_grouped.iloc[i,1]== False:
        sns.barplot(data=df_surname_grouped[i:i+10],x='Family Name',y='Count of Passengers',hue='Transported')
        plt.xlabel('Samples from Families')
        plt.xticks(color='w')
    else:
        sns.barplot(data=df_surname_grouped[i+1:i+11],x='Family Name',y='Count of Passengers',hue='Transported')
        plt.xlabel('Samples from Families')
        plt.xticks(color='w')
    
    j+=1

plt.show()


# **Conclusions (2):**
# 
# No. of family members don't impacting whether the whole family were tranported or not, on contrast there is a slightly prospection that when No. of family members increased, many family members were less chance to survive. 

# **"RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck" Columns and Feature Engineering:**
# 
# Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# 
# Let's first analyse the distribution of amount that passengers payed for Titanic's amenities

# In[7]:


fig, axes = plt.subplots(2, 3, figsize=(30, 15))
axes = axes.flatten()
amenities=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
colors=['g','r','b','y','black']
no=[0,1,2,3,4,5]

for n, i, j in zip(no,amenities,colors):
    sns.histplot(ax = axes[n],x=train[i],kde=True,bins=50,color=j)
    
print("\n Analyzing the graphs here, it turns out that the values of the variables are not normally distributed. \n")


# For feature slection and engineering purpose, I will create a new column "Total_Charge" of the sum of all luxury amenities. then take the log transformation for it

# In[8]:


train['Total_Charge']=train[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)
sns.histplot(x=train['Total_Charge'],kde=True,bins=20,hue=train['Transported'])
plt.show()


# For feature slection and engineering purpose, I will create a new column "Total Charge" of the sum of all luxury amenities, then extract another column "Total_Amenities_Billed_Category" from the new created column consists of 4 categories:
# 
# . 1st category for passengers were not billed any amount for amenities. (No bills)
# 
# . 2nd category for passengers billed low amount for amenities. (Low)
# 
# . 3rd category for passengers billed medium amount for amenities. (Medium)
#  
# . 4th category for passengers billed high amount for amenities. (High)

# In[9]:


train['Total_Amenities_Billed_Category']='no_bills'

train.loc[(train['Total_Charge']>0) & 
            (train['Total_Charge']<=716),'Total_Amenities_Billed_Category']='Low'

train.loc[(train['Total_Charge']>716) & 
            (train['Total_Charge']<=1442),'Total_Amenities_Billed_Category']='Medium'

train.loc[train['Total_Charge']>1442,'Total_Amenities_Billed_Category']='High'

amenities_transported_df=train.loc[train['Transported']==True,'Total_Amenities_Billed_Category'].value_counts()
amenities_not_transported_df=train.loc[train['Transported']==False,'Total_Amenities_Billed_Category'].value_counts()

plt.figure(figsize=[8,6])
plt.subplot(1,2,1)
plt.pie(amenities_transported_df,labels=amenities_transported_df.index,
       explode=[0.1,0,0,0],autopct='%1.1f%%',shadow=True,textprops={'size':14},
       labeldistance=0.6,pctdistance=0.4)
plt.legend(['Transported = True'],loc='upper right',fontsize=14,frameon=False)
plt.axis('equal')

plt.subplot(1,2,2)
plt.pie(amenities_not_transported_df,labels=amenities_not_transported_df.index,
       explode=[0.1,0,0,0],autopct='%1.1f%%',shadow=True,textprops={'size':14},
       labeldistance=0.6,pctdistance=0.4)

plt.axis('equal')
plt.legend(['Transported = False'],loc='upper right',fontsize=14,frameon=False)
plt.subplots_adjust(left=0,right=1.5,wspace=0.5)
print('"\n Analyzing the graphs here, it turns out that the Most transported passengers were not payed for \
Titanic\'s Luxury Anemities\n"')
plt.show()


# **Conclusions (3):**
# 
# Most of passengers espicially "Transported" passengers were not payed any charge for Titanic's luxury anemities, and at the second order, passengers who billed very much were more exposure to not transported.

# **"HomePlanet", "CryoSleep", "Destination", "VIP" Columns:**
# 
# "HomePlanet": The planet the passenger departed from, typically their planet of permanent residence.
# 
# "CryoSleep": Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# 
# "Destination": The planet the passenger will be debarking to.
# 
# "VIP": Whether the passenger has paid for special VIP service during the voyage.
# 
# 
# 1. First of all, we need to know if there is a correlation or relationship between mentioned coulmns with the target variable.
# 
# 2. Then, explore if there are relationships between these columns and other columns.

# In[10]:


plt.figure(figsize=[10,10])
plt.subplot(2,2,1)
sns.countplot(data=train,x='HomePlanet',hue='Transported')
plt.title('HomePlanet vs. Transported')

plt.subplot(2,2,2)
sns.countplot(data=train,x='CryoSleep',hue='Transported')
plt.title('CryoSleep vs. Transported')

plt.subplot(2,2,3)
sns.countplot(data=train,x='Destination',hue='Transported')
plt.title('Destination vs. Transported')

plt.subplot(2,2,4)
sns.countplot(data=train,x='VIP',hue='Transported')
plt.title('VIP vs. Transported')

plt.subplots_adjust(left=0,right=1.5,wspace=0.3,hspace=0.3)
plt.show()


# **Conclusions (4):**
# 
# From above plots, we conclude below information:
# 
# 1. The most of passengers were from "Earth" HomePlanet, although the sum of passengers from "Europa" and "Mars" are larger than "Earth".
# 
# 2. The most of transported passengers were elected to be put into suspended animation (CryoSleep=True), that mean if you were in suspended animation you have more chane to be transported, while if you don't choose CryoSleep, so the chance for not transported is large.
# 
# 3. Most of passengers their destination were "TRAPPIST-1e".
# 
# 4. Most of passengers were not paid extra funds for "VIP" services.

# **"Age" Column:**

# In[11]:


plt.figure(figsize=[14,8])
sns.histplot(data=train,x='Age',hue='Transported',kde=True)
plt.show()


# **Conclusions (5):**
# 
# - Most of passengers were from age range [18:32]
# 
# - For ages range [0:18] the number of transported passengers were bigger than not transported ones especially for those who were new born.
# 
# - For ages range [19:38] the number of transported passengers were smaller than not tranported ones.
# 
# - For ages > 40, the number of transported passengers are almost the same of not tranported ones. 

# **"Cabin" Column:**
# 
# "Cabin": The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# 
# In cabin column we have three information in one column. lets seperate them.

# In[12]:


train[["Cabin_deck","Cabin_no.","Cabin_side"]]=train["Cabin"].str.split("/",expand=True)
train[["Cabin_deck","Cabin_no.","Cabin_side"]].nunique()


# In[13]:


plt.figure(figsize=[14,6])
plt.subplot(1,2,1)
sns.countplot(data=train,x='Cabin_deck',hue='Transported',order=['A','B','C','D','E','F','G','T'])
plt.subplot(1,2,2)
sns.countplot(data=train,x='Cabin_side',hue='Transported')
plt.show()


# Let's go deeper and explore the percentage of Transported / Not Transported Passengers at each combination of Cabin_deck + Cabin_side to get insights of most safe and most dangerous deck/side.

# In[14]:


cabin_s=train[train['Cabin_side']=='S'].groupby(['Cabin_deck','Transported'])['Cabin_side'].count().reset_index()
cabin_p=train[train['Cabin_side']=='P'].groupby(['Cabin_deck','Transported'])['Cabin_side'].count().reset_index()
plt.figure(figsize=[18,8])
plt.subplot(1,2,1)
sns.barplot(data=cabin_s,x='Cabin_deck',y='Cabin_side',hue='Transported')
plt.ylabel('Passengers Sitting at Cabinet Side "S"')
plt.subplot(1,2,2)
sns.barplot(data=cabin_p,x='Cabin_deck',y='Cabin_side',hue='Transported')
plt.ylabel('Passengers Sitting at Cabinet Side "P"')
plt.show()


# **Conclusions (6):**
# 
# 1. Most of passengers were sitting in cabin_deck "F" & "G".
# 
# 2. Numper of transported passengers at cabin_deck (B,C) is larger than not transported ones at the same decks espicially who were setting at Cabinet Side "S"--> Decks (B,C) with Side "S" seem to be the most safe decks/side.
# 
# 3. Numper of not transported passengers at cabin_deck (D,E,F) is larger than transported ones at the same decks espicially who were setting at Cabinet Side "P"--> Decks (D,E,F) with Side "P" seem to be the most dangerous decks/side.
# 
# 4. Numper of transported passengers at cabin_side (S) is larger than not transported ones at the same side --> Generally "S" side is safer than side "P".
# 
# 5. Number of passengers were sitting at Cabin_deck "T" is very small.

# **Final Conclusion:**
# 
# Before last transformation of data to applicable form for model fitting, I want to take a glence to most transported passengers features like:
# 
# - What was most of their Home Planet? 
# 
# - What was most of their CryoSleep? 
# 
# - What was most of their Destination?
# 
# - Whether most of them has paid for VIP services?
# 
# - Whether most of them was alone?
# 
# - Whether most of them has paid for luxury amenities?
# 
# - What was their age range?

# In[15]:


categorical_cols=['HomePlanet', 'CryoSleep','Destination', 'Age','VIP', 'RoomService', 'FoodCourt', 
      'ShoppingMall', 'Spa', 'VRDeck','Transported','is_alone','No_members_in_group', 'Total_Charge',
      'Cabin_deck','Cabin_side']

numerical_cols=['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Transported',
                'is_alone','No_members_in_group', 'Total_Charge']

print("=====================Most of Transported Passengers Statistics=====================\n")
print('Categorical Columns\n')
print(train.loc[train['Transported']==True,categorical_cols].describe(include='object'))
print('\nNumerical Columns\n')
print(train.loc[train['Transported']==True,numerical_cols].describe())
print("\n=====================Most of Not Transported Passengers Statistics=====================\n")
print('Categorical Columns\n')
print(train.loc[train['Transported']==False,categorical_cols].describe(include='object'))
print('\nNumerical Columns\n')
print(train.loc[train['Transported']==False,numerical_cols].describe())
print("\n=============================== End of EDA =====================================\n")


# Now after EDA is completed, many feature engineering has been done on training dataset, that also need to be reflected into testing dataset as well.
# 
# In next code section I will apply each feature engineering I did above on testing dataset. 

# In[16]:


test[['Group','Member']]=test['PassengerId'].str.split('_',expand=True)
x=test.groupby('Group')['Member'].count().sort_values(ascending=False)
test['is_alone']=test['Group'].apply(lambda y: y not in set(x[x>1].index)) 
test['No_members_in_group']=0   
for i in x.items():
    test.loc[test['Group']==i[0],'No_members_in_group']=i[1]
    
test['surname']=test['Name'].str.split(' ',expand=True)[1]

test['Total_Charge']=test[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].sum(axis=1)

test['Total_Amenities_Billed_Category']='no_bills'
test.loc[(test['Total_Charge']>0) & (test['Total_Charge']<=716),'Total_Amenities_Billed_Category']='Low'
test.loc[(test['Total_Charge']>716) & (test['Total_Charge']<=1442),'Total_Amenities_Billed_Category']='Medium'
test.loc[test['Total_Charge']>1442,'Total_Amenities_Billed_Category']='High'

test[["Cabin_deck","Cabin_no.","Cabin_side"]]=test["Cabin"].str.split("/",expand=True)


# # **Missing Values**
# 
# For better missing values processing, I will combine train/test datasets in one set, then I will split it back later. 
# 
# But first we have many columns that extracted from above EDA, some of them are informative and some bacame no informative after feature enginnering, so I will drop the less informative features.

# **Combining & Dropping less informative features:**

# In[17]:


target=train['Transported']  ## Will be used then for model fitting. 
train_rows=train.shape[0]  

combine=pd.concat([train,test],sort=False, ignore_index=True)  ## Combine Taining / Testing Datasets

cols_to_drop=['PassengerId', 'Cabin', 'Name','Transported',  'Group', 'Member', 'No_members_in_group',
             'Total_Amenities_Billed_Category', 'is_alone']

combine=combine.drop(cols_to_drop, axis=1)


# **Exploring Mising Values:**

# In[18]:


count_missing_values=combine.isna().sum()
percentage=np.round((count_missing_values/combine.shape[0])*100,2)
missing_val_df=pd.DataFrame({'count_missing_values':count_missing_values,'percentage %':percentage})
missing_val_df


# Nice! the missing values are relativelly small, but we still need to fill in these missing values for machine learning model fitting and predection.
# 
# For object columns, I'll fill nans by the mode of each column.
# 
# For numerical columns, I will fill nans by median of each column.

# In[19]:


categoriacl_cols=['HomePlanet', 'CryoSleep', 'Destination','VIP', 'surname',
                  'Cabin_deck', 'Cabin_no.','Cabin_side']

numerical_cols=['Age','RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Total_Charge']

def fill_missing(df):
    
    for ft in categoriacl_cols:
        df[ft].fillna(df[ft].mode()[0],inplace=True)
        
    for ft in numerical_cols:
        df[ft].fillna(df[ft].median(),inplace=True)
        
    return df

fill_missing(combine)
combine.isna().any()


# # **Data Preparation:**

# **Features Encoding:**
# 
# After filling the missing values, now our data is almost ready for model fitting, but first each of object feature should be converted to numerical one..
# 
# Therefore, I will use LabelEncoder from sklearn.preprocessing in order to replace each unique value at each feature by a number represents this value, also I will transform numerical features using log transformation.

# In[20]:


def encoder(df):
    
    for ft in categoriacl_cols:
        le=LabelEncoder()
        le.fit(df[ft])
        df[ft]=le.transform(df[ft])
        
    for ft in numerical_cols:
        df[ft]=np.log1p(df[ft])
    
    return df

combine=encoder(combine)
combine.info()


# **Data Splitting:**
# 
# Now data is completely ready to be as an input for Machine Learning part.
# 
# Finally, we have to re-split the combined dataset as it was to train & test datasets.

# In[21]:


## Split combine dataset into train & test:

train=combine.iloc[:train_rows,:]
test=combine.iloc[train_rows:,:]

X=train.values

let=LabelEncoder()
let.fit(target)
y=let.transform(target)

print('X shape is: ',X.shape)
print('y shape is: ',y.shape)


# # **Model Selection**

# Now, we reached the most sexy part in the competition.
# 
# This is a Classification Surprvised Machine Learning problem, so I will choose 4 classifier algorithms,tune thier hyperparameters using RandomizedSearchCV to find the best hyperparameter, an finally, I will take the final prediction through Stacking Voting calssifier.
# 
# The models I choosed to train them on the data are:
# 
# **Random Forest (RF):** RF is a reliable ensemble of decision trees, which can be used for regression or classification problems. Here, the individual trees are built via bagging (i.e. aggregation of bootstraps which are nothing but multiple train datasets created via sampling with replacement) and split using fewer features. The resulting diverse forest of uncorrelated trees exhibits reduced variance; therefore, is more robust towards change in data and carries its prediction accuracy to new data. It works well with both continuous & categorical data.
# 
# **Light Gradient Boosting Machine (LGBM):** LGBM works essentially the same as XGBoost but with a lighter boosting technique. It usually produces similar results to XGBoost but is significantly faster.
# 
# **Categorical Boosting (CatBoost):** CatBoost is an open source algorithm based on gradient boosted decision trees. It supports numerical, categorical and text features. It works well with heterogeneous data and even relatively small data. Informally, it tries to take the best of both worlds from XGBoost and LGBM.

# In[22]:


## CatBoostClassifer Hyperparameters Tunning:
cat_parameters = {
    'learning_rate': [0.5, 0.6, 0.7],
    'depth': [8,9,10], 'l2_leaf_reg': [9,11,10],'iterations': [35, 40, 50]
                 }

cat_model = CatBoostClassifier(silent=True)
cat_grid = RandomizedSearchCV(cat_model, cat_parameters, cv = 5, scoring = 'accuracy', n_jobs = -1, n_iter=20)
cat_grid.fit(X, y)
print('------------------CatBoostClassifier------------------')
print('Best Parameters : ', cat_grid.best_params_)
print()
print('Best Accuracy : ', cat_grid.best_score_)
print('------------------------------------------------------')
best_cat_model=cat_grid.best_estimator_

## RandomForestClassifier Hyperparameters Tunning:
rfc_parameters = {
    'n_estimators': [500,550],'min_samples_split': [7,8,9],
    'max_depth': [10,11,12], 'min_samples_leaf': [4, 5, 6]
                 }

rfc_model = RandomForestClassifier()
rfc_grid = RandomizedSearchCV(rfc_model, rfc_parameters, cv = 5, scoring = 'accuracy', n_jobs = -1, n_iter=20)
rfc_grid.fit(X, y)
print('------------------RandomForestClassifier------------------')
print('Best Parameters : ', rfc_grid.best_params_)
print()
print('Best Accuracy : ', rfc_grid.best_score_)
print('----------------------------------------------------------')
best_rfc_model=rfc_grid.best_estimator_

## LGBMClassifier Hyperparameters Tunning:
lgbm_parameters = {
    'n_estimators': [550,600,650],'learning_rate': [0.0095,0.01,0.02],
    'num_leaves': [16,17,18]
                 }

lgbm_model = LGBMClassifier()
lgbm_grid = GridSearchCV(lgbm_model, lgbm_parameters, cv = 5, scoring = 'accuracy', n_jobs = -1)
lgbm_grid.fit(X, y)
print('------------------LGBMClassifier------------------')
print('Best Parameters : ', lgbm_grid.best_params_)
print()
print('Best Accuracy : ', lgbm_grid.best_score_)
print('--------------------------------------------------')
best_lgbm_model=lgbm_grid.best_estimator_


# In[23]:


## Stacking
stacking_model = StackingClassifier(estimators=[('RF', best_rfc_model),
                                                ('LGBM', best_lgbm_model), 
                                                ('CAT', best_cat_model)
                                               ])
stacking_model.fit(X, y)


# # **Prediction**

# In[24]:


# Prediction
pred = stacking_model.predict(test)
pred = pred.reshape(-1, 1)
vot_pred = let.inverse_transform(pred) ## To return values to True / False
vot_pred = vot_pred.reshape(len(vot_pred),)


# # **Sumbission**

# In[25]:


submission=pd.read_csv('/kaggle/input/spaceship-titanic/sample_submission.csv')
submission['Transported']=vot_pred

submission.to_csv('submission.csv', index = False)

