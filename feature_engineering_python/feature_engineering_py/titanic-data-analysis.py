#!/usr/bin/env python
# coding: utf-8

# # Ever thought of having the power to predict the future? 
# 
# ## Well, Now you can! Today we will work on predicting the survival rate of hapless voyage of the mighty RMS Titanic
# 
# ## And the big Question is - Who will survive and who will not!

# ![titanic](https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg)

# In[1]:


# Importing important libraries to analyze the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Initial inspection of the data

# In[2]:


# Now it's time to look at the ultimate treasure - DATA - Importing in 3...2...1

titanic = pd.read_csv('../input/titanic/train.csv')


# In[3]:


titanic.set_index('PassengerId')


# In[4]:


test_titanic = pd.read_csv('../input/titanic/test.csv')


# In[5]:


test_titanic.shape


# In[6]:


test_titanic.set_index('PassengerId')


# In[7]:


titanic.index


# In[8]:


# A nice data dictionary will help us better understand the data

data ={
'Variable' : ['survival', 'pclass','sex','Age','sibsp','parch','ticket','fare','cabin','embarked'],
    'Definition' : ['Survival','Ticket class','Sex','Age in years','# of siblings / spouses aboard the Titanic',
                    '# of parents / children aboard the Titanic','Ticket number','Passenger fare','Cabin number',
                   'Port of Embarkation']}

Data_Dictionary = pd.DataFrame(data = data,columns=['Variable','Definition','Key'])
Data_Dictionary.fillna(' ', inplace=True)

Data_Dictionary.Key[0] = 'No -> 0, Yes -> 1'
Data_Dictionary.Key[1] = '1 = 1st, 2 = 2nd, 3 = 3rd'
Data_Dictionary.Key[9] = 'C = Cherbourg, Q = Queenstown, S = Southampton'


# In[9]:


titanic.head()


# In[10]:


Data_Dictionary


# In[11]:


# Checking its shape

titanic.shape


# In[12]:


# Checking how the data is organized and its data types. 891 records for survival because test data doesn't have target variable.

titanic.info()


# In[13]:


# Quantitative description of the dataset

titanic.describe()


# In[14]:


# checking null values

titanic.isnull().sum()


# In[15]:


titanic.columns


# In[16]:


titanic.shape


# In[17]:


torr = titanic.corr()
plt.figure(figsize=(10,7))
sns.heatmap(torr)


# ##  Let's analyse each column individully and visualize accordingly
# 
# # 1. Survived

# In[18]:


titanic.Survived.value_counts()

# Since test data does not contain this variable


# In[19]:


round(((titanic[titanic.Survived ==1].SibSp != 0).sum()/342)*100,2)

# Out of 342 people who survived, almost 39% had siblings or their spouse on board


# In[20]:


round(((titanic[titanic.Survived ==0].SibSp != 0).sum()/549)*100,2)

# Out of 549 people who died, around 28% had siblings or their spouse on board


# In[21]:


round(((titanic[(titanic.Survived ==1)].Parch != 0).sum()/342)*100,2)

# Out of 342 people who survived, 32% had their parents or children on board


# In[22]:


round(((titanic[(titanic.Survived ==0)].Parch != 0).sum()/549)*100,2)

# Out of 549 people who died, 19% had their parents or children on board


# In[23]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=titanic.Survived, hue=titanic.Pclass, palette='Set3')

# When looking from class perspective, those who lost their lives belonged to class 3 and naturally those who got out alive belonged to class 1 mostly


# In[24]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=titanic.Survived, hue=titanic.Sex, palette='Set3')

# Majority who died were males


# In[25]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.boxplot(x=titanic.Survived, y = titanic.Age, palette='Set3')

# There is no significant differences between ages of deceased and alive


# In[26]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.violinplot(x=titanic.Survived, y = titanic.Fare, palette='Set3', hue=titanic.Sex)

# The plot indicates that the fare amount and corrsponding casualities are inversly proportional but the dataset mostly contains fare in the range 0-100


# In[27]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=titanic.Survived, hue=titanic.Parch, palette='Set3')


# In[28]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=titanic.Survived, hue=titanic.SibSp, palette='Set3')


# In[29]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.stripplot(x=titanic.Survived, y = titanic.Fare, palette='Set3',linewidth=0.7, jitter=0.4)


# In[30]:


plt.figure(figsize=(12,8))
sns.set_theme(style='whitegrid')
sns.countplot(x=titanic.Survived, hue=titanic.Embarked, palette='Set3')

# People aboard were mostly from southampton and also the ones who lost most lives, which is intuitive!


# # 2. Pclass - 
# ### It is defined as the deck number or the class number to which people were assigned and has total of 3 values i.e 1,2 and 3 

# In[31]:


titanic.Pclass.value_counts()  # Majority are from 3rd class


# In[32]:


titanic.Pclass.isnull().sum() # No missing points


# In[33]:


titanic.groupby('Pclass').mean()


# In[34]:


plt.figure(figsize=(10,8))
sns.set_theme(style='ticks')
sns.countplot(x=titanic.Pclass, hue=titanic.Sex, palette='Set1' )

# Most people belonged to 3rd class with male as the clear majority


# In[35]:


plt.figure(figsize=(10,8))
sns.set_theme(style='ticks')
sns.stripplot(x=titanic.Pclass, y=titanic.Age, palette='Set1', linewidth=0.6, jitter= 0.3)

# We have people of all age group similarly distributed across three classes with elederly people more in class 1


# In[36]:


plt.figure(figsize=(10,8))
sns.set_theme(style='ticks')
sns.boxplot(x=titanic.Pclass, y = titanic.Fare,  palette='Set1', hue = titanic.Sex )

# There are some outliars in fare for people belonging to class 1 and females in that class had expensive tickets than males.


# In[37]:


plt.figure(figsize=(10,8))
sns.set_theme(style='ticks')
sns.countplot(x=titanic.Pclass, hue=titanic.Embarked, palette='Set1')

# Southampton seems to be the most popular or the biggest yard according to the figure below


# # 3. Name

# In[38]:


titanic.Name


# In[39]:


titanic.Name.isnull().sum()


# In[40]:


titanic.Name.map(lambda x : x.split(',')[0]).value_counts()[:15]

# We can also see names of people having same last name and can maybe infer that those are family members


# In[41]:


titanic[titanic.Name.map(lambda x : x.split(',')[0]) == titanic.Name.map(lambda x : x.split(',')[0]).value_counts().index[3]]


# In[42]:


titanic.Name.describe()


# In[43]:


titanic.head()


# # 4. Sex

# In[44]:


titanic.Sex.value_counts()

#Majority are males and no missing values


# In[45]:


titanic.Sex.isnull().sum()


# In[46]:


titanic[titanic.Sex == 'female'].Name.map(lambda x : x.split(',')[1].split('.')[0].strip()).value_counts()

# There are no gender specific titles that are wrongly labelled in female category


# In[47]:


titanic[titanic.Sex == 'male'].Name.map(lambda x : x.split(',')[1].split('.')[0].strip()).value_counts()

# There are no gender specific titles that are wrongly labelled in male category


# In[48]:


plt.figure(figsize=(16,10))
sns.set_theme(style='whitegrid')
sns.stripplot(y=titanic.Age, x = titanic.Sex, hue = titanic.Pclass, palette='tab10',jitter=0.3, linewidth=1,size=10)


# ## 4. Age

# In[49]:


titanic.Age.describe()

# We have some missing values and it is much less than half, so we will impute missing values with a method


# In[50]:


titanic.Age.value_counts()


# In[51]:


sns.distplot(x = titanic.Age)


# In[52]:


titanic[titanic.Age < 1].groupby('Sex').mean()


# In[53]:


titanic[(titanic.Age < 18) & (titanic.Survived == 1)].groupby('Sex').mean()


# In[54]:


titanic.Age = titanic.Age.fillna(titanic.Age.median(),axis=0)


# In[55]:


titanic.Age.describe()


# In[56]:


titanic.head()


# In[57]:


age_bins = pd.cut(titanic.Age, bins=4, include_lowest=True, ordered=True, retbins=True)[0]


# In[58]:


plt.figure(figsize=(12,8))
sns.countplot(x = age_bins, hue=titanic.Sex, palette='tab10')

# Majorly people onboard belonged to the age group of [16,33] in which males had a larger number


# In[59]:


plt.figure(figsize=(12,8))
sns.countplot(x = age_bins, hue=titanic.Survived, palette='tab10')


# In[60]:


plt.figure(figsize=(12,8))
sns.barplot(x = age_bins, y = titanic.Fare, hue=titanic.Sex, palette='tab10')

# Average fare for women was particularly high than the corresponding males in the same age group.


# In[61]:


plt.figure(figsize=(12,8))
sns.countplot(x = age_bins, hue=titanic.Pclass, palette='tab10')

# we see that 3rd class had more younger people and better chances of surviving


# In[62]:


plt.figure(figsize=(12,8))
sns.countplot(x = age_bins, hue=titanic.Embarked, palette='tab10')


# # 5. SibSp - It is defined as the number of siblings/spouse present of the person in consideration

# In[63]:


titanic.SibSp.describe() # No missing values


# In[64]:


titanic.SibSp.value_counts()


# In[65]:


titanic.groupby('SibSp').mean()

# We can see that those with less family members on board had more average survival rate


# In[66]:


titanic.loc[titanic.SibSp == 8]


# In[67]:


titanic.loc[titanic.SibSp == 5]


# In[68]:


(titanic[((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 3)].Survived.sum())/(titanic
                                    [((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 3)].Survived.count())

# Only around 30% people people having at least one family member and belonging to class 3 came out alive and mostly died


# In[69]:


(titanic[((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 2)].Survived.sum())/(titanic
                                    [((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 2)].Survived.count())

# Around 65% people having at least one family member and belonging to class 2 came out alive and that is a pretty good figure


# In[70]:


(titanic[((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 1)].Survived.sum())/(titanic
                                    [((titanic.SibSp > 0) | (titanic.Parch >0)) & (titanic.Pclass == 1)].Survived.count())

# Around 70% people having at least one family member and belonging to class 1 came out alive and that tells us the intuitive result of better facilities and
# better accomodomation with security in class 1.


# In[71]:


plt.figure(figsize=(12,8))
sns.countplot(x=titanic.SibSp, hue=titanic.Sex, palette='pastel')


# In[72]:


plt.figure(figsize=(12,8))
sns.countplot(x=titanic.SibSp, hue=titanic.Embarked, palette='pastel')


# In[73]:


plt.figure(figsize=(12,8))
sns.countplot(x=titanic.SibSp, hue=titanic.Survived, palette='pastel')


# In[74]:


plt.figure(figsize=(16,8))
sns.stripplot(x=titanic.SibSp, y = titanic.Age ,  palette='pastel', size = 7, linewidth=1,jitter=0.2)


# In[75]:


plt.figure(figsize=(16,8))
sns.stripplot(x=titanic.SibSp, y = titanic.Fare ,  palette='pastel', size = 7, linewidth=1,jitter=0.2)


# In[76]:


plt.figure(figsize=(16,8))
sns.violinplot(x=titanic.SibSp, y = titanic.Fare , hue=titanic.Sex, split=True, palette='pastel')


# # 6. Parch - It represents number of Parents/Children present on Titanic at the time of the crash

# In[77]:


titanic.Parch.describe() # No missing values


# In[78]:


titanic.Parch.value_counts()

# We see that values correspoding to 3,4,5,6 Parch are very less. 
#If the sample ends up in the test set during splitting then the classifier would not have seen the category during training and will not be able to 
#encode it.
#In scikit-learn, there are two solutions to bypass this issue:

#1. list all the possible categories and provide it to the encoder via the keyword argument 'categories';
#2. use the parameter handle_unknown.


# In[79]:


titanic.loc[titanic.Parch == 6]


# In[80]:


titanic.loc[titanic.Parch == 5]


# In[81]:


plt.figure(figsize=(14,8))
sns.countplot(x=titanic.Parch, hue=titanic.Sex, palette='flare')


# In[82]:


plt.figure(figsize=(14,8))
sns.countplot(x=titanic.Parch, hue=titanic.Survived, palette='flare')


# In[83]:


plt.figure(figsize=(16,8))
sns.barplot(x=titanic.Parch,y=titanic.Fare, hue=titanic.Survived, palette='flare')


# # 7. Ticket

# In[84]:


titanic.Ticket


# In[85]:


titanic.Ticket.isnull().sum()


# In[86]:


titanic.Ticket.value_counts()[:20]


# In[87]:


titanic[titanic.Ticket == titanic.Ticket.value_counts().index[3]]


# # 8. Fare

# In[88]:


titanic.Fare.describe() # One missing value


# In[89]:


titanic.Fare.fillna(titanic.Fare.mean(),axis=0, inplace=True)


# In[90]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=titanic.Fare, y = titanic.Age)


# In[91]:


plt.figure(figsize=(12,8))
sns.boxplot(x=titanic.Embarked, y = titanic.Fare)


# # 9. Cabin

# In[92]:


titanic.Cabin.isnull().sum()


# In[93]:


titanic.Cabin.describe()

# Out of 1309 samples, we have 1014 absent which is more than 75% 
# Dropping cabin column


# In[94]:


titanic.drop('Cabin', axis=1, inplace=True)


# In[95]:


titanic.head()


# # 10. Embarked

# In[96]:


titanic.head()


# In[97]:


titanic.Embarked.describe()

# we see that it has a dtype of object, which is string


# In[98]:


titanic.Embarked.value_counts() 
# C = Cherbourg, Q = Queenstown, S = Southampton


# In[99]:


titanic.Embarked.isnull().sum()


# In[100]:


titanic.Embarked.fillna(value= titanic.Embarked.mode(),axis=0)


# In[101]:


titanic[titanic.Embarked.isnull()]

# We see that fillna did not work because that function does not recognize them as none type


# In[102]:


type(titanic[titanic.Embarked.isnull()].Embarked.iloc[0]) 

# It's a float type! STRANGE


# In[103]:


# So let us convert our dtype to object type that they are homogenous! 

titanic.Embarked = titanic.Embarked.astype(str)


# ## !
# ##### Learning tip - If your column contains string values, then its numpy dtype will be shown as 'object' but it is not necsesrily true backwords like we saw above. If your column has a dtype of object, they can still contain mixed data types. 

# In[104]:


titanic[titanic.Embarked.isnull()]


# In[105]:


titanic['Embarked'].replace('nan','S', inplace=True)


# In[106]:


titanic.iloc[829]


# In[107]:


titanic[titanic.Embarked == 'nan']


# In[108]:


titanic.Embarked.isnull().sum()


# In[109]:


titanic.groupby('Embarked').mean()
# Cherbourg had a large population of elite people


# In[110]:


titanic.groupby('Embarked').agg(['min']) # We see below that children died mostly, we should find out among the children, who had more chances


# In[111]:


titanic.groupby('Embarked').agg(['max'])


# # Feature Engineering

# In[112]:


test_titanic.isnull().sum()


# In[113]:


test_titanic.Age = test_titanic.Age.fillna(test_titanic.Age.median(),axis=0)


# In[114]:


test_titanic.drop('Cabin', axis=1,inplace=True)


# In[115]:


test_titanic.isnull().sum()


# In[116]:


titanic.head()


# In[117]:


# Since all the siblings/spouses and parents/children constutute one family, we can have the total family members as a feature

titanic['Family_members'] = titanic['SibSp'] + titanic['Parch'] + 1
test_titanic['Family_members'] = test_titanic['SibSp'] + test_titanic['Parch'] + 1


# In[118]:


titanic.head()


# In[119]:


titanic['Title'] = titanic.Name.map(lambda x : (x.split(',')[1].split('.')[0].strip()))


# In[120]:


test_titanic['Title'] = test_titanic.Name.map(lambda x : (x.split(',')[1].split('.')[0].strip()))


# In[121]:


titanic.Title


# In[122]:


titanic.Title.value_counts()

# Commonly occuring titles are Mr, Miss, Mrs, Master and Dr


# #### Let's define those obscure titles or titles native to some country that may not be known to common public 
# 
# ##### Rev - (as the title of a priest) Reverend
# ##### Col - Colonel
# ##### Mlle - Mademoiselle (in French)
# ##### Dona - a woman or girlfriend (in Italian)
# ##### Jonkheer - Honorific of nobility is literally translated as "young lord" or "young lady" in Low Countries
# ##### Mme - Madame (is a traditional alternative for an unmarried woman in France)
# ##### Don - Don, abbreviated as D., is an honorific prefix primarily used in Spain and the former Spanish Empire, Croatia, Italy, and Portugal and its former colonies.

# In[123]:


# Let's check if these titles belong to male or female category in order to classify them properly
# titanic.loc[titanic.Title == 'Rev'] --> All male so adding to Mr 
# titanic.loc[titanic.Title == 'Col'] --> All male so adding to Mr
# titanic.loc[titanic.Title == 'Mlle'] --> Adding to Miss
# titanic.loc[titanic.Title == 'Dona'] --> Adding to Mrs
# titanic.loc[titanic.Title == 'Jonkheer'] --> Adding to Mr
# titanic.loc[titanic.Title == 'Mme'] --> Adding to Miss
# titanic.loc[titanic.Title == 'Don'] --> Adding to Mr
# titanic.loc[titanic.Title == 'the Countess'] --> Adding to Mrs
# titanic.loc[titanic.Title == 'Lady'] --> Adding to Mrs


# In[124]:


# We can combine rare occurings of titles under one title also beacuse it may interfere with our model if one observation occurs in test data and not in train
# data which will lead to hampered accuracy.

titanic.Title = titanic.Title.replace(['Sir','Capt','Don','Col','Jonkheer','Rev','Major'], 'Mr')

titanic.Title = titanic.Title.replace(['Dona','the Countess','Lady'], 'Mrs')

titanic.Title = titanic.Title.replace(['Ms','Mlle','Mme'], 'Miss')

titanic.Title.value_counts()

# Now we have substantial samples for balanced distribution across train and test sets


# In[125]:


test_titanic.Title = test_titanic.Title.replace(['Sir','Capt','Don','Col','Jonkheer','Rev','Major'], 'Mr')

test_titanic.Title = test_titanic.Title.replace(['Dona','the Countess','Lady'], 'Mrs')

test_titanic.Title = test_titanic.Title.replace(['Ms','Mlle','Mme'], 'Miss')

test_titanic.Title.value_counts()


# In[126]:


titanic['Name_length'] = titanic.Name.map(lambda x : len(x)).values

plt.figure(figsize=(12,8))
sns.color_palette('Set2')
sns.boxplot(y= titanic.Name_length, x = titanic.Survived, hue = titanic.Sex, palette='Set2')

# Oh yeah, we can see that there is a slight relationship between people's name and their survival chances.


# In[127]:


test_titanic['Name_length'] = test_titanic.Name.map(lambda x : len(x)).values


# In[128]:


titanic.Name_length.describe()


# In[129]:


def namelength_bins(x):
    int(x)
    if (x > 10) and (x<=20):
        return 0
    elif (x>20) and (x<=25):
        return 1
    elif (x>25) and (x<=30):
        return 2
    elif (x>30) and (x<=85):
        return 3
    
    
titanic.Name_length = titanic.Name_length.map(lambda x : namelength_bins(x))
titanic.Name_length.value_counts()


# In[130]:


test_titanic.Name_length = test_titanic.Name_length.map(lambda x : namelength_bins(x))
test_titanic.Name_length.value_counts()


# In[131]:


# Let's create bins for age column

def age_bins(x):
    int(x)
    if (x>0) and (x<=20):
        return 0
    elif (x>20) and (x<=40):
        return 1
    elif (x>40) and (x<=60):
        return 2
    elif (x>60) and (x<=90):
        return 3
    
    
titanic.Age = titanic.Age.map(lambda x : age_bins(x))
titanic.Age.value_counts()


# In[132]:


test_titanic.Age = test_titanic.Age.map(lambda x : age_bins(x))
test_titanic.Age.value_counts()


# In[133]:


def title_bins(x):
    if x == 'Mr':
        return 0
    elif x == 'Mrs':
        return 1
    elif x == 'Miss':
        return 2
    elif x == 'Master':
        return 3
    else:
        return 4
    
    
titanic.Title = titanic.Title.map(lambda x : title_bins(x))
titanic.Title.value_counts()


# In[134]:


test_titanic.Title = test_titanic.Title.map(lambda x : title_bins(x))
test_titanic.Title.value_counts()


# In[135]:


def embarked_bins(x):
    
    if x == 'C':
        return 0
    elif x=='S':
        return 1
    else:
        return 2
    
    
titanic.Embarked = titanic.Embarked.map(lambda x : embarked_bins(x))
titanic.Embarked.value_counts()


# In[136]:


test_titanic.Embarked = test_titanic.Embarked.map(lambda x : embarked_bins(x))
test_titanic.Embarked.value_counts()


# In[137]:


# We can atleast extract numerical values of the ticket to know more info about it

#titanic.Ticket.map(lambda x : x.split(" ")[-1].strip()).value_counts()


# In[138]:


#titanic.Ticket = titanic.Ticket.map(lambda x : x.split(" ")[-1].strip())


# In[139]:


titanic.head()


# In[140]:


titanic.drop(columns=['Name','Ticket'],axis=1,inplace=True)


# In[141]:


test_titanic.drop(columns=['Name','Ticket'],axis=1,inplace=True)


# In[142]:


test_titanic.head()


# In[143]:


#titanic[titanic.Ticket == 'LINE']


# In[144]:


#titanic.Ticket = titanic.Ticket.replace('LINE', '21171')


# In[145]:


titanic.info()


# In[146]:


test_titanic.info()


# In[147]:


#titanic.Ticket = titanic.Ticket.astype('int64')


# In[148]:


from sklearn.compose import make_column_selector as selector

# Trying sklearn to select to play with dtypes of columns


# In[149]:


column_selector = selector(dtype_include=object)
category = column_selector(titanic)
category


# In[150]:


categorical_data = titanic[category]


# In[151]:


category_test = column_selector(test_titanic)
category_test


# In[152]:


categorical_data_test = test_titanic[category]


# In[153]:


categorical_data_test.head()


# In[154]:


from sklearn.preprocessing import OneHotEncoder


# In[155]:


encoder = OneHotEncoder(sparse=False)


# In[156]:


encoded_columns = encoder.fit_transform(categorical_data)


# In[157]:


encoded_columns_test = encoder.fit_transform(categorical_data_test)


# In[158]:


encoded_columns


# In[159]:


feature_names = encoder.get_feature_names(input_features=categorical_data.columns)


# In[160]:


feature_names_test = encoder.get_feature_names(input_features=categorical_data_test.columns)


# In[161]:


feature_names


# In[162]:


feature_names_test


# In[163]:


features_list = []
for i in feature_names.tolist():
    features_list.append(i.split("_")[-1])


# In[164]:


features_list


# In[165]:


data_encoded = pd.DataFrame(columns=features_list, data = encoded_columns)


# In[166]:


data_encoded_test = pd.DataFrame(columns=features_list, data = encoded_columns_test)


# In[167]:


data_encoded.set_index(titanic.index,inplace=True)


# In[168]:


data_encoded_test.set_index(test_titanic.index,inplace=True)


# In[169]:


new_titanic = pd.concat([titanic,data_encoded],axis=1)


# In[170]:


new_titanic_test = pd.concat([test_titanic,data_encoded_test],axis=1)


# In[171]:


new_titanic.dtypes


# In[172]:


new_titanic_test.dtypes


# In[173]:


new_titanic_test.drop('Sex',axis=1,inplace=True)


# In[174]:


column_selector = selector(dtype_exclude= 'object')


# In[175]:


final_titanic = new_titanic[column_selector(new_titanic)]


# In[176]:


final_titanic


# In[177]:


final_titanic.info()


# In[178]:


new_titanic_test.info()


# In[179]:


plt.figure(figsize=(14,8))
torr = final_titanic.corr()
sns.heatmap(torr)


# # Feature Scaling

# In[180]:


# Since we have a couple of features with really large values that might interfere with out model training, we will scale them to represent a balanced 
# training set

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[181]:


final_titanic.head()


# In[182]:


#final_titanic[['Name_length','Fare']] = scaler.fit_transform(final_titanic[['Name_length','Fare']].values)


# # Model Building

# In[183]:


final_titanic.head()


# In[184]:


target_class = 'Survived'
target = final_titanic.pop(target_class)


# In[185]:


final_titanic


# In[186]:


#final_titanic.set_index('PassengerId')


# In[187]:


new_titanic_test.set_index('PassengerId')


# In[188]:


final_titanic.drop(columns=['SibSp','Parch','female','PassengerId'],axis=1,inplace=True)


# In[189]:


final_titanic.head()


# In[190]:


from sklearn.linear_model import LogisticRegression


# In[191]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# In[192]:


logr = LogisticRegression(C = 10)


# In[193]:


logr.fit(final_titanic,target)


# In[194]:


knn.fit(final_titanic,target)


# In[195]:


new_titanic_test.head()


# In[196]:


new_titanic_test.drop(columns=['SibSp','Parch','female'],axis=1,inplace=True)


# In[197]:


new_titanic_test.isnull().sum()


# In[198]:


#new_titanic_test.dropna(axis=0,inplace=True)


# In[199]:


new_titanic_test.fillna(np.mean(new_titanic_test.Fare), inplace=True)


# In[200]:


sns.heatmap(new_titanic_test.corr())


# In[201]:


new_titanic_test.shape


# In[ ]:





# In[202]:


predictions = pd.DataFrame(data = logr.predict(new_titanic_test.drop('PassengerId',axis=1)),index=new_titanic_test.index)


# In[203]:


from sklearn.metrics import accuracy_score


# # Learning and Validation curves
# 
# ## They allow us to monitor system's performance and gives us ideas to tweak the model to convert it into a better performing model.

# ### Learning curves in a nutshell:
# * Learning curves allow us to diagnose if the model is **overfitting** or **underfitting**.
# * When the model **overfits**, it means that it performs well on the training set, but not not on the validation set. Accordingly, the model is not able to generalize to unseen data. If the model is overfitting, the learning curve will present a gap between the training and validation scores. Two common solutions for overfitting are reducing the complexity of the model and/or collect more data.
# * On the other hand, **underfitting** means that the model is not able to perform well in either training or validations sets. In those cases, the learning curves will converge to a low score value. When the model underfits, gathering more data is not helpful because the model is already not being able to learn the training data. Therefore, the best approaches for these cases are to improve the model (e.g., tuning the hyperparameters) or to improve the quality of the data (e.g., collecting a different set of features).

# ### Validation curves in a nutshell:
# * Validation curves are a tool that we can use to improve the performance of our model. It counts as a way of tuning our hyperparameters.
# * They are different from the learning curves. Here, the goal is to see how the model parameter impacts the training and validation scores. This allow us to choose a different value for the parameter, to improve the model.
# * Once again, if there is a gap between the training and the validation score, the model is probably overfitting. In contrast, if there is no gap but the score value is low, we can say that the model underfits.

# In[204]:


from sklearn.model_selection import learning_curve


# In[205]:


# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt


# In[206]:


# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)


# In[207]:


predictions


# In[208]:


predictions['PassengerId'] = new_titanic_test.PassengerId


# In[209]:


predictions


# In[210]:


predictions = predictions.astype('int64')


# In[211]:


predictions.columns


# In[212]:


predictions.rename({0:'Survived'}, axis=1,inplace=True)


# In[213]:


pred_pass = predictions.PassengerId


# In[214]:


predictions.drop('PassengerId',inplace=True,axis=1)


# In[215]:


predictions.insert(0, 'PassengerId', pred_pass)


# In[216]:


predictions


# In[217]:


predictions.to_csv('titanicfirstsubmission.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




