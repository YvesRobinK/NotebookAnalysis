#!/usr/bin/env python
# coding: utf-8

# # An Example of Data Science work on Titanic Dataset

# ## A Guide on how to perform different feature engineering tasks
# ### From Imputations, implementing statistics to making use of Probability prediction 

# ### <span style="color:green"> If you like my work, learned something out of it or found it useful , an upvotes would be really appreciated :-) </span>

#  
#  
# #### We will start from exploring the data to see what we have to do clean the provided data set. "Always consider in any data science problem we have to perform some exploratory data analysis as first steps" (if data is readily available, if its not like for data mining strategy will be a bit different)
# 
# 

# **it is important to have a clear understanding of what libs you will be using and import/install them beforehand**

# ### <span style="color:green"> Version 3.0: Added new plots from Seaborn 0.11 release

# In[24]:


#important libraries
import numpy as np 
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
#import seaborn as sns

import string
import warnings
warnings.filterwarnings('ignore')


# #### Explanations:
# ##### %matplotlib inline:
# ##### is specifically used here for jupyter notebook to store plots in notebook document 
# 
# ##### warnings: 
# #####    has been imported to avoid raise of warning when a function is deprecated

# ##### The main reason to have seaborn apart from matplot lib is It is used to create more attractive and informative statistical graphics

# In[25]:


# Importing provided dataset one for prediction and one for training and testing
train_frame = pd.read_csv('/kaggle/input/titanic/train.csv')
test_frame = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[26]:


train_frame.head()


#  

# ##### Note:
# 
# ##### shape function provide number of rows and column in a given dataset like (rows, cols)

# In[27]:


print(train_frame.shape)
print(test_frame.shape)


#  

# 
# 
# ##### so our train and pred data have 891 and 418 rows and 12 and 11 cols respectively, df_pred has obviously 1 col less because that is the target col which we have to predict

#  

# In[28]:


# to see what types of data each col contains
train_frame.dtypes


#  

# ##### Explanation:
# ##### Dataframe.Describe()
# 
# ##### provide a good EDA understanding of the dataset in hand, it provide mean, std and fragments of each 25% and is good to have a glimpse of outliers in data prior to visualzations

# In[29]:


train_frame.describe()


#  

# ##### Dataframe.describe(include='all') is used to provide information about categorical data. This helps us to identify what categories we are seeing the most 

#  

# In[30]:


#include all provide an understanding of category in form of unique values in each col and freq of most common value
train_frame.describe(include='all')


#  

# ##### Lets check how many null/missing values we have in both dataset so we can define our strategy of how to treat them

#  

# In[31]:


train_null = train_frame.isnull().sum().sort_values(ascending=False)
train_null


# In[32]:


null_test = test_frame.isnull().sum().sort_values(ascending=False)
null_test


# In[33]:


#performing EDA 
train_frame['Parch'].value_counts()


#  

# #### Note:
# ##### The main reason behind employing visualization is because it is helpful in understanding data/distributions by placing it in a visual context so that patterns, trends and correlations that might not otherwise be detected can be worked with

#  

# In[34]:


#initial visualization, droping missing rows to avoid errors in visuals
male = train_frame[train_frame['Sex']=='male']
female = train_frame[train_frame['Sex']=='female']
x = male[male['Survived']==1].Age.dropna()
x1 = male[male['Survived']==0].Age.dropna()
y = female[female['Survived']==1].Age.dropna()
y1 = female[female['Survived']==0].Age.dropna()


# In[35]:


#lets have some visualization on sex and survial ratio

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
ax = sns.distplot(x, bins=15, label = 'survived', ax = axes[0], kde = False, color = 'g')
ax = sns.distplot(x1, bins=30, label = 'not survived', ax = axes[0], kde = False, color = 'b')
ax.legend()
ax.set_title('Male')
ax = sns.distplot(y, bins=15, label = 'survived', ax = axes[1], kde = False, color = 'y')
ax = sns.distplot(y1, bins=30, label = 'not survived', ax = axes[1], kde = False, color = 'r')
ax.legend()
ax.set_title('Female')
plt.show()


# In[36]:


#survival ration w.r.t to class
sns.barplot(x='Pclass', y='Survived', data=train_frame)


# In[37]:


sns.barplot(x='Parch', y='Survived', data=train_frame)


# In[38]:


dft1 = train_frame.copy()
cat_cols = ['Sex', 'Embarked', 'Survived']

dft1[cat_cols]= dft1[cat_cols].astype('category')


# In[39]:


f,ax =plt.subplots(len(cat_cols),1,figsize=(5,10))
for idx,col in enumerate(cat_cols):
    if col!='Survived':
        sns.countplot(x=col,data=dft1[cat_cols],hue='Survived', ax=ax[idx])


# In[40]:


#Advance Vis
a = sns.FacetGrid(train_frame, hue = 'Survived', aspect=4, palette="Set1" )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , train_frame['Age'].max()))
a.add_legend()


# In[41]:


b = sns.FacetGrid(train_frame, row = 'Sex', col = 'Pclass', hue = 'Survived', palette="Set2")
b.map(plt.hist, 'Age', alpha = .75)
b.add_legend()


# In[42]:


add_1 = sns.FacetGrid(train_frame, col = 'Embarked')
add_1.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
add_1.add_legend()


# In[43]:


sns.scatterplot(x="Age", y="Fare",
                     hue="Survived", palette="Set2",
                     sizes=(20, 200), hue_norm=(0, 7),
                     legend="full", data=train_frame)


# In[44]:


sns.scatterplot(x="Age", y="Fare", hue="Survived", size="Sex", palette="Set2",
                     sizes=(20, 200), hue_norm=(0, 7),
                     legend="full", data=train_frame)


# In[45]:


sns.scatterplot(x="Pclass", y="Parch",
                     hue="Survived", palette="Set1",
                     sizes=(20, 200), hue_norm=(0, 7),
                     legend="full", data=train_frame)


# In[46]:


cols = ['Survived', 'Pclass', 'Sex', 'Parch', 'Fare', 'Embarked']
sns.pairplot(train_frame[cols], diag_kind="hist", hue = 'Survived', palette="Set2")


# In[47]:


sns.jointplot(train_frame['Age'],train_frame['Survived'], kind="hex")
plt.title('Age Vs Survived')
plt.tight_layout()
plt.show()


# In[48]:


sns.jointplot(train_frame['Age'],train_frame['Fare'], kind="reg")
plt.title('Age Vs fare')
plt.tight_layout()
plt.show()


# ### Installing the newest release of Seaborn

# In[49]:


sns.jointplot(data=train_frame, x='Parch', y='Age',
            kind='hex')
plt.tight_layout()
plt.show()


# In[50]:


pip install seaborn==0.11.0


# In[51]:


import seaborn as sns


# In[52]:


sns.set()
sns.displot(data=train_frame, x='Age', y='SibSp', kind='hist',hue='Survived', height=6, aspect=1.2)


# In[53]:


sns.displot(data=train_frame, x='Age', kind='hist',col='Embarked', hue='Survived')


# In[54]:


sns.displot(
    data=train_frame, kind="hist", kde=True,
    x="Survived", col="Embarked", hue="Pclass",
)


# In[55]:


sns.displot(
    data=train_frame, kind="kde", rug=True,
    x="Age", y="Pclass",
    col="Embarked", hue="Survived",
)


#  

# #### Note: 
# ##### We now have some understanding of what we are expecting from different features like for 'Parch' we have decision boundires for survial indications of which classes are going with 100% survival and which are with 0% and so on.

#  

#  

#  

# ## Working on Mising data

# #### Explanation:
# ##### Filling missing values in dataset is of utmost importance because the fate and accuracy of your model rely heavily on your strategy. There are numerous ways for filling your missing data with most easy is to replace it by mean/median or most common value.

# #### What we will do
# ##### Here i have taken 3 dif strategies just for knowledge sharing as how we can fill nan/missing values. 
# 
# 

# #### 1. Filling NaN with Mean and Standard Deviation
# 
# #### 2. Imputations using simple impute/KNN
# 
# #### 3. Using statistics to fill NaN 

#  

# #### Note: 
# ##### sometime it is wise to drop missing data rows/cols mainly when there are too much nan values or if filling them not make any sense. This is part where our understanding of statistics and visualization skills provide us the insights about how to deal with your datset

#  

# In[56]:


df_train1 = train_frame.copy()
df_test1 = test_frame.copy()


# ### 1. Taking mean & std 

# ##### Since Age feature has the most num of NaN, easiest way is to take mean and fill is by +,_ of its std 

#  

# In[57]:


Age = [df_train1, df_test1]

for dataset in Age:
    #making use of both test and train frame
    mean = train_frame["Age"].mean()
    std = test_frame["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # computing random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated in range of mean +/- std
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df_train1["Age"].astype(int)
df_train1["Age"].isnull().sum()


#  

# #### Note:
# ##### This is not the most intelligent or accurate approach since there are different groups w.r.t to pclass, embarked and etc so the model understanding will be biased. So we will not be continuing with this approach

# In[58]:


#re-checking if any nulls of Age feautes 
train_null = df_train1.isnull().sum().sort_values(ascending=False)
train_null


#  

#  

#  

#  

# ### 2. Imputations
# 

# ##### Imputation is the process of replacing missing data with substituted values
# 
# #### Imputations can be performed in many ways as describe below:

# ##### SimpleFill: Replaces missing entries with the mean or median of each column.
# 
# ##### •KNN: Nearest neighbor imputations which weights samples using the mean squared difference on features for which two rows both have observed data.
# 
# ##### •SoftImpute: Matrix completion by iterative soft thresholding of SVD decompositions. Inspired by the softImpute package for R, which is based on Spectral Regularization Algorithms for Learning Large Incomplete Matrices by Mazumder et. al.
# 
# ##### •IterativeSVD: Matrix completion by iterative low-rank SVD decomposition. Should be similar to SVDimpute from Missing value estimation methods for DNA microarrays by Troyanskaya et. al.
# 
# ##### •MICE: Reimplementation of Multiple Imputation by Chained Equations.
# 
# ##### •MatrixFactorization: Direct factorization of the incomplete matrix into low-rank U and V, with an L1 sparsity penalty on the elements of U and an L2 penalty on the elements of V. Solved by gradient descent.
# 
# ##### •NuclearNormMinimization: Simple implementation of Exact Matrix Completion via Convex Optimization by Emmanuel Candes and Benjamin Recht using cvxpy. Too slow for large matrices.
# 
# ##### •BiScaler: Iterative estimation of row/column means and standard deviations to get doubly normalized matrix. Not guaranteed to converge but works well in practice. Taken from Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares.

#  

#  

# ##### For learning purpose will use 2 imputations Simple Impute and KNN since others are out of context for this task
# ##### (You are free to experiment with others as this improve your understanding of different process)

# #### 2.1 Imputing Using Simple Impute

#  

# In[59]:


df1 = train_frame['Age'].isnull()
df2 = pd.DataFrame(df1)
df2.columns = ["new"]
df2.head()


# In[60]:


#using boolean expressions to divide Nan rows
df3 = train_frame[df2['new']==True] #for nan val of age
df4 = train_frame[df2['new']==False]
df3.head()


# In[61]:


df4.head()


# In[62]:


get_ipython().system(' pip install datawig')


# In[63]:


import datawig


# In[64]:


#Using a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns = ['Sex','Pclass', 'Parch', 'Embarked', 'Survived'], output_column = 'Age',
    output_path = 'imputer_model')
#input cols serves as data on whose basis imputation is performed
#output col is one on which imputation will be performed


# In[65]:


#Fit an imputer model on the train data
imputer.fit(train_df=df4, num_epochs=50)


# In[66]:


#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df3)


# In[67]:


imputed["Age_imputed"].isnull().sum()


#  

# ##### As we now have 2 features of 'Age', 1 is orignal with Nan and second with imputated values so droping NaN col.

# In[68]:


del imputed['Age']


# In[69]:


imputed = imputed.rename(columns={'Age_imputed' : 'Age'})


# In[70]:


imputed.head()


# You can see how the imputer has transformed our Age bin

# In[71]:


imputed = imputed[['PassengerId','Survived', 'Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin',
                   'Embarked']]


# In[72]:


df6 = imputed.append(df4)
df6.sort_values(by=['PassengerId'])


# In[73]:


df6["Age"].isnull().sum()


#  

#  

# #### 2.2 Imputing Using KNN

#  

# ##### This part is just to understand how KNN can be useful for filling NaN values, we'll just cover how its done and move forward to next strategy

#  

# In[74]:


from fancyimpute  import KNN 


# In[75]:


cols = ['Survived', 'Pclass', 'SibSp', 'Parch', 'Age']


# In[76]:


df_t1 = train_frame.copy()
df_te1 = test_frame.copy()


# In[77]:


k_n = KNN(k=7).fit_transform(df_t1[cols]) 


# In[78]:


k_n


# ****As it is seen output is array which can be coverted to dataframe as shown below****

# In[79]:


df1 = pd.DataFrame(KNN(k=5).fit_transform(df_t1[cols]) )


# In[80]:


df1.head()


# ##### Now this can be re-added to orignal data frame and processed as rest

#  

#  

# ### 3. Using The power of Statistics

#  

# ##### Let's think more logically and find certain features which have significant impact on missing data, lets start corelations and find which features are similar to Age

# In[81]:


df_train2 = train_frame.copy()
df_test2 = test_frame.copy()


# In[82]:


df_corr = df_train2.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_corr[df_corr['Feature 1'] == 'Age']


#  

# ##### As evident age is mostly correlated with Pclass, this will help us. This time i will use median age by Pclass feature and fill missing value by this correlation

# In[83]:


age_by_pclass_sex = df_train2.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_train1['Age'].median()))


# In[84]:


df_train2['Age'] = df_train2.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# In[85]:


# Filling the missing values in test frame as well for Age with the medians of Sex and Pclass groups
age_by_pclass_sex = df_test2.groupby(['Sex', 'Pclass']).median()['Age']

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Median age of all passengers: {}'.format(df_test2['Age'].median()))


#  

# ##### Notice the difference of Medians for train and test data

# In[86]:


# Filling the missing values in Age with the medians of Sex and Pclass groups
df_test2['Age'] = df_test2.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))


# ##### This seems much better as we used some statistical understanding to fill missing data

# ##### now check if any pending nan left in age feature

# In[87]:


train_null = df_train2.isnull().sum().sort_values(ascending=False)
train_null


# In[88]:


test_null = df_test2.isnull().sum().sort_values(ascending=False)
test_null


#  

# ##### As Embarked has only few missing values we can use the most frequent occuring strategy here

# In[89]:


#filling embarked with most frquent value
common_value = 'S'
embark = [df_train2, df_test2]

for dataset in embark:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)


#  

# ##### Using Median approach for Fare as it is also have few occuring

# In[90]:


med_fare = df_test2.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
med_fare


# In[91]:


# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
fare = [df_test2]

for dfall in fare:
    dfall['Fare'] = dfall['Fare'].fillna(med_fare)


#  

#  

#  

# ## Feature Creation

#  

# ##### It is a part of Feature Engineering and it is process of creating features that don't already exist in the dataset or creating meaning out of not so important features

# ##### We'll start with Name Feature which doesn't hold much information for a model but a have deeper look and you'll find we can actually extract title from it like 'Mr.' , 'Mrs' etc which can be made useful

#  

# In[92]:


#take title out of name
Feature = [df_train2, df_test2]
min_feature = 10

for dataset in Feature:
     dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        
title = (df_train2['Title'].value_counts() < min_feature)
df_train2['Title'] = df_train2['Title'].apply(lambda x: 'Misc' if title.loc[x] == True else x)

df_train2['Title'].value_counts()


# In[93]:


Survived_female = df_train2[df_train2.Sex=='female'].groupby(['Sex','Title'])['Survived'].mean()
Survived_female


# **See how extracting titles from name has given meaning to this new feature and how much it impact**

# In[94]:


Survived_male = df_train2[df_train2.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()
Survived_male


#  

# ### Some fun with waffle chart

# ##### lets visualize our newly created feature in waffle

# In[95]:


get_ipython().system(' pip install pywaffle')


# In[96]:


from pywaffle import Waffle


# In[97]:


waf = {'Mr':517, 'Miss':182, 'Mrs':125, 'Master':40, 'Misc':27}
waf1 = pd.DataFrame(waf.items(), columns=['Title', 'Value'])


# In[98]:


total_values = sum(waf1['Value'])
category_proportions = [(float(value) / total_values) for value in waf1['Value']]

# print out proportions
for i, proportion in enumerate(category_proportions):
    print (waf1.Title.values[i] + ': ' + str(proportion))


# In[99]:


#add waffle for title
width = 40 # width of chart
height = 10 # height of chart

total_num_tiles = width * height # total number of tiles

# compute the number of tiles for each catagory
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

# print out number of tiles per category
for i, tiles in enumerate(tiles_per_category):
    print (waf1.Title.values[i] + ': ' + str(tiles))


# In[100]:


data = {'Mr': 232, 'Miss': 82, 'Mrs': 56, 'Master': 18, 'Misc': 12}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=10, 
    columns=40,
    values=data, 
    cmap_name="tab10",
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    icons='male', icon_size=18, 
    icon_legend=True,
    figsize=(14, 18)
)


#  

# Encoding the feature manually, we can employ One Hot or Label Encoder for this task but this is done to show it can be mapped manually as well if you are not sure if encoders will perform differently on validations/test/train set.

# In[101]:


#encoding
Feature1 = [df_train2, df_test2]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Misc": 5}

for dataset in Feature1:
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    
df_train2['Title'].value_counts()


# In[102]:


sns.scatterplot(x="Age", y="Title", hue="Survived", size="Sex", palette="Set1",
                     sizes=(20, 200), hue_norm=(0, 7),
                     legend="full", data=df_train2)


#  

# ##### Note: The major distinction between 'pd.cut & pd.qcut' is that 'qcut' will calculate the size of each bin in order to make sure the distribution of data in the bins is equal. In other words, all bins will have (roughly) the same number of observations. While 'cut' will create bins on average and each bin may have different vaulues(unequal samples).

#  

# In[103]:


pd.qcut(df_train2['Age'], 5).value_counts()


# In[104]:


pd.qcut(df_train2['Fare'], 5).value_counts()


# In[105]:


age_bin = [df_train2, df_test2]
for dataset in age_bin:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 25), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 80), 'Age'] = 4


# In[106]:


df_test2['Age'].value_counts()


# In[107]:


Fare_bin = [df_train2, df_test2]
for dataset in Fare_bin:
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 513), 'Fare'] = 4


# In[108]:


df_train2['Fare'].value_counts()


# In[109]:


Fig = sns.FacetGrid(df_train2, col = 'Fare', col_wrap=3)
Fig.map(sns.pointplot, 'Age', 'Survived', 'Sex', ci=95.0, palette = 'deep')
Fig.add_legend()


#   

# ##### See how a combination of multiple feature tell us details about target. We can actually create our own probabalistic model with these ratios.

#  

# In[110]:


## some learning
female_mean = df_train2[df_train2.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','Fare'])['Survived'].mean()
female_mean


# In[111]:


male_mean = df_train2[df_train2.Sex=='male'].groupby(['Sex','Pclass', 'Embarked','Fare'])['Survived'].mean()
male_mean


# In[112]:


# Creating Deck column from the first letter of the Cabin column, For Missing using M
df_train2['Deck'] = df_train2['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_decks = df_train2.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp',
                                                                      'Parch','Fare', 'Embarked', 'Cabin', 'PassengerId', 
                                                                          'Ticket', 'Title']).rename(columns={'Name': 'Count'}).transpose()


# In[113]:


df_decks


# ##### Since Deck T  is negligible assigning it to frequent deck value

# In[114]:


deck = df_train2[df_train2['Deck'] == 'T'].index
df_train2.loc[deck, 'Deck'] = 'M'


#  

# #### waffle visual for deck

#  

# In[115]:


waf2 = {'A':15, 'B':47, 'C':59, 'D':33, 'E':32, 'F':13, 'G':4, 'M':647, 'T':1}
waf2 = pd.DataFrame(waf2.items(), columns=['Deck', 'Value'])


# In[116]:


total_values = sum(waf2['Value'])
category_proportions = [(float(value) / total_values) for value in waf2['Value']]

width = 30 # width of chart
height = 10 # height of chart

total_num_tiles = width * height # total number of tiles

# compute the number of tiles for each catagory
tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

# print out number of tiles per category
for i, tiles in enumerate(tiles_per_category):
    print (waf2.Deck.values[i] + ': ' + str(tiles))


# In[117]:


data = {'A':5, 'B':17, 'C':21, 'D':12, 'E':11, 'F':5, 'G':1, 'M':228, 'T':0}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=10, 
    columns=40,
    values=data, 
    cmap_name="tab10",
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    icons='ship', icon_size=18, 
    icon_legend=True,
    figsize=(14, 18)
)


#  

# ##### Bining Deck feature 

# In[118]:


df_train2['Deck'] = df_train2['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_train2['Deck'] = df_train2['Deck'].replace(['D', 'E'], 'DE')
df_train2['Deck'] = df_train2['Deck'].replace(['F', 'G'], 'FG')

df_train2['Deck'].value_counts()


# Same for Test dataset

# In[119]:


df_test2['Deck'] = df_test2['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

dft_decks = df_test2.groupby(['Deck', 'Pclass']).count().drop(columns=['Sex', 'Age', 'SibSp',
                                                                      'Parch','Fare', 'Embarked', 'Cabin', 'PassengerId', 
                                                                          'Ticket', 'Title']).rename(columns={'Name': 'Count'}).transpose()


# In[120]:


df_test2['Deck'] = df_test2['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_test2['Deck'] = df_test2['Deck'].replace(['D', 'E'], 'DE')
df_test2['Deck'] = df_test2['Deck'].replace(['F', 'G'], 'FG')

df_test2['Deck'].value_counts()


#  

#  

# ##### Note: Here we are going to check correlation or odds of surviving with respect to all features. This will give us a boost in understanding which charcteristics of features have more importance.

# In[121]:


###ckeck corr between difernt class
Target = ['Survived']
corr_cols = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare']

for corr in corr_cols:
    if df_train2[corr].dtype != 'float64' :
        print('Surviving Correlation by:',corr)
        print(df_train2[[corr, Target[0]]].groupby(corr, as_index=False).mean())
        print('-'*20, '\n')#'*20 by -' is to make bottom line of each corr


#  

# ##### Note: Apart from heatmaps which are used further in this notebook we can also sortout corr between different features for feature selections as highly correlated feature sometime exibit no extra information than their counterparts

# In[122]:


corr_train = df_train2.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
corr_train.rename(columns={"level_0": "Attribute 1", "level_1": "Attribute 2", 0: 'Correlation Coefficient'}, inplace=True)
corr_train.drop(corr_train.iloc[1::2].index, inplace=True)
corr_train1 = corr_train.drop(corr_train[corr_train['Correlation Coefficient'] == 1.0].index)


# In[123]:


#Train frame correlations check
corr = corr_train1['Correlation Coefficient'] > 0.3
corr_train1[corr]


#  

# ##### The only downside is we cannot track if the correlation is negative (if you want to do it, remove abs() from the code)

# In[124]:


corr_test = df_test2.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
corr_test.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
corr_test.drop(corr_test.iloc[1::2].index, inplace=True)
corr_test1 = corr_test.drop(corr_test[corr_test['Correlation Coefficient'] == 1.0].index)


# In[125]:


# Test frame correlations check
corr1 = corr_test1['Correlation Coefficient'] > 0.3
corr_test1[corr1]


#  

# ##### Same as above but with heatmaps

# In[126]:


fig, axs = plt.subplots(nrows=2, figsize=(25, 20))

sns.heatmap(df_train2.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap="YlGnBu", annot_kws={'size': 14})
sns.heatmap(df_test2.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)
    
axs[0].set_title('Correlations for Training data features', size=15)
axs[1].set_title('Correlations for Test data features', size=15)


axs[0].set_ylim(6.0, 0)
axs[1].set_ylim(6.0, 0)
plt.show()


#  

#  

#  

# ## Model Training and Evaluation

# ##### We'll use multiple model and see they behave w.r.t to training data and choose the best model to submit for the competetion

# In[127]:


#Models for checking, might not use all

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


#  

# ##### Dropping Features not relevant for performace

#  

# In[128]:


df_train2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_test2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[129]:


df_train2.head()


# In[130]:


df_test2.head()


# In[131]:


from sklearn.preprocessing import LabelEncoder


#  

# ##### LabelEncoder:  
# ##### can turn [dog,cat,dog,mouse,cat] into [1,2,1,3,2], but then the imposed ordinality means that the average of dog and mouse is cat. Still there are algorithms like decision trees and random forests that can work with categorical variables just fine and LabelEncoder can be used to store values using less disk space.
# 
# ##### One-Hot-Encoding:   
# ##### has the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space. The disadvantage is that for high cardinality, the feature space can really blow up quickly and you start fighting with the curse of dimensionality. In these cases, I typically employ one-hot-encoding followed by PCA for dimensionality reduction. I find that the judicious combination of one-hot plus PCA can seldom be beat by other encoding schemes. PCA finds the linear overlap, so will naturally tend to group similar features into the same feature.

#  

# In[132]:


class FeatureEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
    
    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# To avoid encoding feature by feautre we employ above method which is written With help of : https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

# In[133]:


df_tr = df_train2.copy()
df_tr = FeatureEncoder(columns = ['Sex','Embarked', 'Deck']).fit_transform(df_tr)


# In[134]:


df_te = df_test2.copy()
df_te = FeatureEncoder(columns = ['Sex','Embarked', 'Deck']).fit_transform(df_te)


# In[135]:


df_te.head()


# In[136]:


df_tr.dtypes


# In[137]:


cols = ['Survived', 'Pclass', 'Sex', 'Parch', 'Fare', 'Embarked']
sns.pairplot(df_tr[cols], diag_kind="hist", hue = 'Survived', palette="Set1")


# In[138]:


sns.swarmplot(x="Sex", y="Age", hue="Survived", data=df_tr)


# In[139]:


df_te = df_te.astype(int)
df_te.dtypes


# ##### Splitting target and features

# ##### iLoc : iloc returns a Pandas Series when one row is selected, and a Pandas DataFrame when multiple rows are selected, or if any column in full is selected   
# 
# whereas,  
# 
# ##### Loc : loc is label-based, which means that you have to specify rows and columns based on their row and column labels
# 

# In[140]:


x_train = df_tr.iloc[:,1:]
y_train = df_tr.iloc[:,:1]


# In[141]:


x_test = df_te.copy()


# In[142]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


#  

# ##### pipelines are set up with the fit/transform/predict functionality, so that we can fit the whole pipeline to the training data and transform to the test data without having to do it individually for everything

# ##### PowerTransformer provides non-linear transformations in which data is mapped to a normal distribution to stabilize variance and minimize skewness.

#  

# In[143]:


#start with PF of degree 3 then we will check till 5 if efficieny increase else we will leave it
pr=PolynomialFeatures(degree=3)
z = pr.fit_transform(x_train)
z.shape


# ##### Notice how applying Polynomial featuring have increase dimensions/variables to the equation

# ##### One more thing to make note of is the higher in terms of 'degree' you go in Polynomial the the more feature it will create and you might end up breaking your code, so it is better to add a break for this model

# In[144]:


Input=[('scale',PowerTransformer()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LogisticRegression())]
pipe=Pipeline(Input)
pipe.fit(x_train,y_train)
lr_score = round(pipe.score(x_train, y_train) * 100, 2)
print("score:", lr_score, "%")


# In[145]:


decision_tree = DecisionTreeClassifier(criterion='entropy', splitter='random') 
decision_tree.fit(x_train, y_train)  
dt_pred = decision_tree.predict(x_test)  
dt_score = round(decision_tree.score(x_train, y_train) * 100, 2)
print("score:", dt_score, "%")


# In[146]:


knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(x_train, y_train)  
knn_pred = knn.predict(x_test)  
knn_score1 = round(knn.score(x_train, y_train) * 100, 2)
print("score:", knn_score1, "%")


# In[147]:


knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(x_train, y_train)  
knn_pred = knn.predict(x_test)  
knn_score = round(knn.score(x_train, y_train) * 100, 2)
print("score:", knn_score, "%")


# In[148]:


# Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
rf_pred = random_forest.predict(x_test)

rf_score = round(random_forest.score(x_train, y_train) * 100, 2)
print("score:", rf_score, "%")


# In[149]:


from sklearn import svm
from sklearn.svm import SVC

model = svm.SVC(C=1, kernel='poly', random_state=0, gamma = 'auto', degree = 5)
model.fit(x_train, y_train)
svm_pred = model.predict(x_test)
svm_score = round(model.score(x_train, y_train) * 100, 2)
print("score:", svm_score, "%")


# In[150]:


clf = XGBClassifier()
clf.fit(x_train, y_train, eval_metric='auc', verbose=True)
xgb_pred = clf.predict(x_test)
xgb_score = round(clf.score(x_train, y_train) * 100, 2)
print("score:", xgb_score, "%")


# In[151]:


clf1 = XGBClassifier(booster='dart', min_split_loss = 1, max_depth= 7)
clf1.fit(x_train, y_train, eval_metric='auc', verbose=True)
xgb_pred = clf1.predict(x_test)
xgb1_score = round(clf.score(x_train, y_train) * 100, 2)
print("score:", xgb1_score, "%")


# In[152]:


perceptron = Perceptron(max_iter=5)
perceptron.fit(x_train, y_train)

percep_pred = perceptron.predict(x_test)

percep_score = round(perceptron.score(x_train, y_train) * 100, 2)
print("score:", percep_score, "%")


# In[153]:


gaussian = GaussianNB() 
gaussian.fit(x_train, y_train)  
nb_pred = gaussian.predict(x_test)  
nb_score = round(gaussian.score(x_train, y_train) * 100, 2)
print("score:", nb_score, "%")


#  

# ##### Adding all model performance in a single df

#  

# In[154]:


#area for Model Score visual
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN_3 ', 'KNN_5', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'XGB Classifier',
              'Decision Tree'],
    'Score': [svm_score, knn_score, knn_score1, lr_score, 
              rf_score, nb_score, percep_score, xgb_score, dt_score]})
result_df = results.sort_values(by='Score', ascending=False)
#result_df = result_df.set_index('Score')
result_df.head(9)


# In[155]:


ax = sns.barplot(x="Model", y="Score", data=result_df, palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[156]:


#area for feature importance visual
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(9)


# **Understand how a feature creation 'Title' has impacted on our model performance**

# In[157]:


plt.figure(figsize=(10, 5))
sns.barplot(x='importance', y=importances.index, data=importances)

plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.title('Random Forest Classifier', size=15)

plt.show()


#  

# ### Predict probabilty via NaiveBayes and Decision Tree

# ##### predict_proba gives you the probabilities for the target (0 and 1 in our case) in array form. The number of probabilities for each row is equal to the number of categories in target variable

# ##### This is a very powerful function of NaiveBayes and Decision Tree (also employed in Random Forest, XGB and so on). Having a strong understanding of probablities and distribution functions combine with with this may beat some of the top models. If you can somehow work on probablities that are in range of 0.4-0.6 which are the real problem for model and create your function/layer between models say: Naive Bayes -> Your Function -> New features -> XGB/LightGBM might actually perform exceptionally well. 
# ##### If you want to get going with this i recommed read about probablities and distributions functions

#  

# In[158]:


nb_p = gaussian.predict_proba(x_train)


# In[159]:


nb_p[0]


# In[160]:


prob_nb = pd.DataFrame(nb_p, columns=['Prob_NotSurvived', 'Prob_Survived']) 


# In[161]:


prob_nb.head()


# In[162]:


pd.cut(prob_nb['Prob_NotSurvived'], 5).value_counts()


# In[163]:


## lets probability which may be cuasing issue for classifier ie .41 to 0.59
prob_issue = prob_nb[prob_nb['Prob_NotSurvived']>=0.41]


# In[164]:


prob_issue = prob_issue[prob_issue['Prob_NotSurvived']<=0.59]


# In[165]:


prob_issue.head()


# #### see how this will effect model performace, even humans will have issue with such odds :D

# In[166]:


dfp = decision_tree.predict_proba(x_train)


# In[167]:


prob_dt = pd.DataFrame(dfp, columns=['Prob_NotSurvived', 'Prob_Survived'])


# In[168]:


pd.cut(prob_dt['Prob_NotSurvived'], 5).value_counts()


# ##### we can further join index with orignal dataframe to check where on which features our Naive Bayes and decision tree model is having issue and with further analysis we can tune our model better but this will be out of scope for this notebook.

# In[169]:


#then end at df test pred

df_submit = pd.DataFrame(columns=['PassengerId', 'Survived'])
df_submit['PassengerId'] = test_frame['PassengerId']
df_submit['Survived'] = dt_pred
#Since i already submited
#df_submit.to_csv('submit01.csv', header=True, index=False)


# ##### Lastly, there are tonnes of options I didn't discuss here, some of the prominent are Cross Validations, Grid Seacrh for paramters tuning, PCA. But remember that they are also essential part of any ML project.

# ##### Feel free to contact me here or at : https://www.linkedin.com/in/muhammad-saad-31740060/
