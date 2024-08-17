#!/usr/bin/env python
# coding: utf-8

# ## **Titanic Dataset**

# In[1]:


# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## **Intro**

# #### **Flowchart**

# ![](https://raw.githubusercontent.com/erjonhub/pics/main/workflow.png)

# #### **Data**

# In[2]:


# import titanic data from Dataset folder

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


# check the data by printing the first 5 rows of each dataset

display(train.shape)
display(train.head())
display(test.shape)
display(test.head())


# In[4]:


# check for missing values by printing the percentage of only the columns with missing values

display(round(train.isnull().mean().loc[train.isnull().mean()>0]*100,2))


# ## **Features EDA + ENG**

# #### **PassengerId**

# PassengerId has no value in the prediction of survival. We do not drop it since it is needed for submission in the format:
# 
# 'PassengerID', 'Survived Prediction'
# 
# Instead we make it as index in both, test and train set

# In[5]:


# Make passenger id as index.

train.set_index('PassengerId', inplace=True)
test.set_index('PassengerId', inplace=True)

train.head()


# #### **Pclass**

# In[6]:


# print Pclass value counts

print(train['Pclass'].value_counts())

# plot the Pclass vs Survived

sns.barplot(x='Pclass', y='Survived', data=train)
plt.title('Pclass vs Survived')
plt.show()

# change the Pclass to categorical data

train['Pclass'] = train['Pclass'].astype('category')
test['Pclass'] = test['Pclass'].astype('category')


# There seems to be a correlation between the class and the probability to survive. Considering that the majority of 1st class cabins were closer to the deck and the rescue boats, it is a result to be expected.

# #### **Name**

# The names themselves have no importance. By using sunames clusters can be created to connect the families together but it is quite advanced. The most important thing of the Name feature is the title which provides additional important information about who the person is: Sex, Age, Profession etc.

# In[7]:


# create a new column with the titles from the name column. Use split and strip to get the title

train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# do the same for the test dataset

test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# print the titles and their counts

print(train['Title'].value_counts())

# plot the title vs survived

sns.barplot(x='Title', y='Survived', data=train)
plt.xticks(rotation=90)
plt.title('Title vs Survived')
plt.show()


# This graph provides no information. 
# 
# It can be made better by combining repeated titles (i.e. Ms & Miss), and grouping together the ones that are rare. 

# In[8]:


# combine the titles with same meaning (Mlle = mademoiselle, Mme = madame)

train['TitleCategories'] = train['Title']
train['TitleCategories'] = train['TitleCategories'].replace('Mlle','Miss')
train['TitleCategories'] = train['TitleCategories'].replace(['Mme', 'Ms'], 'Mrs')

test['TitleCategories'] = test['Title']
test['TitleCategories'] = test['TitleCategories'].replace('Mlle','Miss')
test['TitleCategories'] = test['TitleCategories'].replace(['Mme', 'Ms'],'Mrs')

# combine the titles with low counts into one group 'Other'

train['TitleCategories'] = train['TitleCategories'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir', 'Lady', 'the Countess', 'Dona'], 'Other')
test['TitleCategories'] = test['TitleCategories'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir', 'Lady', 'the Countess', 'Dona'], 'Other')


#print the title categories and their counts

print(train['TitleCategories'].value_counts())

# plot the title categories vs survived

sns.barplot(x='TitleCategories', y='Survived', data=train)
plt.xticks(rotation=90)
plt.title('TitleCategories vs Survived')
plt.show()


# This is a very common approach in titanic dataset. But there is a better one. As we already know or at least confidently assume, the genders of all the people in Other, we distribute them in the existing categories Mr & Mrs.

# In[9]:


# combine the titles with the same gender

train['TitleCategoriesMF'] = train['Title']

train['TitleCategoriesMF'] = train['TitleCategoriesMF'].replace(['Mme', 'Ms', 'Lady', 'the Countess', 'Dona'], 'Mrs')
train['TitleCategoriesMF'] = train['TitleCategoriesMF'].replace(['Mlle'], 'Miss')
train['TitleCategoriesMF'] = train['TitleCategoriesMF'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir'], 'Mr')

test['TitleCategoriesMF'] = test['Title']
test['TitleCategoriesMF'] = test['TitleCategoriesMF'].replace(['Mme', 'Ms', 'Lady', 'the Countess', 'Dona'], 'Mrs')
test['TitleCategoriesMF'] = test['TitleCategoriesMF'].replace(['Mlle'], 'Miss')
test['TitleCategoriesMF'] = test['TitleCategoriesMF'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir'], 'Mr')

# print the title categories and their counts

print(train['TitleCategoriesMF'].value_counts())


# plot the title categories vs survived

sns.barplot(x='TitleCategoriesMF', y='Survived', data=train)
plt.xticks(rotation=90)
plt.title('TitleCategoriesMF vs Survived')
plt.show()


# In[10]:


# drop the name and title columns

train.drop(['Name', 'Title'], axis=1, inplace=True)
test.drop(['Name', 'Title'], axis=1, inplace=True)

train.head()


# Now we have 2 columns with title: Title Categories, which has the low count titles groupped as Other, and Title Categories MF which has them distributed by sex

# #### **Sex**

# In[11]:


# print the sex value counts

print(train.Sex.value_counts())

# plot the sex vs survived

sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Sex vs Survived')
plt.show()


# Statistically, if we predict that every female will survive and every male will die, our prediction wont be terrible, but we can do better.
# 
# This feature does not provide any additional information to the model since we can already tell who are males and females from the Title Categories MF.

# #### **Age**

# This is the most interesting column because there are so many approaches to impute the values and categorize them. 
# 
# k-NN and "mean for the same TitleCategoryMF" were considered but finally the later was selected.

# In[12]:


# print the average age per title category mf

print(train.groupby('TitleCategoriesMF')['Age'].mean())

# plot the average age per title category mf

sns.barplot(x='TitleCategoriesMF', y='Age', data=train)
plt.xticks(rotation=90)
plt.title('TitleCategoriesMF vs Age')
plt.show()


# Mr: Adult Man, 18? - infinity
# 
# Mrs: Married Woman, assumed an adult, 18? - infinity
# 
# Master: Male child, 0 - 18?
# 
# Miss: Unmarried female, 0 - infinity 
# 
# A thing to consider is the category Miss. Unlike males who have Master kids and Mr. for aduls, Mrs. and Miss distinguish only between married and unmarried. The Miss category has an average of 21.8, but many among them are children. One way to tell is checking the average age in TitleCategories according to their Parch. If a miss has no Parents or Children with her, is most probably an adult, no child would travel alone without parents.

# In[13]:


# print the average age of Miss for each Parch value

print(train.loc[train['TitleCategoriesMF']=='Miss'].groupby('Parch')['Age'].mean())

# plot the average age of Miss for each Parch value

sns.barplot(x='Parch', y='Age', data=train[train['TitleCategoriesMF']=='Miss'])
plt.title('Parch vs Age for Miss')
plt.show()


# We see that the Miss that have 0 parents or children, are most probably adults. The ones with 1 and 2 can be mixed. So now we impute the missing values with the mean age of the following categories:
# 
# 1. Average age of Mr.
# 2. Average age of Mrs.
# 3. Average age of Master
# 4. Average age of Miss with 0 parch
# 5. Average age of Miss with 1 parch
# 6. Average age of Miss with 2 parch

# In[14]:


# imput the ages for the missing values with the mean age for each title category mf == Mr, or Mrs, or Master

train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Mr'), 'Age'] = train.loc[train['TitleCategoriesMF'] == 'Mr']['Age'].mean()
train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Mrs'), 'Age'] = train.loc[train['TitleCategoriesMF'] == 'Mrs']['Age'].mean()
train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Master'), 'Age'] = train.loc[train['TitleCategoriesMF'] == 'Master']['Age'].mean()

# input the ages for the missing values with the mean age for each title category mf == Miss, and Parch == 0, 1, 2

train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 0), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 0)]['Age'].mean()
train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 1), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 1)]['Age'].mean()
train.loc[(train['Age'].isnull()) & (train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 2), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 2)]['Age'].mean()

# we do the same for the test set BUT using the mean values from the train set

test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Mr'), 'Age'] = train[train['TitleCategoriesMF'] == 'Mr']['Age'].mean()
test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Mrs'), 'Age'] = train[train['TitleCategoriesMF'] == 'Mrs']['Age'].mean()
test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Master'), 'Age'] = train[train['TitleCategoriesMF'] == 'Master']['Age'].mean()

test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Miss') & (test['Parch'] == 0), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 0)]['Age'].mean()
test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Miss') & (test['Parch'] == 1), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 1)]['Age'].mean()
test.loc[(test['Age'].isnull()) & (test['TitleCategoriesMF'] == 'Miss') & (test['Parch'] == 2), 'Age'] = train[(train['TitleCategoriesMF'] == 'Miss') & (train['Parch'] == 2)]['Age'].mean()


# ![](https://raw.githubusercontent.com/erjonhub/pics/main/imputation.png)

# In[15]:


# check for missing values in the Age column

print(train.Age.isnull().sum())


# The chances of survival can be different by age. The relationship is not linear though, a 15 year old does not have more or less chance than a 16 year old. Dividing by age group or by categories seems sensible.

# In[16]:


# add a new column with age group baby, child, adult, senior

train['AgeGroup'] = 'Adult'

train.loc[(train['Age']<1), 'AgeGroup'] = 'Baby'
train.loc[(train['Age']>=1) & (train['Age']<12), 'AgeGroup'] = 'Child'
train.loc[(train['Age']>=60), 'AgeGroup'] = 'Senior'

test['AgeGroup'] = 'Adult'

test.loc[(test['Age']<1), 'AgeGroup'] = 'Baby'
test.loc[(test['Age']>=1) & (test['Age']<12), 'AgeGroup'] = 'Child'
test.loc[(test['Age']>=60), 'AgeGroup'] = 'Senior'

# print the age group value counts

print(train['AgeGroup'].value_counts())


# plot the age group vs survived

sns.barplot(x='AgeGroup', y='Survived', data=train)
plt.title('AgeGroup vs Survived')
plt.show()


# The distribution seems to be quite imbalanced which brings the question if it is a good idea to make this categorization

# #### **SibSp and Parch**

# The most common method is to add siblings or spouse and parents or children together to see the size of the families and their chances for survival

# In[17]:


# add a column family size

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# print the family size value counts and sort by FamilySize

print(train['FamilySize'].value_counts().sort_index())

# plot the family size vs survived

sns.barplot(x='FamilySize', y='Survived', data=train)
plt.title('FamilySize vs Survived')
plt.show()


# This does not provide much information and is highly imbalanced. Categorizing according to the family size will provide better information:
# 
# 1. Alone
# 2. Small family = 2-4
# 3. Big family > 4

# In[18]:


# divide the family size into 3 categories

train['FamilySizeCategory'] = 'Small'

train.loc[(train['FamilySize']==1), 'FamilySizeCategory'] = 'Alone'
train.loc[(train['FamilySize']>=5), 'FamilySizeCategory'] = 'Big'

# do the same in the test dataset

test['FamilySizeCategory'] = 'Small'

test.loc[(test['FamilySize']==1), 'FamilySizeCategory'] = 'Alone'
test.loc[(test['FamilySize']>=5), 'FamilySizeCategory'] = 'Big'


# print the family size category value counts

print(train['FamilySizeCategory'].value_counts())


# plot the family size category vs survived

sns.barplot(x='FamilySizeCategory', y='Survived', data=train)
plt.title('FamilySizeCategory vs Survived')
plt.show()


# Also quite imbalanced result. Another approach is to create a column if the person is alone or is with family.

# In[19]:


# create a column is alone

train['IsAlone'] = 0
train.loc[(train['FamilySize']==1), 'IsAlone'] = 1

# do the same in the test dataset

test['IsAlone'] = 0
test.loc[(test['FamilySize']==1), 'IsAlone'] = 1

# print the is alone value counts

print(train['IsAlone'].value_counts())


# plot the is alone vs survived

sns.barplot(x='IsAlone', y='Survived', data=train)
plt.title('IsAlone vs Survived')
plt.show()


# #### **Ticket**

# The ticket feature provides no information. While there are approaches that extract the number and / letter and try to make some sense out of it, that idea is not applied in this notebook.

# In[20]:


#  drop the ticket column

train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)

train.head()


# #### **Fare**

# A histogram will show the distribution of the cost of the ticket.

# In[21]:


# plot a histogram of the fare column

plt.hist(train['Fare'], bins=20)
plt.title('Fare')
plt.show()


# Not much to deduct apart from the fact that some people really went out of their way to get a ticket. We can divide in 3 categories: low fare which is probably all the 3rd class, medium fare which is "normal" people, and high fare which only the richest people can afford. 

# In[22]:


# divide the fare into 3 categories and plot it vs survived

train['FareCategory'] = 'Low'

train.loc[(train['Fare']>10) & (train['Fare']<=100), 'FareCategory'] = 'Medium'
train.loc[(train['Fare']>100), 'FareCategory'] = 'High'

test['FareCategory'] = 'Low'

test.loc[(test['Fare']>10) & (test['Fare']<=100), 'FareCategory'] = 'Medium'
test.loc[(test['Fare']>100), 'FareCategory'] = 'High'


print(train.FareCategory.value_counts())


sns.barplot(x='FareCategory', y='Survived', data=train)
plt.title('FareCategory vs Survived')
plt.show()


# While the distribution is imbalanced, there is a clear difference in price ticket and chance of survival. It can be highly correlated with Pclass as well as age. To be treaded carefully

# #### **Cabin**

# There are 3 possible approaches for this column that has 70% of missing values. 
# 
# 1. Extract the first letter to see if there is a relationship between the cabin category and the survival rate. 
# 2. Create a column "has cabin" to see if there is a difference in survival between those with and without cabin number.
# 3. Don't use the column (or drop it) as it gives us no information.

# In[23]:


# extract the first letter of the cabin and save it in a new column

train['CabinCategory'] = train['Cabin'].str[0]
test['CabinCategory'] = test['Cabin'].str[0]

# print the cabin category value counts

print(train['CabinCategory'].value_counts())

# replace cabin T with cabin C

train.loc[(train['CabinCategory']=='T'), 'CabinCategory'] = 'C'

# plot the cabin category vs survived

sns.barplot(x='CabinCategory', y='Survived', data=train)
plt.title('CabinCategory vs Survived')
plt.show()


# There seems no relationship whatsoever. Lets proceed with option 2

# In[24]:


# create a new column for the passengers that have a cabin

train['HasCabin'] = 0
train.loc[train['Cabin'].notnull(), 'HasCabin'] = 1

test['HasCabin'] = 0
test.loc[test['Cabin'].notnull(), 'HasCabin'] = 1

# print the has cabin value counts

print(train['HasCabin'].value_counts())

# plot the has cabin vs survived

sns.barplot(x='HasCabin', y='Survived', data=train)
plt.title('HasCabin vs Survived')
plt.show()


# This looks more promissing as there is a clear difference in survival probability between people with a cabin number and without. Seems random but the explanation might be that people that survived gave more interviews, their records were better kept, and their cabin number ended up in the data.

# #### **Embarked**

# There is no obvious reason why the location of embarkment can affect the prediction.

# In[25]:


# for the embarked column we fill the missing values with the most frequent value

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# print the embarked value counts

print(train['Embarked'].value_counts())

# plot the embarked vs survived

sns.barplot(x='Embarked', y='Survived', data=train)
plt.title('Embarked vs Survived')
plt.show()


# No clear trend is visible. One possibility is that the embarkment can help strengthen the prediction that families stuck together. 

# ## **Model**

# In[26]:


train.head()


# Separate the labels from the training set

# In[27]:


# remove Survived and save it as y_train

y_train = train['Survived']
train.drop('Survived', axis=1, inplace=True)


# The most difficult task is to select the features that contribute to the performance of the model. The parameters were tuned with Randomized Search and of all the models tried, best performance came from Random Forest.

# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter grid for random search
param_grid = {
    'n_estimators': randint(100, 1000), # Number of trees to train
    'max_depth': randint(1, 10) # Maximum depth of the tree (number of splits)
}

# Perform random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=1),
    param_distributions=param_grid,
    n_iter=10,  # Number of parameter settings that are sampled during random search
    cv=5,  # Cross-validation folds
    random_state=1
)

selected_features = ['Pclass', 'Age', 'Embarked', 'TitleCategoriesMF', 'IsAlone', 'FareCategory', 'Parch', 'FareCategory', 'HasCabin']

X_train = pd.get_dummies(train[selected_features], drop_first=True)
X_test = pd.get_dummies(test[selected_features], drop_first=True) 

random_search.fit(X_train, y_train)

# Get the best model from random search
best_model = random_search.best_estimator_

predictions = best_model.predict(X_test)

output = pd.DataFrame({'PassengerId': X_test.index, 'Survived': predictions})

# output to csv

output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# ## Final comments

# I tried a lot of combination not included in this notebook. 7 different models, grid search, knn imputation, various combinations of selected_features etc. I kept what worked best, including having FareCategory twice which was done initially by mistake but it increased the prediction score in kaggle. I don't have an explanation for that, but would love to hear one.
