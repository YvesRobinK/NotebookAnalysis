#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The sinking of the Titanic is one of the most infamous shipwrecks in history.
# 
# On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# 
# While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# 
# In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).
# 
# 
# # Data Description
# 
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.
# 
# 
# Variable Notes
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# 
# Sibling = brother, sister, stepbrother, stepsister
# 
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# 
# Parent = mother, father
# 
# Child = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 
# Variable	Definition	Key
# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex	
# Age	Age in years	
# sibsp	# of siblings / spouses aboard the Titanic	
# parch	# of parents / children aboard the Titanic	
# ticket	Ticket number	
# fare	Passenger fare	
# cabin	Cabin number	
# embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:





# # Loading Libraries

# In[1]:


import pandas as pd #For Data Analysis
import numpy as np # For numerical Computations
import matplotlib.pyplot as plt # For Visualization
import seaborn as sns # For Visualization
import re # For Capturing words
plt.style.use('fivethirtyeight')


# # Loading Datasets

# In[2]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# # Data Information and data types

# In[3]:


# Checking the Datatypes of the columns
train_df.info()


# In[4]:


test_df.info()


# # EDA of training data

# ## 1. Renaming columns

# In[5]:


train_df.head()


# In[6]:


# Converting the column names to lower_case and replacing some headings
train_df.columns = [x.lower() for x in train_df.columns]
train_df.columns


# In[7]:


# Doing the same for test_df
test_df.columns = [x.lower() for x in test_df.columns]


# In[8]:


train_df.rename(columns={
            "passengerid":"passenger_id",
            "pclass":"passenger_class",
            "sibsp":"sibling_spouse",
            "parch":"parent_children"
        }, inplace=True)


# In[9]:


# Doing the same for train df
test_df.rename(columns={
            "passengerid":"passenger_id",
            "pclass":"passenger_class",
            "sibsp":"sibling_spouse",
            "parch":"parent_children"
        }, inplace=True)


# In[10]:


train_df.head()


# ## 2. Finding Missing Values

# In[11]:


train_df.isnull().sum()


# In[12]:


train_df.isnull().sum().plot(kind='bar')


# In[13]:


# Pictorial
sns.heatmap(train_df.isnull(), cbar=False)


# #### Inference (finding missing values): 
# From the above plots we can see, that Columns Age, Cabin, Embarked are missing some values. Going further we can see how we can rectify them

# ## 3. Checking Each Column values and Feature Engineering

# ### 1. Passenger Id

# In[14]:


train_df[["passenger_id"]]


# In[15]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "passenger_id")
plt.show()


# #### Inference: 
# Since passenger_id column is an index column, and it has no relation with survival rate, we can ignore the passenger_id column

# ### 2. Passenger Class

# In[16]:


train_df.passenger_class.unique()


# #### Distribution of passenger class

# In[17]:


train_df.passenger_class.value_counts().plot(kind='pie')


# In[18]:


train_df.passenger_class.value_counts().plot(kind='bar')


# #### Comparison of P Class with survival

# In[19]:


plt.figure(figsize=(12,5))
sns.countplot("passenger_class", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("P Class", fontsize=18)
plt.title("P Class Distribution ", fontsize=20)


# From the above plot, we can see Passengers in Class 1 and 2 were having good survival rate than Passenger in class 3

# In[20]:


train_df.groupby("passenger_class").survived.value_counts(normalize=True).sort_index()


# #### Inference (passenger_class):
# From the above normalized data, we can understand that people in class 1 had 63 % survival rate and class 2 is having 47 % survival rate. 

# ### 3. Name column

# In[21]:


train_df.name.unique()


# Name column is also like Passenger Id column, its just an index for a person.
# 
# But, from this name values, we can see different salutations are given for persons based on their age/royalty/officer.
# 
# We can collect these data first (Feature Engineering), and will analyse whether its supporting survival rate

# In[22]:


# Collecting the salutation words
train_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())


# In[23]:


# Assign these values to a new column
train_df["salutation"] = train_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Doing the same for Tst data
test_df["salutation"] = test_df.name.apply(lambda x: x.split(",")[1].split(".")[0].strip())


# In[24]:


train_df.salutation.value_counts()


# In[25]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df)
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()


# From the above graph, we can see that we have more categories in salutation, we can try to reduce it by mapping
# ( Since some categories are having only a single value, eg: Lady, Sir, Col)

# In[26]:


# Creating Categories
salutation_dict = {
"Capt": "0",
"Col": "0",
"Major": "0",
"Dr": "0",
"Rev": "0",
"Jonkheer": "1",
"Don": "1",
"Sir" :  "1",
"the Countess":"1",
"Dona": "1",
"Lady" : "1",
"Mme": "2",
"Ms": "2",
"Mrs" : "2",
"Mlle":  "3",
"Miss" : "3",
"Mr" :   "4",
"Master": "5"
}


# In[27]:


train_df['salutation'] = train_df.salutation.map(salutation_dict)

# Doing the same for test data
test_df['salutation'] = test_df.salutation.map(salutation_dict)


# In[28]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df)
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()


# Now we have reduced the categories

# In[29]:


train_df.salutation = train_df.salutation.astype('float64')

# Doing the same for Test
test_df.salutation = test_df.salutation.astype('float64')


# #### Distribution of Salutation

# In[30]:


train_df.salutation.value_counts().plot(kind='pie')


# #### Comparison with survival rate

# In[31]:


#plotting countplot for salutations
plt.figure(figsize=(16,5))
sns.countplot(x='salutation', data=train_df, hue="survived")
plt.xlabel("Salutation", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()


# From the above plot we can see that, people in category 1, 2, 3, 5 were having mpre survival rate than other classess.
# 
# People in Category 
# 1. Jonkheer, Don, Sir, Countess, Dona, Lady
# 2. Mme, Ms, Mrs
# 3. Mlle, Miss
# 5. Master
# 
# From this we can see, Ladies and Childrens are having more survival rate. 

# In[32]:


train_df.groupby("salutation").survived.value_counts(normalize=True).sort_index()


# In[33]:


train_df.groupby("salutation").survived.value_counts(normalize=True).sort_index().unstack()


# So we can try to create an another column "sal_sur" based on the above findings

# In[34]:


sal_sur_index = train_df[(train_df.salutation.isin([1.0, 2.0, 3.0, 5.0]))].index

sal_sur_index_test = test_df[(test_df.salutation.isin([1.0, 2.0, 3.0, 5.0]))].index


# In[35]:


train_df["sal_sur"] = 0
train_df.loc[sal_sur_index, "sal_sur"] = 1

# Doing the same for test data

test_df["sal_sur"] = 0
test_df.loc[sal_sur_index_test, "sal_sur"] = 1


# In[36]:


train_df[["sal_sur", "survived"]].head()


# In[37]:


#plotting countplot for salutations Survived
plt.figure(figsize=(16,5))
sns.countplot(x='sal_sur', data=train_df, hue="survived")
plt.xlabel("Salutation Survived", fontsize=16) 
plt.ylabel("Count", fontsize=16)
plt.title("Salutation Survived Count", fontsize=20) 
plt.xticks(rotation=45)
plt.show()


# #### Inference (Name):
# From the above findings, we can see "salutations" plays a good role in survival_rate

# ### 4. Sex

# In[38]:


# Unique values of gender
train_df.sex.unique()


# In[39]:


# Percentage of people
train_df.sex.value_counts(normalize=True)


# #### Distribution of Gender

# In[40]:


train_df.sex.value_counts().plot(kind='pie')


# In[41]:


train_df.sex.value_counts().plot(kind='bar')


# In[42]:


plt.figure(figsize=(12,5))
sns.countplot("sex", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Sex", fontsize=18)
plt.title("Sex Distribution ", fontsize=20)


# In[43]:


train_df.groupby("sex").survived.value_counts(normalize=True).sort_index()


# In[44]:


train_df[['sex', 'survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# From the above findings, we can see 74% of females are having higher survival rate than males. 

# #### Inference (Sex):
# From the above we can see females are having more survival rate than men

# ### 5. Age

# As we discussed at the top Age is having some null values. So first we can concentrate on filling the missing values first. 

# In[45]:


train_df.age.isnull().sum()


# ### Types of filling in the data:
# 
# 1. Filling the missing data with the mean or median value if it’s a numerical variable.
# 2. Filling the missing data with mode if it’s a categorical value.
# 3. Filling the numerical value with 0 or -999, or some other number that will not occur in the data. This can be done so that the machine can recognize that the data is not real or is different.
# 4. Filling the categorical value with a new type for the missing values.
# 
# ### Process for filling missing values in Age
# 1. Since its a continous values, we can use either mean or median - Here we can use <b>Median</b>
# 2. We already having a gouping in name - like Mr, Master, Don. 
# 3. So we can group the individual name category and fill the median value to the missing items
# 

# In[46]:


# Creating a Group based on Sex, Passenger, Salutation
age_group = train_df.groupby(["sex","passenger_class","salutation"])["age"]

# Doing the same for test data
age_group_test = test_df.groupby(["sex","passenger_class","salutation"])["age"]


# In[47]:


# Median of each grop
age_group.median()


# In[48]:


age_group.transform('median')


# In[49]:


# Now we can apply the missing values
train_df.loc[train_df.age.isnull(), 'age'] = age_group.transform('median')

# Doing the same for test data
test_df.loc[test_df.age.isnull(), 'age'] = age_group_test.transform('median')


# In[50]:


# For Checking purpose
train_df.age.isnull().sum()


# Now all the missing values are been filled. 

# #### Distribution of Age

# In[51]:


plt.figure(figsize=(12,5))
sns.histplot(x='age', data=train_df)
plt.title("Total Distribuition and density by Age")
plt.xlabel("Age")
plt.show()


# In[52]:


plt.figure(figsize=(12,5))
sns.histplot(x='age', data=train_df, hue="survived")
plt.title("Distribuition and density by Age and Survival")
plt.xlabel("Age")
plt.show()


# In[53]:


plt.figure(figsize=(12,5))
sns.distplot(x=train_df.age, bins=25)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")
plt.show()


# In[54]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "age")
plt.show()


# From the above we can see that, people in the range of 18 to 40 were having good survival rate. 
# 
# Now we can see, how gender is affecting this values

# #### Male Comparisons

# In[55]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
g = sns.FacetGrid(male_df, col='survived',size=5)
g = g.map(sns.distplot, "age")
plt.show()


# #### Female Comparisons

# In[56]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
g = sns.FacetGrid(female_df, col='survived',size=5)
g = g.map(sns.distplot, "age")
plt.show()


# For Males: With age range 20 to 40 is having a good survival rate. 
# 
# For Females: With age range 18 to 40 is having a goog survival rate.
# 
# Now we can try to use this feature to build a new one 

# In[57]:


age_index = train_df[((train_df.sex=='male') & ( (train_df.age >= 20) & (train_df.age <= 40) )) |
         ((train_df.sex=='female') & ( (train_df.age >= 18) & (train_df.age <= 40) ))   
        ].index


# In[58]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1


# In[59]:


train_df[["age_sur","survived"]]


# In[60]:


train_df.groupby("age_sur").survived.value_counts()


# In[61]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1
plt.figure(figsize=(12,5))
sns.countplot("age_sur", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[62]:


plt.figure(figsize=(12,5))
g = sns.FacetGrid(train_df, col='survived',size=5)
g = g.map(sns.distplot, "age_sur")
plt.show()


# Our newly created features is not good

# In[63]:


print(sorted(train_df.age.unique()))


# In[64]:


# We can try to create categories


# In[65]:


interval = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150) 
cats = list(range(len(interval)-1))

# Applying the pd.cut and using the parameters that we created 
train_df["age_category"] = pd.cut(train_df.age, interval, labels=cats)

# Printing the new Category
train_df["age_category"].head()


# Doing the same for Test Data

# Applying the pd.cut and using the parameters that we created 
test_df["age_category"] = pd.cut(test_df.age, interval, labels=cats)

# Printing the new Category
test_df["age_category"].head()



# In[66]:


train_df.age_category.unique()


# In[67]:


plt.figure(figsize=(12,5))
sns.countplot("age_category", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# #### Comparison with Gender

# In[68]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("age_category", data=male_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist for Male", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[69]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("age_category", data=female_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist for Female", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# From the above two graphs, we can see that Males in age category 0 is having higher survival rate. 
# 
# For Female, in the range 0-6 is having higher survival rate. 
# 
# So now we can update, the new age_survival column based on our findings. 

# In[70]:


age_index = train_df[((train_df.sex=='male') & ( train_df.age_category.isin([0]) )) |
         ((train_df.sex=='female') & ( (train_df.age_category.isin([0,1,2,3,4,5,6])) ))   
        ].index

# Doing the same for Test Data 

age_index_test = test_df[((test_df.sex=='male') & ( test_df.age_category.isin([0]) )) |
         ((test_df.sex=='female') & ( (test_df.age_category.isin([0,1,2,3,4,5,6])) ))   
        ].index


# In[71]:


age_index


# In[72]:


train_df["age_sur"] = 0
train_df.loc[age_index, "age_sur"] = 1

# Doing the same for Test Data 
test_df["age_sur"] = 0
test_df.loc[age_index_test, "age_sur"] = 1

plt.figure(figsize=(12,5))
sns.countplot("age_sur", data=train_df, hue="survived", palette="hls")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Age Dist", fontsize=18)
plt.title("Age Dist ", fontsize=20)


# In[ ]:





# #### Inference(Age):
# From this we can know that, age_sur with category 1 is having higher survival rate

# ### 6. Sibling Spouse

# In[73]:


train_df.sibling_spouse.unique()


# In[74]:


train_df.groupby("sibling_spouse").survived.value_counts(normalize=True).sort_index()


# In[75]:


plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=train_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Sibling Dist", fontsize=18)
plt.title("Sibling Dist ", fontsize=20)


# #### Comparison with Male

# In[76]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=male_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Male Sibling Dist", fontsize=18)
plt.title("Male Sibling Dist ", fontsize=20)


# #### Comparison with Female

# In[77]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("sibling_spouse", data=female_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Female Sibling Dist", fontsize=18)
plt.title("Female Sibling Dist ", fontsize=20)


# #### Inference: 
# On Whole : From the above plot, we can see people with 1, 2 siblings have higher survival rate
# 
# With Male: As usual the survival rate is low for all categories
# 
# With Female : As usual the survival rate is high for all categories.

# ### 7. Parent Children

# In[78]:


train_df.parent_children.unique()


# In[79]:


train_df.groupby("parent_children").survived.value_counts(normalize=True).sort_index()


# In[80]:


plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=train_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("parent_children Dist", fontsize=18)
plt.title("parent_children Dist ", fontsize=20)


# #### Comparison with Male

# In[81]:


male_df = train_df[train_df.sex=='male']
plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=male_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Male parent_children Dist", fontsize=18)
plt.title("Male parent_children Dist ", fontsize=20)


# In[82]:


train_df[train_df.sex=='male'].groupby("parent_children").survived.value_counts(normalize=True).sort_index()


# #### Comparison with Female

# In[83]:


female_df = train_df[train_df.sex=='female']
plt.figure(figsize=(12,5))
sns.countplot("parent_children", data=female_df, hue="survived")
plt.ylabel("Count", fontsize=18)
plt.xlabel("Female parent_children Dist", fontsize=18)
plt.title("Female parent_children Dist ", fontsize=20)


# #### Inference: 
# 
# With Male: As usual the survival rate is low for all categories
# 
# With Female : As usual the survival rate is high for all categories.

# #### We can see something is interesting right, For both sibling_spouse and parent_children, with gender as female its showing higher survival rate (in categories of 0, 1, 2, 3). 
# 
# With this information we can create a new column, like "pc_ss_sur"

# In[84]:


ps_ss_sur_index = train_df[
    (train_df["sex"] == 'female') &
    (
        (train_df["sibling_spouse"].isin([0, 1, 2, 3])) | 
        (train_df["parent_children"].isin([0, 1, 2, 3]))
    )
].index


# Doing the same for Test Data

ps_ss_sur_index_test = test_df[
    (test_df["sex"] == 'female') &
    (
        (test_df["sibling_spouse"].isin([0, 1, 2, 3])) | 
        (test_df["parent_children"].isin([0, 1, 2, 3]))
    )
].index



# In[85]:


train_df["ps_ss_sur"] = 0
train_df.loc[ps_ss_sur_index, "ps_ss_sur"] = 1


# In[86]:


# Doing the same for test data

test_df["ps_ss_sur"] = 0
test_df.loc[ps_ss_sur_index_test, "ps_ss_sur"] = 1


# In[87]:


train_df.ps_ss_sur.corr(train_df.survived)


# ### 8. Fare

# In[88]:


print(sorted(train_df.fare.unique()))


# In[89]:


plt.figure(figsize=(12,5))
sns.set_theme(style="whitegrid")
sns.boxplot(x="survived", y="fare", data=train_df, palette="Set3")
plt.title("Survived Fare Rate")


# In[90]:


train_df.head()


# In[91]:


train_df.fare.fillna(train_df.fare.mean(), inplace=True)

# Doing the same for test data
test_df.fare.fillna(test_df.fare.mean(), inplace=True)


# ### 9. Cabin

# In[92]:


train_df.cabin.isnull().sum()


# We can see that cabin is having more of null values. So instead of filling the missing values, we can create a new feature. 

# In[93]:


cabin_null_index = train_df[train_df.cabin.isnull()].index

# Doing the same for Cabin
cabin_null_index_test = test_df[test_df.cabin.isnull()].index


# In[94]:


train_df["is_cabin"] = 1
train_df.loc[cabin_null_index, "is_cabin"] = 0

# Doing the same for test
test_df["is_cabin"] = 1
test_df.loc[cabin_null_index_test, "is_cabin"] = 0


# In[95]:


train_df.is_cabin.corr(train_df.survived)


# ### 10. Embarked

# As we know before, embarked is having some missing values. We can try to fix that first. 

# In[96]:


train_df.embarked.isnull().sum()


# In[97]:


train_df.embarked.unique()


# #### Distribution of Embarked

# In[98]:


train_df.embarked.value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend()


# In[99]:


sns.displot(x=train_df.embarked)
plt.title("Distribuition of embarked values")
plt.show()


# #### Since the embarked is a categorical values, we can apply mode. So here we will be filling 'S' for all nan

# In[100]:


train_df.embarked.fillna("S", inplace=True)


# In[101]:


# Doing the same for test data
test_df.embarked.fillna("S", inplace=True)


# #### Survival rate based on each embarkment

# In[102]:


sns.barplot(x='embarked', y='survived', data=train_df)


# #### Inference:
# from the above graph we can know that, people who are boarded in C were survived more

# ## Feature Scaling and Feature Selection

# In[103]:


train_df.head()


# In[104]:


train_df.columns


# In[105]:


train_df.sex.replace({"male":0, "female":1}, inplace=True)


# In[106]:


# Doing the same for test data
test_df.sex.replace({"male":0, "female":1}, inplace=True)


# In[107]:


subset = train_df[["passenger_class", "survived","sal_sur", "age_sur", "age_category", "ps_ss_sur", "is_cabin", "sex", "fare"]]
subset_test = test_df[["passenger_class", "sal_sur", "age_sur", "age_category", "ps_ss_sur", "is_cabin", "sex", "fare"]]


# In[108]:


subset


# In[109]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(subset.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# Fare, Sex, is_Cabin, Ps_ss_sur, age_sur, sal_sur were having higher correlation with survival 

# ## Training Testing Set Preparation 

# In[110]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[111]:


X = subset.drop("survived", axis=1)
Y = train_df["survived"]


# In[112]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=10)


# ## Modelling

# In[113]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
lr_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', lr_training_data_accuracy)
lr_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', lr_testing_data_accuracy)


# In[114]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
rf_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', rf_training_data_accuracy)
rf_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', rf_testing_data_accuracy)


# In[115]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
nb_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', nb_training_data_accuracy)
nb_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', nb_testing_data_accuracy)


# In[116]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
knn_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', knn_training_data_accuracy)
knn_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', knn_testing_data_accuracy)


# In[117]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=9000, tol=1e-3))

model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
sgd_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', sgd_training_data_accuracy)
sgd_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', sgd_testing_data_accuracy)


# In[118]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=10)
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
sgd_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', sgd_training_data_accuracy)
sgd_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', sgd_testing_data_accuracy)


# By Comparing, we came to know that Logistic Regression is doing good

# In[119]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
X_test_prediction = model.predict(X_test)
lr_training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', lr_training_data_accuracy)
lr_testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of testing data : ', lr_testing_data_accuracy)


# ## Submissions

# In[120]:


results = model.predict(subset_test)


# In[121]:


sub_df = pd.read_csv("../input/titanic/gender_submission.csv")
sub_df["Survived"] = results
sub_df.to_csv("final_submission_2.csv", index=False)


# Please <b>upvote</b> if you liked it !!!

# # I will be updating this notebook in future as well

# In[ ]:




