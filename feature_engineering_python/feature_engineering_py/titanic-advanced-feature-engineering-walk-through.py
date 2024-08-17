#!/usr/bin/env python
# coding: utf-8

# ## **0. Introduction**
# Welcome to my first kernel! 
# 
# I wrote this kernel **Titanic: Machine Learning from Disaster** as my first data science blog on Kaggle. 
# 
# In my opinion, **better understanding of the data is more important than higher prediction score**. 
# 
# In this kernel, I will focus more on **Exploratory Data Analysis** and **Feature Engineering**, and less on model tuning. I hope this will help to discover the insight of this data set. At the end I am able to use **logistic regression** to achieve score **78.2** with basic feature set, and achieve **0.791** with **target encoding** by **Random Forest** and **0.801** by **SVC**.

# In[1]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


# Load data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train['train_test'] = 1
test['train_test'] = 0
# Concatinate the data. This makes it more convenient to do EDA and feature engineering on the whole data set
all_data = pd.concat([train,test], ignore_index = True)


# ## **1. EDA: First Glance**
# 
# * `PassengerId` is the unique id of the row and it doesn't have any effect on target
# * `Survived` is the target variable we are trying to predict (**0** or **1**):
#     - **1 = Survived**
#     - **0 = Not Survived**
# * `Pclass` (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has **3** unique values (**1**,  **2 **or **3**):
#     - **1 = Upper Class**
#     - **2 = Middle Class**
#     - **3 = Lower Class**
#     **Note**: We should check the relation of Pclass and Survived.
# * `Name` is the passengers's name.
# * `Sex` is the gender of the passengers..
# * `Age` is the age of passengers.
#     - About 20% of Are values are missing.
# * `SibSp` is the total number of the passengers' siblings and spouse.
# * `Parch` is the total number of the passengers' parents and children.
# * `Ticket` is the ticket number of the passenger.
# * `Fare` is the passenger fare.
#     - 1 Fare missing value in test set
# * `Cabin` is the cabin number of the passenger.
#     - About 78% of Cabin values are missing
# * `Embarked` is port of embarkation and it is a categorical feature which has **3** unique values (**C**, **Q** or **S**):
#     - **C = Cherbourg**
#     - **Q = Queenstown**
#     - **S = Southampton**
#     - 2 Embarked missing values in train set. 

# In[3]:


all_data.info() 


# In[4]:


all_data.describe() 


# ## **2. EDA: Feature Distribution and Correlation with the Target Value**
# 
# Figure 1, 2, 3, 4: Plot the distribution of each feature(all_data, and train vs test), and the correlation between each feature and Survived.
# 
# * `Age`: Children have higher chance to survive.
# * `Pclass`: People in Pclass 1 and 2 have higher chance to survive.
# * `Fare`: People who paid more have higher chance to survive.
# * `Sex`: Female have higher chance to survive.
# 
# Figure 5, 6: No significant distribution difference is found between training set and test set.
# 
# Figure 7, 8 ,9: Deeper relation between `Embarked`, `Age`, `Sex` and `Pclass`:
# * `Embarked`: People embarked from C have higher chance to survive, but this seems be relative to Pclass and Sex.
# * `Pclass`:
#     - Pclass 1: Most of women and children survived. Man's survival rate is lower than woman and child, but higher than the one in Pclass 3.
#     - Pclass 2: Most of women and children survived. However man's survival rate is lower than the one in Pclass 3.
#     - Pclass 3: Women and children have higher survival rate than men. Overall the survival rate is lower than Pclass 1 and 2.

# In[5]:


# Figure 1: Distribution of continuous features of all_data
plt.figure(figsize=[14,7])
num_columns = ['Fare','Age','Parch','SibSp']

for i in range(len(num_columns)):
    cur_subplot = 221 + i
    cur_feature = num_columns[i]
    plt.subplot(cur_subplot)
    plt.hist(x = all_data[cur_feature])
#     plt.title(cur_feature + ' Histogram by Survival')
    plt.xlabel(cur_feature)
    plt.ylabel('# of Passengers')
    plt.legend()


# In[6]:


# Figure 2: Distribution of categorical features of all_data
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
cat_columns = ['Sex','Pclass','Embarked']
for i in range(len(cat_columns)):
    curr_feature = cat_columns[i]
    all_data.groupby(curr_feature)['PassengerId'].count().plot(kind = 'bar', stacked = False, ax = axes[i])


# In[7]:


# Figure 3: Distribution / relation of continuous features with Survived
num_columns = ['Fare','Age','Parch','SibSp']
plt.figure(figsize=[14,7])
for i in range(len(num_columns)):
    cur_subplot = 221 + i
    cur_feature = num_columns[i]
    plt.subplot(cur_subplot)
    plt.hist(x = [train[train['Survived']==0][cur_feature], train[train['Survived']==1][cur_feature]], 
             stacked=False,label = ['Dead','Survived'])
    plt.xlabel(cur_feature)
    plt.ylabel('# of Passengers')
    plt.legend()


# In[8]:


# Figure 4: Distribution / relation of categorical features with Survived
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
cat_columns = ['Sex','Pclass','Embarked']
for i in range(len(cat_columns)):
    curr_feature = cat_columns[i]
    train.groupby([curr_feature, 'Survived'])['PassengerId'].count().unstack().plot(kind = 'bar', stacked = False, ax = axes[i])
    axes[i].legend(['Dead','Survived'])


# In[9]:


# Figure 5: Ditribution comparison of continuous features between train and test
plt.figure(figsize=[14,7])
num_columns = ['Fare','Age','Parch','SibSp']
for i in range(len(num_columns)):
    cur_subplot = 221 + i
    cur_feature = num_columns[i]
    plt.subplot(cur_subplot)
    plt.hist(x = [test[cur_feature], train[cur_feature]], stacked=False,label = ['Test','Train'], color = ['skyblue','orange'])
    plt.xlabel(cur_feature)
    plt.ylabel('# of Passengers')
    plt.legend()


# In[10]:


# Figure 6: Ditribution comparison of categorical features between train and test
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
cat_columns = ['Sex','Pclass','Embarked']
for i in range(len(cat_columns)):
    curr_feature = cat_columns[i]
    all_data.groupby([curr_feature, 'train_test'])['PassengerId'].count().unstack().plot(
    kind = 'bar', stacked = False, ax = axes[i], color = ['skyblue','orange'])
    axes[i].legend(['Test','Train'])


# In[11]:


# Figure 7: Deeper relation between Embarked, Pclass and Survived
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
Embarked_features = ['C','Q','S']

for i in range(len(Embarked_features)):
    curr_feature = cat_columns[i]
    temp = train[train.Embarked == Embarked_features[i]]
    temp.groupby(['Pclass', 'Survived'])['PassengerId'].count().unstack().plot(
    kind = 'bar', stacked = False, ax = axes[i], title = ('Embarked_' + Embarked_features[i]))

    axes[i].legend(['Dead','Survived'])


# In[12]:


# Figure 8: Deeper relation between Age, Sex and Pclass
plt.figure(figsize=[14,7])
for i in range(3):
    cur_subplot = 131 + i
    train_temp_pclass_sex = train[(train['Pclass'] == (i+1)) & (train['Sex'] == 'female')]
    plt.subplot(cur_subplot)
    plt.hist(x = [train_temp_pclass_sex[train_temp_pclass_sex['Survived']==0]['Age'], train_temp_pclass_sex[train_temp_pclass_sex['Survived']==1]['Age']], 
             stacked=False,label = ['Dead','Survived'], bins = np.arange(0,80,5))
    plt.xlabel('Pclass' + str(i+1) + '_female')
    plt.ylabel('# of Passengers')
    plt.legend()


# In[13]:


# Figure 9: Deeper relation between Age, Sex and Pclass
plt.figure(figsize=[14,7])
for i in range(3):
    cur_subplot = 131 + i
    cur_feature = num_columns[i]
    train_temp_pclass_sex = train[(train['Pclass'] == (i+1)) & (train['Sex'] == 'male')]
    plt.subplot(cur_subplot)
    plt.hist(x = [train_temp_pclass_sex[train_temp_pclass_sex['Survived']==0]['Age'], train_temp_pclass_sex[train_temp_pclass_sex['Survived']==1]['Age']], 
             stacked=False,label = ['Dead','Survived'], bins = np.arange(0,90,7.5))
    plt.xlabel('Pclass' + str(i+1) + '_male')
    plt.ylabel('# of Passengers')
    plt.legend()


# ## **3. Age: Advanced EDA and Completing**
# 
# `Age`: After checking through the correlation between Age and (Sex, Pclass and Embarked, in Figure 1, 2, 3), no obvious distribution difference was found. This means that most likely the 20% of Age was wiped out on purpose. My **assumption** is that Kaggle did this on **purpose** to imitate missing values in real life. 
# 
# However, there are hidden relations between **Name**, **SibSp**, **Parch** and **Age**:
# * All the people with title Master are under age 14.5.
# * Most of children have tile Master(boy) or Miss(girl). Exceptions 6 out of 109, 5 boys and 1 girl (married).
# * Most of children have Parch > 0. Exception 11 out of 109.
# 
# **Assumption**: People are most likely to be child if they have title **Master** or **Miss**, and **Parch  > 0**.
# 
# This assumption turns out to be true. We can verify it by checking the Ticket, last name, Parch and SibSp. The steps are at the end of section 1.3.1.
# 
# I created a feature **isChild** to show if someone is likely to be the **young, unmarried and travling with parents** in the family. Then the missing **Age** values are filled by the median of corresponding **Pclass**, **Sex** and **isChild**.

# In[14]:


all_data['Age_isMissing'] = 0
all_data.loc[all_data.Age.isnull(), 'Age_isMissing'] = 1


# In[15]:


# Figure 1: Age_isMissing against continuous features, to check if any obvious distribution difference between people with/without age.
num_columns = ['Fare','Parch','SibSp']
plt.figure(figsize=[14,7])
temp = all_data.copy()
for i in range(len(num_columns)):
    cur_subplot = 131 + i
    cur_feature = num_columns[i]
    plt.subplot(cur_subplot)
    plt.hist(x = [temp[temp['Age_isMissing']==0][cur_feature], temp[temp['Age_isMissing']==1][cur_feature]], 
             stacked=False,label = ['Age','NoAge'])
#     plt.title(cur_feature + ' Histogram by Survival')
    plt.xlabel(cur_feature)
    plt.ylabel('# of Passengers')
    plt.legend()


# In[16]:


# Figure 2: Age_isMissing against categorical features, to check if any obvious distribution difference between people with/without age.
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
cat_columns = ['Sex','Pclass','Embarked']
temp = all_data.copy()
for i in range(len(cat_columns)):
    curr_feature = cat_columns[i]
    temp.groupby([curr_feature, 'Age_isMissing'])['PassengerId'].count().unstack().plot(
    kind = 'bar', stacked = False, ax = axes[i])

    axes[i].legend(['Age','NoAge'])


# In[17]:


# Figure 3: Check statistical difference between people with/without age.
all_data.loc[all_data.Age_isMissing == 0, ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'train_test']].describe()


# In[18]:


all_data.loc[all_data.Age_isMissing == 1, ['Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'train_test']].describe()


# In[19]:


# Get the Title from the name
all_data['Name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[20]:


print("Max age of title Master:" + str(all_data[all_data['Name_title'] == 'Master'].Age.max()))


# In[21]:


print('Titles of children:')
all_data[all_data['Age'] < 15].Name_title.value_counts()


# In[22]:


# I am curious about the exceptions.
# By checking Last name and Ticket, passengerId 684 and 687 traveled with their family. They were probably mistaken as Mr instead of Master
# PassengerId7 732 traveled with a 26 years old male and the last name is different. Maybe from one family, and the woman is nanny.
all_data[(all_data['Age'] < 15) & (all_data['Name_title'] == 'Mr')]


# In[23]:


# By checking Ticket and last name, PassengerId 10 is a 14 years old girl, and she is married with a 32.5 years old guy. 
# Not sure if it was legal at the time.
all_data[all_data['Ticket'] == '237736']


# In[24]:


# Check the Parch distribution, when Age < 15
all_data[all_data['Age'] < 15].Parch.value_counts()


# In[25]:


# Make a categorical label for child. It would be convenient later to fill the missing value on Age.
all_data['isChild'] = 0
all_data.loc[all_data['Age'] < 15, 'isChild'] = 1


# In[26]:


all_data.loc[(all_data.Age_isMissing == 1) & ((all_data.Name_title == 'Miss') | 
                                            (all_data.Name_title == 'Master')) & (all_data.Parch > 0),'isChild'] = 1


# ### **Proof** of the **Assumption**: 
# 
# People are most likely to be child if they have title **Master** or **Miss**, and **Parch  > 0**.

# In[27]:


# Print out all the candidates
all_data.loc[(all_data.Age_isMissing == 1) & ((all_data.Name_title == 'Miss') | 
                                            (all_data.Name_title == 'Master')) & (all_data.Parch > 0)]


# For the first candidate, check Ticket 2661:
# 
# The female has title Mrs (married), and has Parch = 2. The two 'Master' both have SibSp = 1 and Parch = 2. 
# 
# We can now assume that ID 1117 is the mom, and ID 66 and 710 are two sons.

# In[28]:


all_data[all_data.Ticket == '2661']


# Another more complex example:
# 
# ID 1234 and 1257 are one male and one female, they are married (same last name), have same SibSp = 1 and Parch = 9. All the rest of family have SibSp = 8 and Parch = 2. So ID 1234 and 1257 are parents and the rest are their children. These children could be older than 15, but the family relationship can be confirmed.
# 
# Since there are not many candidates, I manully went through all of them and confirmed that such family relationship stays true for all candidates.
# 
# For male candidates, title Master indicates that they are children.
# 
# For female candidates, the chance of misassumption on their age is higher, because the title Miss could be used for adults. However, since they are female, they already fall into the other priority group: Sex = female, which have similar survival rate as children in each Pclass, so such possible misassumption will not hurt our model much.

# In[29]:


all_data[all_data.Ticket == 'CA. 2343']


# Fill the missing **Age** values by **Pclass**, **Sex** and **isChild**.

# In[30]:


all_data['Age_isMissing_mean_temp'] = all_data.groupby(['Pclass', 'Sex', 'isChild']).Age.transform('median')
all_data.loc[all_data.Age.isnull(), 'Age'] = all_data.loc[all_data.Age.isnull(), 'Age_isMissing_mean_temp']
all_data.drop('Age_isMissing_mean_temp', axis = 1, inplace = True)


# ## **4. Cabin: Completing and Feature Engineering**
#     
# `Cabin`: Unlike Age, 78% percent of Cabin values are missing, which means it is more likely missing for an actual reason, instead of being wiped on purpose. Typically, grouping values and assigning a value isMissing are good methods to complete such feature. 
# 
# Figure 1,2: by plotting the relation between **Cabin** against **Pclass** we see that:
# * A, B, C, T: 100% Pclass 1
# * D, E: Majority are Pclass 1,
# * F: 62% Pclass 2 and 38% Pclass 3.
# * G: 100% Pclass 3.
# * n: People with missing Cabin value. Majority of n is Pclass 3.
# 
# Figure 3, 4, 5: By combining the relation between **Cabin**, **Pclass**, **Survived**, **Age** and **Sex**, we can explain some uncommon situations:
# * A: Although A has only 1st class, but the majority is **male** and the mean **age** is older, so **survival** rate is **lower**.
# * G: Deck G has only **female** in **3rd** class, which is probably why its **survival** rate is **higher** than the average one of Pclass 3.
# * n: people who have **missing** Cabin values are mostly from 3rd class, and thus the **survival** rate is **lower**. Or the other way around: these 3rd class passengers mostly did not survive, so their Cabin information could not be collected.
# 
# For the rest, the survival rate falls in expectations. Since the uncommon survival rate in Cabin A can be explained by differernt Age and Sex distribution, we can safely group the Cabin_initial values basing on Pclass (Figure 2):
# * A, B, C, T --> 'ABC'
# * D, E --> 'DE'
# * F, G --> 'FG'
# * n --> 'N'

# In[31]:


all_data['Cabin_initial'] = all_data.Cabin.apply(lambda x: str(x)[0])


# In[32]:


# Figure 1: print the total number of passengers regarding Pclass and Cabin deck
all_data.groupby(['Cabin_initial', 'Pclass']).PassengerId.count()


# In[33]:


# Figure 2: left: number of passenger of each Pclass distribution in each Cabin
#           right: percentage of the Pclass distribution in each 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
_ = all_data.groupby(['Cabin_initial', 'Pclass']).PassengerId.count().unstack(fill_value=0).plot.bar(ax = axes[0])

_ = (all_data.groupby(['Cabin_initial', 'Pclass']).PassengerId.count()/ 
     all_data.groupby(['Cabin_initial']).PassengerId.count()).unstack(fill_value=0).plot.bar(ax = axes[1])


# In[34]:


# Figure 3: left: number of survived/not survived passengers in each Cabin
# right: percentage of survived/not survived in each Cabin
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
train_temp = all_data[all_data.train_test == 1] 
_ = train_temp.groupby(['Cabin_initial', 'Survived']).PassengerId.count().unstack(fill_value=0).plot.bar(ax = axes[0])

_ = (train_temp.groupby(['Cabin_initial', 'Survived']).PassengerId.count()/ 
     train_temp.groupby(['Cabin_initial']).PassengerId.count()).unstack(fill_value=0).plot.bar(ax = axes[1])


# In[35]:


all_data['Sex_label'] = 0
all_data.loc[all_data.Sex == 'male', 'Sex_label'] = 1


# In[36]:


# # Figure 4: The higher the Sex_label.mean, the more male than female
all_data.groupby('Cabin_initial').Sex_label.mean()


# In[37]:


# # Figure 5: print the average age of each deck
all_data.groupby('Cabin_initial').Age.mean()


# In[38]:


def group_deck(deck):
    if(deck in ['A','B','C','T']):
        return "ABC"
    elif(deck in ['D','E']):
        return "DE"
    elif (deck in ['F', 'G']):
        return "FG"
    else:
        return "N"

all_data['Cabin_deck'] = all_data['Cabin_initial'].apply(group_deck)


# ## **5. Fare: Feature Engineering and Completing**
# `Fare`: Basing on common sense, we can assume that **Fare** should be related to **Pclass**.
# * Figure 1,2: By plotting the **Fare** within different **Pclass** with bins, we see that the **starting** price for each **Pclass** is:
#      - Plcass 1: 25
#      - Pclass 2: 10
#      - Pclass 3: 7
# 2. Why some 3rd class passenger paid higher fare than 2nd class passenger and even higher than 1st class
#     - During EDA, we saw that one ticket can have more than one person included, so we do **frenquency encoding** on **Ticket** to find the number of occurance of each Ticket (**Tiket_count**), and devide **Fare** by **Tiket_count**, get **Fare_individual**. 
#     
#     Once plot the Fare_individual, we can see that the distribution is **centered** at the **starting price** of each Pclass (Figure 3). For the Fare_individual that is less than the starting price, we can assume that it is due to the family discount (family ticket is cheaper, or ticket of children is cheaper).
#     - Some people are willing to pay more to get better service within the same **Pclass**, we can create a new feature **Fare_differenceFromStart** to indicate if someone paid more or less for the ticket.
#         
#         **Fare_differenceFromStart = Fare_individual - StartingPrice of the Pclass**
#         
# 3. Completing: since the one missing value of Fare has **Ticket_count** = 1, we can just complete it by fill the median of Fare_individual in the corresponding Pclass.
# 
# 3. lastly, in the last plot, we see that no matter the Pclass, people who paid much less than the starting price (paid zero or near zeor) mostly did not survive (survived 1 out of 15).
#     

# In[39]:


# Figure 1: plot the distribution of Fare in each Pclass
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
for i in range(3):
    ax = all_data[all_data.Pclass == (i+1)].Fare.plot.hist(ax = axes[i], title='Plcass = ' + str(i+1))
    ax.set_xlabel("Fare")


# In[40]:


# Figure 2: plot the distribution of Fare in each Pclass with more specific bins
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
_ = all_data[all_data.Pclass == 1].Fare.plot.hist(ax = axes[0], bins = np.arange(0,30,1)).set_xlabel("Fare")
_ = all_data[all_data.Pclass == 2].Fare.plot.hist(ax = axes[1], bins = np.arange(0,20,1)).set_xlabel("Fare")
_ = all_data[all_data.Pclass == 3].Fare.plot.hist(ax = axes[2], bins = np.arange(0,10,1)).set_xlabel("Fare")


# In[41]:


ticket_count = all_data.groupby('Ticket').PassengerId.count()
all_data['Ticket_count'] = all_data['Ticket'].map(ticket_count)
all_data['Fare_individual'] = all_data['Fare']/all_data['Ticket_count']


# In[42]:


# Figure 3: plot the distribution of Fare_individual in each Pclass with more specific bins
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
for i in range(3):
    ax = all_data[all_data.Pclass == (i+1)].Fare_individual.plot.hist(ax = axes[i], title='Plcass = ' + str(i+1))
    ax.set_xlabel("Fare")


# In[43]:


all_data.loc[all_data.Fare_individual.isnull(), 'Fare_individual'] = all_data[all_data['Pclass'] == 3].Fare_individual.median()
all_data.loc[all_data.Fare_individual.isnull(), 'Fare'] = all_data[all_data['Pclass'] == 3].Fare_individual.median()


# In[44]:


all_data['Pclass_startingPrice'] = 0
all_data.loc[all_data.Pclass==1, 'Pclass_startingPrice'] = 25
all_data.loc[all_data.Pclass==2, 'Pclass_startingPrice'] = 10
all_data.loc[all_data.Pclass==3, 'Pclass_startingPrice'] = 7
all_data['Fare_differenceFromStart'] = all_data.Fare_individual - all_data.Pclass_startingPrice


# In[45]:


# Figure 4: plot the Fare_differenceFromStart in each Pclass
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
for i in range(3):
    _ = all_data[all_data.Pclass == (i+1)].Fare_differenceFromStart.plot.hist(ax = axes[i], title='Plcass = ' + str(i+1))
    ax.set_xlabel("Fare")


# In[46]:


# Figure 5: plot Fare_differenceFromStart against Pclass and Survived
num_columns = ['Fare','Parch','SibSp']
plt.figure(figsize=[14,7])
for i in range(len(num_columns)):
    cur_subplot = 131 + i
    cur_feature = num_columns[i]
    plt.subplot(cur_subplot)
    temp = all_data[all_data.Pclass == (i+1)]
    plt.hist(x = [temp[temp['Survived']==0]['Fare_differenceFromStart'], temp[temp['Survived']==1]['Fare_differenceFromStart']], 
             stacked=False, label = ['Dead','Survived'])
    plt.title('Fare_differenceFromStart')
    plt.xlabel('Pclass=' + str(i+1))
    plt.ylabel('# of Passengers')
    plt.legend()


# In[47]:


# The survival rate of people who paid near zero is extremly low, no matter the Pclass
all_data[(all_data.Fare_individual < 2) & (all_data.train_test == 1)].Survived.mean()


# ## **6. Others: Feature Engineering and Completing**
# `Embarked`: There are only 2 missing values in **Embarked**. We can just fill it with the most frequent value within the same **Pclass** and **Sex**.
# 
# `FamilySize`: We can create a new feature **FamilySize = Parch + SibSp + 1**, to indicate the size of the family.
# 
# `GroupSize`: Then we can create another feature **GroupSize = max(Ticket_count, FamilySize)**, to include friends, nanny and family members.

# In[48]:


# The most frequent Embarked value for 1st class female is 'C'. Uncomment to check.
# all_data.groupby(['Pclass', 'Sex', 'Embarked']).size()
all_data.loc[all_data.Embarked.isnull(), 'Embarked'] = 'C'


# In[49]:


all_data['FamilySize'] = all_data.Parch + all_data.SibSp + 1


# In[50]:


all_data['GroupSize'] = all_data[['Ticket_count', 'FamilySize']].max(axis = 1)


# ## **7. First Training**
# 
# Here I only used **Logistic Regression** and **Random Forest**. 
# 
# After fine tunning and poking leader board, both models reached **score 0.784**, which indicates that I probabaly have reached the limit of the current feature set.

# In[51]:


from sklearn.preprocessing import StandardScaler
# Set the feature set
features_to_use = ['Pclass', 'Sex_label', 'Age', 'Embarked', 'train_test', 'Cabin_deck', 
                   'Fare_individual', 'Fare_differenceFromStart', 'GroupSize']
all_features_train = all_data[features_to_use].copy()
# Set the Pclass to str, so we can get one hot encoding of it
all_features_train.Pclass = all_features_train.Pclass.astype(str)
# one hot encoding
all_features_train = pd.get_dummies(all_features_train, columns = ['Pclass', 'Embarked','Cabin_deck'])
# standard scale the numerical features, as logitstic regresstion is sensitive to it
scale = StandardScaler()
to_be_scaled = ['Age', 'Fare_individual', 'Fare_differenceFromStart','GroupSize']
all_features_train[to_be_scaled]= scale.fit_transform(all_features_train[to_be_scaled])


# In[52]:


# get the training and test set
x_train = all_features_train[all_features_train.train_test == 1].copy()
x_train.drop('train_test', axis = 1, inplace = True)
x_test = all_features_train[all_features_train.train_test == 0].copy()
x_test.drop('train_test', axis = 1, inplace = True)

y_train = all_data[all_data.train_test == 1].Survived


# In[53]:


#simple performance reporting function
def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


# In[54]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn import metrics
# Gridsearch the parameters. Uncomment if needed.
# lr = LogisticRegression()
# param_grid = {'max_iter' : [2000],
#               'penalty' : ['l1', 'l2', 'elasticnet'],
#               'C' : np.logspace(-4, 4, 20),
#               'solver' : ['liblinear', 'newton-cg', 'saga']}
                                  
# clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_lr = clf_lr.fit(x_train,y_train)
# clf_performance(best_clf_lr,'Logistic Regression')


# In[55]:


# Logistic regression score: 0.784
lr = LogisticRegression(C = 0.1)
lr.fit(x_train, y_train)
y_pred_test = lr.predict(x_test)

lr_submission = {'PassengerId': all_data[all_data.train_test == 0].PassengerId, 'Survived': y_pred_test.astype(int)}
submission_lr = pd.DataFrame(data=lr_submission)
submission_lr.to_csv('lr_submission_1.csv', index=False)


# In[56]:


from sklearn.ensemble import RandomForestClassifier
# Gridsearch the parameters. Uncomment if needed.
# rf = RandomForestClassifier(random_state = 1)
# param_grid =  {'n_estimators': [100,300,500], 
#                                   'bootstrap': [True,False],
#                                   'max_depth': [3,5,10,20,None],
#                                   'max_features': ['auto','sqrt'],
#                                   'min_samples_leaf': [1,2,4,10],
#                                   'min_samples_split': [2,5,10]}
                                  
# clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_rf_rnd = clf_rf_rnd.fit(x_train,y_train)
# clf_performance(best_clf_rf_rnd,'Random Forest')


# In[57]:


# Randome forest score: 0.784
rf = RandomForestClassifier(random_state = 1, n_estimators = 100, bootstrap = False, max_depth = 5, max_features = 'auto',
                            min_samples_leaf = 1, min_samples_split = 5)
rf.fit(x_train, y_train)
y_pred_test = rf.predict(x_test)

rf_submission = {'PassengerId': all_data[all_data.train_test == 0].PassengerId, 'Survived': y_pred_test.astype(int)}
submission_rf = pd.DataFrame(data=rf_submission)
submission_rf.to_csv('rf_submission_1.csv', index=False)


# ## **9. Target Encoding and Training**
# As both logistic regression and random forest reached the same score 0.784, I probabaly have reached the limit of the feature set.
# Typically, at the later stage of feature engineering, I would add target encoding. 
# 
# Target encoding usually does **NOT** help **understanding** the data better, but more help to **boost** prediction **score**. For this feature set, target encoding Survival rate on **Family** or **Ticket** group would be a good choice, as if some people **survived** in the training set, then the rest of group would have **similar chance** to survive in the test set. 
# 
# When applying target encoding on **Family** group and **Ticket** group, we need to solve the following problems:
# 
# 1. For both Family and Ticket group, some group exist only in **test** set.
# 2. Lots of people are traveling alone. 
#     * For 1 and 2, I created a feature SurvivalRate_NA to indicate if someone is traveling alone, or the group only exist in test set. If so, (Group)SurvivalRate_NA = 1. For people with SurvivalRate_NA = 1, their family survival rate or ticket group survival rate would be 0.
# 3. Do we need to combine Family group survival rate and Ticket group survival rate, or keep them both in feature set?
# 4. If combine, which method? Min, max, mean?
#     * There are 3 cases for Family-Ticket group difference:
#         1. Some people travel with friends. Their family survival rate would be 0.
#         2. Some people travel with family but not on same ticket, so their Ticket survival rate would be closer to 0.
#         3. Majority of families are on the same Ticket.
#     
#     In case A and B, the group bonding is **weaker** than the one of C, because **friends** bonding is **weaker** than **family**, and as ticket could be relative to Cabin posotion, the bonding of family members with different tickets is **weaker** than the one of family of same ticket. By averaging, the result survival rate would be lower in case A and B, but in case C it won't be affected, which represents the actual bonding better.
# 
# **Note**: sometimes people who have the same last name do not come from the same family, but such "mistake" may actually improve the model. One of the weak point of **target encoding** is that it can make the model **overfit** easily. The "mistake" and the averaging of Ticket group and Family group survival rate can help to **reduce overfitting**.

# In[58]:


# get the list of people who are in test set only
train_temp = all_data[all_data.train_test == 1].copy()
test_temp = all_data[all_data.train_test == 0].copy()

ticket_train_list = train_temp.Ticket.value_counts().index.tolist()
ticket_test_list = test_temp.Ticket.value_counts().index.tolist()
ticket_only_test_list = list(set(ticket_test_list) - set(ticket_train_list))


# In[59]:


# if in the list, set Ticket_survivalRate_NA = 1
all_data['Ticket_survivalRate_NA'] = 0
all_data['Ticket_survivalRate_NA'] = all_data['Ticket'].apply(lambda x: 1 if x in ticket_only_test_list else 0)
all_data.loc[all_data.Ticket_count<2, 'Ticket_survivalRate_NA'] = 1


# In[60]:


# map the Ticket group survival rate.
all_data['Ticket_survivalRate'] = all_data['Ticket'].map(all_data[all_data.Ticket_survivalRate_NA == 0].groupby('Ticket').Survived.mean())
# fill NA by zero
all_data['Ticket_survivalRate'] = all_data['Ticket_survivalRate'].fillna(0)


# In[61]:


# get first name as the indicator of family
all_data['FirstName'] = all_data.Name.apply(lambda x: x.split(',')[0].strip())

train_temp = all_data[all_data.train_test == 1].copy()
test_temp = all_data[all_data.train_test == 0].copy()

# do the similar mapping as we did on Ticket
FirstName_train_list = train_temp.FirstName.value_counts().index.tolist()
FirstName_test_list = test_temp.FirstName.value_counts().index.tolist()
FirstName_only_test_list = list(set(FirstName_test_list) - set(FirstName_train_list))

all_data['Family_survivalRate_NA'] = 0
all_data['Family_survivalRate_NA'] = all_data['FirstName'].apply(lambda x: 1 if x in FirstName_only_test_list else 0)
all_data.loc[all_data.FamilySize<2, 'Family_survivalRate_NA'] = 1

all_data['Family_survivalRate'] = all_data['FirstName'].map(all_data[all_data.Family_survivalRate_NA == 0].groupby('FirstName').Survived.mean())
all_data['Family_survivalRate'] = all_data['Family_survivalRate'].fillna(0)


# In[62]:


# Average the survival rate of the family group and ticket group
all_data['SurvivalRate'] = (all_data['Family_survivalRate'] + all_data['Ticket_survivalRate'])/2


# In[63]:


# Just need to simply add the new feature into the train and test set,
x_train['SurvivalRate'] = all_data[all_data.train_test == 1]['SurvivalRate']
x_test['SurvivalRate'] = all_data[all_data.train_test == 0]['SurvivalRate']


# In[64]:


# print out the correlation.
temp = all_data[all_data.train_test == 1][['Survived', 'SurvivalRate','Family_survivalRate','Ticket_survivalRate']]
temp.corr()


# In[65]:


# Gridsearch the parameters. Uncomment if needed.
# lr = LogisticRegression()
# param_grid = {'max_iter' : [2000],
#               'penalty' : ['l1', 'l2', 'elasticnet'],
#               'C' : np.logspace(-4, 4, 20),
#               'solver' : ['liblinear', 'newton-cg', 'saga']}
                                  
# clf_lr = GridSearchCV(lr, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_lr = clf_lr.fit(x_train,y_train))
# clf_performance(best_clf_lr,'Logistic Regression')


# In[66]:


# Score: 0.7894736842105263
lr = LogisticRegression(C = 0.033, max_iter = 2000, penalty = 'l2', solver = 'liblinear')
lr.fit(x_train, y_train)
y_pred_test_lr = lr.predict(x_test)
lr_submission = {'PassengerId': all_data[all_data.train_test == 0].PassengerId, 'Survived': y_pred_test_lr.astype(int)}
submission_lr = pd.DataFrame(data=lr_submission)
submission_lr.to_csv('lr_submission_2.csv', index=False)


# In[67]:


# Gridsearch the parameters. Uncomment if needed.
# rf = RandomForestClassifier(random_state = 1)
# param_grid =  {'n_estimators': [300,500,1000], 
#                                   'bootstrap': [True,False],
#                                   'max_depth': [2,3,4,5,None],
#                                   'max_features': ['auto','sqrt'],
#                                   'min_samples_leaf': [1,2,4],
#                                   'min_samples_split': [1,2,3,5]}
                                  
# clf_rf_rnd = GridSearchCV(rf, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_rf_rnd = clf_rf_rnd.fit(x_train,y_train)
# clf_performance(best_clf_rf_rnd,'Random Forest')


# In[68]:


# Score: 0.7918660287081339
rf = RandomForestClassifier(random_state = 1, bootstrap = True, 
                            max_depth = 3, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 300)
rf.fit(x_train, y_train)
y_pred_test_rf = rf.predict(x_test)
rf_submission = {'PassengerId': all_data[all_data.train_test == 0].PassengerId, 'Survived': y_pred_test_rf.astype(int)}
submission_rf = pd.DataFrame(data=rf_submission)
submission_rf.to_csv('rf_submission_2.csv', index=False)


# In[69]:


from sklearn import svm
# Gridsearch the parameters. Uncomment if needed.
# param_grid = tuned_parameters = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10],
#                                   'C': [.1, 1, 10, 100, 1000]},
#                                  {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
#                                  {'kernel': ['poly'], 'degree' : [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
# clf_svc = GridSearchCV(svc, param_grid = param_grid, cv = 5, verbose = True, n_jobs = -1)
# best_clf_svc = clf_svc.fit(x_train,y_train)
# clf_performance(best_clf_svc,'SVC')


# In[70]:


svc = svm.SVC(random_state=1, C = 1, degree = 2, kernel = 'poly')
svc.fit(x_train, y_train)
y_pred_test_svc = svc.predict(x_test)
svc_submission = {'PassengerId': all_data[all_data.train_test == 0].PassengerId, 'Survived': y_pred_test_svc.astype(int)}
submission_svc = pd.DataFrame(data=svc_submission)
submission_svc.to_csv('svc_submission_1.csv', index=False)


# ## **9. Thank you**
# You made it all the way here!
# 
# After tunning SVC, we reached 80% accuracy. This also means that this data set has more **potential** and can be used for more advanced modeling such as **stacking**. You are more than welcome to try it out yourself.
# 
# Any constructive input is welcome!
# 
# If you find this helpful, please consider up vote :)
