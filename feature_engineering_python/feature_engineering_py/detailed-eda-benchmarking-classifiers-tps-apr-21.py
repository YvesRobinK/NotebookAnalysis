#!/usr/bin/env python
# coding: utf-8

# # Approach:-
# My advice to beginners will be to follow the following workflow for approaching any tabular data for competition:-  
# 1. Read the Problem Statement and understand the requirements carefully.
# 2. READ THE PROBLEM STATEMENT AND REQUIREMENTS AGAIN.
# 3. Look carefully at the Data and do EDA (It will help largely during feature engineering and Inference). It is worth the time.
# 4. Decide the problem category (Classification/Regression).
# 5. Spit data into Kfolds before doing Feature engineering. Because it's very easy to leak data/contaminate the validation set during Feature Engineering. Then the validation set is no longer representative of real data.
# 6. Build Basic model and record the performance.
# 7. Try to improve the performance by Feature engineering, better encoding, synthetic feature creation etc.
# 8. Do feature selection to only use the important/relevant features.
# 9. If that saturates or starts to drop, go back and try using other models and see whichone works better.
# 10. Tune the hyperparameters for the models which seem to work good as per your observations.
# 11. Select some of the best models from previous step and do an ensemble/stacking-blending.
# 12. Submit to the leaderboard and gaze the difference in CV and LB. If it is huge, then most likely there was some overfitting and leakage across folds. Try to identify and rectify the same.
# 13. When done, resubmit and you should see a close result.
# 14. Not satisfied with result/Trying to get better rank? Head over to the Discussion Forum and read through interesting discussions and see what others are trying to do. If happy, implement them and gradually start improving your scores and skills.
# 
# **Happy Kaggling!**
# 
# # Why this Competition?
# This competition provides an unique oppertunity for Data Science beginners to participate in a Hackathon style challenge. It also provides the unique oppertunities for beginners to get their hands dirty and indulge is practical application of ML and do one of the basic tasks machine learning algorithms are capable of doing:- **Classification**.  
# 
# This competition has the right mix to Catergorical and Numerical features we might expect in a practical problem and this helps us know how to leverage both of thhem in conjugation for a Classification task.
# 
# # Problem Statement
# The goal of this competition is to provide a fun, and approachable for anyone, tabular dataset. These competition will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition.  
# 
# The dataset is used for this competition is synthetic but based on a real dataset (in this case, the actual Titanic data!) and generated using a CTGAN.  
# 
# So we are sort of dealing with a variation of actual real-world data and here as Data Scientists are expected to predict the Binary Classification based on these features.
# 
# ## Data Description:-
# There are 6 categorical features and 4 continuous features in the dataset. The label binary class.  
# 
# `survival` : Survival  
# `pclass` : Ticket class  
# `sex` : Sex  
# `Age` : Age in years  
# `sibsp` : # of siblings / spouses aboard the Titanic  
# `parch` : # of parents / children aboard the Titanic  
# `ticket` : Ticket number  
# `fare` : Passenger fare  
# `cabin` : Cabin number  
# `embarked` : Port of Embarkation
# 
# ## Expected Outcome:-
# * Build a model to predict if a person survived this tragic incident or not, given the information above.
# * Grading Metric: **Accuracy**
# 
# ## Problem Category:-
# From the data and objective its is evident that this is a **Binary Classification Problem** in the **Tabular Data** format.
# 
# So without further ado, let's now start with some basic imports to take us through this journey:-

# In[1]:


# Asthetics
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os
from scipy.optimize import fmin as scip_fmin

# Visialisation
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")

# Machine Learning

# Utils
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn import preprocessing
import category_encoders as ce

#Feature Selection
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile, VarianceThreshold

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier


# In[2]:


data_dir = '../input/tabular-playground-series-apr-2021'

train_file_path = os.path.join(data_dir, 'train.csv')
test_file_path = os.path.join(data_dir, 'test.csv')
sample_sub_file_path = os.path.join(data_dir, 'sample_submission.csv')

print(f'Train file: {train_file_path}')
print(f'Train file: {test_file_path}')
print(f'Train file: {sample_sub_file_path}')


# In[3]:


RANDOM_SEED = 42


# In[4]:


def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# In[5]:


seed_everything()


# # Nulls Imputation
# Let's have a basic look around the data we have at hand first

# In[6]:


train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
sub_df = pd.read_csv(sample_sub_file_path)


# Let's see what columns we have in the training data.

# In[7]:


train_df.columns


# Looking at some sample rows...

# In[8]:


train_df.sample(10)


# Let's look at some basic descriptive analysis...

# In[9]:


train_df.describe().T


# In[10]:


test_df.describe().T


# Also, we can explore the cardinality of each feature as shown below...

# In[11]:


train_df.nunique()


# In[12]:


test_df.nunique()


# As we can see from the descriptive analysis before, there are some features which have NaN values both in train and test set. So let's plot what is the situation for all the features...

# In[13]:


nulls_train = np.sum(train_df.isnull())
nullcols_train = nulls_train.loc[(nulls_train != 0)].sort_values(ascending=False)

barplot_dim = (15, 8)
ax = plt.subplots(figsize=barplot_dim)
sns.barplot(x=nullcols_train.index, y=nullcols_train)
plt.ylabel("Null Count", size=20);
plt.xlabel("Feature Name", size=20);
plt.show()
print('There are', len(nullcols_train), 'features with missing values in the training data.')
print(f'Columns containing nulls are: {list(nullcols_train.index)}')


# In[14]:


nulls_test = np.sum(test_df.isnull())
nullcols_test = nulls_test.loc[(nulls_test != 0)].sort_values(ascending=False)

barplot_dim = (15, 8)
ax = plt.subplots(figsize=barplot_dim)
sns.barplot(x=nullcols_test.index, y=nullcols_test)
plt.ylabel("Null Count", size=20);
plt.xlabel("Feature Name", size=20);
plt.show()
print('There are', len(nullcols_test), 'features with missing values in the test data.')
print(f'Columns containing nulls are: {list(nullcols_test.index)}')


# If we convert those numbers into percentage, we can have a basic idea on how much gaps require filling in each feature or whether it is significant to fill or not...

# In[15]:


nulls_train = np.sum(train_df.isnull())
nullcols_train = nulls_train.loc[(nulls_train != 0)].sort_values(ascending=False)
nullcols_train = nullcols_train.apply(lambda x: 100*x/train_df.shape[0])

barplot_dim = (15, 8)
ax = plt.subplots(figsize=barplot_dim)
sns.barplot(x=nullcols_train.index, y=nullcols_train)
plt.ylabel("Null %", size=20);
plt.xlabel("Feature Name", size=20);
plt.show()


# In[16]:


nulls_test = np.sum(test_df.isnull())
nullcols_test = nulls_test.loc[(nulls_test != 0)].sort_values(ascending=False)
nullcols_test = nullcols_test.apply(lambda x: 100*x/test_df.shape[0])

barplot_dim = (15, 8)
ax = plt.subplots(figsize=barplot_dim)
sns.barplot(x=nullcols_test.index, y=nullcols_test)
plt.ylabel("Null %", size=20);
plt.xlabel("Feature Name", size=20);
plt.show()


# ## 1. Cabin
# We can see that the Cabin feature has quite a bit of missing values as compared to others (~ 68%).  
# Also cabin is a categorical feature thus inputing this will be a tricky one. But to keep things simple, let's assume that the people who do not have a cabin number, did not have Cabins and we will treat this as a new category and also create one more feature that captures that those values are synthetically imputed...

# In[17]:


train_df['Cabin_Available'] = train_df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['Cabin_Available'] = test_df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[18]:


train_df['Cabin'].fillna('No Cabin', inplace=True)
test_df['Cabin'].fillna('No Cabin', inplace=True)


# ## 2. Ticket
# Around 5% of the values in ticket number is missing.  
# Generally speaking, ticket number should be very unique to each passenger. But it might also be the case that people booked tickets for their friends and family and multiple people have boarded using the same ticket number. Thus there will not be any stright forward method to inpute this value as well. This similar to Cabin we will treat each missing class as a new category and also create another feature that tells where the values were imputed synthetically...

# In[19]:


train_df['Ticket_Available'] = train_df['Ticket'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['Ticket_Available'] = test_df['Ticket'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[20]:


train_df['Ticket'].fillna('Missing Ticket', inplace=True)
test_df['Ticket'].fillna('Missing Ticket', inplace=True)


# ## 3. Age
# Age is a continuous feature and almost 3% of those values are missing.  
# To keep things simple we can impute the age by the mode of all the available ages. And like earlier we will track the rows which had missing values.

# In[21]:


train_df['Age_Available'] = train_df['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['Age_Available'] = test_df['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[22]:


train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)


# ## 4. Embarked
# Embarked is a categorical feature and almost 0.2% of the values are missing.  
# To keep things simple we can impute embarked with the most popular embarked station. And we will store the missing information in another feature column.

# In[23]:


train_df['Embarked_Available'] = train_df['Embarked'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['Embarked_Available'] = test_df['Embarked'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[24]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)


# ## 5. Fare
# Fare is a continuos feature and close to 0.1% values are missing.  
# It will be a fair asumption to think that fare will be hugely related to ticket class. So we can impute the missing values with the average fare of that class...

# In[25]:


train_df['Fare_Available'] = train_df['Fare'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_df['Fare_Available'] = test_df['Fare'].apply(lambda x: 0 if pd.isnull(x) else 1)


# In[26]:


train_df['Fare'] = train_df.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.mean()))
test_df['Fare'] = test_df.groupby('Pclass').Fare.transform(lambda x: x.fillna(x.mean()))


# Okay, now that we have addressed all the null columns, let's have a sanity check to ensure that we did not miss anything...

# In[27]:


np.sum(train_df.isnull())


# In[28]:


np.sum(test_df.isnull())


# Great, it looks like we successfully managed to impute all the null values. Now moving on to EDA...
# 
# # EDA
# 
# ## 1. Class Imbalance

# In[29]:


ax = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
sns.countplot(x='Survived', data=train_df);
plt.ylabel("No. of Observations", size=20);
plt.xlabel("Survived", size=20);


# Ok, so this is a fairly balanced dataset. We have to keep this in mind while developing our models later.  
# Now let's move on to understanding each individual feature through EDA and some visualizations... Before that l;et's create some helper functions that eases the EDA plotting process.

# In[30]:


def plot_cat_distribution(cat, train_df=train_df):
    ax = plt.subplots(figsize=(12, 6))
    sns.set_style('whitegrid')
    sns.countplot(x=cat, data=train_df);
    plt.ylabel('No. of Observations', size=20);
    plt.xlabel(cat+' Count', size=20);
    plt.show()
    
def plot_cat_response(cat, train_df=train_df):
    ax = plt.subplots(figsize=(8, 5))
    sns.set_style('whitegrid')
    sns.countplot(x=cat, hue='Survived', data=train_df);
    plt.show()


# ## 2. PassengerID
# The data description says it is an unique identifier for each row and does not hold any significant information regarding the passenger. Thus we can plan to drop this feature later while modelling.  
# 
# ## 3. Pclass
# This feature signifies the ticket class of the passenger. It is available as a categorical feature with 3 distinct values:- 1st Class, 2nd Class and 3rd Class. Let's look at their distribution and response towards Survival...

# In[31]:


plot_cat_distribution('Pclass')


# In[32]:


plot_cat_response('Pclass')


# **Observations:-**  
# * Maximum number of passengers were travelling on class 3 tickets.
# * It seems that Class 1 and 2 passengers had a much higher chance of survival than Class 3 passengers.
# 
# ## 4. Name
# Name is again something which is most likely unique to the passenger and less/no effect on the probability of the survival of the person. Any correlation might be coincidental and not necessarily part of the signal. So we can drop this feature later.
# 
# ## 5. Sex
# As obvious as it is, this feature tells us the gender of the passangers. This one might be very significant because in case of evacuation weomen and children are given higher preference. Let's see if the data agrees...

# In[33]:


plot_cat_distribution('Sex')


# In[34]:


plot_cat_response('Sex')


# In[35]:


g = sns.catplot(x="Pclass", hue="Sex", col="Survived",
                data=train_df, kind="count");
g.fig.set_size_inches(10,5)


# **Observation:-**  
# * There were more male passengers on the ship ac comapred to females. But there is not a very high bias towards any specific gender.
# * As expected, the probability of survival as a female is mugh higher as comapred to male passengers.
# * Females from Class-1 had the highest chances of survival followed by Class-2 and then Class-3.
# 
# ## 6. Age
# Age is a continuous variable and one which can again be of high importance. Because in case of evacuation children and old people would be given preference. Let's check what the data says...

# In[36]:


g = sns.displot(data=train_df, x="Age", hue="Survived", kind="kde");
g.fig.set_size_inches(10,5)


# There are more adults on the ship as compared to childeren

# In[37]:


g = sns.catplot(x='Survived', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# The average age of surviving passengers is more than deceased ones, stating that infact children and older people weere evacuated first from the ship. And just because the population of older people is larger than children, the average age is on the higher side.

# In[38]:


g = sns.catplot(x='Sex', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# On an average, the age of females on the boat was higher than males.

# In[39]:


g = sns.catplot(x='Sex', y='Age', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# Older females had the highest chances of survival followed by older men.

# In[40]:


g = sns.catplot(x='Pclass', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# The average age in Class-1 is highest, followed by Class-2 and then Class-3.  
# There are children in all 3 classes of the ship.  
# Class-2 has the oldest preson, followed by Class-3 and then Class-1.

# In[41]:


g = sns.catplot(x='Pclass', y='Age', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# **Observations:-**  
# * There are more adults on the ship as compared to childeren.
# * The average age of surviving passengers is more than deceased ones, stating that infact children and older people weere evacuated first from the ship. And just because the population of older people is larger than children, the average age is on the higher side.
# * On an average, the age of females on the boat was higher than males.
# * Older females had the highest chances of survival followed by older men.
# * The average age in Class-1 is highest, followed by Class-2 and then Class-3.
# * There are children in all 3 classes of the ship.
# * Class-2 has the oldest preson, followed by Class-3 and then Class-1.
# * Older people from Class-1 and Class-2 had better chances of survival as compared to their younger counterparts. But in Class-3, actually younger people had better chances of survival than older people.
# 
# ## 7. SibSp
# SibSp is a integer feature which specifies the number of siblings / spouses of the passenger aboard the Titanic. This might have an influence on survival because usually a human will also try to keep their relatives safe in case of a disaster.  
# We can treat this a ordinal categorical features because there are only finite number of Siblings/Spouse a passenger could have aboard the ship.

# In[42]:


plot_cat_distribution('SibSp')


# In[43]:


plot_cat_response('SibSp')


# In[44]:


# Response Rate
v = train_df.groupby('SibSp').Survived.value_counts().unstack()
v['Ratio'] = v[1]/v[0]
v.reset_index(inplace=True)


# In[45]:


v['Ratio'].mean()


# In[46]:


ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
sns.barplot(x='SibSp', y='Ratio', data=v.sort_values(by=['Ratio'], ascending=False));


# In[47]:


g = sns.catplot(x='SibSp', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[48]:


g = sns.catplot(x='SibSp', y='Age', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(12,5)


# In[49]:


g = sns.catplot(x='Pclass', y='Age', hue='SibSp', kind='box', data=train_df);
g.fig.set_size_inches(12,5)


# **Observations:-**  
# * Most of the people were travelling without Siblings/Spouse.
# * Survival rate of passengers having 2 Sibling/Spouse were the highest.
# * Older and very young passengers were generally travelling without any Siblings/Spouse.
# * Older people travelling with < 3 siblings/spouse had higher chances of survival.
# 
# ## 8. Parch
# Parch is a integer feature which specifies the number of parents / children of the passenger aboard the Titanic. This might have an influence on survival because usually any parent will try to keep their children safe in case of a disaster. Also while evacuating any children, a parent will be accompany them, so that also adds to the equation of survival...  
# We can treat this a ordinal categorical features because there are only finite number of parents / children a passenger could have aboard the ship.

# In[50]:


plot_cat_distribution('Parch')


# In[51]:


plot_cat_response('Parch')


# In[52]:


# Response Rate
v = train_df.groupby('Parch').Survived.value_counts().unstack()
v['Ratio'] = v[1]/v[0]
v.reset_index(inplace=True)


# In[53]:


ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")
sns.barplot(x='Parch', y='Ratio', data=v.sort_values(by=['Ratio'], ascending=False));


# In[54]:


g = sns.catplot(x='Parch', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[55]:


g = sns.catplot(x='Parch', y='Age', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(12,5)


# In[56]:


g = sns.catplot(x='Pclass', y='Age', hue='Parch', kind='box', data=train_df);
g.fig.set_size_inches(14,5)


# **Observations:-**  
# * Most passengers were travelling without any parents/children.
# * Passengers having 1/3/5 parents/children are more likely to survive than others.
# * Age has very little effect on the number of parents/children accompanying a passenger.
# 
# ## 9. Fare
# Fare is a continuous feature that specifies the total fare paid by the passenger for their place on the boat. It should be highly correlated with Class and total number of people on the same ticket.

# In[57]:


g = sns.displot(data=train_df, x="Fare", hue="Survived", kind="kde");
g.fig.set_size_inches(10,5)


# The distribution has a very long tail, we probably need to tranform this while feature engineering.

# In[58]:


g = sns.catplot(x='Survived', y='Fare', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[59]:


g = sns.catplot(x='Sex', y='Fare', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[60]:


g = sns.catplot(x='Sex', y='Fare', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[61]:


g = sns.catplot(x='Pclass', y='Fare', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[62]:


g = sns.catplot(x='Pclass', y='Fare', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# **Observations:-**  
# * Probability distribution of Fare has a very long tail. Needs to be transformed during feature enginering.
# * People who paid higher fare, also had a higher chance of survival.
# * Average fare paid by Male passengers was lower than Female passengers.
# * Higher fare males had a higher chance of survival than lower fare males. But in case of females there was not a big difference.
# * Naturally, Class-1 had the highest fare followed by Class-2 and then Class-3.
# * Higher fare peoples among Class-1 and Class-2 had higher chances of curvival as compared to their lower fare counterparts. However in Class-3, Fare did not matter much in terms of survival probability.
# 
# ## 10. Embarked
# This feature defines the port of Embarkation. We have 3 options in this category:- (C = Cherbourg, Q = Queenstown, S = Southampton).  
# This feature might be important because it will also affect the order of filling of compartments. And due to compartment locations it might have an effect on the survival probability. This combined with Class would act like a spatial proxy for the location of the passenger inside the ship.

# In[63]:


plot_cat_distribution('Embarked')


# In[64]:


g = sns.catplot(x='Embarked', y='Fare', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[65]:


g = sns.catplot(x="Embarked", hue="Pclass",
                data=train_df, kind="count");
g.fig.set_size_inches(10,5)


# In[66]:


plot_cat_response('Embarked')


# In[67]:


g = sns.catplot(x='Embarked', y='Fare', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[68]:


g = sns.catplot(x='Embarked', y='Age', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[69]:


g = sns.catplot(x='Embarked', y='Age', hue='Survived', kind='box', data=train_df);
g.fig.set_size_inches(10,5)


# In[70]:


g = sns.catplot(x="Embarked", hue="Pclass", col="Survived",
                data=train_df, kind="count");
g.fig.set_size_inches(10,5)


# **Observations:-**  
# * Most of the passengers embarked the ship from Southampton.
# * The average fare for people boarding from Southampton is lowest while Cherbourg and Queenstown are similar.
# * As a result, most of the people bording from Southampton are Class-3 passengers while the majority of Cherbourg and Queenstown passengers belong to Class-1 or 2.
# * Similarly, the survival probability of passengers from Cherbourg and Queenstown is higher as comapred to passengers from Southampton.
# * People embarking from Southampton who paid a higher fare had a better chance of survival than people with lower fares. But the same can not be said about the people from Cherbourg and Queenstown.
# * The Average age of people embarking from Southampton is lowest followed by Cherbourg and Queenstown.
# 
# # KFold Splits  
# Before we move on to feature engineering, it is always a good idea to perform cross validation splits. In that way, we will not risk any data leakage and would be more certain of the validation set being aptly represenative of the real world unknown data.

# In[71]:


NUM_SPLITS = 5

train_df["kfold"] = -1
train_df = train_df.sample(frac=1).reset_index(drop=True)
y = train_df.Survived.values
kf = StratifiedKFold(n_splits=NUM_SPLITS)
for f, (t_, v_) in enumerate(kf.split(X=train_df, y=y)):
    train_df.loc[v_, 'kfold'] = f
    
train_df.head()


# # Feature Engineering

# In[72]:


train_df.nunique()


# Some features can be dropped from the dataset, like:-
# 1. We can drop the 'PassengerId' and 'Name' features because their cardinality is so high comapred to the data that, we necessarily will not learn anything from these features. Also The features practically is not relevant to any physical parameter affecting the survival probability.
# 2. The features 'Ticket' and 'Cabin' have some potentially useful information like Cabin number etc, but after retrival of those information these columns can be dropped because their cardinality is just too high for the data.

# In[73]:


drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']


# As found out during EDA, the Fare feature has a very long tail. So as decided, we will take a log transform of the same to shorten the tail and potentially eliminate any ill effects of huge outliers.

# In[74]:


train_df['Fare'] = np.log(train_df['Fare'])
test_df['Fare'] = np.log(test_df['Fare'])


# We can create a synthetic feature which clubs Class-1 and Class-2 together but kepps class-3 separate, because as seen from EDA, the response from Class-1 and 2 were very similar.

# In[75]:


train_df['Clubbed_Class'] = train_df['Pclass'].apply(lambda x: 'Economy' if x == 3 else 'Highend')
test_df['Clubbed_Class'] = test_df['Pclass'].apply(lambda x: 'Economy' if x == 3 else 'Highend')


# We can concatenate Pclass and Sex feature to form a synthetic feature which signifies something like "Class-1 Male Passenger".

# In[76]:


train_df['Pclass_Sex'] = train_df['Pclass'].astype(str) + '_' + train_df['Sex']
test_df['Pclass_Sex'] = test_df['Pclass'].astype(str) + '_' + test_df['Sex']


# Passenger ages and Fares can be bucketed and dictretized into groups.

# In[77]:


train_df['Age_Bin_10'] = pd.cut(train_df['Age'], bins=10, labels=False)
train_df['Age_Bin_50'] = pd.cut(train_df['Age'], bins=50, labels=False)
test_df['Age_Bin_10'] = pd.cut(test_df['Age'], bins=10, labels=False)
test_df['Age_Bin_50'] = pd.cut(test_df['Age'], bins=50, labels=False)


# In[78]:


train_df['Fare_Bin_10'] = pd.cut(train_df['Fare'], bins=10, labels=False)
train_df['Fare_Bin_50'] = pd.cut(train_df['Fare'], bins=50, labels=False)
test_df['Fare_Bin_10'] = pd.cut(test_df['Fare'], bins=10, labels=False)
test_df['Fare_Bin_50'] = pd.cut(test_df['Fare'], bins=50, labels=False)


# We can concatenate Age bins with sex features indicating something like "Old Female".

# In[79]:


train_df['Age_Bin_10_Sex'] = train_df['Age_Bin_10'].astype(str) + '_' + train_df['Sex']
train_df['Age_Bin_50_Sex'] = train_df['Age_Bin_50'].astype(str) + '_' + train_df['Sex']
test_df['Age_Bin_10_Sex'] = test_df['Age_Bin_10'].astype(str) + '_' + test_df['Sex']
test_df['Age_Bin_50_Sex'] = test_df['Age_Bin_50'].astype(str) + '_' + test_df['Sex']


# We can also concatenate Age bins with Class features indicating something like "Young Class-3 passenger".

# In[80]:


train_df['Age_Bin_10_Class'] = train_df['Age_Bin_10'].astype(str) + '_' + train_df['Pclass'].astype(str)
train_df['Age_Bin_50_Class'] = train_df['Age_Bin_50'].astype(str) + '_' + train_df['Pclass'].astype(str)
test_df['Age_Bin_10_Class'] = test_df['Age_Bin_10'].astype(str) + '_' + test_df['Pclass'].astype(str)
test_df['Age_Bin_50_Class'] = test_df['Age_Bin_50'].astype(str) + '_' + test_df['Pclass'].astype(str)


# We can extract various ticket related information from the ticket number...

# In[81]:


train_df['Ticket_len'] = train_df['Ticket'].str.len()
test_df['Ticket_len'] = test_df['Ticket'].str.len()

train_df['Ticket_type'] = train_df['Ticket'].str.replace('\d+', '')
train_df['Ticket_type'] = train_df['Ticket_type'].apply(lambda x: 'Num' if x=='' else x[:3])
test_df['Ticket_type'] = test_df['Ticket'].str[:3].replace('\d+', '')
test_df['Ticket_type'] = test_df['Ticket_type'].apply(lambda x: 'Num' if x=='' else x[:3])


# We can extract various cabin related information from the cabin number...

# In[82]:


train_df['Cabin_type'] = train_df['Cabin'].map(lambda x: str(x)[0])
test_df['Cabin_type'] = test_df['Cabin'].map(lambda x: str(x)[0])


# We can create a synthetic variable stating the family size on board the Titanic for rach passenger...

# In[83]:


train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1


# Similarly we can detemine if the person was travelling alone...

# In[84]:


train_df['IsAlone'] = train_df['FamilySize'] <= 1
test_df['IsAlone'] = test_df['FamilySize'] <= 1


# We can create a feature that takes in if the age is in int or fraction...

# In[85]:


train_df['IsAgeInt'] = (train_df['Age'] == train_df['Age'].map(np.floor)).astype(int)
test_df['IsAgeInt'] = (test_df['Age'] == test_df['Age'].map(np.floor)).astype(int)


# We can create a feature that captures the name and surname separately...

# In[86]:


train_df['Last_Name'] = train_df['Name'].map(lambda x: x.split(', ')[0])
train_df['First_Name'] = train_df['Name'].map(lambda x: x.split(', ')[1])

test_df['Last_Name'] = test_df['Name'].map(lambda x: x.split(', ')[0])
test_df['First_Name'] = test_df['Name'].map(lambda x: x.split(', ')[1])


# In[87]:


name_cols = ['Last_Name', 'First_Name']
train_df[name_cols] = train_df[name_cols].apply(lambda x: x.mask(x.map(x.value_counts())< (0.001*train_df.shape[0]), 'RARE'))

for col in name_cols:
    vals = list(train_df[col].unique())
    test_df[col] = test_df[col].apply(lambda x: 'RARE' if x not in vals else x)


# In[88]:


train_df.drop(drop_columns, axis=1, inplace=True)


# # Feature Encoding

# In[89]:


target_col = ['Survived']
skip_cols = ['kfold']
numerical_cols = [
    'Age', 'SibSp', 'Parch', 'Fare','Age_Bin_10',
    'Age_Bin_50', 'Fare_Bin_10', 'Fare_Bin_50',
    'Ticket_len', 'FamilySize', 'IsAgeInt'
]
categorical_cols = []
for col in train_df.columns:
    if col not in (target_col + skip_cols + numerical_cols):
        categorical_cols.append(col)


# In[90]:


train_df[categorical_cols].head()


# In[91]:


non_numeric_cat_cols = []
for col in categorical_cols:
    if (train_df[col].dtypes == object) or (train_df[col].dtypes == bool):
        non_numeric_cat_cols.append(col)


# For basic benchmarking, let's catboost encode the non numeric categorical variables...

# In[92]:


def label_enc(train_df, test_df, features):
    lbl_enc = preprocessing.LabelEncoder()
    full_data = pd.concat(
        [train_df[features], test_df[features]],
        axis=0
    )
    
    for col in (features):
        print(col)
        if train_df[col].dtype == 'object':
            lbl_enc.fit(full_data[col].values)
            train_df[col] = lbl_enc.transform(train_df[col])
            test_df[col] = lbl_enc.transform(test_df[col])
            
    return train_df, test_df


# In[93]:


def one_hot_enc(train_df, test_df, features):
    OH_enc = preprocessing.OneHotEncoder(sparse=False)
    OH_cols_train = pd.DataFrame(OH_enc.fit_transform(train_df[features]))
    OH_cols_test = pd.DataFrame(OH_enc.transform(test_df[features]))
    
    OH_cols_train.index = train_df[features].index
    OH_cols_test.index = test_df[features].index
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    train_df = pd.concat([train_df, OH_cols_train], axis=1)
    test_df = pd.concat([test_df, OH_cols_test], axis=1)
    
    return train_df, test_df


# In[94]:


def catboost_enc(train_df, test_df, features):
    cb_enc = ce.CatBoostEncoder(cols=features)
    cb_enc.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(cb_enc.transform(train_df[features]).add_suffix('_cb'))
    test_df = test_df.join(cb_enc.transform(test_df[features]).add_suffix('_cb'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[95]:


def hash_enc(train_df, test_df, features, components=400):
    hash_enc = ce.HashingEncoder(cols=features, n_components=components)
    hash_enc.fit(train_df[features])
    
    train_df = train_df.join(hash_enc.transform(train_df[features]).add_suffix('_hash'))
    test_df = test_df.join(hash_enc.transform(test_df[features]).add_suffix('_hash'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[96]:


def target_enc(train_df, test_df, features):
    targ_enc = ce.TargetEncoder(cols=features)
    targ_enc.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(targ_enc.transform(train_df[features]).add_suffix('_targ'))
    test_df = test_df.join(targ_enc.transform(test_df[features]).add_suffix('_targ'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[97]:


def helmhert_enc(train_df, test_df, features):
    helm_enc = ce.HelmertEncoder(cols=features)
    helm_enc.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(helm_enc.transform(train_df[features]).add_suffix('_helm'))
    test_df = test_df.join(helm_enc.transform(test_df[features]).add_suffix('_helm'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[98]:


def looe_enc(train_df, test_df, features):
    loo_enc = ce.LeaveOneOutEncoder(cols=features)
    loo_enc.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(loo_enc.transform(train_df[features]).add_suffix('_looe'))
    test_df = test_df.join(loo_enc.transform(test_df[features]).add_suffix('_looe'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[99]:


def woe_enc(train_df, test_df, features):
    WOE_encoder = ce.WOEEncoder(cols=features)
    WOE_encoder.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(WOE_encoder.transform(train_df[features]).add_suffix('_woe'))
    test_df = test_df.join(WOE_encoder.transform(test_df[features]).add_suffix('_woe'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[100]:


def mee_enc(train_df, test_df, features):
    MEE_encoder = ce.MEstimateEncoder(cols=features)
    MEE_encoder.fit(train_df[features], train_df['Survived'])
    
    train_df = train_df.join(MEE_encoder.transform(train_df[features]).add_suffix('_mee'))
    test_df = test_df.join(MEE_encoder.transform(test_df[features]).add_suffix('_mee'))
    
    train_df = train_df.drop(features, axis=1)
    test_df = test_df.drop(features, axis=1)
    
    return train_df, test_df


# In[101]:


low_cardinality_cols = []
high_cardinality_cols = []

for feat in categorical_cols:
    if train_df[feat].nunique() < 5:
        low_cardinality_cols.append(feat)
    else:
        high_cardinality_cols.append(feat)
        
print(f'Low Cardinality Cols: {low_cardinality_cols}')
print(f'High Cardinality Cols: {high_cardinality_cols}')


# In[102]:


train_df, test_df = one_hot_enc(train_df, test_df, low_cardinality_cols)


# In[103]:


train_df, test_df = looe_enc(train_df, test_df, high_cardinality_cols)


# # Features Selection
# We need to select only the important features for better performance of the model. As unnecessary in best case scenario will not add to any productive calculation of the algorithm or in worst case scenario 'confuse' the model.  
# 
# To do the same let's create a wrapper class that has all the built in statistical tests required to perform feature selection and takes some basic inputs from user and spits out the required features.

# In[104]:


cols = list(train_df.columns)
features = [feat for feat in cols if feat not in skip_cols+target_col]


# In[105]:


# From https://github.com/abhishekkrthakur/approachingalmost
class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring, return_cols=True):
        """
        Custom univariate feature selection wrapper on
        different univariate feature selection models from
        scikit-learn.
        :param n_features: SelectPercentile if float else SelectKBest
        :param problem_type: classification or regression
        :param scoring: scoring function, string
        """
        self.n_features = n_features
        
        if problem_type == "classification":
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,
                "mutual_info_classif": mutual_info_classif
            }
        else:
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression
            }
        if scoring not in valid_scoring:
            raise Exception("Invalid scoring function")
            
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception("Invalid type of feature")
    
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    def transform(self, X):
        return self.selection.transform(X)
    
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)
    
    def return_cols(self, X):
        if isinstance(self.n_features, int):
            mask = SelectKBest.get_support(self.selection)
            selected_features = []
            features = list(X.columns)
            for bool, feature in zip(mask, features):
                if bool:
                    selected_features.append(feature)
                    
        elif isinstance(self.n_features, float):
            mask = SelectPercentile.get_support(self.selection)
            selected_features = []
            features = list(X.columns)
            for bool, feature in zip(mask, features):
                if bool:
                    selected_features.append(feature)
        else:
            raise Exception("Invalid type of feature")
        
        return selected_features


# In[106]:


ufs = UnivariateFeatureSelction(
    n_features=0.8,
    problem_type="classification",
    scoring="f_classif"
)

ufs.fit(train_df[features], train_df[target_col].values.ravel())
selected_features = ufs.return_cols(train_df[features])


# # Models Benchmarking
# We will spawn some of the most popular classifiers here and try to benchmark their performance against one another for this dataset.

# In[107]:


def get_stacking():
    level0 = []
    level0.append(('gauss', GaussianNB()))
    level0.append(('lr', LogisticRegression(solver='liblinear')))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('rf', RandomForestClassifier(n_estimators = 500,
                                                random_state=42)))
    level0.append(('xgb', xgb.XGBClassifier(max_depth=7,
                                            n_estimators=1000,
                                            colsample_bytree=0.8,
                                            subsample=0.8,
                                            learning_rate=0.1,
                                            tree_method='gpu_hist',
                                            gpu_id=0)))
    level0.append(('lgbm', LGBMClassifier(metric='binary_logloss',
                                          objective='binary',
                                          learning_rate=0.01,
                                          seed=42,
                                          n_estimators=1000)))
    level0.append(('cbc', CatBoostClassifier(verbose=0,
                                             n_estimators=1000,
                                             eval_metric='AUC',
                                             task_type='GPU',
                                             devices='0',
                                             random_seed=42)))
    level1 = LogisticRegression()
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# In[108]:


def get_voting(vote_type='hard'):
    models = list()
    models.append(('gauss', GaussianNB()))
    models.append(('lr', LogisticRegression(solver='liblinear')))
    models.append(('knn', KNeighborsClassifier()))
    models.append(('rf', RandomForestClassifier(n_estimators = 500,
                                                random_state=42)))
    models.append(('xgb', xgb.XGBClassifier(max_depth=7,
                                            n_estimators=1000,
                                            colsample_bytree=0.8,
                                            subsample=0.8,
                                            learning_rate=0.1,
                                            tree_method='gpu_hist',
                                            gpu_id=0)))
    models.append(('lgbm', LGBMClassifier(metric='binary_logloss',
                                          objective='binary',
                                          learning_rate=0.01,
                                          seed=42,
                                          n_estimators=1000)))
    models.append(('cbc', CatBoostClassifier(verbose=0,
                                             n_estimators=1000,
                                             eval_metric='AUC',
                                             task_type='GPU',
                                             devices='0',
                                             random_seed=42)))
    
    ensemble = VotingClassifier(estimators=models, voting=vote_type)
    return ensemble


# In[109]:


def get_models():
    models = dict()
    models['gauss'] = GaussianNB()
    models['lr'] = LogisticRegression(solver='liblinear')
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['rf'] = RandomForestClassifier(n_estimators = 500,
                                          random_state=42,
                                          n_jobs=-1)
    models['xgb'] = xgb.XGBClassifier(max_depth=7,
                                      n_estimators=1000,
                                      colsample_bytree=0.8,
                                      subsample=0.8,
                                      nthread=-1,
                                      learning_rate=0.1,
                                      tree_method='gpu_hist',
                                      gpu_id=0)
    models['lgbm'] = LGBMClassifier(metric='binary_logloss',
                                    objective='binary',
                                    seed=42,
                                    learning_rate=0.01,
                                    n_estimators=1000)
    models['cbc'] = CatBoostClassifier(verbose=0,
                                       n_estimators=1000,
                                       eval_metric='AUC',
                                       task_type='GPU',
                                       devices='0',
                                       random_seed=42)
    models['stacking'] = get_stacking()
    models['voting_soft'] = get_voting(vote_type='soft')
    models['voting_hard'] = get_voting(vote_type='hard')
    
    return models

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# In[110]:


get_ipython().run_cell_magic('time', '', "\nX = train_df[selected_features]\ny = train_df[target_col]\n\nmodels = get_models()\nresults = []\nnames = []\n\nfor name, model in models.items():\n    scores = evaluate_model(model, X, y)\n    results.append(scores)\n    names.append(name)\n    print(f'{name} : {round(np.mean(scores),3)} ({round(np.std(scores),3)})')\n")


# In[111]:


ax = plt.subplots(figsize=(12, 6))
plt.boxplot(results, labels=names, showmeans=True)
plt.show()


# # Submission Prediction
# 
# ## 1. Stacked

# In[112]:


get_ipython().run_cell_magic('time', '', "\nmodels = get_models()\nclf = models['stacking']\nX = train_df[selected_features]\ny = train_df[target_col]\n\nclf.fit(X, y)\npreds = clf.predict(test_df[selected_features])\nsubmission = pd.DataFrame()\nsubmission['PassengerId'] = test_df['PassengerId']\nsubmission['Survived'] = preds\n")


# In[113]:


submission.head()


# In[114]:


submission.to_csv('Baseline_Stacked.csv', index=False)


# ## 2. LGBM

# In[115]:


get_ipython().run_cell_magic('time', '', "\nmodels = get_models()\nclf = models['lgbm']\nX = train_df[selected_features]\ny = train_df[target_col]\n\nclf.fit(X, y)\npreds = clf.predict(test_df[selected_features])\nsubmission = pd.DataFrame()\nsubmission['PassengerId'] = test_df['PassengerId']\nsubmission['Survived'] = preds\n")


# In[116]:


submission.to_csv('Baseline_LGBM.csv', index=False)


# ## 3. CatBoost

# In[117]:


get_ipython().run_cell_magic('time', '', "\nmodels = get_models()\nclf = models['cbc']\nX = train_df[selected_features]\ny = train_df[target_col]\n\nclf.fit(X, y)\npreds = clf.predict(test_df[selected_features])\nsubmission = pd.DataFrame()\nsubmission['PassengerId'] = test_df['PassengerId']\nsubmission['Survived'] = preds\n")


# In[118]:


submission.to_csv('Baseline_CBC.csv', index=False)


# ## 4. Logistic Regression

# In[119]:


get_ipython().run_cell_magic('time', '', "\nmodels = get_models()\nclf = models['lr']\nX = train_df[selected_features]\ny = train_df[target_col]\n\nclf.fit(X, y)\npreds = clf.predict(test_df[selected_features])\nsubmission = pd.DataFrame()\nsubmission['PassengerId'] = test_df['PassengerId']\nsubmission['Survived'] = preds\n")


# In[120]:


submission.to_csv('Baseline_LogReg.csv', index=False)


# **If you found this notebook useful and use parts of it in your work, please don't forget to show your appreciation by upvoting this kernel. That keeps me motivated and inspires me to write and share these public kernels.** 😊
