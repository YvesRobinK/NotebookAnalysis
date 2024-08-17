#!/usr/bin/env python
# coding: utf-8

# <p style='text-align:center;font-family: sans-serif;font-weight:bold;color:#616161;font-size:25px;margin: 30px;'>Titanic</p>
# <p style='text-align:center;font-family: sans-serif ;font-weight:bold;color:black;font-size:30px;margin: 10px;'>EDA + Modeling for <font color='#08B4E4'>Beginners</font> (Top3%)</p>
# <p style="text-align:center;font-family: sans-serif ;font-weight:bold;color:#616161;font-size:20px;margin: 30px;">RF, XGB, Voting for classification</p>

# Hello. Nice to meet you. I'm a student who's been fascinated by the charm of Kaggle lately and studying hard. I studied the good notes of many Kagglers through Titanic examples and tried to organize them in my own way. In particular, I focused on visualization, preprocessing, and model tuning methods that are useful for binary classification. After a lot of trial and error, I got the top 3% test score and decided to write a note for other Kagglers, especially beginners like me.
#   
# Through this note, you can learn the following.
#   
#   1. Random Forest, XGB Hyperparameter Tuning: You can optimize your model through Random Search, Grid Search, and sequential tuning.
#   2. Ensemble: Voting: Ensemble: Voting: You can use the Voting algorithm to improve the performance of your taxonomic model. I built and verified a total of 52 types of Voting models.
#   3. Visualization for EDA: You can effectively visualize binary sorting problems.
#   4. Basic preprocessing: To solve binary sorting problems, you can learn preprocessing methods such as pruning, generating derivative variables, transforming variables, and selecting variables.
#   
# If you follow this note and submit the results, you will get a high score (0.806). Hyper parameter tuning takes a lot of time, so I defined a variable called allow_tuning so that users can choose this process. If you want to perform hyperparameter tuning yourself, change allow_tuning to True (it takes a lot of time on the Kagle kernel, so it is recommended that you do-it-yourself environment).

# # Import Modules

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('max_columns', 90)
import matplotlib.pyplot as plt
import seaborn as sns
PALETTE = ['#dd4124','#009473', '#b4b4b4', '#336b87']
BACKCOLOR = '#f6f5f5'
sns.set_palette(PALETTE)

from scipy.stats import norm, probplot, skew
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold, train_test_split, RandomizedSearchCV
from sklearn.neighbors import  KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix

from IPython.core.display import HTML

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


# # Read Data

# In[2]:


# Import training and test data.
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

# When exploring a dataset, it is recommended to use the entire data.
all_data = pd.concat((train, test)).reset_index(drop=True)


# In[3]:


all_data.head(10)


# # Understand each features

# In[4]:


all_data.info()


# Through the above process, we can learn the following facts.  
# 1) There are 1309 points in total.    
# 2) There are a total of 11 variables.  
# 3) Age, Fare and Cabin have missing values.  
# 4) Numerical variables(int, float): PassengerId, Pclass, Age, SibSp, Parch, and Fare.  
# 5) Category variables(object): Name, Sex, Ticket, Cabin, Embarked.

# |Variable|Definition|Key|
# |------|---|---|
# |Survived|Survival|0 = No, 1= Yes|
# |Pclass|Ticket class|1=1st, 2=2nd, 3=3rd|
# |Sex|Sex||
# |Age|Age in years||
# |SibSp|number of siblings / spouses aboard the Titanic||
# |Parch|number of parents / children aboard the Titanic||
# |Ticket|Ticket number||
# |Fare|Passenger fare||
# |Cabin|Cabin number||
# |Embarked|Port of Embarkation|C = Cherbourg, Q = Queenstown, S = Southampton|

#  Reading the meaning of variables has many advantages.    
#   
# You can see the actual data types of all variables. All variables are separated as numerical, categorical data, and categorical data are separated back to ordinal / nominal. However, the variables in the program are divided into (int and float) / object. We cannot determine that all int and float variables mean actual countinuous numbers. Reading the data description table and then dividing it directly is a reliable way to distinguish between data types. For example, Pclass is an int type. However, it is an ordinal variable with keys 1st, 2nd, and 3rd. This variable may later be label encoded and scaled. Embarked is an object variable. You don't know if it's ordinal or nominal. If you look at the table, you can be sure that it is a nominal variable with the keys Cherbourg, Queenstown, and Southampton.

# In[5]:


def multi_table(table_list):
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")


# In[6]:


multi_table([pd.DataFrame(all_data[i].value_counts()) for i in all_data.columns])


# We can be sure of some facts.
#   
# 1) Continuous variables: Age, SibSp, Parch, Fare  
# 2) Ordinal variables: Pclass  
# 3) Nominal variables: PassengerId, Survived, Name, Sex, Ticket, Cabin, Embarked  
# 4) Pclass is already encoded in numbers. Therefore, leave it as it is.  
# 5) Name and Cabin are too many items to use.  

# In[7]:


numerical_vars = ['Age', 'SibSp', 'Parch', 'Fare']
ordinal_vars = ['Pclass']
nominal_vars = ['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']


# # Variables explore

# ## Detail Describe (numeric)

# In[8]:


train0 = train[train.Survived == 0]
train1 = train[train.Survived == 1]
cnt = 0
detail_desc = []
for c in train.columns:
    if c == 'PassengerId':
        continue
    if train[c].dtypes != 'object':
        desc = pd.DataFrame(columns=['feature', 'data', 'type', 'count', 'mean', 'median', 'std', 'min', 'max', 'skew', 'null'])
        desc.loc[0] = [c, 'Train', train[c].dtype.name, train[c].count(), train[c].mean(), train[c].median(), train[c].std(), train[c].min(), train[c].max(), train[c].skew(), train[c].isnull().sum()]
        desc.loc[1] = [c, 'All', train[c].dtype.name, all_data[c].count(), all_data[c].mean(), all_data[c].median(), all_data[c].std(), all_data[c].min(), all_data[c].max(), all_data[c].skew(), all_data[c].isnull().sum()]
        desc.loc[2] = [c, 'Target=0', train0[c].dtype.name, train0[c].count(), train0[c].mean(), train0[c].median(), train0[c].std(), train0[c].min(), train0[c].max(), train0[c].skew(), train0[c].isnull().sum()]      
        desc.loc[3] = [c, 'Target=1', train1[c].dtype.name, train1[c].count(), train1[c].mean(), train1[c].median(), train1[c].std(), train1[c].min(), train1[c].max(), train1[c].skew(), train1[c].isnull().sum()]
        desc = desc.set_index(['feature', 'data'],drop=True)
        detail_desc.append(desc.style.background_gradient())


# In[9]:


train0 = train[train.Survived == 0]
train1 = train[train.Survived == 1]
cnt = 0
detail_desc = []
for c in train.columns:
    if c == 'PassengerId':
        continue
    if train[c].dtypes == 'object':
        desc = pd.DataFrame(columns=['feature', 'data', 'type', 'count', 'null', 'mode', 'value_count'])
        desc.loc[0] = [c, 'Train', train[c].dtype.name, train[c].count(), train[c].isnull().sum(), train[c].mode(), train[c].value_counts()]
        #desc.loc[1] = [c, 'All', train[c].dtype.name, all_data[c].count(), all_data[c].mean(), all_data[c].median(), all_data[c].std(), all_data[c].min(), all_data[c].max(), all_data[c].skew(), all_data[c].isnull().sum()]
        #desc.loc[2] = [c, 'Target=0', train0[c].dtype.name, train0[c].count(), train0[c].mean(), train0[c].median(), train0[c].std(), train0[c].min(), train0[c].max(), train0[c].skew(), train0[c].isnull().sum()]      
        #desc.loc[3] = [c, 'Target=1', train1[c].dtype.name, train1[c].count(), train1[c].mean(), train1[c].median(), train1[c].std(), train1[c].min(), train1[c].max(), train1[c].skew(), train1[c].isnull().sum()]
        desc = desc.set_index(['feature', 'data'],drop=True)
        detail_desc.append(desc.style.background_gradient())


# In[10]:


multi_table(detail_desc)


# I looked more closely at the numerical variables. Basic statistics and missing values of each variable were checked in the training dataset and the entire dataset. And I checked the difference by comparing the statistics when Survived was 0 and 1.

# ## Survived (Dependent, Nominal)

# In[11]:


total_cnt = train.Survived.count()
f, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.countplot(train.Survived, edgecolor='black', linewidth=4, ax=ax)
ax.set_xlabel('Survived', weight='bold', size=13)
ax.set_ylabel('Count', weight='bold', size=13)
ax.set_facecolor(BACKCOLOR)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title(f"Survived's distribution", size=15, weight='bold')
for patch in ax.patches:
    x, height, width = patch.get_x(), patch.get_height(), patch.get_width()
    ax.text(x + width / 2, height + 5, f'{height} / {height / total_cnt * 100:2.2f}%', va='center', ha='center', size=15, bbox={'facecolor': 'white', 'boxstyle': 'round'})

plt.show()


# In[12]:


f, ax = plt.subplots(1,1,figsize=(15, 5))
ax.pie(train.Survived.value_counts(),
       explode=[.01,.01],
       labels=['NonSurvived', 'Survived'],
       autopct='%1.1f%%',
       
)
plt.show()


# Survived:  
# It is important to first determine the distribution of dependent variables. You need to know beforehand whether there are biased and unbalanced problems in both classes.
# Survived is a binary variable consisting of 0 and 1. As a result of drawing the countplot, 0 represents 62% and 1 represents 38%. 0 is more than 1, but one of them is not too biased, so it can be analyzed.

# ## Pclass

# I created a simple function using the countplot I used earlier. You can apply any number of seaborn functions.

# In[13]:


def cat_dist(data, var, hue, msg_show=True):
    total_cnt = data[var].count()
    f, ax = plt.subplots(1, 2, figsize=(25, 8))
    hues = [None, hue]
    titles = [f"{var}'s distribution", f"{var}'s distribution by {hue}"]

    for i in range(2):
        sns.countplot(data[var], edgecolor='black', hue=hues[i], linewidth=4, ax=ax[i], data=data)
        ax[i].set_xlabel(var, weight='bold', size=13)
        ax[i].set_ylabel('Count', weight='bold', size=13)
        ax[i].set_facecolor(BACKCOLOR)
        ax[i].spines[['top', 'right']].set_visible(False)
        ax[i].set_title(titles[i], size=15, weight='bold')
        for patch in ax[i].patches:
            x, height, width = patch.get_x(), patch.get_height(), patch.get_width()
            if msg_show:
                ax[i].text(x + width / 2, height + 3, f'{height} \n({height / total_cnt * 100:2.2f}%)', va='center', ha='center', size=12, bbox={'facecolor': 'white', 'boxstyle': 'round'})
    plt.show()


# In[14]:


cat_dist(train, var='Pclass', hue='Survived')
train.pivot_table(index="Pclass", values="Survived", aggfunc=['count', 'sum', 'mean']).style.background_gradient(vmin=0)


# We can see some facts from the above picture.
# 
# 1) In Pclass, 3 accounts for the largest percentage (55%).  
# 2) For Pclass, 2 is the smallest (21%).  
# 3) Pclass1 has a larger percentage of survivors.  
# 4. Pclass 2 has a higher death rate.  
# 5) Pclass3 has an overwhelming death rate.  
# 
# => Most passengers belong to Pclass 3 and most are dead. The percentage of survivors increases from 3 to 1. This variable is the core variable in the classification because the difference in survival rates is obvious depending on the Pclass.

# ## Sex

# In[15]:


cat_dist(train, var='Sex', hue='Survived')
train.pivot_table(index='Sex', values='Survived', aggfunc=['count', 'sum', 'mean']).style.background_gradient(vmin=0)


# We can see some facts from the above picture.
# 
# 1) 65% of all passengers are male and 35% are female.  
# 2) The survival rate of men (18%) is significantly lower than that of women (74%).  
# 
# => Sex is a key variable in this problem because there are clear differences in survival rates depending on gender.

# ## Pclass and Sex

# In[16]:


# Heatmap can visualize continuous values (or binary variables) in categories and categories.
plt.subplots(figsize=(10, 5))
g = sns.heatmap(train.pivot_table(index='Pclass', columns='Sex', values='Survived'), annot=True, cmap="YlGnBu")
g.set_title('Survived ratio by Pclass and Sex', weight='bold', size=15)
g.set_xlabel('Sex', weight='bold', size=13)
g.set_ylabel('Pclass', weight='bold', size=13)
plt.show()

pd.crosstab([train.Sex, train.Survived], train.Pclass,margins=True).style.background_gradient()


# Women in Pclass 1 are said to have the highest survival rate. On the other hand, men in Pclass 3 have the lowest survival rate.

# ## Age

# In[17]:


def continuous_dist(data, x, y):
    f, ax = plt.subplots(1, 3, figsize=(35, 10))
    sns.violinplot(x=data[x], y=data[y], ax=ax[0], edgecolor='black', linewidth=5)
    sns.boxplot(x=data[x], y=data[y], ax=ax[1])
    sns.stripplot(x=data[x], y=data[y], ax=ax[2])
    for i in range(3):
        ax[i].spines[['top','right']].set_visible(False)
        ax[i].set_xlabel(x, weight='bold', size=20)
        ax[i].set_ylabel(y, weight='bold', size=20)
        ax[i].set_facecolor(BACKCOLOR)
    f.suptitle(f"{y}'s distribution by {x}", weight='bold', size=25)
    plt.show()


# In[18]:


f, ax = plt.subplots(1, 2, figsize=(25, 5))
sns.distplot(train.Age, ax=ax[0])
# sns.distplot(train.loc[train.Survived == 0, 'Age'], ax=ax[1])
# sns.distplot(train.loc[train.Survived == 1, 'Age'], ax=ax[1])
sns.histplot(data=train, x='Age', hue='Survived', ax=ax[1], element='step')

for i in range(2):
    ax[i].spines[['top','right']].set_visible(False)
    ax[i].set_xlabel('Age', weight='bold', size=15)
    ax[i].set_ylabel('Density', weight='bold', size=15)
    ax[i].set_facecolor(BACKCOLOR)
f.suptitle("Age' distribution", weight='bold', size=20)
plt.show()

continuous_dist(train, x='Survived', y='Age')


# You can see some facts from the diagram above.
# 
# Passengers in their 20s and 40s are the most common.
# 2) Age distribution is similar regardless of survival.  
# 3) Most passengers in their 50s and older died (violin, boxplot), but the oldest survived (strippplot).  
# 4) The survival rate of children is higher than that of those in their 20s and 30s.(You can tell by looking at violin)  
# 
# => If you look at the histogram, the age distribution of survival is similar. I observed more closely through violin plots, box plots, and strip plots, and I was convinced that age would affect survival. In particular, the survival rate of infants is high, and the survival rate of people in their 20s and 30s is low. I felt it was necessary to categorize age variables into sections and see the survival rate by age group more clearly.

# In[19]:


import copy

tmp_train = copy.deepcopy(train)
tmp_train['AgeBin'] = 6
for i in range(6):
    tmp_train.loc[(tmp_train.Age >= 10*i) & (tmp_train.Age < 10*(i + 1)), 'AgeBin'] = i
tmp_train.head(3)


# In[20]:


t0 = pd.pivot_table(index='AgeBin', values='Survived', data=tmp_train).style.background_gradient()
t1 = pd.pivot_table(index='Pclass', columns='AgeBin', values='Survived', data=tmp_train).style.background_gradient()
t2 = pd.crosstab([tmp_train.AgeBin, tmp_train.Pclass], [tmp_train.Sex, tmp_train.Survived],margins=True).style.background_gradient(vmax=100)
t3 = pd.pivot_table(index='Sex', columns='AgeBin', values='Survived', data=tmp_train).style.background_gradient()
multi_table([t2, t0, t1, t3])


# In[21]:


cat_dist(tmp_train, var='AgeBin', hue='Survived', msg_show=False)


# We can be sure that the survival rate varies from age to age. There are missing values at the current age. Age is a continuous variable, so we can process missing values with central tendency values such as mean and median. However, I don't think this method is a good way because age is a variable that has a lot to do with survival. After I found other variables that seemed to be more relevant to age, I decided to treat unrecorded passengers as the average age of other passengers who had similar characteristics to me.

# In[22]:


all_data.corr().Age


# In[23]:


continuous_dist(train, x='Pclass', y='Age')


# Check the distribution of Pclasses with relatively high correlation coefficients with Age. The average age is the highest in Pclass1 and the lowest in Pclass3. Through this result, you will be able to process age missing values as average values by Pclass. We haven't generated a derivative variable yet, so we can do this again with the variables that will be added later.

# ## SibSp and Parch

# In[24]:


f, ax = plt.subplots(2, 2, figsize=(25, 10))
sns.distplot(train.SibSp, ax=ax[0][0])
sns.histplot(data=train, x='SibSp', hue='Survived', ax=ax[0][1], element='step')

sns.distplot(train.Parch, ax=ax[1][0])
sns.histplot(data=train, x='Parch', hue='Survived', ax=ax[1][1], element='step')

for i in range(4):
    ax[i//2][i%2].spines[['top','right']].set_visible(False)
    if i < 2:
        ax[i//2][i%2].set_xlabel('SibSp', weight='bold', size=10)
    else:
        ax[i//2][i%2].set_xlabel('Parch', weight='bold', size=10)
    ax[i//2][i%2].set_ylabel('Density', weight='bold', size=10)
    ax[i//2][i%2].set_facecolor(BACKCOLOR)
f.suptitle("SibSp and Parch' distribution", weight='bold', size=20)
plt.show()


# In[25]:


continuous_dist(train, x='Survived', y='SibSp')


# In[26]:


continuous_dist(train, x='Survived', y='Parch')


# In[27]:


t0 = pd.pivot_table(index='SibSp', values='Survived', data=train).style.bar()
t1 = pd.pivot_table(index='Parch', values='Survived', data=train).style.bar()
t2 = pd.pivot_table(index='SibSp', columns='Parch', values='Survived', data=train).style.bar()
multi_table([t0, t1, t2])


# Through this process, you can learn several facts.
# 
# 1) The survival rate of single-person passengers without family members is low (34-35%).  
# 2) The survival rate of passengers with brothers and sisters above 3 will decrease.  
# 3) In the case of passengers who are not with parents or children, the survival rate increases with the number of brothers and sisters, but the survival rate of passengers who are with two parents or children gradually decreases.
# 
# => SibSp and Parch are expected to affect survival rates. Since both variables are family-related variables, it is considered possible to combine them.

# ## Fare

# In[28]:


f, ax = plt.subplots(1, 3, figsize=(25, 5))
sns.distplot(train.Fare, ax=ax[0])
sns.histplot(data=train, x='Fare', hue='Survived', ax=ax[1], element='step')
sns.histplot(data=train[train.Fare < 300], x='Fare', hue='Survived', ax=ax[2], element='step')

for i in range(3):
    ax[i].spines[['top','right']].set_visible(False)
    ax[i].set_xlabel('Fare', weight='bold', size=15)
    ax[i].set_ylabel('Density', weight='bold', size=15)
    ax[i].set_facecolor(BACKCOLOR)
f.suptitle("Fare' distribution", weight='bold', size=20)
plt.show()


# In[29]:


continuous_dist(train, x='Survived', y='Fare')


# In[30]:


tmp_train = copy.deepcopy(train)
tmp_train['FareBin'] = pd.cut(tmp_train.Fare, 10)
tmp_train['FareBin'] = LabelEncoder().fit_transform(tmp_train.FareBin)
tmp_train.head(3)


# In[31]:


cat_dist(tmp_train, var='FareBin', hue='Survived', msg_show=False)


# The process is similar to that of exploring Age earlier. You can see the following:
# 
# 1) Fare has a certain degree of normality, but very low levels account for a certain percentage.  
# 2) Groups with very low Fare have lower survival rates, and the higher the Fare, the higher the survival rate.  
# 
# => Like Age, Fare can be grouped, which is a variable that clearly affects survival.

# ## FareBin and Pclass

# Fare's level and Pclass are considered to be correlated, so multivariate searches using two variables were conducted.

# In[32]:


plt.subplots(figsize=(15, 6))
g = sns.countplot('FareBin', hue='Pclass', data=tmp_train)
g.set_title('Count by FareBin and Pclass', weight='bold', size=20)
g.spines[['top','right']].set_visible(False)
g.set_xlabel('FareBin', weight='bold', size=15)
g.set_ylabel('Pclass', weight='bold', size=15)
g.set_facecolor(BACKCOLOR)
plt.show()


# In[33]:


pd.pivot_table(index='FareBin', columns='Pclass', values='Survived', data=tmp_train).style.bar()


# ## Embarked

# In[34]:


cat_dist(train, var='Embarked', hue='Survived')


# In[35]:


pd.pivot_table(data=train, index='Embarked', values='Survived', aggfunc=['count', 'sum', 'mean']).style.background_gradient()


# We can see some facts from the above picture.
# 
# 1) Embarked accounts for the largest proportion in the order of S, C, and Q.  
# 2) S accounts for the largest percentage, but has the lowest survival rate.  
# 3) Embarked C has the highest survival rate.  
# 
# => Embarked is also a good variable because the survival rate varies significantly depending on the value.

# ## Name

# In[36]:


tmp_all_data = copy.deepcopy(all_data)
t0 = pd.DataFrame(tmp_all_data.Name)
t1 = pd.DataFrame(tmp_all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip()).value_counts())
multi_table([t0, t1])


# You can extract common keywords (Mr, Miss, etc.) by name.

# In[37]:


tmp_all_data['Title'] = tmp_all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
#tmp_all_data['Title'] = tmp_all_data.Title.apply(lambda x: 'Others' if x in list(tmp_all_data.Title.value_counts()[tmp_all_data.Title.value_counts() < 8].index) else x)


# In[38]:


continuous_dist(tmp_all_data, x='Title', y='Age')


# We have determined that we can get hints about age in Title, so we have confirmed the distribution of Age in Title. There seems to be some correlation. It can be used to handle Age missing values.

# ## Cabin

# In[39]:


tmp_train.Cabin.value_counts()


# Cabin is rarely recorded and is subject to deletion. However, some observations may have multiple cabins based on blanks. Therefore, we can come up with a new variable called Cabin count. I can replace the missing value of the Cabin count with a certain value, but I easily processed it to 0.
# 
# Also, the first letter of Cabin starts with an alphabet. This value may indicate the type of Cabin. This value can be processed by CabinClass.

# In[40]:


tmp_train['CabinCnt'] = tmp_train.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
tmp_train['CabinClass'] = tmp_train.Cabin.apply(lambda x: str(x)[0])


# In[41]:


t0 = pd.DataFrame(tmp_train.CabinCnt.value_counts())
t1 = pd.DataFrame(tmp_train.CabinClass.value_counts())
multi_table([t0, t1])


# In[42]:


cat_dist(tmp_train, var='CabinCnt', hue='Survived', msg_show=False)


# In[43]:


cat_dist(tmp_train, var='CabinClass', hue='Survived', msg_show=False)


# After visualizing the survival rates of CabinCnt and CabinClass, we can see that these are the variables that can be used.

# ## Ticket

# In[44]:


tmp_train.Ticket


# In[45]:


tmp_train['IsNumericTicket'] = tmp_train.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
tmp_train['TicketType'] = tmp_train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)


# Tickets also contain several common keywords. Therefore, new variables can be generated after parsing according to specific criteria.

# In[46]:


cat_dist(tmp_train, var='IsNumericTicket', hue='Survived')


# In[47]:


pd.pivot_table(data=tmp_train, index='TicketType', values='Survived').T.style.background_gradient(axis=1)


# # Feature engineering

# Start Feature Engineering based on the contents organized through EDA. Some processes have been simplified (ex: Age Missing Value Processing). You can proceed with this course in many ways as you like.

# In[48]:


# missing values
all_data['Age'] = all_data.Age.fillna(train.Age.median())
all_data['Fare'] = all_data.Fare.fillna(train.Fare.median())
all_data.dropna(subset=['Embarked'], inplace=True)
cabins = all_data.Cabin
all_data.drop(['Cabin'], axis=1, inplace=True)


# In[49]:


# derivative features
all_data['CabinCnt'] = cabins.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['CabinClass'] = cabins.apply(lambda x: str(x)[0])
all_data['IsNumericTicket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['TicketType'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)
all_data['Title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
all_data['Family'] = all_data.SibSp + all_data.Parch


# In[50]:


# feature transform
numeric_vars = ['Age', 'SibSp', 'Parch', 'Fare', 'CabinCnt', 'Family']
ordinal_vars = ['Pclass']
nominal_vars = ['Name', 'Sex', 'Ticket', 'Embarked', 'CabinClass', 'IsNumericTicket', 'TicketType', 'Title']
all_data[nominal_vars] = all_data[nominal_vars].astype('str')

for feature in numeric_vars:
    all_data[feature] = np.log1p(all_data[feature])

scaler = StandardScaler()
numeric_vars = all_data.columns[(all_data.dtypes != 'object') & (all_data.columns != 'PassengerId') & (all_data.columns != 'Survived') & (all_data.columns != 'IsTrain')]
all_data[numeric_vars] = scaler.fit_transform(all_data[numeric_vars])


# In[51]:


# split data
all_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
data_dummies = pd.get_dummies(all_data)
X_train = data_dummies[data_dummies.Survived.notnull()].drop(['Survived'], axis=1)
y_train = data_dummies[data_dummies.Survived.notnull()].Survived
X_test = data_dummies[data_dummies.Survived.isnull()].drop(['Survived'], axis=1)


# In[52]:


X_train.shape, y_train.shape, X_test.shape


# In[53]:


selector = RandomForestClassifier().fit(X_train, y_train)

imps = pd.DataFrame(selector.feature_importances_, X_train.columns, columns=['Importance'])
imps = pd.DataFrame(imps.Importance.sort_values(ascending=False))

plt.subplots(figsize=(20, 10))
g = sns.barplot(x=imps.index, y=imps.Importance)
g.set_xticklabels(g.get_xticklabels(),rotation = 90)
plt.show()


# In[54]:


all_data.Title = all_data.Title.apply(lambda x: 'Others' if x in list(all_data.Title.value_counts()[all_data.Title.value_counts() < 8].index) else x)
all_data.TicketType = all_data.TicketType.apply(lambda x: 'Others' if x in list(all_data.TicketType.value_counts()[all_data.TicketType.value_counts() < 10].index) else x)


# In[55]:


# split data2
data_dummies = pd.get_dummies(all_data)
X_train = data_dummies[data_dummies.Survived.notnull()].drop(['Survived'], axis=1)
X_test = data_dummies[data_dummies.Survived.isnull()].drop(['Survived'], axis=1)


# In[56]:


selector = RandomForestClassifier().fit(X_train, y_train)

imps = pd.DataFrame(selector.feature_importances_, X_train.columns, columns=['Importance'])
imps = pd.DataFrame(imps.Importance.sort_values(ascending=False))

plt.subplots(figsize=(20, 10))
g = sns.barplot(x=imps.index, y=imps.Importance)
g.set_xticklabels(g.get_xticklabels(),rotation = 50)
plt.show()


# # Modeling

# Creating a good model is as important as creating a good variable. A good model refers to a model with the best generalization performance using hyperparameters optimized for a given dataset. To solve this problem, I used Logistic Regression, Knn, Support Vector Machine, Radnom Forest, XGBoost, and Voting Model. Because each model is suitable for binary classification tasks and has different algorithms and characteristics, learning the same dataset can perform differently. In addition, because each model has a different categorization point for key people, it may be better to combine them.
# 
# I used Grid Search to find this model's HYPER PARAMTER. Random Forest hyperparameter tuning is too wide and difficult. It is easier to adjust by narrowing the range using Random Search first. XGBoost has many parameters and a wide range, so tuning is more difficult than other models. Therefore, tuning all hyperparameters at once takes a very long time.
# 
# XGBoost has many hyperparameters, but there are parameters that have a relatively significant impact. This includes learning_rate and n_estimators. I took advantage of early_stopping to get the best learning_rate, and then tuned the hyperparameters such as max_depth, min_child_weight, gamma, subsample, etc. After tuning the core parameters, the optimal parameters are derived by final tuning the learning_rate and n_estimators again.

# In[57]:


# Hyperparameter tuning takes a lot of time. If this variable is False, the tuning process will be omitted and the learning will proceed 
# with the hyperparameters already obtained. If this variable is true, you can proceed with the tuning process directly.
allow_tuning = False


# In[58]:


# This function is a function created by myself to eliminate repeated code generated by tuning XGBoost.
# params_grid_xgb: Combines fixed parameters for grid search in xgboost.
# features: Target features to be tuned using this function
# values: Search parameters for each feature
# X,y: Datasets.
# last: If this value is false, change each value of the GridSearchCV object's best_params to a list for immediate use in the next adjustment.
def xgb_gridsearch(params_grid_xgb, features, values, X, y, last=False):
    x_train, x_test = train_test_split(X, test_size=.2, random_state=42)
    y_train_tmp, y_test_tmp = train_test_split(y, test_size=.2, random_state=42)

    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 42)

    model_xgb = XGBClassifier(use_label_encoder = False, 
                              objective = 'binary:logistic')
    
    for i in range(len(features)):
        params_grid_xgb[features[i]] = values[i]
    search_xgb = GridSearchCV(model_xgb, params_grid_xgb, verbose = 0,
                              scoring = 'neg_log_loss', cv = cv).fit(x_train, y_train_tmp, early_stopping_rounds = 15, 
                                  eval_set = [[x_test, y_test_tmp]], 
                                  eval_metric = 'logloss', verbose = False)
    for i in range(len(features)):
        print(f"{features[i]}: {search_xgb.best_params_[features[i]]}")
    if not last:
        for k, v in search_xgb.best_params_.items():
            search_xgb.best_params_[k] = [v]
    return search_xgb, search_xgb.best_params_


# ## KNN

# In[59]:


if allow_tuning:
    params_knn = {
        'n_neighbors' : range(1, 10),
        'weights' : ['uniform', 'distance'],
        'algorithm' : ['auto', 'ball_tree','kd_tree'],
        'p' : [1,2]
    }
    model_knn = knn()
    search_knn = GridSearchCV(model_knn, params_knn, cv=5, scoring='accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
    print(search_knn.best_params_)


# ## Logistic Regression

# In[60]:


if allow_tuning:
    params_logistic = {
        'max_iter': [2000],
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['liblinear']
    }
    model_logistic = LogisticRegression()
    search_logistic = GridSearchCV(model_logistic, params_logistic, cv=5, scoring='accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
    print(search_logistic.best_params_)


# ## SVC

# In[61]:


if allow_tuning:
    params_svc = [{'kernel': ['rbf'], 'gamma': [.01, .1, .5, 1, 2, 5, 10], 'C': [.1, 1, 10, 100, 1000], 'probability': [True]},
                  {'kernel': ['poly'], 'degree' : [2, 3, 4, 5], 'C': [.01, .1, 1, 10, 100, 1000], 'probability': [True]}]
    model_svc = SVC()
    search_svc = GridSearchCV(model_svc, params_svc, cv=5, scoring='accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
    print(search_svc.best_params_)


# In[62]:


if allow_tuning:
    params_svc = {'kernel': ['rbf'], 'gamma': [i/10000 for i in range(90, 110)], 'C': range(50, 80, 10), 'probability': [True]}
    model_svc = SVC()
    search_svc = GridSearchCV(model_svc, params_svc, cv=5, scoring='accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
    print(search_svc.best_params_)


# ## Random Forest

# First, use Random Search to narrow the search range, and then proceed with Grid Search. Random search results vary from time to time. Increasing n_iter results in more consistent results.

# In[63]:


# if allow_tuning:
#     params_rf = {
#         'n_estimators': range(100, 2000, 200),
#         'criterion':['gini','entropy'],
#         'bootstrap': [True, False],
#         'max_depth': list(range(5, 100, 5)) + [None],
#         'max_features': ['auto','sqrt', 5, 10],
#         'min_samples_leaf': range(2, 11, 2),
#         'min_samples_split': range(2, 11, 2)}
#     model_rf = RandomForestClassifier()
#     search_rf = RandomizedSearchCV(model_rf, params_rf, cv=5,
#                                    scoring='accuracy', n_jobs=-1, verbose=1,
#                                    n_iter=100).fit(X_train, y_train)
#     print(search_rf.best_params_)


# In[64]:


if allow_tuning:
    params_rf = {
        'n_estimators': [95, 100, 105],
        'criterion':['entropy'],
        'bootstrap': [True, False],
        'max_depth': [40, 45, 50],
        'max_features': [4, 5, 6],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [9, 10, 11],
        'random_state': [734]}
    model_rf = RandomForestClassifier()
    search_rf = GridSearchCV(model_rf, params_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1).fit(X_train, y_train)
    search_rf.best_params_['random_state']=242
    search_rf.best_estimator_.random_state=242
    print(search_rf.best_params_)


# ## XGBoost

# In[65]:


if allow_tuning:
    # Initial params.
    params_xgb = {'n_estimators': [1000],
                  'learning_rate': [0.1],
                  'max_depth': [5],
                  'min_child_weight': [1],
                  'gamma': [0],
                  'subsample': [0.8],
                  'colsample_bytree': [0.8],
                  'n_jobs': [-1],
                  'objective': ['binary:logistic'],
                  'use_label_encoder': [False],
                  'eval_metric': ['logloss'],
                  'scale_pos_weight': [1]}
    
    # learning rate tuning.
    search_xgb, params_xgb = xgb_gridsearch(params_xgb, 
                                            ['learning_rate'], 
                                            [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]],
                                            X_train, y_train)
    # max_depth, min_child_weight tuning.
    search_xgb, params_xgb = xgb_gridsearch(params_xgb,
                                            ['max_depth', 'min_child_weight'],
                                            [range(3, 10), range(1, 6)],
                                            X_train, y_train)
    
    # gamma tuning.
    search_xgb, params_xgb = xgb_gridsearch(params_xgb,
                                            ['gamma'],
                                            [[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]],
                                            X_train, y_train)
    search_xgb, params_xgb = xgb_gridsearch(params_xgb,
                                            ['subsample', 'colsample_bytree'],
                                            [[i/100.0 for i in range(75,90,5)], [i/100.0 for i in range(75,90,5)]],
                                            X_train, y_train)
    
    # reg_alpha tuning.
    search_xgb, params_xgb = xgb_gridsearch(params_xgb,
                                            ['reg_alpha'], 
                                            [[1e-5, 1e-2, 0.1, 1, 100]], 
                                            X_train, y_train)
    
    # learning rate re tuning.
    params_xgb['n_estimators'] = [5000]
    search_xgb, params_xgb = xgb_gridsearch(params_xgb,
                                            ['learning_rate'],
                                            [[0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2]],
                                            X_train, y_train, last=True)

    x_train, x_test = train_test_split(X_train, test_size=.2, random_state=42)
    y_train_tmp, y_test_tmp = train_test_split(y_train, test_size=.2, random_state=42)
    model_xgb = XGBClassifier(**params_xgb)
    
    # n_estimators tuning.
    model_xgb = model_xgb.fit(x_train, y_train_tmp, eval_set=[(x_test, y_test_tmp)], eval_metric=['logloss'], early_stopping_rounds=15, verbose=0)
    search_xgb.best_estimator_.n_estimators = model_xgb.best_iteration


# Create each model based on hyperparameter tuning results. If allow_tuning is False, create a model based on the results tuned in advance.

# In[66]:


if allow_tuning:
    model_knn = search_knn.best_estimator_
    model_logistic = search_logistic.best_estimator_
    model_svc = search_svc.best_estimator_
    model_rf = search_rf.best_estimator_
    model_xgb = search_xgb.best_estimator_
else:
    model_knn = knn(algorithm='auto', 
                    n_neighbors=9,
                    p=1, 
                    weights='uniform')
    
    model_logistic = LogisticRegression(C=0.08858667904100823,
                                        max_iter=2000, 
                                        penalty='l2', 
                                        solver='liblinear')
    model_svc = SVC(C=70,
                    gamma=0.0106,
                    kernel='rbf',
                    probability=True)
    
    model_rf = RandomForestClassifier(bootstrap=True,
                                      criterion='entropy',
                                      max_depth=50, max_features=6, 
                                      min_samples_leaf=1, 
                                      min_samples_split=10, 
                                      n_estimators=100,
                                      random_state=734)
    
    model_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=0.8,
                              enable_categorical=False, eval_metric='logloss', gamma=0.8,gpu_id=-1, importance_type=None, interaction_constraints='',
                              learning_rate=0.15, max_delta_step=0, max_depth=5,
                              min_child_weight=1, missing=np.nan, monotone_constraints='()',
                              n_estimators=15, n_jobs=-1, num_parallel_tree=1, predictor='auto',
                              random_state=0, reg_alpha=1e-05, reg_lambda=1, scale_pos_weight=1,
                              subsample=0.8, tree_method='exact', use_label_encoder=False,
                              validate_parameters=1, verbosity=0)

models = {
    'knn': model_knn,
    'logistic': model_logistic,
    'svc': model_svc,
    'rf': model_rf,
    'xgb': model_xgb
}


# ## Voting model
# 
# I also tested the voting model suitable for the classification problem. I created a recursive function that allows users to combine as many as they want in a total of five models.

# In[67]:


import copy

# goal: The number of models to combine.
# estimaors: empty list.
# voting: voting method.
def select_models(start, cnt, goal, estimators, voting):
    if cnt == goal:
        estimators_copy = copy.deepcopy(estimators)
        voting_name = f'{voting}_' + '_'.join([i[0] for i in list(estimators_copy)])
        models[voting_name] = VotingClassifier(estimators=estimators_copy, voting=voting)
        return
    for i in range(start, 5):
        estimators.append(list(models.items())[i])
        select_models(i + 1, cnt + 1, goal, estimators, voting)
        estimators.pop()


# In[68]:


# create voting models
select_models(0, 0, 2, [], 'hard')
select_models(0, 0, 3, [], 'hard')
select_models(0, 0, 4, [], 'hard')
select_models(0, 0, 5, [], 'hard')

select_models(0, 0, 2, [], 'soft')
select_models(0, 0, 3, [], 'soft')
select_models(0, 0, 4, [], 'soft')
select_models(0, 0, 5, [], 'soft')


# In[69]:


# Dictionary for storing results for each model.
result_by_model = pd.DataFrame({'model name': models.keys(), 'model': models.values(), 'score': 0})


# In[70]:


# Cross-validation progresses for all models.
for name, model in models.items():
    result_by_model.loc[result_by_model['model name'] == name, 'score'] = cross_val_score(model, X_train,y_train,cv=5).mean()


# In[71]:


# Cross validation scores of all models.
result_by_model.sort_values('score', ascending=False).reset_index(drop=True)


# I tested all of the above models and found that Random Forest had the highest test score, unlike the cross validation results.

# In[72]:


model_name = 'rf'
models[model_name].fit(X_train, y_train)
y_pred = models[model_name].predict(X_test).astype('int')

submission = pd.DataFrame({'PassengerId': test.PassengerId, 
                              'Survived': y_pred})

submission.to_csv('submission.csv', index = False)

