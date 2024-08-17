#!/usr/bin/env python
# coding: utf-8

# ### <div style="padding: 35px;color:Black;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.definition.org/wp-content/uploads/2018/04/Titanic_11.jpg)"><b><span style='color:#404040'>Titanic model 🚢🌊 Getting Started.. </span></b> </div>
# 
# <br>
# 
# In this kernel I will create a machine learning model on the famous Titanic dataset, which is used by many people all over the world. The objective of this project is to build a model to predict whether passengers on the Titanic would survive or not based on pattern extracted from analysing features.
# 
# <br>
# 
# <p style="text-align:center; ">
# <img src="https://cf.ltkcdn.net/kids/images/std/236793-1600x1200-titanic.jpg" style='width: 500px; height: 350px;'>
# </p>
# 
# <br>
# 
# ### <b><span style='color:#0077b3'>|</span> Domain Knowledge</b>
# 
# <br>
# 
# * `survival`: Target column has two values (0 = No, 1 = Yes).
# * `pclass`:	Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
# * `sex`: male or female.
# * `Age`: Age of passengers in years.	
# * `sibsp`: number of siblings / spouses aboard the Titanic.	
# * `parch`: number of parents / children aboard the Titanic.
# * `ticket`:	Ticket number.
# * `fare`: Passenger fare.
# * `cabin`: Cabin number.	
# * `embarked`: Port of Embarkation has three values (C = Cherbourg, Q = Queenstown, S = Southampton)   
# 
# ✔️ **These variables, in combination with appropriate statistical and machine learning techniques, can help predict whether passengers would survive or not.**
# 
# <br>
# 
# <div style="padding: 20px;color:Black;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.definition.org/wp-content/uploads/2018/04/Titanic_11.jpg)"><b><span style='color:#404040'> Table of contents </span></b> </div>
# 
#     
# |No  | Contents |
# |:---| :---     |
# |1   | [<font color="#1c1c1c"> Introduction </font>](#1)                   
# |2   | [<font color="#1c1c1c"> Data Review </font>](#2)                         
# |3   | [<font color="#1c1c1c"> Explore data analysis </font>](#3)                     
# |4   | [<font color="#1c1c1c"> data preprocessing </font>](#4)                       
# |5   | [<font color="#1c1c1c"> Modeling </font>](#5)      
# |6   | [<font color="#1c1c1c"> Evaluate </font>](#6)              
# |7   | [<font color="#1c1c1c"> Submission </font>](#7)              
# |8   | [<font color="#1c1c1c"> Conclusion </font>](#8)              
#    

# # <a id="1"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>0 | Introduction </span></b> </div>
# 
# ## <b>I <span style='color:#595959'>|</span> Import libraries</b> 

# In[1]:


# linear algebra
import numpy as np 

# data processing
import pandas as pd 
import re

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

#machine learning libraries:
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict

#evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve 
from sklearn.metrics import r2_score,f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve

import os
os.environ['PYTHONWARNINGS']='ignore::FutureWarning'

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.warn("once")


# ## <b>II <span style='color:#595959'>|</span> Import data</b> 

# In[2]:


train_df = pd.read_csv(r"/kaggle/input/titanic/train.csv")
test_df = pd.read_csv(r"/kaggle/input/titanic/test.csv")


# In[3]:


train_df.head(5)


# # <a id="2"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>1 | Data Review </span></b> </div>

# In[4]:


print(f'''The shape of data:

1) training: {train_df.shape}
2) testing: {test_df.shape}
''')


# In[5]:


print(train_df.info())


# In[6]:


train_df.describe()


# In[7]:


train_df.describe(include=['O'])


# In[8]:


round(train_df['Survived'].mean()*100,2)


# In[9]:


print(train_df.dtypes)
print(f"""
Number of float features: {len(train_df.select_dtypes('float').columns)}
Number of int features: {len(train_df.select_dtypes('int').columns)}
Number of object features: {len(train_df.select_dtypes('object').columns)}
""")


# ### **<mark style="color:white;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> Note that </mark>**
# you should put in your mind:
# 
# **<mark style="background-color:#595959;color:white;border-radius:2px;opacity:1.0">1</mark>** Which numerical features are `discrete` or `continuous`?
# 
# **<mark style="background-color:#595959;color:white;border-radius:2px;opacity:1.0">2</mark>** Which categorical features are `nominal`, `ordinal` or `ratio`?
# 
# **<mark style="background-color:#595959;color:white;border-radius:2px;opacity:1.0">3</mark>** Which features are mixed data types?
# 
# 
# This can help us select the appropriate and suitable plots for visualization.

# In[10]:


# columns which have nulls and the percentage of nulls in each column

train_data_na = (train_df.isnull().sum() / len(train_df)) 
train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)
train_missing_data = pd.DataFrame({'mean of nulls' :train_data_na , "number_of_nulls" : train_df[train_data_na.index].isna().sum()})
train_missing_data


# In[11]:


test_data_na = (test_df.isnull().sum() / len(test_df)) 
test_data_na = test_data_na.drop(test_data_na[test_data_na == 0].index).sort_values(ascending=False)
test_missing_data = pd.DataFrame({'mean of nulls' :test_data_na, "number_of_nulls" : test_df[test_data_na.index].isna().sum()})
test_missing_data


# In[12]:


train_df[['Ticket']].duplicated().sum()/len(train_df)*100


# ## <b><span style='color:#595959'>|</span> Observations </b> 
# 
# * The training-set has 891 rows and 11 features + `survived` column (target feature).
# <br>
# 
# * `Categorical` columns: Survived, Sex, and Embarked. `Ordinal` columns: Pclass.
# 
# * `Continous` columns: Age, Fare. `Discrete` columns: SibSp, Parch.
# 
# * `alphanumeric` columns: Ticket and Cabin.
# <br>
# 
# * Around `38.38%` of the training-set survived the Titanic.
# 
# * The passenger ages range from 0.4 to 80.
# 
# * `Sex` column has two values with `65%` male (freq=577/count=891).
# 
# * `Embarked` column has three values. port S used by `72.4%` of passengers.
# 
# * `Ticket` column contains high ratio of duplicates (`23.5%`). we might want to drop it.
# <br>
# 
# * There are three columns in our data have missing values: 
#     * `Cabin` column have almost `77%` null values of its data. we might want to drop it.
#     * 177 value in `Age` column are missed, Around `19%` of its data.
#     * Just two values in `Embarked` are missing, which can easily be filled.
# <br>
# 
# 
# * `SibSp` and `Parch` These features have zero correlation for certain values. We might derive a feature or a set of features from these individual features. 
# 

# # <a id="3"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>2 | Explore data analysis </span></b> </div>
# 
# <br>
# 
# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 1-  Univariative Analysis </mark>**
# 
# <br>
# 

# In[13]:


# Add labels to the end of each bar in a bar chart.

def add_value_labels(ax, spacing=5):

    # For each bar: Place a label    
    for rect in ax.patches:
        
        # Get X and Y placement of label from rect.
        x = rect.get_x() + rect.get_width() / 2
        y = rect.get_height()-3

        # Determine vertical alignment for positive and negative values
        va = 'bottom' if y >= 0 else 'top'

        # Format the label to one decimal place
        label = "{}".format(y)

        # Determine the vertical shift of the label
        # based on the sign of the y value and the spacing parameter
        y_shift = spacing * (1 if y >= 0 else -1)

        # Create the annotation
        ax.annotate(label, (x, y), xytext=(0, y_shift),
                    textcoords="offset points", ha='center', va=va)


# ## <b>I <span style='color:#595959'>|</span> Analysis categorical columns separately</b> 

# In[14]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
plt.title('Survived countplot', size=15)
plot= sns.countplot(data=train_df, x='Survived' ,palette="Set2")
add_value_labels(plot)

plt.subplot(1,3,2)
plt.title('Pclass countplot', size=13)
plot= sns.countplot(data=train_df, x='Pclass', palette="Set2")
add_value_labels(plot)

plt.subplot(1,3,3)
plt.title('Sex countplot', size=13)
plot= sns.countplot(data=train_df, x='Sex', palette='Set2')
add_value_labels(plot)

plt.tight_layout()


# In[15]:


plt.figure(figsize=(13,5))

plt.subplot(1,3,1)
plt.title('Pclass-Survived plot', size=15)
plot= sns.countplot(data=train_df, x='Pclass',hue='Survived' ,palette="Blues")
add_value_labels(plot)

plt.subplot(1,3,2)
plt.title('Sex-Survived plot', size=15)
plot= sns.countplot(data=train_df, x='Sex', hue='Survived' ,palette="Greens")
add_value_labels(plot)

plt.subplot(1,3,3)
plt.title('Embarked-Survived plot', size=15)
plot= sns.countplot(data=train_df, x='Embarked',hue='Survived' ,palette="Reds")
add_value_labels(plot)


# In[16]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# In[17]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()


# ## <b>I <span style='color:#595959'>|</span> Analysis Age column </b> 

# In[18]:


#survived passengers
survived_df= train_df[train_df['Survived']==1]

#non-survived passengers
unsurvived_df= train_df[train_df['Survived']==0]


# In[19]:


plt.figure(figsize=(13,5))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
sns.histplot(data=survived_df, x='Age', kde=True, bins=20,  alpha=0.3 );

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
sns.histplot(data=unsurvived_df, x='Age', kde=True, bins=20, alpha=0.3 );


# In[20]:


Infant_passengers = train_df[train_df['Age']<=5]
Infant_passengers['Survived'].value_counts(normalize=True)


# In[21]:


Old_passengers = train_df[train_df['Age']==80]
Old_passengers['Survived'].value_counts()


# ## <b><span style='color:#595959'>|</span> Observations </b> 
# 
# * Pclass=3 had most passengers(484 passengers), however the most of them didn't survive (112 passengers not-survived about `75.8%`).
# 
# * Most passengers in Pclass=1 survived about `62.9%`. 
# <br>
# 
# * Infant passengers (Age <=5) had high survival rate, about `70.4%` of infant passengers survived.
# 
# * There is only one passengers with 80 years old and he survived.
# 
# * Large number of 15-25 year olds did not survive.
# 
# * There is only one passengers with 80 years old and he is survived
# <br>
# 
# * Female passengers had much better survival rate than males( `74.2%` of female passengers survived but just `18.8%` of males survived.)
# <br>
# 
# * Port S had most passengers(630 passengers) but the most of them didn't survive (420 passengers not-survived about `67%`).
# 
# * The majority of port C passengers survived (86 passengers survived out of 154 about `55.8%`).

# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 2-  Bivariative Analysis</mark>**
# 
# <br>
# 
# ## <b>I <span style='color:#595959'>|</span> Sex and Age analysis </b> 

# In[22]:


plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.title('survived passenger ages')
sns.histplot(data=survived_df, x='Age', hue='Sex', kde=True, bins=20,  alpha=0.3 );

plt.subplot(2,2,2)
plt.title('unsurvived passenger ages')
sns.histplot(data=unsurvived_df, x='Age',hue='Sex', kde=True, bins=20, alpha=0.3 );

plt.subplot(2,2,3)
plt.title('survived passenger ages')
sns.boxplot(x=survived_df['Sex'], y=train_df["Age"],palette="Set2");

plt.subplot(2,2,4)
plt.title('unsurvived passenger ages')
sns.boxplot(x=unsurvived_df['Sex'], y=train_df["Age"],palette="Set2");


# In[23]:


grid = sns.FacetGrid(train_df, col='Sex', row='Survived', aspect=1.2)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[24]:


print('surviving male passengers \n')

print(survived_df[survived_df['Sex']=='male'][['Age']].describe().T)
print('--------------------------------')
print('surviving female passengers \n')

print(survived_df[survived_df['Sex']=='female'][['Age']].describe().T)


# In[25]:


print('non-surviving male passengers \n')

print(unsurvived_df[unsurvived_df['Sex']=='male'][['Age']].describe().T)
print('--------------------------------')
print('non-surviving female passengers \n')

print(unsurvived_df[unsurvived_df['Sex']=='female'][['Age']].describe().T)


# ## <b>II <span style='color:#595959'>|</span> Pclass and Age analysis </b> 

# In[26]:


plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
sns.boxplot(x=survived_df['Pclass'], y=train_df["Age"],palette="Set2");

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
sns.boxplot(x=unsurvived_df['Pclass'], y=train_df["Age"],palette="Set2");


# In[27]:


grid = sns.FacetGrid(train_df, col='Pclass', row='Survived', aspect=1.2)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# ## <b>III <span style='color:#595959'>|</span> Sex and Pclass analysis </b> 

# In[28]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_df, x='Pclass', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_df, x='Pclass', hue='Sex',palette="Set2");
add_value_labels(plot)


# In[29]:


grid = sns.FacetGrid(train_df, col='Sex', aspect=1.2)
grid.map(sns.pointplot,'Pclass', 'Survived')
grid.add_legend();


# ## <b><span style='color:#595959'>|</span> Observations </b> 
# 
# * Average age for non-surviving male passengers is 32, And on the other hand non-surviving female passengers is 25 .
# 
# * Most male passengers aged 20-35 did not survive.
# <br>
# 
# 
# * Infant passengers in Pclass=2 and Pclass=3 mostly survived. 
# <br>
# 
# 
# * Half of the female passengers inside Pclass=3 survive (`50%` of passengers counted 69).
# * All female passengers inside Pclass=1 survived and about `95` of female passengers inside Pclass=2 survived.
# * About `87%` of male passengers inside Pclass=3 and Pclass=2 non-survived but about `36.2%` of them survived in Pclass=1.

# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 3- Multivariative Analysis</mark>**
# 
# <br>
# 
# ## <b>I <span style='color:#595959'>|</span> Sex, Pclass and Embarked analysis </b> 

# In[30]:


grid = sns.FacetGrid(train_df, col='Embarked', aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex')
grid.add_legend();


# ## <b>II <span style='color:#595959'>|</span> Sex, Fare and Embarked analysis </b> 

# In[31]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_df, x='Embarked', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_df, x='Embarked', hue='Sex',palette="Set2");
add_value_labels(plot)


# In[32]:


grid = sns.FacetGrid(train_df, col='Embarked', row='Survived', aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend();


# ## <b><span style='color:#595959'>|</span> Observations </b> 
# 
# * Women on port Q and S have a higher chance of survival. But it's inverse at port C.
# 
# * Men have a high survival probability on port C, but a low probability on port Q or S.
# 
# * Most female passengers inside Pclass=3 on port C and S non-survived but most of them survived on port Q.
# <br>
#  
#  
# * Higher fare paying passengers had better survival. 
# 
# * Passengers on port Q paid less fare.
# 
# * Nearly no male survived on port Q. 
# 
# * Femals on port Q (about `37%` of all port Q passengers) Survived, however they paid small fare.
# 
# 

# # <a id="4"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>3 | data preprocessing </span></b> </div>
# 
# <br>
# 
# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 1-  Drop unuseful columns </mark>**
# <br>
# 
# * Drop `PassengerId` column from the train set, because it won't benefit our model. I won't drop it from the test set, since it's required there for the submission.
# 
# * Drop `Cabin` column, becouse 77% of its data are missing. And a general rule is that, if more than half of the data in a column is missing, it's better to drop it.
# 
# * Drop `Ticket` column, becouse there may not be a correlation between Ticket and survival and its high ratio of duplicates.

# In[33]:


#Drop PassengerId column from the train set
train_df.drop(columns='PassengerId', inplace=True)

#Drop Cabin column.
train_df.drop(columns='Cabin', inplace=True)
test_df.drop(columns='Cabin', inplace=True)

#Drop Ticket column
train_df.drop(columns='Ticket', inplace=True)
test_df.drop(columns='Ticket', inplace=True)


# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 2- Dealing with missing values </mark>**
# <br>
# 
# * We will guess `Age` missing values using random numbers between mean and standard deviation.
# <br>
# 
# * we will fill missing values of `Embarked` columns with mode value. As a reminder, we have to deal with just two missing values.

# In[34]:


datasets=[train_df, test_df]

for dataset in datasets:
    
    #Age column
    
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    nulls = dataset["Age"].isnull().sum()
    
    # compute random numbers between the mean, std and is_null
    random_age = np.random.randint(mean - std, mean + std, size = nulls)
    
    # fill NaN values in Age column with random values generated
    dataset["Age"][dataset["Age"].isna()] = random_age
    dataset["Age"] = train_df["Age"].astype(int)
    
    
    
    #Embarked column
    
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)


# In[35]:


#check
print(train_df['Age'].isna().sum())
print(train_df['Embarked'].isna().sum())


# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 3- Create new columns</mark>**
# <br>
# 
# ## <b>I <span style='color:#595959'>|</span> Create Title column </b> 

# In[36]:


title_list = pd.concat([train_df,test_df])['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1]).unique()
title_list


# In[37]:


# replacing all titles with mr, mrs, miss, master, and boy 
def replace_titles(x):
    title=x['Title'].strip()
    
    if (x['Age']<13): return 'Boy'
    
    if title in ['Don', 'Rev', 'Col','Capt','Sir','Major','Jonkheer']: return 'Mr'
    
    elif title in ['Countess', 'Mme']: return 'Mrs'
    
    elif title in ['Mlle', 'Ms','Lady','Dona']: return 'Miss'
    
    elif title =='Dr':
        
        if x['Sex']=='male': return 'Mr'
        else: return 'Mrs'
        
    else: return title
    
for dataset in datasets:
    
    #create a new columns containing the title for each name
    dataset['Title'] = dataset['Name'].apply(lambda x: re.findall(r'[, ]\w+[.]',x)[0][:-1])
    
    #apply replacing title function to all titles
    dataset['Title'] = dataset.apply(replace_titles, axis=1)


# In[38]:


print(f'Train data has : {train_df["Title"].unique()}')
print()
print(train_df["Title"].value_counts())


# ## <b>II <span style='color:#595959'>|</span> Create FamilyCount column </b> 
# 
# * We can create a `FamilyCount` feature which combines `Parch` (number of parents and children) and `SibSp` (number of siblings and spouses) columns. This will enable us to drop Parch and SibSp from our datasets.
# 

# In[39]:


for dataset in datasets:
    
    #create FamilyCount column.
    dataset['FamilyCount'] = dataset['SibSp'] + dataset['Parch']+1


# In[40]:


train_df['FamilyCount'].value_counts()


# ## <b>III <span style='color:#595959'>|</span> Create IsAlone column </b> 
# 
# 
# * Create a `IsAlone` feature which contain two values (0 or 1). 0 when family count is 1 means there is one alone person and 1 when family count is more than 1.

# In[41]:


for dataset in datasets:
    
    #create IsAlone column.
    dataset.loc[dataset['FamilyCount'] > 1, 'IsAlone'] = 0
    dataset.loc[dataset['FamilyCount'] == 1, 'IsAlone'] = 1   
    dataset['IsAlone'] = dataset['IsAlone'].astype(int)


# In[42]:


train_df['IsAlone'].value_counts()


# In[43]:


train_df.groupby(['IsAlone', 'Survived'])['Survived'].count()


# In[44]:


#survived passengers
survived_df= train_df[train_df['Survived']==1]

#non-survived passengers
unsurvived_df= train_df[train_df['Survived']==0]


# In[45]:


grid = sns.FacetGrid(train_df, col='Sex', aspect=1.2)
grid.map(sns.pointplot,'IsAlone', 'Survived')
grid.add_legend();


# In[46]:


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title('survived passenger ages')
plot=sns.countplot(data=survived_df, x='IsAlone', hue='Sex',palette="Set2");
add_value_labels(plot)

plt.subplot(1,2,2)
plt.title('unsurvived passenger ages')
plot=sns.countplot(data=unsurvived_df, x='IsAlone', hue='Sex',palette="Set2");
add_value_labels(plot)


# ## <b><span style='color:#595959'>|</span> Observations </b> 
# 
# * Most of alone men non-survived (About `85%`), on other hand about `72%` of alone women survived. 
# * Almost half of the passengers who are not alone survived(About `50.5%`). 
# 

# you can check this link for [basic feature engineering with the titanic data.](https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/) 
# 

# In[47]:


#drop columns
for dataset in datasets:
    dataset.drop(columns=['SibSp','Parch','Name'], inplace=True)


# In[48]:


train_df.head()


# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 4- Create Categories</mark>**
# <br>
# 
# ## <b>I <span style='color:#595959'>|</span> Age column</b> 
# Create Age bands and determine correlations with Survived.
# <br>
# 
# It is important to place attention on how you form these groups, since you don't want for example that 80% of your data falls into group 1.

# In[49]:


train_df['Bands'] = pd.cut(train_df['Age'], 5)
train_df[['Bands', 'Survived']].groupby(['Bands'], as_index=False).mean()


# In[50]:


train_df['Bands'].value_counts()


# In[51]:


for dataset in datasets:    
    
    #create categories
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_df.head()


# ## <b>II <span style='color:#595959'>|</span> Fare column</b> 
# 
# Frist we will check for missing values then we can create bands.

# In[52]:


print(train_df['Fare'].isna().sum())
print(test_df['Fare'].isna().sum())


# * there is only one missing value in test set. So, let's fill it with median.

# In[53]:


test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df.head()


# In[54]:


train_df['Bands'] = pd.cut(train_df['Fare'], 4)
train_df[['Bands', 'Survived']].groupby(['Bands'], as_index=False).mean()


# In[55]:


train_df['Bands'].value_counts(normalize=True)


# <div style="border-radius:10px;border:#404040 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# For the 'Fare' feature, we can't do the same as with the 'Age' feature because 95% of the values fall into the first category. Fortunately, we can use "qcut()" function, that we can use to see, how we can form the categories.
# </div>
# 

# In[56]:


train_df['Bands'] = pd.qcut(train_df['Fare'], 4)
train_df[['Bands', 'Survived']].groupby(['Bands'], as_index=False).mean()


# In[57]:


train_df['Bands'].value_counts(normalize=True)


# Now it looks good.

# In[58]:


for dataset in datasets:
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df.head(10)


# In[59]:


#drop Bands column
train_df.drop(columns='Bands',inplace=True)


# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 5- Encode categorical features</mark>**
# <br>
# 
#  We have three categorical features Sex, Embarked and Title. So, we should convert them ti numeric.

# In[60]:


for dataset in datasets:
    
    #encode Sex column
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
    
    #encode Embarked column
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    #encode Title column
    dataset['Title'] = dataset['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Boy": 4}).astype(int)


# In[61]:


test_df.head()


# In[62]:


train_df.head()


# In[ ]:





# # <a id="5"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>4 | Modeling </span></b> </div>
# 
# <br>
# 
# We have chosen to employ a variety of models, namely **<span style='color:#404040'>SVM</span>** , **<span style='color:#404040'>RandomForest</span>** , **<span style='color:#404040'>KNeighbors Classifier</span>** , **<span style='color:#404040'>Decision Tree Classifier</span>** , **<span style='color:#404040'>Logistic Regression</span>** and **<span style='color:#404040'>Gradient Boosting Classifier</span>**.
# 
# 
# These algorithms are known for their distinct strengths when dealing with diverse data types and structures.
# 
# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 1- Model Building</mark>**
# <br>
# 
# ## <b>I <span style='color:#595959'>|</span> Define models</b> 
# 

# In[63]:


X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()


# In[64]:


Models = [
        
        ("SVM",      SVC()),     #Support Vector Machines
               
        ("kNN",      KNeighborsClassifier(n_neighbors = 3)),    #KNeighborsClassifier

        ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),   #Logistic Regression model

        ("DT_model", DecisionTreeClassifier(random_state=42)),    #Decision tree model

        ("RF_model", RandomForestClassifier(random_state=42, n_jobs=-1)),   #Random Forest model

        ("GradientBoosting",GradientBoostingClassifier(max_depth=2,     #GradientBoosting model
                                                      n_estimators=100))
        ]


# ## <b>II <span style='color:#595959'>|</span> Train our models</b> 
# 

# In[65]:


accuracies = {}
models = {}
model = Models
for name,model in Models:
    model.fit(X_train, y_train)
    models[name] = model
    acc = model.score(X_train, y_train)*100
    accuracies[name] = acc
    print("{} Accuracy Score : {:.3f}%".format(name,acc))


# In[66]:


models_res = pd.DataFrame(data=accuracies.items())
models_res.columns = ['Model','Test score']
models_res.sort_values('Test score',ascending=False)


# ## <b>III <span style='color:#595959'>|</span> Estimate the performance of our models</b> 
# 
# 
# ### **<mark style="color:#404040;border-radius:20px;opacity:1.5;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> Cross-Validation  </mark>**
# <br>
# 
# 
# Cross-validation is used to estimate the performance of a model on unseen data. It helps assess how well the model generalizes to unseen data. The basic idea behind cross-validation is to split the available dataset into multiple subsets or folds.
# 
# <br>
# 
# <div style="border-radius:10px;border:#595959 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# This approach makes the best use of all the data we are given, so it's particularly useful when the sample size is small.</div>
# 
# <br>
# The most common type is k-fold cross-validation. In k-fold cross-validation, the data is divided into k equal-sized folds. The model is trained on k-1 folds and tested on the remaining fold.
# 
# This process is repeated k times, with each fold serving as the test set once. The performance metric is then averaged across all the folds to get an estimate of the model's performance.
# 
# ### **<mark style="color:white;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> Note that </mark>**
# 
# Calculate the value of mean and standard deviation of averaged across all the folds and the  best model that have a good mean accracy and a low value of standard deviation. 
# 
# 

# In[67]:


models_data = {'min_score':{},'max_score':{},'mean_score':{},'std_dev':{}}
for name, model in Models:   
    
    # get cross validation score for each model:
    cv_results = cross_val_score(model, X_train, y_train, 
                                 cv=5, scoring="accuracy" )
    
    # output:
    
    #min accuracy.
    min_score = round(min(cv_results)*100, 4)
    models_data['min_score'][name] = min_score
     
    #max accuracy.
    max_score = round(max(cv_results)*100, 4)
    models_data['max_score'][name] = max_score
    
    #average accuracy.
    mean_score = round(np.mean(cv_results)*100, 4)
    models_data['mean_score'][name] = mean_score
    
    #standard deviation of the data to see degree of variance in the results.
    std_dev = round(np.std(cv_results), 4)
    models_data['std_dev'][name] = std_dev


# In[68]:


models_df = pd.DataFrame(models_data)
models_df


# ## <b>IV <span style='color:#595959'>|</span> Chose the best model</b> 
# 
# We found that `decision tree` model with K-Fold Cross Validation reached a good mean accracy and a lowest value of standard deviation. The value of std is extremely low, which means that our model has a very low variance. The model performed good on all test sets. which mean that our model has no overfitting.
# 
# ## **<mark style="color:#404040;border-radius:20px;opacity:1.5;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> Decision Tree model  </mark>**
# <br>
# 
# **<span style='color:#8e9b9a'>Decision tree</span>** builds tree branches and each branch can be considered as an if-else statement. The branches develop by partitioning the dataset into subsets based on most important features. Final classification happens at the leaves of the decision tree.
# 
# ▶ **<span style='color:#8e9b9a'>decision tree common hyperparameters: </span>** max_depth, min_samples_split, min_samples_leaf; max_features.
# 
# <p style="text-align:center; ">
# <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/1*fSlQBEta5GKjNgZGsVTVTA.png" style='width: 500px; height: 300px;'>
# </p>
# 
# 
# ### **<span style='color:#8e9b9a'> Advantages of Decision Tree</span>**
# * Extremely fast at classifying unknown records.
# * Easy to interpret for small-sized trees.
# * Their accuracy is comparable to other classification techniques for many simple data sets.
# * Exclude unimportant features.
# 
# ### **<span style='color:#8e9b9a'> Disadvantages of Decision Tree</span>**
# * Easy to overfit.
# * Decision tree models are often biased toward splits on features having a large number of levels.
# * Small changes in the training data can result in large changes to decision logic.
# * Large trees can be difficult to interpret and the decisions they make may seem counter-intuitive.
# 
# [Check it out for more about Decision Tree](https://builtin.com/data-science/classification-tree).

# In[69]:


# Decision Tree
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)

acc_decision_tree = round(DT_model.score(X_train, y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")


# ## <b>V <span style='color:#595959'>|</span> Hyperparameter Tuning </b> 
# 
# how to optimize your model built using the previous section using the GridSearchCV.
# 
# ## **<mark style="color:#404040;border-radius:20px;opacity:1.5;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> Grid search  </mark>**
# <br>
# Grid search is used for hyperparameter tuning, where a predefined set of hyperparameters is explored to find the combination that yields the best performance. It is often used in conjunction with cross-validation to evaluate the performance.
# 
# let's tune our model. The first step is creating a range of hyperparameters that we want to evaluate. And Create a `GridSearchCV` model that includes your classifier and hyperparameter grid.
# 
# 
# 
# [Click here for more](https://towardsdatascience.com/tuning-the-hyperparameters-of-your-machine-learning-model-using-gridsearchcv-7fc2bb76ff27)
# 

# In[70]:


#our hyperparameter grid
params = {
    "max_depth":range(10,50,10) ,
    "min_samples_leaf" : [1, 5, 10, 25, 50, 70],
    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
    "max_features": ['auto', 'sqrt']
}
model = GridSearchCV(
    DT_model,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train,y_train)


# In[71]:


# Extract best hyperparameters
print(model.best_params_)
print("-------------")
print(model.best_score_)
print("-------------")
print(model.best_estimator_)


# In[72]:


# Random Forest
DT_model = model.best_estimator_

DT_model.fit(X_train, y_train)

acc_decision_tree = round(DT_model.score(X_train, y_train) * 100, 2)
print(round(acc_decision_tree,2), "%")


# In[73]:


# Get feature names from training data
features = X_train.columns

# Extract importances from model
importances = model.best_estimator_.feature_importances_

# Create a series with feature names and importances
feat_imp = pd.Series(importances,index=features).sort_values()

# Plot 10 most important features
feat_imp.plot(kind='bar')
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");


# # <a id="6"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>5 | Evaluate </span></b> </div>
# 
# <br>
# 
# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 1- Confusion Matrix</mark>**
# <br>
# 
# **<span style='color:#595959'>Confusion matrix</span>** is a table that summarize the performance of classification models by compared the actual values with predicted values.
# 
# **It shows:**
# 
# **<span style='color:#595959'>True Posetive(TP):</span>** correctly prediction as a positive (belong to positive class and predicted as positive)
# 
# **<span style='color:#595959'>False Positive(FP):</span>** incorrectly prediction as a positive (belong to negative class but predicted as positive)
# 
# **<span style='color:#595959'>False Negative(FN):</span>** incorrectly prediction as a negative (belong to positive class but predicted as negative)
# 
# **<span style='color:#595959'>True Negative(TN):</span>** correctly prediction as a negative (belong to negative class and predicted as negative)
# 
# <br>
# 
# <div style="border-radius:10px;border:#595959 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# note: A good model is one which has high True Posetive(TP) and True Negative(TN) rates
# </div>
# 
# <p style="text-align:center; ">
# <img src="https://www.researchgate.net/profile/Francois_Waldner/publication/340876219/figure/download/fig4/AS:883619027501056@1587682601588/Illustrative-error-matrix-Shaded-areas-represent-data-that-are-not-available-for.ppm" style='width: 400px; height: 300px;'>
# </p>
# 

# In[74]:


# Create a confusion matrix
predictions = cross_val_predict(DT_model, X_train, y_train, cv=3)

cm = confusion_matrix(y_train, predictions)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="gray_r", fmt="d", cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# A confusion matrix gives you a lot of information about how well your model does, but theres a other ways to get even more.

# In[ ]:





# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 2- Precision,Recall and F1-Score</mark>**
# <br>
# 
# **<span style='color:#595959'>Precision</span>**
# 
# Precision is a measure of how many of the true positive predictions were actually correct. 
# 
# <p class="formulaDsp">
# \[ Precision = \frac{TP}{TP + FP} \]
# </p>
# 
# **<span style='color:#595959'>Recall</span>**
# 
# Recall (or Sensitivity) is a measure of how many of the actual positive cases were identified correctly.
# 
# <p class="formulaDsp">
# \[ Recall = \frac{TP}{TP + FN} \]
# </p>
# 
# **<span style='color:#595959'>F1-Score</span>**
# 
# The F1 score is the harmonic mean of Precision and Recall and tries to find the balance between precision and recall. 
# 
# * F1-Score is important when the poth precision and recall are equally important and it helps to find a trade-off between recall and precision.
# 
# <p class="formulaDsp">
# \[ F1 Score = \frac{2 * Precision * Recall}{Precision + Recall} \]
# </p>
# 

# In[75]:


# Calculate precision, recall, and F1-score
precision = precision_score(y_train,predictions, average='weighted')
recall = recall_score(y_train,predictions, average='weighted')
f1 = f1_score(y_train,predictions, average='weighted')

# Print evaluation metrics
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1))


# * Our model predicts `80%` of the passengers survival correctly (precision). The recall tells us that it predicted the survival of `80%` of the people who actually survived.
# 
# * F1-score favors classifiers that have a similar precision and recall. So, we have about `80%` F-score. The score is fairly good.

# ## **<mark style="color:#404040;border-radius:5px;opacity:1.0;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"> 3- Auc - Roc curve and score</mark>**
# <br>
# 
# It tells how much model is capable of distinguishing between classes.
# Higher the AUC, better the model is at distinguishing between survived and not.
# 
# ▶ **<span style='color:#595959'>ROC curve:</span>** is a graph showing the performance of a classification model at all classification thresholds.
# 
# <br>
# 
# <div style="border-radius:10px;border:#595959 solid;padding: 15px;background-color:#ffffff00;font-size:100%;text-align:left">
# This curve plots two parameters:
#     True Positive Rate(TPR), False Positive Rate(FPR)
# </div>
# 
# <br>
# 
# ▶ **<span style='color:#595959'>AUC</span>**
# 
# AUC stands for `Area under the ROC Curve` That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).
# 
# AUC provides an aggregate measure of performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example.
# 
# <br>
# 
# ▶ **<span style='color:#595959'> ROC AUC Score:</span>**  is the corresponding score to the ROC AUC Curve. It simply measure the area under the curve, which is called AUC.

# In[76]:


#probabilities of our predictions
y_scores = DT_model.predict_proba(X_train) 

Roc_Auc_Score = roc_auc_score(y_train, y_scores[:,1])
print("ROC-AUC-Score:", Roc_Auc_Score)


# the score is good enough.

# In[77]:


FPR, TPR, thresholds = roc_curve(y_train, y_scores[:,1])

plt.plot(FPR, TPR)
plt.plot([0, 1], [0, 1], 'g')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Auc - Roc curve',fontsize=15);


# * Our model should be as far away from green line(purely classifier) as possible. Our model seems to be very good.
# 
# Now, I think our model is good enough to submit the predictions for the test-set.

# # <a id="7"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>6 | Submission </span></b> </div>
# 
# <br>
# 

# In[78]:


#predictions for the test-set.
Y_prediction = DT_model.predict(X_test)

#submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)


# # <a id="8"></a>
# <div style="padding: 30px;color:white;margin:10;font-size:170%;text-align:left;display:fill;border-radius:10px;background-color:#F1C40F;overflow:hidden;background-image: url(https://th.bing.com/th/id/R.3be5e39321febd4c1f758691e109c8bd?rik=lkhAzU4ulIuL0Q&riu=http%3a%2f%2fupload.wikimedia.org%2fwikipedia%2fcommons%2f5%2f56%2fRMS_Titanic_2.jpg&ehk=VbIvq0%2b63pXvH%2bl3Ln4tlz%2bwppwyoN%2fUTZDuGVYTHiQ%3d&risl=&pid=ImgRaw&r=0)"><b><span style='color:#404040'>7 | Conclusion 🎉 </span></b> </div>
# 
# <br>
# 
# Finnaly, we finish that. I really happy becouse I think this kernal is useful enough to put upvote. 😂💥

# ***
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 4.5em; font-weight: bold; font-family: Arial;">THANK YOU!</span>
# </div>/
# 
# <br>
# <br>
# 
# <div style="text-align: center;">
#     <span style="font-size: 5em;">✔️</span>
# </div>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.4em; font-weight: bold; font-family: Arial; max-width:1200px; display: inline-block;">
#        If you find this notebook useful, I really would appreciate your upvote!
#    </span>
# </div>
# 

# In[ ]:




