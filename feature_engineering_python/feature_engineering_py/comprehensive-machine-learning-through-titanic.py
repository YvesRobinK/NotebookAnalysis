#!/usr/bin/env python
# coding: utf-8

# # Titanic
# <img src="https://res.cloudinary.com/dk-find-out/image/upload/q_80,w_1920,f_auto/MA_00079563_yvu84f.jpg" style="width: 650px;"/>

# # Introduction
# 
# Titanic movie is one of the most beautiful Love story movie I have ever seen. Love affair of Jack and Rose start in Ship and they enjoy the company of each other. Jack was a poor boy while Rose belong to a rich family and engaged to some other person Caledon. The ship drowns by bumping with a Iceberg.
# * > The sinking of the Titanic is one of the most disgraceful shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This tragedy shocked the international community and led to better safety regulations for ships.
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew, Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# In this notebook, I will go through the whole process of creating a machine learning model on the famous Titanic dataset, which is used by many people all over the world. It provides information on the fate of passengers on the Titanic, summarized according to economic status (class), sex, age and survival.
# In this challenge, we are asked to predict whether a passenger on the titanic would have been survived or not.

# # Import libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # Getting the data

# In[2]:


data = pd.read_csv('../input/train.csv')
data.head(5)


# From the table above, we can note a few things. First of all, that we need to convert a lot of features into numeric ones later on, so that the machine learning algorithms can process them. Furthermore, we can see that the features have widely different ranges, that we will need to convert into roughly the same scale. We can also spot some more features, that contain missing values (NaN = not a number), that wee need to deal with.

# # Data Exploration/Analysis

# In[3]:


data.shape


# In[4]:


data.info()


# * Survived: The Survived variable is our outcome or dependent variable. It is a binary nominal datatype of 1 for survived and 0 for did not survive. All other variables are potential predictor or independent variables.
# * PassengerID and Ticket: The PassengerID and Ticket variables are assumed to be random unique identifiers, that have no impact on the outcome variable. Thus, they will be excluded from analysis.
# * Pclass: The Pclass variable is an ordinal datatype for the ticket class, a proxy for socio-economic status (SES), representing 1 = upper class, 2 = middle class, and 3 = lower class.
# * Name: The Name variable is a nominal datatype. It could be used in feature engineering to derive the gender from title, family size from surname, and SES from titles like doctor or master. Since these variables already exist, we'll make use of it to see if title, like master, makes a difference.
# * Sex and Embarked: The Sex and Embarked variables are a nominal datatype. They will be converted to dummy variables for mathematical calculations.
# * Age and Fare: The Age and Fare variable are continuous quantitative datatypes.
# * SibSp: The SibSp represents number of related siblings/spouse aboard and Parch represents number of related parents/children aboard. Both are discrete quantitative datatypes. This can be used for feature engineering to create a family size and is alone variable.
# * Cabin: The Cabin variable is a nominal datatype that can be used in feature engineering for approximate position on ship when the incident occurred. However, since there are many null values, it does not add value and thus is excluded from analysis.

# * The training-set has 891 examples and 11 features + the target variable (survived). 2 of the features are floats, 5 are integers and 5 are objects. 
# * Through this data we would like to find the effect of various factors such as age, sex, station of Embarkment,their class and no. of relatives present on survival chances of passangers.our cabin column has lots of null values.so we would not like to modify it much. there is only 2 entries in embarked column having null values,so we will replace it with mode value of point of embarktion

# In[5]:


data.describe()


# Above we can see that 38% out of the training-set survived the Titanic. We can also see that the passenger ages range from 0.4 to 80. On top of that we can already detect some features, that contain missing values, like the ‘Age’ feature.

#  total number of survived passanger 

# In[6]:


import pandas_profiling
pandas_profiling.ProfileReport(data)


# # Visulaly analyzing

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
data.hist(figsize=(12,8))
plt.figure()


# ### Survived

# In[8]:


import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col = "Survived"
grouped = data[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])
layout = {'title': 'Survived(0 = No, 1 = Yes)'}
fig = go.Figure(data = [trace], layout = layout)
iplot(fig)


# It is clear that the no of people survived is less than the number of people who died.
# > From the total passenger on the titanic 61.6% people died and 31.6% survived. If we assume all the passenger died in that infamous incident then our accuracy is about to 62%. But we have to analyse more and train our model to predict more accurate value.

# In[9]:


col = "Sex"
grouped = data[col].value_counts().reset_index()
grouped = grouped.rename(columns = {col : "count", "index" : col})

## plot
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])
layout = {'title': 'Sex(male, female)'}
fig = go.Figure(data = [trace], layout = layout)
fig.layout.template='presentation'
iplot(fig)


# 64.8% of the total passenger are male and 35.2% are female.

# In[10]:


x=data
d1=x[x['Survived']==0]
d2=x[x['Survived']==1]


# In[11]:


col='Sex'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="#6ad49b"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name="1", marker=dict())
y = [trace1, trace2]
layout={'title':"surviving rate male vs female",'xaxis':{'title':"Sex"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='plotly'
iplot(fig)


# Look at the above figure. Sex seems to be very interesting feature. The survival rate of men is much less than that of women. Only 81 women died out of 344. But 109 men survived out of 577. It means that women were given high priority while Rescue.

# In[12]:


col='Pclass'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#9467bd"))
layout={'title':"Pclass count",'xaxis':{'title':"pclass"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='presentation'
iplot(fig)


# There are 3 classes of passengers. Class1 216 passangers Class2 184 passangers and Class3 491 passangers, here we see that the no. of passangers in the class 3 is more than class 1 and class 3. 

# In[13]:


col='Pclass'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="#d62728"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name="1", marker=dict(color='#6ad49b'))
y = [trace1, trace2]
layout={'title':"surviving rate in Pclass",'xaxis':{'title':"Pclass"},'barmode': 'relative'}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)


# * That looks amazing. It is usually said that Money can’t buy Everything, But it is clearly seen that passengers of Class 1 are given high priority while Rescue. There are greater number of passengers in Class 3 than Class 1 and Class 2 but very few, almost 25% in Class 3 survived. In Class 2, survival and non-survival rate is 49% and 51% approx. While in Class 1 almost 68% people survived. So money and status matters here.
# * it is clear that women survival rate in Class 1 is about 95–96%, as only 3 out of 94 women died. So, it is now more clear that irrespective of Class, women are given first priority during Rescue. Because survival rate for men in even Class 1 is also very low. From this conclusion, PClass is also a important feature

# In[14]:


col='Embarked'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#bcbd22"))
layout={'title':"Embarked Count",'xaxis':{'title':"Embarked"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='plotly_dark'
iplot(fig)


# The 3 embarked category in the dataset, among them 168 belongs to category C, 77 belongs to category Q and 644 belongs to category s.

# In[15]:


col='Embarked'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="#bcbd22"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name="1", marker=dict(color='#8c564b'))
y = [trace1, trace2]
layout={'title':"surviving rate in Embarked",'xaxis':{'title':"Embarked"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)


# Embarked seems to be correlated with survival, depending on the gender.
# * Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C. Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S

# In[16]:


col='SibSp'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#e377c2"))
layout={'title':"siblings/spouse Count",'xaxis':{'title':"SibSp"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='presentation'
iplot(fig)


# SibSp and Parch would make more sense as a combined feature, that shows the total number of relatives, a person has on the Titanic. I will create it above and also a feature that sows if someone is not alone

# In[17]:


col='SibSp'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="#17becf"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name="1", marker=dict(color='#ff7f0e'))
y = [trace1, trace2]
layout={'title':"surviving rate in SibSp",'xaxis':{'title':"SibSp"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)


# There are many interesting facts with this feature. Above plot shows that if a passanger is alone in ship with no siblings, survival rate is 34.5%. The graph decreases as no of siblings increase. This is interesting because, If I have a family onboard, I will save them instead of saving myself. But there’s something wrong, the survival rate for families with 5–8 members is 0%. Is this because of PClass? Yes this is PClass, The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.

# In[18]:


col='Age'
v1=x[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="0", marker=dict(color="rgb(63, 72, 204)"),text= data.Name)
y = [trace1]
layout={'title':"Age count with name",'xaxis':{'title':"Age"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)


# In[19]:


print('Oldest Passenger was of:',data['Age'].max(),'Years')
print('Youngest Passenger was of:',data['Age'].min(),'Years')
print('Average Age on the ship:',data['Age'].mean(),'Years')


# In[20]:


col='Age'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="0", marker=dict(color="#d62728"),text= data.Name)
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="1", marker=dict(color='#bcbd22'),text= data.Name)
y = [trace1, trace2]
layout={'title':"surviving rate on the basic of Age with their names",'xaxis':{'title':"Age"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='plotly_dark'
iplot(fig)


# From the above plots, I found the following observations
# (1) First priority during Rescue is given to children and women, as the persons<5 are save by large numbers
# (2) The oldest saved passenger is of 80
# (3) The most deaths were between 30–40
# >  The no of children is increasing from Class 1 to 3, the number of children in Class 3 is greater than other two. 2) Survival rate of children, for age 10 and below is good irrespective of Class 3) Survival rate between age 20–30 is well and is quite better for women.

# In[21]:


col='Parch'
v2=x[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v2[col], y=v2["count"], name="Emb",  marker=dict(color="#a678de"))
layout={'title':"Parch Count",'xaxis':{'title':"Parch"}}
fig = go.Figure(data=[trace1], layout=layout)
fig.layout.template='presentation'
iplot(fig)


# Whether a passenger is alone or have family, from the plot we see that most of the person are alone and there's a family which have 6 members.

# In[22]:


col='Parch'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Bar(x=v1[col], y=v1["count"], name="0", marker=dict(color="#17becf"))
trace2 = go.Bar(x=v2[col], y=v2["count"], name="1", marker=dict(color='rgb(63, 72, 204)'))
y = [trace1, trace2]
layout={'title':"surviving rate on the basic of Parch",'xaxis':{'title':"Parch"},'barmode': 'relative'}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='presentation'
iplot(fig)


# In[23]:


col='Fare'
v1=x[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="fare", marker=dict(color="#9467bd"))
y = [trace1]
layout={'title':"Farecount",'xaxis':{'title':"Fare"}}
fig = go.Figure(data=y, layout=layout)
iplot(fig)


# In[24]:


col='Fare'
v1=d1[col].value_counts().reset_index()
v1=v1.rename(columns={col:'count','index':col})
v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
v1=v1.sort_values(col)
v2=d2[col].value_counts().reset_index()
v2=v2.rename(columns={col:'count','index':col})
v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
v2=v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name="0", marker=dict(color="#17becf"),text= data.Name)
trace2 = go.Scatter(x=v2[col], y=v2["count"], name="1", marker=dict(color='#bcbd22'),text= data.Name)
y = [trace1, trace2]
layout={'title':"surviving rate on the basic of fare with their names",'xaxis':{'title':"Fare"}}
fig = go.Figure(data=y, layout=layout)
fig.layout.template='plotly_dark'
iplot(fig)


# In[25]:


plt.style.use('fivethirtyeight')
grid = sns.FacetGrid(data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', bins=20)


# In[26]:


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

# specify hue="categorical_variable"
sns.boxplot(y='Age', x='Survived', hue="Pclass", data=data)
plt.show()


# > we see that more no. of passangers survived in upper classes and also children were almost certain to survive if they belonged to higher classes.

# ### Age Histogram based on Gender and Survived

# In[27]:


grid = sns.FacetGrid(data, col='Survived', row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', bins=15)


# In[28]:


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

# specify hue="categorical_variable"
sns.boxplot(y='Age', x='Survived', hue="Sex", data=data)
plt.show()


# ### Age histogram based on Embarked,Survived

# In[29]:


grid = sns.FacetGrid(data, col='Survived', row='Embarked', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', bins=15)


# In[30]:


plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

# specify hue="categorical_variable"
sns.boxplot(y='Age', x='Survived', hue="Embarked", data=data)
plt.show()


# #### >> we can clearly see that women had very higher survival rate.

# ### Age violinplot based on Pclass,Survived
# 

# In[31]:


f,ax=plt.subplots(1,2,figsize=(12,6))
sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


#  **>>** You can see that men have a high probability of survival when they are between 18 and 30 years old, which is also a little bit true for women but not fully. For women the survival chances are higher between 14 and 40.
#  
#  **>>** For men the probability of survival is very low between the age of 5 and 18, but that isn’t true for women. Another thing to note is that infants also have a little bit higher probability of survival.

# **>>** For males, the survival chances decreases with an increase in age.Survival chances for Womens Passenegers aged 20-50 from Pclass1 is high,As we had seen earlier, the Age feature has 177 null values.

# In[32]:


f,ax=plt.subplots(2,2,figsize=(12,6))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# Maximum passenegers boarded from S. Majority of them being from Pclass3.The Passengers from C look to be lucky as a good proportion of them survived.Port Q had almost 95% of the passengers were from Pclass3, passengers from Pclass3 around 81% didn't survive.

# Embarked seems to be correlated with survival, depending on the gender.
# Women on port Q and on port S have a higher chance of survival. The inverse is true, if they are at port C. Men have a high survival probability if they are on port C, but a low probability if they are on port Q or S.
# Pclass also seems to be correlated with survival. We will generate another plot of it below.

# Here we see clearly, that Pclass is contributing to a persons chance of survival, especially if this person is in class 1. 

# # Correlation
# It is used to describe the linear relationship between two continuous variables (e.g., height and weight). In general, correlation tends to be used when there is no identified response variable. It measures the strength (qualitatively) and direction of the linear relationship between two or more variables.

# In[33]:


data.corr()


# ## Let's see feature corelation

# In[34]:


data.groupby('Pclass',as_index=False)['Survived'].mean()


# there seems to have little corelation among survived and Pclass.Higher class passangers are more likely to survive.

# In[35]:


data.groupby('Sex',as_index=False)['Survived'].mean()


# that's pretting amazing correlation. females are 4 times more likely to survive

# In[36]:


data.groupby('Embarked',as_index=False)['Survived'].mean()


# In[37]:


data.groupby('SibSp',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)


# Having more sibling can be corelated to less survival rate

# In[38]:


data.groupby('Parch',as_index=False)['Survived'].mean().sort_values(by='Survived',ascending=False)


# Persons with more parents or children seem to have low chances.Let's divide our age data into bands and let's see if there is some corelation.

# In[39]:


fig,ax = plt.subplots(figsize=(8,7))
ax = sns.heatmap(data.corr(), annot=True,linewidths=.5,fmt='.1f')
plt.show()


# # Data cleaning

# In[40]:


data['NewSibSp'] = 5 # let something more than 5 be 5 (others)

data.loc[(data.SibSp.values == 0),'NewSibSp']= 0
data.loc[(data.SibSp.values == 1),'NewSibSp']= 1
data.loc[(data.SibSp.values == 2),'NewSibSp']= 2
data.loc[(data.SibSp.values == 3),'NewSibSp']= 3
data.loc[(data.SibSp.values == 4),'NewSibSp']= 4


# In[41]:


data['NewParch'] = 3 # let something more than 3 be 3 (others)

data.loc[(data.Parch.values == 0),'NewParch']= 0
data.loc[(data.Parch.values == 1),'NewParch']= 1
data.loc[(data.Parch.values == 2),'NewParch']= 2


# In[42]:


data['Agroup'] = 1

data.loc[(data.Age.values < 24.0),'Agroup']= 0
data.loc[(data.Age.values > 30.0),'Agroup']= 2

data.head()


# In[43]:


for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')

data.head()


# In[44]:


def survpct(a):
  return data.groupby(a).Survived.mean()

survpct('Initial')


# In[45]:


total=data.isnull().sum()
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()


# In[46]:


data['Age'] = data.groupby('Initial')['Age'].apply(lambda x: x.fillna(x.mean()))
data.head()


# In[47]:


data['Fare']=np.log1p(data["Fare"])


# In[48]:


data.drop(['PassengerId','Name','Age','Parch','Ticket','Cabin'],axis=1,inplace=True)


# # Creating Dummy Variables

# In[49]:


data=pd.get_dummies(data)
data.head()


# In[50]:


data.shape


# # Building Machine Learning Models &  Train Data
# Now we will train several Machine Learning models and compare their results. Note that because the dataset does not provide labels for their testing-set, we need to use the predictions on the training set to compare the algorithms with each other. Later on, we will use cross validation.

# In[51]:


X = data.drop(['Survived'], axis = 1)
y = data.Survived.values


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.svm import SVC


# In[54]:


#LogisticRegression
lr_c=LogisticRegression(random_state=0)
lr_c.fit(X_train,y_train)
lr_pred=lr_c.predict(X_test)
lr_cm=confusion_matrix(y_test,lr_pred)
lr_ac=accuracy_score(y_test, lr_pred)

#SVM classifier
svc_c=SVC(kernel='linear',random_state=0)
svc_c.fit(X_train,y_train)
svc_pred=svc_c.predict(X_test)
sv_cm=confusion_matrix(y_test,svc_pred)
sv_ac=accuracy_score(y_test, svc_pred)

#SVM regressor
svc_r=SVC(kernel='rbf')
svc_r.fit(X_train,y_train)
svr_pred=svc_r.predict(X_test)
svr_cm=confusion_matrix(y_test,svr_pred)
svr_ac=accuracy_score(y_test, svr_pred)

#RandomForest
rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
rdf_c.fit(X_train,y_train)
rdf_pred=rdf_c.predict(X_test)
rdf_cm=confusion_matrix(y_test,rdf_pred)
rdf_ac=accuracy_score(rdf_pred,y_test)

# DecisionTree Classifier
dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)
dtree_c.fit(X_train,y_train)
dtree_pred=dtree_c.predict(X_test)
dtree_cm=confusion_matrix(y_test,dtree_pred)
dtree_ac=accuracy_score(dtree_pred,y_test)

#KNN
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
knn_cm=confusion_matrix(y_test,knn_pred)
knn_ac=accuracy_score(knn_pred,y_test)


# In[55]:


print('LogisticRegression_accuracy:\t',lr_ac)
print('SVM_regressor_accuracy:\t\t',svr_ac)
print('RandomForest_accuracy:\t\t',rdf_ac)
print('DecisionTree_accuracy:\t\t',dtree_ac)
print('KNN_accuracy:\t\t\t',knn_ac)
print('SVM_classifier_accuracy:\t',sv_ac)


# # What is Support Vector Machine?
# The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.
# Support Vector Machine (SVM) is primarily a classier method that performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels. SVM supports both regression and classification tasks and can handle multiple continuous and categorical variables. For categorical variables a dummy variable is created with case values as either 0 or 1.

# # How does SVM work?
# The basics of Support Vector Machines and how it works are best understood with a simple example. Let’s imagine we have two tags: red and blue, and our data has two features: x and y. We want a classifier that, given a pair of (x,y) coordinates, outputs if it’s either red or blue. We plot our already labeled training data on a plane:

# <img src="https://monkeylearn.com/blog/wp-content/uploads/2017/06/plot_original.png" style="width: 350px;"/>

# A support vector machine takes these data points and outputs the hyperplane (which in two dimensions it’s simply a line) that best separates the tags. This line is the decision boundary: anything that falls to one side of it we will classify as blue, and anything that falls to the other as red.

# <img src="https://monkeylearn.com/blog/wp-content/uploads/2017/06/plot_hyperplanes_2.png" style="width: 350px;"/>

# But, what exactly is the best hyperplane? For SVM, it’s the one that maximizes the margins from both tags. In other words: the hyperplane (remember it’s a line in this case) whose distance to the nearest element of each tag is the largest.

# <img src="https://monkeylearn.com/blog/wp-content/uploads/2017/06/plot_hyperplanes_annotated.png" style="width: 350px;"/>

# And that’s the basics of Support Vector Machines!
# 1.A support vector machine allows you to classify data that’s linearly separable.
# 2.If it isn’t linearly separable, you can use the kernel trick to make it work.
# However, for text classification it’s better to just stick to a linear kernel.
# Compared to newer algorithms like neural networks, they have two main advantages: higher speed and better performance with a limited number of samples (in the thousands). This makes the algorithm very suitable for text classification problems, where it’s common to have access to a dataset of at most a couple of thousands of tagged samples.

# In[56]:


plt.figure(figsize=(20,10))
plt.subplot(2,3,1)
plt.title("LogisticRegression_cm")
sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,3,2)
plt.title("SVM_regressor_cm")
sns.heatmap(sv_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,3,3)
plt.title("RandomForest")
sns.heatmap(rdf_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)
plt.subplot(2,3,4)
plt.title("SVM_classifier_cm")
sns.heatmap(svr_cm,annot=True,cmap="Reds",fmt="d",cbar=False)
plt.subplot(2,3,5)
plt.title("DecisionTree_cm")
sns.heatmap(dtree_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,3,6)
plt.title("kNN_cm")
sns.heatmap(knn_cm,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.show()


# ### Confusion matrix
# The first row is about the not-survived-predictions: 364 passengers were correctly classified as not survived (called true negatives) and 60 where wrongly classified as not survived (false positives).
# 
# The second row is about the survived-predictions: 96passengers where wrongly classified as survived (false negatives) and 192re correctly classified as survived (true positives).
# 
# A confusion matrix gives you a lot of information about how well your model does, but theres a way to get even more, like computing the classifiers precision.

# ### Plotting the Accuracy of the models
# Here we plot the performance or the accuracy of the different machine learning model, in this plot we observe that the different models have diffrent performence.

# In[57]:


model_accuracy = pd.Series(data=[lr_ac,sv_ac,svr_ac,rdf_ac,dtree_ac,knn_ac], 
                index=['LogisticRegression','SVM_classifier','SVM_regressor',
                                      'RandomForest','DecisionTree_Classifier','KNN'])
fig= plt.figure(figsize=(10,6))
model_accuracy.sort_values().plot.barh()
plt.title('Model Accracy')


# # Summary
# We started with the data exploration where we got a feeling for the dataset, checked about missing data and learned which features are important. During this process we used seaborn and matplotlib to do the visualizations. During the data preprocessing part, we computed missing values, converted features into numeric ones, grouped values into categories and created a few new features. Afterwards we started training  machine learning models, and applied cross validation on it. 
# Of course there is still room for improvement, like doing a more extensive feature engineering, by comparing and plotting the features against each other and identifying and removing the noisy features. Another thing that can improve the overall result on the kaggle leaderboard would be a more extensive hyperparameter tuning on several machine learning models. You could also do some ensemble learning.Lastly, we looked at it’s confusion matrix and computed the models precision.

# # I hope this kernel is helpfull for you  --> upvote will appreciate me for further work.
# 
