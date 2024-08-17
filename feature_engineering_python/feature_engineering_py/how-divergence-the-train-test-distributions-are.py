#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The divergence between train and test data is very important, because if there is high divergnce between them, our data engineering and model evaluations can not perform good enough. 
# 
# In this notebook, I'll try to measure the divergency between train/test data of Titanic and hopefully make some insights about the dataset.

# In[1]:


import numpy as np 
import pandas as pd
from math import log2
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train = pd.read_csv(r'../input/titanic/train.csv')
test =  pd.read_csv(r'../input/titanic/test.csv')
All = pd.concat([train, test], sort=True).reset_index(drop=True)
All.head()


# # Data Preparation

# Here the data will prepare to a numerical and workable format. I used the same process and code as in this notebook:
# 
# https://www.kaggle.com/code/khashayarrahimi94/knn-xgboost-svc-ensemble-with-just-5-feature

# In[3]:


All["Cabin_dumb"]=All["Cabin"]
for i in range(All.shape[0]):
    if pd.isnull(All["Cabin"][i])== False:
        All["Cabin_dumb"][i] = All["Cabin"][i][0]
    else:
        All["Cabin_dumb"][i] =0
All["Cabin_dumb"]


# In[4]:


All['Family'] = All['SibSp']+All['Parch']


# In[5]:


from sklearn.feature_selection import mutual_info_classif as MIC
mi_score = MIC(train.loc[: , ['Age' ,'Pclass','Parch','Fare','SibSp' ]].values.astype('int'),
               train.loc[: , ['Age']].values.astype('int').reshape(-1, 1))
Feature2 = ['Age' ,'Pclass','Parch','Fare','SibSp' ]
Mutual_Information_table = pd.DataFrame(columns=['Feature1', 'Feature2', 'MIC'], index=range(5))
Mutual_Information_table['Feature1'] = 'Age'
for feature in range(5):
    Mutual_Information_table['Feature2'][feature] = Feature2[feature]
for value in range(5):
    Mutual_Information_table['MIC'][value] = mi_score[value]
Mutual_Information_table


# In[6]:


age_by_pclass_sex = round(All.groupby(['Sex', 'Pclass']).median()['Age'])

for pclass in range(1, 4):
    for sex in ['female', 'male']:
        print('Mean age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))
print('Mean age of all passengers: {}'.format(round(All['Age'].mean())))

# Filling the missing values in Age with the medians of Sex and Pclass groups
All['Age'] = All.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(round(x.median())))


# In[7]:


All[All['Embarked'].isnull()]
All['Embarked'] = All['Embarked'].fillna('S')


# In[8]:


All[All['Fare'].isnull()]
mean_fare = All.groupby(['Pclass', 'Parch', 'SibSp']).Fare.mean()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
All['Fare'] = All['Fare'].fillna(mean_fare)


# In[9]:


All['Ticket_Frequency'] = All.groupby('Ticket')['Ticket'].transform('count')
freq = All.head(891)['Ticket_Frequency'].value_counts().tolist()
Ticket_freq = All.head(891)['Ticket_Frequency'].unique().tolist()

death = []
for n in Ticket_freq:
    k = 0
    for i in range(891):
        if (All.head(891)['Ticket_Frequency'][i] == n) & (All.head(891)['Survived'][i] == 0):
            k = k+1
    death.append(k)    
     
survive_rate = []
for j,w in zip(death,freq):
    rate = (w-j)/w
    survive_rate.append(rate)

Survive_rate_index = {}
for u,r in zip(Ticket_freq,survive_rate):
    Survive_rate_index[u] = r
Survive_rate_index


# In[10]:


new_ticket_freq = []
for i in range(All.shape[0]):
    new_ticket_freq.append(Survive_rate_index[All['Ticket_Frequency'][i]])
new_ticket_freq = pd.DataFrame(new_ticket_freq)
All['Ticket_Frequency'] = pd.DataFrame(new_ticket_freq)


# In[11]:


for name in All["Name"]:
    All["Title"] = All["Name"].str.extract("([A-Za-z]+)\.",expand=True)

title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Dona": "Other"
                     ,"Dr":"Other","Rev":"Other", "Mrs":"Woman","Ms":"Woman","Miss":"Woman"}

All.replace({"Title": title_replacements}, inplace=True)


# In[12]:


All = pd.get_dummies(All, columns=['Embarked','Pclass', 'Sex','Cabin_dumb','Title','Family'],
                     prefix=['Embarked','Pclass', 'Sex','Cabin_dumb','Title','Family'])
All.head()


# In[13]:


All.drop(['Ticket','Cabin','Name','PassengerId'], axis=1, inplace=True)


# # Train and Test Distribution

# Here we have clean and prepare train and test dataset and we define some functions that compute probability distributions of each columns and for train and test pat of datatest and plot them, which enable us to compare these distributions and thei divergencies. 

# In[14]:


train_cleaned = All.head(891)
test_cleaned = All.tail(418)


# In[15]:


def Prob_train(feature):
    l = []
    for j in train_cleaned[train_cleaned.columns[feature]].unique().tolist():
        l.append((train_cleaned[train_cleaned.columns[feature]].value_counts()[j])/train_cleaned.shape[0])
    return l

def Prob_test(feature):
    l = []
    for j in test_cleaned[test_cleaned.columns[feature]].unique().tolist():
        l.append((test_cleaned[test_cleaned.columns[feature]].value_counts()[j])/test_cleaned.shape[0])
    return l
    
def train_distribution(feature):
    Feature_prob = {}
    for u,r in zip(train_cleaned[train_cleaned.columns[feature]].unique().tolist(),Prob_train(feature)):
        Feature_prob[u] = r
    return Feature_prob
        
def test_distribution(feature):
    Feature_prob = {}
    for u,r in zip(test_cleaned[test_cleaned.columns[feature]].unique().tolist(),Prob_test(feature) ):
        Feature_prob[u] = r
    return Feature_prob

def train_events(feature):
    return train_cleaned[train_cleaned.columns[feature]].unique().tolist()

def test_events(feature):
    return test_cleaned[test_cleaned.columns[feature]].unique().tolist()


# In[16]:


def Prob_train_normal(feature):
    l = []
    for j in All[All.columns[1]].unique().tolist():
        if list(set(train_cleaned[train_cleaned.columns[feature]].tolist())).__contains__(j) == False:
            l.append(0.000001)
        else:
            l.append((train_cleaned[train_cleaned.columns[feature]].value_counts()[j])/train_cleaned.shape[0])
    return l

def Prob_test_normal(feature):
    l = []
    for j in All[All.columns[1]].unique().tolist():
        if list(set(test_cleaned[test_cleaned.columns[feature]].tolist())).__contains__(j) == False:
            l.append(0.000001)
        else:
            l.append((test_cleaned[test_cleaned.columns[feature]].value_counts()[j])/test_cleaned.shape[0])
    return l


# In[17]:


def distribution_plot(feature):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 2))
    font = {'family':'serif','color':'blue','size':15}
    axes[0].bar(train_events(feature), Prob_train(feature))
    axes[0].set(xlabel=train_cleaned.columns[feature], ylabel='Probability')
    axes[0].set_title("Train",fontdict = font)

    axes[1].bar(test_events(feature), Prob_test(feature))
    axes[1].set(xlabel=test_cleaned.columns[feature], ylabel='Probability')
    axes[1].set_title("Test",fontdict = font)
    return fig.tight_layout() 


# In[18]:


import matplotlib.pyplot as plt
for i in range(All.shape[1]):
    if i !=4:
        distribution_plot(i)


# ## Kullback-Leibler Divergence

# 

# In mathematical statistics, the Kullback–Leibler divergence, ${\displaystyle D_{\text{KL}}(P\parallel Q)}$ (also called relative entropy), is a statistical distance: a measure of how one probability distribution Q is different from a second, reference probability distribution P.
# 
# For discrete probability distributions ${\displaystyle P}$ and ${\displaystyle Q}$ defined on the same probability space, ${\displaystyle {\mathcal {X}}}$, the relative entropy from ${\displaystyle Q}$ to ${\displaystyle P}$ is defined to be:
# 
#  ${\displaystyle D_{\text{KL}}(P\parallel Q)=\sum _{x\in {\mathcal {X}}}P(x)\log \left({\frac {P(x)}{Q(x)}}\right)}$
#  
#  For distributions ${\displaystyle P}$ and ${\displaystyle Q}$ of a continuous random variable, relative entropy is defined to be the integral:
# 
#  ${\displaystyle D_{\text{KL}}(P\parallel Q)=\int _{-\infty }^{\infty }p(x)\log \left({\frac {p(x)}{q(x)}}\right)\,dx}$
#  
# *Wikipedia*

# In[19]:


def kl_divergence(p, q):
    return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))


# In[20]:


KL_divergences = []
for i in range(All.shape[1]):
    KL_divergences.append(kl_divergence(Prob_train_normal(i),Prob_test_normal(i)))
    print('KL(Prob_train || Prob_test) for %s is: %.3f bits' % (All.columns[i],kl_divergence(Prob_train_normal(i),Prob_test_normal(i))))


# In[21]:


plt.figure(figsize=(16, 5))
plt.plot(All.columns.tolist(), KL_divergences)
plt.xlabel("Feature")
plt.ylabel("KL")
plt.xticks(All.columns.tolist(),rotation=90)
plt.show()


# ## Jensen–Shannon Divergence

# Jensen–Shannon divergence is another method of measuring the similarity between two probability distributions.
# 
# 
# ${\displaystyle {\rm {JSD}}(P\parallel Q)={\frac {1}{2}}D(P\parallel M)+{\frac {1}{2}}D(Q\parallel M)}$
# 
# *Wikipedia*

# In[22]:


js_divergences = []
for i in range(All.shape[1]):
    js_divergences.append(distance.jensenshannon(Prob_train_normal(i),Prob_test_normal(i)))
    print('JSD(Prob_train || Prob_test) for %s is: %.5f bits' % (All.columns[i],distance.jensenshannon(Prob_train_normal(i),
                                                                        Prob_test_normal(i))))


# In[23]:


plt.figure(figsize=(16, 5))
plt.plot(All.columns.tolist(), js_divergences)
plt.xlabel("Feature")
plt.ylabel("JSD")
plt.xticks(rotation=90)
plt.show()


# # Result
# 
# 

# As we can see in plots, the divergence between train and test dataset for each column is very low. Therefore, our models and EDA should work well and not harm from divergence distributions.
# A little considerations for columns "Age" and "Fare" is needed, that I will explain them soon.
# 
# **High divergence in column "Survived" is obviousely because of its unknown values in test dataset.
