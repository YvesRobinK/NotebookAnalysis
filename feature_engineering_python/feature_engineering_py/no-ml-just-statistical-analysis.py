#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Briefly, in this notebook I try to solve the Spaceship Titanic competition JUST by statistical methods and probability and not Machine Learning.
# 
# ## Statistical Analysis
# You may hear Bayesian Inference or probabily work with it. Here I used this method but in a slightly different way;
# Defining Bayesian Score instead of Bayesian Probability.
# 
# ## Bayes Theorem
# This theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
# ### Simple Form:
# ${\displaystyle P(A\mid B)={\frac {P(B\mid A)P(A)}{P(B)}}}$
# ### Extended Form:
# Often, for some partition ${A_j}$ of the sample space, the event space is given in terms of $P(A_j)$ and $P(B | A_j)$. It is then useful to compute P(B) using the law of total probability:
# 
# ${\displaystyle P(B)={\sum _{j}P(B|A_{j})P(A_{j})}}$
# 
# Or equivalently:
# ${\displaystyle P(B)={\sum _{j}P(B 	\cap A_{j})}}$
# 
# ## Bayesian Score
# As in our case the event we want to predict (Transported) is not completely consists of the dataset's features, the bayesian theorem assumptions doesn't satisfy. Therefore, I use the idea behind this great theorem and define a new random variable, named Bayesian Score.
# 
# To understand the restriction of using this theorem, consider the $Ω$ as sample space, then we have:
# 
# $Columns = \big \{PassengerId, HomePlanet, CryoSleep, Cabin, Destination, Age,VIP, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck,Name\big \}$
# 
# 
# ${\displaystyle Ω \neq \bigcup_{\alpha \in Columns} \alpha }$
# 
# 
# Therefore:
# 
# ${\displaystyle P(Transported=0 \,or\, 1) \neq {\sum _{\alpha \in Columns}P(Transported \cap \alpha)}}$
# 
# ### BS: Bayesian Score
# 
# ${\displaystyle BS(B)={\prod _{j}P(B|A_{j})P(A_{j})}}$
# 
# Or equivalently:
# ${\displaystyle BS(B)={\prod _{j}P(B 	\cap A_{j})}}$
# 
# ### Finally
# 
# ${\displaystyle BS\big (Transported = 0 \,or\, 1 \big )={\prod _{\alpha \in Columns}P \big ( (Transported = 0 \,or\, 1) \cap \alpha \big)}}$
# 
# ## Advantage
# 
# As you can see in the following, one of the benefit of this type of statistical analysis is that, **You don't need to fill missing values** in the data which is almost one of the most time consuming and chanllenging task in data analysis.

# In[1]:


import numpy as np 
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re
import math
import collections


# In[2]:


train = pd.read_csv(r'../input/spaceship-titanic/train.csv')
test = pd.read_csv(r'../input/spaceship-titanic/test.csv')
submission = pd.read_csv(r'../input/spaceship-titanic/sample_submission.csv')
All = pd.concat([train, test], sort=False).reset_index(drop=True)
All


# # Data Preparation
# 
# In this section I prepare the data including extract meaningful part from some of columns, seperate them, encoding categorical features to numeric and so on.

# In[3]:


def family(data):
    
    data['Family'] = train['Name']

    for i in range(data.shape[0]):
        if (data['Name'].isnull()[i] ==False):
            data['Family'][i] = data['Name'][i].split(' ')[1]
    return data
family(All)


# In[4]:


def passengerID(data):
    
    data['family_id'] = data['PassengerId']
    data['second_id'] = data['PassengerId']
    for i in range(data.shape[0]):
        
        data['family_id'][i] = data['PassengerId'][i][:4]
        data['second_id'][i] = data['PassengerId'][i][5:]
    return data

passengerID(All)


# In[5]:


def separate_cabin(data):
    data['Cabin_deck'] = data['Cabin']
    #data['Cabin_num'] = data['Cabin']
    data['Cabin_side'] = data['Cabin']
    for i in range(data.shape[0]):

        if data['Cabin'].isnull()[i] ==False:

            data['Cabin_deck'][i] = data['Cabin'][i][0]
     #       data['Cabin_num'][i] = int(re.findall(r'\d+', data['Cabin'][i])[0])
            data['Cabin_side'][i] = data['Cabin'][i][-1]

    data.drop(['Cabin'],axis=1,inplace=True)
    return data
separate_cabin(All)


# In[6]:


All['HomePlanet'].replace(['Earth', 'Europa','Mars'], [0, 1,2], inplace=True)
All['CryoSleep'].replace([False, True], [0, 1], inplace=True)                       
All['Destination'].replace(['TRAPPIST-1e', '55 Cancri e','PSO J318.5-22'], [0, 1,2], inplace=True)
All['VIP'].replace([False, True], [0, 1], inplace=True)
All['Transported'].replace([False, True], [0, 1], inplace=True)
All['Cabin_deck'].replace(['F', 'G','E','B','C','D','A','T'], [0, 1,2,3,4,5,6,7], inplace=True)
All['Cabin_side'].replace(['S', 'P'], [0, 1], inplace=True)
All


# # Numeric Cols Peparation
# 
# Due to large number of unique values of some columns like $\big \{VRDeck,Spa,FoodCourt,RoomService,ShoppingMall \big \}$ we need to partitioning them to smaller groups. I did this by exploring each of this columns and their relation to the target (Transported).
# In other words, for better partitioning, we need to group each interval of values which have less diversity in order of their Transported values.
# As an example if the label values of an interal has $n$ False and $0$ True and another interval has $\dfrac {m}{2}$ False and $\dfrac {m}{2}$ True, the former is the better choice, beacuse it separates the feature to more distinct part and therefore high variance probability. 
# 

# In[7]:


sns.relplot(data=All.head(train.shape[0]), kind="line",x="Age", y="Transported")


# In[8]:


train_1 = All.head(train.shape[0])
VR_Tran = []
for i in range(train_1.shape[0]):
    
    if train_1['VRDeck'][i] > 4000:
        VR_Tran.append(train_1["Transported"][i])
print("Transported=False:",VR_Tran.count(0)/len(VR_Tran))        


# In[9]:


for i in range(All.shape[0]):
    
    if All['VRDeck'][i] > 1000:
        All['VRDeck'][i]=1
    elif All['VRDeck'][i] > 100:
        All['VRDeck'][i]=2
    elif All['VRDeck'][i] > 50:
        All['VRDeck'][i]=3
    elif All['VRDeck'][i] > 10:
        All['VRDeck'][i]=4
    elif All['VRDeck'][i] > 0:
        All['VRDeck'][i]=5
    elif All['VRDeck'][i] ==0:
        All['VRDeck'][i]=6


# In[10]:


train_1 = All.head(train.shape[0])
Spa_Tran = []
for i in range(train_1.shape[0]):
    
    if train_1['Spa'][i] > 3000:
        Spa_Tran.append(train_1["Transported"][i])
print("Transported=False:",Spa_Tran.count(0)/len(Spa_Tran))        


# In[11]:


for i in range(All.shape[0]):
    
    if All['Spa'][i] > 1500:
        All['Spa'][i]=1
    elif All['Spa'][i] > 1000:
        All['Spa'][i]=2
    elif All['Spa'][i] > 500:
        All['Spa'][i]=3
    elif All['Spa'][i] > 100:
        All['Spa'][i]=4
    elif All['Spa'][i] > 10:
        All['Spa'][i]=5
    elif All['Spa'][i] > 1:
        All['Spa'][i]=6
    elif All['Spa'][i] ==1:
        All['Spa'][i]=7
    elif All['Spa'][i] ==0:
        All['Spa'][i]=8


# In[12]:


Food_Tran = []
for i in range(train_1.shape[0]):
    
    if train_1['FoodCourt'][i] > 17000:
        
        Food_Tran.append(train_1["Transported"][i])
        
print("Transported=False:",Food_Tran.count(1)/len(Food_Tran))   


# In[13]:


for i in range(All.shape[0]):
    
    if All['FoodCourt'][i] > 15000:
        All['FoodCourt'][i]=1
    elif All['FoodCourt'][i] > 8760:
        All['FoodCourt'][i]=2
    elif All['FoodCourt'][i] > 1500:
        All['FoodCourt'][i]=3
    elif All['FoodCourt'][i] > 500:
        All['FoodCourt'][i]=4
    elif All['FoodCourt'][i] > 100:
        All['FoodCourt'][i]=5
    elif All['FoodCourt'][i] > 10:
        All['FoodCourt'][i]=6
    elif All['FoodCourt'][i] > 0:
        All['FoodCourt'][i]=7
    elif All['FoodCourt'][i]==0:
        All['FoodCourt'][i]=8


# In[14]:


Room_Tran = []
for i in range(train_1.shape[0]):
    
    if train_1['RoomService'][i] > 3682:
        Room_Tran.append(train_1["Transported"][i])
print("Transported=False:",Room_Tran.count(0)/len(Room_Tran))     


# In[15]:


for i in range(All.shape[0]):
    
    if All['RoomService'][i] > 1200:
        All['RoomService'][i]=1
    elif All['RoomService'][i] > 500:
        All['RoomService'][i]=2
    elif All['RoomService'][i] > 50:
        All['RoomService'][i]=3
    elif All['RoomService'][i] > 10:
        All['RoomService'][i]=4
    elif All['RoomService'][i] > 0:
        All['RoomService'][i]=5
    elif All['RoomService'][i] == 0:
        All['RoomService'][i]=6


# In[16]:


Shop_Tran = []
for i in range(train_1.shape[0]):
    
    if train_1['ShoppingMall'][i] > 5000:
        
        Shop_Tran.append(train_1["Transported"][i])
        
print("Transported=True:",Shop_Tran.count(1)/len(Shop_Tran))   


# In[17]:


for i in range(All.shape[0]):
    
    if All['ShoppingMall'][i] > 2000:
        All['ShoppingMall'][i]=1
    elif All['ShoppingMall'][i] > 1000:
        All['ShoppingMall'][i]=2
    elif All['ShoppingMall'][i] > 500:
        All['ShoppingMall'][i]=3
    elif All['ShoppingMall'][i] > 0:
        All['ShoppingMall'][i]=4
    elif All['ShoppingMall'][i] == 0:
        All['ShoppingMall'][i]=5


# In[18]:


for i in range(All.shape[0]):
    
    try:
         All['Family'][i] = All['Family'].value_counts()[All['Family'][i]]
    except Exception:
        pass
All 


# In[19]:


Label = All['Transported']
All.drop(['family_id','Transported','Name','PassengerId'],axis=1,inplace=True)

All['second_id'] = All['second_id'].tolist()
for i in range(All.shape[0]):
    All['second_id'][i] = int(All['second_id'][i])
All['Transported'] = Label
All


# In[20]:


def BS_Prob(col):
    data = All.head(train.shape[0])
    values = data[col].value_counts().index.tolist()
    class_dist_0 = []
    class_dist_1 = []
    probability_0 = {}
    probability_1 = {}
    for i in values:
        
        try:
            
            a = All.head(train.shape[0]).groupby(col)['Transported'].value_counts()[i][0]
            b = All.head(train.shape[0]).groupby(col)['Transported'].value_counts()[i][1]
            class_dist_0.append(a)
            class_dist_1.append(b)
            
        except Exception:
            pass
      
        if (a*b)!=0:
            p = a/(a+b)
            probability_0[i]= p #probability of col=i and Transported=0
            probability_1[i] = (1-p)
        elif a == 0:
            probability_0[i] = 0.001
            probability_1[i] = (1-p)
        elif b == 0:
            probability_0[i] = 0.999
            probability_1[i] = (1-p)

    return probability_0,probability_1,class_dist_0,class_dist_1,values


# In[21]:


Return = BS_Prob('Age')
df = pd.DataFrame({'Transported_0': Return[2],'Transported_1': Return[3]}, index=Return[3])
ax = df.plot.bar(rot=0)
Return[0]


# In[22]:


cols = All.columns.tolist()[:-1]
def predict(record):

    p0,p1 = 1,1
    for i in cols:
        if np.isnan(record[i]) == False:
            
            prob = BS_Prob(i)
            
            prob1 = round(prob[0][int(record[i])],2)
            prob2 = round(prob[1][int(record[i])],2)
        else:
            prob1 = 1
            prob2 = 1
            
        p0 = p0 * prob1
        p1 = p1 * prob2
    return p0,p1


# In[23]:


get_ipython().run_cell_magic('time', '', 'result_test = []\nTrue_prob = []\nFalse_prob = []\ntest = All.tail(test.shape[0])\n\nfor i in range(test.shape[0]):\n    Pre_test = predict(test.iloc[i])\n    \n    True_prob.append(Pre_test[1])\n    False_prob.append(Pre_test[0])\n    \n    print(i,Pre_test[0] , Pre_test[1])\n    \n    if Pre_test[0] > Pre_test[1]:\n        result_test.append(0)\n    else:\n         result_test.append(1)\n')


# In[24]:


submission['Transported'] = result_test
submission.Transported = submission.Transported.replace({1:True, 0:False})                   


# # Improvement
# 
# Here again by exploring some of features, we can obtain high confindence in prediction just by single columns, but in small interval of values.
# Actually, as you see the probabilities which are greater than %98, we can predict Transpoted value for records, that their values on some features are in the high confindence intervals.

# In[25]:


train0 = pd.read_csv(r'../input/spaceship-titanic/train.csv')
test0 = pd.read_csv(r'../input/spaceship-titanic/test.csv')
All0 = pd.concat([train0, test0], sort=False).reset_index(drop=True)
test_1 = All0.tail(test.shape[0])
test_1=test_1.reset_index(drop=True)
test_1


# In[26]:


for i in range(test_1.shape[0]):
    
    if test_1['Spa'][i] > 3000:
        
        print(i,submission['Transported'][i])
        
        submission['Transported'][i] = False


# In[27]:


for i in range(test_1.shape[0]):
    
    if test_1['VRDeck'][i] > 4000:
        
        print(i,submission['Transported'][i])
        
        submission['Transported'][i] = False


# In[28]:


VR_Tran = []
for i in range(test_1.shape[0]):
    
    if test_1['RoomService'][i] > 3682: 
        
        print(i,submission['Transported'][i])
        
        submission['Transported'][i] = False


# In[29]:


for i in range(test_1.shape[0]):
    
    if test_1['FoodCourt'][i] > 17000:
        
        print(i,test_1['FoodCourt'][i],submission['Transported'][i])
        
        submission['Transported'][i] = True


# In[30]:


for i in range(test_1.shape[0]):
    
    if test_1['ShoppingMall'][i] > 5000:
        
        print(i,submission['Transported'][i])
        
        submission['Transported'][i] = True


# In[31]:


submission.to_csv('submission.csv', index=False)

