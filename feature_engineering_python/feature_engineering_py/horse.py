#!/usr/bin/env python
# coding: utf-8

# # <p style="background-color:lightgray; font-family:verdana; font-size:250%; text-align:center; border-radius: 15px 20px;">🟠Libraries and Data import // First look at Data o.o/🟠</p>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


# In[2]:


path = "/kaggle/input/playground-series-s3e22/"
train = pd.read_csv(path+"train.csv")
test = pd.read_csv(path+"test.csv")
sub = pd.read_csv(path+"sample_submission.csv")
org= pd.read_csv("/kaggle/input/horse-survival-dataset/horse.csv")


# In[3]:


train=pd.concat([train.drop("id",axis=1), org], ignore_index=True)


# In[4]:


train.head().T


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# 1.   **id** : Cardinal variable, has no meeaning for our modeling. Drop
# 1.   **surgery** : 'yes', 'no' Nominal variable, there we will need label encoding.
# 1.     **age** : 'adult', 'young' Nominal variable, there we will need label encoding.
# 1.     **hospital_number** : Numeric, but need generate another variable from this data which shows how many times a horse gone to hospital, can drop after done. (tried, couldnt execute another data, dropped column)
# 1.  **rectal_temp** :  Ratio Data, here we will need to control effects of temperature on target variable, if a certain temperature shows a certain death we can create an ordinal data from here. Normal temp is 37.8
# 1.    **pulse** : Ratio Data, here we will need to control effects of pulse on target variable, if a certain pulse shows a certain death we can create an ordinal data from here.Normal is 30-40, but athletic horses may have a rate of 20-25
# 1.  **respiratory_rate** : Ratio Data, here we will need to control effects of rate on target variable, if a certain rate shows a certain death we can create an ordinal data from here.Normal rate is 8 to 10
# 1.     **temp_of_extremitiess** : Nominal data, need to control effects on target, cool and cold may be correlated with death rate.
# 1.  **peripheral_pulse** : Nominal data, need to control effects on target, Increased and Absent may cause problems.
# 1.  **mucous_membrane** : Ordinal data, dark cyanotic > pale cyanotic > bright red / injected > pale pink > bright pink > normal pink
# 1.  **capillary_refill_time** : Ordinal data, more than 3 sec means poor circulation.
# 1.  **pain** : Nominal data, alert means higher lived rate, other ones doesn't matter. (alert = 1, rest = 0)
# 1. **peristalsis** :  Ordinal data, an indication of the activity in the horse's gut, absent  > hypomotile   > hypermotile > normal has higher death rate.
# 1. **abdominal_distention** : Ordinal data, severe > moderate > sight > none has higher death rate
# 1.  **nasogastric_tube** : Ordinal data, shows how much horse farts 🤣, significant > sight > none
# 1.  **nasogastric_reflux** : Ordinal data, more than 1, less than 1 and none
# 1.  **nasogastric_reflux_ph** : Ratio Data, don't need to touch, just check skewness and outliners
# 1.  **rectal_exam_feces** : Ordinal Data, Absent means there is a problem, should rank others based on death rate.
# 1.  **abdomen** : Ordinal Data, we should rank based on death/lived rate.
# 1.  **packed_cell_volume** : Ratio Data , normal range is 30 to 50. The level rises as the circulation becomes compromised or as the animal becomes dehydrated.
# 1.  **total_protein** : Ratio Data , normal values lie in the 6-7.5. The level rises as the circulation becomes compromised or as the animal becomes dehydrated. (we will need to create a dehydrated column)
# 1.  **abdomo_appearance** : Nominal Data, we should onehot encoding this one.
# 1.  **abdomo_protein** : Ratio Data, higher means riskier for gut.   
# 1.  **surgical_lesion** : Nominal Data, we should label encoding this one.
# 1.  **lesion_1** :  Ratio Data, check skewness and outliners.
# 1.  **lesion_2** : Means nothing, can drop.
# 1.  **lesion_3** : Means nothing, can drop.
# 1.  **cp_data** : Nominal Data, we should label encoding this one.
# 1.  **outcome** : Target Variable

# In[5]:


train.info()


# In[6]:


train.describe().T


# In[7]:


number_columns = ['rectal_temp','pulse','respiratory_rate','nasogastric_reflux_ph','packed_cell_volume',
                  'total_protein','abdomo_protein','lesion_1',
                  'lesion_2','lesion_3']
categorical_columns= ['surgery','age','hospital_number','temp_of_extremities','peripheral_pulse','mucous_membrane','capillary_refill_time',
                       'pain','peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','rectal_exam_feces',
                       'abdomen','abdomo_appearance','surgical_lesion','cp_data']


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
#   * for the data in train that doesnt exist on test we make them NaN

# In[8]:


for column in categorical_columns:
    print(f'Extra categories in test not in train for column: {column}')
    print(set(train[column].unique())-set(test[column].unique()))


# In[9]:


train['pain'] = train['pain'].replace({'slight':np.nan})
train['peristalsis'] = train['peristalsis'].replace({'distend_small':np.nan})
train['nasogastric_reflux'] = train['nasogastric_reflux'].replace({'slight':np.nan})
train['rectal_exam_feces'] = train['rectal_exam_feces'].replace({'serosanguious':np.nan})


# In[10]:


for column in categorical_columns:
    print(f'Extra categories in test not in train for column: {column}')
    print(set(train[column].unique())-set(test[column].unique()))


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# * for the data in test that doesnt exist on train we make them NaN

# In[11]:


for column in categorical_columns:
 print(f'Extra categories in test not in train for column: {column}')
 print(set(test[column].unique())-set(train[column].unique()))


# In[12]:


test['pain'] = test['pain'].replace({'moderate':np.nan})


# # <p style="background-color:lightgray; font-family:verdana; font-size:250%; text-align:center; border-radius: 15px 20px;">🟠EDA and Feature Engineering 🟠</p>

# <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Categorical colunms</h1>

# In[13]:


sns.countplot(x='surgery', hue='outcome', data=train)
plt.show()


# In[14]:


train['surgery'] =  train['surgery'] .map({'yes': 1, 'no': 0})
test['surgery'] =  test['surgery'].map({'yes': 1, 'no': 0,})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  surgery : 'yes', 'no' Nominal variable, there we will need simple label encoding

# In[15]:


sns.countplot(x='age', hue='outcome', data=train)
plt.show()


# In[16]:


train['age'] =  train['age'] .map({'young': 0, 'adult': 1})
test['age'] =  test['age'].map({'young': 0, 'adult': 1,})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  age : 'adult', 'young' Nominal variable, there we will need label encoding.

# In[17]:


sns.countplot(x='temp_of_extremities', hue='outcome', data=train)
plt.show()


# In[18]:


train['temp_of_extremities'] =  train['temp_of_extremities'].map({"normal": 0,"warm": 1,"cool": 2,"cold": 3})
test['temp_of_extremities'] =  test['temp_of_extremities'].map({"normal": 0,"warm": 1,"cool": 2,"cold": 3})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  temp_of_extremitiess : Nominal data, need to control effects on target, cool and cold may be correlated with death rate.

# In[19]:


sns.countplot(x='peripheral_pulse', hue='outcome', data=train)
plt.show()


# In[20]:


train['peripheral_pulse'] =  train['peripheral_pulse'].map({"normal": 0,"increased": 1,"reduced": 2,"absent": 3})
test['peripheral_pulse'] =  test['peripheral_pulse'].map({"normal": 0,"increased": 1,"reduced": 2,"absent": 3})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  peripheral_pulse : Nominal data, need to control effects on target, Increased and Absent may cause problems.

# In[21]:


sns.countplot(x='mucous_membrane', hue='outcome', data=train)
plt.show()


# In[22]:


train["mucous_membrane"]=train["mucous_membrane"].map({"normal_pink": 0,"bright_pink": 1,"pale_pink": 2,"bright_red": 3,
                                                                "pale_cyanotic": 4,
                                                                "dark_cyanotic": 5 })
test["mucous_membrane"]=test["mucous_membrane"].map({"normal_pink": 0,"bright_pink": 1,"pale_pink": 2,
                                                                "bright_red": 3,
                                                                "pale_cyanotic": 4,
                                                                "dark_cyanotic": 5 })


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  mucous_membrane : Ordinal data, dark cyanotic >  pale cyanotic > bright red / injected > pale pink > bright pink > normal pink

# In[23]:


sns.countplot(x='capillary_refill_time', hue='outcome', data=train)
plt.show()


# In[24]:


train['capillary_refill_time'] =  train['capillary_refill_time'] .map({"less_3_sec": 2,"more_3_sec": 4,"3": 3,})
test['capillary_refill_time'] =  test['capillary_refill_time'].map({"less_3_sec": 2,"more_3_sec": 4,"3": 3,})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  capillary_refill_time : Ordinal data, more than 3 sec means poor circulation.

# In[25]:


sns.countplot(x='pain', hue='outcome', data=train)
plt.show()


# In[26]:


train['pain'] =  train['pain'] .map({'alert': 0, 'mild_pain': 1,'depressed':2,'severe_pain':3,'extreme_pain':4 })
test['pain'] =  test['pain'].map({'alert': 0, 'mild_pain': 1,'depressed':2,'severe_pain':3,'extreme_pain':4 })


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  pain : Ordinal data, higher pain, looks like higher death rate, by avoiding slight in train and moderate in test, map function will convert them to NaN

# In[27]:


sns.countplot(x='peristalsis', hue='outcome', data=train)
plt.show()


# In[28]:


train['peristalsis'] =  train['peristalsis'] .map({"normal": 0,"hypermotile": 1,"hypomotile": 2,"absent": 3 })
test['peristalsis'] =  test['peristalsis'].map({"normal": 0,"hypermotile": 1,"hypomotile": 2,"absent": 3 })


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  peristalsis : Ordinal data, an indication of the activity in the horse's gut, absent > hypomotile > hypermotile > normal has higher death rate, distend_small will set as NaN.

# In[29]:


sns.countplot(x='abdominal_distention', hue='outcome', data=train)
plt.show()


# In[30]:


train['abdominal_distention'] =  train['abdominal_distention'] .map({"none": 0,"slight": 1,"moderate": 2,"severe": 3,})
test['abdominal_distention'] =  test['abdominal_distention'].map({"none": 0,"slight": 1,"moderate": 2,"severe": 3,})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  abdominal_distention : Ordinal data, severe > moderate > sight > none has higher death rate

# In[31]:


sns.countplot(x='nasogastric_tube', hue='outcome', data=train)
plt.show()


# In[32]:


train['nasogastric_tube'] =  train['nasogastric_tube'] .map({"none": 0,"slight": 1,"significant": 2})
test['nasogastric_tube'] =  test['nasogastric_tube'].map({"none": 0,"slight": 1,"significant": 2})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  nasogastric_tube : Ordinal data, shows how much horse farts 🤣, significant > sight > none

# In[33]:


sns.countplot(x='nasogastric_reflux', hue='outcome', data=train)
plt.show()


# In[34]:


train['nasogastric_reflux'] =  train['nasogastric_reflux'] .map({"none": 0,"less_1_liter": 0,"more_1_liter": 1})
test['nasogastric_reflux'] =  test['nasogastric_reflux'].map({"none": 0,"less_1_liter": 0,"more_1_liter": 1})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  nasogastric_reflux : Ordinal data, more than 1, less than 1 and none, ignore slight, looks like there isn't much diffrence between none and less_1_liter.
#     

# In[35]:


sns.countplot(x='rectal_exam_feces', hue='outcome', data=train)
plt.show()


# In[36]:


train['rectal_exam_feces'] =  train['rectal_exam_feces'] .map({"absent": 0,"decreased": 1,"normal": 2,"increased": 3,})
test['rectal_exam_feces'] =  test['rectal_exam_feces'].map({"absent": 0,"decreased": 1,"normal": 2,"increased": 3,})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  rectal_exam_feces : Ordinal Data, Absent means there is a problem, should rank others based on death rate. Ignore serosanguious

# In[37]:


sns.countplot(x='abdomen', hue='outcome',data=train)
plt.show()


# In[38]:


train['abdomen'] =  train['abdomen'] .map({"other": 0,"firm": 1,"normal": 2,"distend_large": 3, "distend_small": 4})
test['abdomen'] =  test['abdomen'].map({"other": 0,"firm": 1,"normal": 2,"distend_large": 3, "distend_small": 4})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  abdomen : Ordinal Data, we should rank based on death/lived rate.

# In[39]:


sns.countplot(x='abdomo_appearance',hue='outcome',data=train)


# In[40]:


train["abdomo_appearance"]=train["abdomo_appearance"].map({"serosanguious": 0,"cloudy": 1,"clear": 2})
test["abdomo_appearance"]=test["abdomo_appearance"].map({"serosanguious": 0,"cloudy": 1,"clear": 2})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  abdomo_appearance : Nominal Data, we did one hot encode this one.
#    

# In[41]:


sns.countplot(x='surgical_lesion',hue='outcome',data=train)


# In[42]:


train['surgical_lesion'] =  train['surgical_lesion'] .map({"no": 0,"yes": 1})
test['surgical_lesion'] =  test['surgical_lesion'].map({"no": 0,"yes": 1})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  surgical_lesion : Nominal Data, we did label encoding this one.

# In[43]:


sns.countplot(x='cp_data',hue='outcome',data=train)


# In[44]:


train['cp_data'] =  train['cp_data'] .map({"no": 0,"yes": 1})
test['cp_data'] =  test['cp_data'].map({"no": 0,"yes": 1})


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  cp_data : Nominal Data, we did label encoding this one.

# <h1 style="background-color:lightgray;font-family:newtimeroman;font-size:350%;text-align:center;border-radius: 15px 50px;">Numerical colunms</h1>

# In[45]:


plt.figure(figsize=(14,15))
for idx,column in enumerate(number_columns[1:9]): #without id,hospitalId, lesion_2, lesion_3
    plt.subplot(len(number_columns)//2,2,idx+1)
    sns.histplot(x=column, hue="outcome", data=train,bins=30,kde=True)
    plt.title(f"{column} Distribution")
    plt.ylim(0,200)
    plt.tight_layout()


# In[46]:


plt.figure(figsize=(8,8))
corr=train[number_columns[1:-3]].corr(numeric_only=True)
mask= np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, linewidths=1, mask=mask)


# In[47]:


def les_sep(i):
    if len(str(i))==1:
        return ["00","0","0","0"]

    elif len(str(i))==2:
        two=list(str(i))
        two.insert(1,"0")
        two.insert(2,"0")
        return two

    elif len(str(i))==3:
        tres=list(str(i))
        tres.insert(0,"00")
        return tres


    elif len(str(i))==4:
        return list(str(i))

    else :
        five=list(str(i))
        if (five[3]=="1") & (five[4]=="0"):
            five.append("".join(five[3:5]))
            five.pop(3)
            five.pop(3)
        else:
            five.insert(0,("".join(five[0:2])))
            five.pop(1)
            five.pop(1)
        return five


# In[48]:


[les_sep(each) for each in train["lesion_1"]]


# In[49]:


train["lesions"]=[les_sep(each) for each in train["lesion_1"]]
test["lesions"]=[les_sep(each) for each in test["lesion_1"]]
train.head()


# In[50]:


train_expanded = train["lesions"].apply(pd.Series).add_prefix('les')
test_expanded = test["lesions"].apply(pd.Series).add_prefix('les')


# In[51]:


train_expanded.head()


# In[52]:


train = pd.concat([train, train_expanded], axis=1)
test = pd.concat([test, test_expanded], axis=1)


# In[53]:


train.head()


# In[54]:


test.head()


# In[55]:


train[["les0","les1","les2","les3"]]=train[["les0","les1","les2","les3"]].astype(int)
test[["les0","les1","les2","les3"]]=test[["les0","les1","les2","les3"]].astype(int)


# # 🛠️Feature Engineering 🛠️
# 
# 

# In[56]:


train["outcome"]=train["outcome"].map({"died": 0,
                                       "euthanized": 1,
                                       "lived": 2
                                       })


# In[57]:


train.drop(["lesions","lesion_1"],axis=1, inplace=True)
test.drop(["lesions","lesion_1"],axis=1, inplace=True)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# *  Converted outcome to numeric temporarily  for modeling.

# In[58]:


def dehydration(df):
    df["dehydrated"] = df["packed_cell_volume"] / df["total_protein"]  
    df["pulse_high"] = df["pulse"] * df["respiratory_rate"]
    df["protein"] =  df["total_protein"] -  df["abdomo_protein"]
    
dehydration(train)
dehydration(test)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# * dehydrated calculates cell's dehydration
# * pulse_high calculates how high pulse effect on horse
# * protein calculates how much extra protein horse has

# # <p style="background-color:lightgray; font-family:verdana; font-size:250%; text-align:center; border-radius: 15px 20px;">🟠Modeling🟠</p>
# 

# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Split</p>

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train.drop(columns=["outcome"] )
y= train["outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1, random_state = 30)


# In[60]:


X_train.head()


# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Feature Importance</p>

# In[61]:


num_feats=['rectal_temp', 'pulse', 'respiratory_rate',
       'nasogastric_reflux_ph', 'packed_cell_volume', 'total_protein',
       'abdomo_protein',"protein","dehydrated","pulse_high"]
cat_feats=["hospital_number",'surgery',
 'age',
 'temp_of_extremities',
 'peripheral_pulse',
 'mucous_membrane',
 'capillary_refill_time',
 'pain',
 'peristalsis',
 'abdominal_distention',
 'nasogastric_tube',
 'nasogastric_reflux',
 'rectal_exam_feces',
 'abdomen',
 'abdomo_appearance',
 'surgical_lesion',
 'cp_data',"les0","les1","les2","les3","lesion_2","lesion_3"]


# In[62]:


pip install autofeatselect


# In[63]:


from autofeatselect import FeatureSelector
#Create Feature Selector Object
feat_selector = FeatureSelector(modeling_type='classification',
                                X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test,
                                numeric_columns=num_feats,
                                categorical_columns=cat_feats,
                                seed=42)

#Note: Hyperparameters and objective function of LightGBM can be changed.
lgbm_importance_df = feat_selector.lgbm_importance(hyperparam_dict=None,
                                                   objective=None,
                                                   return_plot=True)


# In[64]:


lgbm_importance_df.tail(10)


# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
#  Looks like
# 
# * abdomen
# * peristalsis
# * peripheral_pulse
# * les2
# * lesion_2
# * lesion_3
#      are not much important for our modeling, we can drop them.

# In[65]:


drop_list=lgbm_importance_df["feature"][-6:]
X_train.drop(drop_list, axis=1, inplace=True)
X_test.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)


# In[66]:


get_ipython().system('pip install lazypredict')


# In[67]:


y_test


# In[71]:


from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit( X_train, X_test , y_train, y_test)
models


# <a id="1"></a>
# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">CatBoost Classifier</p>

# In[72]:


from catboost import CatBoostClassifier
import optuna

def objective_cat(trial):
    """Define the objective function"""

    params = {
        "iterations" : trial.suggest_int("iterations", 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        "depth" : trial.suggest_int("depth", 1, 10),
        "l2_leaf_reg" : trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        "bootstrap_type" : trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        "random_strength" : trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature" : trial.suggest_float("bagging_temperature", 0.0, 10.0),
        "od_type" : trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        "od_wait" : trial.suggest_int("od_wait", 10, 50),
        "verbose" : False
        
    }


    model_cat = CatBoostClassifier(**params)
    model_cat.fit(X_train, y_train)
    y_pred = model_cat.predict(X_test)
    return accuracy_score(y_test,y_pred)


# In[73]:


study_cat = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_cat.optimize(objective_cat, n_trials=50,show_progress_bar=True)


# In[74]:


# Print the best parameters
print('Best parameters', study_cat.best_params)


# In[75]:


cat = CatBoostClassifier(**study_cat.best_params, verbose=False)
cat.fit(X_train, y_train)
y_pred = cat.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))


# <a id="1"></a>
# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">LightGBM Classifier</p>

# In[76]:


from lightgbm import LGBMClassifier
import optuna

def objective_lgb(trial):
    """Define the objective function"""

    params = {
        'objective': trial.suggest_categorical('objective', ['multiclass']),
        'metric': trial.suggest_categorical('metric', ['multi_logloss']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
        "random_state" : trial.suggest_categorical('random_state', [42]),
    }


    model_lgb = LGBMClassifier(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    return accuracy_score(y_test,y_pred)


# In[77]:


study_lgb = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_lgb.optimize(objective_lgb, n_trials=50,show_progress_bar=True)


# In[78]:


# Print the best parameters
print('Best parameters', study_lgb.best_params)


# In[79]:


lgb = LGBMClassifier(**study_lgb.best_params)
lgb.fit(X_train, y_train)
y_pred = lgb.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))


# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">XGBoost Classifier</p>

# In[80]:


from xgboost import XGBClassifier
import optuna
def objective_xg(trial):
    """Define the objective function"""

    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 300, 700),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 0.5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        'eval_metric':trial.suggest_categorical('eval_metric', ['mlogloss']),
    }
    model_xgb = XGBClassifier(**params)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    return accuracy_score(y_test,y_pred)


# In[81]:


study_xgb = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_xgb.optimize(objective_xg, n_trials=50,show_progress_bar=True)


# In[82]:


# Print the best parameters
print('Best parameters', study_xgb.best_params)


# In[83]:


xgb = XGBClassifier(**study_xgb.best_params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))


# In[84]:


lgb_imp=(lgb.feature_importances_-lgb.feature_importances_.min())/(lgb.feature_importances_.max()-lgb.feature_importances_.min())
xgb_imp=(xgb.feature_importances_-xgb.feature_importances_.min())/(xgb.feature_importances_.max()-xgb.feature_importances_.min())
importances=pd.DataFrame({"Features": X_train.columns , "Importance_LGBM":lgb_imp*100, "Importance_XGB":xgb_imp*100})
importances


# # <p style="padding:10px;background-color:orange;margin:0;color:black;font-family:newtimeroman;font-size:100%;text-align:center;border-radius: 15px 50px;overflow:hidden;font-weight:500">Voting and Stacking Classifier</p>

# In[85]:


from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('cat', cat),
                                      ('lgbm', lgb), 
                                      ('xgb', xgb)], voting='soft')
voting.fit(X_train,y_train)
voting_pred = voting.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, voting_pred))


# In[86]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(voting,X_test, y_test,display_labels=("died", "euthanized", "lived"),cmap="RdPu");


# In[87]:


from sklearn.ensemble import StackingClassifier
stk = StackingClassifier(estimators=[('lgbm', lgb), 
                                      ('xgb', xgb)])
stk.fit(X_train,y_train)
stk_pred = stk.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, stk_pred))


# In[88]:


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(stk,X_test, y_test,display_labels=("died", "euthanized", "lived"),cmap="RdPu")


# # <p style="background-color:lightgray; font-family:verdana; font-size:250%; text-align:center; border-radius: 15px 20px;">🟠Prediction🟠</p>

# In[89]:


sub["outcome"]=voting.predict(test.drop(columns=["id"] ))
sub["outcome"]=sub["outcome"].map({ 0: "died",
                                    1: "euthanized",
                                    2: "lived"
                                       })
sub.to_csv('submission.csv',index=False)
sub

