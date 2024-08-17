#!/usr/bin/env python
# coding: utf-8

# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:10px 10px;font-weight:bold;border:2px solid #AB47BC;">Spaceship Titanic</p>
# 

# <center><img src= "https://raw.githubusercontent.com/ashwinshetgaonkar/kaggle-kernel-images/main/spaceship.jpg" alt ="spaceship" style='width:600px;'></center><br>
# 

# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Context</p>

# <h3>
# The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.
# 
# While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!
# 
# 
# 
# <center><img src= "https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg" alt ="anamoly" style='width:600px;'></center><br>
# 
# 
# 
# To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.
# 
# Help save them and change history!</h3>

# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Data Description</p>

# <h4>
#     <ul>
#         <li><b style='color:#AB47BC'>train.csv</b> - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.</li>
#         
# <li><b style='color:#AB47BC'>PassengerId</b> - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.</li>
# <li><b style='color:#AB47BC'>HomePlanet </b>- The planet the passenger departed from, typically their planet of permanent residence.</li>
# <li><b style='color:#AB47BC'>CryoSleep </b>- Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.</li>
# <li><b style='color:#AB47BC'>Cabin </b>- The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.</li>
# <li><b style='color:#AB47BC'>Destination</b> - The planet the passenger will be debarking to.</li>
# <li><b style='color:#AB47BC'>Age </b>- The age of the passenger.</li>
# <li><b style='color:#AB47BC'>VIP</b> - Whether the passenger has paid for special VIP service during the voyage.</li>
# <li><b style='color:#AB47BC'>RoomService, FoodCourt, ShoppingMall, Spa, VRDeck </b>- Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.</li>
# <li><b style='color:#AB47BC'>Name </b>- The first and last names of the passenger.</li>
# <li><b style='color:#AB47BC'>Transported </b>- Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.</li>
# <br>        
# <li><b style='color:#AB47BC'>test.csv</b> - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.</li>
#  <br>
# <li><b style='color:#AB47BC'>sample_submission.csv</b> - A submission file in the correct format.</li>
# <li><b style='color:#AB47BC'>PassengerId </b>- Id for each passenger in the test set.</li>
# <li><b style='color:#AB47BC'>Transported </b>- The target. For each passenger, predict either True or False.  </li>      
#         
#         
#  </ul>    
# </h4>
#     
#     

#  <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Importing Libraries</p>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook',font_scale=1.25)
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')
import optuna
from IPython.core.display import HTML,display
import xgboost as xgb
from xgboost import XGBClassifier
from optuna.integration import LightGBMPruningCallback
from optuna.visualization import plot_optimization_history,plot_param_importances
from sklearn.metrics import log_loss,accuracy_score
import gc


# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Importing Data</p>

# In[2]:


train_df=pd.read_csv('../input/spaceship-titanic/train.csv')
test_df=pd.read_csv('../input/spaceship-titanic/test.csv')
sub_df=pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
train_df.head()


# In[3]:


# check for duplicate values
train_df.duplicated().sum()


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>There are no duplicate values.</li>
#     </ul>
# </h3>
# 

# In[4]:


# check the number of rows and columns
train_row,train_col=train_df.shape[0],train_df.shape[1]
test_row,test_col=test_df.shape[0],test_df.shape[1]

text=f"<h3>The training dataset has {train_row} rows and {train_col} columns.<br><br>The test dataset has {test_row} rows and {test_col} columns.</h3>"
display(HTML(text))


# In[5]:


# check for missing values
train_df.isna().sum()


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>There are a lot of missing values which needs to be imputed suitably.</li>
#     </ul>
# </h3>
# 

# In[6]:


test_df.isna().sum()


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>There are a lot of missing values which needs to be imputed suitably.</li>
#     </ul>
# </h3>
# 

# In[7]:


# check the data types
train_df.dtypes


# In[8]:


# check the cardinality of the columns
train_df.nunique()


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>There are 5 numerical features: Age,RoomService,FoodCourt,ShoppingMall,Spa,VRDeck.</li><br>
#      <li>There are 4 categotical features: HomePlanet,CryoSleep,Destination,VIP.</li><br>
#      <li>There are 3 descriptive features: PassengerId,Cabin,Name.</li><br>
#      <li>Here the Target variable is categorical: Transported.</li><br>
#     </ul>
# </h3>
# 

# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">EDA</p>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 210px">Univariate Analysis</span>

# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Numerial Features</span>

# In[9]:


plt.figure(figsize=(10,7))
sns.histplot(data=train_df,x='Age',color='lightblue',stat='density',element='step')
sns.kdeplot(data=train_df,x='Age',color='red');
display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 250px;font-weight:bold'> Distribution of Age</h3>"))


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>The distribution of Age follows the Normal Distribution which is always beneficial.</li>
#     </ul>
# </h3>

# In[10]:


luxery_features=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
plt.figure(figsize=(24,10))
display(HTML("<h3 style='color:#AB47BC;font-size:22px;font-weight:bold;text-align:center'> Distribution of Luxery Features</h3>"))
for i,col in enumerate(luxery_features):
    plt.subplot(2,3,i+1)
    sns.histplot(data=train_df,x=col,color='lightblue',stat='density',element='step',bins=4)
    sns.kdeplot(data=train_df,x=col,color='red');
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>Majority of the passengers did not make any expenditure.</li><br>
#      <li>It will be a good idea to club all the expenditures under a common term.</li>
#     </ul>
# </h3>
# 

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Categorical Features</span>

# In[11]:


categorical_features=['HomePlanet', 'CryoSleep', 'Destination','VIP' ]
display(HTML("<h3 style='color:#AB47BC;font-size:22px;font-weight:bold;text-align:center'> Distribution of Categorical Features</h3>"))
plt.figure(figsize=(24,6))
for i,col in enumerate(categorical_features):
    plt.subplot(1,4,i+1)
    sns.countplot(data=train_df,x=col,palette='tab10')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li> Majority of the Passengers were from HomePlanet Earth.</li><br>
#      <li> Majority of the Passengers opted not to CryoSleep.</li><br>
#      <li> Majority of the Passengers were enroute to planet TRAPPIST-1e.</li><br>
#      <li> Almost all of the Passengers did not opt for VIP pass.</li>
#     </ul>
# </h3>
# 

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Distribution of Target</span>

# In[12]:


plt.figure(figsize=(10,6))
colors=['#AB47BC','#26C6DA']
plt.pie(train_df['Transported'].value_counts(),labels=['True','False'],autopct='%.1f%%',explode=[0.01,0.01],colors=colors);
plt.ylabel('Transported');


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>The dataset is well balanced.</li>
#     </ul>
# </h3>
# 

# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Bivariate Analysis</span>
# 

# In[13]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;font-weight:bold;padding:0 0 0 100px'>Distribution of Age with respect to Target variable</h3>"))
plt.figure(figsize=(10,7))
sns.histplot(data=train_df,x='Age',color='lightblue',stat='density',element='step',hue='Transported',kde=True);


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li> Upto an age of 12 there is a higher probability of being successully transported.</li><br>
#      <li> From 12-18 the probability of being successfully transported is slightly more.</li><br>
#      <li> From 18-40 the probability of not being successfully transported is slightly more.</li><br>
#      <li> From 40 onwards the probability of both events is kinda same.</li>
#     </ul>
# </h3>
# 
# 

# In[14]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;text-align:center;font-weight:bold'> Distribution of Categorical Features with respect to Target</h3>"))

categorical_features=['HomePlanet', 'CryoSleep', 'Destination','VIP' ]
plt.figure(figsize=(24,6))
for i,col in enumerate(categorical_features):
    plt.subplot(1,4,i+1)
    sns.countplot(data=train_df,x=col,hue='Transported',palette='Set1')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li> Passengers from HomePlanet Europa have a higher probability of being successfully transported.</li><br>
#      <li> Passengers that opted for Cryosleep have a high chance of being unsuccessfully transported.</li><br>
#      <li> No informative inference can be drawn from distribution of Distination.</li><br>
#      <li> It can be said that target is independent of VIP feature value ,so I will be dropping this feature.</li>
#     </ul>
# </h3>
# 

# In[15]:


# droping the VIP column
train_df.drop(columns=['VIP'],inplace=True)
test_df.drop(columns=['VIP'],inplace=True)


# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Missing values Imputation</p>

# In[16]:


# copying the values of train_df into a temp_df to perform experiments
temp_df=train_df.copy()


# In[17]:


def check_nan_values(df,name):   
    plt.figure(figsize=(12,6))
    sns.heatmap(df.isna().T, cmap='viridis')
#     plt.title(f'Heatmap of missing values of {name}',fontsize=20);
    display(HTML(f"<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 180px;font-weight:bold'> Heatmap of missing values of {name}</h3>"))
    plt.xticks([])


# In[18]:


check_nan_values(temp_df,'temp_df')


# In[19]:


temp_df['na_count']=temp_df.isna().sum(axis=1)
plt.figure(figsize=(10,4))
sns.countplot(data=temp_df, x='na_count', hue='Transported',palette='Set1')
# plt.title('Number of missing entries by passenger')
display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 0px;font-weight:bold'> Distribution of Number of missing values with respect to Target</h3>"))
temp_df.drop('na_count', axis=1, inplace=True)


# 
# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>The presence of missing values does not favour any of the target class.</li>
#     </ul>
# </h3>
# 

# In[20]:


# using median value of Age 
median_age=temp_df['Age'].median()
temp_df['Age'].fillna(median_age,inplace=True)
test_df['Age'].fillna(median_age,inplace=True)


# In[21]:


# Find mode of each categorical feature
cat_cols=['HomePlanet','CryoSleep','Destination']
modes=temp_df[cat_cols].mode()
modes


# In[22]:


# using mode(most frequently occuring value for imputing missing values)
for col in cat_cols:
    temp_df[col].fillna(modes[col][0],inplace=True)   
    test_df[col].fillna(modes[col][0],inplace=True)


# In[23]:


# assigning value 'ZZ/1899/ZZ' for missing values in Cabin
temp_df['Cabin'].fillna('ZZ/1899/ZZ',inplace=True)
test_df['Cabin'].fillna('ZZ/1899/ZZ',inplace=True)


# In[24]:


luxery_features


# In[25]:


# imputing value of 0 for missing values in luxery features
for col in luxery_features:
    temp_df[col].fillna(0,inplace=True)
    test_df[col].fillna(0,inplace=True)


# In[26]:


# assigning value 'No Name' for missing values in Name
temp_df['Name'].fillna('No Name',inplace=True)
test_df['Name'].fillna('No Name',inplace=True)


# In[27]:


check_nan_values(temp_df,'temp_df')


# In[28]:


check_nan_values(test_df,'test_df')


# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Feature Engineering</p>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Binning Age</span>

# In[29]:


temp_df['Age_binned']=pd.cut(temp_df['Age'],bins=[-1,12,18,40,100],labels=['child','teenage','mid-age','old'])

test_df['Age_binned']=pd.cut(test_df['Age'],bins=[-1,12,18,40,100],labels=['child','teenage','mid-age','old'])


# In[30]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 50px;font-weight:bold'> Distribution of Age_binned</h3>"))

plt.figure(figsize=(12,6))
temp_df['Age_binned'].value_counts().plot(kind='pie');


# In[31]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 50px;font-weight:bold'> Distribution of Age_binned with respect to Target</h3>"))

plt.figure(figsize=(8,6))
sns.countplot(data=temp_df,x='Age_binned',hue='Transported',palette='Set1');


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>This strengthens the obsevations I made before binnig the Age feature.</li>
#     </ul>
# </h3>
# 

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Aggregating Luxery Features</span>

# In[32]:


temp_df['total_expenditure']=0.0
for col in luxery_features:
    temp_df['total_expenditure']+=temp_df[col]
    
test_df['total_expenditure']=0.0
for col in luxery_features:
    test_df['total_expenditure']+=test_df[col]
    

temp_df['zero_expenditure']=(temp_df['total_expenditure']==0.0).astype(int)
test_df['zero_expenditure']=(test_df['total_expenditure']==0.0).astype(int)


temp_df.drop(columns=luxery_features,inplace=True)
test_df.drop(columns=luxery_features,inplace=True)


# In[33]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;text-align:center;font-weight:bold'> Distribution of total_expenditure</h3>"))
plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.histplot(data=temp_df,x='total_expenditure',hue='Transported',stat='density',kde=False,bins=50)
plt.title(f"Hist plot")
plt.subplot(1,2,2)
sns.kdeplot(data=temp_df,x='total_expenditure',hue='Transported')
plt.title(f"Kde plot")
plt.tight_layout()
plt.show()


# 
# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>It follows the Power Law.</li><br>
#      <li>The passengers who made zero expenditure have a higher probability of being successfully transported.</li>
#     </ul>
# </h3>

# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Extracting the Group size</span>

# In[34]:


temp_df['PassengerId'][0]


# In[35]:


temp_df['Group']=temp_df['PassengerId'].apply(lambda x:x.split('_')[0]).astype('int')

test_df['Group']=test_df['PassengerId'].apply(lambda x:x.split('_')[0]).astype('int')


# In[36]:


temp_df['Group_size']=temp_df['Group'].apply(lambda x:temp_df['Group'].value_counts()[x])
common_groups=temp_df['Group'].unique()
test_df['Group_size']=test_df['Group'].apply(lambda x:temp_df['Group'].value_counts()[x] if x in common_groups else 1)


# In[37]:


# distribution of Groupsize with respect to Target
display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 80px;font-weight:bold'> Distribution of Group size with respect to Target</h3>"))
plt.figure(figsize=(10,7))
sns.countplot(data=temp_df,x='Group_size',hue='Transported',palette='Set1');


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>Passsengers having Group size = 1 have higher probability of being unsuccessfully transported.</li><br>
#      <li>It will be a good idea to explicitly inform this by using an additional feature,'Solo'.
#     </ul>
# </h3>
# 

# In[38]:


temp_df['Solo']=(temp_df['Group_size']==1).astype('int')
test_df['Solo']=(test_df['Group_size']==1).astype('int')


# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 210px">Decomposing the descriptive feature 'Cabin'</span>
# 

# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Splitting the various parameters</span>
# 

# In[39]:


temp_df['Cabin_deck']=temp_df['Cabin'].apply(lambda x: x.split('/')[0])
temp_df['Cabin_num']=temp_df['Cabin'].apply(lambda x: x.split('/')[1])
temp_df['Cabin_side']=temp_df['Cabin'].apply(lambda x: x.split('/')[2])


test_df['Cabin_deck']=test_df['Cabin'].apply(lambda x: x.split('/')[0])
test_df['Cabin_num']=test_df['Cabin'].apply(lambda x: x.split('/')[1])
test_df['Cabin_side']=test_df['Cabin'].apply(lambda x: x.split('/')[2])


# In[40]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 0px;text-align:center;font-weight:bold'> Distribution of Cabin_deck & Cabin_side with respect to Target</h3>"))

cabin_features=['Cabin_deck','Cabin_side']
plt.figure(figsize=(24,8))
for i,col in enumerate(cabin_features):
    plt.subplot(1,2,i+1)
    sns.countplot(data=temp_df,x=col,hue='Transported',palette='Set1')
    plt.title(f"Distribution of {col}",fontsize=20)
plt.tight_layout()
plt.show()


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>The probability of being successfully transported varies as per the Cabin_deck the passenger is in,It has max. probability in B and least in F/E</li><br>
#      <li>Passengers in Cabin_side S have a higher chance of being successfully transported in comparison to those in P.</li>
#  
# </ul>
# </h3>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Binning Cabin_num</span>
# 

# In[41]:


bins=[x for x in range(-1,1901,100)]+[9000]
labels=[i+100 for i in range(len(bins)-1)]
temp_df['Cabin_num_binned']=pd.cut(temp_df['Cabin_num'].astype('int'),bins=bins,labels=labels)

test_df['Cabin_num_binned']=pd.cut(test_df['Cabin_num'].astype('int'),bins=bins,labels=labels)


# In[42]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 150px;text-align:left;font-weight:bold'> Distribution of Cabin_num_binned with respect to Target</h3>"))

plt.figure(figsize=(16,6))
sns.countplot(data=temp_df,x='Cabin_num_binned',hue='Transported',palette='Set1')
plt.xticks(rotation=80)
plt.show()


# 
# 
# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>In the  binned feature it is prominent that passengers in the first 3 categories have a higher chance of being successfully transported as compared to others.</li><br>
#      <li>It would be a good idea to explicitly encode this information using an additional feature,'is_in_first_3'</li>
#  
# </ul>
# </h3>

# In[43]:


temp_df['is_in_first_3']=(temp_df['Cabin_num_binned']<=102).astype('int')
test_df['is_in_first_3']=(test_df['Cabin_num_binned']<=102).astype('int')


# 
# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 210px">Decomposing the descriptive feature 'Name'</span>

# In[44]:


# extracting the lastname
temp_df['LastName']=temp_df['Name'].apply(lambda x:x.split(" ")[-1])
test_df['LastName']=test_df['Name'].apply(lambda x:x.split(" ")[-1])


# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Family size</span>

# In[45]:


temp_df['Family_size']=temp_df['LastName'].apply(lambda x: temp_df['LastName'].value_counts()[x])

common_family_names=temp_df['LastName'].unique()
test_df['Family_size']=test_df['LastName'].apply(lambda x: temp_df['LastName'].value_counts()[x] if x in common_family_names else 1)


# In[46]:


# Assigning Family size = 0 whose names were imputed
temp_df.loc[temp_df['Name']=='No Name','Family_size']=0
test_df.loc[test_df['Name']=='No Name','Family_size']=0


# In[47]:


display(HTML("<h3 style='color:#AB47BC;font-size:22px;padding:0px 0px 0px 150px;text-align:left;font-weight:bold'> Distribution of Family_size with respect to Target</h3>"))

plt.figure(figsize=(16,6))
sns.countplot(data=temp_df,x='Family_size',hue='Transported',palette='Set1')
plt.xticks(rotation=80)
plt.show()


# 
# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>It follows the Normal Distribution.</li><br>
#      <li>Passengers with Family_size: ,3-5 have a higher chance of being successfully transported as compared to its </li><br>
#      
#  
# </ul>
# </h3>

# In[48]:


features_to_drop=['PassengerId','Cabin','Group','Cabin_num','Name','LastName','Age']
temp_df.drop(columns=features_to_drop,inplace=True)
test_df.drop(columns=features_to_drop,inplace=True)


# In[49]:


temp_df.shape,test_df.shape


# In[50]:


columns_to_encode=['Cabin_deck','Cabin_side','HomePlanet', 'CryoSleep', 'Destination','Age_binned']
for col in columns_to_encode:
    temp_df[col]=temp_df[col].astype('category').cat.codes
    # for test_df 
    test_df[col]=test_df[col].astype('category').cat.codes

label_encoder=LabelEncoder()
temp_df['Transported']=label_encoder.fit_transform(temp_df['Transported'])


# In[51]:


train_df['Transported']=label_encoder.transform(train_df['Transported'])


# In[52]:


temp_df.info()


# In[53]:


X=train_df.drop(columns='Transported')
y=train_df[['Transported']]


# In[54]:


X.head()


# In[55]:


y.head()


# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;">Spliting the data into train and test</p>

# In[56]:


xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20,stratify=y.values)


# In[57]:


xtrain.shape,ytrain.shape,xtest.shape,ytest.shape


# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Defining Data Imputation & Feature Engineering Function</span>

# In[58]:


def data_preprocessing(x,x_):
    x_train=x.copy()
    x_val=x_.copy()
    
    median_age=x_train['Age'].median()
    x_train['Age'].fillna(median_age,inplace=True)
    x_val['Age'].fillna(median_age,inplace=True)


    cat_cols=['HomePlanet','CryoSleep','Destination']
    modes=x_train[cat_cols].mode()

    for col in cat_cols:
        x_train[col].fillna(modes[col][0],inplace=True)   
        x_val[col].fillna(modes[col][0],inplace=True)

            
    x_train['Cabin'].fillna('ZZ/1899/ZZ',inplace=True)
    x_val['Cabin'].fillna('ZZ/1899/ZZ',inplace=True)

    for col in luxery_features:
        x_train[col].fillna(0,inplace=True)
        x_val[col].fillna(0,inplace=True)
            
    x_train['Name'].fillna('No Name',inplace=True)
    x_val['Name'].fillna('No Name',inplace=True)

    x_train['Age_binned']=pd.cut(x_train['Age'],bins=[-1,12,18,40,100],labels=['child','teenage','mid-age','old'])
    x_val['Age_binned']=pd.cut(x_val['Age'],bins=[-1,12,18,40,100],labels=['child','teenage','mid-age','old'])

            
    x_train['total_expenditure']=0.0
    x_val['total_expenditure']=0.0
    for col in luxery_features:
        x_train['total_expenditure']+=x_train[col]
        x_val['total_expenditure']+=x_val[col]
                    
    x_train['zero_expenditure']=(x_train['total_expenditure']==0.0).astype(int)
    x_val['zero_expenditure']=(x_val['total_expenditure']==0.0).astype(int)

    x_train.drop(columns=luxery_features,inplace=True)
    x_val.drop(columns=luxery_features,inplace=True)

    
    
    x_train['Group']=x_train['PassengerId'].apply(lambda x:x.split('_')[0]).astype('int')

    x_val['Group']=x_val['PassengerId'].apply(lambda x:x.split('_')[0]).astype('int')


    x_train['Group_size']=x_train['Group'].apply(lambda x:x_train['Group'].value_counts()[x]).astype('int')
    common_groups=x_train['Group'].unique()
    x_val['Group_size']=x_val['Group'].apply(lambda x:x_train['Group'].value_counts()[x] if x in common_groups else 1)

    x_train['Solo']=(x_train['Group_size']==1).astype('int')
    x_val['Solo']=(x_val['Group_size']==1).astype('int')
    
    
    x_train['Cabin_deck']=x_train['Cabin'].apply(lambda x: x.split('/')[0])
    x_train['Cabin_num']=x_train['Cabin'].apply(lambda x: x.split('/')[1])
    x_train['Cabin_side']=x_train['Cabin'].apply(lambda x: x.split('/')[2])


    x_val['Cabin_deck']=x_val['Cabin'].apply(lambda x: x.split('/')[0])
    x_val['Cabin_num']=x_val['Cabin'].apply(lambda x: x.split('/')[1])
    x_val['Cabin_side']=x_val['Cabin'].apply(lambda x: x.split('/')[2])



    bins=[x for x in range(-1,1901,100)]+[9000]
    labels=[i+100 for i in range(len(bins)-1)]
    x_train['Cabin_num_binned']=pd.cut(x_train['Cabin_num'].astype('int'),bins=bins,labels=labels)

    x_val['Cabin_num_binned']=pd.cut(x_val['Cabin_num'].astype('int'),bins=bins,labels=labels)



    x_train['is_in_first_3']=(x_train['Cabin_num_binned']<=102).astype('int')
    x_val['is_in_first_3']=(x_val['Cabin_num_binned']<=102).astype('int')
    
            
        
    x_train['LastName']=x_train['Name'].apply(lambda x:x.split(" ")[-1])
    x_val['LastName']=x_val['Name'].apply(lambda x:x.split(" ")[-1])


    x_train['Family_size']=x_train['LastName'].apply(lambda x: x_train['LastName'].value_counts()[x])

    common_family_names=x_train['LastName'].unique()
    x_val['Family_size']=x_val['LastName'].apply(lambda x: x_train['LastName'].value_counts()[x] if x in common_family_names else 1)


            # Assigning Family size = 0 whose names were imputed
    x_train.loc[x_train['Name']=='No Name','Family_size']=0
    x_val.loc[x_val['Name']=='No Name','Family_size']=0


    features_to_drop=['PassengerId','Cabin','Group','Cabin_num','Name','LastName','Age']
    x_train.drop(columns=features_to_drop,inplace=True)
    x_val.drop(columns=features_to_drop,inplace=True)
    
    
    columns_to_encode=['Cabin_deck','Cabin_side','HomePlanet', 'CryoSleep', 'Destination','Age_binned']
    for col in columns_to_encode:
        x_train[col]=x_train[col].astype('category').cat.codes
        x_val[col]=x_val[col].astype('category').cat.codes
    
    return x_train,x_val


# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Modelling</p>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 210px">With Default Parameters</span>
# 

# In[63]:


# define the Kfold splits
cv=StratifiedKFold(n_splits=5,shuffle=True)
    
def compute_performance_with_default_parameters(models,x,y):
    # to store scores
    xtrain=x.copy()
    ytrain=y.copy()
    results=[]
    for name,model in tqdm(models.items()):
        result={}
        result['name']=name
        result['train_score']=0
        result['test_score']=0
        
        for train_idx,val_idx in cv.split(xtrain,ytrain):
            x_train,x_val=xtrain.iloc[train_idx],xtrain.iloc[val_idx]
            y_train,y_val=ytrain.iloc[train_idx],ytrain.iloc[val_idx]
                
            x_train,x_val=data_preprocessing(x_train,x_val)
            clf=model
            if name=='XGBoost':
                clf.fit(x_train.values,y_train.values,eval_metric='logloss')
            else:
                clf.fit(x_train.values,y_train.values)
                
            train_pred=clf.predict(x_train.values)
            test_pred=clf.predict(x_val.values)

            result['train_score']+=accuracy_score(y_train.values,train_pred)
            result['test_score']+=accuracy_score(y_val.values,test_pred)
            del x_train,x_val,y_train,y_val
            gc.collect()
        result['train_score']/=5
        result['test_score']/=5
        results.append(result)
        
    return pd.DataFrame(results)
        


# In[64]:


models={'Random Forest':RandomForestClassifier(n_jobs=-1),'LightGBM':LGBMClassifier(n_jobs=-1),'XGBoost':XGBClassifier(n_jobs=-1)}
results=compute_performance_with_default_parameters(models,xtrain,ytrain)
results


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>LightGBM model gave the best performance.</li>
#     </ul>
# </h3>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Hyperparameter tunning LightGBM using Optuna</span>
# 

# In[65]:


def objective(trial):
    
    params = {
    'objective':'binary',
    'n_estimators':trial.suggest_categorical('n_estimators',[3000]),
    'num_leaves':trial.suggest_int('num_leaves',100,600,step=2),
    'subsample':trial.suggest_float("subsample",0.4,0.9,step=0.1),
    'min_child_samples':trial.suggest_int("min_child_samples",40,400,step=10),
    'colsample_bytree':trial.suggest_float("colsample_bytree",0.4,0.9,step=0.1),
    'learning_rate':trial.suggest_categorical("learning_rate",[0.01]),   
#     'learning_rate':trial.suggest_loguniform("learning_rate",1e-4,0.1),

#     "max_depth": trial.suggest_int("max_depth", 5,17,1),
    'reg_alpha':trial.suggest_float('reg_alpha',0.0,50),
    'reg_lambda':trial.suggest_float('reg_lambda',0.0,50),
    "min_split_gain": trial.suggest_float("min_split_gain", 0.0,20),
    'subsample_freq' : trial.suggest_categorical("subsample_freq", [1]),
            }
    
    # to store scores
    results=[]
    for train_idx,val_idx in cv.split(xtrain,ytrain.values):
        x_train,x_val=xtrain.iloc[train_idx],xtrain.iloc[val_idx]
        y_train,y_val=ytrain.iloc[train_idx],ytrain.iloc[val_idx]
        
        x_train,x_val=data_preprocessing(x_train,x_val)

        
        model_1=LGBMClassifier(**params)
        model_1.fit(
               x_train.values,y_train.values,
               eval_set=[(x_val.values,y_val.values)],
               eval_metric='binary_logloss',
        callbacks=[lightgbm.early_stopping(100,verbose=0),LightGBMPruningCallback(trial, "binary_logloss")]
        )
    
        pred=model_1.predict(x_val.values)
        score=log_loss(y_val.values,pred)
        results.append(score)
        del x_train,x_val,y_train,y_val
        gc.collect()
    return np.mean(results)


# In[66]:


optuna.logging.set_verbosity(optuna.logging.WARNING)


# In[67]:


get_ipython().run_cell_magic('time', '', 'study = optuna.create_study(direction="minimize")\nstudy.optimize(objective, n_trials=100,show_progress_bar=True,n_jobs=1)\nprint("Number of finished trials: {}".format(len(study.trials)))\n')


# In[75]:


trial = study.best_trial
best_params_lgbm=trial.params
study.best_value


# In[76]:


text="<h3 style='color:blue'>"+f"Best Params :<br><br><pre>{best_params_lgbm:}"+"</h3>"
display(HTML(text))


# In[77]:


x_train,x_test=data_preprocessing(xtrain,xtest)
model_1=LGBMClassifier(**best_params_lgbm).fit(x_train.values,ytrain.values)
print(f"Train Accuracy:{model_1.score(x_train.values,ytrain.values)}")
print(f"Test Accuracy:{model_1.score(x_test.values,ytest.values)}")


# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Hyperparameter tunning XGBoost using Optuna</span>
# 

# In[78]:


def objective(trial):
    
    params = {
        
   'objective': 'binary:logistic',
    'tree_method':'gpu_hist',
    'subsample':trial.suggest_float("subsample",0.4,0.9,step=0.1),
    'min_child_weight':trial.suggest_int("min_child_weight",30,500),
    'colsample_bytree':trial.suggest_float("colsample_bytree",0.4,0.9,step=0.1),
    'learning_rate':trial.suggest_categorical("learning_rate",[0.01,0.03,0.06,0.08]),   
    "max_depth": trial.suggest_int("max_depth", 5,15,1),
    'reg_alpha':trial.suggest_float('reg_alpha',0.0,100),
    'reg_lambda':trial.suggest_float('reg_lambda',0.0,100),
    "gamma": trial.suggest_float("gamma", 0,5),
    'eval_metric': 'logloss'
            }
    
    # to store scores
    results=[]
    for train_idx,val_idx in cv.split(xtrain,ytrain.values):
        x_train,x_val=xtrain.iloc[train_idx],xtrain.iloc[val_idx]
        y_train,y_val=ytrain.iloc[train_idx],ytrain.iloc[val_idx]
        
        x_train,x_val=data_preprocessing(x_train,x_val)

        
        
        xg_train=xgb.DMatrix(x_train.values,y_train.values)
        xg_val=xgb.DMatrix(x_val.values,y_val.values)
        
        clf=xgb.train(params,xg_train,num_boost_round=2000,evals=[(xg_val,'val')],early_stopping_rounds=100,verbose_eval=400)
    
        pred=clf.predict(xg_val)
        score=log_loss(y_val.values,pred)
        results.append(score)
        del x_train,x_val,y_train,y_val
        gc.collect()
    return np.mean(results)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'study = optuna.create_study(direction="minimize")\nstudy.optimize(objective, n_trials=50,)\nprint("Number of finished trials: {}".format(len(study.trials)))\n')


# In[80]:


trial = study.best_trial
best_params_xgb=trial.params
study.best_value


# In[81]:


text="<h3 style='color:blue'>"+f"Best Params :<br><br><pre>{best_params_xgb:}"+"</h3>"
display(HTML(text))


# In[83]:


best_params_xgb['eval_metric']='logloss'
best_params_xgb['objective']= 'binary:logistic'
best_params_xgb['tree_method']='gpu_hist'


# In[84]:


x_train,x_test=data_preprocessing(xtrain,xtest)
model_1=XGBClassifier(**best_params_xgb).fit(x_train.values,ytrain.values)
print(f"Train Accuracy:{model_1.score(x_train.values,ytrain.values)}")
print(f"Test Accuracy:{model_1.score(x_test.values,ytest.values)}")


# <h3>  <b style='color:#AB47BC;font-size:22px;'>Inference </b>:
#  <ul>
#      <li>LightGBM gave better performance than XGBoost.</li>
#     </ul>
# </h3>

# <span style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:30px 60px;font-weight:bold;border:2px solid #AB47BC;padding:0px 20px">Prediction on Test Dataset</span>

# In[85]:


results=[]
for train_idx,val_idx in tqdm(cv.split(X,y.values)):
    x_train,x_val=X.iloc[train_idx],X.iloc[val_idx]
    y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
        
    x_train,x_val=data_preprocessing(x_train,x_val)    

    model_1=LGBMClassifier(**best_params)
    model_1.fit(
               x_train.values,y_train.values,
               eval_set=[(x_val.values,y_val.values)],
               eval_metric='binary_logloss',
        callbacks=[lightgbm.early_stopping(100,verbose=0)]
        )
    del x_train,x_val,y_train,y_val
    gc.collect()
    pred=model_1.predict_proba(test_df.values)
    results.append(pred)


# In[ ]:


# results=[]
# for train_idx,val_idx in tqdm(cv.split(X,y.values)):
#     x_train,x_val=X.iloc[train_idx],X.iloc[val_idx]
#     y_train,y_val=y.iloc[train_idx],y.iloc[val_idx]
        
#     x_train,x_val=data_preprocessing(x_train,x_val)    

    
#     xg_train=xgb.DMatrix(x_train.values,y_train.values)
#     xg_val=xgb.DMatrix(x_val.values,y_val.values)
        
#     clf=xgb.train(best_params_xgb,xg_train,num_boost_round=3000,evals=[(xg_val,'val')],early_stopping_rounds=100,verbose_eval=400)
    
        
#     del x_train,x_val,y_train,y_val
#     gc.collect()
#     xg_test=xgb.DMatrix(test_df.values)
#     pred=clf.predict(xg_test)
#     results.append(pred)


# In[ ]:


# len(results),results[0].shape


# In[ ]:


# results[0]


# In[87]:


final_pred=(sum(results)/5).argmax(axis=1)
# final_pred=(sum(results)/5)


# In[ ]:


# plot_optimization_history(study)


# In[ ]:


# plot_param_importances(study)


# In[88]:


sub_df['Transported']=label_encoder.inverse_transform(final_pred)
# sub_df['Transported']=label_encoder.inverse_transform(np.round(final_pred).astype('int'))


# In[89]:


sub_df.head()


# In[90]:


sub_df.to_csv('submission.csv',index=False)


# <h2 style='text-align:center;color:#FF3355;font-weight:bold'> Do share your feedback in the comments section,I hope you found it to be helpful.🙌</h2>
# 

# <p style="background-color:#AB47BC;color:white;font-size:22px;text-align:center;border-radius:10px 10px;font-weight:bold;border:2px solid #AB47BC;">Thank you😄!!!!!!</p>
