#!/usr/bin/env python
# coding: utf-8

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⛵ Titanic-Machine Learning From Disaster ⛵</b></div>

# <h3 align="center" style="font-size: 35px; color: #800080; font-family: Georgia;">
#     <span style="color: #008080;"> Author:</span> 
#     <span style="color: black;">Kumod Sharma .📄🖋️</span>
# </h3>

# <div align="center">
#   <img src="https://miro.medium.com/v2/resize:fit:1358/1*nNcgXmjqRLF16e1jhiGDlg.jpeg" alt="Image Description" width="1200px" height="300px">
# </div>

# ----

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 🎬 Introduction 🎬</b></div>

# <div style="border-radius:10px;border:black solid;padding: 15px;background-color:white;font-size:110%;text-align:left">
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:17px">
# 
#    
#   
# <h3 align="left"><font color=purple>📝 Project Objective:</font></h3><br> 
# 
# 1. This project revolves around <b>performing binary classification</b>, where we aim to categorize passengers into <b>two groups: those who survived the Titanic shipwreck and those who did not.</b><br>
#     
# 2. We will leverage the Titanic Survival dataset provided by Kaggle, which <b>contains information about passengers, including their attributes and survival outcomes.</b><br>
# 3. The primary objective of this project is to <b>harness the power of machine learning</b> to create a predictive model.<br>
# 4. This <b>model will be designed to predict which passengers among those on board the Titanic survived the tragic shipwreck.</b><br>
# 5. By achieving this objective, we aim to <b>gain insights into the factors and attributes that influenced survival rates</b> during this historic event.<br>
# 6. Ultimately, our <b>goal is to create a reliable and accurate model that can predict survival outcomes for passengers</b> based on the available data, helping us understand the dynamics of this disaster more comprehensively.<br></div></div>

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 📝 Project Content 📝</b></div>

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color:##F0E68C ; font-size:100%; text-align:left">
# 
# <div style="font-family:Georgia;background-color:'#DEB887'; padding:30px; font-size:17px">
# 
# <h3 align="left"><font color=brown>📊 Table of Contents:</font></h3><br>
# 
# 1. <b>📚 Importing Libraries: </b>Import all the essentials libraries for Data Manipulation,Visualization & Data Analysis.<br>
#     
# 2. <b>⏳ Loading Datasets: </b> Load the dataset into a suitable data structure using pandas.<br>
# 3. <b>🧠 Basic Understanding of Data: </b>To Understand the basic information of the data for better analysis.<br>
# 4. <b>📊 Exploratory Data Analysis: </b>To gain insights, discover patterns, and understand the characteristics of the data before applying further analysis.<br>
# 5. <b>💡 Feature Engineering: </b> To create new relevant features to generate more actionable insights.<br>
# 6. <b>📈 Statistical Analysis: </b>To assess the significance and impact of different features on the target variable, identify the most important variables.<br>
# 7. <b>⚙️ Data Preprocessing: </b>To clean, transform, and restructure the data in order to make it suitable for analysis and model building.<br>
# 8. <b>🎯 Model Creation and Evaluation: </b>To create ML model and evaulate the model performance using different metrics.<br>
# 9. <b> 🎈 Conclusion: </b>Conclude the project by summarizing the key findings and limitations related to employee attrition.<br></div></div>

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b> 📚 Importing Libraries 📚</b></div>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid",font_scale=1.5)
sns.set(rc={"axes.facecolor":"#FFFAF0","figure.facecolor":"#FFFAF0"})
sns.set_context("poster",font_scale = .7)
pd.set_option("display.max.rows",None)
pd.set_option("display.max.columns",None)

from scipy import stats as st
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from imblearn.over_sampling import SMOTE


# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>⏳ Loading Datset ⏳</b></div>

# In[2]:


train_df = pd.read_csv("../input/titanic/train.csv")
test_df = pd.read_csv("../input/titanic/test.csv")


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>Feature- </b> Description </h2>

# > * **PassengerId:** This feature indicates the Id of each passengers.
# > * **Survived:** this features indicates whether the passenger has survived or not. **0 means Not- Survived** & **1 means Survived**
# > * **Pclass:** A proxy for socio-economic status (SES) **1st = Upper**, **2nd = Middle**, **3rd = Lower**
# > * **Name:** This feature is indicating the names of individual passengers.
# > * **Sex:** This feature is indicating the gender of the passengers.
# > * **Age:** this feature is indicating the age of individual passengers.
# > * **SibSp:** The dataset defines family relations in this way...**Sibling** = brother, sister, stepbrother, stepsister,  **Spouse** = husband, wife (mistresses and fiancés were ignored)
# > * **Parch:** The dataset defines family relations in this way. **Parent: [Mother or Father]**, **Child: [Daughter, Son, Stepdaughter, Stepson]**.
#             Some children travelled only with a nanny, therefore parch=0 for them.
# > * **Ticket:** This feature is showing the Ticket Number of each passengers.
# > * **Fare:** The amount paid by the passenger to get the Ticket.
# > * **Cabin:** This feature is indicating the cabin deck & number of individual passengers.
# > * **Embarked:** Embarked implies where the traveler mounted from. There are three possible values for Embark — **Southampton, Cherbourg, and Queenstown**. More than **70% of the people boarded from Southampton**

# ---

# <a id="1"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Verdana;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>🧠 Basic Understanding of Data 🧠</b></div>

# ### 1. Checking shape of both Training & Testing Datasets.

# In[3]:


print("The shape of Training Datasets is:",train_df.shape)
print("The shape of Testing Datsets is:",test_df.shape)


# ----

# ### 2. Showing Training & Testing Data.

# In[4]:


train_df.head().style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# In[5]:


test_df.head().style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We can observe that we don't have <b>Survived</b> feature in our <b>Test Dataset.</b><br>
# * So we have to build a model using Training Data and <b>predict</b> that feature for our Testing Data.<br>

# -----

# ### 3. Getting Basic Information of Data

# In[6]:


train_df.info()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We can observe that only <b>Name, Sex, Ticket, Cabin & Embarked</b> features are having data-type <b>object</b>.<br>
# * But in reality <b>Survived & Pclass</b>  stores categorical values.<br>
# * So befor <b>EDA</b> we will <b>replace</b> those values with their respective categorical values to <b>generate more accurate insights.</b><br>

# ---

# ### 4. Descriptive Statistical Analysis of Numerical Features on Training Data.

# In[7]:


train_df.describe(percentiles=[0.25,0.50,0.60,0.65,0.70,0.75,0.80]).T.style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * The **Survived feature** stores **almost 60 percentile** of data as **0**, which means most **more than half passengers didn't survived.**<br>
# * In Age feature we can observe **50% & 75% percentiles and mean value close to 30**, so most of the **passengers were near top age 30.**<br>
# * The **SibSp feature stores almost 60% percentile of data as 0**, so **most of the passengers are travelling without their Siblings or Spouse.**<br>
# * The **Parch feature stores almost 75% percentile of data as 0**, so **most of the passengers are travelling without their Parent or Childen.**<br>
# * There's **huge difference between mean and maximum value of fares**, so we can say that the **distribution of the data must be right-skewed.**<br>

# ----

# 
# ### 5. Checking Cardinality of Categorical Features in  Both Datasets.

# In[8]:


unique = train_df.select_dtypes("object").nunique().to_frame().rename(columns={0:"Cardiniality in Training Data"})
unique["Cardiniality in Testing Data"] = test_df.select_dtypes("object").nunique()
unique.style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * Features like **Name, Ticket & Cabin** are having **high cardniality.**
# * We **normally drop features** with high cardinality but in this project I have tried to do **Feature Engineeering on those features.**<br>
# * So that we can **avoid Data Loss** because more amount of data leads to more accurate prediction by the model.<br>

# ------

# ### 6. Checking Duplicates in our Datasets

# In[9]:


print(f"Duplicate in Training Data is:{train_df.duplicated().sum()},({100*train_df.duplicated().sum()/len(train_df)}%)")
print(f"Duplicate in Testing Data is:{test_df.duplicated().sum()},({100*test_df.duplicated().sum()/len(test_df)}%)")


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We can clearly observe that we **don't have any duplicate values** in our both training & testing datasets.<br>
# * So we can say that we don't have any type of **Data lekage** in our both datasets.<br>

# -----

# ### 7. Checking Total Number & Percentage of Missing Values in Training Datasets.

# In[10]:


df = train_df.isnull().sum()[train_df.isnull().sum()>0].to_frame().rename(columns={0:"Number of Missing Values"})
df["% of Missing Values"] = round(df["Number of Missing Values"]*100/len(train_df),2)
df.style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * The **Cabin Feature** is having **more than 77% of missing values.** So we will **drop** this cabin feature.<br>
# * The **Age Feature** is also having **almost 20% of missing values** but we **can't drop** this feature as this feature might be **relevant to target variable.**<br>
# * The **Embarked Feature** is having only **2 missing values** so we will **simply impute** those missing values.<br>

# ----

# ### 8. Checking Total Number & Percentage of Missing Values in Testing Datasets.

# In[11]:


df = test_df.isnull().sum()[test_df.isnull().sum()>0].to_frame().rename(columns={0:"Number of Missing Values"})
df["% of Missing Values"] = round(df["Number of Missing Values"]*100/len(test_df),2)
df.style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * The **Cabin Feature** is having **more than 78% of missing values.** So we will **drop** this cabin feature.<br>
# * The **Age Feature** is also having **more than 20% of missing values** but we **can't drop** this feature as this feature might be **relevant to target variable.**<br>
# * The **Fare Feature** is having only **1 missing values** so we will **simply impute** that missing value.<br>

# ---

# ### 9. Dropping Features with High Missing Values & High Cardiniality.

# In[12]:


train_df.drop(columns=["Cabin","Name","Ticket"],inplace=True)
test_df.drop(columns=["Cabin","Name","Ticket"],inplace=True)


# ### 10. Replacing Discrete Values with Categorical Values for better Visualization.

# In[13]:


train_df["Survived"].replace({0:"Not-Survived",1:"Survived"},inplace=True)

train_df["Pclass"].replace({1:"Upper_Class",2:"Middle_Class",3:"Lower_Class"},inplace=True)
test_df["Pclass"].replace({1:"Upper_Class",2:"Middle_Class",3:"Lower_Class"},inplace=True)


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We know that in **Survived feature 0 means not-Survived & 1 means Survived**<br>
# * Similarly, we know that in **Pclass feature 1 means Upper, 2 means middle and 3 means lower.**<br>
# * So for better visualization let's **replace** those numerical values with their original values so that we can **generate more accurate insights.**<br>

# -----

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Exploratory Data Analysis</b></div>

# ## 1. Visualizing Target Variable "Survived Column".

# In[14]:


survived = train_df["Survived"].value_counts()

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Rate of Passenger Survived",fontweight="black",size=16,pad=20)
plt.pie(survived.values, labels=survived.index, autopct="%.2f%%", textprops={"fontweight":"black","size":15},
        colors = ["#AC1F29","#1d7874"])
white_circle = plt.Circle((0,0),0.3,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)


plt.subplot(1,2,2)
ax=sns.barplot(y=survived.values,x=survived.index,palette=["#8B0000","#1d7874"])
ax.bar_label(ax.containers[0],fontweight="black",size=15)
plt.title("No. of Passengers Survived",fontweight="black",size=15,pad=20)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **More than 60% of passengers** was **not able to survive** during the **titanic disaster.**<br>
# * The **ratio** of survived to non-survived passengers is **60:40 approx**, so we can **can't consider the target varible as imbalanced**<br>

# ---

# ## 2. Visualizing Pclass Feature w.r.t Survived Feature.

# In[15]:


plt.figure(figsize=(13.5,6))
ax = sns.countplot(x="Pclass" ,hue="Survived",data=train_df,palette="RdPu")
plt.title("Passenger Survived w.r.t their Pclass",fontweight="black",size=18,pad=20)
for container in ax.containers:
    ax.bar_label(container,fmt="%.1f%%",fontweight="black",size=15,
                labels=[f'{height / len(train_df) * 100:.1f}%' for height in container.datavalues])
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * Out of **55% of lower class passenger, 42% of passengers were not able to survive only 13% of passenger survived**, which is **huge difference** in survived and not-survived passengers who were **travelling in lower class.**
# * Out of **24% of Upper class passenger, 15% of passengers survived and 9% was not able to survive**, so we can say **almost double passengers survived in upper class.**
# * The **ratio** of survived to not-survived **passengers in middle class is approxiamately equal.**
# * Since **Survived feature is being affected by Pclass feature too much**, so this feature **seems relevant for model training.**

# ---

# ## 3. Visualizing Sex Feature w.r.t Survived Feature.

# In[16]:


survived = train_df[train_df["Survived"]=="Survived"]
not_survived = train_df[train_df["Survived"]=="Not-Survived"]

count_1 = survived["Sex"].value_counts()
count_2 = not_survived["Sex"].value_counts()


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.pie(count_1.values, labels=count_1.index, autopct="%.f%%",textprops={"fontweight":"black","size":15},colors = ['#6495ED', '#87CEEB'])
plt.title("Gender Wise Survived Passengers",fontweight="black",size=18,pad=10)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)

plt.subplot(1,2,2)
plt.pie(count_2.values, labels=count_2.index, autopct="%.f%%",textprops={"fontweight":"black","size":15},colors = ['#FF8000','#FFB366'])
plt.title("Gender Wise Not-Survived Passengers",fontweight="black",size=18,pad=10)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
#     
# * The **survival ratio of male & female is almost 7:3**, which is more than **double for female compared to male passengers.**<br>
# * The **Non-Survival ratio of male & female is almost 17:3**, which is **more than five times male compared to female.**<br>
# * So the **male passengers have very low chances of survival**, where as the **female passengers have very high chances of survival.** <br>

# ---

# ## 4. Visualizing Age Feature w.r.t Survived Feature.

# In[17]:


male_data = train_df[train_df['Sex'] == 'male']
female_data = train_df[train_df['Sex'] == 'female']

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.histplot(x="Age", hue="Survived", data=male_data, kde=True,palette="Blues")
plt.title("Male Passengers vs Age",fontweight="black",size=15,pad=15)
plt.xticks(list(range(0,81,5)))

plt.subplot(1,2,2)
sns.histplot(x="Age", hue="Survived", data=female_data, kde=True,palette="RdPu")
plt.title("Female Passenegers vs Age",fontweight="black",size=15,pad=15)
plt.xticks(list(range(0,61,5)))
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We can easily observe that **most of the passenger were between 20 to 35 ages**.<br>
# * We can observe that **age from 0-10 are highly Survived when compared with Not Survived.** Specially for small childrens.<br>
# * We can observe that **age from 11 to 65 are comparatively less Survived when compared with Not Survived.**<br>
# * We can observe that passengers having **age more than 65 have negligible chance of being Survival.**<br>
# * We can do **Feature Engineering** to create a new feature Age Category **by splitting age into different categories.**<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>5. Visualizing- </b> SibSp Feature </h2>

# In[18]:


grouped_data = train_df.groupby(['SibSp', 'Survived']).size().reset_index(name='Count')

plt.figure(figsize=(15, 6))

ax = sns.barplot(x='SibSp', y='Count', hue='Survived', data=grouped_data, palette=["#FFCCCB", "#87CEEB"], ci=None)
ax.bar_label(ax.containers[0],fontweight="black",size=15,color="brown")
ax.bar_label(ax.containers[1],fontweight="black",size=15,color= "blue")
plt.title('SibSP Count Distribution by Survived (Log Scale)',fontweight="black",size=15,pad=15)
plt.legend(title='Survived', loc='upper right')
plt.yscale('log')  # Use a logarithmic scale on the y-axis
plt.ylabel('Count (log scale)')
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **Passengers travelling with 0 sibling/spouse have almost half chance of Survival.**<br>
# * **Passengers travelling with 1 sibling/spouse have higher chance of Survival.**<br>
# * **Passengers travelling with 2 sibling/spouse have almost equal chance of Survival.**<br>
# * **Passengers travelling with more than 2 sibling/spouse have almost negligible chance of Survival.**<br>
# 
# <b>Note: </b><br>
# 
# * We can do **Feature Engineering** to create a new feature SibSp Category in which we can indicate the following things.<br>
# 1. **If the passengers is travelling with 0 sibling/spouse than we can categorize them as No Sibling/Spouse.**<br>
# 2. **If the passengers is travelling with 1 or 2 sibling/spouse than we can group them together as Average Sibling/Spouse.**<br>
# 3. **If the passengers is travelling with more than 2 sibling/spouse than we an group them as Extra Sibling/Spouse.**<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>6.Visualizing- </b> Parch Feature </h2>

# In[19]:


grouped_data = train_df.groupby(['Parch', 'Survived']).size().reset_index(name='Count')

plt.figure(figsize=(15, 6))

ax = sns.barplot(x='Parch', y='Count', hue='Survived', data=grouped_data, palette=["#FFCCCB", "#90EE90"], ci=None)
ax.bar_label(ax.containers[0],fontweight="black",size=15,color="brown")
ax.bar_label(ax.containers[1],fontweight="black",size=15,color= "green")
plt.title('Parch Count Distribution by Survived (Log Scale)',fontweight="black",size=15,pad=15)
plt.legend(title='Survived', loc='upper right')
plt.yscale('log')  # Use a logarithmic scale on the y-axis
plt.ylabel('Count (log scale)')
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
#     
# * If the **passengers is travelling with 0 parent/children than they have almost half chance of Survival.**<br>
# * If the **passengers is travelling with 1 or 2 parent/children than they have almost equal chane of Survival.**<br>
# * We **can't make any observation for passengers travelling with more than 2 parent/children because there are very few datapoints.**<br>
# 
# <b>Note: </b><br>
# 
# * We can do **Feature Engineering** and can create a new feature Parch category which will indicate following thing:<br>
# 1. If the **passengers is travelling with 0 parent/children than we can group them as one category No Parents/Children.**<br>
# 2. If the **passengers is travelling with 1 or 2 parent/children than we can group them as Average Parents/Children.**<br>
# 3. Since we **couldn't make any observation for passengers travelling with more than 2 parents/children so we can group them as one category Extra parents/Children.**<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>7. Visualizing- </b> Fare Feature </h2>

# In[20]:


fare_50 = train_df[train_df["Fare"]<51]
fare_500 = train_df[train_df["Fare"]>50]


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.histplot(x="Fare",data=fare_50,hue="Survived",kde="True",palette="Blues")
plt.title("Fare Charges ($ 0-50) vs Survival",fontweight="black",size=15,pad=15)

plt.subplot(1,2,2)
sns.histplot(x="Fare",data=fare_500,hue="Survived",kde="True",palette="Oranges")
plt.title("Fare Charges ($ 51-500) vs Survival",fontweight="black",size=15,pad=15)


plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **Passengers having Fare charge between 0 to 5 are having almost negligible chance of Survival when compared with Non Survived Passenegers.**<br>
# * **Passengers having Fare charge between 6 to 40 are having almost equal chance of Survival when comapred with Non Survived Passenegrs.**<br>
# * **Passengers having Fare charge greater than 50 are having very high chance of Survival when comparedd with Non Survived Passenergs.**<br>
# 
# <b>Note: </b><br>
# 
# * **Passengers having Fare charge between 41 to 50 are having No Survived Passengers which can be taken as exception case.**<br>
# * Some of the **Passengers paid fare around 500 due to which the fare distribution is right-skewed so we have to transform it to normal-distribution.**<br>
# * We can do **Feature Engineering** and can create a new feature Fare Range in which we can **split fare into different categories.**<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>8. Visualizing- </b> Embarked Feature </h2>

# In[21]:


embarked = train_df["Embarked"].value_counts()
labels = ["Southampton","Cherbourg","Queenstown"]

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.pie(x=embarked.values,labels=labels,startangle=30,autopct="%.f%%",textprops={"fontweight":"black","size":15},colors=sns.color_palette("Blues"))
plt.title("Embarked Feature Distribution",fontweight="black",size=15,pad=15)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)


plt.subplot(1,2,2)
ax = sns.countplot(x="Embarked",data=train_df,hue="Survived",palette="Blues")
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.title("Embarked vs Survival",fontweight="black",size=15,pad=15)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
#     
# * From pie-chart we can observe that **72% of passengers are from Southampton.**<br>
# * From countplot we can observe following things:<br>
# 1. **If the passengers is from Southampton than they have almost half chance of Survival.**<br>
# 2. **If the passengers is from Cherbourg than they have High chance of Survival.**<br>
# 3. **If the passengers is from Queenstown than they have almost equal or little less chance of Survival.**<br>

# ----

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Feature Engineering</b></div>

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">1. Creating New Feature - <b> Family-Size </b></h2> 

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
# 
# 1. We know that **passengerId stores the information of a single passenger.**<br>
# 2. We also know that **SibSp stores the values indicating number of siblings or spouse the passenger is travelling with.**<br>
# 3. We also know that **Parch stores the values indiccating number of parents or childrens the passenger is travelling with.**<br>
# 4. So we create a **new feature family_size by adding all the values of SibSp & Parch features of each individual passengers.**<br>

# In[22]:


#Displaying a sample of Data with required Columns.

train_df[["PassengerId","SibSp","Parch"]].sample(5)


# In[23]:


#Creating "Family_Size" Feature in Training Dataset and Testing Dataset...

train_df["Family_Size"] = (train_df["SibSp"] + train_df["Parch"] + 1)
test_df["Family_Size"] = (train_df["SibSp"] + train_df["Parch"] + 1)


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> Family_Size </b>Feature</h2> 

# In[24]:


plt.figure(figsize=(15,6))
ax=sns.countplot(x="Family_Size",hue="Survived",data=train_df,palette=["#8B0000","#1d7874"])
ax.bar_label(ax.containers[0],fontweight="black",size=15,color="#8B0000")
ax.bar_label(ax.containers[1],fontweight="black",size=15,color="#1d7874")
plt.title("Family_Size vs Survived",fontweight="black",size=20,pad=15)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# 1. We can observe that **more than 50% of Passenegrs are travellig Solo** and the **maximum family size is of 11 members.**<br>
# 2. **Passengers travelling Solo has very low chances of Survival** when comapred to solo non-survival passenegers.<br>
# 3. **Passengers traveeling with 2 to 3 members has high chances of Survival** comapred to non-survial passenegers.<br>
# 4. We **Can"t make any inference** fro passengers travelling with **more than 4 members** because there are less records availabe to generate insights.<br>
# 5. Since there is **class-imbalance** between categories we can do **further feature engineering by creating a new feature Solo_Traveller** which can **indicate whether the passenger is travelling alone or with family.**<br>

# ****

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">2. Creating New Feature - <b> Solo-Traveller </b></h2> 

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
# 
# 1. We can create a **new feature Solo_traveller using Family_Size feature which will store boolean values.**<br>
# 2. **If the passenger is travelling alone than True** and **If the passenger is travelling with their families than False.**<br>

# In[25]:


# Creating New Feature "Solo Traveller" on Training Dataset and Testing Dataset...

train_df["Solo_Traveller"] = (train_df["Family_Size"]==1)
test_df["Solo_Traveller"] = (test_df["Family_Size"]==1)


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> Solo-Traveller </b>Feature</h2> 

# In[26]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
z = train_df["Solo_Traveller"].value_counts().to_frame()
plt.pie(z["Solo_Traveller"], labels=z.index, autopct="%0.f%%", textprops={"fontweight":"black","size":15},colors=sns.color_palette("pastel"))
plt.title("Solo Traveller Distribution",fontweight="black",size=15,pad=15)

plt.subplot(1,2,2)
ax=sns.countplot(x="Solo_Traveller",hue="Survived",data=train_df,palette="pastel")
plt.title("Solo_Traveller vs Survival",fontweight="black",size=15,pad=15)
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * We can observe that **60% of passengers are travelling solo and 40% of passengers are travelling with atleast 1 family members.**<br>
# * **Passengers Travelling Solo has very low chance of survival** compared to non-survival passengers.<br>
# * **Passengers Travelling with atleast 1 family member are having almost equal chance of survival**,compared to non-survival passenegers.<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">3. Creating New Feature - <b> Age-Category </b></h2> 

# In[27]:


# Performig Descriptive Analysis on AGE Feature.

train_df["Age"].agg(["min","median","mean","std","max"]).to_frame().T


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
#   
# 1. If the **age is between 0 to 12** than we can group them as one category **Children**<br>
# 2. If the **age is between 12 to 20** than we can group them as one category **Teenager**<br>
# 3. If the **age is between 20 to 50** than we can group them as one category **Adults**<br>
# 4. If the **age is above 50** than we can group them as one category **Senior Citizens**<br>

# In[28]:


# Creating New Feature "Age_Category" in Training and Testing Datasets...

train_df["Age_Category"] = pd.cut(train_df["Age"], bins=[0,12,20,50,100], labels=["Children","Teenager","Adult","Senior Citizen"])
test_df["Age_Category"] = pd.cut(test_df["Age"], bins=[0,12,20,50,100], labels=["Children","Teenager","Adult","Senior Citizen"])


# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> Age-Category </b>Feature</h2> 

# In[29]:


plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
z = train_df["Age_Category"].value_counts().to_frame()
plt.pie(z["Age_Category"], labels=z.index, autopct="%0.f%%",startangle=60,textprops={"fontweight":"black","size":15}, colors=sns.set_palette("pastel"))
plt.title("Age Categories Distribution",fontweight="black",size=15,pad=15)

plt.subplot(1,2,2)
ax=sns.countplot(x="Age_Category",hue="Survived",data=train_df)
plt.title("Age_Category vs Survived",fontweight="black",size=15,pad=15)
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * Most of the **Passengers are Adult** followed by the **Teenager passengers.**<br>
# * But the **chance of survival** for Adult and Teenager Passengers are **very low.**<br>
# * The **chance of survival** for **children are high** where as for **senior citizen its almost half.**<br>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">4. Creating New Feature - <b> SibSp-Category </b></h2> 

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
# 
# * Recall the insights we gained while doing **EDA on SibSp Feature**, we came to know that we can **group sibsp column on the below criteria.**<br>
#     
#     1. If the passengers is travelling with **0 sibling/spouse** than we can categorize them as **No Sibling/Spouse.**<br>
#     2. If the passengers is travelling with **1 or 2 sibling/spouse** than we can group them together as **Average Sibling/Spouse.**<br>
#     3. If the passengers is travelling with **more than 2 sibling/spouse** than we an group them as **Extra Sibling/Spouse.**<br></div>

# In[30]:


## Creating Function so that we can create new feature "SibSp category" for both Training & Testing Datasets.

def sibsp_category(df):
    sibsp = []
    
    for i in df["SibSp"]:
        if i==0:
            sibsp.append("No Sibling/Spouse")
        elif i==1 or i==2:
            sibsp.append("Average Sibling/Spouse")
        else:
            sibsp.append("Extra Sibling/Spouse")
    df["SibSp_Category"] = sibsp


# In[31]:


## Calling the SibSp Category Function..

sibsp_category(train_df)
sibsp_category(test_df)


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> SibSp-Category </b>Feature</h2> 

# In[32]:


sibsp = train_df["SibSp_Category"].value_counts()


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.pie(x=sibsp.values,labels=sibsp.index, autopct="%0.f%%",textprops={"fontweight":"black","size":15},colors=sns.set_palette("Set2"))
plt.title("SibSp_Categories Distribution",fontweight="black",size=18,pad=20)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)


plt.subplot(1,2,2)
ax=sns.countplot(x="SibSp_Category",hue="Survived", data=train_df,palette="Oranges")
plt.title("SibSp Category Countplot",fontweight="black",size=18,pad=20)
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.xticks(size=12)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * Most of the **Passengers are travelling without siblings or spouse.**<br> 
# * If the passengers has **no siblings or no spouse** than they have **almost half chances of Survival.**<br>
# * If the passengers has **average siblings/spouse** than they have **very High chances of Survival.**<br>
# * If the passengers has **Extra siblings/spouse** than they have **negligible chances of Survival.**<br></div>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">5. Creating New Feature - <b> Parch-Category </b></h2> 

# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
# 
# * Recall that we gained some insights while doing EDA on Parch Feature.We gained that we can create a new feature Parch category.<br>   
#     1. If the passengers is travelling with **0 parents/children** than we can group them as one category **Without Parents/Children.**<br>
#     2. If the passengers is travelling with **1 or more parents/children** than we can group them as **With Parents/Children.**<br></div>

# In[33]:


## Creating a function so that we can create new feature for both Training & Testing Dataset at once

def parch_category(df):
    parch = []    
    for i in df["Parch"]:
        if i==0:
            parch.append("Withot Parents/Children")
        else:
            parch.append("With Parents/Children")
        
    df["Parch_Category"] = parch


# In[34]:


## Calling the Parch Category Function...

parch_category(train_df)
parch_category(test_df)


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> Parch-Category </b>Feature</h2> 

# In[35]:


parch = z = train_df["Parch_Category"].value_counts().to_frame()

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.pie(parch["Parch_Category"],labels=parch.index, autopct="%0.f%%",textprops={"fontweight":"black","size":13})
plt.title("Parch Categories Distribution",fontweight="black",size=15)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)


plt.subplot(1,2,2)
ax=sns.countplot(x="Parch_Category",hue="Survived",data=train_df,palette="Blues")
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.title("Parch Category Countplot",fontweight="black",size=15,pad=15)
plt.xticks(size=10)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **Most of the passengers** are travelling **withoout Parents or children.**<br>
#     1. If the passenger is having **Without Parents/Childrens** then they have **half chances of Survival.**<br>
#     2. If the passenger is having **With Parents/Childrens** then they have **equal chances of Survival.**<br></div>

# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">6. Creating New Feature - <b> Fare-Range </b></h2> 

# In[36]:


# Descriptive Analysis on Fare Feature...

train_df["Fare"].agg(["min","median","mean","std","var","max"]).to_frame().T


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; font-family:Georgia; padding:30px; font-size:17px">
# 
# <b>💡 Steps for feature engineering</b><br>
# 
# * We can create a new feature **Fare Range by splitting Fare into different ranges**.<br>
# * NOTE: We have **1 missing value** in our testing data so we will keep that in mind while doing featrue engineering.<br>
#     1. Passengers having Fare charge between **0 to 14.45** can be grouped as **Low Fare Group.**<br>
#     2. Passengers having Fare charge between **14.45 to 32.20** can be grouped as **Moderate Fare Group.**<br>
#     3. Passengers having Fare charge **greater than 32.20** can be grouped as **High Fare Passengers.**<br></div>

# In[37]:


# Creating a Function so that we can create new feature on both Training & Testing Datasets at once...


def fare_range(df):
    fare = []
    
    for i in df["Fare"]:
        if i <= 14.45:
            fare.append("Low Fare Group")
        elif i <= 32.20:
            fare.append("Moderate Fare Group")
        elif i>32.20:
            fare.append("High Fare Group")
        else:
            fare.append(np.nan)       #Because of NaN value present in test dataset.++
    
    df["Fare_Range"] = fare


# In[38]:


# Calling the Fare_Range Function...

fare_range(train_df)
fare_range(test_df)


# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;">Visualizing- <b> Fare-Range </b>Feature</h2> 

# In[39]:


fare_range = train_df["Fare_Range"].value_counts().to_frame()

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.pie(fare_range["Fare_Range"], labels=fare_range.index, autopct="%0.f%%",textprops={"fontweight":"black","size":13},colors=sns.set_palette("Blues"))
plt.title("Fare Range Categories",fontweight="black",size=18,pad=20)
white_circle = plt.Circle((0,0),0.4,fc="white")
fig=plt.gcf()
fig.gca().add_artist(white_circle)

plt.subplot(1,2,2)
ax=sns.countplot(x="Fare_Range",hue="Survived",data=train_df,palette="Oranges")
ax.bar_label(ax.containers[0],fontweight="black",size=15)
ax.bar_label(ax.containers[1],fontweight="black",size=15)
plt.title("Fare Range vs Survived",fontweight="black",size=18,pad=15)
plt.xticks(size=12)
plt.tight_layout()
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **Almost 50%** of passengers were travelling with **low fares** and **other 50%** were travelling with **moderate to high fares** combined.<br>
# * **Fare feature is higly positively correlated with survived column.**<br>
#     1. Passengers with **Low Fare Charge** has **very low chance of survival.**<br>
#     2. Passengers with **Moderate Fare Chareg** has **moderate chance of survival.**<br>
#     3. Passengers with **High Fare Charge** has **high chance of survival.**<br>

# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Feature  Selection</b></div>

# ### 1. Performing Chi-Square Test to find the Association between Categorical Features & Survived feature.

# In[40]:


from scipy.stats import chi2_contingency


# In[41]:


cat_cols = ["Pclass","Sex","SibSp","Parch","Embarked","Family_Size","Solo_Traveller","Age_Category","SibSp_Category","Parch_Category","Fare_Range"]

for column in cat_cols:
    contingency_table = pd.crosstab(train_df[column], train_df['Survived'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"P_value for columns {column} is --->","{:.15f}".format(p))


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * All the features are showcasing **significant association with Survived Feature** becuase all the features are having **P-value less than 0.05**.<br>
# * However, we can created **new features (Parch_Category & SibSp_Category)** so we can remove the **original column (Parch & SibSp)** while feature selection.<br></div>

# ---

# ### 2. Performing Anova-Test to find the Association between Numerical & Continous Features.

# In[42]:


from scipy.stats import f_oneway


# In[43]:


for column in ["Age","Fare"]:
    survived = train_df[train_df["Survived"]=="Survived"][column].dropna()
    not_survived = train_df[train_df["Survived"]=="Not-Survived"][column].dropna()
    
    anova_test = f_oneway(survived, not_survived)
    print(f"P-value of {column} is---->", "{:.20f}".format(anova_test.pvalue))


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * Both the features are showcasing **significant association with Survived Feature** becuase both the features are having **P-value less than 0.05**.<br>
# * So we will **include** both these features while **feature selection.**

# ### 3. Creating New DataFrame with all Relevant Features w.r.t Survived Features.

# In[44]:


new_train_df = train_df.drop(columns=["Parch","SibSp"]).copy()
new_test_df = test_df.drop(columns=["Parch","SibSp"]).copy()


# In[45]:


new_train_df.head()


# ----

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Data Pre-Processing</b></div>

# ### 1. Checking Missing Values & Their Percentage

# In[46]:


null_df = new_train_df.isnull().sum().to_frame().rename(columns={0:"No. of Null Values in Train Data"})
null_df["No. of Null Values in Test Data"] = new_test_df.isnull().sum()
null_df


# ### 2. Computing Missing Values in Numerical Features.

# In[47]:


median_age_survived = new_train_df[new_train_df['Survived'] == 'Survived']['Age'].median()
median_age_not_survived = new_train_df[new_train_df['Survived'] == 'Not-Survived']['Age'].median()


new_train_df.loc[(new_train_df['Survived'] == 'Survived') & (new_train_df['Age'].isnull()), 'Age'] = median_age_survived
new_train_df.loc[(new_train_df['Survived'] == 'Not-Survived') & (new_train_df['Age'].isnull()), 'Age'] = median_age_not_survived

new_test_df["Age"] = new_test_df["Age"].fillna(value=new_test_df["Age"].median())
new_test_df["Fare"] = new_test_df["Fare"].fillna(value=new_test_df["Fare"].median())


# ### 3. Computing Missing Values in Categorical Features.

# In[48]:


imputer1 = SimpleImputer(strategy="most_frequent")    


new_train_df[["Age_Category","Embarked"]]= imputer1.fit_transform(new_train_df[["Age_Category","Embarked"]])
new_test_df[["Age_Category","Fare_Range"]]= imputer1.fit_transform(new_test_df[["Age_Category","Fare_Range"]])


# ### 4. Confiriming Filling of Missing Values.

# In[49]:


print("Missing Values left in Training Dataset is:",new_train_df.isnull().sum().sum())
print("Missing Values left in Testing Dataset is:",new_test_df.isnull().sum().sum())


# ### 5 Checking Duplicacy in Data.

# In[50]:


print("Duplicate values in training data is: ",train_df.duplicated().sum())
print("Duplicate values in testing data is: ",test_df.duplicated().sum())


# ### 6. Dropping PassengerId Feature from both datasets.

# In[51]:


test_id = new_test_df["PassengerId"]

new_train_df.drop(columns=["PassengerId"],inplace=True)
new_test_df.drop(columns=["PassengerId"],inplace=True)


# ### 7. Plotting Distribution of Continous Numerical Features.

# In[52]:


plt.figure(figsize=(16,6))

for i,column in enumerate(["Age","Fare"]):
    plt.subplot(1,2,i+1)
    sns.distplot(x=new_train_df[column],color="Orange")
    plt.title(f"Skewness of {column} column is: {round(new_train_df[column].skew(),2)}",fontweight="black",size=18,pad=15)


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
# 
# * **Age Feature** is **moderate right skewed** where as **Fare Feature** is **highly right skewed.**<br>
# * To achieve a **Normal Distribution** in **Fare Feature** we can use **Log-Transformation** technique.<br></div>

# ### 8. Applying Log Transformation Fare Feature.

# In[53]:


#Log Transformation on Fare Feature...

new_train_df["Fare"] = np.log(1+new_train_df["Fare"])
new_test_df["Fare"]=np.log(1+new_test_df["Fare"])


# ### 9. Applying One-Hot Encoding on Categorical Features.

# In[54]:


cat_cols = new_train_df.select_dtypes(include=["object","bool"]).columns.tolist()
cat_cols.remove("Survived")

#NOTE:- One-hot encoding can make your model more interpretable because it explicitly shows the contribution of each category to the prediction. 

new_train_df = pd.get_dummies(columns=cat_cols, data=new_train_df)
new_test_df = pd.get_dummies(columns=cat_cols, data=new_test_df)


# ### 10. Enocding Survived Column in Training Data.

# In[55]:


new_train_df["Survived"] = new_train_df["Survived"].replace({"Survived": 0, "Not-Survived": 1})


# ### 11. Segregating Features & Labels For Model Training.

# In[56]:


X = new_train_df.drop(columns=["Survived"])
y = new_train_df["Survived"]


# ### 12. Splitting Data for Model Training & Model Testing.

# In[57]:


x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)


# In[58]:


print("Shape of x_train:",x_train.shape,"Shape of y_train:",y_train.shape)
print("Shape of x_test:",x_test.shape,"Shape of y_test:",y_test.shape)


# -------------------------------------------------------------------------------------

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Model Building For Un-Sclaed Data</b></div>

# In[59]:


sns.reset_defaults()
sns.set(style="darkgrid",font_scale=1.5)

training_score = []
testing_score = []
precision_value = []
recall_value = []
f1_value = []


# In[60]:


#Creating a function to train different models on the same dataset...

def model_prediction(model):
    
    model.fit(x_train, y_train)
    x_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    train_score = accuracy_score(y_train, x_train_pred) * 100
    test_score = accuracy_score(y_test, y_test_pred) * 100
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1= f1_score(y_test, y_test_pred)
    
    training_score.append(train_score)
    testing_score.append(test_score)
    precision_value.append(precision)
    recall_value.append(recall)
    f1_value.append(f1)
    
    print(f"Accuracy Score of {model} model on Training Data is: {train_score:.2f}")
    print(f"Accuracy Score of {model} model on Testing Data is: {test_score:.2f}")
    print("\n------------------------------------------------------------------------")
    print(f"Precision Score of {model} model is: {precision:.2f}")
    print(f"Recall Score of {model} model is: {recall:.2f}")
    print(f"F1 Score of {model} model is: {f1:.2f}")
    print("\n------------------------------------------------------------------------")
    print(f"Confusion Matrix of {model} model is:")
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(data=cm, annot=True, cmap="Blues")
    plt.show()


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>1. DecisionTree Classifier </b> Model </h2>

# In[61]:


model_prediction(DecisionTreeClassifier())


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>2. Random Forest Classifier </b> Model </h2>

# In[62]:


model_prediction(RandomForestClassifier())


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>3. AdaBoost Classifier </b> Model </h2>

# In[63]:


model_prediction(AdaBoostClassifier())


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>4. Gradient Boosting Classifier </b> Model </h2>

# In[64]:


model_prediction(GradientBoostingClassifier())


# -----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>5.LGBM Classifier</b> Model </h2>

# In[65]:


model_prediction(LGBMClassifier())


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>6. XGBoost Classifier</b> Model </h2>

# In[66]:


model_prediction(XGBClassifier(verbosity=0))


# ----

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b>7. CatBoost Classifier</b> Model </h2>

# In[67]:


model_prediction(CatBoostClassifier(verbose=False))


# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Model's Performance Comparison</b></div>

# In[68]:


models = ["Decision Tree","Random Forest","Ada Boost","Gradient Boost","LGBM","XGBoost","CatBoost"]


# In[69]:


df = pd.DataFrame({"Algorithms":models,
                   "Training Score":training_score,
                   "Testing Score":testing_score,
                   "Precision Score":precision_value,
                   "Recall Score":recall_value,
                   "f1 Score":f1_value})


# In[71]:


df


# ### Plotting above results using column-bar chart.

# In[72]:


df.plot(x="Algorithms",y=["Training Score","Testing Score"], figsize=(16,6),kind="bar",
        title="Performance Visualization of Different Models",colormap="Set1")
plt.show()


# <div style="border-radius:10px; border:#808080 solid; padding: 15px; background-color: ##F0E68C ; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color=brown>💬 Inference:</font></h3>
#    
# * Highest performance was give by GradientBoosting approx 86%.<br>
# * Second Highest performance was given by LGBM approx 84%<br>
# * But RandomForest, XGBoost & CatBoost Model performance was also good.<br>
# * So we will do Hyper-Parameter Tunning on these five Models.<br></div>

# ----------------------------------------------------------------------------------------

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Hyper-Parameter Tunning of GradientBoosting Model</b></div>

# In[74]:


gbc_model = GradientBoostingClassifier()


# In[75]:


parameters1 = {'n_estimators': [100,300,500,550],
               'min_samples_split':[7,8,9],
               'max_depth': [10,11,12], 
               'min_samples_leaf':[4,5,6]}


# In[76]:


grid_search1 = GridSearchCV(gbc_model, parameters1, cv=5, n_jobs=-1)


# In[77]:


grid_search1.fit(x_train,y_train)


# In[78]:


grid_search1.best_score_


# In[79]:


best_parameters1 = grid_search1.best_params_
best_parameters1


# ### Creating GradientBoost Model Using Best Parameters

# In[80]:


gradientboost_clf = GradientBoostingClassifier(**best_parameters1)


# In[81]:


gradientboost_clf.fit(x_train,y_train)


# In[82]:


x_test_pred1 = gradientboost_clf.predict(x_test)


# In[83]:


accuracy_score(y_test,x_test_pred1)


# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Hyper Parameter Tunning of LGBM Model</b></div>

# In[85]:


lgbm_model = LGBMClassifier()


# In[86]:


parameters2 = {"n_estimators":[100,300,500,600,650],
              "learning_rate":[0.01,0.02,0.03],
              "random_state":[0,42,48,50],
               "num_leaves":[16,17,18]}


# In[87]:


grid_search2 = GridSearchCV(lgbm_model, parameters2, cv=5, n_jobs=-1)


# In[88]:


grid_search2.fit(x_train,y_train)


# In[ ]:


grid_search2.best_score_


# In[ ]:


best_parameters2 = grid_search2.best_params_
best_parameters2


# ### Creating LGBM Model Using Best Parameters.

# In[ ]:


lgbm_clf  = LGBMClassifier(**best_parameters2)


# In[ ]:


lgbm_clf.fit(x_train,y_train)


# In[ ]:


x_test_pred2 = lgbm_clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,x_test_pred2)


# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Hyper Parameter Tunning of CatBoost Model</b></div>

# In[ ]:


cat_model = CatBoostClassifier(verbose=False)


# In[ ]:


parameters3 = {"learning_rate":[0.1,0.3,0.5,0.6,0.7],
              "random_state":[0,42,48,50],
               "depth":[8,9,10],
               "iterations":[35,40,50]}


# In[ ]:


grid_search3 = GridSearchCV(cat_model, parameters3, cv=5, n_jobs=-1)


# In[ ]:


grid_search3.fit(x_train,y_train)


# In[ ]:


grid_search3.best_score_


# In[ ]:


best_parameters3 = grid_search3.best_params_
best_parameters3


# ### Creating Cat Boost Model Using Best Parameters

# In[ ]:


catboost_clf = CatBoostClassifier(**best_parameters3,verbose=False)


# In[ ]:


catboost_clf.fit(x_train,y_train)


# In[ ]:


x_test_pred3 = catboost_clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,x_test_pred3)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 
# 💡 We can clearly observe that our CatBoost Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br></div>

# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Hyper Parameter Tunning of XGBoost Model</b></div>

# In[ ]:


model4 = XGBClassifier()


# In[ ]:


parameters4 = {"n_estimators":[50,100,150],
             "random_state":[0,42,50],
             "learning_rate":[0.1,0.3,0.5,1.0]}


# In[ ]:


grid_search4 = GridSearchCV(model4, parameters4, cv=5, n_jobs=-1)


# In[ ]:


grid_search4.fit(x_train,y_train)


# In[ ]:


grid_search4.best_score_


# In[ ]:


best_parameters4 = grid_search4.best_params_
best_parameters4


# ### Creating XGBoost Model Using Best Parameters

# In[ ]:


xgboost_clf = XGBClassifier(**best_parameters4)


# In[ ]:


xgboost_clf.fit(x_train,y_train)


# In[ ]:


x_test_pred4 = xgboost_clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,x_test_pred4)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
#     
# 💡 We can clearly observe that our XGBoost Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br></div>

# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Hyper Parameter Tunning of Random-Forest Model</b></div>

# In[ ]:


model5 = RandomForestClassifier()


# In[ ]:


parameters5 = {'n_estimators': [100,300,500,550],
               'min_samples_split':[7,8,9],
               'max_depth': [10,11,12], 
               'min_samples_leaf':[4,5,6]}
    


# In[ ]:


grid_search5 = GridSearchCV(model5, parameters5, cv=5, n_jobs=-1)


# In[ ]:


grid_search5.fit(x_train,y_train)


# In[ ]:


grid_search5.best_score_


# In[ ]:


best_parameters5 = grid_search5.best_params_
best_parameters5


# ### Creating Random Forest Model Using Best Parameters

# In[ ]:


randomforest_clf = RandomForestClassifier(**best_parameters5)


# In[ ]:


randomforest_clf.fit(x_train,y_train)


# In[ ]:


x_test_pred5 = randomforest_clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,x_test_pred5)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observations</b><br>
# 
# 💡 We can clearly observe that Random Forest Model is having best fitting.<br>
# 💡 Model doesn't have any overfitting or underfitting<br></div>

# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Stacking Model Classifier</b></div>

# In[ ]:


stacking_model = StackingClassifier(estimators=[('GBoost', gradientboost_clf),
                                                ('LGBM', lgbm_clf),
                                                ('CAT Boost', catboost_clf),
                                                ("XGBoost", xgboost_clf),
                                                ("RF",randomforest_clf)])


# In[ ]:


stacking_model.fit(x_train, y_train)


# In[ ]:


x_train_pred6 = stacking_model.predict(x_train)


# In[ ]:


x_test_pred6 = stacking_model.predict(x_test)


# In[ ]:


print("Stacking Model accuracy on Training Data is:",accuracy_score(y_train,x_train_pred6)*100)


# In[ ]:


print("Stacking Model accuracy on Testing Data is:",accuracy_score(y_test,x_test_pred6)*100)


# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Observation</b><br>
# 
# 💡 We can observe that our Stacking Model is having kind of Best Fitting<br>
# 💡 Stacking Model is not having any kind of over_fitting or under_fitting<br>
# 💡 So, we can use this Stacking Model to predict our test_data and then submit it on kaggle.<br></div>

# ---

# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Predicting Test Data using Stacking Model</b></div>

# **Extracting & Removing PassengerId from Test Data**

# In[ ]:


test_df.rename(columns={"Pclass_1":"Pclass_Lower_Class","Pclass_2":"Pclass_Middle_Class","Pclass_3":"Pclass_Upper_Class"},inplace=True)


# In[ ]:


df = test_df[["PassengerId"]]
test_df.drop(columns=["PassengerId"],inplace=True)


# In[ ]:


pred = stacking_model.predict(test_df)


# In[ ]:


pred


# <a id="2"></a>
# # <div style="padding:20px;color:white;margin:0;font-size:35px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;background-color:#254E58;overflow:hidden"><b>Submission DataFrame Format</b></div>

# In[ ]:


df.head()


# In[ ]:


df["Survived"] = pred


# In[ ]:


df.head()


# **Submission**

# In[ ]:


df.to_csv("Titanic_Survival_Prediction.csv",index=False)


# ---

# <a id="1.1"></a>
# <h2 style="font-family: Verdana; font-size: 28px; font-style: normal; font-weight: normal; text-decoration: none; text-transform: none; letter-spacing: 2px; color: black; background-color: #ffffff;"><b> Conclusion </b></h2>

# <div style="font-family:Georgia;background-color:aliceblue; padding:30px; font-size:17px">
# 
# <b>Titanic Survival Project</b><br>
#     
# 💡 The main objective of this project was to predict whether the passengers will be transported to alternate dimensions or not using the independent features given.<br>
# 
# <b>Key-Points</b><br>
#     
# 💡 We were havinng very few usefull independent features in the dataset.<br> 
# 💡 So I have done various feature engineering to create some new relevant features for better predictions.<br>
# 💡 The main objective of feature engineering was to avoid data loss.<br>
# 💡 I have used different classifiers machine learning techniques for predictions.<br>
# 💡 Then I  have compared all the preddictions given by different classifier models.<br>
# 💡 Then I have selected the best performing classifier modles.<br>
# 💡 The best performing Models were XGboost,LGBM, CatBoost,& GradientBoosting<br>
# 💡 But this models were having overfiiting.<br>
# 💡 So to reduce overfitting from the model I have done Hyper-Parameter Tunning<br>
# 💡 Then I haved used Stacking Ensemble Technique to boost my predictions.<br>
# 💡 In stacking Model I have used all the Models created after Hyper-Parameter Tunning.<br>
# 💡 In the end I have used Stacking Model to predict our test data.<br></div>
