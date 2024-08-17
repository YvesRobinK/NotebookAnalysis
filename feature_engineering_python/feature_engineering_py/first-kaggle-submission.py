#!/usr/bin/env python
# coding: utf-8

# <font color="red" size=5><center>TITANIC: MACHINE LEARNING FROM DISASTER</center></font>

#  In this notebook we will see how different people either survived or lost their lives who were present on the great RMS Titanic. 
#  
#  
# <center><img src="https://www.printwand.com/blog/media/2012/01/titanic-sinking.jpg" width="500px"></center>

# <font color="red" size=3>Please upvote this kernel if you like it. It motivates me to produce more quality content :)</font>

# 
# <a class="anchor" id="toc"></a>
# <div style="background: #f9f9f9 none repeat scroll 0 0;border: 1px solid #aaa;display: table;font-size: 95%;margin-bottom: 1em;padding: 20px;width: 600px;">
# <h1>Contents</h1>
# <ul style="font-weight: 700;text-align: left;list-style: outside none none !important;">
# <li style="list-style: outside none none !important;font-size:17px"><a href="#1">1 Understanding the data at hand</a></li>
# 
# <li style="list-style: outside none none !important;font-size:17px"><a href="#2">2 Exploratory Data Analysis</a></li>
#     
# <li style="list-style: outside none none !important;font-size:17px"><a href="#3">3 Feature Engineering</a></li>
#     
# <li style="list-style: outside none none !important;font-size:17px"><a href="#4">4 Modelling</a></li>
#       <ul style="font-weight: 700;text-align: left;list-style: outside none none !important;">    
# 
# </ul>
# </div>

# In[1]:


import pandas as pd 
import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import scipy.stats as ss
from statsmodels.formula.api import ols
from scipy.stats import zscore

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('timeit', '')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('darkgrid')

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

plt.rc('font', **font)


# <a id="1"></a> 
# # 1. Understanding the data at hand

# In[2]:


dftrain=pd.read_csv("../input/train.csv")
dftest=pd.read_csv("../input/test.csv")
test=dftest.copy()


# In[3]:


dftrain.info()


# In[4]:


dftrain.head().T


# In[5]:


dftrain.info()


# In[6]:


print('Number of null values :',dftrain.isnull().sum().sum())


# In[7]:


df = dftrain.copy()


# In[8]:


print("Survival rates of MALES and FEMALES")
print("**********************************\n\n")
male1=df.loc[(df.Survived==1) &(df.Sex=='male'),:].count()
print("MALE:\n\n",male1) 
female1=df.loc[(df.Survived==1) & (df.Sex=='female'),:].count()
print("\nFEMALE:\n\n",female1)


# In[9]:


print('*'*6,' NULL VALUES ',"*"*6,'\n')
print(df.isnull().sum())


# In[10]:


print("Mean age for each Pclass :\n\n",df.groupby("Pclass").Age.mean())


# <a id="2"></a> 
# # 2. Exploratory Data Analysis

# In[11]:


# Missing data
import missingno as msno
msno.bar(dftrain.sample(890),(28,10),color='red')

plt.title('MISSING VALUES',fontsize=25)


# In[12]:


corr = dftrain.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(corr,mask=mask,square=True,annot=True,fmt='0.2f',linewidths=.8,cmap="Accent",robust=True)


# In[13]:


# Stats and Visualisation of Survival Rate
sns.factorplot(x="Sex",col="Survived", data=df , kind="count",size=6, aspect=.7,palette=['crimson','lightblue'])

malecount=pd.value_counts((df.Sex == 'male') & (df.Survived==1))
femalecount=pd.value_counts((df.Sex=='female') & (df.Survived==1))
totalmale,totalfemale=pd.value_counts(df.Sex)


# In[14]:


print("MALE survived \n{} \n\n\nFEMALE survived \n{}".format(malecount/totalmale,femalecount/totalfemale))


# From above statistics it is clear that Women were given more preference than Men while evacuation  

# In[15]:


#Clear representation of Ages of passengers and to which Class they belonged
plt.figure(figsize=(10,10))
sns.swarmplot(x="Sex",y="Age",hue='Pclass',data=df,size=8 ,palette=['orange','brown','purple'])


# The above graph makes it clear that most of the people were aged between 20-50

# In[16]:


plt.figure(figsize=(10,10))
sns.swarmplot(x="Sex",y="Age",hue='Survived',data=df,size=8,palette='viridis')


# It is clear from vizualisation that most of the survivors were children and women 

# In[17]:


sns.factorplot(x="Sex", hue = "Pclass" , col="Survived", data=df , kind="count",size=7, aspect=.7,palette=['blue','green','yellow'])


# Most of the people who died were from Passenger Class 3 irrespective of Gender

# In[18]:


pd.crosstab([df.Sex,df.Survived],df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# The above stats show us survival of each class and its clear the ones in better class had a better chance of survival
# ## Power of money

# In[19]:


sns.factorplot(x="Survived",col="Embarked",data=df ,hue="Pclass", kind="count",size=8, aspect=.7,palette=['crimson','darkblue','purple'])


# Most of the embarkments were from class : S
# 
# Least embarkments were from class : Q

# In[20]:


sns.factorplot(x="Sex", y="Survived",col="Embarked",data=df ,hue="Pclass",kind="bar",size=7, aspect=.7)


# In[21]:


# Correlation Heatmap 
context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df['Sex_bool']=df.Sex.map(context1)
df["Embarked_bool"] = df.Embarked.map(context2)

corr = df[['PassengerId', 'Survived', 'Pclass', 'Sex_bool', 'Age', 'SibSp',
       'Parch', 'Fare' , 'Embarked_bool']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(9, 7))
    ax = sns.heatmap(corr,mask=mask,square=True,annot=True,fmt='0.2f',linewidths=.8,cmap="jet",robust=True)


# ## Inferences from the above heatmap
# *  PassengerId is a redundant column as its very much less related to all other attributes , we can remove it .
# * Also , Survived is related indirectly with Pclass and also we earlier proved that as Pclass value increases Survival decreases
# * Pclass and Age are also inversely related and can also be proven by the following cell that as Pclass decreases , the mean of the Age      increases , means the much of the older travellers are travelling in high class .
# * Pclass and fare are also highly inversely related as the fare of Pclass 1 would obviously be higher than corresponding Pclass 2 and 3 .
# * Also , people with lower ages or children are travelling with their sibling and parents more than higher aged people (following an                inverse relation) , which is quite a bit obvious .
# * Parch and SibSp are also highly directly related
# * Sex_bool and Survived people are highly inversely related , i.e. females are more likely to survive than men

# In[22]:


for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i
df[['Age','Age_bin']].head(20)


# In[23]:


plt.figure(figsize=(20,20))
sns.set(font_scale=1)
sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex',kind="bar", data=df)


# In[24]:


df.describe().T


# In[25]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftrain['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftrain["Age"].mean()
std_age = dftrain["Age"].std()
count_nan_age = dftrain["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftrain["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)


# In[26]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftest['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftest["Age"].mean()
std_age = dftest["Age"].std()
count_nan_age = dftest["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftest["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)


# In[27]:


for x in [dftrain, dftest , df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i
fig, axes = plt.subplots(2,1)
fig.set_size_inches(15, 12)
sns.kdeplot(df.Age_bin , shade=True, color="green" , ax= axes[0])
sns.kdeplot(df.Fare , shade=True, color="orange" , ax= axes[1])


# ## Inferences from the above graph
# * Most passengers were aged from 20-40
# * Most passengers paid nearly 40 units $/Rs
# 
# ### As the graph is left-skewed we can use log scale or sqrt scale to change this

# In[28]:


family_df = dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df["Fsize"] = family_df.SibSp + family_df.Parch + 1

family_df.head()


# In[29]:


plt.figure(figsize=(15,5))

plt.title("FAMILY_SIZE vs SURVIVAL")
sns.countplot(x='Fsize', hue="Survived", data=family_df)


# Bigger the family lesser the chance of survival
# 

# <a id="3"></a> 
# # 3.Feature Engineering

# In[30]:


dftrain.info()


# In[31]:


dftest.info()


# In[32]:


dftrain.loc[[61,829],"Embarked"] = 'C'


# In[33]:


dftrain["Age"] = age_slice
dftrain=dftrain.drop('Age_bin',axis=1)
dftrain.info()


# In[34]:


dftrain['Fsize']=family_df['Fsize']


# In[35]:


family_df_t= dftest.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_t["Fsize"] = family_df_t.SibSp + family_df_t.Parch + 1
dftest['Fsize']=family_df_t['Fsize']
family_df_t.head()
dftest['Fsize']=family_df_t['Fsize']


# In[36]:


dftest.loc[[152],"Fare"] = 10


# In[37]:


family_df_tr= dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_tr["Fsize"] = family_df_tr.SibSp + family_df_tr.Parch + 1

family_df_tr.head()
dftrain['Fsize']=family_df_tr['Fsize']


# In[38]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )


#  Using Chi- Squared test at 5% we get the above results telling us which are important  features
# 

# In[39]:


# Make a copy of the titanic data frame
dftrain['Title'] = dftrain['Name']
# Grab title from passenger names
dftrain["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)

rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
dftrain['Title'].replace(rare_titles, "Rare title", inplace=True)
# Also reassign mlle, ms, and mme accordingly
dftrain['Title'].replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)


# Making some changes in the titles 

# In[40]:


cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize','Title','SibSp','Parch']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )  


# In[41]:


dftest=dftest.drop(['Ticket','PassengerId'],axis=1)


# In[42]:


dftest['Title'] = dftest['Name']
# Grab title from passenger names

dftest["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)


# In[43]:


dftrain=dftrain.drop('Name',axis=1)


# In[44]:


dftrain.head().T


# In[45]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
context3= {"Mr":0 , "Mrs":1 , "Miss":2,'Master':3}

dftrain['Sex_bool']=dftrain.Sex.map(context1)
dftrain["Embarked_bool"] = dftrain.Embarked.map(context2)
dftrain['Title']=dftrain.Title.map(context3)


# Above we are creating boolean values for the model to understand 

# In[46]:


dftrain=dftrain.drop(['PassengerId','Cabin','Ticket'],axis=1)
dftrain=dftrain.drop(['Embarked','Sex'],axis=1)
dftrain.head().T


# In[47]:


dftest.head().T


# In[48]:


dftest=dftest.drop(['Name','Sex','Embarked'],axis=1)


# In[49]:


for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i


# In[50]:


for x in [dftrain, dftest,df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ x['Fare'] <= i*10, 'Fare_bin'] = i


# In[51]:


dftrain=dftrain.drop('Age',axis=1)
dftest=dftest.drop('Age',axis=1)


# In[52]:


dftrain=dftrain.convert_objects(convert_numeric=True)


# In[53]:


def change_type(df):
    float_list=list(df.select_dtypes(include=["float"]).columns)
    print(float_list)
    for col in float_list:
        df[col]=df[col].fillna(0).astype(np.int64)
        
    return df    
change_type(dftrain)   
dftrain.dtypes


# In[54]:


dftrain.head().T


# In[55]:


x=dftrain.iloc[:,1:].values
y=dftrain.iloc[:,0].values
print(dftrain.columns)
print(dftest.columns)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)


# In[56]:


dftest=dftest.convert_objects(convert_numeric=True)
change_type(dftest)    
dftest.dtypes


# <a id="4"></a> 
# # 4. Modelling

# In[57]:


MLA = []
Z = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , KNeighborsClassifier() , GaussianNB() ,
    RandomForestClassifier() , GradientBoostingClassifier()]
X = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "KNeighborsClassifier" , "GaussianNB" ,
    "RandomForestClassifier" , "GradientBoostingClassifier"]

for i in range(0,len(Z)):
    model = Z[i]
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    MLA.append(accuracy_score(pred , y_test))
MLA    


# In[58]:


d = { "Accuracy" : MLA , "Algorithm" : X }
dfm = pd.DataFrame(d)
dfm


# In[59]:


sns.barplot(dfm['Accuracy'],dfm['Algorithm'])


# # 5.Tuning the parameters

# In[60]:


#Logistic Regression 
params={'C':[1,100,0.01,0.1,1000],'penalty':['l2','l1']}
logreg=LogisticRegression()
gscv=GridSearchCV(logreg,param_grid=params,cv=10)
get_ipython().run_line_magic('timeit', 'gscv.fit(x,y)')


# In[61]:


print("BEST PARAMS: ",gscv.best_params_)
logregscore=gscv.best_score_
print("BEST SCORE:",logregscore)


# In[62]:


print("SCORE:",gscv.score(X_test,y_test))


# In[63]:


#KNN
param={'n_neighbors':[3,4,5,6,8,9,10],'metric':['euclidean','manhattan','chebyshev','minkowski'] }       
knn = KNeighborsClassifier()
gsknn=GridSearchCV(knn,param_grid=param,cv=10)
gsknn.fit(x,y) 

print("BEST PARAMS: ",gsknn.best_params_)
print("BEST SCORE:",gsknn.best_score_)


# In[64]:


Survived=gsknn.predict(X_test)


# In[65]:


print("SCORE :",gsknn.score(X_test,y_test))


# In[66]:


#Random Forest
rfcv=RandomForestClassifier(n_estimators=500,max_depth=6)
rfcv.fit(X_train,y_train)
rfcv.predict(X_test)
print("SCORE:", rfcv.score(X_test,y_test))


# In[67]:


#Gradient Boosting
gbcv=GradientBoostingClassifier(learning_rate=0.001,n_estimators=2000,max_depth=5)
gbcv.fit(X_train,y_train)
gbcv.predict(X_test)
print("SCORE:",gbcv.score(X_test,y_test))


# In[68]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, gscv.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rfcv.predict_proba(X_test)[:,1])
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, gsknn.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, ada_thresholds = roc_curve(y_test, gbcv.predict_proba(X_test)[:,1])

plt.figure(figsize=(9,9))
log_roc_auc = roc_auc_score(y_test, gscv.predict(X_test))
print ("logreg model AUC = {} " .format(log_roc_auc))
rf_roc_auc = roc_auc_score(y_test, rfcv.predict(X_test))
print ("random forest model AUC ={}" .format(rf_roc_auc))
knn_roc_auc = roc_auc_score(y_test, gsknn.predict(X_test))
print ("KNN model AUC = {}" .format(knn_roc_auc))
gbc_roc_auc = roc_auc_score(y_test, gbcv.predict(X_test))
print ("GBC Boost model AUC = {}" .format(gbc_roc_auc))
# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression')

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest')

# Plot Decision Tree ROC
plt.plot(knn_fpr, knn_tpr, label=' KnnClassifier')

# Plot GradientBooseting Boost ROC
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingclassifier')

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate',linestyle="--")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")

plt.arrow(0.5, 0.1, -0.12, 0.2, head_width=0.02, head_length=0.05, fc='k', ec='k')
plt.show()


# The arrow is pointing towards `Base Rate`.  If any line is below it then model is performing in the worst manner possible.

# In[69]:


test_PassengerId = pd.read_csv('../input/gender_submission.csv')['PassengerId']
submission = pd.concat([pd.DataFrame(test_PassengerId), pd.DataFrame({'Survived':Survived})], axis=1)
submission.to_csv('submission.csv', index=False)

