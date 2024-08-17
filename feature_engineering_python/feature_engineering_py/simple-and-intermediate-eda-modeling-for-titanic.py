#!/usr/bin/env python
# coding: utf-8

# ## **Objective** : Predict if a passenger can survive on the titanic or not.

# ## Import Libraries :

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Read Data :

# In[2]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[3]:


train.head()


# # 1. Exploratory Data Analysis
# - Identify the target :
# Survived
# - Number of lines and columns :
# (891, 12)
# - Type of variables :
# Quantitative variables : 7
# Qualitative : 5
# - Identification of missing values :
#     - a lot of Nan variables in Cabin column = 70 % -->  missing values can be a significant information in that case
#     - many Nan variables in Age column = 20 % 
# 

# In[4]:


n = len(train)
train.shape


# In[5]:


train.dtypes.value_counts()


# In[6]:


train.isna().sum() / train.shape[0] * 100


# In[7]:


plt.figure(figsize=(10, 5))
sns.heatmap(train.isna())


# In[8]:


train.info()


# In[9]:


train.describe(include='all')


#  ## Target visualization :

# In[10]:


(train.Survived.value_counts() / train.shape[0] * 100).plot.bar(title='Target distribution')


#  ## Quantitative variables :

# In[11]:


quantitative_col = [ 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare']

for col in quantitative_col :
    plt.figure(figsize=(10, 5))
    sns.distplot(train[col])


# - The target is not perfectly distributed on Survived and unSurvived peaple, so using F-score as a metric is a good option
# - Pclass, SibSp and Parch can be encoded as they contain just some diffirent values 
# - Age and Fare may be normalized

#  ## Qualitative variables :

# In[12]:


train.select_dtypes('object').columns


# In[13]:


qualitative_col = ['Sex', 'Ticket', 'Cabin', 'Embarked']

for col in qualitative_col :
    print(f'{col :-<50} {train[col].unique()}')


# - Embarked and Sex columns can easily be encoded
# - for Ticket, cabin and name columns we can extract other meaningful ones = Feature engineering 

#  ## Relationship between target and variables :

# In[14]:


train.drop('PassengerId', axis = 1, inplace = True)


# In[15]:


# color palette from seaborn
cm = sns.light_palette("green", as_cmap=True)
 
# Visualizing the DataFrame with set precision
train.corr().style.background_gradient(cmap=cm).set_precision(2)


# In[16]:


survived_people = train[train.Survived == 1]
unsurvived_people = train[train.Survived == 0]


# In[17]:


for col in quantitative_col :
    plt.figure()
    sns.distplot(survived_people[col], label='survived')
    sns.distplot(unsurvived_people[col], label='unsurvived')
    plt.legend()


# In[18]:


for col in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'] :
    pd.crosstab(train['Survived'], train[col]).plot.bar()


# let's see what hapens with SibSp and Parch :

# In[19]:


data = [train, test]

for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'

axes = sns.factorplot('relatives','Survived', 
                      data=train, aspect = 2.5, )


# - Pclass and Fare are highly correlated with our target and between each other, also SibSp and Parch have significant correlation with it and also between each other
# - The younger you are the more likely to survive
# - More people in class 3 died
# - females have more chance to survive
# - Alone People have more chance to dy and if you travel with 1 to 3 people you have more chance to survive
# - People who travel from C have more chance to not survive

# # 2. Pre-Processing :

# In[20]:


# concatenate train and test set for the pre-processing
df = train.append(test).reset_index(drop=True)


# ## Feature Engineering :

# In[21]:


for i in range(len(df)):
    if not(pd.isnull(df['Cabin'].iloc[i])):
        df['Cabin'].iloc[i]=df['Cabin'].iloc[i][0] 
    else :
        df['Cabin'].iloc[i]='No'


# In[22]:


# add familly size column
df['Fsize'] = df['Parch'] + df['SibSp'] + 1


# In[23]:


df['travelled_alone'] = 'No'
df.loc[df.Fsize == 1, 'travelled_alone'] = 'Yes'


# In[24]:


df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Capt', 'Col', 'Rev', 'Don', 'Countess', 'Jonkheer', 'Dona', 'Sir', 'Dr', 'Major', 'Dr'], 'Rare')
df['Title'] = df['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Miss')
df['Title'] = df['Title'].replace(['Lady'], 'Mrs')


# ## Imputation :

# In[25]:


df.Embarked.fillna(train.Embarked.mode()[0], inplace = True)


# In[26]:


mean = train["Age"].mean()
std = train["Age"].std()

is_null = df["Age"].isnull().sum()
# compute random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
# fill NaN values in Age column with random values generated
age_slice = df["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

df["Age"] = age_slice
df["Age"] = df["Age"].astype(int)


# In[27]:


df['Fare'].fillna(train['Fare'].mean(), inplace = True)


# In[28]:


plt.figure(figsize=(12,6))
plt.scatter(train.Fare, train.Survived)


# In[29]:


train.Fare.max()


# In[30]:


test.Fare.max()


# In[31]:


train[train.Fare>300].Fare.count() # there is just three elements between 300 and 5** so we will remove it


# In[32]:


test[test.Fare>300].Fare.count()


#     There are probably other outliers in the training data. However, removing all them may affect badly our models if ever there were also outliers in the test data. 
#   
#   Outliers removal is note always safe. 
#   
#   Ps : I tried with removing these three samples (Fare column) and the result become worse!  So I will keep it.

# ## Encoding :

# In[33]:


df.columns


# In[34]:


features = ["Sex", "Pclass","travelled_alone", "Cabin", "Embarked", "Title"]


# In[35]:


df=pd.get_dummies(df,columns=features,drop_first=True)


# In[36]:


df.head(2)


# In[37]:


df.drop(['Name', 'Ticket'], axis = 1, inplace = True)


# In[38]:


train = df[:n ] # the three outliers
test = df[n:]


# In[39]:


train.isna().sum() / train.shape[0] * 100


# In[40]:


test.isna().sum() / test.shape[0] * 100


# ## Train test split :

# In[41]:


X = train.drop(['Survived','PassengerId'], axis = 1)
y = train.Survived
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # 3. Modeling and Evaluation :

# In[42]:


def evaluation(model):
    
    model.fit(x_train, y_train)
    ypred = model.predict(x_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_score, val_score = learning_curve(model, x_train, y_train,
                                              cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    
    


# In[43]:


model = RandomForestClassifier(random_state=0)


# In[44]:


evaluation(model)


# In[45]:


preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))


# In[46]:


RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())


# In[47]:


dict_of_models = {'RandomForest': RandomForest,
                  'SVM': SVM,
                  'KNN': KNN
                 }


# In[48]:


for name, model in dict_of_models.items():
    print(name)
    evaluation(model)


# In[49]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[50]:


SVM


# In[51]:


hyper_params = {'svc__gamma':[1e-3, 1e-4, 0.0005],
                'svc__C':[1, 10, 100, 1000, 3000], 
               'pipeline__polynomialfeatures__degree':[2, 3],
               'pipeline__selectkbest__k': range(45, 60)}


# In[52]:


grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4,
                          n_iter=40)

grid.fit(x_train, y_train)
print(grid.best_params_)


# In[53]:


evaluation(grid.best_estimator_)


# In[54]:


submit = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[55]:


submit['Survived']=grid.predict(test.drop(['Survived', 'PassengerId'], axis = 1)).astype('int')
submit.to_csv('submission.csv',index=False)


# In[56]:


submit.head(2)


# ### If you find this notebook useful, please don't forget to upvote it!
