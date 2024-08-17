#!/usr/bin/env python
# coding: utf-8

# ### Updating...

# Soon to be completed...

# ## 1. Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Loading dataset

# In[2]:


df = pd.read_csv('../input/titanic/train.csv')


# ## 3. Inspecting data

# In[3]:


df.head()


# In[4]:


df.info()


# ### So we have to deal with NaN values based on the info for ("Age", "Cabin", "Embarked")

# In[5]:


df.describe()


# * About 38% of the passengers survived.
# * Mean age was around 30 years.
# * Passengers paid something around 32.
# * Half of the passengers travelled with siblings or spouse.

# ## 4. Removing unuseful columns

# * It's obvious that PassengerID has nothing to do with our model so simply just dropped it!

# In[6]:


df["Cabin"].value_counts()


# * It looks like that cabin number is to spreaded to be used in the model but let's visualise it and then make the decision.
# * At most 4 rows share the same cabin so it doens't help much and it's better to drop it.

# In[7]:


df["Fare"].value_counts()


# In[8]:


df["Fare"].describe()


# In[9]:


sns.displot(df["Fare"])


# * __"Fare"__ Distplot is pretty much skewed, maybe it's better to use the **log form**.
# * And clearly we don't have to remove "Fare" column.

# In[10]:


df["Ticket"].value_counts()


# * Ticket column is like Cabin, so spreaded so we just remove it!

# First we check for duplicate names and then remove the Name column

# In[11]:


df["Name"].duplicated().sum()


# * No duplicates!

# But wait, maybe we can do a feature engineering and extract titles and form a new columns based on them!

# ### Feature Engineering Alert!

# In[12]:


# Apply function to make the new column

df["Title"] = df["Name"].apply(lambda p : p.split()[1])


# Now let's see how this `Title` looks like

# In[13]:


df["Title"].value_counts()


# Under 40 observations won't contribute much to the model. so let's just perform __one hot encoding__ to these titles and lets see what happens!

# First of all let's transform everything beside [Mr., Miss., Mrs., Master.] to `Pal.`  

# In[14]:


accetable_titles = ["Mr.", "Miss.", "Mrs.", "Master."]
def change_title(tit):
    if tit in accetable_titles:
        return tit
    else:
        return "Pal."


# In[15]:


df ["Title"] = df["Title"].apply(lambda p : change_title(p))


# Now we're done with `Name` and it's time to say bye!

# In[16]:


df.drop(["Cabin","Ticket","PassengerId","Name"],axis = 1, inplace=True)


# In[17]:


df.head(10)


# ## 5. Looking for missing data

# In[18]:


def missing_percent(df):
    nan_percent= 100*(df.isnull().sum()/len(df))
    nan_percent= nan_percent[nan_percent>0].sort_values()
    return nan_percent    


# In[19]:


missing_percent(df)


# __1. Let's deal with Age__

# In[20]:


df["Age"].isnull().sum()


# __Roughly 20% of the age data is missing, so we just put the mean age for each sex__.

# In[21]:


import math
female_mean, male_mean = df.groupby("Sex")["Age"].mean()
def fill_age(age,sex):
    if math.isnan(age):
        if sex == "male":
            return male_mean
        else:
            return female_mean
    else:
        return age


# In[22]:


df["Age"] = df.apply(lambda row : fill_age(row["Age"],row["Sex"]),axis = 1)
# df['Q'] = df.apply(lambda row: EOQ(row['D'], row['p'], ck, ch), axis=1)


# __2. Now it's embarked time!__

# In[23]:


df["Embarked"].isnull().sum()


# * Only 2 rows are missing so we just delete those rows.

# In[24]:


df = df.dropna()


# ## 6. EDA!

# First things first lets see how `Sex` could affect your life back then!

# In[25]:


sns.barplot(x = df["Sex"], y = df["Survived"])
plt.show()


# ##### Females tend to survive more by a large margin!

# In[26]:


sns.barplot(x = df["Embarked"], y = df["Survived"])
plt.show()


# ###### Less survival rate by S embark, So there must be some relations!

# In[27]:


df.head()


# In[28]:


sns.boxplot(x = df["Embarked"], y = df["Fare"])
plt.show()


#  Strong outliers over 400 price, Let's just remove them for now and take a better look 

# In[29]:


index = df[df["Fare"] > 450 ].index
df.drop(index, axis = 0, inplace=True)


# In[30]:


plt.figure(figsize=(8,5))
sns.boxplot(x = df["Embarked"], y = df["Fare"])
plt.show()


# ##### 1. C embark is more expensive, maybe we can conclude that it's more premium than the others and maybe have better health guards and whatever, but clearly, if you paid more you had better chances of surviving.
# ##### 2. But Q embark is cheaper than S and interestingly, it has a better survival rate, so maybe your placement in ship matters more because these embarks show each cluster of people who are entering the ship simultaneously.

# In[31]:


plt.figure(figsize=(9,7))
sns.boxplot(x = df["Embarked"], y = df["Fare"],hue = df["Pclass"])
plt.show()


# ##### So Q is mainly for the lower class and C can match our theory, Richer people got on board from C.

# ##### So now let's explore the mean age of each embark:

# In[32]:


sns.boxplot(y = df["Age"], x = df["Embarked"])
plt.show()


# ##### It's pretty much normal so nothing to be worry about!

# In[33]:


sns.countplot(x = df["Pclass"], hue = df["Survived"])
plt.show()


# ##### Wow, such discrimination! unfortunately, class 3 almost didn't make it alive :(

# Money really could've bought you, your life!

# In[34]:


sns.factorplot('Pclass','Survived',hue='Sex',data=df)
plt.show()


# Women on class 1 and 2 was almost invincble, Hmmm...

# In[35]:


plt.figure(figsize=(10,8))
plt.yticks(range(0,110,5))
sns.violinplot(x = df["Pclass"], y = df["Age"], hue = df["Survived"],split = True)


# Looks like rich kids stole the show! (Doin fancy stuff on board =))

# Early on, we concluded that rich people got on board from C embark Let's see now if Pclass agrees with us or Nah?

# In[36]:


sns.countplot(df["Pclass"], hue = df["Embarked"])


# At least we can agree on that Q embark wasn't for riches.

# In[37]:


sns.countplot(df["Parch"], hue=df["Survived"])
plt.show()


# In[38]:


plt.plot(df.groupby("Parch")["Survived"].mean())
plt.xlabel("Parch")
plt.ylabel("Chance of survival")


# Huge dropoff in Parch = 4 and then a sudden rise in 5. 
# Maybe some sort of correlation is there!

# What about our new belvoed `Title` Column? 

# In[39]:


sns.countplot(df["Title"], hue = df["Survived"])
plt.show()


# Poor `Mr.`'s they're the real __martyrs__ of the titanic. `Mrs.` and `Miss.` did pretty well surviving. No more feminism talks for now then =))  

# Corr matrix time!

# In[40]:


a = df.corr()
plt.figure(figsize=(8,8))
k = 10
cols = a.nlargest(k, 'Survived')['Survived'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# #### As we can see `Parch` and `SubSp` don't contribute much to the corr matrix so let's do a feature engineering and create a new columns based on `Parch` and `SubSp`.

# ### Feature Engineering Alert!

# In[41]:


df["Family"] = df["Parch"] + df["SibSp"] + 1


# Now let's see the corr matrix again.

# In[42]:


a = df.corr()
plt.figure(figsize=(8,8))
k = 10
cols = a.nlargest(k, 'Survived')['Survived'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[43]:


sns.scatterplot(x = df["Fare"], y = df["Survived"])
plt.axvline(500, color = 'r')


# * One outlier after 500 

# In[44]:


index = df[df["Fare"] > 500].index
df.loc[index,:]


# In[45]:


df.drop(index,inplace=True)


# In[46]:


sns.scatterplot(x = df["Fare"], y = df["Survived"])
plt.axvline(500, color = 'r')


# ## 7. Feature Selection

# We have to change our categorical features to numerical.

# In[47]:


def change_sex(sex):
    if sex == "male":
        return 1
    elif sex == "female":
        return 0


# __Male will be 1 and female will be 0__

# In[48]:


df["Sex"] = df.apply(lambda row : change_sex(row["Sex"]),axis = 1)


# But this approach isn't ideal because we're turning them into ordinal format and it's not right for our model.

# In[49]:


df["Embarked"].value_counts()


# * S -> 0
# * C -> 1
# * Q -> 2

# In[50]:


def change_em(em):
    if em == "S":
        return 0
    elif em == "C":
        return 1
    elif em == "Q":
        return 2


# In[51]:


df["Embarked"] = df.apply(lambda row : change_em(row["Embarked"]),axis = 1)


# ## Update:

# For example, in `Embarked` mapped `Q` to 2 and `C` to 1, so it means `Q`s values are double of `C`. It isn't the best way to deal with categorical data.

# Now let's use dummy variable for the better performance

# In[52]:


emb = df["Embarked"]
sex = df["Sex"]
title = df["Title"]
pcl = df["Pclass"]


# In[53]:


s = pd.get_dummies(sex, columns=["Sex"], prefix="Sex_is" )
e = pd.get_dummies(emb, columns=["Embarked"], prefix="Embarked_is" )
t = pd.get_dummies(title, columns=["Title"], prefix="Title_is" )
p = pd.get_dummies(pcl,columns=["Pclass"],prefix="Pclass_is")


# In[54]:


main = s.join(e)
main = main.join(t)
main = main.join(p)


# In[55]:


df = df.join(main)


# In[56]:


df = df.drop(["Embarked","Sex","Pclass","Title"], axis = 1)


# In[57]:


df


# In[58]:


X = df.drop("Survived",axis = 1)
y= df["Survived"]


# ## 8. Split data

# In[59]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)


# ## 9. Scaling the features

# In[60]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()


# In[61]:


scaler.fit(X_train)


# In[62]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Now Let's try Logistic Reggresion

# ## 10. Train the model

# In[63]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# ## 11. Predicting test data

# In[64]:


y_pred = model.predict(X_test)
y_pred


# ## 12. Evaluating the model

# In[65]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# In[66]:


accuracy_score(y_test, y_pred)


# In[67]:


confusion_matrix(y_test, y_pred)


# In[68]:


plot_confusion_matrix(model, X_test, y_test)


# In[69]:


print(classification_report(y_test, y_pred))


# ### Scored 0.84 with Logistic Regression method

# ## Now Let's do it with KNN method

# First of all we should find the best k value, and we use GridSearch and Elbow method for this purpose.

# ## 13. Elbow method for finding the best K values

# In[70]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

test_error_rate = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    
    y_pred_knn = knn.predict(X_test)
    
    e = 1 - accuracy_score(y_test,y_pred_knn)
    test_error_rate.append(e)
    


# In[71]:


test_error_rate


# In[72]:


plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), test_error_rate, label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel('K Value')


# #### So elbow method shows that best K Values = 5

# ## Now Let's see what the Grid search method predicts for the K value

# ## 14. Creating the pipeline

# In[73]:


# Scaler for pipeline

sc = StandardScaler()


# In[74]:


# Model for pipeline

knn_model = KNeighborsClassifier()


# In[75]:


# Operations of pipeline

operations = [("Scaler",sc),("KNN", knn_model)]


# In[76]:


from sklearn.pipeline import Pipeline

pipe = Pipeline(operations)


# ## 15. Finding the best K value with Grid Search

# In[77]:


from sklearn.model_selection import GridSearchCV

k_values= list(range(1,20))


# In[78]:


# Pipeline keys

pipe.get_params().keys()


# In[79]:


param_grid = {"KNN__n_neighbors" : k_values}


# In[80]:


cv_classifier = GridSearchCV(pipe,param_grid=param_grid,scoring="accuracy")
cv_classifier.fit(X_train,y_train)


# In[81]:


# Now let's see what's the best K values

cv_classifier.best_estimator_.get_params()


# #### Grid search suggests for K values of 15

# #### Now we check for both 15 and 5

# ## 16. Final KNN model

# In[82]:


# KNN for 5

knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_5.fit(X_train,y_train)
y_knn_pred_5 = knn_model_5.predict(X_test)


# In[83]:


# KNN for 15

knn_model_15 = KNeighborsClassifier(n_neighbors=15)
knn_model_15.fit(X_train,y_train)
y_knn_pred_15 = knn_model_15.predict(X_test)


# ## 17. Evaluating the KNN models

# In[84]:


# Model Report for k = 5

print(classification_report(y_test, y_knn_pred_5))


# In[85]:


# Model Report for k = 15

print(classification_report(y_test, y_knn_pred_15))


# #### So clearly K = 5 is our best value for KNN method

# In[ ]:




