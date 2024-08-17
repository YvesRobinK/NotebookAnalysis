#!/usr/bin/env python
# coding: utf-8

# ##please upvote the kernle if you have found it useful. It motivates me a lot

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


train=pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test=pd.read_csv('../input/forest-cover-type-prediction/test.csv')


# In[3]:


train.info()


# ### EDA Column wise

# In[4]:


train['Elevation'].min()


# In[5]:


sns.distplot(train.Elevation,rug=True)
plt.grid()


# In[6]:


train['Elevation'].describe()


# In[7]:


sns.boxplot(train['Elevation'])


# In[8]:


sns.violinplot(x=train['Cover_Type'],y=train['Elevation'])
plt.grid()


# In[9]:


train.Aspect.describe()


# In[10]:


sns.distplot(train.Aspect)
plt.grid()


# In[11]:


sns.violinplot(x=train['Cover_Type'],y=train['Aspect'])
plt.grid()


# In[12]:


train['Slope'].describe()


# In[13]:


sns.distplot(train['Slope'])
print(train.Slope.skew())


# In[14]:


# apply the sqrt transformation to reduce the skewness 
sns.distplot(np.sqrt(train['Slope']+1))
print(np.sqrt(train['Slope']+1).skew())


# In[15]:


train.Slope=np.sqrt(train.Slope+1)


# In[16]:


test.Slope=np.sqrt(test.Slope+1)


# In[17]:


sns.distplot(test['Slope'],color='red')
plt.title('test.slope')


# In[18]:


train.Horizontal_Distance_To_Hydrology.describe()


# In[19]:


train['dist_hydr']=np.sqrt(train['Vertical_Distance_To_Hydrology']**2 + train['Horizontal_Distance_To_Hydrology']**2)
test['dist_hydr']=np.sqrt(test['Vertical_Distance_To_Hydrology']**2 + test['Horizontal_Distance_To_Hydrology']**2)


# In[20]:


sns.distplot(train['dist_hydr'])


# In[21]:


sns.distplot(np.sqrt(1+train['dist_hydr']))


# In[22]:


train['dist_hydr']=np.sqrt(1+train['dist_hydr'])
test['dist_hydr']=np.sqrt(1+test['dist_hydr'])



# In[23]:


sns.distplot(train.Horizontal_Distance_To_Hydrology,color='orange')


# In[24]:


sns.distplot(np.sqrt(train.Horizontal_Distance_To_Hydrology),color='orange')


# In[25]:


# apply the sqrt transformation
train.Horizontal_Distance_To_Hydrology=np.sqrt(1+train.Horizontal_Distance_To_Hydrology)
test.Horizontal_Distance_To_Hydrology=np.sqrt(1+test.Horizontal_Distance_To_Hydrology)


# In[26]:


# vertical distance to the hydrology column
sns.violinplot(x=train.Cover_Type,y=train.Vertical_Distance_To_Hydrology)


# In[27]:


sns.distplot(train.Vertical_Distance_To_Hydrology)


# In[28]:


# It is clearly an indication that there are some outliers
# By looking at the violin plot they may produce some good results   
sns.boxplot(train.Vertical_Distance_To_Hydrology)
plt.title('train.Vertical_Distance_To_Hydrology')


# In[29]:


# It is better to not remove outliers by looking at the both training and test plot  
sns.boxplot(test.Vertical_Distance_To_Hydrology)


# In[30]:


print(train.Hillshade_9am.describe())


# In[31]:


sns.distplot(train.Hillshade_9am)


# In[32]:


sns.boxplot(train.Hillshade_9am)
plt.grid()


# In[33]:


# to find the impact of  of outliers consider the violinplot
sns.violinplot(x=train.Cover_Type,y=train.Hillshade_9am)


# In[34]:


# both train and test datasets have points below the (Q1-1.5IQR)
# so let us assume that outliers have some significant impact on prediction 
sns.boxplot(train.Hillshade_9am,color='red')
plt.title('test_Hillshade')


# In[35]:


sns.boxplot(train.Hillshade_Noon)


# In[36]:


sns.violinplot(y=train.Hillshade_Noon,x=train.Cover_Type)


# In[37]:


sns.boxplot(test.Hillshade_Noon)


# In[38]:


sns.distplot(train.Hillshade_Noon)
plt.grid()
plt.title('train_Hillshae_Noon')


# In[39]:


sns.boxplot(train.Hillshade_3pm)


# In[40]:


sns.distplot(train.Hillshade_3pm,color='green')
plt.grid()


# In[41]:


sns.violinplot(x=train.Cover_Type,y=train.Hillshade_3pm)


# In[42]:


train.head()


# In[43]:


#all the remaining columns Wilderness_Area and soil type are binary variables
# checking whethere they contain other than zero or one
col=list(train.columns)

for i in range(11,55):
    filter=(train.iloc[:,i]!=0) & (train.iloc[:,i]!=1)
    if (filter.sum()!=0):
        print(col[i])
    


# In[44]:


train['Horizontal_Distance_To_Fire_Points'].describe()


# In[45]:


sns.boxplot(train['Horizontal_Distance_To_Fire_Points'])


# In[46]:


sns.violinplot(x=train.Cover_Type,y=train.Horizontal_Distance_To_Fire_Points)


# In[47]:


# there has been a matchin of patter in train and test sets 
sns.boxplot(test.Horizontal_Distance_To_Fire_Points)


# In[48]:


sns.distplot(train.Horizontal_Distance_To_Fire_Points)
plt.grid()
print(train.Horizontal_Distance_To_Fire_Points.skew())


# In[49]:


# After applying the log transformation there has been decrease in the skewness
sns.distplot(np.log(1+train.Horizontal_Distance_To_Fire_Points))
plt.grid()
print(np.log(1+train.Horizontal_Distance_To_Fire_Points.skew()))


# In[50]:


train['Horizontal_Distance_To_Fire_Points']=np.log(1+train.Horizontal_Distance_To_Fire_Points)
test['Horizontal_Distance_To_Fire_Points']=np.log(1+test.Horizontal_Distance_To_Fire_Points)


# ### Preprocessing and Feature Engineering

# In[51]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
sd=StandardScaler()


# In[52]:


# standardizing the columns except 'soil type and wilderness_area since they are binary  

df_train=train.iloc[:,1:11]
df_train['dist_hydr']=train['dist_hydr']
df_train.info()


# In[53]:


# similarly slice the columns for the test datset

df_test=test.iloc[:,1:11]
df_test['dist_hydr']=test['dist_hydr']
df_test.info()


# In[54]:


sd.fit(df_train)
df_train=sd.transform(df_train)


# In[55]:


df_train[:10,1]


# In[56]:


train.iloc[:,1:11]=df_train[:,0:10]


# In[57]:


train['dist_hydr']=df_train[:,10]


# In[58]:


df_test=sd.transform(df_test)
test.iloc[:,1:11]=df_test[:,0:10]
test['dist_hydr']=df_test[:,10]


# In[59]:


# drop id both from train and test columns
train.drop(columns=['Id'],axis=1,inplace=True)


# In[60]:


Id=test['Id']
test.drop(columns=['Id'],axis=1,inplace=True)


# In[61]:


train_corr=train.corr()


# In[62]:


# correlated columsn with target lable are plotted in descending order
# we can eliminate least correlated columns when we have hign dimmensional data which in not inour case 
train_corr['Cover_Type'].abs().sort_values(ascending=False)


# In[63]:


sns.heatmap(train_corr)


# In[64]:


# Creating a new features by adding higlhy correlated features with target 
# also independent variables should not correlate with each other

print(train_corr.loc['Soil_Type38','Soil_Type39'])
print(train_corr.loc['Soil_Type38','Wilderness_Area1'])
print(train_corr.loc['Soil_Type39','Wilderness_Area1'])


# In[65]:


train['soil_type38,39']=train['Soil_Type38']+train['Soil_Type39']
train['soil_38_Wilde_area_1']=train['Soil_Type38']+train['Wilderness_Area1']
train['soil_39_Wilde_area_1']=train['Soil_Type39']+train['Wilderness_Area1']

test['soil_type38,39']=test['Soil_Type38']+test['Soil_Type39']
test['soil_38_Wilde_area_1']=test['Soil_Type38']+test['Wilderness_Area1']
test['soil_39_Wilde_area_1']=test['Soil_Type39']+test['Wilderness_Area1']


# In[66]:


#seperating the target

X=train.drop(columns='Cover_Type',axis=1)
y=train['Cover_Type']


# In[67]:


X.info()


# In[68]:


from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=0.25)


# In[69]:


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


# In[70]:


clf_accuracy=[]


# In[71]:


# Logistic Regression
lg=LogisticRegression(max_iter=1000)
lg.fit(x_train,y_train)
pred=lg.predict(x_valid)
clf_accuracy.append(accuracy_score(y_valid,pred))
print(accuracy_score(y_valid,pred))


# In[72]:


#plot the accuracy for different values of neighbor
# from the below plot take n_neighnors=4 as it gives the optimal value

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()


l=[i for i in range(1,11)]
accuracy=[]

for i in l:
    model=KNeighborsClassifier(n_neighbors=i,weights='distance')
    model.fit(x_train,y_train)
    pred=model.predict(x_valid)
    accuracy.append(accuracy_score(y_valid,pred))
    
plt.plot(l,accuracy)
plt.title('knn_accuracy plot')
plt.xlabel('neighbors')
plt.ylabel('accuracy')
plt.grid()

print(max(accuracy))

clf_accuracy.append(max(accuracy))


# In[73]:


# Support Vector Machines
from sklearn.svm import SVC
model=SVC(kernel='rbf')
model.fit(x_train,y_train)
pred=(model.predict(x_valid))
clf_accuracy.append(accuracy_score(y_valid,pred))
print(accuracy_score(y_valid,pred))


# In[74]:


# Random Forest Classfier
rand=RandomForestClassifier()
rand.fit(x_train,y_train)
pred=rand.predict(x_valid)
clf_accuracy.append(accuracy_score(y_valid,pred))
print(accuracy_score(y_valid,pred))


# In[75]:


# xgboost
xgb=XGBClassifier(max_depth=7)
xgb.fit(x_train,y_train)
pred=xgb.predict(x_valid)
clf_accuracy.append(accuracy_score(y_valid,pred))
print(accuracy_score(y_valid,pred))


# In[76]:


# Naive Bayes Classifier
nb=GaussianNB()
nb.fit(x_train,y_train)
pred=nb.predict(x_valid)
clf_accuracy.append(accuracy_score(y_valid,pred))
print(accuracy_score(y_valid,pred))


# In[77]:


classifier_list=['log_regression','knn','svm','rforest','xgboost','nbayes']


# In[78]:


sns.barplot(x=clf_accuracy,y=classifier_list)
plt.grid()
plt.xlabel('accuracy')
plt.ylabel('classifier')
plt.title('classifier vs accuracy plot')

#leela_submission=pd.DataFrame({'Id': Id,'Cover_Type':stack_res})


# In[79]:


# we use random forest for us final prediction
# We fit the whole training data given to us 
rand=RandomForestClassifier()
rand.fit(X,y)
pred=rand.predict(test)


# In[80]:


leela_submission=pd.DataFrame({'Id': Id,'Cover_Type':pred})
leela_submission.to_csv('leela_submision.csv',index=False)


# In[ ]:




