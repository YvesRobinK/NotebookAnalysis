#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[3]:


df_train


# In[4]:


df_train.label.unique()


# # Explanatory Data Analysis

# In[5]:


plt.figure(figsize=(8,6))
ax = sns.countplot(x='label',data=df_train)

plt.title("Label Distribution")
total= len(df_train.label)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')


# In[6]:


df_train.describe()


# In[7]:


df_train.sum(axis=1)


# In[8]:


df_train.shape


# In[9]:


pixels = df_train.columns.tolist()[1:]
df_train["sum"] = df_train[pixels].sum(axis=1)

df_test["sum"] = df_test[pixels].sum(axis=1)


# In[10]:


df_train.groupby(['label'])['sum'].mean()


# In[11]:


# separate target values from df_train
targets = df_train.label
features = df_train.drop("label",axis=1)


# In[12]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features[:] = scaler.fit_transform(features)
df_test[:] = scaler.transform(df_test)


# In[13]:


del df_train


# In[14]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=2)
Y_sklearn = sklearn_pca.fit_transform(features)


# In[15]:


Y_sklearn


# In[16]:


#referred to https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html and  https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction


with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 8))
    for lab, col in zip((0,1,2,3,4,5,6,7,8,9),
                       ('blue','red','green','yellow','purple','black','brown','pink','orange','beige')):
        plt.scatter(Y_sklearn[targets==lab, 0],
                    Y_sklearn[targets==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# In[17]:


features.index


# In[18]:


sklearn_pca_3 = sklearnPCA(n_components=3)
Y_sklearn_3 = sklearn_pca_3.fit_transform(features)
Y_sklearn_3_test = sklearn_pca_3.transform(df_test)


# In[19]:


# Store results of PCA in a data frame
result=pd.DataFrame(Y_sklearn_3, columns=['PCA%i' % i for i in range(3)], index=features.index)


# In[20]:


result


# In[21]:


my_dpi=96
plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)

with plt.style.context('seaborn-whitegrid'):
    my_dpi=96
    fig = plt.figure(figsize=(10, 10), dpi=my_dpi)
    ax = fig.add_subplot(111,projection ='3d')
    for lab, col in zip((0,1,2,3,4,5,6,7,8,9),
                       ('blue','red','green','yellow','purple','black','brown','pink','orange','beige')):
        plt.scatter(Y_sklearn[targets==lab, 0],
                    Y_sklearn[targets==lab, 1],
                    label=lab,
                    c=col,s =60)                
        
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title("PCA on the Handwriting Data")
    plt.show()


# In[22]:


encoder = LabelEncoder()
targets[:] = encoder.fit_transform(targets[:])


# In[23]:


X_train,X_val, y_train,y_val = train_test_split(result,targets,random_state=1)


# # Making a Model and Predictions

# In[24]:


# 3 Principal Components
model = XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, 
                        num_classes=10)

history = model.fit(X_train, y_train,eval_set =[(X_val,y_val)],early_stopping_rounds =50)
acc = accuracy_score(y_val, model.predict(X_val))
print(f"Accuracy: , {round(acc,3)}")







# In[25]:


X_train,X_val, y_train,y_val = train_test_split(features,targets,random_state=1)


# In[26]:


model = XGBClassifier(max_depth=5, objective='multi:softprob', n_estimators=1000, 
                        num_classes=10)

history = model.fit(X_train, y_train,eval_set =[(X_train,y_train),(X_val,y_val)],early_stopping_rounds =5)
acc = accuracy_score(y_val, model.predict(X_val))
print(f"Accuracy: , {round(acc,3)}")


# In[27]:


results = model.evals_result()


# In[28]:


from matplotlib import pyplot
# plot learning curves
plt.figure(figsize=(10, 8))
pyplot.plot(results['validation_0']['mlogloss'], label='train')
pyplot.plot(results['validation_1']['mlogloss'], label='test')
# show the legend
pyplot.legend()
plt.xlabel('iterations')
plt.ylabel('mlogloss')
# show the plot
pyplot.show()


# In[29]:


from xgboost import plot_importance
ax = plot_importance(model,max_num_features=10)
fig = ax.figure
fig.set_size_inches(10,8)
plt.show()


# In[30]:


predictions = model.predict(df_test)



# In[31]:


output = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
output['Label'] = predictions
output.to_csv('submission.csv',index=False)

