#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import itertools
import random 
import warnings
warnings.filterwarnings("ignore")
random.seed(0)


# In[ ]:


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df =  pd.read_csv('/kaggle/input/titanic/test.csv')

# Join the train and test dataframes so the data preprocessing 
# will be done simultaneously in both datasets 
full_df = train_df.append(test_df, ignore_index=True)
print(f'There are {full_df.shape[0]} rows and {full_df.shape[1]} columns in the full dataframe.')


# In[ ]:


def data_preprocessing(df):
  
  # Label-encode the sex of a passenger 
  df['Sex'] = df['Sex'].replace(['male'],0)
  df['Sex'] = df['Sex'].replace(['female'],1)

  # Initialize new columns 
  df['title'] = np.NaN
  df['alone'] = np.NaN
  df['cabin_class'] = np.NaN

  # Identify if a passenger is alone in the ship 
  for i,_ in enumerate(df['alone']):
    if df['SibSp'][i] + df['Parch'][i] == 0:
      df['alone'][i] = 1
    else:
      df['alone'][i] = 0 

  # Handle missing values
  cols = ['SibSp','Parch','Fare','Age']
  for col in cols:
    df[col].fillna(df[col].median(), inplace = True)
    
  # Feature-engineer the cabin-class 
  for i,row in enumerate(df['Cabin']):
    # Get cabin class 
    df['cabin_class'][i] =  str(row)[:1]

  # Count the cabin distribution per class (if available) 
  cabin_distribution = {}
  count = 0 
  for row in df['cabin_class']:
    if row != 'n':
      count += 1 
      if row not in cabin_distribution:
        cabin_distribution[row] = 1 
      else:
        cabin_distribution[row] +=1 

  # Calculate the probability of being in a sepcific cabin-class  
  cabin_pdf = {k:v / count for k, v in cabin_distribution.items()}

  # Calculate the cumulative probability of being in a specific cabin-class 
  keys, vals = cabin_pdf.keys(), cabin_pdf.values()
  cabin_cdf = dict(zip(keys, itertools.accumulate(vals)))
  cabin_cdf = sorted(cabin_cdf.items(), key=lambda x: x[1])    

  # Randomly assign cabin-classes to passengers that are missing the cabin 
  # field, based on the probabilities calculated above 
  for i,row in enumerate(df['cabin_class']):
    random_num = random.random()
    if row == 'n':
      if random_num < cabin_cdf[0][1]:
        df['cabin_class'][i] =  cabin_cdf[0][0]
      elif cabin_cdf[0][1] <= random_num < cabin_cdf[1][1]:
        df['cabin_class'][i] =  cabin_cdf[1][0]

      elif cabin_cdf[1][1] <= random_num < cabin_cdf[2][1]:
        df['cabin_class'][i] =  cabin_cdf[2][0]
      
      elif cabin_cdf[2][1] <= random_num < cabin_cdf[3][1]:
        df['cabin_class'][i] =  cabin_cdf[2][0]

      elif cabin_cdf[3][1] <= random_num < cabin_cdf[4][1]:
        df['cabin_class'][i] =  cabin_cdf[3][0]

      elif cabin_cdf[3][1] <= random_num < cabin_cdf[4][1]:
        df['cabin_class'][i] =  cabin_cdf[4][0]

      elif cabin_cdf[4][1] <= random_num < cabin_cdf[5][1]:
        df['cabin_class'][i] =  cabin_cdf[4][0]
      
      elif cabin_cdf[5][1] <= random_num < cabin_cdf[6][1]:
        df['cabin_class'][i] =  cabin_cdf[5][0]

      elif cabin_cdf[6][1] <= random_num < cabin_cdf[7][1]:
        df['cabin_class'][i] =  cabin_cdf[6][0]
      else:
        df['cabin_class'][i] = cabin_cdf[7][0]

  # Perform feature engineering to obtain additional title-info 
  for i,row in enumerate(df['Name']):
    # Get person's title 
    df['title'][i] = row.split(',')[1].split('.')[0]

  # Embarked one-hot encoding 
  embarked_dummies = pd.get_dummies(df.Embarked, prefix='Embarked')
  df = pd.concat([df, embarked_dummies], axis=1)

  # Person's title one-hot encoding 
  title_dummies = pd.get_dummies(df.title, prefix='title')
  df = pd.concat([df, title_dummies], axis=1)

  # Cabin class one-hot encoding 
  cabin_class_dummies = pd.get_dummies(df.cabin_class, prefix = 'cabin_class')
  df = pd.concat([df, cabin_class_dummies], axis = 1)

  #Remove unecessary columns 
  del df['Name']
  del df['PassengerId']
  del df['title']
  del df['Embarked']
  del df['Cabin']
  del df['Ticket']
  del df['cabin_class']

  return df 


# In[ ]:


# Preprocess the data and create the train / test sets 
full_df = data_preprocessing(full_df)
X_train = full_df[:891]
y_train = full_df['Survived'][:891]
X_test = full_df[891:]
del X_train['Survived']
del X_test['Survived']


print(f'There are {X_train.shape[0]} rows and {X_train.shape[1]} columns in the training data.\n')
print(f'There are {X_test.shape[0]} rows and {X_test.shape[1]} columns in the test data.')


# In[ ]:


# Fit the model 
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(max_iter=1000,C=0.175,random_state=42)
LR.fit(X_train, y_train)
training_accuracy = LR.score(X_train, y_train)
predictions = LR.predict(X_test)
print("Training accuracy: %.2f%%" % (training_accuracy * 100.0))


# In[ ]:


predictions = [int(x) for x in predictions]
submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})
submission.to_csv('submission.csv',index = False)

