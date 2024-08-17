#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 


# In[2]:


train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Concatenate the train and test dataframes so the preprocessing is applied to both 
full_data = pd.concat([train_data,test_data]).reset_index(drop = True)

sale_price = train_data['SalePrice'].reset_index(drop=True)
# Remove the Sale Price dependent variable from the combined dataset 
del full_data['SalePrice']

print(f'Train dataframe contains {train_data.shape[0]} rows and {train_data.shape[1]} columns.\n')
print(f'Test dataframe contains {test_data.shape[0]} rows and {test_data.shape[1]} columns.\n')
print(f'The merged dataframe contains {full_data.shape[0]} rows and {full_data.shape[1]} columns.')


# In[3]:


# Drop columns with more than 45% missing data 
cols_to_drop = []
for column in full_data:
  if full_data[column].isnull().sum() / len(full_data) >= 0.4:
    cols_to_drop.append(column)
full_data.drop(cols_to_drop, axis=1, inplace=True)

print(f'{len(cols_to_drop)} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')


# In[4]:


# Now replace the NA values with the median for the numerical 
# columns and scale the data
scaler = MinMaxScaler()

columns = full_data.columns.values
for column in columns:
  if full_data[column].dtype == np.int64 or full_data[column].dtype == np.float64:
    full_data[column] = full_data[column].fillna(full_data[column].median())
    full_data[column] = scaler.fit_transform(np.array(full_data[column]).reshape(-1,1))

# Print the updated data  
full_data.head()


# In[5]:


# Calculate the correlation of the numerical variables with the Sale Price 
# Use the training dataset that inludes the Sale Price variable  

corr = train_data.corr()
plt.subplots(figsize=(19,10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, vmax=0.7, cmap = cmap, square=True)

cols_to_drop = []
# Get the correlation of the dependent variable with the rest of the features
sale_price_corr = train_data.corr()['SalePrice'][:-1] 

# Drop all the columns from the full data that correlate < |0.12| with the sale price, 
# since will add little value to the model 
for column,row in sale_price_corr.iteritems():
  if abs(float(row)) < 0.12:
    cols_to_drop.append(column)
full_data.drop(cols_to_drop, axis=1, inplace=True)

print(f'{len(cols_to_drop)} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')


# In[6]:


# Drop the columns that have > 6 unique categorical classes
count = 0 
columns = full_data.columns.values
for column in columns:
  if full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() > 6:
    count += 1 
    full_data.drop(column, axis = 1, inplace = True)

print(f'{count} columns dropped, the full dataset now comprises of {full_data.shape[1]} variables.')


# In[7]:


# Replace nas with the most frequent occurring value in the categorical data 
full_data = full_data.fillna(full_data.mode().iloc[0])


# In[8]:


# Label / one-hot encode the categorical variables
# One-hot encode the columns that have > 2 categorical variables
# Label-encode the columns that have only 2 categorical variables 

# Instanciating the labelencoder
labelencoder = LabelEncoder()
cols_to_drop = []

columns = full_data.columns.values
for column in columns:
    if full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() > 2: 
      dummies = pd.get_dummies(full_data[column], prefix = str(column))
      cols_to_drop.append(column)
      full_data = pd.concat([full_data, dummies], axis = 1)
    elif full_data[column].dtype not in (np.int64, np.float64) and full_data[column].nunique() < 3: 
      full_data[column] = labelencoder.fit_transform(full_data[column])
      cols_to_drop.append(column)

full_data.drop(cols_to_drop, axis = 1, inplace = True)
print(f'The new dataframe comprises of {test_data.shape[0]} rows and {test_data.shape[1]} columns.\n')


# In[9]:


#Now that the data have been processes split again into train and test 
train_df = full_data[:train_data.shape[0]]
test_df =  full_data[train_data.shape[0]:]


# In[10]:


import tensorflow as tf
#Set random seed
tf.random.set_seed(42)

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(1000), 
                            tf.keras.layers.BatchNormalization(), 
                            tf.keras.layers.Dense(100), 
                            tf.keras.layers.Dense(100),
                            tf.keras.layers.Dropout(0.1), 
                            tf.keras.layers.Dense(1) 
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0003), 
                          metrics=['mae'])

# Fit the model and save the history 
history = model.fit(train_df,sale_price, epochs=400, verbose=0)

# Plot the model trained for 400 total epochs loss curves
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

DNN_predictions = model.predict(test_df)
DNN_predictions = tf.squeeze(DNN_predictions, axis = 1)
DNN_predictions = np.array(DNN_predictions)


# In[11]:


from xgboost import XGBRegressor

xgboost = XGBRegressor(learning_rate=0.008,
                       n_estimators=6000,
                       max_depth=8,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

xgb = xgboost.fit(train_df,sale_price)
training_accuracy = xgb.score(train_df,sale_price)
print("Training accuracy: %.2f%%" % (training_accuracy * 100.0))
xgb_predictions = xgb.predict(test_df)


# In[12]:


predictions = [(xgb_pred + DNN_pred) / 2 for xgb_pred, DNN_pred in zip(xgb_predictions, DNN_predictions)]
submission = pd.DataFrame({'ID':test_data['Id'],'SalePrice':predictions})
submission.to_csv('submission.csv',index = False)
submission

