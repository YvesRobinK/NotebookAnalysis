#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train_path = "/kaggle/input/playground-series-s3e17/train.csv"
df = pd.read_csv(train_path)
df.head(3)


# In[3]:


df.info()


# In[4]:


df = df.drop(columns=['id', 'Product ID'], axis=1)
df.head()


# In[5]:


df.describe(include='object')


# In[6]:


df['Type'] = pd.Categorical(df['Type'])
num_col = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
cat_col = ['Type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']


# In[7]:


def plot_grouped_data(df, target_col, numerical_vars):
    target_groups = df.groupby(target_col)

    # Plot KDE for numerical variables
    num_plots_per_row = 3
    num_plots = len(numerical_vars)
    num_rows = (num_plots - 1) // num_plots_per_row + 1

    for i, var in enumerate(numerical_vars):
        if i % num_plots_per_row == 0:
            plt.figure(figsize=(15, 5))

        plt.subplot(1, num_plots_per_row, (i % num_plots_per_row) + 1)
        for group_name, group_data in target_groups:
            sns.kdeplot(data=group_data, x=var, label=str(group_name))
        plt.xlabel(var)
        plt.ylabel('Density')
        plt.legend(title=target_col)

        if (i + 1) % num_plots_per_row == 0 or (i + 1) == num_plots:
            plt.tight_layout()
            plt.show()


# In[8]:


plot_grouped_data(df,'Machine failure', num_col)

# Inference: Two classes numerical feature's distributions overlap a lot.


# In[9]:


for col in cat_col:
    print(col, df[col].unique())


# In[10]:


def plot_categorical_bar_grouped(df, target_col, cat_cols):
    num_plots_per_row = 3
    num_plots = len(cat_cols)
    num_rows = (num_plots - 1) // num_plots_per_row + 1

    for i, col in enumerate(cat_cols):
        if i % num_plots_per_row == 0:
            plt.figure(figsize=(15, 5))

        plt.subplot(1, num_plots_per_row, (i % num_plots_per_row) + 1)
        sns.countplot(data=df, x=col, hue=target_col)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.title(f'Bar Plot of {col} by {target_col}')
        plt.legend(title=target_col)

        if (i + 1) % num_plots_per_row == 0 or (i + 1) == num_plots:
            plt.tight_layout()
            plt.show()


# In[11]:


plot_categorical_bar_grouped(df, 'Machine failure', cat_col)


# In[12]:


def plot_categorical_count_with_percentage(df, target_col):
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x=target_col)
    plt.title(f'Bar Plot of {target_col}')

    # Calculate the percentage for each category
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')

    plt.show()
    
plot_categorical_count_with_percentage(df, 'Machine failure')

#Inference: this is the imbalanced dataset


# Feature engineering

# In[13]:


def generate_extra_features(data):

    # Total Work Factor
    data['Total Work Factor'] = data[['HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)

    # Temperature Difference
    data['Temperature Difference'] = data['Process temperature [K]'] - data['Air temperature [K]']

    # Power Factor
    data['Power Factor'] = data['Rotational speed [rpm]'] * data['Torque [Nm]']

    # Tool Utilization
    max_tool_wear = data.groupby('Type')['Tool wear [min]'].transform('max')
    data['Tool Utilization'] = data['Tool wear [min]'] / max_tool_wear
    
    # Interaction Terms
    data['Interaction Term'] = data['Rotational speed [rpm]'] * data['Torque [Nm]']

    # Polynomial Features
    data['Temp^2'] = data['Process temperature [K]'] ** 2
    data['Temp^3'] = data['Process temperature [K]'] ** 3

    return data


# In[14]:


df = generate_extra_features(df)
df.head()


# Modelling

# In[15]:


from sklearn.model_selection import train_test_split

X = df.drop(columns=['Machine failure'],axis=1)
y = df['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[16]:


def get_categorical_and_numeric_columns(df):
    categorical_columns = []
    numeric_columns = []

    for column in df.columns:
        if column in ['Type']:
            categorical_columns.append(column)
        elif column not in ['Machine failure','TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            numeric_columns.append(column)
    
    encoded_categorical = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    return categorical_columns, numeric_columns, encoded_categorical


categorical_columns, numeric_columns, encoded_categorical  = get_categorical_and_numeric_columns(df)
print('Categorical columns:' , categorical_columns)
print('Numerical columns:', numeric_columns)
print('Encoded coloumns:', encoded_categorical)


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Define the transformations for each column type
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(), categorical_columns),
        ('numeric', StandardScaler(), numeric_columns)
    ])

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', XGBClassifier()),
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)


# In[18]:


# Predict on the test set
y_pred_train = pipeline.predict_proba(X_train)[:,-1]

# Predict on the test set
y_pred_test = pipeline.predict_proba(X_test)[:,-1]
y_pred_test


# In[19]:


from sklearn.metrics import roc_auc_score

# Calculate ROC-AUC score
roc_auc_train = roc_auc_score(y_train, y_pred_train)
print(f"ROC-AUC Score Train: {roc_auc_train:.2f}")

roc_auc_test = roc_auc_score(y_test, y_pred_test)
print(f"ROC-AUC Score Test: {roc_auc_test:.2f}")


# Submit to the competition

# In[20]:


test_path = "/kaggle/input/playground-series-s3e17/test.csv"
df = pd.read_csv(test_path)

df = generate_extra_features(df)

X = df.drop(columns=['id', 'Product ID'], axis=1)
X.head()


# In[21]:


submission = pd.DataFrame()
submission['id'] = df['id']

submission['Machine failure'] = pipeline.predict_proba(X)[:,-1]
submission.head()


# In[22]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




