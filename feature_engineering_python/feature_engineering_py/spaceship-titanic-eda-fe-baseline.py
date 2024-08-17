#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! 😊

# In[1]:


import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("../input/spaceship-titanic/train.csv")
test = pd.read_csv("../input/spaceship-titanic/test.csv")
submission = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")

display(train.head())
display(test.head())
display(submission.head())


# In[2]:


print("Train shape: ", train.shape)
print("Test shape: ", test.shape)


# # Data Cleaning

# In[3]:


print("Train dublicated num: ", train.duplicated().sum())
print("Test dublicated num: ", test.duplicated().sum())


# In[4]:


def check_objcol_quality(df, object_cols, fold=5):
    from sklearn.model_selection import StratifiedKFold
    
    X = df.drop("Transported", 1)
    y = df[["Transported"]]
    
    good_label_cols = []
    bad_label_cols = []
    for col in object_cols:
        skf = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)
        goodness_key = True
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
            y_train, y_test = y.loc[train_index, :], y.loc[test_index, :]
            
            goodness_key = np.logical_and(goodness_key, set(X_train[col]) == set(X_test[col])) 
        if goodness_key:
            good_label_cols.append(col)
        else:
            bad_label_cols.append(col)
    
    print("good_label_cols len: ", len(good_label_cols))
    print(good_label_cols)
    print("bad_label_cols len: ", len(bad_label_cols))
    print(bad_label_cols)
    


# In[5]:


numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(train.columns) - set(numeric_cols))
object_cols.remove("Transported")

print("numeric_cols: ", numeric_cols)
check_objcol_quality(train, object_cols, fold=10)


# In[6]:


display(train.isna().sum())
display(test.isna().sum())


# In[7]:


import missingno as msno

sorted_train = train.sort_values("Age")
msno.matrix(sorted_train);


# # Feature Engineering

# In[8]:


train[["Cabin_1", "Cabin_2", "Cabin_3"]] = train["Cabin"].str.split("/", expand=True)
check_objcol_quality(train, ["Cabin_1", "Cabin_2", "Cabin_3"], fold=10)


# In[9]:


train.drop(["Cabin_1", "Cabin_2", "Name", "Cabin"], axis=1, inplace=True)
train.shape


# # Distributions

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=train, x='Age', hue="Transported", ax=ax);


# In[11]:


fig, axes = plt.subplots(2, 3, figsize=(16, 8))
sns.countplot(data=train, x='HomePlanet', hue="Transported", ax=axes[0][0]);
sns.countplot(data=train, x='CryoSleep', hue="Transported", ax=axes[1][1]);
sns.countplot(data=train, x='Destination', hue="Transported", ax=axes[0][2]);
sns.countplot(data=train, x='VIP', hue="Transported", ax=axes[1][0]);
sns.countplot(data=train, x='Cabin_3', hue="Transported", ax=axes[0][1]);
sns.countplot(data=train, x='Transported', ax=axes[1][2]);


# In[12]:


sns.pairplot(data=train, kind="scatter", hue="Transported", diag_kind="auto", corner=True, 
            x_vars=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'],
            y_vars=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']);


# In[13]:


def plot_umap(embedding, df, col, ax=None):
    colors = pd.factorize(df.loc[:, col])
    colors_dict = {}
    for index, label in enumerate(df[col].unique()):
        colors_dict[index] = label
    color_list = sns.color_palette(None, len(df[col].unique()))
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(12,12))
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);
    else:
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);


# In[14]:


import umap

train_dropna = train.dropna()
embedding = umap.UMAP(n_neighbors=10,
                      min_dist=0.3,
                      metric='correlation').fit_transform(train_dropna[numeric_cols])


# In[15]:


fig, axes = plt.subplots(2, 2, figsize=(16, 16))
plot_umap(embedding, train_dropna, "Transported", ax=axes[0][0])
plot_umap(embedding, train_dropna, "CryoSleep", ax=axes[0][1])
plot_umap(embedding, train_dropna, "HomePlanet", ax=axes[1][0])
plot_umap(embedding, train_dropna, "Destination", ax=axes[1][1])


# # P-Values

# In[16]:


def p_value_warning_background(cell_value):
    highlight = 'background-color: lightcoral;'
    default = ''
    if cell_value > 0.05:
            return highlight
    return default

def calculate_p_values(df, target, numeric_cols):
    from scipy.stats import pearsonr
    
    p_values_list = []
    for c in numeric_cols:
        p = round(pearsonr(df.loc[:,target], df.loc[:,c])[1], 4)
        p_values_list.append(p)
    p_values_target_list = np.array(p_values_list)
    p_values_target_list = p_values_target_list.reshape(len(numeric_cols), 1)
    p_values_df = pd.DataFrame(p_values_target_list, columns=[target], index=numeric_cols)
    return p_values_df


# In[17]:


p_values_df = calculate_p_values(train_dropna, "Transported", numeric_cols)
p_values_df.style.applymap(p_value_warning_background)


# # Correlations

# In[18]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(train[numeric_cols + ["Transported"]].corr(), annot=True, ax=ax, 
            xticklabels=numeric_cols + ["Transported"],
            yticklabels=numeric_cols + ["Transported"]);


# # Modelling

# In[19]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret[full]\n')


# In[20]:


from pycaret.classification import *

numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(train.columns) - set(numeric_cols))
object_cols.remove("Transported")
ignore_cols = ["PassengerId"]

clf = setup(data=train,
            target='Transported',
            numeric_features = numeric_cols,
            categorical_features = object_cols,
            ignore_features = ignore_cols,
            session_id = 42,
            use_gpu = False,
            silent = True,
            fold = 10,
            n_jobs = -1)


# In[21]:


model_rf = create_model('rf')


# In[22]:


plot_model(model_rf, plot='error')


# In[23]:


plot_model(model_rf, plot='confusion_matrix')


# # Submission

# In[24]:


import gc
gc.collect()
_, _, test['Cabin_3'] = test["Cabin"].str.split("/", expand=True)
unseen_predictions = predict_model(model_rf, data=test)
unseen_predictions.head()


# In[25]:


assert(len(test.index)==len(unseen_predictions))
sub = pd.DataFrame(list(zip(submission.PassengerId, unseen_predictions.Label)),columns = ['PassengerId', 'Transported'])
sub.to_csv('submission.csv', index = False)
sub.head()


# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns

train_test_preds = pd.DataFrame()
train_test_preds['label'] = list(train['Transported']) + list(unseen_predictions['Label'])
train_test_preds['train_test'] = 'Test preds'
train_test_preds.loc[0:len(train[['Transported']]), 'train_test'] = 'Training'

fig, ax = plt.subplots(figsize=(16,3))
sns.countplot(data=train_test_preds, x='label', hue='train_test', ax=ax)
plt.xticks(rotation=90)
plt.show()

