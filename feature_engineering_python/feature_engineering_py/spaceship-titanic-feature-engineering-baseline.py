#!/usr/bin/env python
# coding: utf-8

# In this notebook, we establish a logisitc regression baseline for the Spaceship Titanic dataset. Specifically, we do the following:
# 
# - Feature Engineering Based On EDA
# - Imputing Missing Values Based On Insights (and Kaggle Discussions!)
# - Logistic Regression Baselines, where we also judge the usability of 2 dense categorical features (extracted from `PassengerId` and `Cabin` respectively)
# 
# By the end of this notebook, we will carry out 7 (!) experiments and have a definitive accuracy score that a final model should try to beat.
# 
# This is Part 2 of a three part series.
# 
# * Part 1: [Spaceship Titanic - Exploratory Data Analysis](https://www.kaggle.com/code/defcodeking/spaceship-titanic-exploratory-data-analysis)
# * Part 2: [Spaceship Titanic - Logistic Regression Baselines](https://www.kaggle.com/code/defcodeking/spaceship-titanic-logistic-regression-baselines) (you are here!)
# * Part 3: [Ensembling (And Optuna 😉) Is All You Need!](https://www.kaggle.com/code/defcodeking/ensembling-and-optuna-is-all-you-need)

# # Imports

# In[1]:


from sklearn import linear_model, preprocessing, impute, model_selection, metrics
from scipy.stats import boxcox
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.set_style("ticks")
sns.despine()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()


# # Config

# In[3]:


DATA_DIR = "../input/spaceship-titanic"

def filepath(filename):
    return os.path.join(DATA_DIR, filename)


# # Load Dataset

# In[4]:


train_df = pd.read_csv(filepath("train.csv"), index_col="PassengerId")
test_df = pd.read_csv(filepath("test.csv"), index_col="PassengerId")

# Add PassengerId since we need it for feature engineering
train_df["PassengerId"] = train_df.index
test_df["PassengerId"] = test_df.index

len(train_df), len(test_df)


# # Initial Feature Engineering
# 
# See [Spaceship Titanic - Exploratory Data Analysis](https://www.kaggle.com/code/defcodeking/spaceship-titanic-exploratory-data-analysis) for more details.
# 
# > Note: All features extraced from `Cabin` will be engineered after missing values are imputed but the function is created here.

# In[5]:


expenditure_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']


# ## `GroupId`, `GroupSize` and `Alone`
# 
# The `from_passengerId()` function extracts features out of the `PassengerId` feature.

# In[6]:


def from_passengerId(df):
    split_id = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"] = split_id[0]
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("count")
    
    # Indicates whether the passenger was traveling alone or not
    df["Alone"] = df["GroupSize"] == 1
    
    return df

train_df = from_passengerId(train_df)
test_df = from_passengerId(test_df)


# In[7]:


train_df.head()


# ## Presence of Missing Values
# 
# The function `mssing_value_features()` takes a list of columns and adds a new column indicating whether there is a missing value present or not. It also adds an additional feature called `TotalExpense_missing` which indicates whether `TotalExpense` (the sum of all the expenditure columns) is missing when nulls are not ignored. This sort of "summarises" the missing values in the expenditure columns.

# In[8]:


def missing_value_features(df, columns, expenditure_columns):
    for column in columns:
        df[f"{column}_missing"] = df[column].isna()
    
    # An additional feature which encodes whether TotalExpense is missing if NAs are not ignored
    df["TotalExpense_missing"] = df[expenditure_columns].sum(axis=1, skipna=False).isna()
    return df

columns = ["RoomService", "FoodCourt", "ShoppingMall", "Cabin", "VIP"]
train_df = missing_value_features(train_df, columns, expenditure_columns)
test_df = missing_value_features(test_df, columns, expenditure_columns)


# In[9]:


train_df.head()


# ## `TotalExpense`
# 
# The function `from_expenditure_features()` extracts a feature from all the expenditure columns which is their sum (ignoring nulls).

# In[10]:


def from_expenditure_features(df, expenditure_columns):
    df["TotalExpense"] = df[expenditure_columns].sum(axis=1)
    return df

train_df = from_expenditure_features(train_df, expenditure_columns)
test_df = from_expenditure_features(test_df, expenditure_columns)


# In[11]:


train_df.head()


# ## `CabinDeck`, `CabinNum` and `CabinSide`
# 
# The function `from_cabin()` splits the `Cabin` feature into its three constituents: `Deck`, `Num` and `Side`.

# In[12]:


def from_cabin(df):
    df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split("/", expand=True)
    return df


# # Categorical Missing Values
# 
# Missing values in some categorical features will be filled by the mode. Specifically, `HomePlanet`, `CryoSleep` and `Destination` will use the feature-level mode, while `Cabin` will use group-level mode taken in two ways. First, the null values will be filled by the group-mode based on `GroupId`. This will leave passengers who were travelling alone or those who belong to a group which has null values in `Cabin` for every member. These will be filled by the group-mode based on `HomePlanet` and `Destination`.
# 
# `VIP` is a feature that cannot be filled with the mode. Instead, we will use some heuristics discovered by other Kagglers and posted on the competitions discussion page. As it turns out, the heuristics will not be enough and so, the remaining values will be filled according to the probability distribution of `VIP`.

# ## `HomePlanet`, `CryoSleep`, `Destination`

# In[13]:


def simple_mode_replacement(df, columns):
    df[columns] = df[columns].fillna(df[columns].mode().iloc[0])
    return df

columns = ["HomePlanet", "CryoSleep", "Destination"]
train_df = simple_mode_replacement(train_df, columns)
test_df = simple_mode_replacement(test_df, columns)


# In[14]:


train_df[columns].isna().any()


# In[15]:


train_df.head()


# ## `Cabin`

# In[16]:


def group_mode_replacement(df, groupby, column):
    # Find all passengers belonging to groups where at least one member has a non-null column value
    temp = df.groupby(groupby).filter(lambda x: x[column].notna().any())
    
    # Replace by mode
    func = lambda x: x.fillna(x.mode().iloc[0]) if x.isna().any() else x
    temp[column] = temp.groupby(groupby)[column].transform(func)
    
    # Update the original dataframe
    df.loc[temp.index, column] = temp[column]
    
    return df

train_df = group_mode_replacement(train_df, groupby="GroupId", column="Cabin")
test_df = group_mode_replacement(test_df, groupby="GroupId", column="Cabin")


# In[17]:


train_df.head()


# There are still 99 passengers with null `Cabin` values in the train set and 63 in the test set. These can be filled with the mode of groups by `HomePlanet` and `Destination`.

# In[18]:


train_df["Cabin"].isna().sum()


# In[19]:


test_df["Cabin"].isna().sum()


# In[20]:


train_df = group_mode_replacement(train_df, groupby=["HomePlanet", "Destination"], column="Cabin")
test_df = group_mode_replacement(test_df, groupby=["HomePlanet", "Destination"], column="Cabin")


# In[21]:


train_df["Cabin"].isna().sum()


# In[22]:


test_df["Cabin"].isna().sum()


# Now that all the null values have been filled in for `Cabin`, we will extract the features from `Cabin`.

# In[23]:


train_df = from_cabin(train_df)
test_df = from_cabin(test_df)


# In[24]:


train_df.head()


# In[25]:


columns = ["Cabin", "CabinDeck", "CabinNum", "CabinSide"]
train_df[columns].isna().any()


# In[26]:


test_df[columns].isna().any()


# ## `VIP`

# In[27]:


train_df["VIP"].isna().sum(), test_df["VIP"].isna().sum()


# The following heuristics will be followed (see [Some rules to fill NaNs](https://www.kaggle.com/competitions/spaceship-titanic/discussion/315987)):
# 
# - Passengers who have zero spending and are not in cryo sleep are not VIPs.
# - Passengers who are below or at the age of 12 are not VIPs.
# - Passengers from Earth are not VIPs.
# - Mars VIPs have `Age` >= 18, no `CryoSleep` and never go to "55 Cancri e"

# In[28]:


def impute_vip_for_no_spend(df):
    df.loc[
        (df["VIP"].isna()) & (df["TotalExpense"] == 0.0) & (~df["CryoSleep"]), "VIP"
    ] = False
    return df

def impute_vip_for_children(df):
    df.loc[(df["VIP"].isna()) & (df["Age"] <= 12), "VIP"] = False
    return df

def impute_vip_for_earthlings(df):
    df.loc[(df["VIP"].isna()) & (df["HomePlanet"] == "Earth"), "VIP"] = False
    return df

def impute_vip_for_martians(df):
    df.loc[
        (df["VIP"].isna())
        & (df["Age"] >= 18)
        & (~df["CryoSleep"])
        & (df["Destination"] != "55 Cancri e"),
        "VIP",
    ] = True
    return df


# In[29]:


def impute_vip(df):
    df = impute_vip_for_no_spend(df)
    df = impute_vip_for_children(df)
    df = impute_vip_for_earthlings(df)
    df = impute_vip_for_martians(df)
    return df

train_df = impute_vip(train_df)
test_df = impute_vip(test_df)


# In[30]:


train_df["VIP"].isna().sum(), test_df["VIP"].isna().sum()


# The remaining values are filled using the proportion of VIPs and non-VIPs. 

# In[31]:


def impute_vip_by_prob(df):
    probs = df["VIP"].value_counts() / df["VIP"].notna().sum()
    values = np.random.choice([False, True], size=df["VIP"].isna().sum(), p=probs)
    df.loc[df["VIP"].isna(), "VIP"] = values
    df["VIP"] = df["VIP"].astype(bool)
    return df

train_df = impute_vip_by_prob(train_df)
test_df = impute_vip_by_prob(test_df)


# In[32]:


train_df["VIP"].isna().sum(), test_df["VIP"].isna().sum()


# # Drop Unnecessary Features
# 
# `PassengerID`, `Cabin` and `Name` will be dropped from the dataset. It may be so that there are features that can be extracted from `Name` but most of the other Kagglers have reported decrease in accuracy when using any features from `Name`.

# In[33]:


drop = ["PassengerId", "Cabin", "Name"]
train_df = train_df.drop(drop, axis=1)
test_df = test_df.drop(drop, axis=1)


# In[34]:


train_df.head()


# Now, all the null values are in the numerical columns.

# In[35]:


train_df.isna().any()


# # Feature Encoding Categorical Variables (Except `CabinNum` and `GroupId`)
# 
# To encode the features, we will combine the two datasets into one. The functions `concat_train_test()` and `split_train_test()` handle the combining and resplitting of the datasets respectively.
# 
# `CabinNum` and `GroupSize` will not be not be encoded since they are part of the experiment.

# In[36]:


# Take out labels from training data
def concat_train_test(train, test, has_labels=False):
    transported = None
    
    # Since the test set doesn't have labels
    # If there are labels in the train set
    # They need to be dropped
    if has_labels is True:
        transported = train["Transported"].copy()
        train = train.drop("Transported", axis=1)

    # Store indices so that they can be used to
    # Split the dataset again
    train_index = train.index
    test_index = test.index

    # Concatenate the two datasets
    df = pd.concat([train, test])

    return df, train_index, test_index, transported


def split_train_test(df, train_index, test_index, transported=None):
    # Get the training set in the df according to index
    train_df = df.loc[train_index, :]
    
    # If transported is passed
    # Add it to the dataframe
    if transported is not None:
        train_df["Transported"] = transported
        
    # Get the test set in the df according to the index
    test_df = df.loc[test_index, :]
    
    return train_df, test_df


# In[37]:


# Combine the datasets
df, train_idx, test_idx, transported = concat_train_test(train_df, test_df, has_labels=True)
df.head()


# In case of logistic regression, all binary categorical features need to be encoded as `0` or `1`, while non-binary categorical features need to be one-hot encoded.
# 
# We will first convert all Boolean columns to `int` so that they are `0` or `1`.

# In[38]:


def bool2int(df):
    # Find all bool columns
    columns = [column for column in df.columns if df[column].dtype.name == "bool"]
    # Convert to integer
    df[columns] = df[columns].astype(int)
    
    return df


df = bool2int(df)


# In[39]:


df.head()


# Then, we will encode `CabinSide` as `0` or `1`.

# In[40]:


df["CabinSide"] = df["CabinSide"].map({"S": 0, "P": 1})


# Finally, we will use `pd.get_dummies()` to one-hot encode the non-binary features.

# In[41]:


to_be_encoded = ["HomePlanet", "Destination", "GroupSize", "CabinDeck"]
df = pd.get_dummies(df, columns=to_be_encoded)


# In[42]:


df.head()


# In[43]:


df.columns


# Now, we will split the dataset again before imputing numerical missing values. The splitting is important since while encoding is not influenced by the distribution of datapoints across the two datasets, imputation of missing values is. We want the two datasets to maintain the difference in their distributions and not be influenced by each other.

# In[44]:


train_df, test_df = split_train_test(df, train_idx, test_idx, transported=transported)


# In[45]:


train_df.head()


# In[46]:


test_df.head()


# 
# # Numerical Features Missing Values
# 
# All missing values in numerical features are imputed using KNN.
# 
# For now, we will drop `GroupId` and `CabinNum`. The goal is to train a total of 7 baselines:
# 1. Without `CabinNum` and `GroupId`
# 2. Only with `CabinNum` One-Hot Encoded.
# 3. Only with `CabinNum` Label Encoded
# 4. Only with `GroupId` One-Hot Encoded.
# 5. Only with `GroupId` Label Encoded.
# 6. With both `CabinNum` and `GroupId` One-Hot Encoded.
# 7. With both `CabinNum` and `GroupId` Label Encoded.
# 
# This will tell us whether these features actually contribute to the model or not and the best encoding for them, allowing us to keep the best features.

# In[47]:


def impute_missing_using_knn(df, numeric_cols, has_labels=False):
    x = df
    
    # We should not use the labels for imputing
    # So, if there are labels, drop them
    if has_labels is True:
        transported = df["Transported"]
        x = df.drop("Transported", axis=1)
        
    # Standardize the numerical columns
    scaler = preprocessing.StandardScaler()
    x[numeric_cols] = scaler.fit_transform(x[numeric_cols])
    
    # Impute missing values
    imputer = impute.KNNImputer(n_neighbors=5, weights="distance")
    # Note: x is now a NumPy array
    x = imputer.fit_transform(x)
    
    # Add the labels again if they were dropped
    if has_labels is True:
        x = np.hstack((x, transported.values.reshape(-1, 1)))
        
    return pd.DataFrame(x, columns=df.columns, index=df.index)


# In[48]:


# Store CabinNum and GroupId for later use before dropping
train_cabin_num = train_df["CabinNum"]
train_group_id = train_df["GroupId"]

test_cabin_num = test_df["CabinNum"]
test_group_id = test_df["GroupId"]


# In[49]:


to_drop = ["GroupId", "CabinNum"]
numeric_cols = ["Age", "TotalExpense"] + expenditure_columns

train_df = impute_missing_using_knn(train_df.drop(to_drop, axis=1), numeric_cols, has_labels=True)
test_df = impute_missing_using_knn(test_df.drop(to_drop, axis=1), numeric_cols)


# In[50]:


train_df.head()


# In[51]:


test_df.head()


# In[52]:


train_df.isna().any()


# In[53]:


test_df.head()


# # Create Folds
# 
# We will create 5 folds. This leads to a 6954 samples in the training set and 1739 samples in the validation set for each fold.

# In[54]:


# Reset index so that the output of kf.split() can be used directly
train_df = train_df.reset_index()

# Add column for fold index and initialize kf
train_df["kfold"] = -1
kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)

for idx, (_, val_idx) in enumerate(kf.split(train_df)):
    train_df.loc[val_idx, "kfold"] = idx

# Restore the index
train_df = train_df.set_index("PassengerId")
train_df.head()


# In[55]:


len(train_df[train_df["kfold"] != 0])


# Let's save these two files so that we can reuse them in other notebooks.

# In[56]:


train_df.to_csv("train_prepared.csv", index=False)
test_df.to_csv("test_prepared.csv")


# # Logistic Regression

# We first define the training loop.

# In[57]:


def train(df):
    # Add prediction column
    df["preds"] = pd.NA
    
    # Need to drop target, predictions and kfold
    # In each training iteration
    drop = ["Transported", "preds", "kfold"]
    
    for fold in range(5):
        train = df[df["kfold"] != fold]
        
        # Get training features and labels
        y_train = train["Transported"].values
        X_train = train.drop(drop, axis=1).values
        
        val = df[df["kfold"] == fold]
        
        # Get validation features and labels
        y_val = val["Transported"].values
        X_val = val.drop(drop, axis=1).values
        
        # The default max_iter is too small for this dataset
        model = linear_model.LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Predict on the validation set
        preds = model.predict(X_val)
        df.loc[val.index, "preds"] = preds
        
        # Calculate accuracy
        acc = metrics.accuracy_score(y_val, preds)
        print(f"Fold {fold + 1} - Accuracy = {acc: .4f}")
    
    # Convert target, prediction to integer
    df[drop] = df[drop].astype(int)
    
    # Calculate overall accuracy
    acc = metrics.accuracy_score(df["Transported"].values, df["preds"].values)
    print(f"Overall accuracy = {acc: .4f}")
    
    return df


# ## 1. Without `CabinNum` and `GroupId`

# In[58]:


# Send copy so that original dataset remains unchanged
train_df_exp1 = train(train_df.copy())


# Around 61% of the mistakes are for passengers traveling alone.

# In[59]:


wrong = train_df_exp1[train_df_exp1["Transported"] != train_df_exp1["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 2. Only With `CabinNum` One-Hot Encoded
# 
# Here we one-hot encode `CabinNum`. For this, we need to combine the train and test sets again and use `pd.get_dummies()`. We define a helper function which can be reused for `GroupId`. This function takes the training and test dataframes, the label that should be assigned to the column being encoded, and the training and test values for the column.

# In[60]:


def add_onehot_column(train_df, test_df, column_label, train_values, test_values):
    folds = train_df["kfold"]
    
    # Create copies so that original dataframes remain unchanged
    train_df_with_col = train_df.drop("kfold", axis=1)
    test_df_with_col = test_df.copy()

    # Add the column to the dataframes
    train_df_with_col[column_label] = train_values
    test_df_with_col[column_label] = test_values

    # Merge, one-hot encode and then split
    df, train_idx, test_idx, transported = concat_train_test(
        train=train_df_with_col,
        test=test_df_with_col,
        has_labels=True
    )
    
    df = pd.get_dummies(df, columns=[column_label])
    
    train_df_with_col, test_df_with_col = split_train_test(
        df=df,
        train_index=train_idx,
        test_index=test_idx,
        transported=transported
    )
    
    # Add the folds column
    train_df_with_col["kfold"] = folds
    
    return train_df_with_col, test_df_with_col


# In[61]:


# Get a new dataframe with one-hot encoded `CabinNum`
train_df_cabinnum_oh, test_df_cabinnum_oh = add_onehot_column(
    train_df=train_df,
    test_df=test_df,
    column_label="CabinNum",
    train_values=train_cabin_num,
    test_values=test_cabin_num,
)


# In[62]:


train_df_cabinnum_oh.head()


# In[63]:


test_df_cabinnum_oh.head()


# In[64]:


train_df_exp2 = train(train_df_cabinnum_oh.copy())


# Around 60% of the mistakes are in case of passengers traveling alone.

# In[65]:


wrong = train_df_exp2[train_df_exp2["Transported"] != train_df_exp2["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 3. Only With `CabinNum` Label Encoded
# 
# For this, we need to merge the two datasets, label encode `CabinNum` and then split them. Let's create a function which can be reused for `GroupId`. This function is similar to the one used for one-hot encoding and takes the same parameters.

# In[66]:


def add_labelencoded_column(train_df, test_df, column_label, train_values, test_values):
    folds = train_df["kfold"]
    
    # Create copies so that original dataframes remain unchanged
    train_df_with_col = train_df.drop("kfold", axis=1)
    test_df_with_col = test_df.copy()

    # Add the column to the dataframes
    train_df_with_col[column_label] = train_values
    test_df_with_col[column_label] = test_values

    # Merge, label encode and then split
    df, train_idx, test_idx, transported = concat_train_test(
        train=train_df_with_col,
        test=test_df_with_col,
        has_labels=True
    )
    
    # To label encode, we first get the labels
    # Then, we create a dictionary like {'a': 0, 'b': 1, ...}
    # Finally, we use the .map() method to change the values
    levels = df[column_label].value_counts().index
    mapping = {level: idx for idx, level in enumerate(levels)}
    df[column_label] = df[column_label].map(mapping)
    
    train_df_with_col, test_df_with_col = split_train_test(
        df=df,
        train_index=train_idx,
        test_index=test_idx,
        transported=transported)
    
    # Add the folds column
    train_df_with_col["kfold"] = folds
    
    return train_df_with_col, test_df_with_col


# In[67]:


# Add CabinNum as a label encoded column
train_df_cabinnum_le, test_df_cabinnum_le = add_labelencoded_column(
    train_df=train_df,
    test_df=test_df,
    column_label="CabinNum",
    train_values=train_cabin_num,
    test_values=test_cabin_num
)


# In[68]:


train_df_cabinnum_le.head()


# In[69]:


test_df_cabinnum_le.head()


# In[70]:


train_df_exp3 = train(train_df_cabinnum_le.copy())


# Around 60.2% of the mistakes are in case of passengers who were traveling alone.

# In[71]:


wrong = train_df_exp3[train_df_exp3["Transported"] != train_df_exp3["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 4. Only with `GroupId` One-Hot Encoded
# 
# The process will be similar to `CabinNum`.

# In[72]:


# Get a new dataframe with one-hot encoded GroupId
train_df_groupid_oh, test_df_groupid_oh = add_onehot_column(
    train_df=train_df.copy(),
    test_df=test_df.copy(),
    column_label="GroupId",
    train_values=train_group_id,
    test_values=test_group_id,
)


# In[73]:


train_df_groupid_oh.head()


# In[74]:


test_df_groupid_oh.head()


# In[75]:


train_df_exp4 = train(train_df_groupid_oh.copy())


# Around 60.7% of the mistakes are in case of passengers who were traveling alone.

# In[76]:


wrong = train_df_exp4[train_df_exp4["Transported"] != train_df_exp4["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 5. Only With `GroupId` Label Encoded
# 
# The process will be similar to `CabinNum`.

# In[77]:


# Add GroupId as a label encoded column
train_df_groupid_le, test_df_groupid_le = add_labelencoded_column(
    train_df=train_df,
    test_df=test_df,
    column_label="GroupId",
    train_values=train_group_id,
    test_values=test_group_id
)


# In[78]:


train_df_groupid_le.head()


# In[79]:


test_df_groupid_le.head()


# In[80]:


train_df_exp5 = train(train_df_groupid_le.copy())


# Around 60.2% of the mistakes are in case of passengers who were traveling alone.

# In[81]:


wrong = train_df_exp5[train_df_exp5["Transported"] != train_df_exp5["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 6. With Both `CabinNum` and `GroupId` One-Hot Encoded

# In[82]:


# Get new dataframe with one-hot encoded CabinNum
train_df_both_oh, test_df_both_oh = add_onehot_column(
    train_df=train_df,
    test_df=test_df,
    column_label="CabinNum",
    train_values=train_cabin_num,
    test_values=test_cabin_num,
)

# Get final dataframe with one-hot encoded GroupId
train_df_both_oh, test_df_both_oh = add_onehot_column(
    train_df=train_df_both_oh,
    test_df=test_df_both_oh,
    column_label="GroupId",
    train_values=train_group_id,
    test_values=test_group_id,
)


# In[83]:


train_df_both_oh.head()


# In[84]:


test_df_both_oh.head()


# In[85]:


train_df_exp6 = train(train_df_both_oh.copy())


# Around 60% of the mistakes are in case of passengers who were traveling alone.

# In[86]:


wrong = train_df_exp6[train_df_exp6["Transported"] != train_df_exp6["preds"]]
wrong["GroupSize_1"].value_counts()


# ## 7. With Both `CabinNum` and `GroupId` Label Encoded

# In[87]:


# Add CabinNum as a label encoded column
train_df_both_le, test_df_both_le = add_labelencoded_column(
    train_df=train_df,
    test_df=test_df,
    column_label="CabinNum",
    train_values=train_cabin_num,
    test_values=test_cabin_num
)

# Add GroupId as a label encoded column
train_df_both_le, test_df_both_le = add_labelencoded_column(
    train_df=train_df_both_le,
    test_df=test_df_both_le,
    column_label="GroupId",
    train_values=train_group_id,
    test_values=test_group_id
)


# In[88]:


train_df_both_le.head()


# In[89]:


test_df_both_le.head()


# In[90]:


train_df_exp7 = train(train_df_both_le.copy())


# Around 60% of the mistakes are in case of passengers who were traveling alone.

# In[91]:


wrong = train_df_exp7[train_df_exp7["Transported"] != train_df_exp7["preds"]]
wrong["GroupSize_1"].value_counts()


# # Additional Files
# 
# The following three experiments yield close results:
# 
# - With `CabinNum` label encoded
# - With `GroupId` label encoded
# - With both `CabinNum` and `GroupId` label encoded.
# 
# Thus, we will also save the datasets used for these experiments so that they can be used in future notebooks.

# In[92]:


# `CabinNum` label encoded
train_df_cabinnum_le.to_csv("train_prepared_cabinnum_le.csv", index=False)
test_df_cabinnum_le.to_csv("test_prepared_cabinnum_le.csv")

# `GroupId` label encoded
train_df_groupid_le.to_csv("train_prepared_groupid_le.csv", index=False)
test_df_groupid_le.to_csv("test_prepared_groupid_le.csv")

# Both label encoded
train_df_both_le.to_csv("train_prepared_both_le.csv", index=False)
test_df_both_le.to_csv("test_prepared_both_le.csv")


# # Conclusion
# 
# The experiment suggests that we are better off not using `CabinNum` in the model and using `GroupId` with label encoding. Across the board, label encoding has beaten one-hot encoding. This is most likely due to the dense nature of these features.
# 
# We also find out that most of the mistakes are in case of passengers who were traveling alone. In each model, passengers traveling alone make up ~60% of the mistakes. The most likely culprit behind this is how the missing values are computed but this requires further investigation.
# 
# 
# |                   **Experiment**                   | **Fold 1** | **Fold 2** | **Fold 3** | **Fold 4** | **Fold 5** | **Overall** |
# |:--------------------------------------------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:-----------:|
# |          Without `CabinNum` and `GroupId`          |   0.7867   |   0.7861   |   0.7936   |   0.7975   |   0.7969   |    0.7921   |
# |           With only `CabinNum` (One-Hot)           |   0.7832   |   0.7872   |   0.7941   |   0.7992   |   0.7831   |    0.7894   |
# |        With only `CabinNum` (Label Encoded)        |   0.7849   | **0.7913** |   0.7953   |   0.7952   | **0.8021** |    0.7937   |
# |            With only `GroupId` (One-Hot)           |   0.7821   |   0.7878   |   0.7959   |   0.7957   |   0.7940   |    0.7911   |
# |         With only `GroupId` (Label Encoded)        | **0.7913** |   0.7890   | **0.7999** |   0.7940   |   0.7975   |  **0.7943** |
# |    With both `CabinNum` and `GroupId` (One-Hot)    |   0.7832   |   0.7890   |   0.7878   | **0.8003** |   0.7854   |    0.7891   |
# | With both `CabinNum` and `GroupId` (Label Encoded) |   0.7815   |   0.7901   |   0.7993   |   0.7900   |   0.7992   |    0.7920   |
# 
# 
# Thank you for reading!
