#!/usr/bin/env python
# coding: utf-8

# #### In this notebook I tried making new features from Name column. For those that created so far. Only the new feature "IsLastNameDublicated" improved the score.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('../input/tabular-playground-series-apr-2021/test.csv')
submission_df = pd.read_csv('../input/tabular-playground-series-apr-2021/sample_submission.csv')


# In[2]:


df.head(2)


# #### For the simplicity, I will use now only features that don't have missing values. And without the PassengerId.

# In[3]:


names_cols_missing_values = list(df.columns[df.isna().any()])
names_cols_missing_values

df.drop(names_cols_missing_values, axis=1, inplace=True)


# #### Creating new features from the Name column.

# In[4]:


comma_and_space = 2
df["NameLength"] = df.Name.apply(lambda x : len(x) - comma_and_space)

df["NameFirstChar"] = df.Name.apply(lambda x : x[0])

df["IsNameDublicated"] = df.Name.duplicated() 

remove_comma = -1
df["FirstName"] = df.Name.apply(lambda x : x.partition(' ')[0][:remove_comma])

df["LastName"] = df.Name.apply(lambda x : x.split()[-1])

df["LastNameFirstChar"] = df.LastName.apply(lambda x : x[0])

df["IsFirstNameDublicated"] = df.FirstName.duplicated()

df["IsLastNameDublicated"] = df.LastName.duplicated()

df["IsFullNameDublicated"] = df.Name.duplicated()

def check_two_same_sequence_letters(x):
    current_letter = ""
    for i in x:
        if (i == current_letter):
            return True
        current_letter = i
    else:
        return False
df["FirstNameTwoSecLetters"] = df.FirstName.apply(lambda x : check_two_same_sequence_letters(x))
df["LastNameTwoSecLetters"] = df.LastName.apply(lambda x : check_two_same_sequence_letters(x))


# In[5]:


df.head(2)


# In[6]:


def df_preparation(the_df):
    the_df["Survived"] = the_df["Survived"].astype('int8')
    the_df["Pclass"] = the_df["Pclass"].astype('int8')
    the_df["Sex"] = the_df["Sex"].astype('category').cat.codes
    the_df["Name"] = the_df["Name"].astype('category').cat.codes
    the_df["SibSp"] = the_df["SibSp"].astype('int8')
    the_df["Parch"] = the_df["Parch"].astype('int8')
    # new features
    the_df["NameLength"] = the_df["NameLength"].astype('int8')
    the_df["NameFirstChar"] = the_df["NameFirstChar"].astype('category').cat.codes
    the_df["IsNameDublicated"] = the_df["IsNameDublicated"].astype('int8')
    the_df["FirstName"] = the_df["FirstName"].astype('category').cat.codes
    the_df["LastName"] = the_df["LastName"].astype('category').cat.codes
    the_df["LastNameFirstChar"] = the_df["LastNameFirstChar"].astype('category').cat.codes
    the_df["IsFirstNameDublicated"] = the_df["IsFirstNameDublicated"].astype('int8')
    the_df["IsLastNameDublicated"] = the_df["IsLastNameDublicated"].astype('int8')
    the_df["IsFullNameDublicated"] = the_df["IsFullNameDublicated"].astype('int8')
    df["FirstNameTwoSecLetters"] = df["FirstNameTwoSecLetters"].astype('int8')
    df["LastNameTwoSecLetters"] = df["LastNameTwoSecLetters"].astype('int8')
    
    return the_df

df = df_preparation(df).copy()   


# #### Checking the columns.

# In[7]:


df.info()


# #### Making the models

# In[8]:


random_forest_model = RandomForestClassifier(random_state=0)


# In[9]:


def make_model(new_column_name, chosen_columns, the_df):
    X = the_df[chosen_columns]
    y = the_df.Survived
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
    
    random_forest_model.fit(X_train,y_train)
    prediction_score = random_forest_model.score(X_test,y_test)
    print( '{:<25} score: {:>1.5f}'.format(new_column_name, prediction_score) )


# In[10]:


make_model("Without new features", ["Pclass", "Sex", "SibSp", "Parch"], df)
make_model("Name", ["Name", "Sex", "SibSp", "Parch", "Name"], df)
print("\n          New Features:\n")
make_model("NameLength", ["Pclass", "Sex", "SibSp", "Parch", "NameLength"], df)
make_model("NameFirstChar", ["Pclass", "Sex", "SibSp", "Parch", "NameFirstChar"], df)
make_model("IsNameDublicated", ["Pclass", "Sex", "SibSp", "Parch", "IsNameDublicated"], df)
make_model("FirstName", ["Pclass", "Sex", "SibSp", "Parch", "FirstName"], df)
make_model("LastName", ["Pclass", "Sex", "SibSp", "Parch", "LastName"], df)
make_model("LastNameFirstChar", ["Pclass", "Sex", "SibSp", "Parch", "LastNameFirstChar"], df)
make_model("IsFirstNameDublicated", ["Pclass", "Sex", "SibSp", "Parch", "IsFirstNameDublicated"], df)
make_model("IsLastNameDublicated", ["Pclass", "Sex", "SibSp", "Parch", "IsLastNameDublicated"], df)
make_model("IsFullNameDublicated", ["Pclass", "Sex", "SibSp", "Parch", "IsFullNameDublicated"], df)
make_model("FirstNameTwoSecLetters", ["Pclass", "Sex", "SibSp", "Parch", "FirstNameTwoSecLetters"], df)
make_model("LastNameTwoSecLetters", ["Pclass", "Sex", "SibSp", "Parch", "LastNameTwoSecLetters"], df)


# #### So far, only IsLastNameDublicated slightly improved the score. I performed submission and it improved my score.
