#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello!
# 
# In this kernel you will find my approach to this classification problem which will include EDA, data cleaning, feature engineering, model selection and parameters tuning. So without further ado, let's get started!

# # Table of contents:
# 
# 1. Meeting our data
# 
# 2. Visualization and data analysis
#     
#     2.1 Target variable and numerical data
#     
#     2.2 Categorical data
#     
# 3. Data cleaning
# 
#     3.1 Dealing with null values
#     
#     3.2 Dealing with outliers
#     
# 4. Back to visualization 
# 
# 5. Feature engineering and one-hot encoding
# 
# 6. Creating and evaluating a model
# 
#     6.1 Evaluating models and making a choice
#     
#     6.2 Parameter tuning and submitting results

# # 1. Meeting our data

# In[1]:


import numpy as np
import pandas as pd

train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col = 'PassengerId')
test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col = 'PassengerId')


# In[2]:


train.shape


# In[3]:


test.shape


# In[4]:


train.tail()


# In[5]:


test.head()


# Getting a glimpse of null values using .info().

# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


train.dtypes.unique()


# In[9]:


test.dtypes.unique()


# In[10]:


train.select_dtypes(include = ['object']).describe()


# In[11]:


train.drop('Survived', axis = 1).select_dtypes(exclude = ['object']).describe()


# In[12]:


target = train.Survived.copy()
target


# In[13]:


target.isna().any()


# In[14]:


target.loc[target == 1].size / target.size


# In[15]:


target.describe()


# In[16]:


train.drop('Survived', axis = 1).columns.equals(test.columns)


# # 2. Visualization and data analysis

# Setting up a seaborn library for pretty visualizations. :)

# In[17]:


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.set_style('whitegrid')


# # 2.1 Target variable and numerical data

# In[18]:


plt.figure(figsize = (16, 6))
sns.countplot(x = train.Survived, palette = 'Blues_r')


# In[19]:


def plot_grid(data, fig_size, grid_size, plot_type, target = ''):
    """
    Custom function for plotting grid of plots.
    It takes: DataFrame of data, size of a grid, type of plots, string name of target variable;
    And it outputs: grid of plots.
    """
    fig = plt.figure(figsize = fig_size)
    if plot_type == 'histplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.histplot(data[column_name], kde = True, color = 'royalblue', stat = 'count')
    if plot_type == 'boxplot':
        for i, column_name in enumerate(data.select_dtypes(exclude = 'object').columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.boxplot(x = data[column_name], color = 'royalblue')
    if plot_type == 'countplot':
        target = data[target]
        for i, column_name in enumerate(data.drop(target.name, axis = 1).columns):
            fig.add_subplot(grid_size[0], grid_size[1], i + 1)
            plot = sns.countplot(x = data[column_name], hue = target, palette = 'Blues_r')
            plot.legend(loc = 'upper right', title = target.name)
    plt.tight_layout()


# In[20]:


plot_grid(train.drop('Survived', axis = 1), (16, 6), (2,3), 'histplot')


# In[21]:


pd.pivot_table(train, index = 'Survived', values = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass'], aggfunc = 'mean')


# In[22]:


plot_grid(train.select_dtypes(exclude = 'object').drop(['Fare', 'Age'], axis = 1), (16, 6), (1, 3), 'countplot', 'Survived')


# In[23]:


pd.pivot_table(train, index = 'Survived', values = ['SibSp', 'Parch', 'Pclass'], aggfunc = (lambda x: x.mode()[0]))


# In[24]:


print(f"{pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train, index = 'Survived', columns = 'SibSp', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train, index = 'Survived', columns = 'Parch', values = 'Name', aggfunc ='count')}")


# In[25]:


plt.figure(figsize = (16, 6))
sns.heatmap(train.corr(), 
            annot = True,
            fmt = '.2f',
            square = True,
            cmap = "Blues_r", 
            mask = np.triu(train.corr()))


# In[26]:


plot_grid(train.drop('Survived', axis = 1), (16, 6), (2,3), 'boxplot')


# In[27]:


print(f'Percent of values < 1 in Parch feature: {(train.Parch[train.Parch < 1].size / train.shape[0]) * 100}')


# By analyzing everything above we now have some ideas about what's going on. People who had better chances to survive the disaster tend to be:
# 
# - Younger;
# 
# - In first or second class;
# 
# - Spent more money on a ticket;
# 
# - Had a few family members on board.
# 
# Also, we can see that Parch feature is useless by itself as more than 76 percent of it are zeroes.

# # 2.2 Categorical data

# In[28]:


plot_grid(pd.concat([train.select_dtypes(include = 'object').drop(['Name', 'Ticket', 'Cabin'], axis = 1), target], axis = 1), (16, 6), (2,1), 'countplot', 'Survived')


# In[29]:


print(f"{pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Name', aggfunc ='count')} \n\n" +
      f"{pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Name', aggfunc ='count')}")


# In[30]:


train.select_dtypes(include = 'object').nunique().sort_values(ascending = False)


# Now we have an even bigger picture:
# 
# - Women had better chances of surviving;
# 
# - Embarked feature probably has a relationship with Pclass;
# 
# - We have to simplify or remove Name, Cabin and Ticket features because of high cardinality.

# # 3. Data cleaning

# # 3.1 Dealing with null values

# In[31]:


train_test = pd.concat([train.drop('Survived', axis = 1), test], keys = ['train', 'test'], axis = 0)
missing_values = pd.concat([train_test.isna().sum(),
                            (train_test.isna().sum() / train_test.shape[0]) * 100], axis = 1, 
                           keys = ['Values missing', 'Percent of missing'])
missing_values.loc[missing_values['Percent of missing'] > 0].sort_values(ascending = False, by = 'Percent of missing').style.background_gradient('Blues')


# By analyzing each of this features closely I decided to fill NaN values like this:
# 
# - In Cabin with None;
# 
# - In Age with a median because data has some outliers;
# 
# - In Fare with 0; (cause it looks like there's a group of people that didn't pay for their tickets probably staff members)
# 
# - Embarked with train[(train.Fare < 85) & (train.Fare > 75) & (train.Cabin.str.contains('B'))].Emberked.mode().

# In[32]:


train_cleaning = train.drop('Survived', axis = 1).copy()
test_cleaning = test.copy()

train_cleaning['Cabin'].fillna('None', inplace = True)
test_cleaning['Cabin'].fillna('None', inplace = True)

train_cleaning['Age'].fillna(train_cleaning['Age'].median(), inplace = True)
test_cleaning['Age'].fillna(train_cleaning['Age'].median(), inplace = True)

train_cleaning['Embarked'].fillna(train_cleaning[(train_cleaning.Fare < 85) & (train_cleaning.Fare > 75) & 
                                                 (train_cleaning.Cabin.str.contains('B'))].Embarked.mode()[0], 
                                  inplace = True)

test_cleaning['Fare'].fillna(0, inplace = True)


# In[33]:


train_cleaning.isnull().sum().max()


# In[34]:


test_cleaning.isnull().sum().max()


# In[35]:


fig, axs = plt.subplots(2, 2, figsize = (16, 6))

sns.histplot(train.Age, kde = True, color = 'black', ax = axs[0, 0])
axs[0, 0].set_title('Ages in train with NaN values')
sns.histplot(train_cleaning.Age, kde = True, color = 'royalblue', ax = axs[0, 1])
axs[0, 1].set_title('Ages in train without NaN values')

sns.histplot(test.Age, kde = True, color = 'black', ax = axs[1, 0])
axs[1, 0].set_title('Ages in test with NaN values')
sns.histplot(test_cleaning.Age, kde = True, color = 'royalblue', ax = axs[1, 1])
axs[1, 1].set_title('Ages in test without NaN values')

fig.tight_layout()


# # 3.2 Dealing with outliers

# After looking closely into data and testing out different options I decided to remove only one outlier (observation with Age == 80), but I left my function in the cell below, so you can experiment with it if you would like. Because this part with outliers is kinda subjective.

# In[36]:


# def get_outliers(X_y, cols):
#     """
#     Custom function for dealing with outliers.
#     It takes: DataFrame of data, list of columns;
#     And it returns: list of unique indexes of outliers.(Also it outputs all outliers with indexes for each column)
#     (value is considered an outlier if absolute value of its z-score is > 3)
#     """
#     outliers_index = []
#     for col in cols:
#         right_outliers = X_y[col][(X_y[col] - X_y[col].mean()) / X_y[col].std() > 3]
#         left_outliers = X_y[col][(X_y[col] - X_y[col].mean()) / X_y[col].std() < -3]
#         all_outliers = right_outliers.append(left_outliers)
#         outliers_index += (list(all_outliers.index))
#         print('{} right outliers:\n{} \n {} left outliers:\n{} \n {} has TOTAL {} rows of outliers\n'.format(col, right_outliers, col, left_outliers, col, all_outliers.count()))
#     outliers_index = list(set(outliers_index)) # Removing duplicates
#     print('There are {} unique rows with outliers in dataset'.format(len(outliers_index)))
#     return outliers_index


# In[37]:


# cols = ['Age']
X_y = pd.concat([train_cleaning, target], axis = 1)
# outliers_index = get_outliers(X_y, cols)

X_y.drop(X_y.loc[X_y.Age == 80].index, axis = 0, inplace = True)
# X_y.drop(list(set(outliers_index + 
#                   list(X_y.loc[(X_y.Fare > 500) | (X_y.SibSp == 8)].index))), axis = 0, inplace = True)


# In[38]:


train_cleaning = X_y.drop('Survived', axis = 1).copy()
target_cleaned = X_y.Survived.copy()

# target_cleaned = target


# In[39]:


train_cleaning.shape[0] == target_cleaned.shape[0]


# # 4. Back to visualization

# Answering some questions I came up with during previous steps.

# In[40]:


plt.figure(figsize = (16, 6))
plot = sns.countplot(x = train_cleaning.Cabin.loc[train_cleaning.Cabin != 'None'].str.split().apply(lambda x: len(x)), 
                     hue = (train_cleaning.SibSp + train_cleaning.Parch + 1))
plot.set_title('Relationship between number of places in cabin and family size')
plot.set_xlabel('Number of places in cabin')
plot.set_ylabel('Count')
plot.legend(loc = 'upper right', title = 'Family size')


# The plot above shows relationship between number of places in cabin each person has and the size of their family. (excluding observations with Cabin = 'None').

# In[41]:


plt.figure(figsize = (16, 6))
plot = sns.countplot(x = train_cleaning.Cabin.loc[train_cleaning.Cabin != 'None'].str.split().apply(lambda x: len(x)), 
                     hue = (train_cleaning.Pclass), palette = 'Blues_r')
plot.set_title('Relationship between number of places in cabin and ticket class')
plot.set_xlabel('Number of places in cabin')
plot.set_ylabel('Count')
plot.legend(loc = 'upper right', title = 'Ticket Class')


# In[42]:


plt.figure(figsize = (16, 6))
plot = sns.countplot(x = train_cleaning.loc[train_cleaning.Cabin == 'None'].Cabin, hue = train_cleaning.Pclass, palette = 'Blues_r')
plot.set_title('Relationship between Pclass and absence of Cabin')


# Relationship between Pclass and absence of Cabin. Soooo None value doesn't really tell us a story because I doubt that people in first or second (or even third) class didn't have a cabin. We will take care of it in Feature Engineering part.

# In[43]:


fig, axs = plt.subplots(1, 2, figsize = (16, 6))
sns.histplot(hue = target_cleaned, x = train_cleaning.Age.loc[train_cleaning.Sex == 'male'], palette = {0 : 'black', 1 : 'royalblue'}, ax = axs[0])
axs[0].set_title('Male Age distribution')
sns.histplot(hue = target_cleaned, x = train_cleaning.Age.loc[train_cleaning.Sex == 'female'], palette = {0 : 'black', 1 : 'royalblue'}, ax = axs[1])
axs[1].set_title('Female Age distribution')
plt.tight_layout()


# In[44]:


plt.figure(figsize = (16,6))
plot = sns.kdeplot(hue = target_cleaned, x = train_cleaning.Age, palette = {0 : 'black', 1 : 'royalblue'}, fill = True)
plot.set_title('Age distribution')


# In[45]:


plt.figure(figsize = (16, 6))
plot = sns.countplot(x = train_cleaning.Embarked, hue = train_cleaning.Pclass, palette = 'Blues_r')
plot.set_title('Relationship between Embarked and Pclass')


# # 5. Feature engineering and one-hot encoding

# In[46]:


train_test_cleaning = pd.concat([train_cleaning, test_cleaning], keys = ['train', 'test'], axis = 0)
train_test_cleaning


# As with outliers, I decided to leave everything I tried with feature engineering here, so you can try different things yourself if you want to. In my opinion this is the most interesting and rewarding (in terms of model accuracy) part, and I had a lot of fun trying out different features. :)

# In[47]:


train_test_cleaning['CabinLetter'] = train_test_cleaning.Cabin.str.split().apply(lambda x: x[-1][0].strip().lower() if x[0] != 'None' else np.nan)


# Imputing NaN values in CabinLetter with mode according to its Pclass.

# In[48]:


train_test_cleaning.xs('train').groupby('Pclass').CabinLetter.apply(lambda x: x.value_counts().index[0])


# In[49]:


train_cleaning_new = train_test_cleaning.xs('train').copy()
test_cleaning_new = train_test_cleaning.xs('test').copy()

train_cleaning_new['CabinLetter'] = train_cleaning_new.groupby('Pclass')['CabinLetter'].apply(lambda x: x.fillna(x.mode()[0]))

for i in train.Pclass.unique():
    test_cleaning_new.loc[test_cleaning_new.Pclass == i, 'CabinLetter'] = test_cleaning_new.loc[test_cleaning_new.Pclass == i, 'CabinLetter'].fillna(train_cleaning_new.loc[train_cleaning_new.Pclass == i].CabinLetter.mode()[0])

train_test_cleaning = pd.concat([train_cleaning_new, test_cleaning_new], keys = ['train', 'test'], axis = 0)


# In[50]:


# train_test_cleaning['CabinLetter'] = train_test_cleaning.groupby('Pclass')['CabinLetter'].apply(lambda x: x.fillna(x.mode()[0]))
# train_test_cleaning['CabinCount'] = train_test_cleaning.Cabin.str.split().apply(lambda x: len(x) if x[0] != 'None' else 0)
# train_test_cleaning['SecondCabinF'] = train_test_cleaning.Cabin.str.split().apply(lambda x: 1 if ((len(x) > 1) & ('F' in x)) else 0)
# train_test_cleaning['CabinNumbersSum'] = train_test_cleaning.Cabin[train_test_cleaning.Cabin != 'None'].str.replace('[A-Z|\s]', ' ', regex = True).apply(lambda x: sum(list(map(int, x.split()))))
# train_test_cleaning['CabinNumbersSum'].fillna(0, inplace = True)

train_test_cleaning['NameStatus'] = train_test_cleaning.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip().lower())#.loc[train_test_cleaning.Fare > 0]
# train_test_cleaning['NameStatus'].fillna('staff', inplace = True) # Fare == 0
train_test_cleaning['NameStatus'] = train_test_cleaning['NameStatus'].apply(lambda x: 'ms' if (x == 'mlle' or x == 'miss' or 
                                                                                               x == 'mme' or x == 'mrs' or x == 'lady' or 
                                                                                               x == 'the countess' or x == 'dona') 
#                                                                             else 'staff' if (x == 'capt')
                                                                            else 'mr' if (x == 'sir' or x == 'don'  or x == 'major' or 
                                                                                          x == 'col' or x == 'rev' or x == 'capt' or x == 'jonkheer' or
                                                                                          x == 'master')
#                                                                             else 'boy' if (x == 'master')
                                                                            else x)
train_test_cleaning.loc[(train_test_cleaning['NameStatus'] == 'dr') & (train_test_cleaning['Sex'] == 'male'), 'NameStatus'] = train_test_cleaning.loc[(train_test_cleaning['NameStatus'] == 'dr') & (train_test_cleaning['Sex'] == 'male'), 'NameStatus'].apply(lambda x: 'mr')
train_test_cleaning.loc[(train_test_cleaning['NameStatus'] == 'dr') & (train_test_cleaning['Sex'] == 'female'), 'NameStatus'] = train_test_cleaning.loc[(train_test_cleaning['NameStatus'] == 'dr') & (train_test_cleaning['Sex'] == 'female'), 'NameStatus'].apply(lambda x: 'ms')

# train_test_cleaning['TicketIsNumeric'] = train_test_cleaning.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
# train_test_cleaning['NumericTicketNumbers'] = train_test_cleaning.Ticket.apply(lambda x: int(x) if x.isnumeric() else 0)
# train_test_cleaning['TicketLen'] = train_test_cleaning.Ticket.apply(lambda x: len(x.replace('.', '').replace('/', '').replace(' ', '')))
train_test_cleaning['TicketNumbers'] = train_test_cleaning.Ticket.apply(lambda x: int(x) if x.isnumeric() else 0 if x == 'LINE' else int(x.split(' ')[-1]))
train_test_cleaning['TicketLetters'] = train_test_cleaning.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.', '').replace('/', '').lower() 
                                                                        if len(x.split(' ')[:-1]) > 0 else x.lower() if x == 'LINE' else 'none')

train_test_cleaning['FamilySize'] = train_test_cleaning.SibSp + train_test_cleaning.Parch + 1
train_test_cleaning['FamilySize'] = train_test_cleaning['FamilySize'].apply(lambda x: 'no family' if (x == 1) 
                                                                            else 'medium' if (x == 2 or x == 3 or x == 4)                                                                    
                                                                            else 'large')

train_test_cleaning['AgeGroup'] = train_test_cleaning['Age'].apply(lambda x: 'infant' if (x < 1) 
                                                                   else 'child' if (x >= 1 and x <= 11)                                                                    
                                                                   else 'teen' if (x >= 12 and x <= 17)
                                                                   else 'adult' if (x >= 18 and x <= 64)
                                                                   else 'adult+')


# In[51]:


train_test_cleaning


# In[52]:


train_cleaning_target_cleaned = pd.concat([train_test_cleaning.xs('train'), target_cleaned], axis = 1)
train_cleaning_target_cleaned


# In[53]:


print(f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'CabinLetter', values = 'Name', aggfunc ='count')} \n\n" +
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'CabinCount', values = 'Name', aggfunc ='count')} \n\n" +
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'SecondCabinF', values = 'Name', aggfunc ='count')} \n\n" +
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', values = 'CabinNumbersSum', aggfunc ='mean')} \n\n\n\n" +
      
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', values = 'TicketLen', aggfunc ='mean')} \n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', values = 'TicketNumbers', aggfunc = (lambda x: x.mode()[0]))} \n\n" +
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', values = 'TicketLetters', aggfunc ='mean')} \n\n" +
#       f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketIsNumeric', values = 'Name', aggfunc ='count')} \n\n\n\n" +
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'AgeGroup', values = 'Name', aggfunc ='count')} \n\n" +
      
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'NameStatus', values = 'Name', aggfunc ='count')} \n\n" +
      
      f"{pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'FamilySize', values = 'Name', aggfunc ='count')}")


# In[54]:


plt.figure(figsize = (20, 6))
sns.countplot(x = train_cleaning_target_cleaned.NameStatus, hue = train_cleaning_target_cleaned.Survived, palette = 'Blues_r')


# In[55]:


plt.figure(figsize = (16, 6))
sns.countplot(x = train_cleaning_target_cleaned.TicketLetters.loc[train_cleaning_target_cleaned.TicketLetters != 'none'].sort_values(), hue = train_cleaning_target_cleaned.Survived, 
              palette = 'Blues_r')


# In[56]:


pd.pivot_table(train_cleaning_target_cleaned, index = 'Survived', columns = 'TicketLetters', values = 'Name', aggfunc = 'count')


# In[57]:


train_cleaning_target_cleaned.select_dtypes(include = 'object').nunique().sort_values(ascending = False)


# In[58]:


plot_grid(train_cleaning_target_cleaned.drop('Survived', axis = 1), (16, 6), (2, 3), 'histplot')


# In[59]:


plot_grid(train_cleaning_target_cleaned.drop(['Name', 'Ticket', 'Cabin', 'Age', 'Fare', 'TicketNumbers', 'TicketLetters'],
                                             axis = 1), (16, 6), (3, 3), 'countplot', 'Survived')


# In[60]:


plt.figure(figsize = (16,6))
sns.heatmap(train_cleaning_target_cleaned.corr(),
            annot = True,
            fmt = '.2f',
            square = True,
            cmap = "Blues_r",
            mask = np.triu(train_cleaning_target_cleaned.corr()))


# Now we drop features with high cardinality and a few extra ones (reasons for dropping every single one of them are provided in the cell below as comments).

# In[61]:


to_drop = ['Name',# High cardinality
           'Ticket',# High cardinality
           'Cabin',# High cardinality
           'Sex',# NameStatus tells us the same story
           'Age',# I decided to use AgeGroup for the sake of simplicity instead
           'Parch']# Useless on its own because it consists almost entirely of zeroes, and besides we have a better feature FamilySize

train_test_cleaned = train_test_cleaning.drop(to_drop, axis = 1).copy()
train_test_cleaned


# In[62]:


train_test = pd.get_dummies(train_test_cleaned)
train_test


# In[63]:


X_train_full, X_test = train_test.xs('train'), train_test.xs('test')
X_train_full


# In[64]:


y_train_full = target_cleaned
y_train_full


# # 6. Creating and evaluating a model

# In[65]:


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV 


# # 6.1 Evaluating models and making a choice

# In[66]:


def test_estimators(X, y, estimators, labels, cv):
    ''' 
    A function for testing multiple estimators.
    It takes: full train data and target, list of estimators, 
              list of labels or names of estimators,
              cross validation splitting strategy;
    And it returns: a DataFrame of table with results of tests
    '''
    result_table = pd.DataFrame()

    row_index = 0
    for est, label in zip(estimators, labels):

        est_name = label
        result_table.loc[row_index, 'Model Name'] = est_name

        cv_results = cross_validate(est,
                                    X,
                                    y,
                                    cv = cv,
                                    n_jobs = -1)

        result_table.loc[row_index, 'Test accuracy'] = cv_results['test_score'].mean()
        result_table.loc[row_index, 'Test Std'] = cv_results['test_score'].std()
        result_table.loc[row_index, 'Fit Time'] = cv_results['fit_time'].mean()

        row_index += 1

    result_table.sort_values(by=['Test accuracy'], ascending = False, inplace = True)

    return result_table


# In[67]:


lr = LogisticRegression()
dt = DecisionTreeClassifier(random_state = 1)
rf = RandomForestClassifier(random_state = 1)
svc = make_pipeline(StandardScaler(), SVC(probability = True))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier())


estimators = [lr,
              dt,
              rf,
              svc, 
              knn,]

labels = ['Log Regression',
          'Decision Tree',
          'Random Forest',
          'SVC', 
          'KNN',]

results = test_estimators(X_train_full, y_train_full, estimators, labels, cv = 10)
results.style.background_gradient(cmap = 'Blues')


# Now if we would take our Random Forest model in this state (without tuning any parameters) fit it, get predictions and submit results, then we will score only **0.75837** on a test set. And the reason for this drop in accuracy is overfitting. For more information about it, I recommend checking out [this amazing kernel](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic#Overfitting-and-underfitting-the-Titanic). In short, Titanic train data set gets overfitted even by the humble Desicion Tree, so using something more complex than Random Forest would probably be overkill for this task.

# # 6.2 Parameter tuning and submitting results

# In order to deal with overfitting we will tune parameters using GridSearchCV. I shortened lists of parameters for the sake of saving some time, because finding good parameters can be time consuming, but I will explain the strategy I used:
# 
# - for 'max_depth' I tried using 30-50% of number of features in train data and some values a bit outside this interval;
# - for 'max_features' I tried small values from 5 to 10 and just like with 'max_depth' ended up a bit outside from this interval;
# - for 'n_estimators' I started with something like [50, 100, 300, 500] and then narrowed it down;
# - for 'min_samples_leaf' and 'min_samples_split' I only tried values that you can see in the cell below.

# In[68]:


rf_params = {'random_state': [1],
             'max_depth': [10, 11, 12],
             'max_features': [18],
             'min_samples_leaf': [1, 2],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [113]}

grid = GridSearchCV(rf, 
                    rf_params,
                    cv = 10,   
                    n_jobs = -1)

grid.fit(X_train_full, y_train_full)


# In[69]:


grid.best_params_


# In[70]:


rf = RandomForestClassifier(**grid.best_params_)

cv_results = cross_val_score(rf, X_train_full, y_train_full, cv = 10)

print(f'All results: {cv_results} \n\n' +
      f'Mean: {cv_results.mean()} \n\n' +
      f'Std: {cv_results.std()}')


# In[71]:


rf.fit(X_train_full, y_train_full)
predictions = rf.predict(X_test)


# In[72]:


submission = pd.DataFrame({'PassengerId': X_test.index,
                           'Survived': predictions})
submission.to_csv('submission.csv', index = False)

