#!/usr/bin/env python
# coding: utf-8

# # **Estimation of (Non)-Survival in the Titanic Dataset using Machine Learning Models**

# ### import libraries

# In[1]:


get_ipython().system('pip install lazypredict')


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate, validation_curve, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lazypredict.Supervised import LazyClassifier
from lightgbm import LGBMClassifier
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' %x)


# ### import dataset

# In[3]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
test.head()


# In[4]:


df_survived = train[['Survived']]
train = train.drop(['Survived'], axis=1)
train.shape, test.shape, df_survived.shape
df = pd.concat([train, test], ignore_index=True)
df.shape
df.head()


# # Exploratory Data Analysis

# ### check dataframe

# In[5]:


def check_dataframe(dataframe, head=5):
    print('\n', '#' * 20, 'head'.upper(), 20 * '#')
    display(dataframe.head(head))
    print('\n', '#' * 20, 'tail'.upper(), 20 * '#')
    display(dataframe.tail(head))
    print('\n', '#' * 20, 'shape'.upper(), 20 * '#')
    print(dataframe.shape)
    print('\n', '#' * 20, 'dtypes'.upper(), 20 * '#')
    print(dataframe.dtypes)
    print('\n', '#' * 20, 'columns'.upper(), 20 * '#')
    print(dataframe.columns)
    print('\n', '#' * 20, 'info'.upper(), 20 * '#')
    print(dataframe.info())
    print('\n', '#' * 20, 'any null values'.upper(), 20 * '#')
    print(dataframe.isnull().values.any())
    print('\n', '#' * 20, 'null values'.upper(), 20 * '#')
    print(dataframe.isnull().sum().sort_values(ascending=False))
    print('\n', '#' * 20, 'descriptive statistics'.upper(), 20 * '#')
    display(dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


check_dataframe(df)


# ### grabbing categorical, numerical and cardinal variables

# In[6]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # categorical variables
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'O']
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != 'O']
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == 'O']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # numerical variables
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # reporting section
    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variables: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # keeping the calculated values
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols = [col for col in num_cols if 'PassengerId' not in col]
num_cols
cat_but_car


# ### summary analysis of categorical variables

# In[7]:


def cat_summary(dataframe, col_name, plot=False):
    print('\n', '#' * 10, col_name.upper(), 10 * '#')
    print(pd.DataFrame({
        col_name.upper(): dataframe[col_name].value_counts(),
        'RATIO (%)': round(100 * (dataframe[col_name].value_counts() / len(dataframe)), 2)
    }))

    if plot:
        px.histogram(dataframe, x=dataframe[col_name]).show()

for col in cat_cols:
    cat_summary(df, col, plot=True)


# ### summary analysis of numerical variables

# In[8]:


def num_summary(dataframe, numerical_col, plot=False):
    print('\n', '#' * 10, numerical_col.upper(), '#' * 10)
    print(pd.DataFrame({
        numerical_col.upper(): round(dataframe[numerical_col].describe().T, 2)
    }))

    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.boxplot(y=dataframe[numerical_col])
        plt.ylabel(numerical_col.upper())
        plt.subplot(1, 2, 2)
        sns.histplot(x=dataframe[numerical_col])
        plt.xlabel(numerical_col.upper())
        plt.show(block=True)

        
for col in num_cols:
    num_summary(df, col, plot=True)


# ### determining high correlated variables

# In[9]:


def high_correlated_cols(dataframe, corr_th=0.90, plot=False):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, cmap='RdBu', annot=True, annot_kws={'fontsize': 10})
        plt.show(block=True)

    return drop_list

high_correlated_cols(df, plot=True)   


# # Feature engineering

# In[10]:


# Cabin Bool
df['new_cabin_bool'] = df['Cabin'].notnull().astype('int')
# Name Count
df['new_name_count'] = df['Name'].str.len()
# Name word count
df['new_name_word_count'] = df['Name'].apply(lambda x: len(str(x).split(' ')))
# Name Dr
df['new_name_dr'] = df['Name'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr.')]))
# Name title
df['new_title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# Family size
df['new_family_size'] = df['SibSp'] + df['Parch'] + 1
# Age Pclass
df['new_age_pclass'] = df['Age'] * df['Pclass']
# Is alone?
df.loc[((df['SibSp'] + df['Parch']) > 0), 'new_is_alone'] = 'No'
df.loc[((df['SibSp'] + df['Parch']) == 0), 'new_is_alone'] = 'Yes'
# Age level
df.loc[(df['Age'] < 18), 'new_age_cat'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'new_age_cat'] = 'mature'
df.loc[(df['Age'] >= 56), 'new_age_cat'] = 'senior'
# Sex-Age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'new_sex_cat'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'new_sex_cat'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'new_sex_cat'] = 'seniormale'                                       
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'new_sex_cat'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'new_sex_cat'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'new_sex_cat'] = 'seniorfemale'
# Delete the variables 'Cabin', 'Name', and 'Ticket'
df.drop(['Cabin', 'Name'], axis=1, inplace=True)
# Fill the missing valus of age with its median
df['Age'] = df['Age'].fillna(df.groupby('new_title')['Age'].transform('median'))
# Age Pclass
df['new_age_pclass'] = df['Age'] * df['Pclass']
# Age level
df.loc[(df['Age'] < 18), 'new_age_cat'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'new_age_cat'] = 'mature'
df.loc[(df['Age'] >= 56), 'new_age_cat'] = 'senior'
# Sex-Age
df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'new_sex_cat'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'new_sex_cat'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'new_sex_cat'] = 'seniormale'                                       
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'new_sex_cat'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age'] <= 50)), 'new_sex_cat'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'new_sex_cat'] = 'seniorfemale'
# Fill the missing valus of Embarked with its mode
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0)

# ticket_prefix
def get_prefix(x):
    # return prefix if prefix is not a number else return 'X'
    pre = x.split(' ')[0]
    if not pre.isnumeric():
        return pre
    return 'X'

df['ticket_prefix'] = df['Ticket'].apply(get_prefix)

tp_count = df['ticket_prefix'].value_counts()
# change prefixes with a frequency of less than 5 to 'X'
df['ticket_prefix'] = df['ticket_prefix'].apply(lambda x: x if  tp_count[x] > 4 else 'X')

# remove Ticket
df.drop('Ticket', axis=1, inplace=True)
df


# In[11]:


# Let's show the prequence of the variable 'ticket_prefix'
px.histogram(df, x='ticket_prefix').show()


# # Encoding

# In[12]:


# Label encoding
def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)
df


# In[13]:


# OneHot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 20 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
df.head()


# # Scaling

# In[14]:


cols = [col for col in df.columns if 'PassengerId' not in col]
df[cols] = RobustScaler().fit_transform(df[cols])    
df.head()


# In[15]:


df_train = df.iloc[:891, :]
df_train = pd.concat([df_train, df_survived], axis=1)
df_train


# In[16]:


df_test = df.iloc[891:, :]
df_test


# # Modeling

# ## Selecting the independent and dependent variables

# In[17]:


y = df_train['Survived']
X = df_train.drop(['PassengerId', 'Survived'], axis=1)


# # Lazy Regressor

# In[18]:


X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, y_train, X_test, y_test)
print(models)


# # AdaBoost Classifier

# In[19]:


ada_boost = AdaBoostClassifier().fit(X, y)

# estimating the errors before hyperparameter optimization
adaboost_cv_results = cross_validate(ada_boost, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', adaboost_cv_results['test_accuracy'].mean())
print('F1 Score:', adaboost_cv_results['test_f1'].mean())
print('Roc Auc Score:', adaboost_cv_results['test_roc_auc'].mean())

# hypreparameter optimization
print('--'*50)
print(ada_boost.get_params())
print('--'*50)
adaboost_params = {'learning_rate': [0.01, 0.05, 0.1],
              'n_estimators': [50, 100, 300, 500, 1000]}

# Using GridSearchCv method
adaboost_best_grid = GridSearchCV(ada_boost, adaboost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

adaboost_best_grid.best_params_
print('--'*50)

# final model
adaboost_final = ada_boost.set_params(**adaboost_best_grid.best_params_, random_state=1).fit(X, y)

# estimating the errors after hyperparameter optimization
adaboost_cv_results = cross_validate(adaboost_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', adaboost_cv_results['test_accuracy'].mean())
print('F1 Score:', adaboost_cv_results['test_f1'].mean())
print('Roc Auc Score:', adaboost_cv_results['test_roc_auc'].mean())

# prediction
print('--'*50)
adaboost_final.predict(X)


# In[20]:


# confusion matrix
y_pred = adaboost_final.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(adaboost_final, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# In[21]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(adaboost_final, X)


# # LightGBM Model

# In[22]:


# model establishing
lgbm_model = LGBMClassifier(random_state=1).fit(X, y)

# estimating the errors before hyperparameter optimization
lgbm_cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', lgbm_cv_results['test_accuracy'].mean())
print('F1 Score:', lgbm_cv_results['test_f1'].mean())
print('Roc Auc Score:', lgbm_cv_results['test_roc_auc'].mean())

# hypreparameter optimization
print('--'*50)
print(lgbm_model.get_params())
print('--'*50)
lgbm_params = {'learning_rate': [0.01, 0.05, 0.1],
              'n_estimators': [100, 300, 500, 1000]}

# Using GridSearchCv method
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_
print('--'*50)

# final model
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=1).fit(X, y)

# estimating the errors after hyperparameter optimization
lgbm_cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', lgbm_cv_results['test_accuracy'].mean())
print('F1 Score:', lgbm_cv_results['test_f1'].mean())
print('Roc Auc Score:', lgbm_cv_results['test_roc_auc'].mean())

# prediction
print('--'*50)
lgbm_final.predict(X)


# In[23]:


# confusion matrix
y_pred = lgbm_final.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(lgbm_final, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# In[24]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X)


# # Random Forest Classififer

# In[25]:


# model establishing
rf_model = RandomForestClassifier().fit(X, y)

# estimating the errors before hyperparameter optimization
rf_cv_results = cross_validate(rf_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', rf_cv_results['test_accuracy'].mean())
print('F1 Score:', rf_cv_results['test_f1'].mean())
print('Roc Auc Score:', rf_cv_results['test_roc_auc'].mean())

# hypreparameter optimization
print('--'*50)
print(rf_model.get_params())
print('--'*50)

rf_params = {'max_depth': [5, 8, None],
             'max_features': [3, 5, 7, 'auto'],
            'min_samples_split': [2, 5, 8, 15, 20],
            'n_estimators': [100, 200, 500]}

# Using GridSearchCv method
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
print('--'*50)

# final model
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=1).fit(X, y)

# estimating the errors after hyperparameter optimization
rf_cv_results = cross_validate(rf_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', rf_cv_results['test_accuracy'].mean())
print('F1 Score:', rf_cv_results['test_f1'].mean())
print('Roc Auc Score:', rf_cv_results['test_roc_auc'].mean())

# prediction
print('--'*50)
rf_final.predict(X)


# In[26]:


# confusion matrix
y_pred = rf_final.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(rf_final, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# In[27]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)


# # Decision Tree Classifier

# In[28]:


# model establishing
cart_model = DecisionTreeClassifier(random_state=1)

# estimating the errors before hyperparameter optimization
cart_cv_results = cross_validate(cart_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', cart_cv_results['test_accuracy'].mean())
print('F1 Score:', cart_cv_results['test_f1'].mean())
print('Roc Auc Score:', cart_cv_results['test_roc_auc'].mean())

# hypreparameter optimization
print('--'*50)
print(cart_model.get_params())
print('--'*50)
cart_params = {'max_depth': range(1, 11),
               'min_samples_split': range(2, 20)}

# Using GridSearchCv method
cart_best_grid = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

# final model
cart_final = cart_model.set_params(**cart_best_grid.best_params_, random_state=1).fit(X, y)

# estimating the errors after hyperparameter optimization
cart_cv_results = cross_validate(cart_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', cart_cv_results['test_accuracy'].mean())
print('F1 Score:', cart_cv_results['test_f1'].mean())
print('Roc Auc Score:', cart_cv_results['test_roc_auc'].mean())

# prediction
print('--'*50)
cart_final.predict(X)


# In[29]:


# confusion matrix
y_pred = cart_final.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(cart_final, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# In[30]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(18, 18))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final, X)


# # Logistic Regression

# In[31]:


# model establishing
log_model = LogisticRegression().fit(X, y)

# estimating the errors before hyperparameter optimization
log_cv_results = cross_validate(log_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', log_cv_results['test_accuracy'].mean())
print('F1 Score:', log_cv_results['test_f1'].mean())
print('Roc Auc Score:', log_cv_results['test_roc_auc'].mean())
print('--'*50)

# bias and weights
log_model.intercept_
log_model.coef_
print('--'*50)

# confusion matrix
y_pred = log_model.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(log_model, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# # KNN Model

# In[32]:


knn_model = KNeighborsClassifier().fit(X, y)
print(knn_model.get_params())


# In[33]:


# model establishing
knn_model = KNeighborsClassifier().fit(X, y)

# estimating the errors before hyperparameter optimization
knn_cv_results = cross_validate(knn_model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', knn_cv_results['test_accuracy'].mean())
print('F1 Score:', knn_cv_results['test_f1'].mean())
print('Roc Auc Score:', knn_cv_results['test_roc_auc'].mean())
print('--'*50)

# hypreparameter optimization
print('--'*50)
print(knn_model.get_params())
print('--'*50)
knn_params = {'n_neighbors': range(2, 50)}

# Using GridSearchCv method
knn_best_grid = GridSearchCV(knn_model, knn_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
knn_best_grid.best_params_
print('--'*50)

# final model
knn_final = knn_model.set_params(**knn_best_grid.best_params_).fit(X, y)

# estimating the errors after hyperparameter optimization
knn_cv_results = cross_validate(knn_final, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc'])
print('Accuracy:', knn_cv_results['test_accuracy'].mean())
print('F1 Score:', knn_cv_results['test_f1'].mean())
print('Roc Auc Score:', knn_cv_results['test_roc_auc'].mean())

# prediction
print('--'*50)
knn_final.predict(X)


# In[34]:


# confusion matrix
y_pred = knn_final.predict(X)
print(confusion_matrix(y, y_pred))
print('--'*50)

# classification report
print(classification_report(y, y_pred))
print('--'*50)

# figure of classification report
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
print('--'*50)

# getting roc curve
plot_roc_curve(knn_final, X, y)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1])
plt.show()


# # Let's operate the models on test set

# In[35]:


X_test = df_test.drop(['PassengerId'], axis = 1)
X_test


# In[36]:


df_test['Survived'] = lgbm_final.predict(X_test)
df_test


# In[37]:


submission_df = df_test[['PassengerId', 'Survived']]
submission_df.head()


# In[38]:


submission_df.to_csv('Submission.csv', index=False)


# In[ ]:




