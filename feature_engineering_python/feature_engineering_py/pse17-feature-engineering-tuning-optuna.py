#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This episode's problem revolves around predicting Machine Failures. The dataset is synthetically generated based on this dataset. The data contains 136429 rows and 14 columns, 5 of which are binary features of failure areas and 1 binary feature indicating the presence of machine failure regarding the area.
# 
# Predicting machine failure can be crucial in order to support predictive maintenance, which will support both effective business and operations management. Successfully implementing a predictive maintenance model will lower production costs while also keeping production quality high.
# <img src = "https://eloquentarduino.github.io/wp-content/uploads/2020/07/Binary-classification.png">

# # Importing the dataset

# In[1]:


import numpy as np
import pandas as pd 
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import random
import math
import warnings
warnings.filterwarnings("ignore")

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
gren = Style.BRIGHT + Fore.GREEN
res = Style.RESET_ALL


# In[2]:


train_df = pd.read_csv("/kaggle/input/playground-series-s3e17/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e17/test.csv")
sub_df = pd.read_csv("/kaggle/input/playground-series-s3e17/sample_submission.csv")
original = pd.read_csv("/kaggle/input/machine-failure-predictions/machine failure.csv")


# In[3]:


train_df.shape, test_df.shape, sub_df.shape, original.shape


# The size of the dataset seems large enough.

# In[4]:


train_df['is_generated'] = 0
test_df['is_generated'] = 0
original['is_generated'] = 1


# In[5]:


train_df = pd.concat([train_df, original], axis = 0)
train_df.drop(['id','UDI'], axis = 1, inplace = True)


# In[6]:


train_df.isna().sum()


# In[7]:


print(f'Training duplicated rows {train_df.duplicated().sum()}')
print(f'Test-Data, duplicated rows {test_df.drop(["id"],axis = 1).duplicated().sum()}')
#There are 523 duplicated values in test data
train_df.drop_duplicates(inplace = True)


# In[8]:


df = pd.concat([train_df, test_df], axis = 0) #Creating a single dataframe that will make preprocessing easy
df.shape


# # Exploring the Data
# According to the documentation of the AI4I 2020 Predictive Maintenance Dataset
# 
# 1. **Type**: consisting of a letter L, M, or H for low, medium and high as product quality variants.
# 2. **air temperature [K]**: generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.
# 3. **process temperature [K]**: generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
# 4. **rotational speed [rpm]**: calculated from a power of 2860 W, overlaid with a normally distributed noise.
# 5. **torque [Nm]**: torque values are normally distributed around 40 Nm with a Ïƒ = 10 Nm and no negative values.
# 6. **tool wear [min]**: The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
# 7. **machine failure**: whether the machine has failed in this particular datapoint for any of the following failure modes are true.
# 
# `The machine failure consists of five independent failure modes`
# 
# 8. **tool wear failure (TWF)**: the tool will be replaced of fail at a randomly selected tool wear time between 200 ~ 240 mins.
# 9. **heat dissipation failure (HDF)**: heat dissipation causes a process failure, if the difference between air and process temperature is below 8.6 K and the rotational speed is below 1380 rpm.
# 10. **power failure (PWF)**: the product of torque and rotational speed (in rad/s) equals the power required for the process. If this power is below 3500 W or above 9000 W, the process fails.
# 11. **overstrain failure (OSF)**: if the product of tool wear and torque exceeds 11,000 minNm for the L product variant (12,000 M, 13,000 H), the process fails due to overstrain.
# 12. **random failures (RNF)**: each process has a chance of 0,1 % to fail regardless of its process parameters.
# 
# `If at least one of the above failure modes is true, the process fails and the 'machine failure' label is set to 1.`

# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

palette_color = sns.color_palette('dark')
palette1 = ['dimgrey','crimson']
palette2 = ['crimson', 'dimgrey']
palette3 = ['darkgreen', 'orange']
palette4= ['salmon','mediumseagreen']


# In[10]:


train_df['Machine failure'].value_counts().plot(kind = 'pie', labels = ['No','Yes'], autopct='%.3f%%', colors= palette3)


# ## Product ID

# In[11]:


temp = train_df['Product ID'].value_counts().head(10)
temp_df = train_df[train_df['Product ID'].isin(temp.index)]
# sns.countplot(data = temp_df, x = 'Product ID', hue = 'Machine failure', palette = palette1)
machine_failure_counts = temp_df.groupby('Product ID')['Machine failure'].value_counts().unstack(fill_value=0)
machine_failure_counts.plot(kind='bar', stacked=True, color=palette1)
plt.ylabel("Count")
plt.title("Top 10 Product IDs with Machine failure")
plt.legend()


# ## Type

# In[12]:


fig= plt.figure(figsize=(11,4))
type_count = train_df['Type'].value_counts()
ax1= plt.subplot(1,2,1)
ax1.pie(type_count.values, labels = type_count.index, autopct='%1.1f%%')
ax1.set_title("Product count by Type ")


ax2= plt.subplot(1,2,2)
type_failure_rate = train_df.groupby('Type')['Machine failure'].mean().reset_index()
sns.barplot(x='Type', y='Machine failure', data=type_failure_rate,  alpha= 0.9, width= 0.4, ax= ax2)
ax2.set_title('Machine failure Rate by Type')
ax2.set_ylabel('Failure Rate')
ax2.set_ylim([0, 0.03])
for p in ax2.patches:
    ax2.annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=9, color='black', xytext=(0, 8),
                textcoords='offset points')

plt.subplots_adjust(wspace=0.7)
plt.show()


# ## Air Temperature

# In[13]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['Air temperature [K]'], kde = True, color = 'crimson')
plt.title('Histplot of Air Temparature')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Air temperature [K]', x = 'Machine failure', palette = palette1)
plt.title('boxplot of Air temperature')
plt.subplots_adjust(wspace=0.35)


# ## Process Temperature

# In[14]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['Process temperature [K]'], kde = True, color = 'dimgrey')
plt.title('Histplot of Process Temparature')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Process temperature [K]', x = 'Machine failure', palette = palette2)
plt.title('boxplot of Process temperature')
plt.subplots_adjust(wspace=0.35)


# ## Rotational Speed

# In[15]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['Rotational speed [rpm]'], kde = True, color = 'green')
plt.title('Histplot')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Rotational speed [rpm]', x = 'Machine failure', palette =palette3 )
plt.title('boxplot')
plt.subplots_adjust(wspace=0.35)


# In[16]:


plt.figure(figsize = (12,6))
train_df['speed_Range'] = pd.cut(train_df['Rotational speed [rpm]'], [1150, 1300, 1450, 1600, 1850, 2100, 2500, 2800])
ax = sns.countplot(x="speed_Range", hue="Machine failure", data=train_df,palette=palette1)

# Annotate the count values on top of each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2, height + 10),
                ha='center')

plt.title('Countplot of Speed Range')
plt.show()


# More than `95%` of the roational speed data points are in only this range of `1300, 1850`. It signifies there are many outliers which is distorting the shape of our the the `Rotational Speed` attribute.

# ## Torque

# In[17]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['Torque [Nm]'], kde = True, color = 'grey')
plt.title('Histplot')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Torque [Nm]', x = 'Machine failure', palette = ['crimson', 'dimgray'])
plt.title('boxplot')
plt.subplots_adjust(wspace=0.35)


# ## Tool wear

# In[18]:


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
sns.histplot(train_df['Tool wear [min]'], kde = True, color = 'mediumseagreen')
plt.title('Countplot')

plt.subplot(1,2,2)
sns.boxplot(data= train_df, y = 'Tool wear [min]', x = 'Machine failure', palette = palette4)
plt.title('boxplot')
plt.subplots_adjust(wspace=0.35)


# ## Binary Columns Plot

# In[19]:


for binary in ([['TWF', 'HDF', 'PWF'],['OSF','RNF','is_generated']]):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4)) 
    palette = ['dimgray', 'red']
    # Plot the count plots for the first three binary columns
    for i, col in enumerate(binary):
        ax = axs[i]  # Select the appropriate subplot
        sns.countplot(data=train_df, x=col, hue='Machine failure', ax=ax, palette= palette)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2, height + 10), ha='center')

        ax.set_title(f'Count plot of {col}')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


# In[20]:


corr_matrix = train_df.corr().round(3)
colormap = plt.cm.RdBu_r
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize = (15,12))
sns.heatmap(corr_matrix,linewidths=0.1, vmax=1.0, vmin=-1.0, annot = True, 
            linecolor='white',cmap= colormap, square = True, fmt='.2g', annot_kws={"size": 14}, mask = mask)


# Following columns have strong correlation with our target columns:
# 1. `TWF`
# 2. `HDF`
# 3. `PWF`
# 4. `OSF`

# ## Modifying column names
# It is required because algorithms like`xgboost` and `Lightgbm` will throw error when found names of attributes with `[]`

# In[21]:


import re
def clean_feature_names(data):
    cleaned_columns = []
    for column in data.columns:
        cleaned_column = re.sub('[^A-Za-z0-9_]+', '', column)
        cleaned_columns.append(cleaned_column)
    data.columns = cleaned_columns
    return data


# In[22]:


train_df = clean_feature_names(train_df)
test_df = clean_feature_names(test_df)
df = clean_feature_names(df)


# # Helper Functions

# ## Helper Function- I
# In this problem, our evaluation metric is `roc_auc_score`.ROC AUC (Receiver Operating Characteristic Area Under the Curve) is a performance metric commonly used in binary classification tasks. It quantifies the ability of a classifier to distinguish between positive and negative classes by measuring the area under the receiver operating characteristic curve. The ROC curve plots the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds. A higher ROC AUC score indicates better classification performance, with a score of 1 representing a perfect classifier and a score of 0.5 representing a random classifier. For more understanding, explore the below graph.
# <img src = "https://mlwhiz.com/images/roc-auc-curves-explained/5_hu84485d9ac406291b53c5109d7ec0e2a3_312173_1500x0_resize_box_2.png">

# In[23]:


def train_classifier(clf,x_train,y_train,x_test,y_test, to_return= False):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    score = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])
    if to_return is False:
        cm = confusion_matrix(y_pred, y_test)
        if score>0.92:
                print(f'{gren}{cm}')
                print(f'{gren}Ruc_score is {score}')
                print('*'*90)
        else:
            print(f'{red}{cm}')
            print(f"{red}Not a good model with score = {score}")
            print('x'*90)
        return clf
    return score


# ## Helper Function - II
# It will generate submission file for us.

# In[24]:


def submit_file(model, X_test, filename = 'submission.csv'):
    preds = model.predict_proba(X_test)
    predicted_prob = [pred[1] for pred in preds]
    sub_df['Machine failure'] = predicted_prob
    sub_df.to_csv(filename, index = False)
    print(f'{blu}File successfully created with name {filename}')


# # Training a Baseline Model
# As we saw in the heatmap, some columns have strong correlation with our target columns, so in this section. I will be using only those strong_correlation features for training purpose.
# 
# This type of model helps us to set a base score which helps us to check whether our model is performing better than a random, simple model. Normally we use random value generator but for now, I am using strong correlation columns as it will also save our time.

# In[25]:


strong_correlation_col = [ 'TWF', 'HDF', 'PWF', 'OSF','RNF']
base_train = train_df[strong_correlation_col]
base_test = test_df[strong_correlation_col]


# In[26]:


from sklearn.model_selection import train_test_split
base_X_train, base_X_val, base_y_train, base_y_val = train_test_split(base_train, train_df['Machinefailure'], test_size = 0.2, random_state = 15)


# In[27]:


base_model = CatBoostClassifier(verbose = False, auto_class_weights = 'Balanced')
train_classifier(base_model, base_X_train, base_y_train, base_X_val, base_y_val)


# Our model is performing quite good without any processing of data, probably this is due to the dependecy of our target column on these columns.

# # Feature Engineering

# ## Dropping `id` and column.

# In[28]:


df.drop(['id'], axis = 1, inplace = True)


# In[29]:


df.columns


# ## `Product ID`

# In[30]:


#This step is part of post processing, earlier I was dropping this column but I found that this is useful one.
df['ProductID'] = df['ProductID'].str.replace('M','').str.replace('L','').str.replace('H','').astype(int)


# In[31]:


mf = df.copy()


# In[32]:


df = mf.copy()


# ## `RotationalSpeed`: 
# ### Power
# Calculate the power consumed by the machine using the rotational speed and torque. Power is an important factor that can influence machine failure, so incorporating it as a feature might be informative.
# 
# Power = Torque * (2 * pie * rotationalspeed)/60

# In[33]:


df['Power'] = df['TorqueNm'] * df['Rotationalspeedrpm']


# ### Speed to Torque Ratio

# In[34]:


df['RotationalSpeed_TorqueRatio'] = df['Rotationalspeedrpm'] / df['TorqueNm']


# ### Eficiency Index:
# I found that, this feature is not adding any additional value to the model in postprocessing.

# In[35]:


df['EfficiencyIndex'] = df['Power'] / df['Toolwearmin']


# ## Temperatures:
# Calculating the temperature difference, ratio, variability between the air temperature and process temperature. These features can capture the variations and relative changes between the two temperatures, which might be indicative of machine failure.

# In[36]:


# Calculate temperature difference
df['TemperatureDifference'] = df['ProcesstemperatureK'] - df['AirtemperatureK']
# Calculate temperature variability
df['TemperatureVariability'] = df[['AirtemperatureK', 'ProcesstemperatureK']].std(axis=1)
# Calculate temperature ratio
df['TemperatureRatio'] = df['ProcesstemperatureK'] / df['AirtemperatureK']
#Calculcating Average Temperature
df['AverageTemperature'] = (df['ProcesstemperatureK'] + df['AirtemperatureK'])/2


# ## ToolWearMin:
# ### ToolWearRate
# Calculating the rate of tool wear by dividing the tool wear (in minutes) by the maximum tool wear. This feature can capture the wear and tear of the machine tool and its relationship with failure probability.

# In[37]:


# Calculate tool wear rate
max_tool_wear = df['Toolwearmin'].max()
df['ToolWearRate'] = df['Toolwearmin'] / max_tool_wear
# Calculate temperature change rate
df['TemperatureChangeRate'] = df['TemperatureDifference'] / (np.where(df['Toolwearmin']==0, 2,df['Toolwearmin']))
df['Torque_ToolwearRatio'] = df['TorqueNm'] / (np.where(df['Toolwearmin']==0, 2,df['Toolwearmin']))


# ## Failure

# In[38]:


df['TotalFailures'] = df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1)


# In[39]:


# This step is part of PostProcessing
# df.drop(['TWF','HDF','PWF','OSF'], axis =1, inplace = True)
df.drop(['RNF'], axis =1, inplace = True)
#Earlier in this notebook, I found that these columns are not useful as they have completely aligned with 
# target column, so due to this reason I am removing these columns. 


# # Outlier Detection and Removal
# Using simple I.Q.R method with some buffer to declare a value as 

# In[40]:


numeric_cols = ['ProductID','AirtemperatureK', 'ProcesstemperatureK', 
                'Rotationalspeedrpm','TorqueNm', 'Toolwearmin','Power',
                'TemperatureDifference','TemperatureRatio','TemperatureVariability',
               'RotationalSpeed_TorqueRatio', 'ToolWearRate','TemperatureChangeRate',
                'Torque_ToolwearRatio','AverageTemperature'
               ,'EfficiencyIndex']


# In[41]:


# print(df.quantile(0.75)+1.5*(df.quantile(0.75) - df.quantile(0.25)), df.quantile(0.25)-1.5*(df.quantile(0.75) - df.quantile(0.25)))
data = {'Attributes': df.drop('Type', axis = 1).columns,
             'upper_bound' : df.quantile(0.75)+1.5*(df.quantile(0.75) - df.quantile(0.25)),
             'lower_bound':  df.quantile(0.25)-1.5*(df.quantile(0.75) - df.quantile(0.25))}
bound_df = pd.DataFrame(data)
bound_df = bound_df.reset_index(drop=True)
bound_df


# In[42]:


def replace_outliers(data):
    for col in data[numeric_cols]:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        if col not in ['Rotationalspeedrpm', 'Power']:
            data[col] = np.clip(data[col], lower, upper)
        else: 
            data[col] = np.clip(data[col], lower-100, upper+200)

    return data


# In[43]:


df = replace_outliers(df)


# ## Generating Polynomial features for continous features

# In[44]:


features_list = ['AirtemperatureK', 'ProcesstemperatureK', 'Rotationalspeedrpm', 'TorqueNm', 'Toolwearmin']
for feat in features_list:
        df[f'{feat}Squared'] = df[feat] ** 2
        df[f'{feat}Cubed'] = df[feat] ** 3
        df[f'{feat}Log'] = df[feat].apply(lambda x: math.log(x) if x > 0 else 0)


# # Scaling 

# In[45]:


new_cols = ['AirtemperatureKSquared', 'AirtemperatureKLog','ProcesstemperatureKSquared',
                     'ProcesstemperatureKLog','RotationalspeedrpmSquared', 'RotationalspeedrpmLog', 
                     'TorqueNmSquared','TorqueNmLog','ToolwearminSquared','ToolwearminLog', 'TorqueNmCubed', 'ToolwearminCubed',
                    'AirtemperatureKCubed','ProcesstemperatureKCubed','RotationalspeedrpmCubed']
numeric_cols = numeric_cols+ new_cols


# In[46]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# # Encoding Catgorical Columns

# In[47]:


## This is part of PostProcessing
## type_copy = df["Type"].copy()
# from sklearn.preprocessing import OrdinalEncoder
# encoder = OrdinalEncoder()
# df["Type"] = encoder.fit_transform(np.array(df["Type"]).reshape(-1,1))
df = pd.get_dummies(df, columns = ['Type'])
df.drop('Type_M', axis = 1, inplace = True)


# In[48]:


df.describe()


# # Splitting input and output cols

# In[49]:


target = df['Machinefailure']
y = target.dropna()


# In[50]:


df.drop('Machinefailure', axis = 1, inplace = True)


# In[51]:


df.shape


# In[52]:


X = df.iloc[:train_df.shape[0],:]
X_test = df.iloc[train_df.shape[0]:,:]
X.shape, X_test.shape


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.15, random_state = 15)


# # Model Training and Evaluation

# In[54]:


from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
#for validation
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[55]:


models = {
    'cat': CatBoostClassifier(verbose = False, random_state = 42, auto_class_weights='Balanced'),
    'lgbm': LGBMClassifier(is_unbalance = True, random_state = 42, metric = 'auc'),
    'xgb': XGBClassifier(eval_metric='auc',random_state = 42,objective="binary:logistic"),
    'DT' : DecisionTreeClassifier(random_state = 42),
    'RFC': RandomForestClassifier(random_state = 42, class_weight='balanced'),
    'LR' : LogisticRegression(),
    'brf' : BalancedRandomForestClassifier(random_state=42),
    'mlp' : MLPClassifier(random_state = 42),
#     'svm' : SVC(probability = True),
#     'KNN' : KNeighborsClassifier(n_neighbors= 9),
    'hgb':HistGradientBoostingClassifier(class_weight='balanced', random_state = 42),
#     'gbc': GradientBoostingClassifier(),
#     'bc' : BaggingClassifier(),
    'adc':AdaBoostClassifier(random_state = 42),
    'gnb' : GaussianNB(),
#     'mnb': MultinomialNB(),
    'bnb': BernoulliNB()
}


# In[56]:


cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
cv_splits = list(cv.split(X,y))


# In[57]:


get_ipython().run_cell_magic('time', '', 'for i in range(len(models)):\n    scores = []\n    model = list(models.values())[i]\n    print(f\'{blu}For {list(models.keys())[i]}\')\n    for j, (train_index, test_index) in enumerate(cv_splits):\n        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[test_index]\n        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[test_index]\n        \n        score = train_classifier(model, X_train_fold, y_train_fold, \n                                 X_val_fold, y_val_fold, to_return = True)\n        scores.append(score)\n    print(f\'{blu}{scores}\')\n    mean_score = np.mean(scores)\n    if mean_score>0.91:\n            print(f\'{gren} Mean Ruc_score is {mean_score}\')\n            print(\'*\'*90)\n    else:\n            print(f"{red}Not a good model with Mean score = {mean_score}")\n            print(\'x\'*90)\n')


# ## Insights:
# <h3 style = "display: inline;">Plan</h3> <p> We will hypertune 5 best models on the basis of above score and make predictions of them and for the final prediction, we will build a voting classifier using these hypertuned models. So, lets do it.</p>
# 
# <h3 style = "display: inline;">Top-5 models are</h3>
# 
# 1. XGBoost
# 2. HistGradientBooster
# 3. BalancedRandomForest
# 4. AdaBoostClassifier
# 5. LGBMClassifier
# 6. CatBoostClassifier

# # HyperParameter Tuning

# In[58]:


import optuna
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import VotingClassifier, StackingClassifier


# In[59]:


model = BalancedRandomForestClassifier(random_state= 42)
print(train_classifier(model, X_train, y_train, X_val, y_val))
feature_importances = model.feature_importances_

# Sort the feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
# Print the feature importances
plt.figure(figsize = (14, 8))
sns.barplot(y =X_train.columns[sorted_indices], x =  sorted_feature_importances)
# for feature_index, importance in zip(sorted_indices, sorted_feature_importances):
#     print(f"Feature: {X_train.columns[feature_index]}, Importance: {importance}")


# ## `XGBClassifier()`

# In[60]:


get_ipython().run_cell_magic('time', '', '# imbalance_ratio = len(y_train[y_train == 1]) / len(y_train[y_train == 0])\ndef xgb_objective(trial):\n    params = {\n        \'n_estimators\': trial.suggest_int(\'n_estimators\', 100, 1100),\n        \'learning_rate\': trial.suggest_loguniform(\'learning_rate\', 0.001, 0.3),\n        \'max_depth\': trial.suggest_int(\'max_depth\', 3, 15),\n        \'subsample\': trial.suggest_uniform(\'subsample\', 0.5, 1.0),\n        \'colsample_bytree\': trial.suggest_uniform(\'colsample_bytree\', 0.5, 1.0),\n    }\n\n    classifier = XGBClassifier(**params, random_state=42,eval_metric=\'auc\',\n                                objective = "binary:logistic")\n    classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)\n\n    y_pred_proba = classifier.predict_proba(X_val)[:, 1]\n    roc_auc = roc_auc_score(y_val, y_pred_proba)\n\n    return roc_auc\n\nxgb_study = optuna.create_study(direction=\'maximize\')\nxgb_study.optimize(xgb_objective, n_trials=15)\n\n# Print the best hyperparameters and corresponding ROC AUC scores - XGBClassifier\nxgb_best_params = xgb_study.best_params\nxgb_best_score = xgb_study.best_value\nprint("XGB Best Hyperparameters: ", xgb_best_params)\nprint("XGB Best ROC AUC Score: ", xgb_best_score)\n')


# In[61]:


xgb_model = XGBClassifier(**xgb_best_params, objective = "binary:logistic", eval_metric='auc',random_state = 42)
train_classifier(xgb_model,X_train, y_train, X_val, y_val)
submit_file(xgb_model, X_test, filename = 'xgb.csv')
plt.figure(figsize = (14, 8))
sns.barplot(y =X_train.columns, x =  xgb_model.feature_importances_, 
            order=X_train.columns[np.argsort(-xgb_model.feature_importances_)])
plt.show()


# ## `LGBMClassifier(is_unbalance = True)`

# In[62]:


get_ipython().run_cell_magic('time', '', 'def lgbm_objective(trial):\n\n    params = {\n        \'n_estimators\': trial.suggest_int(\'n_estimators\', 100, 900),\n        \'learning_rate\': trial.suggest_loguniform(\'learning_rate\', 0.001, 0.1),\n        \'max_depth\': trial.suggest_int(\'max_depth\', 3, 10),\n        \'num_leaves\': trial.suggest_int(\'num_leaves\', 10, 1000),\n        \'subsample\': trial.suggest_uniform(\'subsample\', 0.5, 1.0),\n        \'colsample_bytree\': trial.suggest_uniform(\'colsample_bytree\', 0.5, 1.0),\n    }\n\n    classifier = LGBMClassifier(**params, is_unbalance = True,metric = \'auc\' )\n    classifier.fit(X_train, y_train)\n    y_pred_proba = classifier.predict_proba(X_val)[:, 1]\n\n    # Calculate ROC AUC score for validation predictions\n    roc_auc = roc_auc_score(y_val, y_pred_proba)\n\n    return roc_auc\n\nstudy = optuna.create_study(direction=\'maximize\')\nstudy.optimize(lgbm_objective, n_trials=15)\n\n# Print the best hyperparameters and corresponding ROC AUC score\nlgbm_best_params = study.best_params\nlgbm_best_score= study.best_value\nprint("Best Hyperparameters: ",lgbm_best_params)\nprint("Best ROC AUC Score: ", lgbm_best_score)\n')


# In[63]:


lgbm_model = LGBMClassifier(**lgbm_best_params, is_unbalance = True, metric = 'auc')
model = train_classifier(lgbm_model, X_train, y_train, X_val, y_val)
submit_file(lgbm_model, X_test, filename = 'lgbm.csv')
plt.figure(figsize = (14, 8))
sns.barplot(y =X_train.columns, x =  lgbm_model.feature_importances_, 
            order=X_train.columns[np.argsort(-lgbm_model.feature_importances_)])
plt.show()


# ## `BalancedRandomForestClassifier`

# In[64]:


get_ipython().run_cell_magic('time', '', 'def brc_objective(trial):\n    # Define the hyperparameters to optimize\n    params = {\n        \'n_estimators\': trial.suggest_int(\'n_estimators\', 100, 1000),\n        \'max_depth\': trial.suggest_categorical(\'max_depth\', [3, 5, 10, None]),\n        \'min_samples_split\': trial.suggest_int(\'min_samples_split\', 2, 10),\n        \'min_samples_leaf\': trial.suggest_int(\'min_samples_leaf\', 1, 10),\n        \'max_features\': trial.suggest_categorical(\'max_features\', [\'auto\', \'sqrt\', \'log2\', None])\n    }\n\n    classifier = BalancedRandomForestClassifier(**params, random_state = 42)\n    classifier.fit(X_train, y_train)\n\n    # Predict probabilities for the validation data\n    y_pred_proba = classifier.predict_proba(X_val)[:, 1]\n\n    # Calculate ROC AUC score for validation predictions\n    roc_auc = roc_auc_score(y_val, y_pred_proba)\n\n    return roc_auc\n\nstudy = optuna.create_study(direction=\'maximize\')\nstudy.optimize(brc_objective, n_trials=15)\n\n# Print the best hyperparameters and corresponding ROC AUC score\nbrc_best_params = study.best_params\nbrc_best_score = study.best_value\nprint("Best Hyperparameters: ", brc_best_params)\nprint("Best ROC AUC Score: ", brc_best_score)\n')


# In[65]:


brc_model = BalancedRandomForestClassifier(**brc_best_params, random_state = 42)
train_classifier(brc_model, X_train, y_train, X_val, y_val)
submit_file(brc_model, X_test, filename = 'brc.csv')
plt.figure(figsize = (14, 8))
sns.barplot(y =X_train.columns, x =  brc_model.feature_importances_, 
            order=X_train.columns[np.argsort(-brc_model.feature_importances_)])
plt.show()


# ## `HistGradientBoostingClassifier`

# In[66]:


get_ipython().run_cell_magic('time', '', 'def objective(trial):\n    params = {\n        \'learning_rate\': trial.suggest_float(\'learning_rate\', 0.01, 0.2),\n        \'max_iter\': trial.suggest_int(\'max_iter\', 100, 1000),\n        \'max_depth\': trial.suggest_int(\'max_depth\', 3, 10),\n        \'l2_regularization\': trial.suggest_float(\'l2_regularization\', 0.0, 1.0)\n    }\n\n    clf = HistGradientBoostingClassifier(**params, random_state = 42 )\n    clf.fit(X_train, y_train)\n    \n    y_pred_proba = clf.predict_proba(X_val)[:, 1]\n    roc_auc = roc_auc_score(y_val, y_pred_proba)\n    \n    return roc_auc\n\nstudy = optuna.create_study(direction=\'maximize\')\nstudy.optimize(objective, n_trials=15)\n\nhgbc_best_params = study.best_params\nhgbc_best_score = study.best_value\n\nprint("Best Hyperparameters:", hgbc_best_params)\nprint("Best ROC AUC Score:", hgbc_best_score)\n')


# In[67]:


hgbc_model = HistGradientBoostingClassifier(**hgbc_best_params, random_state = 42, class_weight='balanced')
train_classifier(hgbc_model, X_train, y_train, X_val, y_val)
submit_file(hgbc_model, X_test, filename = 'hgbc.csv')
# sns.barplot(y =X_train.columns, x =  hgbc_model.feature_importances_, order=X_train.columns[np.argsort(hgbc_model.feature_importances_)])
# plt.show()


# ## `CatBoostClassifier(verbose = False, auto_class_weights = 'Balanced')`

# In[68]:


get_ipython().run_cell_magic('time', '', 'def catboost_objective(trial):\n    params = {\n        \'n_estimators\': trial.suggest_int(\'n_estimators\', 100, 1100),\n        \'learning_rate\': trial.suggest_loguniform(\'learning_rate\', 0.001, 0.3),\n        \'max_depth\': trial.suggest_int(\'max_depth\', 3, 15),\n        \'subsample\': trial.suggest_uniform(\'subsample\', 0.5, 1.0),\n        \'colsample_bylevel\': trial.suggest_uniform(\'colsample_bylevel\', 0.5, 1.0),\n    }\n\n    classifier = CatBoostClassifier(**params,verbose = False,random_state = 42, auto_class_weights=\'Balanced\',  eval_metric = \'AUC\')\n    classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)\n\n    y_pred_proba = classifier.predict_proba(X_val)[:, 1]\n    roc_auc = roc_auc_score(y_val, y_pred_proba)\n\n    return roc_auc\n\n# Create the Optuna study and optimize the objective functions\ncatboost_study = optuna.create_study(direction=\'maximize\')\ncatboost_study.optimize(catboost_objective, n_trials=15)\n\n# Print the best hyperparameters and corresponding ROC AUC scores - CatBoostClassifier\ncatboost_best_params = catboost_study.best_params\ncatboost_best_score = catboost_study.best_value\nprint(f"{blk}CatBoost Best Hyperparameters: ", catboost_best_params)\nprint(f"{blk}CatBoost Best ROC AUC Score: ", catboost_best_score)\n')


# In[69]:


get_ipython().run_cell_magic('time', '', "catboost_model = CatBoostClassifier(**catboost_best_params,verbose = False,random_state = 42, auto_class_weights='Balanced')\ntrain_classifier(catboost_model, X_train, y_train, X_val, y_val)\nsubmit_file(catboost_model, X_test, filename = 'cat.csv')\nplt.figure(figsize = (14, 8))\nsns.barplot(y =X_train.columns, x =  catboost_model.feature_importances_, \n            order=X_train.columns[np.argsort(-catboost_model.feature_importances_)])\nplt.show()\n")


# # Final Prediction

# In[70]:


cat = CatBoostClassifier(**catboost_best_params,verbose = False, auto_class_weights='Balanced')
xgb = XGBClassifier(**xgb_best_params, objective = "binary:logistic")
lgbm = LGBMClassifier(**lgbm_best_params, is_unbalance = True)
brc = BalancedRandomForestClassifier(**brc_best_params)
hgbc = HistGradientBoostingClassifier(**hgbc_best_params)
abc = AdaBoostClassifier(n_estimators = 10, estimator = cat)


# In[71]:


get_ipython().run_cell_magic('time', '', "voting = VotingClassifier([('abc',abc),('xgb',xgb),('lgbm',lgbm),('brc',brc), ('hgbc',hgbc)], voting = 'soft')\ntrain_classifier(voting, X_train, y_train, X_val, y_val)\n")


# In[72]:


submit_file(voting, X_test, filename = 'submission.csv')

