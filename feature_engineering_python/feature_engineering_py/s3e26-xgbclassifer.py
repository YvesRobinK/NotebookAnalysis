#!/usr/bin/env python
# coding: utf-8

# <p align="center" style="text-align:center; align-items: center;">
#     <img src="https://cdn-icons-png.flaticon.com/512/7350/7350859.png" width=150/>
# </p>
# 
# <a href="https://www.flaticon.com/free-icons/cirrhosis" title="cirrhosis icons">Cirrhosis icons created by Freepik - Flaticon</a>
# 
# # S3E26 | XGBClassifer
# 
# > Cirrhosis is a late stage of scarring (fibrosis) of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism. Each time your liver is injured, it tries to repair itself. In the process, scar tissue forms. As the cirrhosis progresses, more and more scar tissue forms, making it difficult for the liver to function (decompensated cirrhosis). Advanced cirrhosis is life-threatening.
# 
# **I combine my predicitons with results from other notebooks. Please upvote their work first.**
# * [PS3E26 | Cirrhosis Survial Prediction | Multiclass](https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass) by [MASTER JIRAIYA](https://www.kaggle.com/arunklenin)
# * [PS3E25 - 🦠 Cirrhosis Multi-Class Solution](https://www.kaggle.com/code/dreygaen/ps3e25-cirrhosis-multi-class-solution) by [LUCAS ANTOINE](https://www.kaggle.com/dreygaen)
# * [🛑Multi-Class 📈Prediction of 🦧Cirrhosis Outcomes](https://www.kaggle.com/code/satyaprakashshukl/multi-class-prediction-of-cirrhosis-outcomes) by [SATYA](https://www.kaggle.com/satyaprakashshukl)
# 
# **=> Additionally, you are welcome to utilize the output from my notebook for your submissions.**
# 
# ## Notebook Objective | Summary
# 
# We create some additonal features and evaluate the performance of different approaches and feature importances on this enhanced dataset. Approaches covered:
# 
# * ❌ LogisticRegression
# * ❌ DecisionTreeClassifier
# * ❌ RandomForestClassifier
# * ❌ SVC
# * ❌ KNN
# * ✅ XGBClassifier
# * ❓ CatBoostClassifier
# * ✅ LGBMClassifier
# * ❌ Keras Sequential NN
# 
# **Model parameterization for above models is included, however, it is not optimized..**
# 
# ## ToDos 📓
# 
# * `CatBoostClassifier` was removed since "I was new to `optuna`, check if optimized model can compete with other approaches
# * When I run `XGBClassifier` and `LGBMClassifier` on the complete dataset, I obtain a difference of 8 features from the TOP-20 most important features (`Risk_Score`, `Diag_Month`, `Age_Years`, `Diagnosis_Date`, `Liver_Function_Index`, `SGOT`, `Albumin`, `Tryglicerides`). Optimize model parameters on different data features. Functions are already capable of dealing with different inputs, however, parameters need to be tuned.
# * Currently, there are no dimension reducing methods implemented
# * Number of optimal features is also not specified
#     * A recursive feature elimination (RFE) algorithm could be interesting, e.g. use `sklearn.feature_selection.RFE`
# 
# ## Dataset Description
# 
# Dataset description, taken from the original data source [Cirrhosis Patient Survival Prediction](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction/data) and added some _general insights_.
# 
# | Variable Name   | Role    | Type        | Demographic | Description                                                                                                                                                                                                                                                                                                 | Units   | Missing Values |
# | --------------- | ------- | ----------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | :------------: |
# | `ID`            | ID      | Integer     | -           | Unique identifier                                                                                                                                                                                                                                                                                           | -       |       No       |
# | `N_Days`        | Other   | Integer     | -           | Number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986                                                                                                                                                                                          | -       |       No       |
# | `Drug`          | Feature | Categorical | -           | Type of drug: D-penicillamine or placebo. _Type of medication may impact the effectiveness of treatment, thus affecting status._                                                                                                                                                                            | -       |      Yes       |
# | `Age`           | Feature | Integer     | Age         | Age. _Age could be related to disease progression; older patients may have a different status trajectory._                                                                                                                                                                                                  | Days    |       No       |
# | `Sex`           | Feature | Categorical | Sex         | Gender: M (male) or F (female). _Biological sex may influence disease patterns and response to treatment, thereby affecting status._                                                                                                                                                                        | -       |       No       |
# | `Ascites`       | Feature | Categorical | -           | Presence of ascites: N (No) or Y (Yes). _The accumulation of fluid in the abdomen, often a sign of advanced liver disease, which could indicate a poorer status._                                                                                                                                           | -       |      Yes       |
# | `Hepatomegaly`  | Feature | Categorical | -           | Presence of hepatomegaly: N (No) or Y (Yes). _Enlargement of the liver. If present, it might suggest more serious liver disease and potentially a poorer status._                                                                                                                                           | -       |      Yes       |
# | `Spiders`       | Feature | Categorical | -           | Presence of spiders: N (No) or Y (Yes). _Spider angiomas are small, spider-like capillaries visible under the skin, associated with liver disease and could indicate a more advanced disease affecting status._                                                                                             | -       |      Yes       |
# | `Edema`         | Feature | Categorical | -           | Presence of edema: N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy). _Swelling caused by excess fluid trapped in the body's tissues, often worsening the prognosis and indicating poorer status._ | -       |       No       |
# | `Bilirubin`     | Feature | Continuous  | -           | Serum bilirubin. _High levels can indicate liver dysfunction and may correlate with more advanced disease and poorer status._                                                                                                                                                                               | mg/dL   |       No       |
# | `Cholesterol`   | Feature | Integer     | -           | Serum cholesterol. _While not directly related to liver function, abnormal levels can be associated with certain liver conditions and overall health status._                                                                                                                                               | mg/dL   |      Yes       |
# | `Albumin`       | Feature | Continuous  | -           | Albumin. _Low levels can be a sign of liver disease and can indicate a poorer status due to the liver's reduced ability to synthesize proteins._                                                                                                                                                            | g/dL    |       No       |
# | `Copper`        | Feature | Integer     | -           | Urine copper. _Elevated in certain liver diseases (like Wilson's disease), and could affect status if levels are abnormally high._                                                                                                                                                                          | µg/day  |      Yes       |
# | `Alk_Phos`      | Feature | Continuous  | -           | Alkaline phosphatase. _An enzyme related to the bile ducts; high levels might indicate blockage or other issues related to the liver._                                                                                                                                                                      | U/Liter |      Yes       |
# | `SGOT`          | Feature | Continuous  | -           | SGOT. _An enzyme that, when elevated, can indicate liver damage and could correlate with a worsening status._                                                                                                                                                                                               | U/mL    |      Yes       |
# | `Triglycerides` | Feature | Integer     | -           | Triglycerides. _Though mainly a cardiovascular risk indicator, they can be affected by liver function and, by extension, the status of the patient._                                                                                                                                                        | -       |      Yes       |
# | `Platelets`     | Feature | Integer     | -           | Platelets per cubic. _Low platelet count can be a result of advanced liver disease and can indicate a poorer status._                                                                                                                                                                                       | mL/1000 |      Yes       |
# | `Prothrombin`   | Feature | Continuous  | -           | Prothrombin time. _A measure of how long it takes blood to clot. Liver disease can cause increased times, indicating poorer status._                                                                                                                                                                        | s       |      Yes       |
# | `Stage`         | Feature | Categorical | -           | Histologic stage of disease (1, 2, 3, or 4). _The stage of liver disease, which directly correlates with the patient's status - the higher the stage, the more serious the condition._                                                                                                                      | -       |      Yes       |
# | `Status`        | Target  | Categorical | -           | Status of the patient: C (censored), CL (censored due to liver tx), or D (death)                                                                                                                                                                                                                            | -       |       No       |
# 

# In[1]:


import sys
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import xgboost as xgb
from xgboost import plot_importance
from ydata_profiling import ProfileReport

pd.set_option('display.max_columns', None)
# colors = ["#078533", "#850728", "#fcba03"]
sns.set_palette(sns.color_palette("Set2"))
print(f"Running on {sys.version}")

KAGGLE_ENV = True
GENERATE_REPORTS = False

# If we are running in a kaggle environment, then we need to use the kaggle
# provided data paths.
if KAGGLE_ENV:
    data_path = "../input/"
else:
    data_path = "./data/"


# In[2]:


df_train = pd.read_csv(data_path + "playground-series-s3e26/train.csv").drop(
    ["id"], axis=1
)
df_test = pd.read_csv(data_path + "playground-series-s3e26/test.csv")
test_IDs = df_test.id
df_test = df_test.drop("id", axis=1)
df_sample_sub = pd.read_csv(data_path + "playground-series-s3e26/sample_submission.csv")
df_supp = pd.read_csv(
    data_path + "cirrhosis-patient-survival-prediction/cirrhosis.csv"
)[df_train.columns]

# Merge supplementary data
df_train = pd.concat(objs=[df_train, df_supp]).reset_index(drop=True)

LABEL = "Status"
CAT_FEATS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
NUM_FEATS = [x for x in df_train.columns if x not in CAT_FEATS and x != LABEL]
ORG_FEATS = df_train.drop(LABEL, axis=1).columns.tolist() # original feats

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")


# # EDA | Descriptive Statistics
# 

# In[3]:


df_train


# In[4]:


get_ipython().run_cell_magic('time', '', 'if GENERATE_REPORTS:\n    # Generate the profile report\n    profile = ProfileReport(df_train, title="YData Profiling Report - Cirrhosis")\n    profile.to_notebook_iframe()\n')


# In[5]:


desc_df = df_train.describe(include="all")
desc_df = desc_df.T
desc_df['unique'] = desc_df['unique'].fillna(df_train.nunique())
desc_df['count'] = desc_df['count'].astype('int16')
desc_df['missing'] = df_train.shape[0] - desc_df['count']
desc_df


# We can see that there **are** missing values (in combined train dataset). However, no duplicates were found. 
# 
# **NOTE THAT `Prothrombin` ONLY SHOWS 50 DISTINCT VALUES, HENCE IT MIGHT BE WORTH CONSIDERING THIS FEATURE AS CATEGORICAL (same might hold for other features also)**
# 
# Let's check the distribution of the target variable `Status`.

# In[6]:


# Counting the observations for each category
status_counts = df_train[LABEL].value_counts()
labels = status_counts.index
sizes = status_counts.values

# Calculating the percentage of each category
percentages = 100.*sizes/sizes.sum()

# Creating the pie chart with percentages in the labels
plt.figure(figsize=(10, 6))
plt.pie(sizes, labels=[f"{l}, {s:.1f}%" for l, s in zip(labels, percentages)], startangle=90)
plt.gca().set_aspect("equal")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), labels=labels, title=LABEL)
plt.title(f"Distribution of {LABEL}")
plt.show()


# We can see that the target variable is imbalanced. Also most of the patients are censored (meaning that the patient was lost to follow-up or the study ended before the patient died or received a liver transplant). This is a common problem in survival analysis and we will deal with it later.
# 
# > Also note that the `Status` CL is only represented with a few observations. It might make sense to deal with CL observations in a special way, e.g. drop them from classification.
# 
# Let us now check the distribution of the features. We will start with the categorical features.

# In[7]:


plt.figure(figsize=(14, len(CAT_FEATS) * 2))
for i, col in enumerate(CAT_FEATS):
    plt.subplot(len(CAT_FEATS) // 2 + 1, 3, i + 1)
    sns.countplot(x=col, hue=LABEL, data=df_train)
    plt.title(f"{col} vs {LABEL}")
    plt.tight_layout()


# Let's check the distribution of the continuous features.

# In[8]:


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, ax in enumerate(axes.flatten()):
    sns.violinplot(x=LABEL, y=NUM_FEATS[i], data=df_train, ax=ax)
    # Set x ticks to be the original labels (inverse transform)
    ax.set_title(f"{NUM_FEATS[i]} vs {LABEL}")
plt.tight_layout()
plt.show()


# We obtain a few outliers. We will remove them in the feature engineering section. Also our preliminary assumptions on the different influences of each feature on `Status` from the beginning seem to hold on a first sight (main effects only, no confounding influences respected).
# 
# Let's check the correlation between the features.

# In[9]:


plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(df_train[NUM_FEATS].corr(), dtype=bool))
sns.heatmap(df_train[NUM_FEATS].corr(), annot=True, mask=mask)
plt.show()


# There are no highly correlated features. However, we can see a medium positive correlation between `Copper` and `Bilirubin` and a medium negative correlation between `Albumin` and `Bilirubin`. This is expected since `Copper` and `Bilirubin` are both related to liver function (especially in presence of liver disfunction). Also, `Albumin` and `Bilirubin` are negatively correlated since `Albumin` is a protein produced by the liver and `Bilirubin` is a waste product of the breakdown of red blood cells.
# 
# From [Dee Dee's](https://www.kaggle.com/ddosad) EDA [notebook](https://www.kaggle.com/code/ddosad/ps3e26-visual-eda-automl-cirrhosis-outcomes) we can also see that correlations among train and test data nearly follow the same pattern.
# 
# Let's further analyze the correlations and `Status` clusters in a pairplot.

# In[10]:


pairplot = sns.pairplot(df_train[NUM_FEATS + [LABEL]].sample(frac=.01), 
                 hue=LABEL, 
                 corner=True)


# **It looks like (to me) that some features can be seperated quite effectively with a linear approach (only inspected 2D space). Hence, a classical SVM approach might perform not too bad ...**
# 
# **Also, the feature distributions (histogram) show some nice "bell shaped" curves ...**

# # Feature Engineering
# 
# **TODO** I should include an "overview" table of the features and tranformations of this notebook...

# In[11]:


# SET THE DATA VERSION FOR STORING DATA
DATA_VERSION = 24

CAT_FEATS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
NUM_FEATS = [x for x in df_train.columns if x not in CAT_FEATS and x != LABEL]

# Copy dataframes and modify them
df_train_mod = df_train.copy()
df_test_mod = df_test.copy()

print(f"Train shape: {df_train_mod.shape}")
print(f"Test shape: {df_test_mod.shape}")
assert df_train_mod.shape[1]-1 == df_test_mod.shape[1]


# ## Impute Missing Values
# 
# Inspired by this [notebook](https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass). Please upvote if you like the work.
# 
# However, I need to adjust the parameter optimization with`optuna` (other notebook) when dealing with imputed values... Currently, not satisfying - therefore, I just drop observations with missing values.

# In[12]:


# Missing categories
missing_cat=[f for f in df_train_mod.columns if df_train_mod[f].dtype=="O" if df_train_mod[f].isna().sum()>0]
train_missing_pct = df_train_mod[missing_cat].isnull().mean() * 100
test_missing_pct = df_train_mod[missing_cat].isnull().mean() * 100

missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])
print(missing_pct_df)

cat_params={
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
}
def store_missing_rows(df, features):
    missing_rows = {}
    
    for feature in features:
        missing_rows[feature] = df[df[feature].isnull()]
    
    return missing_rows

def fill_missing_categorical(train, test, target, features, max_iterations=10):
    df = pd.concat([train.drop(columns=target), test], axis="rows")
    df = df.reset_index(drop=True)

    # Step 1: Store the instances with missing values in each feature
    missing_rows = store_missing_rows(df, features)

    # Step 2: Initially fill all missing values with "Missing"
    for f in features:
        df[f] = df[f].fillna("Missing_" + f)

    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in features:
            # Skip features with no missing values
            rows_miss = missing_rows[feature].index

            missing_temp = df.loc[rows_miss].copy()
            non_missing_temp = df.drop(index=rows_miss).copy()
            missing_temp = missing_temp.drop(columns=[feature])

            other_features = [x for x in df.columns if x != feature and df[x].dtype == "O"]

            X_train = non_missing_temp.drop(columns=[feature])
            y_train = non_missing_temp[[feature]]

            catboost_classifier = CatBoostClassifier(**cat_params)
            catboost_classifier.fit(X_train, y_train, cat_features=other_features, verbose=False)

            # Step 4: Predict missing values for the feature and update all N features
            y_pred = catboost_classifier.predict(missing_temp)
            
            # Convert y_pred to strings if necessary
            if y_pred.dtype != "O":
                y_pred = y_pred.astype(str)

            df.loc[rows_miss, feature] = y_pred

    train[features] = np.array(df.iloc[:train.shape[0]][features])
    test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train, test


# In[13]:


# Missing numerical values
missing_num=[f for f in df_train_mod.columns if df_train_mod[f].dtype!="O" and df_train_mod[f].isna().sum()>0]
train_missing_pct = df_train_mod[missing_num].isnull().mean() * 100
test_missing_pct = df_test_mod[missing_num].isnull().mean() * 100
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis=1, keys=['Train %', 'Test%'])
print(missing_pct_df)

cb_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'random_state': 42,
        }
lgb_params = {
            'n_estimators': 50,
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
        }
def rmse(y1,y2):
    return(np.sqrt(mean_squared_error(y1,y2)))

def fill_missing_numerical(train,test,target, features, max_iterations=10):
    train_temp=train.copy()
    if target in train_temp.columns:
        train_temp=train_temp.drop(columns=target)
        
    
    df=pd.concat([train_temp,test],axis="rows")
    df=df.reset_index(drop=True)
    
    # Step 1: Store the instances with missing values in each feature
    missing_rows = store_missing_rows(df, features)
    
    # Step 2: Initially fill all missing values with "Missing"
    for f in features:
        df[f]=df[f].fillna(df[f].mean())
    
    cat_features=[f for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])]
    dictionary = {feature: [] for feature in features}
    
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in features:
            # Skip features with no missing values
            rows_miss = missing_rows[feature].index
            
            missing_temp = df.loc[rows_miss].copy()
            non_missing_temp = df.drop(index=rows_miss).copy()
            y_pred_prev=missing_temp[feature]
            missing_temp = missing_temp.drop(columns=[feature])
            
            
            # Step 3: Use the remaining features to predict missing values using Random Forests
            X_train = non_missing_temp.drop(columns=[feature])
            y_train = non_missing_temp[[feature]]
            
            model = CatBoostRegressor(**cb_params)
#             if iteration>3:
#                 model = lgb.LGBMRegressor()
            model.fit(X_train, y_train,cat_features=cat_features, verbose=False)
            
            # Step 4: Predict missing values for the feature and update all N features
            y_pred = model.predict(missing_temp)
            df.loc[rows_miss, feature] = y_pred
            error_minimize=rmse(y_pred,y_pred_prev)
            dictionary[feature].append(error_minimize)  # Append the error_minimize value

    for feature, values in dictionary.items():
        iterations = range(1, len(values) + 1)  # x-axis values (iterations)
        plt.plot(iterations, values, label=feature)  # plot the values
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Minimization of RMSE with iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    train[features] = np.array(df.iloc[:train.shape[0]][features])
    test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train,test


# In[14]:


DROP_MISSING = True
if DROP_MISSING:
    # TODO: this is just preliminary ... not very elegant
    df_train_mod = df_train_mod.dropna()
    df_test_mod = df_test_mod.dropna()
else:
    df_train_mod, df_test_mod = fill_missing_categorical(df_train_mod, df_test_mod, LABEL, missing_cat, 5)
    df_train_mod, df_test_mod = fill_missing_numerical(df_train_mod, df_test_mod, LABEL, missing_num, 5)


# ## Label & Feature Encoding
# 
# Since some of the features are categorical, we will transform them.
# 
# | Encoding Technique | Type of Variable | Support High Cardinality | Handle Unseen Variables | Cons |
# |--------------------|------------------|--------------------------|-------------------------|------|
# | Label Encoding | Nominal | Yes | No | Unseen Variables |
# | Ordinal Encoding | Ordinal | Yes | Yes | Categories interpreted as numerical values |
# | One-Hot / Dummy Encoding | Nominal | No | Yes | Dummy Variable Trap <br> Large dataset |
# | Target Encoding | Nominal | Yes | Yes | Target Leakage <br> Uneven Category Distribution |
# | Frequency / Count Encoding | Nominal | Yes | Yes | Similar encodings |
# | Binary Encoding | Nominal | Yes | Yes | Irreversible |
# | Hash Encoding | Nominal | Yes | Yes | Information Loss or Collision |
# 
# (Thanks to this [notebook](https://www.kaggle.com/code/satyaprakashshukl/multi-class-prediction-of-cirrhosis-outcomes))
# 
# * **One-Hot-Encoding** for features: `Edema`
# * **Ordinal Encoding** for features: `Stage`, 
# * **Binary-Encoding** for features: `Drug`, `Sex`, `Ascites`, `Hepatomegaly`, `Spiders`
# * **LabelEncoding** for label: `Status`
# 
# **Encoding `Edema` with One-Hot-Encoding rather then OrdinalEncoding (N: 0, S: 1, Y: 2) improved the score a bit.**
# 
# ### Label Encoding

# In[15]:


# Encode the label
label_encoder = LabelEncoder()
df_train_mod[LABEL] = label_encoder.fit_transform(df_train_mod[LABEL])


# ### Feature Encoding 

# In[16]:


encoders = {
    'Drug': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['Placebo', 'D-penicillamine']]),
    'Sex': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Ascites': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Hepatomegaly': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    'Spiders': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    # 'Edema': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['N', 'S', 'Y']]),
    'Edema': OneHotEncoder(),
    'Stage': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
}


# In[17]:


for feat, enc in encoders.items():
    if isinstance(enc, OrdinalEncoder):
        df_train_mod[feat] = enc.fit_transform(df_train_mod[[feat]]).astype('int32')
        df_test_mod[feat] = enc.transform(df_test_mod[[feat]]).astype('int32')
    if isinstance(enc, OneHotEncoder):
        # Transform and get new column names
        new_cols = enc.fit_transform(df_train_mod[[feat]]).toarray().astype('int8')
        # col_names = [f"{feat}_{cat}" for cat in enc.categories_[0]]
        col_names = enc.get_feature_names_out()
        
        # Add new columns to the dataframe
        df_train_mod[col_names] = new_cols
        df_train_mod.drop(feat, axis=1, inplace=True)  # Drop original column
        
        # Repeat for the test set
        new_cols_test = enc.transform(df_test_mod[[feat]]).toarray().astype('int8')
        df_test_mod[col_names] = new_cols_test
        df_test_mod.drop(feat, axis=1, inplace=True)


# ## Additional Features
# 
# We will create some additional features. An explanation is provided below:
# 
# | Transformer Class     | Type | Description |
# |-----------------------|------|-------------|
# | `DiagnosisDateTransformer`   | `num` | Calculates 'Diagnosis_Date' by subtracting 'N_Days' from 'Age'. This could provide a more direct measure of time since diagnosis, relevant for analysis.          |
# | `AgeBinsTransformer`         | `cat` | Categorizes 'Age' into bins (19, 29, 49, 64, 99), converting a continuous variable into a categorical one for simplified analysis.                 |
# | `BilirubinAlbuminTransformer`| `num` | Creates a new feature 'Bilirubin_Albumin' by multiplying 'Bilirubin' and 'Albumin', potentially highlighting interactions between these two variables.             |
# | `NormalizeLabValuesTransformer`| `num` | Normalizes laboratory values (like 'Bilirubin', 'Cholesterol', etc.) to their z-scores, standardizing these features for modeling purposes.                       |
# | `DrugEffectivenessTransformer`| `num` | Generates a new feature 'Drug_Effectiveness' by combining 'Drug' and 'Bilirubin' levels. This assumes that changes in 'Bilirubin' reflect drug effectiveness.   |
# | `SymptomScore(Cat)Transformer`    | `num` | Summarizes the presence of symptoms ('Ascites', 'Hepatomegaly', etc.) into a single 'Symptom_Score', simplifying the representation of patient symptoms.      |
# | `LiverFunctionTransformer`   | `num` | Computes 'Liver_Function_Index' as the average of key liver function tests, providing a comprehensive metric for liver health.                                    |
# | `RiskScoreTransformer`       | `num` | Calculates 'Risk_Score' using a combination of 'Bilirubin', 'Albumin', and 'Alk_Phos', potentially offering a composite risk assessment for patients.              |
# | `TimeFeaturesTransformer`    | `num` | Extracts 'Year' and 'Month' from 'N_Days', converting a continuous time measure into more interpretable categorical time units.                                    |
# 

# In[18]:


class DiagnosisDateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Diagnosis_Date'] = X['Age'] - X['N_Days']
        return X
    
class AgeYearsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Age_Years'] = round(X['Age'] / 365.25).astype("int16")
        return X

class AgeGroupsTransformer(BaseEstimator, TransformerMixin):
    """Older people might be hit harder (interaction) by health issues. Also can cover lifestyle influences, i.e.
    alcohol consumption etc."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Use years from above, min=26, max=78
        X['Age_Group'] = pd.cut(X['Age_Years'], bins=[19, 29, 49, 64, 99], labels = [0, 1, 2, 3]).astype('int16')
        return X

class BilirubinAlbuminTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Bilirubin_Albumin'] = X['Bilirubin'] * X['Albumin']
        return X

class DrugEffectivenessTransformer(BaseEstimator, TransformerMixin):
    # Placeholder concept, assuming 'Bilirubin' improvement is a measure of effectiveness
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Drug_Effectiveness'] = X['Drug'] * X['Bilirubin']
        return X

class SymptomScoreTransformer(BaseEstimator, TransformerMixin):
    # From data set explanations above let's add all the "bad" symptoms
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
        symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema_N', 'Edema_S', 'Edema_Y']
        X['Symptom_Score'] = X[symptom_columns].sum(axis=1)
        return X
    
class SymptomCatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.symptom_columns = ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema_N', 'Edema_S', 'Edema_Y']
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        X_copy = X.copy()
        symptom_scores = X_copy[self.symptom_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        self.encoder.fit(symptom_scores.values.reshape(-1, 1))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        symptom_scores = X_transformed[self.symptom_columns].apply(lambda row: ''.join(row.values.astype(str)), axis=1)
        
        encoded_features = self.encoder.transform(symptom_scores.values.reshape(-1, 1)).toarray().astype("int8")
        encoded_feature_names = self.encoder.get_feature_names_out(input_features=['Symptom_Score'])

        # Drop the original symptom columns and add the new encoded features
        # X_transformed.drop(columns=self.symptom_columns, inplace=True)
        X_transformed[encoded_feature_names] = pd.DataFrame(encoded_features, index=X_transformed.index)
        
        return X_transformed


class LiverFunctionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        liver_columns = ['Bilirubin', 'Albumin', 'Alk_Phos', 'SGOT']
        X['Liver_Function_Index'] = X[liver_columns].mean(axis=1)
        return X

class RiskScoreTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Risk_Score'] = X['Bilirubin'] + X['Albumin'] - X['Alk_Phos']
        return X

class TimeFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Diag_Year'] = (X['N_Days'] / 365).astype(int)
        X['Diag_Month'] = ((X['N_Days'] % 365) / 30).astype(int)
        return X
    
class ScalingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_feats = NUM_FEATS + ['Diagnosis_Date', 'Age_Years', 'Bilirubin_Albumin', 'Drug_Effectiveness', 
                                      'Symptom_Score', 'Liver_Function_Index', 'Risk_Score', 'Diag_Year', 'Diag_Month']

    def fit(self, X, y=None):
        self.scaler.fit(X[self.num_feats])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[self.num_feats] = self.scaler.transform(X_scaled[self.num_feats])
        return X_scaled

# Define the pipeline
pipeline = Pipeline([
    ('diagnosis_date', DiagnosisDateTransformer()),
    ('age_years', AgeYearsTransformer()),
    ('age_groups', AgeGroupsTransformer()),
    ('bilirubin_albumin', BilirubinAlbuminTransformer()),
    ('drug_effectiveness', DrugEffectivenessTransformer()),
    ('symptom_score', SymptomScoreTransformer()),
    ('symptom_cat_score', SymptomCatTransformer()),
    ('liver_function', LiverFunctionTransformer()),
    ('risk_score', RiskScoreTransformer()),
    ('time_features', TimeFeaturesTransformer()),
    #('scaling', ScalingTransformer()),
    # ... ?
])

# Apply the pipeline to your dataframes
df_train_mod = pipeline.fit_transform(df_train_mod)
df_test_mod = pipeline.transform(df_test_mod)

# Update the CAT_FEATS
CAT_FEATS = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage', #old
             'Age_Group', 'Symptom_Score'] # new 
# Update the NUM_FEATS ????


# ## Outlier Detection & Removal
# 
# As stated before the numerical features look "kinda" normally distributed. Hence observations with feature values that are more than 6 standard deviations from the mean are  considered outliers and we want to remove them.

# In[19]:


tmp_df = df_train_mod.copy()

# Calculate the mean and standard deviation for each column
means = tmp_df[NUM_FEATS].mean()
std_devs = tmp_df[NUM_FEATS].std()

# Define a threshold for what you consider to be an outlier, typically 3 standard deviations from the mean
n_stds = 6
thresholds = n_stds * std_devs

# Detect outliers
outliers = (np.abs(tmp_df[NUM_FEATS] - means) > thresholds).any(axis=1)

print(f"Detected {sum(outliers)} that are more than {n_stds} SDs away from mean...")


# In[20]:


# The resulting boolean series can be used to filter out the outliers
outliers_df = tmp_df[outliers]

# Overwrite the train data
df_train_mod = tmp_df[~outliers].reset_index(drop=True)
print(f"Train data shape after outlier removal: {df_train_mod.shape}")


# # Dimensionality Reduction
# 
# Following this nice comment of [Vilius Pėstininkas](https://www.kaggle.com/code/markuslill/s3e26-xgbclassifer-lgbmclassifier/comments#2561287) we will try to reduce the dimensionality of our data.
# 
# **TBD!!!**

# In[21]:


def tsne_with_feature_selection_and_pca(data, num_feats, target_column, n_components=2, top_n_features=10, pca_components=None):
    """
    Select top features based on feature importance, optionally apply PCA, and then use t-SNE for visualization.

    Parameters:
    data (DataFrame): The input data.
    num_feats (list): List of numerical feature column names.
    target_column (str): The name of the target column.
    n_components (int): Number of dimensions for t-SNE (2 or 3). Default is 2.
    top_n_features (int): Number of top features to select based on importance. Default is 10.
    pca_components (int or None): Number of PCA components to retain before applying t-SNE. If None, PCA is not applied.
    """
    
    global label_encoder
    
    # Standardizing the numerical features
    scaler = StandardScaler()
    numerical_data_scaled = scaler.fit_transform(data[num_feats])

    # Random Forest for feature importances
    rf = RandomForestClassifier(random_state=42)
    rf.fit(numerical_data_scaled, data[target_column])
    importances = rf.feature_importances_

    # Selecting top_n_features
    indices = np.argsort(importances)[-top_n_features:]
    selected_features = [num_feats[i] for i in indices]

    # Data for t-SNE
    tsne_data = numerical_data_scaled[:, indices]

    # Optionally applying PCA
    if pca_components is not None and pca_components < len(selected_features):
        pca = PCA(n_components=pca_components)
        tsne_data = pca.fit_transform(tsne_data)

    # Applying t-SNE
    tsne = TSNE(n_components=n_components, learning_rate='auto', init='random', perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(tsne_data)

    # Creating a DataFrame for the t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=[f'Component {i+1}' for i in range(n_components)])
    tsne_df[target_column] = label_encoder.inverse_transform(data[target_column].values)

    # Visualizing using Plotly
    if n_components == 3:
        fig = px.scatter_3d(tsne_df, x='Component 1', y='Component 2', z='Component 3', color=target_column)
    else:
        fig = px.scatter(tsne_df, x='Component 1', y='Component 2', color=target_column)
    
    fig.update_layout(width=800, height=600)
    fig.show()

df_train_red = df_train_mod
# tsne_with_feature_selection_and_pca(df_train_red, NUM_FEATS, LABEL, n_components=3, top_n_features=10, pca_components=None)


# In[22]:


# Features to combine 
# All
#df_train_pca = df_train_mod.drop([LABEL], axis=1)
#df_test_pca = df_test_mod

# Numerical feats
df_train_pca = df_train_mod[NUM_FEATS]
df_test_pca = df_test_mod[NUM_FEATS]

# Some, the feats here are taken iteratively from previous runs
#FEATS = ['Platelets', 'Copper', 'Alk_Phos', 'Diagnosis_Date', 'SGOT', 'Age', 'N_Days']
#PCA_FEATS = [c for c in df_train_mod.drop(LABEL, axis=1).columns.values if c not in FEATS]
#df_train_pca = df_train_mod[PCA_FEATS]
#df_test_pca = df_test_mod[PCA_FEATS]

pca = PCA(n_components=10)
df_train_pca = pca.fit_transform(df_train_pca)
df_test_pca = pca.transform(df_test_pca)

print(f"Explained variance per component: {np.round(pca.explained_variance_ratio_, 1)}")


# In[23]:


eps_expl_var_treshold = 0
n_pcas = np.sum(np.round(pca.explained_variance_ratio_, 3) > eps_expl_var_treshold)
pca_c_names = [f"PCA_{i}" for i in range(n_pcas)]
print(f"PCA column names: {pca_c_names}")
df_train_mod[pca_c_names] = df_train_pca[:,0:n_pcas]
df_test_mod[pca_c_names] = df_test_pca[:,0:n_pcas]


# ## Store Feature Engineering Results
# 
# For parameter optimization we store the modified dataframe and find optimal parameters in a different notebook.

# In[24]:


# Store modified (not resampled)
df_train_mod.to_csv(f"train_mod_v{DATA_VERSION}.csv")
df_test_mod.to_csv(f"test_mod_v{DATA_VERSION}.csv")


# # Model Selection 
# 
# We have a few potential models that can be utilized to solve this classification problem
# 
# | Approach | Description | Pros | Cons | Python Libraries |
# |----------|-------------|------|------|------------------|
# | **Logistic Regression** | A statistical model that uses a logistic function to model a binary dependent variable. | - Simple and interpretable <br> - Fast training and prediction <br> - Good for linearly separable data | - Not suitable for complex relationships in data <br> - Can't capture non-linear patterns | `scikit-learn`, `LogisticRegression` |
# | **Decision Trees** | A non-parametric supervised learning method used for classification and regression. | - Easy to interpret and visualize <br> - Can handle both numerical and categorical data <br> - No need for feature scaling | - Prone to overfitting <br> - Can be unstable with small changes in data <br> - Biased towards dominant classes | `scikit-learn`, `DecisionTreeClassifier` |
# | **Random Forests** | An ensemble learning method (bagging) that operates by constructing multiple decision trees during training. | - Handles overfitting better than decision trees <br> - Works well with both classification and regression <br> - Can handle large data sets with higher dimensionality | - More complex and less interpretable than decision trees <br> - Longer training time <br> - Can be memory intensive | `scikit-learn`, `RandomForestClassifier` |
# | **Boosting** | (Not really a model but an ensembling method) Applies weak models sequentially, focusing on errors of previous models. Includes AdaBoost, Gradient Boosting, etc. | - Often provides high accuracy <br> - Handles different types of data <br> - Offers feature importance scores | - Prone to overfitting without proper tuning <br> - Time-consuming to train <br> - Less interpretable compared to simpler models | `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost` |
# | **Support Vector Machines (SVM)** | Supervised learning models with associated learning algorithms used for classification and regression. Divides the feature space with *separating hyperplanes*. | - Effective in high dimensional spaces <br> - Works well with clear (linear) margin of separation <br> - Different Kernel functions can be specified for the decision function | - Not suitable for large data sets <br> - Sensitive to noisy data <br> - Requires careful choice of kernel and regularization term | `scikit-learn`, `SVC` |
# | **Naive Bayes** | A set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features. | - Simple and fast <br> - Performs well with categorical input variables <br> - Good for large data sets | - Based on the assumption of independent predictors, which is rarely true <br> - Not good for regression tasks | `scikit-learn`, `GaussianNB`, `MultinomialNB` |
# | **K-Nearest Neighbors (KNN)** | A non-parametric method used for classification and regression. | - Simple and easy to implement <br> - No training period <br> - Naturally handles multi-class cases | - Slow on large data sets <br> - Sensitive to irrelevant features and scale of data <br> - Requires memory for all training data | `scikit-learn`, `KNeighborsClassifier` |
# | **Neural Networks/Deep Learning** | A set of algorithms, modeled loosely after the human brain, designed to recognize patterns. | - Very powerful and flexible <br> - Can model complex non-linear relationships <br> - Good for large data sets and high-dimensional data | - Requires a lot of data <br> - Computationally intensive <br> - Prone to overfitting and requires tuning | `tensorflow`, `keras`, `pytorch` |
# 
# > We will run a selection of the above models and will check their performance on train and validation sets. The optimal model parameterization is not performed yet!

# ## Feature Selection & Parameter Tuning
# 
# Parameter tuning is done in another notebook.
# We do not want all engineered features in our models. Therefore, we only select the potential interesting ones.

# In[25]:


#from sklearn.feature_selection import RFECV

#df_feat_sel_train = df_train_mod
#feat_sel_model = xgb.XGBClassifier()
#rfe = RFECV(estimator=feat_sel_model, min_features_to_select=1, step=1, n_jobs=-1, verbose=1)

# Preliminary, should be done of course in a train-val loop
#rfe.fit(df_feat_sel_train.drop(LABEL, axis=1), df_feat_sel_train[LABEL])

#print("Feature Ranking: ", rfe.ranking_)
#rfe.transform(df_feat_sel_train.drop(LABEL, axis=1))

#sel_feats = df_feat_sel_train.drop(LABEL, axis=1).columns[rfe.support_].values
#print(len(sel_feats))
#print(sel_feats.tolist())


# In[26]:


#import statsmodels.api as sm

#feat_check_model = sm.MNLogit(df_train_mod[LABEL], df_train_mod[['Platelets', 'Copper', 'Alk_Phos', 'Diagnosis_Date', 'SGOT', 'Age'] + pca_c_names])
#res = feat_check_model.fit()
#print(res.summary())


# In[27]:


# All
# FEATS = df_train_mod.drop(LABEL, axis=1).columns.tolist()
# Some
FEATS = ['Platelets', 'Copper', 'Alk_Phos', 'Diagnosis_Date', 'SGOT', 'Age', 'N_Days', 'Cholesterol', 
         'Tryglicerides', 'Albumin', 'Bilirubin', 'Prothrombin', 'Symptom_Score', 'Stage', 'Drug', 
         'Hepatomegaly', 'Spiders', 'Sex', 'Edema_N', 'Edema_S', 'Edema_Y']
# FEATS = FEATS + pca_c_names
# FEATS = sel_feats.tolist()
print(f"Number of feats: {len(FEATS)}")
print(f"Features used: {FEATS}")

# Local optuna optimization test (cross val score: .4129666179319006), 21 FEATS
xgb_params = {'objective': 'multi_logloss', 'early_stopping_rounds': 50, 'max_depth': 9, 'min_child_weight': 8, 'learning_rate': 0.0337716365315986, 'n_estimators': 733, 'subsample': 0.6927955384688348, 'colsample_bytree': 0.1234702658812108, 'reg_alpha': 0.18561628377665318, 'reg_lambda': 0.5565488299127089, 'random_state': 42}


# In[28]:


# Just use the selected features
df_train_final = df_train_mod[FEATS + [LABEL]]
df_test_final = df_test_mod[FEATS]


# ## Model Selection

# In[29]:


def validate_models(models: list[dict],
                    data: pd.DataFrame, 
                    label=LABEL,
                    n_splits=5,
                    n_repeats=1,
                    seed=43):
    """Run models and test them on validation sets. The optimal parameters 
    should be retrieved from previous runs e.g. GridSearchCV etc."""
    
    # TODO: the model dicts should contain the FEATS (since different FEATS should be used)
    
    train_scores, val_scores = {}, {}
    
    pbar = tqdm(models)
    for model in pbar:
        
        # Model needs to be a dict (before tuple) since I need a mutable datatype
        # to insert the average validation score in the end
        model_str = model["name"]
        model_est = model["model"]
        model_feats = model["feats"]
        
        pbar.set_description(f"Processing {model_str}...")
        
        train_scores[model_str] = []
        val_scores[model_str] = []
    
        # I think I should drop the seed when I blend the models together
        # -> they will be trained on different datasets
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

        for i, (train_idx, val_idx) in enumerate(skf.split(data[model_feats], data[label])):
            pbar.set_postfix_str(f"Fold {i+1}/{n_splits}")
            
            X_train, y_train = data[model_feats].loc[train_idx], data[label].loc[train_idx]
            X_val, y_val = data[model_feats].loc[val_idx], data[label].loc[val_idx]
            
            if model_str in ["lgb_cl"]:
                callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
                model_est.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif model_str in ["xgb_cl", "cat_cl"]:
                model_est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            elif model_str in ["voting_clf"]:
                pass # TODO: find a solution
            else:
                model_est.fit(X_train, y_train)
                
            train_preds = model_est.predict_proba(X_train[model_feats])
            valid_preds = model_est.predict_proba(X_val[model_feats])
            train_score = log_loss(y_train, train_preds)
            val_score = log_loss(y_val, valid_preds)
            train_scores[model_str].append(train_score)
            val_scores[model_str].append(val_score)
            
            #print(f"{model_str} | Fold {i + 1} | " +
            #      f"Train log_loss: {round(train_score, 4)} | " +
            #      f"Valid log_loss: {round(val_score, 4)}")
        
        model["avg_val_score"] = np.mean(val_scores[model_str])
            
    return models, pd.DataFrame(train_scores), pd.DataFrame(val_scores)


# Let's just run a couple of models in our `validate_models` function.

# In[30]:


get_ipython().run_cell_magic('time', '', '\nxgb_cl = xgb.XGBClassifier(**xgb_params)\n\nmodels = [\n    {"name": "xgb_cl", "model": xgb_cl, "feats": FEATS},\n]\n\nmodels, train_scores, val_scores = validate_models(models=models, \n                                                   data=df_train_final, \n                                                   n_splits=10,\n                                                   n_repeats=1)\n')


# Let's inspect the performances on the train and validation sets. 

# In[31]:


fig, axes = plt.subplots(1, 2, figsize=(15, 6))

eps = .05
hl = .39
min_score = train_scores.min().min()-eps
max_score = val_scores.max().max()+eps

def calculate_ticks(min_score, max_score, num_ticks=10):
    return np.linspace(min_score, max_score, num_ticks)

ticks = calculate_ticks(min_score, max_score)

_ = sns.boxplot(train_scores, ax=axes[0])
_ = axes[0].set_title('Train Scores')
_ = axes[0].set_ylim(min_score, max_score)
_ = axes[0].set_yticks(ticks)
_ = axes[0].yaxis.grid(True)
_ = axes[0].axhline(y=hl, color='r', linestyle='--', lw=.7)
_ = axes[0].text(-1, hl, f"{hl}", c="red")

_ = sns.boxplot(val_scores, ax=axes[1])
_ = axes[1].set_title('Validation Scores')
_ = axes[1].set_ylim(min_score, max_score)
_ = axes[1].set_yticks(ticks)
_ = axes[1].yaxis.grid(True)
_ = axes[1].axhline(y=hl, color='r', linestyle='--', lw=.7)
_ = axes[1].text(-1, hl, f"{hl}", c="red")


# In[32]:


# Print results in DataFrame
model_res = pd.concat([train_scores.describe(), val_scores.describe()], axis=1)
model_res.columns = ['Train', 'Validation']
print(model_res)

# Plot results as lineplot
_ = sns.lineplot(pd.concat([train_scores, val_scores], keys=["Train Score", "Validation Score"], axis=1), markers=True, dashes=False)
plt.axhline(y=0.39, color='r', linestyle='--', lw=.5)
plt.title('Train vs Validation Scores')
plt.xlabel('Index')
plt.ylabel('Scores')
plt.show()


# **Oversampling, the `CL` cases resulted in an improvement of the training and validation score... However, this is an artifact since I have more observations that can now be classified correctly... The validation set for training should reflect the original distribution of labels!**
# 
# *From the discussion [here](https://www.kaggle.com/competitions/playground-series-s3e26/discussion/459897) and Ravi's kind [reponse](https://www.kaggle.com/competitions/playground-series-s3e26/discussion/459897#2552015) I follow his suggestion and I do not refit the model on the whole train dataset again.*

# # Ensemble Methods (Not used any more)!
# 
# 
# 
# | Ensemble Method | Description | Pros | Cons | Python Libraries |
# |-----------------|-------------|------|------|------------------|
# | **Voting Classifier** | Combines predictions from multiple models. Hard voting uses mode of predictions, while soft voting uses average probabilities. | - Simple to implement <br> - Reduces variance and improves performance <br> - Effective for combining models with diverse strengths | - Limited by the best individual model <br> - Hard voting may ignore probability estimates <br> - Requires careful selection of models to combine | `scikit-learn` (`VotingClassifier`) |
# | **Bagging** | Trains the same algorithm on different subsets of the data and then averages the predictions (e.g. RandomForests). | - Reduces variance; avoids overfitting <br> - Effective with unstable models like decision trees <br> - Improves model accuracy without increasing complexity | - Models may become very similar, limiting benefits <br> - Less effective if individual models are biased <br> - Computationally intensive for large datasets | `scikit-learn` (`BaggingClassifier`) |
# | **Stacking** | Trains a new model to combine predictions of several base models. See [here](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/) for an example.| - Leverages strengths of each model <br> - Can outperform individual models <br> - Flexible in choice of base and meta-models | - Complex to implement; risk of overfitting <br> - Training can be time-consuming <br> - Model selection and tuning can be challenging | `scikit-learn` (`StackingClassifier`) |
# | **Blending** | Combines predictions using weighted average or other simple methods. | - Simpler than stacking; less risk of overfitting <br> - Quick to implement and run <br> - Effective when models have similar performance | - Might not capture complex patterns as well as stacking <br> - Weaker than stacking or boosting in handling diverse data types <br> - Requires a good mix of models for effective blending | Manual implementation with `numpy`, `pandas` |
# | **Custom Ensemble Methods** | Develop a tailored ensemble method for specific problems. | - Customization can lead to better performance on specific tasks <br> - Flexibility to address unique aspects of the data <br> - Potential to outperform standard methods | - Requires deep understanding of problem and models <br> - Can be time-consuming to develop and test <br> - Risk of overfitting if not carefully designed | `numpy`, `pandas`, `scikit-learn`, `tensorflow`, etc. |
# 
# 
# 
# 

# In[33]:


class MyAvgVoting(BaseEstimator, ClassifierMixin):
    """A basic voting method that just averages all estimator predictions and 
    predicts the class with the highest vote."""
    def __init__(self, estimators, weighted=False):
        self.estimators = estimators
        # Whether to average according to validation scores
        self.weighted = weighted

    def fit(self, X, y):
        for _, est in self.estimators:
            est["model"].fit(X, y)
        return self
    
    def create_avg_prob_predictions(self, X):
        predictions = np.array([est["model"].predict_proba(X) for est in self.estimators])
        if self.weighted:
            # Note: we need the inverse of the val_score since lower values are "better"
            weights = [{"name": est["name"], "value": 1/est["avg_val_score"]} for est in self.estimators]
            print(f"Weights are:\n{pd.DataFrame(weights)}")
            return np.average(predictions, axis=0, weights=[w["value"] for w in weights])
        return np.average(predictions, axis=0)
            
    def predict(self, X):
        avg_predictions = self.create_avg_prob_predictions(X)
        return np.argmax(avg_predictions, axis=1)

    def predict_proba(self, X):
        avg_predictions = self.create_avg_prob_predictions(X)
        return avg_predictions

    def score(self, X, y):
        pass

voting_ests = models
voting_clf = MyAvgVoting(voting_ests, weighted=False) 
# no fitting needed


# In[34]:


model_fin = models[0]['model']


# # Model Analysis
# 
# We inspect the predictions on the train data and check how accurate we classify `Status`.
# 
# **TBD: elaborate more...**

# In[35]:


y_hat = model_fin.predict(df_train_final[FEATS])
ConfusionMatrixDisplay.from_predictions(df_train_final[LABEL], y_hat, normalize='true', display_labels=label_encoder.classes_)
plt.show()


# Let's analyze the feature importances from `XGBClassifier` and `LGBMClassifier`.

# In[36]:


# Creating Pandas Series for feature importances
xgb_feat_importances = pd.Series(xgb_cl.feature_importances_, index=df_train_final[FEATS].columns)

# Plotting both feature importances in subplots
fig, axes = plt.subplots(1, 1, figsize=(15, 6))

xgb_feat_importances.nlargest(20).plot(kind='barh', title='XGB Feature Importances')

plt.tight_layout()
plt.show()
print(f"Total sorted XGBClassifier importances: {xgb_feat_importances.nlargest(99).index.tolist()}")


# **-> It seems like (when we want to utilize / ensemble both approaches), that we should use different features for each model**

# # Submission
# 

# In[37]:


y_test_hat = model_fin.predict_proba(df_test_final[FEATS])
assert y_test_hat.shape == (df_test.shape[0], 3)


# In[38]:


submission_labels = ["Status_C", "Status_CL", "Status_D"]

sub = pd.DataFrame(
    {"id": test_IDs, **dict(zip(submission_labels, y_test_hat.T))}
)
sub.head()


# # Submission Ensemble

# In[39]:


# Reading other DataFrames
sub_list = [
    sub,
    #pd.read_csv('/kaggle/input/ps3e26-cirrhosis-survial-prediction-multiclass/submission.csv'),
    #pd.read_csv('/kaggle/input/multi-class-prediction-of-cirrhosis-outcomes/submission.csv'),
    #pd.read_csv('/kaggle/input/ps3e25-cirrhosis-multi-class-solution/submission.csv')
    pd.read_csv('/kaggle/input/other-notebooks/Multi-Class Prediction of Cirrhosis Outcomes_V16.csv'),
    pd.read_csv('/kaggle/input/other-notebooks/PS3E25 -  Cirrhosis Multi-Class Solution_V10.csv'),
    pd.read_csv('/kaggle/input/other-notebooks/PS3E26  Cirrhosis Survial Prediction  Multiclass_V8.csv')
]

# Calculate the mean across all DataFrames
sub_ensemble = pd.concat(sub_list).groupby('id')[submission_labels].mean()

# Normalizing the columns to sum to 1
sub_ensemble[submission_labels] = sub_ensemble[submission_labels].div(sub_ensemble[submission_labels].sum(axis=1), axis=0).fillna(0)
sub_ensemble = sub_ensemble.reset_index()

# Saving and displaying the result
sub_ensemble.to_csv('submission.csv', index=False)
sub_ensemble.head()

