#!/usr/bin/env python
# coding: utf-8

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Introduction</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Hello Kagglers, I just wanted to share with you another implementation of a trainer class like the one I did in my other Titanic notebook <a href="https://www.kaggle.com/code/maxdiazbattan/titanic-top-5-competition-class-v1-blending">[link]</a>, I really like the final result. Greetings to all! </p>
# 

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Table of contents</div>
# - [1- Libraries](#libraries)
# - [2- Data loading](#data-load)
# - [3- Folds creation](#folds)
# - [4- Exploratory data analysis (EDA)](#eda)
# - [5- Feature engineering](#fe)
# - [6- Feature selection](#fs)
# - [7- Modeling](#modeling)
# - [8- Blending](#blend)

# <a id="libraries"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 1 | Libraries</div>

# In[1]:


get_ipython().system('pip install feature_engine -q')


# In[2]:


# Data handling
import numpy as np 
import pandas as pd 

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn import model_selection, preprocessing, pipeline, metrics, impute, compose

# Encoders & Feature engine
from category_encoders import TargetEncoder, CatBoostEncoder,GLMMEncoder, CountEncoder
from feature_engine.encoding import *
from feature_engine.outliers import *

# Feature selection
import eli5
from eli5.sklearn import PermutationImportance

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn import linear_model, neighbors, naive_bayes, svm

# Remove warning
import warnings
warnings.filterwarnings('ignore')

# Viz styling
sns.set_style('darkgrid')
plt.style.use('ggplot')


# <a id="data-load"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 2 | Data loading</div>

# In[3]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
submission = pd.read_csv('../input/titanic/gender_submission.csv')


# <a id="folds"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 3 | Folds creation</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Usually always it's recommended split the data in folds first </p>
# 

# In[4]:


pd.DataFrame(train.Survived.value_counts()).T


# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Because it's an imbalance problem I'm going to use stratified k fold</p>

# In[5]:


skf= model_selection.StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
train['kfold'] = -1
def kfold (df):
    df = df.copy()
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X = df, y=df.Survived)):
        df.loc[valid_idx, 'kfold'] = int(fold)    
    return df


# In[6]:


train = kfold(train)


# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# For a better analysis I'm going to concat the 2 dataframes</p>

# In[7]:


test['Survived'] = -1


# In[8]:


combined_df = pd.concat([train,test], axis = 0)


# <a id="eda"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 4 | Exploratory data analysis (EDA)</div>

# In[9]:


combined_df.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')


# In[10]:


features = [feature for feature in train.columns if feature not in ('PassengerId','Survived','kfold')]
categoric_features = [feature for feature in train[features].columns if train[feature].dtype =='O']
numeric_features = [feature for feature in train[features].columns if feature not in categoric_features]


# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Checking the categories length</p>

# In[11]:


{feature: len(train[feature].unique()) for feature in train.select_dtypes('object')}


# In[12]:


{feature: len(test[feature].unique()) for feature in test.select_dtypes('object')}


# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Besides of this categories, we have also Pclass</p>

# In[13]:


sns.set_style ('darkgrid')
sns.palplot(sns.color_palette('rainbow'))
sns.set_palette('rainbow')


# In[14]:


plt.figure (figsize = (20,15))
for i, feature in enumerate (numeric_features):
    plt.subplot (4,2, i*1 + 1 )
    sns.histplot (data = train, x = train[feature], hue='Survived', linewidth=2)


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px ">   
# <b> Insights: </b> We can see how almost half of the first class people survived, as opposed to the third class where less than a third did. Traveling alone gives almost a 50% chance of survival. Age is a bit right skew, and Fare much more, candidate for a log transformation.</p>

# In[15]:


plt.figure (figsize = (20,15))
for i, feature in enumerate (numeric_features):
    plt.subplot (4,2, i*1 + 1 )
    sns.boxplot (data = train, x = train[feature], linewidth=2)


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> There is some outliers in the data in the columns Age, Sibsp, Parch, and Fare.</p>

# In[16]:


missing = (combined_df[features].isna().mean() * 100).round(2).sort_values(ascending=False)


# In[17]:


pd.DataFrame(missing).rename(columns={0:'Missing %'}).style.bar(subset=['Missing %'], color='#CC0000')


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; border: ">   
# <b> Insights: </b> Cabin it's the feature with most missing values, almost 78% of the data it's missing. </p>

# In[18]:


plt.figure(figsize=(12,6))
sns.countplot(x = train["Survived"], hue = "Sex", data=train, edgecolor='black',linewidth=2)
plt.ylabel('Number of people' ,weight='bold', size=13)
plt.xlabel('Survived' ,weight='bold', size=13)
plt.title('Survival count by Gender',weight='bold', size=14);


# In[19]:


plt.figure (figsize = (12,6))
sns.barplot(x = 'Sex', y ='Survived', data = train, edgecolor='black',linewidth=2);
plt.ylabel('Survival Probability' ,weight='bold', size=13)
plt.xlabel('Sex' ,weight='bold', size=13)
plt.title('Survival Probability by Gender',weight='bold', size=14);


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> By far women are more likely to survive. </p>

# In[20]:


plt.figure(figsize=(12,6))
sns.countplot(x = train["Survived"], hue = "Pclass", data=train, edgecolor='black',linewidth=2)
plt.ylabel('Number of people',weight='bold', size=13)
plt.title('Survival count by Passenger class',weight='bold', size=14);


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> Rich people also has a greater opportunity to survive. </p>

# In[21]:


plt.figure(figsize=(12,6))
sns.countplot(x = train["Survived"], hue = "SibSp", data=train, edgecolor='black',linewidth=2)
plt.ylabel('Number of people',weight='bold', size=13)
plt.title('Survival count by sibiling and spouses',weight='bold', size=14);


# In[22]:


plt.figure(figsize=(12,6))
sns.countplot(x = train["Survived"], hue = "Parch", data=train, edgecolor='black',linewidth=2)
plt.xlabel('Survived',weight='bold', size=13)
plt.ylabel('Number of people',weight='bold', size=13)
plt.title('Survival count by Parch',weight='bold', size=14);


# In[23]:


plt.figure(figsize=(12,6))
sns.swarmplot(data=train, x=(train['SibSp'] + train['Parch']), y=train['Fare'], hue=train['Survived'])
plt.xlabel('Family Size',weight='bold', size=13)
plt.ylabel('Fare amount',weight='bold', size=13)
plt.title('Survival by Fare & Family Size ',weight='bold', size=14);


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> Travel alone or just with one Sibsp or Patch gives the highest chance to survive. This may also be due to the fact that the smaller families are the ones with more first- or second-class people. </p>

# In[24]:


plt.figure(figsize=(12,6))
sns.boxplot(x = combined_df["Pclass"], y = combined_df["Age"], hue='Sex', data = combined_df, linewidth=2)
plt.xlabel('Pclass',weight='bold', size=13)
plt.ylabel('Age',weight='bold', size=13)
plt.title('Age by Passenger class',weight='bold', size=14);


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> The "oldest" people are the richest, and also men are elderly compared to women. </p>

# <a id="fe"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 5 | Feature engineering </div>

# In[25]:


(train.groupby('Ticket')['Survived'].sum() / train.groupby('Ticket')['Survived'].count()).to_frame().rename(columns={'Survived':'SR_Ticket'}).reset_index()


# In[26]:


def preprocessing_inputs (df):
    df = df.copy()
    
    # Feature Engineering:
    
    # Name
    # Extracting the Name feature and creating a new feature just with the title 
#     ds['TitlePrefix'] = ds['TitlePrefix'].replace(['Capt', 'Dr', 'Major', 'Rev', 'Col'], 'Officials')
#     ds['TitlePrefix'] = ds['TitlePrefix'].replace(['Lady', 'Countess', 'Don', 'Sir', 'Jonkheer', 'Dona'], 'High_Class')
#     ds['TitlePrefix'] = ds['TitlePrefix'].replace(['Mlle', 'Ms'], 'Miss')
#     ds['TitlePrefix'] = ds['TitlePrefix'].replace(['Mme'], 'Mrs')

    
    df['Title'] = df['Name'].apply(lambda x: x.split('.')[0]).apply(lambda x : x.split(',')[1]).str.strip().replace({'Mrs':'Mr','Mlle':'Miss','Ms':'Miss','Mme':'Miss','Dona':'Miss','Lady':'Miss'})
    # Last name extraction 
    df['LastName'] = df['Name'].str.extract('^(.+?),', expand = False)
    
    # Age
    # Creating a flag if the Age value is null
    df['AgeFlag'] = df['Age'].map(lambda x: 1 if pd.isnull(x) else 0)
    # Filling the NA values
    median_age = df.groupby(by=['Pclass','Sex'])['Age'].median().to_frame().rename(columns={'Age':'MedianAge'}).reset_index()
    df = df.merge(median_age, on=['Pclass','Sex'], how='left')
    # Creating bins from the Age feature
    df['AgeBin'] = pd.cut(df['MedianAge'].astype(int), 5, labels=False)
    median_age_title = df.groupby(by='Title')['Age'].median().to_frame().rename(columns={'Age':'MedianAgeTitle'}).reset_index()
    # Calculating the median Age by Tittle
    df = df.merge(median_age_title, on=['Title'], how='left')
    df['MedianAgeTitle'] = df['MedianAgeTitle'].fillna(df['MedianAgeTitle'].mode()[0])
    
    # SibSp & Parch
    # Math transform on Sib and Parch
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    
    # Ticket
    # Extracting the ticket number
    df['TicketNumber'] = df['Ticket'].apply(lambda x: x.split(' ')).apply(lambda x : x[1] if len (x) > 1 else x[0]).str.strip()
    df['TicketNumber'].replace({'LINE': '-1', 'Basle': '-2', 'L':'-3', 'B':'-4', '2.':'2'}, inplace=True)
    # Creating a flag if the Ticket value is null
    df['TicketFlag'] = df['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
    # Extracting the first letter on the ticket feature
    df['TicketCode'] = df['Ticket'].apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','') if len(x.split(' ')[:-1]) > 0 else 'None')
    sr_ticket = (df.groupby('Ticket')['Survived'].sum() / df.groupby('Ticket')['Survived'].count()).to_frame().rename(columns={'Survived':'SR_Ticket'}).reset_index()
    df = df.merge(sr_ticket, on=['Ticket'], how='left')
    
    # Fare
    # Calculating the median Fare by Pclass, Sex, Family # Ver de separar el fare por Pclass, Sex, Family
    median_fare = df.groupby(by=['Pclass','Sex','Family'])['Fare'].median().to_frame().rename(columns={'Fare':'MedianFare'}).reset_index()
    df = df.merge(median_fare, on=['Pclass','Sex','Family'], how='left')
    # Creating a feature by splitting the Fare in 3 different classes
    df['SocialClassByFare'] = df['Fare'].apply(lambda x : 'Rich' if x > df['Fare'].quantile(0.75) else ( 'Poor' if x < df['Fare'].quantile(0.25) else 'Midd' ))
    # Creating bins from the Age feature
    df['FareBin'] = pd.qcut(df['Fare'], 3, labels=False)
    
    # Cabin
    # Extracting the first letter on the Cabin feature
    cabin_mode = df.groupby(by=['Pclass','Sex'])['Cabin'].agg(lambda x:x.value_counts().index[0]).to_frame().rename(columns={'Cabin':'CabinMode'}).reset_index()
    df = df.merge(cabin_mode, on=['Pclass','Sex'], how='left')
    df['CabinCode'] = df['Cabin'].apply(lambda x : str(x)).apply(lambda x: 'U' if x == 'nan' else x[0])
    # Extracting the length of the Cabin feature
    df['CabinLen'] = df['CabinMode'].apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
    # Creating a flag if the Cabin value is null
    df['CabinFlag'] = df['Cabin'].map(lambda x: 1 if pd.isnull(x) else 0)
    # Calculating the mode for Cabin
    median_fare_cabin = df.groupby(by='CabinMode')['Fare'].median().to_frame().rename(columns={'Fare':'MedianFareCabin'}).reset_index()
    df = df.merge(median_fare_cabin, on=['CabinMode'], how='left')
    
    # Log transform fare
    df['Fare'] = np.log(df['Fare']+1)
    
    # Embarked
    # Replacing the null values on Embarked with the mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Split the dataframe
    train = df.query("Survived != -1").copy()
    train['Survived'] = train['Survived'].astype(int)
    
    test = df.query("Survived == -1").copy()
    test.drop(['Survived', 'kfold'], axis = 1, inplace=True)
                
    return train, test


# In[27]:


train_df, test_df = preprocessing_inputs(combined_df)


# <a id="fs"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 6 | Feature selection</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
#  It's a very small dataset, so this feature selection part it's not so important, but I think it's a good practice to apply it for educational purposes anyway. </p>

# <p style="font-size:18px; font-family:verdana; line-height: 1.7em; margin-left:20px">  
# <b> Permutation feature importance </b> </p>

# In[28]:


features = [feature for feature in train_df.columns if feature not in ['PassengerId','Name','Cabin','Ticket','Survived','kfold']]
numeric_features = [feature for feature in features if train_df[feature].dtype !='O']
categoric_features = [feature for feature in features if feature not in numeric_features]


# In[29]:


selector = cb.CatBoostClassifier (n_estimators=1000, random_state=0, verbose=0)


# In[30]:


def eli5_feature_selection(df, subset, features, numeric_features, categoric_features, target, folds, models, random_state=42):
    
    list_results = []
    
    df = df.sample(frac=subset, random_state=random_state).reset_index()

    for fold in range(folds):
        print(f' Fold number = {fold}')
        X_train = df[df.kfold != fold].reset_index(drop=True)
        X_valid = df[df.kfold == fold].reset_index(drop=True)

        y_train = X_train[target]
        y_valid = X_valid[target]
               
        # Encoding
            #  encoder
        encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[categoric_features] = encoder.fit_transform(X_train[categoric_features])
        X_valid[categoric_features] = encoder.transform(X_valid[categoric_features])
                
        # Preprocessed's dfs
        X_train = X_train[numeric_features+categoric_features]
        X_valid = X_valid[numeric_features+categoric_features]
        
        model = models.fit(X_train, y_train)
        
        perm = PermutationImportance(model, random_state=random_state).fit(X_valid, y_valid)
        weights = eli5.show_weights(perm, top=len(X_valid.columns), feature_names=X_valid.columns.tolist())
        result = pd.read_html(weights.data)[0]
        eli_features = pd.DataFrame(result)
        eli_features.Weight = eli_features.Weight.str.split(' ').apply(lambda x: x[0]).astype('float32')
        eli_features.set_index('Feature', inplace=True)
        
        list_results.append(eli_features)

    results = pd.concat(list_results, axis=1)
    results = results.mean(1)
    
    return results


# In[31]:


results = eli5_feature_selection(train_df, 1., features, numeric_features, categoric_features, 'Survived', 5, selector)


# In[32]:


pi_features = results[results>0].sort_values(ascending=False)#.index.to_list()


# In[33]:


pi_features


# In[34]:


results_df = pd.DataFrame(results).reset_index().rename(columns={0:'PI_value'}).sort_values(by='PI_value',ascending=False)


# In[35]:


plt.figure(figsize=(15,5))
sns.barplot(data=results_df, y='Feature',  x='PI_value');


# In[36]:


len(pi_features)


# <div class="alert alert-info" style="border-radius:5px; font-size:15px; font-family:verdana; line-height: 1.7em">
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Insights: </b> We can see how Sex, Cabin, Ticket, Fare, Embarked, Pclass, Tittle, are the most important features. </p>

# <a id="modeling"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 7 | Modeling</div>

# 
# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Trainer class </b> </p>

# In[37]:


class Trainer:
    
    """
    Args:
        - model: Any ML model to train.
        - model_name: The corresponding model name to be used to identify it in the training process.
        - fold: Fold number.
        - model_params: Hyperparameters of the respective model.
    """
    
    def __init__(self, model, model_name, fold):
     
        self.model_ = model
        self.model_name = model_name
        self.fold = fold
        
        self.test_preds = []
              
    def fit(self, xtrain, ytrain, xvalid, yvalid):
        
        """
        Fits an instance of the model for a particular dataset.
        Args:
            - xtrain: Train data.
            - ytrain: Train target.
            - xvalid: Validation data.
            - yvalid: Validation target.
        """
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid
        
        if self.model_name.startswith('B'):
            self.model_.fit(self.xtrain, self.ytrain, early_stopping_rounds=10, eval_set=[(self.xvalid, self.yvalid)],verbose=False)
        if self.model_name.startswith('T'):
            self.model_.fit(self.xtrain, self.ytrain.values.reshape(-1,1), eval_set=[(self.xvalid, self.yvalid.values.reshape(-1,1))])
        else:
            self.model_.fit(self.xtrain, self.ytrain)
        
        return self.model_
        
    def pred_evaluate(self,name, oofs,val_idx):
        
        """
        Makes predictions for each model on the valid data provided.
        Args:
            - name: Model name.
            - oofs: oofs data.
            - val_idx: Validation indicies.

        """
        
        preds_valid = self.model_.predict(self.xvalid)
        score = metrics.accuracy_score(self.yvalid, preds_valid)
        oofs.loc[val_idx, f"{name}_preds"] = preds_valid
        
        print(f'fold = {self.fold} | score = {score:.4f}')
        
        return score, preds_valid, oofs
    
    def blend(self, models, folds, xtest, column_id, features):        
        """
        Makes a blend of the trained models.
        Args:
            - models: Models to blend.
            - xtest: Test data preprocess.
            - column_id: Id.
            - features: training final.
        """
        
        y_test_pred = pd.DataFrame(xtest.loc[:,column_id])    
        X_test = xtest[features]

        models_n = np.array_split(list(models.items()), len(models)/folds)
        for model_folds in models_n:
            model_test_preds = []
            for fold, model in model_folds:
                preds_test = model.predict(X_test)
                model_test_preds.append(preds_test)
                
            # Get the model name
            name =  fold.split('_')[0]
            # Assign mean folds predictions to preds dataframe
            y_test_pred.loc[:,f'{name}_preds'] = np.mean(np.column_stack (model_test_preds), axis=1).astype(int)
        
        return y_test_pred


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Models </b> </p>

# In[38]:


class Models:
    BXGB = xgb.XGBClassifier(random_state=0, objective = 'reg:squarederror')
    BLGBM = lgb.LGBMClassifier(random_state=0)
    BCB = cb.CatBoostClassifier(random_state=0, verbose=False)
    LR = linear_model.LogisticRegression(solver='liblinear')
    RI = linear_model.RidgeClassifier()
    KNN = neighbors.KNeighborsClassifier()
    SVC = svm.SVC()


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Final Features </b> </p>

# In[39]:


features = ['Pclass','Sex', 'Fare', 'Embarked',  'Title', 'LastName', 'AgeFlag', 'MedianAge', 'AgeBin', 'MedianAgeTitle','Family', 
            'TicketNumber', 'TicketFlag', 'MedianFare', 'FareBin','CabinMode', 'CabinCode', 'CabinLen','MedianFareCabin']

# ['Pclass', 'Sex', 'MedianAge', 'Fare', 'Embarked', 'Title', 'AgeBin', 'Family', 'TicketNumber', 'SocialClassByFare', 'FareBin', 'CabinCode','MedianFare','CabinMode']


# In[40]:


categoric_features = [feature for feature in train_df[features] if train_df[feature].dtype =='O']
numeric_features = [feature for feature in train_df[features] if feature not in categoric_features+['PassengerId','kfold','Survived']]  


# In[41]:


ordinal_features = [feature for feature in train_df[categoric_features].columns if len(train_df[feature].unique()) <= 3 and feature not in ['Survived']]
high_card_features = [feature for feature in train_df[categoric_features].columns if len(train_df[feature].unique()) > 3 and feature not in ['Survived','PassengerId']+ordinal_features]


# In[42]:


numeric_features, ordinal_features, high_card_features


# In[43]:


len(features)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Training </b> </p>

# In[44]:


oofs_dfs = []
models_trained = {}

iterator = iter([[getattr(Models, attr), attr] for attr in dir(Models) if not attr.startswith("__")])

for mdls in iterator:

    oofs_scores = []
    
    model = mdls[0]
    name = mdls[1]
       
    oofs = train_df[["PassengerId","Survived", "kfold"]].copy()
    oofs[f"{name}_preds"] = None

    print(f' Model {name}')
    for fold in range(5):
        
        X_train = train_df[train_df.kfold != fold]
        X_valid = train_df[train_df.kfold == fold]
        
        val_idx = X_valid.index.to_list()
        
        y_train = X_train['Survived']
        y_valid = X_valid['Survived']
        
        # Scaling
        scl = preprocessing.RobustScaler()
        X_train[numeric_features] = scl.fit_transform(X_train[numeric_features])
        X_valid[numeric_features] = scl.transform(X_valid[numeric_features])    
                
        # Outliers handlingh
        capper = Winsorizer(capping_method='gaussian', tail='right', fold=3, variables=numeric_features)
        X_train[numeric_features] = capper.fit_transform(X_train[numeric_features])
        X_valid[numeric_features] = capper.transform(X_valid[numeric_features])

        # Encoding
            # OHE
        ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore').fit(X_train[categoric_features])
        encoded_cols = list(ohe.get_feature_names(categoric_features))
    
        X_train[encoded_cols] = ohe.transform(X_train[categoric_features])
        X_valid[encoded_cols] = ohe.transform(X_valid[categoric_features])
        
        # Preprocessed's dfs
        X_train = X_train[numeric_features+encoded_cols]
        X_valid = X_valid[numeric_features+encoded_cols]
        
        # Trainer class initialization
        trainer = Trainer(model=model, model_name=name,fold=fold)
        
        # Fit the trainer
        model_trained = trainer.fit(X_train, y_train, X_valid, y_valid)
        
        # Evaluate
        scores, valid_preds, oofs = trainer.pred_evaluate(name, oofs, val_idx)
        models_trained[f'{name}_{fold}'] = model_trained
        oofs_scores.append(scores)
   
    oofs_dfs.append(oofs)
    print(f' oofs score = {np.mean(oofs_scores):.4f}')
    print()


# In[45]:


final_valid_df = pd.concat(oofs_dfs, axis=1).iloc[:, [0,1,2,3,7,11,15,19,23,27]]


# In[46]:


features_blend =  [feature for feature in final_valid_df.columns if 'preds' in feature]


# In[47]:


final_valid_df.insert(loc = 2, column = 'blend_preds', value = final_valid_df[features_blend].mode(1)[0])


# In[48]:


final_valid_df.sample(5)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# <b> Preprocessing test dataframe </b> </p>

# In[49]:


X_test = test_df.copy()
X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())
X_test['FareBin'] = X_test['FareBin'].fillna(X_test['FareBin'].median())
X_test[numeric_features] = scl.transform(X_test[numeric_features])
X_test[numeric_features] = capper.transform(X_test[numeric_features])
X_test[encoded_cols] = ohe.transform(X_test[categoric_features])


# In[50]:


final_test_df = trainer.blend(models_trained, 5, X_test, 'PassengerId', numeric_features+encoded_cols)


# In[51]:


final_test_df.set_index('PassengerId').mode(1)[0].reset_index().rename(columns={0:'Survived'}).to_csv('submission.csv', index=False)


# In[52]:


final_test_df.sample(5)


# <a id="blend"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 8 | Blending</div>

# In[53]:


def blend(valid_df, test_df, features, target, folds):

    train = valid_df.copy()
    test = test_df.copy()

    scores = []
    valid_preds_lr = {}
    test_preds_lr = []

    for fold in range(folds):

        X_train = train[train.kfold != fold].reset_index(drop=True)
        X_valid = train[train.kfold == fold].reset_index(drop=True)

        X_test = test[features]

        X_valid_ids = X_valid.PassengerId.values.tolist()

        y_train = X_train[target]
        y_valid = X_valid[target]

        X_train = X_train[features]
        X_valid = X_valid[features]

        # Model
        model = linear_model.LogisticRegression()
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        preds_test = model.predict(X_test)

        valid_preds_lr.update(dict(zip(X_valid_ids,preds_valid)))
        test_preds_lr.append(preds_test)

        acc = metrics.accuracy_score(y_valid, preds_valid)
        scores.append(acc)

        print(f' Fold = {fold}, ACC = {acc:.10f}')
    print(f'Mean score = {np.mean(scores):.10f} Std = {np.std(scores):.3f} ')
    
    level_0_valid_preds = pd.DataFrame.from_dict(valid_preds_lr, orient='index').reset_index().rename(columns = {'index':'PassengerId', 0:'lr_pred_0'})
    level_0_valid_preds = submission.copy()
    level_0_valid_preds.Survived = np.mean(np.column_stack (test_preds_lr), axis=1 ).astype(int)
    
    return level_0_valid_preds


# In[54]:


blend_submission = blend(final_valid_df, final_test_df, features_blend, 'Survived', 5)


# In[55]:


blend_submission.Survived.value_counts()


# In[56]:


blend_submission.to_csv('blend_submission.csv', index = False)


# <p style="font-size:20px; font-family:verdana; line-height: 1.7em">   
# Thanks for read my notebook. Greetings! </p>
