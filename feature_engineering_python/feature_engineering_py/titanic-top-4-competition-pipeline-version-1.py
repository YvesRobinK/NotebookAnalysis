#!/usr/bin/env python
# coding: utf-8

# 
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Introduction</div>
# 
# <p style="font-size:16px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Going through Kaggle I have come across amazing notebooks and they gave me ideas to make a trainer class that also works for different competitions. In fact, a tailored version of this class powers my current entry in the Tabular Playground series, where with minimal preprocessing and feature engineering, I'm aiming for the top 4%. I'd love to hear your thoughts on this trainer class and welcome any suggestions! You can also explore my detailed Titanic kernel featuring another trainer class and a comprehensive EDA here: <a href="https://www.kaggle.com/code/maxdiazbattan/titanic-competition-class-v2-updated">[link]</a>. Happy kaggling! </p>
# 

# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> - | Table of contents</div>
# 
# * [1-Libraries](#section-one)
# * [2-Data loading](#section-two)
# * [3-Preprocessing and Feature engineering](#section-three)
# * [4-Training](#section-four)
# * [5-Blending](#section-five)

# <a id="section-one"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 1 | Libraries</div>

# In[1]:


# Data handling
import pandas as pd
import numpy as np

# Sklearn 
from sklearn import model_selection, preprocessing, impute , metrics

# Models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn import linear_model, ensemble

# Remove warnings
import warnings
warnings.filterwarnings('ignore')


# <a id="section-two"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 2 | Data loading</div>

# In[2]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
submission = pd.read_csv('../input/titanic/gender_submission.csv')


# In[3]:


combined_df = pd.concat([train,test], axis = 0)


# <a id="section-three"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 3 | Preprocessing and Feature engineering</div>

# In[4]:


def preprocessing_inputs (df):
    df = df.copy()
    
    # Feature Engineering:
    # Name:
    # Casting the Name feature and creating a new feature just with the title and replacing bad caetgorization with the most representative category
    df['Title'] = df['Name'].apply(lambda x: x.split('.')[0]).apply(lambda x : x.split(',')[1]).str.strip().replace({'Mrs':'Mr','Mlle':'Miss','Ms':'Miss','Mme':'Miss','Dona':'Miss','Lady':'Miss'})
    
    # Last name extraction (Not used)
    df['LastName'] = df['Name'].str.extract('^(.+?),', expand = False)
    
    # PClass: (Not used)
    pclass_g = df.groupby(by=['Pclass'])['Age'].agg(['min','max','count','mean','std','skew'])
    pclass_g.rename(columns={c: f"{c}_pclass" for c in pclass_g.columns.to_list()}, inplace=True)
    pclass_g.reset_index(inplace=True)
    df = df.merge(pclass_g, on='Pclass', how='left')
    
    # Age:
    df['Age'].fillna(df['Age'].median(), inplace = True)
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5, labels=False)
    
    # Sib & Parch
    # Math transform on Sib and Parch & binning
    df['Family'] = df['SibSp'] + df['Parch'] + 1
    
    # Ticket:
    # Ticket number
    df['TicketNumber'] = df['Ticket'].apply(lambda x: x.split(' ')).apply(lambda x : x[1] if len (x) > 1 else x[0]).str.strip()
    df['TicketNumber'].replace({'LINE': '-1', 'Basle': '-2', 'L':'-3', 'B':'-4', '2.':'2'}, inplace=True)
    
    # Fare:
    df['SocialClassByFare'] = df['Fare'].apply(lambda x : 'Rich' if x > df['Fare'].quantile(0.75) else ( 'Poor' if x < df['Fare'].quantile(0.25) else 'Midd' ))
    df['FareBins'] = pd.qcut(df['Fare'], 3, labels=False)
    
    # Cabin:
    df['CabinCode'] = df['Cabin'].apply(lambda x : str(x)).apply(lambda x: 'U' if x == 'nan' else x[0])
    
    # Embarked:
    df['Embarked'] = df['Embarked'].apply(lambda x : str(x)).apply(lambda x: 'U' if x == 'nan' else x)
    
    # Split the dataframe
    train = df.query("Survived == Survived").copy()
    train['Survived'] = train['Survived'].astype(int)
    
    test = df.query("Survived != Survived").copy()
    test.drop(['Survived'], axis = 1, inplace=True)
    
    return train, test


# In[5]:


train_df, test_df = preprocessing_inputs(combined_df)


# <a id="section-four"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 4 | Trainer class and Modeling </div>

# In[6]:


class ModelTrain():
            
    """
    Trainer class which is responsible for imputing, encoding, scaling and model training.
    Args:
        - train: Train dataframe.
        - test: Test dataframe.
        - sub: Submission dataframe.
        - n_splits: Number of folds.
        - num_feats: Numerical features (list).
        - cat_feats: Categorical features (list).
        - model_name: The corresponding model name to be used to identify it in the training process.
        - model: Model to train.
        - preprocessing: Preprocessing process (boolean).
        - impute_type: Impute type for missing values.
        - encode_type: Encoding type for categorical features.
        - scale_type: Scale or transforming.
    """
    
    def __init__(self, train, test, sub, n_splits, features, num_feats, cat_feats, target, model_name, model,
                 preprocessing=False, impute_type=False, encode_type=False, scale_type=False):
        
        self.train = train
        self.test = test
        self.sub = sub
        self.n_splits = n_splits
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.target = target
        self.model_name = model_name
        self.model = model
        self.preprocessing = preprocessing
        self.impute_type = impute_type
        self.encode_type = encode_type
        self.scale_type = scale_type
        
        self.valid_preds = {}
        self.test_preds = []
        
    def kfold(self):
        
        """
        Folds creation.
        """
            
        n_splits = self.n_splits
        target = self.target
        df = self.train.copy()
        df['kfold'] = -1
               
        skf= model_selection.StratifiedKFold (n_splits=n_splits, shuffle=True, random_state=0)
        for fold, (train_idx, valid_idx) in enumerate (skf.split(X=df, y=df[target].values)):
            df.loc[valid_idx,'kfold'] = fold
        return df
    
    def imputer_(self, xtrain, xvalid, xtest):
        
        """
        Impute the missing values.
        Args:
            - xtrain: Train dataframe.
            - xvalid: Validation dataframe.
            - xtest: Test dataframe.
        """
            
        self.xtrain = xtrain
        self.xvalid = xvalid
        self.xtest = xtest
        
        num_feats = self.num_feats

        if self.impute_type == 'SI':
            si = impute.SimpleImputer()
            self.xtrain[num_feats] = si.fit_transform(self.xtrain[num_feats])
            self.xvalid[num_feats] = si.transform(self.xvalid[num_feats])
            self.xtest[num_feats] = si.transform(self.xtest[num_feats])
            return self.xtrain[num_feats], self.xvalid[num_feats], self.xtest[num_feats]
        
        elif self.impute_type == 'KNN':
            knn = impute.KNNImputer()
            self.xtrain[num_feats] = knn.fit_transform(self.xtrain[num_feats])
            self.xvalid[num_feats] = knn.transform(self.xvalid[num_feats])
            self.xtest[num_feats] = knn.transform(self.xtest[num_feats])
            return self.xtrain[num_feats], self.xvalid[num_feats], self.xtest[num_feats]
        
        else:
            raise Exception ('Impute type not supported, supported types SI or KNN.')
    
    def encoder_(self, xtrain, xvalid, xtest):
        
        """
        Encode categorical values.
        Args:
            - xtrain: Train dataframe.
            - xvalid: Validation dataframe.
            - xtest: Test dataframe.
        """
        
        self.xtrain = xtrain
        self.xvalid = xvalid
        self.xtest = xtest
        cat_feats = self.cat_feats
        
        if self.encode_type == 'OHE':
            ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
            ohe.fit(self.xtrain[cat_feats])
            encoded_cols = list(ohe.get_feature_names(cat_feats))
            self.xtrain[encoded_cols] = ohe.fit_transform(self.xtrain[cat_feats].fillna('-9999'))
            self.xvalid[encoded_cols] = ohe.transform(self.xvalid[cat_feats].fillna('-9999'))
            self.xtest[encoded_cols] = ohe.transform(self.xtest[cat_feats].fillna('-9999'))
            return self.xtrain, self.xvalid, self.xtest, encoded_cols
        
        elif self.encode_type == 'LBL':
            lbl_ = preprocessing.LabelEncoder()
            encoded_cols = list(self.train[cat_feats].columns)
            for c in encoded_cols:
                self.xtrain.loc[:, c] = lbl_.fit_transform(self.xtrain[c].fillna('-9999'))
                self.xvalid.loc[:, c] = lbl_.transform(self.xvalid[c].fillna('-9999'))
                self.xtest.loc[:, c] = lbl_.transform(self.xtest[c].fillna('-9999'))
            return self.xtrain, self.xvalid, self.xtest, encoded_cols
        
        elif self.encode_type == 'ORD':
            # Works with sklearn v 0.24 by setting the handle_unknown parameter
            ord_ = preprocessing.OrdinalEncoder()
            encoded_cols = list(self.train[cat_feats].columns)
            self.xtrain[encoded_cols] = ord_.fit_transform(self.xtrain[cat_feats].fillna('-9999'))
            self.xvalid[encoded_cols] = ord_.transform(self.xvalid[cat_feats].fillna('-9999'))
            self.xtest[encoded_cols] = ord_.transform(self.xtest[cat_feats].fillna('-9999'))
            return self.xtrain, self.xvalid, self.xtest, encoded_cols
        
        else:
            raise Exception ('Encoded type not supported, supported types OHE, LBL, ORD.')
            
    def scaler_(self, xtrain, xvalid, xtest):
        
        """
        Scale the numerical values.
        Args:
            - xtrain: Train dataframe.
            - xvalid: Validation dataframe.
            - xtest: Test dataframe.
        """
        
        self.xtrain = xtrain
        self.xvalid = xvalid
        self.xtest = xtest
        num_feats = self.num_feats
        
        if self.scale_type == 'STD':
            std = preprocessing.StandardScaler()
            self.xtrain[num_feats] = std.fit_transform(self.xtrain[num_feats])
            self.xvalid[num_feats] = std.transform(self.xvalid[num_feats])
            self.xtest[num_feats] = std.transform(self.xtest[num_feats])
            return self.xtrain[num_feats], self.xvalid[num_feats], self.xtest[num_feats]
        
        elif self.scale_type == 'RBT':
            rbt = preprocessing.RobustScaler()
            self.xtrain[num_feats] = rbt.fit_transform(self.xtrain[num_feats])
            self.xvalid[num_feats] = rbt.transform(self.xvalid[num_feats])
            self.xtest[num_feats] = rbt.transform(self.xtest[num_feats])
            return self.xtrain[num_feats], self.xvalid[num_feats], self.xtest[num_feats]
        
        else:
            raise Exception ('Scaler type not supported, supported types STD or RBT.')
            
    def preprocessor(self, X_train, X_valid, X_test):
        
        if self.impute_type != False:
            xtrain, xvalid, xtest = self.imputer_(X_train, X_valid, X_test)
        if self.scale_type != False:
            xtrain, xvalid, xtest = self.scaler_(xtrain, xvalid, xtest)
        if self.encode_type != False:
            xtrain_e, xvalid_e, xtest_e, cols = self.encoder_(X_train, X_valid, X_test)           

        X_train = pd.concat([xtrain[self.num_feats], xtrain_e[cols]], axis=1).values
        X_valid = pd.concat([xvalid[self.num_feats], xvalid_e[cols]], axis=1).values
        X_test = pd.concat([xtest[self.num_feats], xtest_e[cols]], axis=1).values
        
        return X_train, X_valid, X_test
                   
    
    def train_test (self):     
        
        self.train_df = self.kfold().copy()
        scores = []
               
        for fold in range(self.n_splits):
        
            X_train = self.train_df[self.train_df.kfold != fold].reset_index(drop=True)
            X_valid = self.train_df[self.train_df.kfold == fold].reset_index(drop=True)

            X_test = self.test[self.num_feats+self.cat_feats].copy() #
            
            X_valid_ids = X_valid.PassengerId.values.tolist()

            y_train = X_train[self.target]
            y_valid = X_valid[self.target]

            X_train = X_train[self.num_feats+self.cat_feats]
            X_valid = X_valid[self.num_feats+self.cat_feats]
            
            # Preprocessing
            if self.preprocessing:
                X_train, X_valid, X_test = self.preprocessor(X_train, X_valid, X_test)
            
            # Training & Predicting
            model.fit(X_train, y_train) 

            preds_valid = model.predict(X_valid)
            preds_test = model.predict(X_test)
            
            self.valid_preds.update(dict(zip(X_valid_ids,preds_valid )))
            self.test_preds.append(preds_test)

            acc = metrics.accuracy_score(y_valid, preds_valid)
            scores.append(acc)
            
            print(f'Fold = {fold}, ACC = {acc:.10f}')
        print(f'Mean score {self.model_name} = {np.mean(scores):.10f}')
        print()
        
        valid_df = pd.DataFrame.from_dict(self.valid_preds, orient='index').reset_index().rename(columns = {'index':'PassengerId', 0:f'preds_{self.model_name}'})
        
        test_df = self.sub.copy()
        test_df.drop(self.target, axis=1, inplace=True)
        test_df.loc[:,f'preds_{self.model_name}'] = np.mean(np.column_stack (self.test_preds), axis=1 ).astype(int)
        
        return valid_df , test_df, self.train_df


# In[7]:


models = {
          'XGB': xgb.XGBRFClassifier(random_state = 0, objective = 'reg:squarederror'),
          'LGBM': lgb.LGBMClassifier(random_state = 0),
          'CB': cb.CatBoostClassifier(random_state = 0, verbose = False),
          'LR': linear_model.LogisticRegression(solver = 'liblinear'),
          'RI': linear_model.RidgeClassifier()
}


# ## Final feature selection

# In[8]:


features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'AgeBin', 'Family', 'TicketNumber', 'SocialClassByFare', 'FareBins', 'CabinCode']


# In[9]:


ordinal_features = ['Sex', 'AgeBin', 'SocialClassByFare', 'FareBins', 'Title']
ohe_features = ['TicketNumber','CabinCode', 'Embarked']
categoric_features = ordinal_features+ohe_features
continuous_features = [feature for feature in features if feature not in (ordinal_features+ohe_features) ]


# ## Training

# In[10]:


dfs_valid = []
dfs_test = []
for name, model in models.items():
    trainer = ModelTrain(train_df, test_df, submission, 5, features , continuous_features, categoric_features, 'Survived', name, 
                         model, preprocessing=True, impute_type='KNN', scale_type='RBT', encode_type='OHE')
    df_valid, df_test, train_df = trainer.train_test()
    dfs_valid.append(df_valid)
    dfs_test.append(df_test)


# In[11]:


final_valid_df = pd.concat(dfs_valid, axis = 1)


# In[12]:


final_test_df = pd.concat(dfs_test, axis=1)


# <a id="section-five"></a>
# # <div style="color:#fff;display:fill;border-radius:10px;background-color:#20BAFA;text-align:left;letter-spacing:0.1px;overflow:hidden;padding:20px;color:white;overflow:hidden;margin:0;font-size:100%"> 5 | Blending</div>
# 
# <p style="font-size:15px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# I will utilize a Logistic Regression model as the meta-model for blending in this instance. While this is effective, you might also consider trying a weighted average of the predictions, which often yields better results. </p>

# In[13]:


df_valid = final_valid_df.iloc[:,[0,1,3,5,7,9]]
df_test = final_test_df.iloc[:,[0,1,3,5,7,9]]


# In[14]:


train_blend = pd.merge(train_df , df_valid, on='PassengerId', how='left')
test_blend = pd.merge(test_df , df_test, on='PassengerId' , how='left')


# In[15]:


features_blend =  ['preds_XGB','preds_LGBM','preds_CB', 'preds_LR', 'preds_RI']


# In[16]:


# Meta Model LR

train = train_blend.copy()
test = test_blend.copy()

scores = []
valid_preds_lr = {}
test_preds_lr = []

for fold in range(5):
    
    X_train = train[train.kfold != fold].reset_index(drop=True)
    X_valid = train[train.kfold == fold].reset_index(drop=True)
    
    X_test = test[features_blend].copy() #
    
    X_valid_ids = X_valid.PassengerId.values.tolist()

    y_train = X_train['Survived']
    y_valid = X_valid['Survived']
    
    X_train = X_train[features_blend]
    X_valid = X_valid[features_blend]
    
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


# In[17]:


level1_valid_preds_lr = pd.DataFrame.from_dict(valid_preds_lr, orient='index').reset_index().rename(columns = {'index':'PassengerId', 0:'lr_pred_1'})

level1_test_preds_lr = submission.copy()
level1_test_preds_lr.Survived = np.mean(np.column_stack (test_preds_lr), axis=1 ).astype(int)
level1_test_preds_lr.to_csv('submission.csv', index = False)


# <p style="font-size:22.5px; font-family:verdana; line-height: 1.7em; margin-left:20px">   
# Thanks for taking the time to read my notebook, greetings! </p>
