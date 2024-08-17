#!/usr/bin/env python
# coding: utf-8

# <h1>Tabular Playground- Feb 2021

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("../input/tabular-playground-series-feb-2021/train.csv")


# In[3]:


data.head()


# <h3>Dropping the ID column

# In[4]:


data.drop(columns=["id"],inplace=True)


# In[5]:


data.shape


# In[6]:


data.dtypes


# <h3>datatype of all the variables look good

# In[7]:


data.describe().T


# <h3>Visually mean and median looks to be close for most of the variables implying no presence of outliers, however further investigation is required

# In[8]:


data.nunique()


# <h3>All the numeric variables are pretty much unique

# <h2>Storing numerical, categorical and target variables separately. This makes EDA simpler

# In[9]:


numeric_features = data.select_dtypes("float64").columns[:-1]
numeric_features


# In[10]:


categorical_features = data.select_dtypes("object").columns
categorical_features


# In[11]:


target = 'target'


# <h2>Looking for % change between min and max of numeric variables.

# In[12]:


data[numeric_features].apply(lambda x: (np.abs((x.max() / x.min())-1) * 100),axis=0)


# <h3>Few numeric variables have huge % diff between min and max value

# <h2>Looking for zero or near-zero variance in numeric variables by looking at std in realtion with median

# In[13]:


data[numeric_features].apply(lambda x: ((x.std() / x.median())) * 100,axis=0)


# <h3>All the continuous variables have good standard dev (variability) relative to the median.

# In[14]:


data.isnull().mean()


# <h3>No missing values

# <h2>Looking at cardinality of categorical features

# In[15]:


for i in categorical_features:
    print(f'{i}\n{(np.round((data[i].value_counts() / len(data[i]))*100,3))}\n\n')


# <h3>Few categorical variables have rare labels, which can be combined together. cat4 maybe dropped as 99% of it is made by a single category

# <h2>Normality checks

# In[16]:


data[numeric_features].hist(figsize=(20,20));


# <h3>Visually few continuous variables seem to be Gaussian. However, we need to ensure this using Kolmogorov-Smirnov test and qq-plots.

# In[17]:


for i in numeric_features:   
    print(f"Kolmogorov-Smirnov: {i} : {'Not Gaussian' if stats.kstest(data[i],'norm')[1]<0.05 else 'Gaussian'}")


# In[18]:


for i in numeric_features:   
    stats.probplot(data[i],plot=plt)
    plt.title(i)
    plt.show()


# QQ-plot and Kolmogorov-Smirnov test confirms that no continuous variable is normally distributed.

# In[19]:


fig,ax = plt.subplots(5,2,figsize=(15,20),sharey=False)
row = col = 0
for n,i in enumerate(categorical_features):
    if (n % 2 == 0) & (n > 0):
        row += 1
        col = 0
    sns.boxplot(x=data[i],y=data[target],ax=ax[row,col])
    ax[row,col].set_title(f"Target vs {i}")
    ax[row,col].set_xlabel("")
    col += 1
    
    
plt.show();


# <h3>Most of the categorival variables show no relationship with the target. But few show up some sort of relationship.

# <h2>Outlier detection

# In[20]:


fig,ax = plt.subplots(7,2,figsize=(15,30),sharey=False)
row = col = 0
for n,i in enumerate(numeric_features):
    if (n % 2 == 0) & (n > 0):
        row += 1
        col = 0
    sns.boxplot(y=data[i],ax=ax[row,col])
    ax[row,col].set_title(f"{i}")
    ax[row,col].set_ylabel("")
    col += 1
    
    
plt.show();


# <h3>Few numeric variables have outliers as shown in box plots

# <h2>Correlation

# In[21]:


fig = plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),mask=np.triu(data.corr()),annot=True,cbar=False,fmt=".2f",robust=True);


# <h3>No continuous variable is linearly correlated with the target. However, there is a fair amount of multicollinearity.

# In[22]:


fig,ax = plt.subplots(7,2,figsize=(15,40),sharey=False)
row = col = 0
for n,i in enumerate(numeric_features):
    if (n % 2 == 0) & (n > 0):
        row += 1
        col = 0
    sns.scatterplot(x=i,y=target,data=data,ax=ax[row,col])
    ax[row,col].set_title(f"Target vs {i}")
    ax[row,col].set_xlabel("")
    col += 1
    
    
plt.show();


# In[23]:


fig = plt.figure(figsize=(12,12))
sns.heatmap(data.corr(method='kendall'),mask=np.triu(data.corr()),annot=True,cbar=False,fmt=".2f",robust=True);


# <h3>The scatter plots, Pearson coeff and Kendall coeff show no linear or monotonic relationship between the numeric and target variables

# <h2>Recursively eliminating multicollinarity in numeric features

# In[24]:


X_vif  = data[numeric_features].copy()


# In[25]:


vif =  [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif)


# In[26]:


flag = True
correlated_features_to_delete = []
while flag == True:
    vif =  pd.Series([variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])])
    if vif.max() >= 10:
        max_vif_col = pd.Series(X_vif.columns)[vif.argmax()]
        correlated_features_to_delete.append(max_vif_col)
        X_vif.drop(columns=max_vif_col,inplace=True)
    else:
        flag = False


# In[27]:


correlated_features_to_delete


# <h2>Checking for multicollinearity after removing the correlated features

# In[28]:


X_vif.head(2)


# In[29]:


vif =  [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif)


# In[30]:


multi_coll_features = pd.DataFrame({'feature':correlated_features_to_delete})


# In[31]:


multi_coll_features.to_csv("correlated_features.csv",index=False)


# <h2>Target variable distribution

# In[32]:


data['target'].describe()


# In[33]:


sns.boxplot(y=data['target']);


# In[34]:


data['target'].plot(kind="kde");


# In[35]:


data['target'].quantile(0.01)


# In[36]:


data['target'].quantile(0.995)


# In[37]:


stats.probplot(data['target'],plot=plt);


# <h1>EDA findings</h1>
# <h3><ol>
#     <li>No strong relationship seen between numeric and target vaiables.</li>
#     <li>Box plots reveal relationship among few categorical variables and target variable.</li>
#     <li>There exists multicollinearity among numeric variables.</li>
#     <li>There are few categorical variables with rare labels.</li>
#     <li>No numeric variable has Gaussian distribution.</li>
#     <li>Few numeric variables have outliers</li>
#     <li>There are no numeric variables with zero or near zero variance relative to median</li>

# <h2>The above EDA shows that an extensive feature engineering is required for linear models to work on this data, since most of the assumptions like feature normality, non-multicollinearity, linear relationship b/w input features and target are not met. The relationship between the input features and target is also not looking strong and simple. Hence, trying non-linear models may be helpful. We'll build a baseline model with RandomForest.

# <h1>Vanilla RF with cv=3 results</h1>
# Train_RMSE: [0.32163441, 0.32111051, 0.32184251]
# <br>Test_RMSE: [0.8589247 , 0.86120012, 0.85887925]
# <br>Train_R2: [0.86871826, 0.86878779, 0.86855603]
# <br>Test_R2: [0.06125013, 0.06140248, 0.0612373 ]
# <br>Mean Test RMSE as % of Mean Target: 0.11529479488995717
# 
# <br><h1>These results show that the model is extremely overfit.

# <h1>Feature Engineering</h1>

# <h2>Storing the non-rare labels (labels > 5%) in a csv file would help in test set preparation.

# In[38]:


non_rare = pd.DataFrame()
for i in categorical_features:
    var_dist = data[i].value_counts().copy()
    var_dist = (var_dist / var_dist.sum()).copy()
    non_rare = pd.concat([non_rare,pd.DataFrame({i:var_dist[var_dist>0.05].index})],axis=1).copy()

non_rare.to_csv('./non_rare_categories.csv',index=False)


# In[39]:


non_rare = pd.read_csv('./non_rare_categories.csv')
non_rare


# <h2>Looking at the distribution of variables before and after combining 'rare' labels would give an idea to proceed further. 

# In[40]:


new_data = data.copy()
for i in non_rare.columns:
    new_data.loc[(new_data[i].isin(non_rare[i]) == False), i] = "Rare"

fig,ax = plt.subplots(10,2,figsize=(20,40))
row = col = 0
for n,i in enumerate(non_rare.columns):
    cat_dist = data[i].value_counts().copy()
    cat_dist = np.round((cat_dist / cat_dist.sum()) * 100,1).copy()
    cat_dist.plot(kind="bar",ax=ax[row,0],sharey=False)
    ax[row,0].set_title(i + " Before Adding Rare Label")
    for n,j in enumerate(cat_dist.index):
        ax[row,0].text(x=n-0.2,y=cat_dist[j]+0.1,s=str(cat_dist[j]) + "%")
    
    
    new_cat_dist = new_data[i].value_counts().copy()
    new_cat_dist = np.round((new_cat_dist / new_cat_dist.sum()) * 100,1).copy()
    new_cat_dist.plot(kind="bar",ax=ax[row,1])
    ax[row,1].set_title(i + " After Adding Rare Label")
    for n,j in enumerate(new_cat_dist.index):
        ax[row,1].text(x=n-0.2,y=new_cat_dist[j]+0.1,s=str(new_cat_dist[j]) + "%")
    
    
    row += 1
plt.show()


# <h2>Looking at the above distribution plots, there is no huge change in the categorical variable distribution. Hence, the rare labels can be combined.

# In[41]:


for i in non_rare.columns:
    data.loc[(data[i].isin(non_rare[i]) == False), i] = "Rare"


# In[42]:


X = data.drop(columns="target").copy()
y = data["target"].copy()


# In[43]:


# target_non_outliers = y.loc[(y>=5) | (y<=10)].index
# y = y[target_non_outliers].copy()
# X = X.iloc[target_non_outliers,:].copy()


# <h2>Not removing correlated variables as it reduced the model performance

# In[44]:


#correlated_features = list(pd.read_csv('./correlated_features.csv')['feature'])
#correlated_features


# In[45]:


#X.drop(columns=correlated_features,inplace=True)


# In[46]:


X.columns


# In[47]:


x_col = list(X.columns)


# In[48]:


len(X.columns)


# <h2>Commenting the code, since, it takes hours for the GridSearch to complete.

# In[49]:


#ct = ColumnTransformer(transformers=[['oe',OrdinalEncoder(),categorical_features]],remainder='passthrough')


# In[50]:


# pipeline = Pipeline(steps=[['ord_encoder',ct],
#                           ['rfe',RFE(estimator=xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1))],
#                           ['regressor',xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1)]])


# In[51]:


# param_grid = {'rfe__n_features_to_select': range(8,21,2),
#              'regressor__n_estimators':[200,500],
#              'regressor__max_depth':[4,7,10,12],
#              'regressor__reg_lambda':[0.01,0.1,1,10,100]}


# In[52]:


# gscv = GridSearchCV(estimator=pipeline,
#                    param_grid=param_grid,
#                    scoring="neg_root_mean_squared_error",
#                    cv=2,
#                    n_jobs=-1,
#                    return_train_score=True,
#                    verbose=11)


# In[53]:


#gscv.fit(X,y)


# In[54]:


#gscv.best_estimator_.get_params()


# {'memory': None,
#  'steps': [('rfe',
#    RFE(estimator=XGBRegressor(base_score=None, booster=None,
#                               colsample_bylevel=None, colsample_bynode=None,
#                               colsample_bytree=None, gamma=None, gpu_id=None,
#                               importance_type='gain', interaction_constraints=None,
#                               learning_rate=None, max_delta_step=None,
#                               max_depth=None, min_child_weight=None, missing=nan,
#                               monotone_constraints=None, n_estimators=100,
#                               n_jobs=-1, num_parallel_tree=None, random_state=11,
#                               reg_alpha=None, reg_lambda=None,
#                               scale_pos_weight=None, subsample=None,
#                               tree_method='gpu_hist', validate_parameters=None,
#                               verbosity=None),
#        n_features_to_select=18)),
#        
#        
#        
#        XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=0,
#                 importance_type='gain', interaction_constraints='',
#                 learning_rate=0.300000012, max_delta_step=0, max_depth=4,
#                 min_child_weight=1, missing=nan, monotone_constraints='()',
#                 n_estimators=200, n_jobs=-1, num_parallel_tree=1, random_state=11,
#                 reg_alpha=0, reg_lambda=100, scale_pos_weight=1, subsample=1,
#                 tree_method='gpu_hist', validate_parameters=1, verbosity=None)]],
#        
#        

# In[55]:


#gscv.best_estimator_


# Pipeline(steps=[('rfe',
#                  RFE(estimator=XGBRegressor(base_score=None, booster=None,
#                                             colsample_bylevel=None,
#                                             colsample_bynode=None,
#                                             colsample_bytree=None, gamma=None,
#                                             gpu_id=None, importance_type='gain',
#                                             interaction_constraints=None,
#                                             learning_rate=None,
#                                             max_delta_step=None, max_depth=None,
#                                             min_child_weight=None, missing=nan,
#                                             monotone_constraints=None,
#                                             n_estimators=10...
#                               colsample_bytree=1, gamma=0, gpu_id=0,
#                               importance_type='gain',
#                               interaction_constraints='',
#                               learning_rate=0.300000012, max_delta_step=0,
#                               max_depth=4, min_child_weight=1, missing=nan,
#                               monotone_constraints='()', n_estimators=200,
#                               n_jobs=-1, num_parallel_tree=1, random_state=11,
#                               reg_alpha=0, reg_lambda=100, scale_pos_weight=1,
#                               subsample=1, tree_method='gpu_hist',
#                               validate_parameters=1, verbosity=None)]])

# In[56]:


#gscv.best_score_ * -1


# 0.8540331315535704

# In[57]:


ct = ColumnTransformer(transformers=[['oe',OrdinalEncoder(),categorical_features]],remainder='passthrough')


# In[58]:


pipeline = Pipeline(steps=[['ord_encoder',ct],
                          ['rfe',RFE(estimator=xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1),
                                    n_features_to_select=20)],
                          ['regressor',xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1,
                                                       max_depth=4,n_estimators=200,reg_lambda=100)]])


# In[59]:


pipeline.fit(X,y)


# <h2>Decoding the pipeline

# In[60]:


features_after_oe = pd.Series(categorical_features)
features_after_oe = list(features_after_oe.append(pd.Series(x_col)[pd.Series(x_col).isin(features_after_oe)==False]))
features_after_oe


# In[61]:


features_selected_rfe = []
for n,i in enumerate(features_after_oe):
    if pipeline["rfe"].support_[n] == True:
        features_selected_rfe.append(i)
        
    print(f'{i}: {pipeline["rfe"].support_[n]}')


# In[62]:


features_selected_rfe


# <h2>Finding Feature Importance For Further Feature Engineering

# In[63]:


feat_imp = (pd.DataFrame(pipeline['regressor'].get_booster().get_score(importance_type="gain"),index=[0]).T).reset_index()
feat_imp['index'] = feat_imp['index'].str.replace('f',"").astype('int')
feat_imp.sort_values(by="index",inplace=True)
feat_imp['index'] = features_selected_rfe
feat_imp.sort_values(by=0,ascending=False,inplace=True)
feat_imp.columns = ["Feature","Imp"]
feat_imp


# In[64]:


pd.DataFrame({'Feature':features_selected_rfe,'Imp':pipeline['regressor'].feature_importances_}).sort_values(by='Imp',ascending=False)


# In[65]:


X = data.drop(columns="target").copy()
y = data["target"].copy()


# In[66]:


# target_non_outliers = y.loc[(y>=5) | (y<=10)].index
# y = y[target_non_outliers].copy()
# X = X.iloc[target_non_outliers,:].copy()


# In[67]:


#correlated_features = list(pd.read_csv('./correlated_features.csv')['feature'])
#correlated_features


# In[68]:


#X.drop(columns=correlated_features,inplace=True)


# In[69]:


X.columns


# <h1>Creating New Features By Combining The Most Important Features

# In[70]:


X['cat2p6'] = X['cat2'] + X['cat6']
X['cat6p1'] = X['cat6'] + X['cat1']
X['cat2p1'] = X['cat2'] + X['cat1']

X['cat2p0'] = X['cat2'] + X['cat0']
X['cat6p0'] = X['cat6'] + X['cat0']
X['cat1p0'] = X['cat1'] + X['cat0']


# In[71]:


new_categorical_features = list(categorical_features).copy()
new_categorical_features.extend(['cat2p6','cat6p1','cat2p1','cat2p0','cat6p0','cat1p0'])


# In[72]:


new_categorical_features


# In[73]:


ct = ColumnTransformer(transformers=[['oe',OrdinalEncoder(),new_categorical_features]],remainder='passthrough')


# In[74]:


pipeline = Pipeline(steps=[['ord_encoder',ct],
                          ['rfe',RFE(estimator=xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1),
                                    n_features_to_select=22)],
                          ['regressor',xgb.XGBRegressor(tree_method='gpu_hist',random_state=11,n_jobs=-1,
                                                       max_depth=4,n_estimators=200,reg_lambda=100)]])


# In[75]:


cv = cross_validate(estimator=pipeline,X=X,y=y,scoring='neg_root_mean_squared_error',cv=5,n_jobs=-1,return_train_score=True)


# <h2>Training RMSE

# In[76]:


cv['train_score'] *-1


# In[77]:


np.mean(cv['train_score'] *-1)


# <h2>CV RMSE

# In[78]:


cv['test_score'] *-1


# In[79]:


np.mean(cv['test_score'] *-1)


# In[80]:


pipeline.fit(X,y)


# <h2>Learning Curve

# In[81]:


train_size,train_scores,test_scores = learning_curve(estimator=pipeline,X=X,y=y,cv=5,scoring="neg_root_mean_squared_error",random_state=42)
train_scores = np.mean(-1*train_scores,axis=1)
test_scores = np.mean(-1*test_scores,axis=1)
lc = pd.DataFrame({"Training_size":train_size,"Training_loss":train_scores,"Validation_loss":test_scores}).melt(id_vars="Training_size")


# In[82]:


sns.lineplot(data=lc,x="Training_size",y="value",hue="variable");


# <h3>The flat validation loss shows that adding additional examples will not help and training error is high and increasing, hence there is no overfitting for sure. But the high training error suggests the model is underfit.

# <h1>Test Set Preparation

# In[83]:


test = pd.read_csv('../input/tabular-playground-series-feb-2021/test.csv')


# In[84]:


test.columns


# In[85]:


test_ids = test["id"].copy()


# In[86]:


test.drop(columns="id",inplace=True)


# In[87]:


non_rare = pd.read_csv('./non_rare_categories.csv')


# In[88]:


for i in non_rare.columns:
    test.loc[(test[i].isin(non_rare[i]) == False), i] = "Rare"


# In[89]:


#correlated_features = list(pd.read_csv('./correlated_features.csv')['feature'])
#correlated_features


# In[90]:


#test.drop(columns=correlated_features,inplace=True)


# In[91]:


test['cat2p6'] = test['cat2'] + test['cat6']
test['cat6p1'] = test['cat6'] + test['cat1']
test['cat2p1'] = test['cat2'] + test['cat1']

test['cat2p0'] = test['cat2'] + test['cat0']
test['cat6p0'] = test['cat6'] + test['cat0']
test['cat1p0'] = test['cat1'] + test['cat0']


# In[92]:


test.columns


# In[93]:


prediction = pipeline.predict(test)


# In[94]:


pd.read_csv('../input/tabular-playground-series-feb-2021/sample_submission.csv').head()


# In[95]:


len(test_ids) == len(prediction)


# In[96]:


submission = pd.DataFrame({'id':test_ids,'target':prediction})


# In[97]:


submission.head()


# In[98]:


submission.shape


# In[99]:


submission.to_csv('submission_6.csv',index=False)


# <h2>
#     Training RMSE is 0.829710734<br><br>
#     CV RMSE is 0.8454180152461298<br><br>
#     Submission score is 0.84502 (Top 53%) at the time of writing<br><br>
#     i.e. the model is not overfit, but the learning curve shows bias. The baseline model built using Random Forest had a CV RMSE of 0.859668. The final model built shows just a marginal improvement in performance compared to the baseline model.The top score in the leaderboard at the time of writing was 0.84100. The difference between my score and the top score is just 0.00402. Since the model has bias, using more complex models like catboost, LightGBM and rigorous hyperparameter tuning to reduce the bias might be helpful in increasing the score. Since, the model doesn't have high variance, the submission score and cv score are very close to each other. This might continue in the public leaderboard as well. REMOVING CORRELATED VARIABLES REDUCED THE MODEL PERFORMANCE.
