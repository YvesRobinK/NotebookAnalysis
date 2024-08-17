#!/usr/bin/env python
# coding: utf-8

# # ♦ Mercedes Benz - The Best or Nothing ♦

# Github link : https://github.com/deadskull7/Mercedes-Benz-Challenge-78th-Place-Solution-Private-LB-0.55282-Top-2-Percent-

# ### Down is the full kernel . . . .

# In[ ]:


import matplotlib.pyplot as plt
import cv2
from pylab import rcParams

rcParams['figure.figsize'] = 50,20
img=cv2.imread("../input/private-score/score.JPG")
plt.imshow(img)


# <img src="http://starchop.altervista.org/wp-content/uploads/2015/02/Mercedes-Benz-Logo-Rain-HD-Wallpaper.jpg"   />

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/"))
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv')
test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv')
df = train


# In[ ]:


print(train.shape)
train.head()


# 
# **370 numerical features and 8 categorical features**

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)       #  numeric dataframe
objects = ['O']
df_cat = df.select_dtypes(include=objects)
print(df_num.shape,df_cat.shape)
print(df_cat.columns,'\n','--------------------------------------------------------------------------------','\n',df_num.columns)


# **Looking into each categorical feature **

# In[ ]:


for i in df_cat.columns:
    print('The unique values in '+i+' are: ',df[i].nunique(),'\n',df_cat[i].unique(),'\n',"--------------------------------------------------------------------------------")


# In[ ]:


print(df.isnull().sum().sum(axis=0))


# **A separate dataframe to study only categorical features and there mutual relationship and also the one with target column y.**

# In[ ]:


temp=df.y.values
df_cat['y']=temp
print(df_cat.head())


# ### Outlier detection and removal .... A bit cleaning....

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(14,9)})
plt.subplot(221)
plt.title("Outlier Detection in target column via Boxplot")
plt.ylabel("Values of y")
plt.grid(True)
sns.boxplot(y=df["y"],color='gold')
plt.subplot(222)
plt.title("Outlier Detection in target column via Histogram")
plt.grid(True)
ax = sns.distplot(df.y,color='green',bins=22)
plt.show()


# **This clearly shows the outliers are above a value of approx. 137.5. Well we will remove outliers after 150.**

# In[ ]:


sns.set(rc={'figure.figsize':(20,7)})
plt.title("y Analysis")
plt.ylabel("Values of y")
plt.scatter(range(df.shape[0]),np.sort(df.y.values),color='orange')


# **A very distinct and conspicuous point around 275 in boxplot and also the green area in histogram. This noise has to removed.**

# In[ ]:


print((df.loc[df.y>150,'y'].values))
df=df[df.y<150]
print("Removing outliers based on above information and setting 150 as a threshold value . . . . . . . . . . . . . . . . . . . . ")
print(df.shape)
df_cat=df_cat[df_cat.y<150]
df_num=df_num[df_num.y<150]


# **y wrt ID of dataframe.**

# In[ ]:


sns.set(rc={'figure.figsize':(20,7)})
sns.regplot(x='ID', y='y', data=df,color='maroon')


# **This shows a very slight decreasing trend of y wrt ID , maybe cars later in series took less time in test bench. This gives ID an importance while estimating y.**

# In[ ]:


from scipy import stats
rcParams['figure.figsize'] = 15, 7
res = stats.probplot(df['y'], plot=plt)


# The y values of the dataset appears to be skewed

# Taking Log Transformation

# In[ ]:


res = stats.probplot(np.log1p(train["y"]), plot=plt)


# 
# ## Now lets see some jitter on boxplots . . .

# In[ ]:


rcParams['figure.figsize'] = 22, 8
for i in df_cat.columns:
    if i not in 'y':
        plt.figure()
        plt.xlabel=i
        sns.stripplot(x=i, y="y", data=df,jitter=True, linewidth=1,order=np.sort(df[i].unique()))
        sns.boxplot(x=i, y="y", data=df, order=np.sort(df[i].unique()))
        plt.show()


# 
# 
# In these stripplots with boxplots superimposed,  we find the following:
# 
# *     X0 and X2 have a large amount of diversity in their levels. Among those two, X0 shows the most obvious effect of grouping.
# 
# *     The lowest y values (i.e. shortest times) appear to be predominantly caused by 6 feature levels: X0:az, X0:bc, X1:y, X2:n, X5:h, X5:x. Together, these ones are a pretty good predictor for having low y.
# 
# *     Level X0:aa appears to have a notably higher average y than all other features, but consists only of two data points. This is very obvious with the jitter plots.
# 
# *     X3, X5, X6, X8 and to a certain extent X1 show distributions that are largely similar among the different levels

# 
# **Bivariate analysis using Cross-tabulation**

# This also shows the gradient of change using colour change. As you can see as in X2 is the most popular category and leaving most of them with zero. Similarly it can be tested on any 2 more than two categories at the same time to check concurrent occurences of any pairs, triads, quadruplets , etc . . . . 

# In[ ]:


pd.crosstab([df_cat.X2], [df_cat.X0], margins=True).style.background_gradient(cmap='autumn_r')


# ## Further data cleaning . . .  

# 
# * Removing the features from the main dataframe that are involving zero variance or are having constant value inorder to remove redundancy and increase model performance later.
# * Also checking the individual correlation of the features and getting some idea about individual feature importance.
# * There are total 13 variables with zero variance , therefore they must be dropped.
# * Checking for duplicate features in this large set.
# * Feature selection multiple times .....

# Removing columns with zero ovariance

# In[ ]:


temp = []
for i in df_num.columns:
    if df[i].var()==0:
        temp.append(i)
print(len(temp))
print(temp)


# Again setting a threshold of 0.01 for variance for each column and removing them too. The removed columns are also being removed from all the temporary dataframes.

# In[ ]:


count=0
low_var_col=[]
for i in test.columns:
    if test[i].dtype == 'int64':
        if test[i].var()<0.01:
            low_var_col.append(i)
            count+=1
print(count)

df.drop(low_var_col,axis=1,inplace=True)
df_num.drop(low_var_col,axis=1,inplace=True)
test.drop(low_var_col,axis=1,inplace=True)


# Turn out to be there are 146 columns for removal purpose.

# 
# Updating the df_num dataframe after droping the features from original dataframe df.

# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)


# Getting the dictionary of important correlated features with target column y

# ## Some important feature correlations with the target variable.

# ** Taking 0.25 as threshold on grounds of experimental changes . . .**

# In[ ]:


dic={}
for i in df_num.columns:
    if i!='y':
        if df[i].corr(df.y)>0.25 or df[i].corr(df.y)<-0.25:
            dic[i]=df[i].corr(df.y)
print("Important Features with there respective correlations are ",'\n','---------------------------------------------------------','\n',dic)


# **This states that X29, X54, X76, X127, X136, X162, X166, X178,  X232,  X250,  X261, X263, X272,  X276, X279, X313, X314, X328  are important features later we will select using some selection techniques. **

# **But ,  YOU MUST SEE THAT SOME FEATURES ARE HAVING SAME CORRELATIONS THAT COULD INDICATE THE POSSIBLE DUPLICATE FEATURES. Lets check them too . . **

# In[ ]:


print(df.X119.corr(df.X118),'\n', df.X29.corr(df.X54) ,'\n', df.X54.corr(df.X76) ,'\n', df.X263.corr(df.X279))


# 
# This shows that are dataframe is containing some duplicate features which are having correlation of approx. 1. We will remove this redundancy also using some feature selection . .  .

# **Duplicate features. **

# In[ ]:


# Dublicate features
d = {}; done = []
cols = df.columns.values
for c in cols: d[c]=[]
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(df[cols[i]] == df[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0: 
        dub_cols += d[k]        
print('Dublicates:','\n', dub_cols)


# Checking correlations among a set of duplicate features and preparing pairs who are highly correlated.

# **Again, correlation threshold of 0.9 has been judged and taken after multiple experiments .....**

# In[ ]:


corrs=[]
high_corr=[]
for i in range(0,len(dub_cols)):
    for j in range(i+1,len(dub_cols)):
        if df[dub_cols[i]].corr(df[dub_cols[j]]) >=0.90:
            corrs.append(df[dub_cols[i]].corr(df[dub_cols[j]]))
            high_corr.append((dub_cols[i],dub_cols[j]))
print(corrs)
print("\n")
print(high_corr)


# In[ ]:


df.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)


# In[ ]:


test.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)
df_num.drop(['X279','X76','X37','X134','X147','X222','X244','X326'] , axis=1 , inplace=True)


# Label encoding the categorical features 

# **This dataset has some real problem with the number of categories.**

# There are different number of categories in train and test datset. Encountered

# In[ ]:


from sklearn import preprocessing
categorical=[]
for i in df.columns:
    if df[i].dtype=='object':
        le = preprocessing.LabelEncoder()
        le.fit(list(df[i].values) + list(test[i].values))
        print("Categories in the encoded order from 1 to the size of "+i+" are : ")
        print(le.classes_)
        print("--------------------------------------------------------------------------")
        df[i] = le.transform(list(df[i].values))
        test[i] = le.transform(list(test[i].values))
        categorical.append(i)


# #### Now, finding correlations of each category with other . The increasing or decreasing class encoded value can be found from the categories written in the encoded order above.

# In[ ]:


correlation_map = df[df.columns[1:10]].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(9,10)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)


# 
# **Preparing the data for feature importance**

# In[ ]:


import xgboost as xgb
train_y = df["y"].values
train_X = df.drop(['y'], axis=1)

def xgb_r2_score(preds, final):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'n_trees': 520, 
    'eta': 0.0045,
    'max_depth': 4,
    'subsample': 0.98,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), # base prediction = mean(target)
    'silent': 1
}

final = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params), final, num_boost_round=200, feval=xgb_r2_score, maximize=True)

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(model, max_num_features=40, height=0.8, ax=ax, color = 'coral')
print("Feature Importance by XGBoost")
plt.show()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:40]

plt.subplots(figsize=(10,10))
plt.title("Feature importances by RandomForestRegressor")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), feat_names[indices], rotation='horizontal')
plt.ylim([-1, len(indices)])
plt.show()


# **There seems a difference between the feature importances by the two models. You can check above, RandomForest is giving the feature importance more on the basis of the important correlations of target wrt numerical features that we have already figured out above.**

# # Lets do some Feature Engineering . . .
# 
# * Following are some features that I have engineered after multiple trials  .
# * These are just the results (the new features engineered) of the work that I have done for long.
# * Since the features are anonymised, that makes pretty much difficult to do feature engineering.
# * I have made some 2 way and 3 way interactions of the features which prove to be pretty much useful.
# * Also there correlations are higher than their parent features which makes them even better.
# * Feel free to write new features engineered by yourself in the comment section below.

# <img src="https://media.giphy.com/media/fQoCOuFL7DlR6zYRnw/giphy.gif" />

# Taking X314 and X315

# In[ ]:


df['X314_plus_X315'] = df.apply(lambda row: row.X314 + row.X315, axis=1)
test['X314_plus_X315'] = test.apply(lambda row: row.X314 + row.X315, axis=1)


# In[ ]:


print("Correalation between X314_plus_X315 and y is :  ",df.y.corr(df['X314_plus_X315']))
print("Which makes it pretty much high !! Awesome !!")


# Taking X122 and X128

# In[ ]:


#df['X122_plus_X128'] = df.apply(lambda row: row.X122 + row.X128, axis=1)
#test['X122_plus_X128'] = test.apply(lambda row: row.X122 + row.X128, axis=1)


# In[ ]:


#print("Correlation between X122_plus_X128 and y is :  ",df.y.corr(df['X122_plus_X128']))


# Taking X118 , X314 and X315

# In[ ]:


df['X118_plus_X314_plus_X315'] = df.apply(lambda row: row.X118 + row.X314 + row.X315, axis=1)
test['X118_plus_X314_plus_X315'] = test.apply(lambda row: row.X118 + row.X314 + row.X315, axis=1)


# In[ ]:


print("Correalation between X118_plus_X314_plus_X315 and y is :  ",df.y.corr(df['X118_plus_X314_plus_X315']))
print("Which makes it pretty much high !! Awesome !!")


# Taking X10 and X54

# In[ ]:


df["X10_plus_X54"] = df.apply(lambda row: row.X10 + row.X54, axis=1)
test["X10_plus_X54"] = test.apply(lambda row: row.X10 + row.X54, axis=1)
print("Correalation between X10_plus_X54 and y is :  ",df.y.corr(df['X10_plus_X54']))


# Taking X10 and X29

# In[ ]:


df["X10_plus_X29"] = df.apply(lambda row: row.X10 + row.X29, axis=1)
test["X10_plus_X29"] = test.apply(lambda row: row.X10 + row.X29, axis=1)
print("Correalation between X10_plus_X29 and y is :  ",df.y.corr(df['X10_plus_X29']))


# Updating the dataframe for feature importance , the one we used above.

# In[ ]:


train_X['X314_plus_X315']=df['X314_plus_X315']
#train_X['X122_plus_X128']=df['X122_plus_X128']
train_X['X118_plus_X314_plus_X315']=df['X118_plus_X314_plus_X315']
train_X["X10_plus_X54"] = df["X10_plus_X54"]
train_X["X10_plus_X29"] = df["X10_plus_X29"]


# * Taking the numeric dataframe df_num and finding all the features with very high correlation an dchecking for them. Also making the pairs of them as above.
# * Turns out to be a list of 63 features again that are highly correlated. 
# **Again, the value of 0.95 has been experimentally judged and taken , there is no thumb rule to take the threshold value.**

# In[ ]:


corr_val=[]
same_features=[]
for i in range(0,len(df_num.columns)-1):
    for j in range(i+1,len(df_num.columns)):
        temp_corr=df[df_num.columns[i]].corr(df[df_num.columns[j]])
        if temp_corr>=0.95 or temp_corr<=-0.95: 
            same_features.append((df_num.columns[i],df_num.columns[j]))
            corr_val.append(temp_corr)
print(len(corr_val))
print(same_features)


# In[ ]:


booler = np.ones(400)
for i in same_features:
    if booler[int(i[1][1:])]==1:
        booler[int(i[1][1:])]=0
        df_num.drop(i[1],axis=1,inplace=True)
        df.drop(i[1],axis=1,inplace=True)
        test.drop(i[1],axis=1,inplace=True)
        train_X.drop(i[1],axis=1,inplace=True)
    elif booler[int(i[0][1:])]==1:
        booler[int(i[0][1:])]=0
        df_num.drop(i[0],axis=1,inplace=True)
        df.drop(i[0],axis=1,inplace=True)
        test.drop(i[0],axis=1,inplace=True)
        train_X.drop(i[0],axis=1,inplace=True)


# **STEPS for removing above redundancy . . .**

# * Booler is the array of 1 and 0 for keeping the track of the features that we have dropped from the multiple dataframes and allowing the execution of cell without off any error of column not found. (Also the features are being repeated in multiple pairs.) 
# * Initially the booler is taken as all of 1,  considering the fact that all features are present in the dataframe and later we would make them zeroes one by one. 
# * The steps are like , we will target the 2nd feature of each pair and check for its existence, if its present then remove it or else if check for the 1st feature in the pair , and if this also is not present then simply skip that particular pair of feature. 
# * The booler would be used to check the existence of that features in dataframes.

# This dataset is very dirty !! Believe it ....We have to clean it to the utmost level we can to feed into our model to achieve the high accuracy that we aspire.

# ## That look great the new features engineered have outperformed the existing features in the data in the RandomForrest feature importance plot.

# In[ ]:


model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
feature_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:40]

plt.subplots(figsize=(10,10))
plt.title("Feature importances by RandomForestRegressor")
plt.ylabel("Features")
plt.barh(range(len(indices)), importances[indices], color="green", align="center")
plt.yticks(range(len(indices)), feature_names[indices], rotation='horizontal')
plt.ylim([-1, len(indices)])
plt.show()

final = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params), final, num_boost_round=1350, feval=xgb_r2_score, maximize=True)

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(model, max_num_features=40, height=0.8, ax=ax,color = 'coral')
print("Feature Importance by XGBoost")
plt.show()


# Though in XGboost they have earned a little less position but still higher enough to be considered as good work for model performance.

# Taking a total of 378 feature count to 185...

# In[ ]:


print(train_X.shape , test.shape)


# In[ ]:


list(set(train_X.columns)-set(test.columns))


# ### Below is the code for one hot encoding and creating a sparse matrix of around 211 features. But commented it out because it gave me degraded performance. Don't know why ....if anybody knows the answer than please comment it below... I would love to listen.

# In[ ]:


'''from sklearn.preprocessing import OneHotEncoder
total_hot=np.concatenate( (train_X.values[:,1:9], test.values[:,1:9]), axis=0)
enc = OneHotEncoder()
enc.fit(total_hot)
total_hot=enc.transform(total_hot)'''


# Making a dense matrix from sparse

# In[ ]:


'''total_hot.todense().shape'''


# Appending the encoded categories to ID vector and then appending the rest dataframe of numerical features to this newly formed dataframe. Similarly doing this to test matrix.

# In[ ]:


'''train_hot=total_hot.todense()[:4194,:]
test_hot=total_hot.todense()[4194:8404,:]
print(train_hot.shape)
train_X_hot=np.concatenate( (train_X.values[:,0].reshape(4194,1),train_hot) , axis=1)
test_hot=np.concatenate( (test.values[:,0].reshape(4209,1),test_hot) , axis=1)
train_X_hot=np.concatenate( (train_X_hot,train_X.values[:,9:]) , axis=1)
test_hot=np.concatenate( (test_hot,test.values[:,9:]) , axis=1)'''


# In[ ]:


'''print(train_X_hot.shape, test_hot.shape)'''


# Later tried to append the PCA , SVD ,sparse random projections to the dataframe but still got degraded model performance...Please do tell me if anybody knows the answer for this....

# Using 12 as components so as to still retain a variance of ~98%.

# In[ ]:


'''from sklearn.decomposition import PCA
pca=PCA(n_components=6 , random_state=7)
pca.fit(train_X_hot)
pca_train_X = pca.transform(train_X_hot)
pca_test = pca.transform(test_hot)

print(pca.explained_variance_ratio_.sum())
print("--------------------------------------------------------------")
print(pca.components_)
print("--------------------------------------------------------------")
print(pca.components_.shape)
print("--------------------------------------------------------------")
print(pca_train_X.shape , pca_test.shape)
'''


# Validating our XGboost...Finding the best hyperparameters.

# ### Till here I was in 20 % on private leaderboard but after hyperparameter tuning I landed in top 2% .

# In[ ]:


import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2, random_state=420)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(test)

xgb_params = {
    'n_trees': 500, 
    'eta': 0.0050,
    'max_depth': 3,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), # base prediction = mean(target)
    'silent': 1
}

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

clf = xgb.train(xgb_params, d_train, 1050 , watchlist, early_stopping_rounds=70, feval=xgb_r2_score, maximize=True, verbose_eval=10)


# Now, training the whole dataset on selected parameters so as to avoid any data loss.

# In[ ]:


d_train = xgb.DMatrix(train_X, label=train_y)
#d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(test)

xgb_params = {
    'n_trees': 500, 
    'eta': 0.0050,
    'max_depth': 3,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.mean(train_y), 
    'silent': 1
}

def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

watchlist = [(d_train, 'train')]

clf = xgb.train(xgb_params, d_train, 1050 , watchlist, early_stopping_rounds=70, feval=xgb_r2_score, maximize=True, verbose_eval=10)


# ## Making the submission file ...Check yourself for the authentication of script claiming 78th place on private leaderboard with a score of 0.55282 which is Top 2 %. 

# ## Check for output tab of the notebook and check the score after submitting it . . . 

# In[ ]:


Answer = clf.predict(d_test)

sub = pd.DataFrame()
sub['ID'] = test.ID
sub['y'] = Answer
sub.to_csv('mercedes_benz_The_best_or_Nothing.csv', index=False)


# In[ ]:


sub.head()


# ## Please upvote if you like . . . 

# In[ ]:




