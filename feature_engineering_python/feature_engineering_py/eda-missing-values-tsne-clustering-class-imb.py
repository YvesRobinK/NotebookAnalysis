#!/usr/bin/env python
# coding: utf-8

# <h1><center> Jane Street Market Prediction </h1>
# 

# ![](https://storage.googleapis.com/kaggle-organizations/3761/thumbnail.png?r=38)

# <h2> Table of content </h2>
# <ul> <li> <a href="#preparation"> Preparation </a> </li>
#     <li> <a href="#load_data"> Load The Datas </a> </li>
#     <li> <a href="#first_look"> First Look at The Data </a> </li>
#     <li> <a href="#eda"> Explorations Data analysis </a> </li>
#     <li> <a href="missing_values"> Missing Values </a> </li> 
#     <li> <a href="feature_engineering"> Feature engineering </a> </li>
#     <li> <a href="feat_exploration"> Features dataset explorations </a> </li>
#     <li> <a href="modeling"> Modeling </a> </li>
#     <li> <a href="submission"> Submission </a> </li>
# </ul>
# <hr>

# <h2 id=preparation> Preparation </h2>

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.express as px 
import matplotlib.gridspec as gridspec
from  collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE as tsne
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import gc
import pickle
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_curve,auc,roc_auc_score
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import KFold



# In[2]:


def save_pickle(dict_param,nom_fich):
    with open(nom_fich,"wb") as f :
        pickle.dump(dict_param,f)


# <h2 id=load_data> Load The Datas

# In[3]:


# The root path
path = "/kaggle/input/jane-street-market-prediction/"

# Load the training datas.
train = pd.read_csv(path + "train.csv")

# Load the metadata pertaining to the anonymized features.
features = pd.read_csv(path + "features.csv")


# <h2 id=first_look> First Look at the Data </h2>

# In[4]:


# Take a look at the training data
train.head()


# In[5]:


# info about training datas.
train.info()


# In[6]:


# some statistics on the training datas.
train.describe()


# In[7]:


# Display missing values per column

missing_table = pd.DataFrame({c:(train[c].isna().sum()/len(train))*100 for c in train.columns},index=["% missing values"])

missing_table


# ===> It's obvious , that we should treat effeciently the little number of missing values in the column from the feature_120 to the feature_129.

# <h2 id=eda> Explorations Data analysis </h2>

# In[8]:


# Display the histogram 
fig,axes = plt.subplots(nrows=45,ncols=3,figsize=(25,250))

for i in range(2,137):
    sns.distplot(train.iloc[:,i],ax=axes[(i-2)//3,(i-2)%3])


# - The most signal features seems have a normal gaussian distribution, and they are zero centered.
# 
# - The return features, also are zero centred gaussian distributed. Which it can been seen, that the chance to make profit or to lose, is the same at any time of the day . 
# 

# In[9]:


# Compute the correlation between pair features.

correlation_table = train[[train.columns[i] for i in range(2,137)]].corr()


# In[10]:


# Display the correlation table of pair features.
correlation_table


# In[11]:


def detect_correlated_features(df,threshold=0.5):
    """This function will try detect features who have correlation grower than the introduced 
     threshold value.
     
     @param df(DataFrame): The dataframe who resume the correlation values between features.
     @param threshold(int) : the threshold that the function, will use as reference to detect
                             correlated features.
     @return list(List): list of tuple, who resume features that have correlation grower than
                           the introduced threshold.
     """
    correlated= defaultdict(list)
    for col in df.columns:
        dex = list(df.columns).index(col)
        for ind in df.index[dex+1:] :
            if df.loc[col,ind] > threshold:
               correlated[col].append (ind)
                
    return correlated


# In[12]:


# Detect the highly correlated features.Which they had coefficient correlation grower than 0.9.
correlated_features = detect_correlated_features(correlation_table,threshold=0.9)


# In[13]:


# Display a table showing the high correlated features.
ax_features = correlated_features.keys()
ay_features = []
for f in correlated_features.values():
    ay_features.extend(f)
ay_features = np.unique(ay_features)


# In[14]:


# Set up the matplotlib figure
f , ax = plt.subplots(figsize=(70,50))
sns.heatmap(correlation_table.loc[ax_features,ay_features],cmap='BrBG',annot=True,square=True,vmin=-1,vmax=1,\
            linewidths=0.5,cbar_kws={"shrink": .5})


# Let's see that we can save only one feature from each pair highly correlated features.

# In[15]:


# List of features to drop because of they are highly correlated 
# with others features in the train dataset.
features_to_drop = [f for f in ay_features if f not in ax_features]


# In[16]:


# train datas after removing features assigned to drop list of columns.
train_df = train[[ f for f in list(train.columns) if ((f not in features_to_drop) or (f =="resp"))]]


# Let's now, study the correlation of each feature with the feature named resp, which should be the feature that make deciders to move on for the trading or not.

# In[17]:


# Compute the correlation betwen the features named resp, and the reste of features
label_correlation = pd.DataFrame({c:train_df["resp"].corr(train_df[c]) for\
                                  c in train_df.columns if c!="resp" and c!="feature_0"},index=["action"])


# In[18]:


# Visualize the correlation in a table named label_correlation.
l = len(list(label_correlation.columns)) # compute the number  of features for label_correlation dataset
col = list(label_correlation.columns) # list of columns names of label_correlation dataset

# Because of , the high number of features in our dataset, we will try divide them into 5,
# in order to get more clear chart.
fig ,axes = plt.subplots(nrows=5,ncols=1,figsize=(60,30))
level = [0,int(l/5),int(2*l/5),int(3*l/5),int(4*l/5),l]
for i in range(len(level)-1):
    sns.heatmap(label_correlation.loc[:,col[level[i]:level[i+1]]],annot=True,cmap='BrBG',\
                linewidths=0.5,vmin=-1,vmax=1,cbar_kws={"shrink": .5},square=True,ax=axes[i])


# Except resp_{1,2,3,4} values that represent returns over different time horizons, there is no evident linear relation between retained features and the feature of return.

# In[19]:


# correlation between the binary feature named feature_0 and the target values.
train_df.loc[:,"target"] = list(((train_df["resp"] > 0) & (train_df["resp_1"] >0) & (train_df["resp_2"]>0)\
                     & (train_df["resp_3"]>0) & (train_df["resp_4"]>0)).astype("int"))
train_df.loc[:,"vl"] = list(train_df["target"].values)
pvt_table=train_df[["feature_0","target"]].pivot_table(index=["feature_0"],columns=["target"],aggfunc=len)
#nb_1 = len(train_df.loc[train_df["target"]==1,:])
#nb_0 = len(train_df.loc[train_df["target"]==0,:])
#pvt_table[1] = pvt_table[1]/nb_1
#pvt_table[0] = pvt_table[0]/nb_0
tx= train_df["feature_0"].value_counts()
ty = train_df["target"].value_counts() 
tx = pd.DataFrame(tx)
ty = pd.DataFrame(ty)
n = len(train_df)
tx.columns = ["values"]
ty.columns = ["values"]

cnt = tx.dot(ty.T)/n
ind = cnt.index
pvt_table = pvt_table.loc[ind,:]
mesure = (cnt - pvt_table)**2/cnt
xin = mesure.sum().sum()

del(train_df["target"])
del(train_df["vl"])
#del(nb_1)
#del(nb_0)
fig = plt.figure(figsize=(12,8))
sns.heatmap(mesure,annot=True,linewidths=0.5,cmap="BrBG",vmin=0,vmax=1)
plt.title("Correlation table between feature_0 and the target value ",size=15,color="red")
print("The total correlation between feature_0 and the target equal to {}".format(xin))
#del(pvt_table)


# <font color=redblue> We can conclude that, there is no correlation between feature_0 and the target values.

# <h2 id=missing_values> Missing Values:

# In[20]:


# Display train_df missing values before imputations.
missing_b_imputation = pd.DataFrame({c:(train_df[c].isna().sum()) for c in train_df.columns},index=["% missing values"])
print("The number of missing values in the train_df dataframe before imputation processing :{}".format(\
                                                                                                      missing_b_imputation.sum().sum()))
missing_b_imputation


# In[21]:


# identify , which column has missing values in the new dataset named train_df

features_with_missing_values = [] # list of features , has missing values.
    
for f in list(train_df.columns):
    if missing_table.loc["% missing values",f] > 0 :
        features_with_missing_values.append(f)


# In[22]:


# train a linear model regression for each feature, had missing values with his one of correlated
# feature
for f in features_with_missing_values :
    model = LinearRegression()
    if  len(correlated_features[f]) > 0 :
        correlated = correlated_features[f][0]
        if correlated in train.columns :
           model.fit(train.loc[(train[correlated].notna()) & (train[f].notna()),correlated].values.reshape(-1,1),\
              train.loc[(train[correlated].notna()) & (train[f].notna()),f])
           values_to_impute = train_df.loc[(train[f].isna()) & (train[correlated].notna()),f]
           imputer = train.loc[(train[f].isna())&(train[correlated].notna()),correlated].values
           if (len(values_to_impute) > 0) & (len(imputer) > 0) :
              train_df.loc[(train[f].isna()) & (train[correlated].notna()),f] = model.predict(train.loc[(train[f].isna())&(train[correlated].notna()),correlated].values.\
                                                      reshape(-1,1))
    
    
    


# In[23]:


#Display train_df missing values after imputation with linear regression
missing_a_imputation = pd.DataFrame({c:(train_df[c].isna().sum()) for c in train_df.columns},index=["Number missing values"])
print("The number of missing values in the train_df dataframe after imputation processing :{}".format(\
                                                                                                      missing_a_imputation.sum().sum()))
missing_a_imputation


# ==> Although this approach to remplace missing values , is very efficient .It was able to impute only about 6% of the prior number of missing values.

# In[24]:


# for the rest of the missing values, we will use the mean value as imputer for each features.
for f in features_with_missing_values:
    train_df.fillna(train_df[f].mean(),inplace=True)


# <h2 id=feature_engineering> Feature Engineering </h2>

# The feature that we want to predict through this project , is the feature which can give signal to the investor to move on with the trading or not. The decision of the investor is based on the return of the trading ,so they should predict which one can make profit. Therefore we will create new feature named "action" wich give 1 when there is positive return and 0 if it is not.

# In[25]:


#resp_cols = ['resp_1', 'resp_2', 'resp_3', 'resp', 'resp_4']


# In order to enhance the predictive quality of our model, we will check the correlation between the global return and the differents returns in differents time horizons.

# In[26]:


#
returns = ["resp_1","resp_2","resp_3","resp_4","resp"]
                                                             
datas = pd.DataFrame({c:(train_df[c]>0).astype("int") for c in returns})
datas["val"] = datas.loc[:,"resp"].values
fig , ax = plt.subplots(2,2,figsize= (15,12))
for i in range(len(returns)-1):
    k = i // 2
    l = i % 2
    piv_resp_1_resp = pd.pivot_table(datas,index= returns[i],columns="resp",values="val",aggfunc="count")
    ty = datas["resp"].value_counts()
    tx = datas[returns[i]].value_counts()
    tx = pd.DataFrame(tx)
    ty = pd.DataFrame(ty)
    ind = piv_resp_1_resp.index
    col = piv_resp_1_resp.columns
    tx.columns = ["values"]
    ty.columns = ["values"]
    n = len(datas)
    cnt = tx.dot(ty.T)/n
    cnt = cnt.loc[ind,col]
    mesure = (cnt - piv_resp_1_resp) ** 2 /cnt
    xid = mesure.sum().sum()
    mesure = mesure 
    sns.heatmap(mesure,annot=True,linewidths=0.5,cmap="BrBG",ax=ax[k,l],vmin=0,vmax=1)
    ax[k,l].set_title("The Correlation Table Between the feature resp and {} ".\
                      format(returns[i]),size=12,color="red")
    print("The total correlation between the feature {} and the resp feature equal to {}".\
          format(returns[i],xid))
   


# <font color=redblue> We can conclude that there is high correlation between the return in the different horizons time and the return feature, which confirm that there features are highly correlated with the return feature. For this reason we will choice our feature action , that will be our target, as the combinaison of all theses features.

# In[27]:


# Define new feature named action , which can help investor to make decidion.

train_df["action"] = ((train_df["resp"] > 0) & (train_df["resp_1"] >0) & (train_df["resp_2"]>0)\
                     & (train_df["resp_3"]>0) & (train_df["resp_4"]>0)).astype("int")
#Y = np.stack((train_df[c]>0).astype("int") for c in resp_cols).T


# <h2 id=feat_exploration> Features dataset explorations :

# In[28]:


# Load the features data.
features = pd.read_csv(path + "features.csv")


# In[29]:


# Let's take a look at the features data 
features.head()


# In[30]:


# some statistics on the datas.
features.info()


# **We can notice , that fortunatelly the datas don't have any missing values.**

# In[31]:


# Change the type of features dataframe to int.
features.set_index('feature',inplace=True)
features = features.astype("int8")


# In[32]:


#T_sne implementation
t_sne = tsne(n_components=2,random_state=42).fit_transform(features.values)


# In[33]:


# Plotting the embedding of features datas in two dimension using the technique of TSNE.
fig,ax = plt.subplots(1,1,figsize=(10,6))
plt.scatter(t_sne[:,0],t_sne[:,1],cmap="coolwarm")
plt.grid(True)
plt.title("t_SNE")
plt.suptitle("Dimmensionality Reduction using TSNE technique")


# ==> The chart of Dimmensionality Reduction above , shows that the datas can be divided into some number of clusters. Indeed, we notice obviously, many accumulation of points, which are distant and they can be considered as a separated clusters.

# In[34]:


# We will choice the best number of cluster, who can give best performance using 
# silhoute coefficient.

clusters_number = [4,8,12,16,20,24,28] # number of clusters to test in order to choice the best one.
silhouette_performances = {} # Dictionnary of performance.

for cl_n in clusters_number :
    kmeans = KMeans(n_clusters=cl_n)
    kmeans.fit(features.values)
    sc=silhouette_score(features.values,kmeans.labels_)
    silhouette_performances[sc] = cl_n


# In[35]:


# plot the the cluster performances in function of the number of clusters.
fig,ax = plt.subplots(1,1,figsize=(15,8))
plt.plot(list(silhouette_performances.values()),list(silhouette_performances.keys()))
plt.title("Silhouette score ")
plt.xlabel("Clusters number",fontsize=10)
plt.ylabel("Silhouette score ",fontsize=10)
plt.ylim([0.15,0.28])
best_performace = np.max(list(silhouette_performances.keys()))
ab = silhouette_performances[best_performace]
text = "best performance"
plt.annotate(text,xy=(ab,best_performace),arrowprops=dict(facecolor='black', shrink=0.05),\
            xytext=(ab+2,best_performace + 0.005))
plt.tick_params(axis="x",labelsize=15)
plt.tick_params(axis="y",labelsize=15)


# Let's now visualize theses defined clusters in two dimension, using T_SNE

# In[36]:


best_model = KMeans(n_clusters=20)
best_model.fit(features.values)


# In[37]:


fig,ax = plt.subplots(1,1,figsize=(15,8))
plt.scatter(t_sne[:,0],t_sne[:,1],c=best_model.labels_)
plt.suptitle("Visualize clusters using T_SNE")
plt.title("T_SNE")
plt.grid(True)


# <h2 id=modeling> Modeling :</h2>

# In[38]:


train_df = train_df.query('date > 85').reset_index(drop = True) 


# In[39]:


features = list(train_df.columns) # list of retaind features 
features.remove("weight")
features.remove("resp_1")
features.remove("resp_2")
features.remove("resp_3")
features.remove("resp_4")
features.remove("resp")
features.remove("action")
features.remove("ts_id")
features.remove("date")
features.remove("feature_0")


# In[40]:


save_pickle(features,"features_names")


# In[41]:


fig=px.pie(train_df.loc[train_df["weight"] > 0,:],names="action",title="Class imballance")
fig.update_layout(title={"x":0.475,"y":0.9,"xanchor":"center","yanchor":"top"})


# ==> The chart above , show that there is a class imballance , that we should correctly tackled in order to not biased the performance of our model.

# In order to tackle correctly the class imbalnce problem, the easiest way to succefully generalise is to use more datas.The problem is that out-of-the-box classifiers like logistic regression or random forest tend to generalize by discarding the rare class. One easy best practice is building n models that use all the samples of the rare class and n-differing samples of the abundant class. Given that you want to ensemble 10 models, you would keep e.g. the 1.000 cases of the rare class and randomly sample 10.000 cases of the abundant class. Then you just split the 10.000 cases in 10 chunks and train 10 different models.</font>
# 
# ![](https://www.kdnuggets.com/wp-content/uploads/imbalanced-data-2.png)
# 
# In our case, the size of the abundant class is three time as bigger as the rarely class. So we need to train three models , by splitting the datas of the abundant class to three sample , and use the the datas of the rare class for each model training.

# In[42]:


abundant_class = train_df.loc[(train_df["weight"] > 0) & (train_df["action"]==0),:] # extract datas which concern abundant datas
rare_class = train_df.loc[(train_df["weight"] > 0)&(train_df["action"]==1),:]   # extract datas which concern rares datas.


# In[43]:


# The mean values of each feature to use in order to impute missing values in real production.
imputer = np.mean(train_df[features].values,axis=0)
save_pickle(imputer,"features_imputation")


# In[44]:


abundant_class = abundant_class.sample(frac=1)
rare_class = rare_class.sample(frac=1)


# In[45]:


l = len(rare_class) # the size of data which concern rare class.
df1 = abundant_class.iloc[:l,:].append(rare_class) # 1st chunk of datas to train first model
df2 = abundant_class.iloc[l:2*l,:].append(rare_class) # 2nd chunk of datas to train second model
df3 = abundant_class.iloc[2*l:,:].append(rare_class) # 3nd chunk of datas to train the third model.

df1 = df1.sample(frac=1) # shuffle the datas
df2 = df2.sample(frac=1) # shuffle the datas
df3 = df3.sample(frac=1) # shuffle the datas


# In[46]:


len(df1),len(df2),len(df3)


# In[47]:


# reduce the memory charge
del(train_df)
del(train)


# In[48]:


# retained features and label to train differents models on differents chunk of datas.
training= [df1[features],df2[features],df3[features]] 
targets = [df1["action"],df2["action"],df3["action"]]


# In[49]:


datas = []
for i in range(3):
    xtr,xval,ytr,yval = train_test_split(training[i].values,targets[i].values,test_size=0.1,\
                                        stratify=targets[i].values)
    datas.append(((xtr,ytr),(xval,yval)))


# In[50]:


# refresh memory
del(df1)
del(df2)
del(df3)
del(abundant_class)
del(rare_class)
del(features_with_missing_values)
del(correlation_table)
del(missing_table)


# In[51]:


gc.collect()


# In[52]:


# modeling step 
params={"num_leaves":300,
       "max_bin":450,
       "feature_fraction":0.52,
       "bagging_fraction":0.52,
       "objective":"binary",
       "learning_rate":0.05,
       "boosting_type":"gbdt",
       "metric":"auc"
       }
#kf = KFold(n_splits=3,shuffle=True,random_state=111)
models = [] # list of model , we will train 
for i in range(3):
    xtr = datas[i][0][0]
    ytr = datas[i][0][1]
    xval = datas[i][1][0]
    yval = datas[i][1][1]
    #xval = val_datas[j].loc[:,features]
    #yval = val_datas[j].loc[:,"action"]
    d_train = lgbm.Dataset(xtr,label=ytr)
    d_eval = lgbm.Dataset(xval,label=yval,reference=d_train)
    clf = lgbm.train(params,d_train,valid_sets=[d_train,d_eval],num_boost_round=1500,\
                    early_stopping_rounds=50,verbose_eval=50)
    clf.save_model("weights_{}".format(i))
    models.append(clf)


# import tensorflow as tf 
# from tensorflow.keras.layers import Input,Dense, Dropout,BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.activations import swish
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.optimizers import Adam
# SEED = 1111
# 
# def create_model(n_features,dr_rate,hidden_units,n_labels,label_smoothing,lr):
#     inp = Input(shape=(n_features,))
#     x = BatchNormalization()(inp)
#     x = Dropout(dr_rate[0])(x)
#     for i in range(len(hidden_units)):
#         x = Dense(hidden_units[0])(x)
#         x = BatchNormalization()(x)
#         x = swish(x)
#         x = Dropout(dr_rate[i+1])(x)
#     out = Dense(n_labels,activation="sigmoid")(x)
#     model = Model(inputs=inp,outputs=out)
#     
#     model.compile(loss = BinaryCrossentropy(label_smoothing=label_smoothing),\
#                   optimizer = Adam(learning_rate=lr),metrics= tf.keras.metrics.AUC(name="AUC"))
#     return model

# tf.random.set_seed(SEED)
# np.random.seed(SEED)
# models = []
# n_features = len(features)
# dr_rate = [0.2,0.2,0.2,0.2]
# hidden_units = [150,150,150]
# label_smoothing = 1e-2
# lr = 1e-3
# batch_size = 5000
# for i in range(len(datas)):
#     hist = []
#     j = (i+1)% 2
#     clf = create_model(n_features,dr_rate,hidden_units,1,label_smoothing,lr)
#     clf.fit(datas[i].values,target[i].values,validation_data = (datas[j].values,\
#                                                                 target[j].values),batch_size=batch_size,epochs=4)
#     
#     hist.append(clf)
#     models.append(hist[-1])

# In[53]:


fig,ax = plt.subplots(1,3,figsize=(10,20))
for i in range(3):
    lgbm.plot_importance(models[i],ax=ax[i])


# <h2 id=submission> Submission </h2>

# In[54]:


import janestreet
env = janestreet.make_env()


# In[55]:


th = 0.5000


# In[56]:


for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df["weight"].item() > 0 :
        x_tt = test_df.loc[:, features].values
        if np.isnan(x_tt.sum()):
           x_tt = np.nan_to_num(x_tt) + np.isnan(x_tt) * imputer
        pred = np.mean(np.stack([model.predict(x_tt) for model in models]),axis=0).T
        pred_df.action = np.where(pred >= th, 1, 0).astype(int)
    else :
        pred_df.action = 0
    
    env.predict(pred_df)


# <font color=red> <b> PLease leave your comments to enhance the work , or upvote if you like it !</font>
