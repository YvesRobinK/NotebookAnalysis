#!/usr/bin/env python
# coding: utf-8

# <div style="color: #fff7f7;
#            display:fill;
#            border-radius:10px;
#            border-style: solid;
#            border-color:#424949;
#            text-align:center;
#            background-color:#003a6c ;
#            font-size:15px;
#            letter-spacing:0.5px;
#            padding: 0.7em;
#            text-align:left" >
#     <h1 style="text-align:center;font-weight: 20px; color:White;">
#        Feature Processing and Model Selection of Titanic😁</h1>
# </div>

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
sns.set_theme()


# In[2]:


df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/test.csv')
data = df_train.append(df_test)


# In[3]:


df_train.head()


# In[4]:


df_train.isnull().sum()


# In[5]:


df_test.isnull().sum()


# In[6]:


df_train.duplicated().sum()


# In[7]:


print("Training set size",len(df_train))
print("Test set size",len(df_train))


# In[8]:


data


# # 1 EDA
# <div style="color: #fff7f7;
#            display:fill;
#            border-radius:10px;
#            border-style: solid;
#            border-color:#424949;
#            text-align:center;
#            background-color:#003a6c ;
#            font-size:15px;
#            letter-spacing:0.5px;
#            padding: 0.7em;
#            text-align:left">
# <center style="font-size:30px">Some things that need to be explained🍅</center>
# <hr>特征处理主要包括以下几个方面：
# <li>1.年龄和票价的分箱数量要尽量多一点。
# <li>2.通过名字/票价两个特征进行配合，构建一个表示家庭死亡状态的特征。
# <li>3.Cabin列应该留下，用其他值标识缺失值，它显著的提升了0.2分。<hr>
# <li>1. The number of bins for age and fare should be as large as possible.
# <li>2. The two features of name/fare are combined to construct a feature that represents the death state of the family.
# <li>3. The Cabin column should be left, and other values are used to identify missing values, which significantly improves by 0.2 points.
# </div>

# In[9]:


g = sns.FacetGrid(data[:891],col="Pclass")
g.map_dataframe(sns.pointplot,x="Sex",y="Survived")


# - 生存比例跟阶层存在关系，在1/2阶级中，男性的存活比例要超过女性。但是在3阶级中，女性的存活比例要超过男性。
# - The survival rate has a relationship with the class. In the 1/2 class, the survival rate of men exceeds that of women. But in the three classes, the survival rate of women exceeds that of men.

# In[10]:


g = sns.FacetGrid(data[:891],col="Pclass",hue="Survived")
g.map_dataframe(sns.kdeplot,x="Fare")
g.add_legend()


# - 我们注意到，1阶级的人票价分布比较平缓，且存在很高的票价，这说明票价存在严重的右偏。
# - We have noticed that the fare distribution of class 1 people is relatively flat, and there are high fares, which shows that the fares are seriously skewed to the right.

# In[11]:


g = sns.FacetGrid(data[:891],col="Sex",hue="Survived")
g.map_dataframe(sns.histplot,x="Age")
g.add_legend()


# - 无论是男性还是女性，儿童或者未成年活下来的比例都挺高。
# - 从总体来看，女性生存的比例要超过男性，这是毋庸置疑的。
# -Whether it is male or female, the percentage of children or minors surviving is quite high.
# -From an overall point of view, there is no doubt that the proportion of women surviving is higher than that of men.

# In[12]:


g = sns.FacetGrid(data[:891],col="Sex")
g.map_dataframe(sns.pointplot,x="Embarked",y="Survived")
g.add_legend()


# - 登船港口的生存比例比较平缓
# - The survival rate at the embarkation port is relatively flat

# In[13]:


g = sns.FacetGrid(data[:891],col="Embarked")
g.map_dataframe(sns.histplot,x="Age",y="Survived")


# - C港口存活下来的人集中在20岁左右。
# - S港口死亡的人集中在20岁左右。
# - Q港口死亡的人集中在35岁左右。
# -The survivors of Port C are concentrated in their 20s.
# -The deaths in Port S are concentrated in their 20s.
# -People who died at Port Q were mostly around 35 years old.

# In[14]:


data["family"] = data["Parch"] + data["SibSp"]
data = data.drop(columns=["Parch","SibSp"])  #delete  Parch and Sibsp
#-------------------------------------------------
g = sns.FacetGrid(data[:891],col="Sex")
g.map_dataframe(sns.pointplot,x="family",y="Survived")


# - 当独自一个人旅游或者携带家人/朋友的人数超过7个人时，生存率会很低。无论是男性还是女性群体，3个人旅游的生存比例最高。
# - 这说明家人的数量也是存活的重要影响因素。
# - When traveling alone or when there are more than 7 people with family/friends, the survival rate will be very low. Regardless of whether it is a male or a female group, the three persons have the highest survival rate in tourism.
# - This shows that the number of family members is also an important factor in survival.

# # 2 Feature engineering 👍
# <div style="color: #fff7f7;
#            display:fill;
#            border-radius:10px;
#            border-style: solid;
#            border-color:#424949;
#            text-align:center;
#            background-color:#003a6c ;
#            font-size:15px;
#            letter-spacing:0.5px;
#            padding: 0.7em;
#            text-align:left">
# <li> Cabin该列缺失比例过高，我们有三个选择：考虑删除，或者增加一个新的特征，表示是否存在缺失值，或者在原有的基础上，将缺失值标识为Miss。
# <li> Name列很有用，一方面我们可以提取每个人的称谓，根据称谓来填充年龄缺失值，另一方面提取每个人的名字，跟后续的票号/票价联系起来，可以确定部分人的家庭存活情况。
# <li> 我们可以考虑对年龄/票价进行分箱，至于选择分箱的类型，可以参考等宽/等频分箱。如果是使用跟距离相关的算法，分箱后会很有用，或者对特征进行缩放。<hr>
# <li>The percentage of missing values in Cabin column is too high. We have three options: consider deleting, or adding a new feature to indicate whether there are missing values, or mark the missing value as Miss on the original basis.
# <li>The Name column is very useful. On the one hand, we can extract the title of each person and fill in missing age values based on the title. On the other hand, we can extract the name of each person and link it with the subsequent ticket number/fare to determine the survival of some people’s families. condition.
# <li>We can consider classifying the age/fare. As for the type of classification, please refer to the equal width/equal frequency classification. If you are using a distance-related algorithm, it will be useful after binning or scaling features.
# </div>

# ### 2.1 填充缺失值 Fill in missing values

# In[15]:


from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=3)
data["Age"] = knn.fit_transform(data["Age"].values.reshape(-1,1)).ravel()


# In[16]:


data["Fare"] = data["Fare"].fillna(data[:891]["Fare"].mode()[0])
data["Embarked"] = data["Embarked"].fillna(data[:891]["Embarked"].mode()[0])


# In[17]:


data.head()


# ### 2.2 Name and Ticket---add family_survived

# In[18]:


import re
data["Ticket_num"] = data["Ticket"].apply(lambda x:"".join(re.findall(r"\d+",x)))
data["Name_last"] = data["Name"].apply(lambda x:x.split(",",1)[0])


# In[19]:


#设置一个默认值，可以设置为0.5，或者-1，只要区别于0和1即可。由于0和1的均值是0.5，所以我们设置为0.5
#通过名字和票价确定家庭
Def = 0.5
data["family_survived"] = Def

#通过名字和票价确定家庭
da_group = data.groupby(["Name_last","Fare"])
for group,all_data in da_group:
    if (len(all_data) >= 1): #是否是一个家庭。Is it a family？
            for index, row in all_data.iterrows():
                if (row['family_survived'] == 0) or (row['family_survived']== 0.5):
                    surviv_max = all_data.drop(index)['Survived'].max()
                    surviv_min = all_data.drop(index)['Survived'].min()   #删除本人的记录，查看家庭其他人的生存状况。Delete my records and view the living conditions of others in the family.
                    ids = row['PassengerId']
                    if (surviv_max == 1.0):
                        data.loc[data['PassengerId'] == ids, 'family_survived'] = 1
                    elif (surviv_min==0.0):
                        data.loc[data['PassengerId'] == ids, 'family_survived'] = 0

print("family survival:", data.loc[data['family_survived'] == 1].shape[0])


# In[20]:


#通过名字和票号确定家庭
da_group = data[["PassengerId","Name_last","Survived","family","Fare","Ticket_num"]].groupby(["Name_last","Ticket_num"])
for group,all_data in da_group:
    if (len(all_data) >= 1): #判断是否是一个家庭。Is it a family？
        for index, row in all_data.iterrows():
            surviv_max = all_data.drop(index)['Survived'].max()
            surviv_min = all_data.drop(index)['Survived'].min()   #删除本人的记录，查看家庭其他人的生存状况。Delete my records and view the living conditions of others in the family.
            ids = row['PassengerId']
            if (surviv_max == 1.0):
                data.loc[data['PassengerId'] == ids, 'family_survived'] = 1
            elif (surviv_min==0.0):
                data.loc[data['PassengerId'] == ids, 'family_survived'] = 0

print("family survival:", data.loc[data['family_survived'] == 1].shape[0])


# In[21]:


#通过票号相同确定家庭
da_group = data.groupby(["Ticket_num"])
for group,all_data in da_group:
    if (len(all_data) >= 1): #是否是一个家庭。Is it a family？
            for index, row in all_data.iterrows():
                if (row['family_survived'] == 0) or (row['family_survived']== 0.5):
                    surviv_max = all_data.drop(index)['Survived'].max()
                    surviv_min = all_data.drop(index)['Survived'].min()   #删除本人的记录，查看家庭其他人的生存状况。Delete my records and view the living conditions of others in the family.
                    ids = row['PassengerId']
                    if (surviv_max == 1.0):
                        data.loc[data['PassengerId'] == ids, 'family_survived'] = 1
                    elif (surviv_min==0.0):
                        data.loc[data['PassengerId'] == ids, 'family_survived'] = 0

print("family survival:", data.loc[data['family_survived'] == 1].shape[0])


# In[22]:


data = data.drop(columns=["Name","Ticket","Ticket_num","Name_last"])


# In[23]:


data.head()


# ### 2.3 Age and Fare

# In[24]:


data["Age"] = pd.qcut(data["Age"],7)


# In[25]:


data["Fare"] = pd.qcut(data["Fare"],7)


# In[26]:


data["Age"]


# ### 2.4 Cabin

# In[27]:


data["Cabin"] = data["Cabin"].apply(lambda x:str(x)[0])


# In[28]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
# col = ["Sex","Age","Fare","Embarked","Cabin"]  for循环的方式出现错误。换一种办法。
# for i in col:
#     la = LabelEncoder()
#     la.fit(data.loc[:,i].values.reshape(-1, 1))
#     data[i] = la.transform(data.loc[:,i].values.reshape(-1, 1))
la = LabelEncoder()
la.fit(data["Sex"])
data["Sex"] = la.transform(data["Sex"])

la.fit(data["Age"])
data["Age"] = la.transform(data["Age"])

la.fit(data["Fare"])
data["Fare"] = la.transform(data["Fare"])

la.fit(data["Embarked"])
data["Embarked"] = la.transform(data["Embarked"])

la.fit(data["Cabin"])
data["Cabin"] = la.transform(data["Cabin"])

all_col = ["Sex","Age","Fare","Embarked","Pclass","family","family_survived","Cabin"]
for i in all_col:
    std = StandardScaler()
    std.fit(data[:891].loc[:,i].values.reshape(-1, 1)) #traindata
    data[i] = std.transform(data.loc[:,i].values.reshape(-1, 1))


# In[29]:


data.head()


# In[30]:


df_train = data[:891]
df_test = data[891:]


# # 3 Model

# In[31]:


df_train["Survived"].value_counts(normalize=True)


# In[32]:


from sklearn.model_selection import StratifiedKFold,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score
from xgboost import XGBClassifier


# In[33]:


target = df_train.iloc[:,1]
feature = df_train.iloc[:,2:]
stk = StratifiedKFold(n_splits=5,random_state=123,shuffle=True)
for train_index,test_index in stk.split(feature,target):
    x_train,y_train = feature.iloc[train_index],target.iloc[train_index]
    x_test,y_test = feature.iloc[test_index],target.iloc[test_index]


# #### 3.1 GBDT

# In[34]:


#Deault
gbdt = GradientBoostingClassifier(random_state=123)
gbdt.fit(x_train,y_train)
gbdt_pre = gbdt.predict(x_test)
print("gbdt recall:",round(recall_score(y_test,gbdt_pre),2))
print("gbdt precision:",round(precision_score(y_test,gbdt_pre),2))
print("gbdt f1_score:",round(f1_score(y_test,gbdt_pre),2))
print("gbdt rou_auc_score:",round(roc_auc_score(y_test,gbdt_pre),2))


# In[35]:


params = {
    
    "max_depth":(2,8),
    "learning_rate":(0.01,0.1),
    "min_samples_leaf":(1,3),
    "n_estimators":(100,300)
}
gbdt_gd = RandomizedSearchCV(gbdt,param_distributions=params,cv=5,scoring="roc_auc")
gbdt_gd.fit(x_train,y_train)
print(gbdt_gd.best_score_)
print(gbdt_gd.best_estimator_)
print(gbdt_gd.best_params_)


# In[36]:


gbdt_pre = gbdt_gd.predict(x_test)
print("gbdt_gd recall:",round(recall_score(y_test,gbdt_pre),2))
print("gbdt_gd precision:",round(precision_score(y_test,gbdt_pre),2))
print("gbdt_gd f1_score:",round(f1_score(y_test,gbdt_pre),2))
print("gbdt_gd rou_auc_score:",round(roc_auc_score(y_test,gbdt_pre),2))


# #### 3.2 Adaboost

# In[37]:


#Deault
adaboost = AdaBoostClassifier(random_state=123)
adaboost.fit(x_train,y_train)
adaboost_pre = gbdt.predict(x_test)
print("adaboost recall:",round(recall_score(y_test,adaboost_pre),2))
print("adaboost precision:",round(precision_score(y_test,adaboost_pre),2))
print("adaboost f1_score:",round(f1_score(y_test,adaboost_pre),2))
print("adaboost rou_auc_score:",round(roc_auc_score(y_test,adaboost_pre),2))


# In[38]:


params = {
    
    "learning_rate":(0.01,0.2),
    "n_estimators":(50,300)
}
ada_gd = RandomizedSearchCV(adaboost,param_distributions=params,cv=5,scoring="roc_auc")
ada_gd.fit(x_train,y_train)
print(ada_gd.best_score_)
print(ada_gd.best_estimator_)
print(ada_gd.best_params_)


# In[39]:


ada_pre = ada_gd.predict(x_test)
print("ada_gd recall:",round(recall_score(y_test,ada_pre),2))
print("ada_gd precision:",round(precision_score(y_test,ada_pre),2))
print("ada_gd f1_score:",round(f1_score(y_test,ada_pre),2))
print("ada_gd rou_auc_score:",round(roc_auc_score(y_test,ada_pre),2))


# ### 3.3 RF

# In[40]:


rf = RandomForestClassifier(random_state=123)
rf.fit(x_train,y_train)
rf_pre = rf.predict(x_test)
print("rf recall:",round(recall_score(y_test,rf_pre),2))
print("rf precision:",round(precision_score(y_test,rf_pre),2))
print("rf f1_score:",round(f1_score(y_test,rf_pre),2))
print("rf rou_auc_score:",round(roc_auc_score(y_test,rf_pre),2))


# In[41]:


params = {
    "max_depth":(3,10),
    "n_estimators":(50,300),
    "min_samples_leaf":(1,5),
    "min_samples_split":(0.1,0.2),
    "min_impurity_decrease":(0,0.2),
}
rf_gd = RandomizedSearchCV(rf,param_distributions=params,cv=5,scoring="roc_auc")
rf_gd.fit(x_train,y_train)
print(rf_gd.best_score_)
print(rf_gd.best_estimator_)
print(rf_gd.best_params_)


# In[42]:


rfgd_pre = rf_gd.predict(x_test)
print("rf_gd recall:",round(recall_score(y_test,rfgd_pre),2))
print("rf_gd precision:",round(precision_score(y_test,rfgd_pre),2))
print("rf_gd f1_score:",round(f1_score(y_test,rfgd_pre),2))
print("rf_gd rou_auc_score:",round(roc_auc_score(y_test,rfgd_pre),2))


# ### 3.4 KNN

# In[43]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn_pre = knn.predict(x_test)
print("knn recall:",round(recall_score(y_test,knn_pre),2))
print("knn precision:",round(precision_score(y_test,knn_pre),2))
print("knn f1_score:",round(f1_score(y_test,knn_pre),2))
print("knn rou_auc_score:",round(roc_auc_score(y_test,knn_pre),2))


# In[44]:


params = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'], 'leaf_size': list(range(1,50,5)), 
               'n_neighbors': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22]}
knn_gd=GridSearchCV(knn, param_grid = params, verbose=True, 
                cv=10, scoring = "roc_auc")
knn_gd.fit(x_train, y_train)
print(knn_gd.best_score_)
print(knn_gd.best_estimator_)
print(knn_gd.best_params_)


# In[45]:


knngd_pre = knn_gd.predict(x_test)
print("knngd_pre recall:",round(recall_score(y_test,knngd_pre),2))
print("knngd_pre precision:",round(precision_score(y_test,knngd_pre),2))
print("knngd_pre f1_score:",round(f1_score(y_test,knngd_pre),2))
print("knngd_pre rou_auc_score:",round(roc_auc_score(y_test,knngd_pre),2))


# ### 3.5 XGboost

# In[46]:


#Default
xgb = XGBClassifier(random_state=123,eval_metric="logloss",use_label_encoder=False)
xgb.fit(x_train,y_train)
xgb_pre = xgb.predict(x_test)
print("xgb recall:",round(recall_score(y_test,xgb_pre),2))
print("xgb precision:",round(precision_score(y_test,xgb_pre),2))
print("xgb f1_score:",round(f1_score(y_test,xgb_pre),2))
print("xgb rou_auc_score:",round(roc_auc_score(y_test,xgb_pre),2))


# In[47]:


params ={
    "max_depth":(2,8),
    "n_estimators":(100,400),
    "learning_rate":(0.5,1),
    "gamma":(0,0.2),
    "reg_lambda":(0.1,0.3),
    "reg_alpha":(0.1,0.3),
    "colsample_bytree":(0.8,1),
}

xgb_gd = RandomizedSearchCV(xgb,param_distributions=params,cv=5,scoring="roc_auc")
xgb_gd.fit(x_train,y_train)
print(xgb_gd.best_score_)
print(xgb_gd.best_estimator_)
print(xgb_gd.best_params_)


# In[48]:


xgbgd_pre = xgb_gd.predict(x_test)
print("xgb_gd recall:",round(recall_score(y_test,xgbgd_pre),2))
print("xgb_gd precision:",round(precision_score(y_test,xgbgd_pre),2))
print("xgb_gd f1_score:",round(f1_score(y_test,xgbgd_pre),2))
print("xgb_gd rou_auc_score:",round(roc_auc_score(y_test,xgbgd_pre),2))


# ### 3.6 Logistic

# In[49]:


logstic = LogisticRegression(random_state=123)
logstic.fit(x_train,y_train)
logstic_pre = logstic.predict(x_test)
print("logstic recall:",round(recall_score(y_test,logstic_pre),2))
print("logstic precision:",round(precision_score(y_test,logstic_pre),2))
print("logstic f1_score:",round(f1_score(y_test,logstic_pre),2))
print("logstic rou_auc_score:",round(roc_auc_score(y_test,logstic_pre),2))


# In[50]:


prams = {
    
    "C":[0.1,0.2,0.3,0.4,0.5],
    "solver":["newton-cg","lbfgs","sag","saga"]
}

logstic_gd=GridSearchCV(logstic, param_grid = prams, verbose=True, 
                cv=10, scoring = "roc_auc")
logstic_gd.fit(x_train, y_train)
print(logstic_gd.best_score_)
print(logstic_gd.best_estimator_)
print(logstic_gd.best_params_)


# In[51]:


logsticgd_pre = logstic_gd.predict(x_test)
print("logstic recall:",round(recall_score(y_test,logsticgd_pre),2))
print("logstic precision:",round(precision_score(y_test,logsticgd_pre),2))
print("logstic f1_score:",round(f1_score(y_test,logsticgd_pre),2))
print("logstic rou_auc_score:",round(roc_auc_score(y_test,logsticgd_pre),2))


# In[52]:


df_t = df_test.iloc[:,2:]
ids = df_test.iloc[:,0]
df_t


# - soft VotingClassifier

# In[53]:


# from sklearn.ensemble import VotingClassifier
# esti = [("gbdt",gbdt_gd),("rf",rf_gd),("adaboost",ada_gd),("knn",knn_gd),("xgb",xgb_gd)]
# softvoting = VotingClassifier(estimators=esti,voting="soft")
# softvoting.fit(x_train, y_train)
# softvoting_pre = softvoting.predict(x_test)
# print("softvoting recall:",round(recall_score(y_test,softvoting_pre),2))
# print("softvoting precision:",round(precision_score(y_test,softvoting_pre),2))
# print("softvoting f1_score:",round(f1_score(y_test,softvoting_pre),2))
# print("softvoting rou_auc_score:",round(roc_auc_score(y_test,softvoting_pre),2))


# - hard VotingClassifier

# In[54]:


# voting = VotingClassifier(estimators=esti,voting="hard")
# voting.fit(x_train, y_train)
# voting_pre = voting.predict(x_test)
# print("hard voting recall:",round(recall_score(y_test,voting_pre),2))
# print("hard voting precision:",round(precision_score(y_test,voting_pre),2))
# print("hard voting f1_score:",round(f1_score(y_test,voting_pre),2))
# print("hard voting rou_auc_score:",round(roc_auc_score(y_test,voting_pre),2))

#硬投票的效果不如软投票


# - Stacking

# In[55]:


# esti = [("gbdt",gbdt_gd),("rf",rf_gd),("adaboost",ada_gd),("knn",knn_gd),("xgb",xgb_gd)]
# stack = StackingClassifier(estimators=esti,final_estimator=logstic_gd,cv=5)
# stack.fit(x_train, y_train)
# stack_pre = voting.predict(x_test)
# print("stack recall:",round(recall_score(y_test,stack_pre),2))
# print("stack precision:",round(precision_score(y_test,stack_pre),2))
# print("stack f1_score:",round(f1_score(y_test,stack_pre),2))
# print("stack rou_auc_score:",round(roc_auc_score(y_test,stack_pre),2))

#Stack不如软投票


# In[56]:


sub = pd.DataFrame()
sub["PassengerId"] = ids
sub["Survived"] = knn_gd.predict(df_t)
sub["Survived"] = sub["Survived"].astype(int)
sub


# In[57]:


sub.to_csv("submission.csv",index=None)
print("End!")


# <div style="color: #fff7f7;
#            display:fill;
#            border-radius:10px;
#            border-style: solid;
#            border-color:#424949;
#            text-align:center;
#            background-color:#003a6c ;
#            font-size:15px;
#            letter-spacing:0.5px;
#            padding: 0.7em;
#            text-align:left">
#     <h1 style="text-align:center;font-weight: 20px; color:White;">
#        Result🧐</h1>
#     <li>当我删除Cabin时，提交的模型分数是0.79，当我加入Cabin时，将缺失值表示为N,提交的模型分数达到了0.80382。
#     <li>当我将年龄和票价分箱调整为5个时，模型分数下降到0.78。然而当我调整为7个分箱时，模型分数达到0.80382。
#     <li>软投票以及硬投票还有模型堆叠Stack的方式，效果不如KNN，软投票等方式达到的分数是0.78。但是在软投票中删除逻辑回归的模型，提交的分数可以达到0.80143。<hr>
#     <li>When I deleted Cabin, the submitted model score was 0.79. When I joined Cabin, the missing value was expressed as N, and the submitted model score reached 0.80382.
#     <li>When I adjusted the age and fare bins to 5, the model score dropped to 0.78. However, when I adjusted to 7 bins, the model score reached 0.80382.
#     <li>Soft voting and hard voting, as well as the way of stacking models, are not as effective as KNN, and the score achieved by soft voting is 0.78. But if the logistic regression model is deleted in the soft voting, the submitted score can reach 0.80143.<hr>
#     可能需要改进的地方：
# <li>针对年龄缺失值填充，我们可以考虑使用阶级特征或者提取名字特征中的称谓，对年龄来进行缺失值填充，而不是使用暴力的KNN填补的方式。
# <li>家庭死亡特征：我们可以不仅可以通过Ticket和Fare以及Name-last，或许我们还可以使用Cabin来综合判断，毕竟相伴出游的家庭，肯定也会从相同的Embarked登船。<hr>
# Possible areas for improvement:
# <li> For the filling of missing values for age, we can consider using class features or extracting titles from name features to fill in missing values for age instead of using violent KNN filling.
# <li>Family death characteristics: We can not only use Ticket, Fare and Name-last, but perhaps we can also use Cabin to make a comprehensive judgment. After all, families traveling with us will definitely board the same Embarked.
# </div>
