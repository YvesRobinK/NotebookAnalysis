#!/usr/bin/env python
# coding: utf-8

# ## 이번 프로젝트의 흐름

# 1. 간단한 전처리만 수행한 데이터(결측치, categorical feature encoding)를 바로 Random Forest에 돌린 결과를 baseline으로 선정.
# (Linear Regression은 사용하지 않았습니다. 왜냐면 categorical feature와 binary feature가 많은 경우에 동작하지 않을 것을 어느정도 예상했기 때문입니다)
# 
# 
# 2. 3가지 feature engineering 방법을 사용합니다.
# - 우선 categorical feature는 모두 one-hot encoding을 합니다.
# 
# 
# > 2-1. Correlation check (threshold는 0.9로 세팅)
# 
# > 2-2. feature importance가 0.1을 넘는 feature만 선택
# 
# > 2-3. PCA로 90%를 보존하는 차원으로 차원감소
# 
# 
# 
# 3. 해당 모델마다 동일한 범위 내에서 optuna로 hyper-parameter tuning을 수행하여 성능을 비교합니다. 비교하여 성능이 높은 feature engineering 기법을 고릅니다.
# 
# > train-validation split은 9:1로 합니다.
# 
# 
# 4. LightGBM으로 모델을 변경 후, optuna로 hyper-parameter tuning을 열심히 돌립니다.
# 
# 
# 5. 최종 모델 선정

# ## 1. 라이브러리, 데이터 불러오기

# In[1]:


# 데이터분석 4종 세트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 모델들, 성능 평가
# (저는 일반적으로 정형데이터로 머신러닝 분석할 때는 이 2개 모델은 그냥 돌려봅니다. 특히 RF가 테스트하기 좋습니다.)
from sklearn.ensemble import RandomForestRegressor
from lightgbm.sklearn import LGBMRegressor

# KFold(CV), partial : optuna를 사용하기 위함
from sklearn.model_selection import KFold
from functools import partial

# hyper-parameter tuning을 위한 라이브러리, optuna
import optuna


# In[2]:


# flag setting
feature_reducing = "feature_importance" # "correlation" / "feature_importance" / "PCA"


# In[3]:


# 데이터를 불러옵니다.
train = pd.read_csv("../input/mercedes-benz-greener-manufacturing/train.csv.zip")
test = pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")
train


# ## 2. EDA

# In[4]:


# y의 분포
plt.figure(figsize=(12, 6))
sns.histplot(data=train, x="y")
plt.show()


# #### 찾은 특징들
# 
# 
# 1. 결측치 : 없음
# 
# 
# 2. dtype이 object인 column : X0 ~ X8까지 8개. (categorical feature)
# 
# > -> 어떻게 처리할지 고민해야함. (Ordinal Encoding VS One-Hot Encoding)
# 
# > -> categorical feature들은 종류 정보들이 알파벳으로 되어있으며(anomynized) 이 정보들 대비 target값의 차이가 있는지 확인.
# (특별하게 관련 없음)
# 
# > -> binary feature들중에서 0만 가지고 있는 column들이 있음.
# 
# > -> 정보가 충분하지 않다고 판단(target value와의 관련성 0) 삭제.
# 
# 
# 3. target distribution
# -> train data에 180을 넘는 데이터가 하나 있음. 이 데이터를 outlier라고 생각하고 제거.

# ### 3. 전처리

# #### 결측치 처리

# In[5]:


# 결측치가 있는 column
train[train.isnull().any(axis=1)]


# In[6]:


test[test.isnull().any(axis=1)]


# #### feature 구분
# 
# - X0 ~ X8 : categorical feature
# 
# - other features : binary feature(0 / 1)

# In[7]:


categorical_features = train.columns[2:10]
categorical_features


# In[8]:


temp = train.columns[10:]
temp


# In[9]:


card1 = train.columns[train.nunique() == 1]
card1


# In[10]:


binary_features = np.setdiff1d(temp, card1)
print("%d features - %d features = %d binary features" % (len(temp), len(card1), len(binary_features)))


# ### feature engineering
# 
# 1. Correlation
# 
# 
# 2. feature importance
# 
# 
# 3. PCA

# In[11]:


# feature engineering을 위해 tempX, y 생성
total = pd.concat([train, test])
split_point = len(train)
total_OHE = pd.get_dummies(data=total, columns=categorical_features)
y = train.y
tempX = total_OHE.drop(columns=["ID", "y"])
tempX = tempX.drop(columns=card1)
trainX = tempX[:split_point]
testX = tempX[split_point:]
print(trainX.shape, testX.shape, y.shape)


# In[12]:


trainX # sponge


# In[13]:


testX


# In[14]:


# 1. correlation

# 중복정보가 있는 column 제거하기 위해 상관계수를 확인해봅니다.
def remove_collinearity(X, threshold):
    """
    X : feature matrix
    threshold : 다중공선성을 제거할 column을 고르는 기준 값. [0, 1]
    """
    
    corr = X.corr()
    candidate_cols = []
    
    for x in corr.iterrows():
        idx, row = x[0], x[1] # decoupling tuple
        # 해당 row는 이미 처리가 되어서 볼 필요가 없다.
        if idx in candidate_cols:
            continue
        #print(row[row > 0.7].index[1:])
        candidates = row[row > threshold].index[1:]

        # 자기 자신을 제외하고 threshold를 넘는 column이 있다면,
        if len(candidates) != 0:
            for col in candidates:
                candidate_cols.append(col)           
    
    return candidate_cols

def find_feature_importance(X, model, show_plot):

    feat_names = X.columns.values
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()
    
    important_features = X.columns[importances >= 0.005]
    return important_features
    
def apply_PCA(X, show_plot):
    from sklearn.decomposition import PCA
    # training data와 test data를 모두 PCA를 이용하여 차원 감소를 수행합니다.
    pca = PCA(n_components=0.90) # 원래 데이터의 90%를 보존하는 차원.
    pca_090 = pca.fit(X) # 학습 및 변환
    reduced_X = pca_090.transform(X)
    print(reduced_X.shape)
    
    if show_plot:
        labels = [f"PC{x}" for x in range(1, reduced_X.shape[1]+1)]
        pca_090_variance = np.round(pca_090.explained_variance_ratio_.cumsum()*100, decimals=1)
        plt.figure(figsize=(25,5))
        plt.bar(x=range(1, len(pca_090_variance)+1), height=pca_090_variance, tick_label=labels)

        plt.xticks(rotation=90, color='indigo', size=15)
        plt.yticks(rotation=0, color='indigo', size=15)
        plt.title('Scree Plot',color='tab:orange', fontsize=25)
        plt.xlabel('Principal Components', {'color': 'tab:orange', 'fontsize':15})
        plt.ylabel('Cumulative percentage of explained variance ', {'color': 'tab:orange', 'fontsize':15})
        plt.show()
        
        X_train_pca_df = pd.DataFrame(reduced_X, columns=labels)
        display(X_train_pca_df)

    return pca_090, X_train_pca_df


# In[15]:


# PCA 적용
if feature_reducing == "correlation":
    threshold = 0.7
    correlated_features = remove_collinearity(trainX, threshold)
    correlated_features = set(correlated_features) # 중복 제거
    print("%d Correlation features over %.2f" % (len(correlated_features), threshold))
    
    X = trainX.drop(columns=correlated_features)
    print(X.shape)
    
elif feature_reducing == "feature_importance":
    show_plot = True
    model = RandomForestRegressor(max_features="sqrt", n_jobs=-1, random_state=0xC0FFEE)
    model.fit(trainX, y)
    important_features = find_feature_importance(trainX, model, show_plot)
    X = trainX[important_features]
    print(X.shape)
    
elif feature_reducing == "PCA":
    show_plot = True
    pca_model, X = apply_PCA(trainX, show_plot)
    print(X.shape)


# ### 4. 학습 데이터 분할

# In[16]:


# 첫번째 테스트용으로 사용하고, 실제 학습시에는 K-Fold CV를 사용합니다.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0xC0FFEE)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### 5. 학습 및 평가 (debugging 용도)

# In[17]:


print("\nFitting Random Forest...")
model = RandomForestRegressor(max_features='sqrt', n_jobs=-1)
model.fit(X_train, y_train)


# In[18]:


# metric은 그때마다 맞게 바꿔줘야 합니다.
from sklearn.metrics import r2_score
evaluation_metric = r2_score


# In[19]:


print("Prediction")
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)


train_score = evaluation_metric(y_train, pred_train)
test_score = evaluation_metric(y_test, pred_test)

print("Train Score : %.4f" % train_score)
print("Test Score : %.4f" % test_score)


# ### 6. Hyper-parameter Tuning

# - optuna를 갈아넣습니다!

# In[20]:


get_ipython().system('pip install plotly')


# In[21]:


# For Regression

def optimizer(trial, X, y, K):
    # 조절할 hyper-parameter 조합을 적어줍니다.
    n_estimators = trial.suggest_int("n_estimators", 50, 200)
    max_depth = trial.suggest_int("max_depth", 8, 30)
    max_features = trial.suggest_categorical("max_features", ['auto', 'sqrt', 'log2'])
    
    
    # 원하는 모델을 지정합니다, optuna는 시간이 오래걸리기 때문에 저는 보통 RF로 일단 테스트를 해본 뒤에 LGBM을 사용합니다.
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  max_features=max_features,
                                  n_jobs=-1,
                                  random_state=0xC0FFEE)
    
    
    # K-Fold Cross validation을 구현합니다.
    folds = KFold(n_splits=K)
    scores = []
    
    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = evaluation_metric(y_val, preds)
        scores.append(score)
    
    
    # K-Fold의 평균 loss값을 돌려줍니다.
    return np.mean(scores)


# In[22]:


K = 5 # Kfold 수
opt_func = partial(optimizer, X=X_train, y=y_train, K=K)

rf_study = optuna.create_study(study_name="RF", direction="maximize") # regression task에서 R^2를 최대화!
rf_study.optimize(opt_func, n_trials=15)


# In[23]:


# optuna가 시도했던 모든 실험 관련 데이터
rf_study.trials_dataframe()


# In[24]:


print("Best Score: %.4f" % rf_study.best_value) # best score 출력
print("Best params: ", rf_study.best_trial.params) # best score일 때의 하이퍼파라미터들


# In[25]:


# 실험 기록 시각화
optuna.visualization.plot_optimization_history(rf_study)


# In[26]:


# hyper-parameter들의 중요도
optuna.visualization.plot_param_importances(rf_study)


# ### 7. 테스트 및 제출 파일 생성

# In[27]:


final_rf_model = RandomForestRegressor(n_estimators=rf_study.best_trial.params["n_estimators"],
                                 max_depth=rf_study.best_trial.params["max_depth"],
                                 max_features=rf_study.best_trial.params["max_features"])

final_rf_model.fit(X, y) # finalize model


# In[28]:


testX


# In[29]:


# PCA 적용
if feature_reducing == "correlation":
    test = testX.drop(columns=correlated_features)
    print(X.shape)
    
elif feature_reducing == "feature_importance":
    test = testX[important_features]
    print(X.shape)
    
elif feature_reducing == "PCA":
    test = pca_model.transform(testX)
    print(X.shape)


# In[30]:


prediction = final_rf_model.predict(test)
prediction


# In[31]:


submission = pd.read_csv("../input/mercedes-benz-greener-manufacturing/sample_submission.csv.zip")
submission


# In[32]:


submission["y"] = prediction
submission


# In[33]:


submission.reset_index(drop=True).to_csv(f"rf_submission_{feature_reducing}.csv", index=False)


# ### 9. LightGBM으로 변경!

# Reference : https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

# In[34]:


# For Regression

def optimizer(trial, X, y, K):
    
    import os
    
    param = {
        'objective': 'regression', # 회귀
        'verbose': 0, 
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1e-2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1),
    }

    model = LGBMRegressor(**param, n_jobs=os.cpu_count())
    
    # K-Fold Cross validation을 구현합니다.
    folds = KFold(n_splits=K)
    scores = []
    
    for train_idx, val_idx in folds.split(X, y):
        X_train = X.iloc[train_idx, :]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx, :]
        y_val = y.iloc[val_idx]
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=25)
        preds = model.predict(X_val)
        score = evaluation_metric(y_val, preds)
        scores.append(score)
    
    
    # K-Fold의 평균 loss값을 돌려줍니다.
    return np.mean(scores)


# In[35]:


K = 5 # Kfold 수
opt_func = partial(optimizer, X=X_train, y=y_train, K=K)

lgbm_study = optuna.create_study(study_name="LGBM", direction="maximize") # regression task에서 R^2를 최대화!
lgbm_study.optimize(opt_func, n_trials=15)


# In[36]:


# optuna가 시도했던 모든 실험 관련 데이터
lgbm_study.trials_dataframe()


# In[37]:


print("Best Score: %.4f" % lgbm_study.best_value) # best score 출력
print("Best params: ", lgbm_study.best_trial.params) # best score일 때의 하이퍼파라미터들


# In[38]:


# 실험 기록 시각화
optuna.visualization.plot_optimization_history(lgbm_study)


# In[39]:


# hyper-parameter들의 중요도
optuna.visualization.plot_param_importances(lgbm_study)


# In[40]:


trial = lgbm_study.best_trial
trial_params = trial.params

final_lgb_model = LGBMRegressor(**trial_params)
final_lgb_model.fit(X, y) # finalize model


# In[41]:


# PCA 적용
if feature_reducing == "correlation":
    test = testX.drop(columns=correlated_features)
    print(X.shape)
    
elif feature_reducing == "feature_importance":
    test = testX[important_features]
    print(X.shape)
    
elif feature_reducing == "PCA":
    test = pca_model.transform(testX)
    print(X.shape)
    
prediction = final_lgb_model.predict(test)
submission["y"] = prediction
display(submission)
submission.reset_index(drop=True).to_csv(f"lgbm_submission_{feature_reducing}.csv", index=False)


# In[ ]:




