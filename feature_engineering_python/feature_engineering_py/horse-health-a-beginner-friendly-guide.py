#!/usr/bin/env python
# coding: utf-8

# # Introduction 💥
# ***
# Welcome to this notebook, where we will explore the data from the Horse Health competition, which is about predicting the health status of horses using various features.
# 
# > Beacuse I got late to this party this notebook is not focused on getting a top score, but rather on creating a **notebook that is profoundly explained for beginners starting on Kaggle**.
# 
# By taking time to read about the data science fundamentals I’ve dusted off a bit of old knowledge. **So I hope that even the seasoned Kagglers can take something from this simple notebook.** With that said join me on this journey as we discover how data science can help us improve the well-being of our equine friends.
# 
# ## Table of Contents
# 1. [Libraries 📖](#Libraries-📖)
# 2. [Load data 📂](#Load-data-📁)
# 3. [Data analysis 📊](#Data-analysis-📊)
# 4. [Feature engineering 🛠️](#Feature-engineering-🛠️)
#     1. [Drop lesion_3](#Drop-lesion_3)
# 5. [Preprocessing ⚙️](#Preprocessing-⚙️)
#     1. [Splitting dataset](#Splitting-dataset)
#     2. [Pipelines](#Pipelines)
#     3. [Main transformer](#Main-transformer)
#     2. [Target encoder](#Target-encoder)
#     3. [Transforming](#Transforming)
#     4. [Results](#Results)
# 6. [Modeling 🪄](#Modeling-🪄)
#     1. [Splitting training set](#Model-training)
#     2. [Hyperparameter optimization](#Model-training)
#     3. [Models](#Models)
#     4. [Model evaluation](#Model-evaluation)
#     5. [Model training](#Model-training)
#     6. [Feature importance](#Feature-importance)
# 7. [Submission 🏆](#Submission-🏆)
# 8. [Thank you ✨](#Thank-you-✨)

# # Libraries 📖
# ***

# In[1]:


# Holy grail
import pandas as pd
import numpy as np

# Sklearn
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score # evaluation metric used for leaderboard scoring in this competition

# Statistics
import scipy.stats as stats

# Visualization
from matplotlib import pyplot as plt # pyplot is an easy to use scripting interface for plotting as oppose to more advanced artistic interface
import seaborn as sns # seaborn is even higher level graphing library built on top of matplotlib

# Machine learning
import optuna # used for finding good hyperparameters for a model
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# # Load data 📁
# ***

# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv', index_col='id')


# In[3]:


original = pd.read_csv('/kaggle/input/horse-survival-dataset/horse.csv')

train = pd.concat([train, original], ignore_index=True).drop_duplicates()


# # Data analysis 📊
# ***

# In[4]:


train.head()


# In[5]:


train.info()


# <div style="border-radius: 10px; border: #27374D solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📉 Columns types :</b>
#     <ul>
#         <li><b>11</b> numeric columns</li>
#         <li><b>17</b> categorical columns</li>
#     </ul>
# </div>

# In[6]:


train.isna().sum().sort_values()


# <div style="border-radius: 10px; border: #ffac00 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b> ⚠️ Missing values:</b> We will have to keep <i>rectal_exam_feces</i> and <i>abdomen</i> features in mind because they have significantly more missing data that the rest.
# </div>

# In[7]:


num = train.select_dtypes(include=['int64','float64']).columns

df = pd.concat([train[num].assign(Source = 'Train'), test[num].assign(Source = 'Test')], ignore_index = True)

# Use of more advanced artistic matplotlib interface (see the axes)
fig, axes = plt.subplots(len(num), 3 ,figsize = (16, len(num) * 4), gridspec_kw = {'hspace': 0.35, 'wspace': 0.3, 'width_ratios': [0.80, 0.20, 0.20]})

for i,col in enumerate(num):
    ax = axes[i,0]
    sns.kdeplot(data = df[[col, 'Source']], x = col, hue = 'Source', palette=['#456cf0', '#ed7647'], linewidth = 2.1, warn_singular=False, ax = ax) # Use of seaborn with artistic interface
    ax.set_title(f"\n{col}",fontsize = 9)
    ax.grid(visible=True, which = 'both', linestyle = '--', color='lightgrey', linewidth = 0.75)
    ax.set(xlabel = '', ylabel = '')

    ax = axes[i,1]
    sns.boxplot(data = df.loc[df.Source == 'Train', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#456cf0', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Train", fontsize = 9)

    ax = axes[i,2]
    sns.boxplot(data = df.loc[df.Source == 'Test', [col]], y = col, width = 0.25, linewidth = 0.90, fliersize= 2.25, color = '#ed7647', ax = ax)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title("Test", fontsize = 9)

plt.suptitle(f'\nDistribution analysis - numerical features\ndash by YANG ZHOU\n',fontsize = 12, y = 0.9, x = 0.57)
plt.show()


# <div style="border-radius: 10px; border: #00d65c solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>✅ Great :</b> Distributions of individual numerical features are very close to each other so there doesn't seem to be any data drift!
# </div>
# 
# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left; margin-top: 10px;">
#     <b>📔 Matplotlib layers:</b> Matplotlib provides two main interfaces for creating plots: the pyplot interface (also known as the scripting layer) and the object interface (also known as the artist layer). Object interface used in above dash provides more control and flexibility, allowing you to create more complex and customized charts. It requires you to create a figure and an axis explicitly. Basically if you ever see <code>ax</code> or <code>fig</code> used then you know your dealing with a artistic layer.
# </div>

# In[8]:


sns.set_palette('rainbow')

num = train.drop(columns='outcome').select_dtypes(include=['object']).columns

df = pd.concat([train[num].assign(Source = 'train'), 
                test[num].assign(Source = 'test')], 
               axis=0, ignore_index = True)

fig, axes = plt.subplots(len(num), 2 ,figsize = (12, len(num) * 4.2))

for i,col in enumerate(num):
    train_dist = df.loc[df.Source == 'train', [col]].value_counts()
    test_dist = df.loc[df.Source == 'test', [col]].value_counts()
    
    ax = axes[i,0]
    ax.pie(train_dist, shadow=True, explode=[.05]*len(train_dist), autopct='%.1f%%')
    ax.legend([category[0] for category in train_dist.index], loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title(f'Train {col}',fontsize = 9)

    ax = axes[i,1]
    ax.pie(test_dist, shadow=True, explode=[.05]*len(test_dist), autopct='%.1f%%')
    ax.legend([category[0] for category in test_dist.index], loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    ax.set(xlabel = '', ylabel = '')
    ax.set_title(f'Test {col}',fontsize = 9)


plt.suptitle(f"\nDistribution analysis - categorical features\n",fontsize = 15, y = 0.9, x = 0.57)
plt.show()


# <div style="border-radius: 10px; border: #00d65c solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>✅ Great :</b> Distributions of individual categorical columns have almost identical distributions so there is no data drift!
# </div>

# In[9]:


target_dist = train['outcome'].value_counts()

plt.pie(target_dist, shadow=True, explode=[.05,.05, .05], autopct='%.1f%%')

plt.title('Target distribution', size=18)
plt.legend(target_dist.index, loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

plt.figure(figsize=(5,10))

plt.show()


# <div style="border-radius: 10px; border: #00d65c solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>✅ Great :</b> Our target is pretty well balanced. After EDA it seems that this competition will be fairly easy and light weight. There is little missing values, not many features, balanced target distribution and it's not a time series.
# </div>

# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left; margin-top: 10px;">
#     <b>📔 Feature/target distribution :</b> Investing distributions of your numerical and categorical features is one of the most crucial steps in data analysis. This way you understand data better and can discover a lot of red flags hiding in it like:
#     <ul>
#         <li><b>data drift</b> - training and testing data distriubtion are different</li>
#         <li><b>imbalanced target</b> - one class has significantly more samples than the other(s), which can lead to biased models that favor the majority class, only in classificaiton problems</li>
#         <li><b>outliers</b> - data points that differ from the other observations in a dataset</li>
#         <li><b>bimodality</b> - can indicate that your data is actually a mix of two different distributions</li>
#     </ul>
#     If a column has one or more of those issues you can try dropping it and seeing if it improves your leaderboard score.
# </div>

# In[10]:


corr_matrix = train.select_dtypes(include=np.number).corr()
mask = np.triu(corr_matrix)

plt.figure(figsize=(15,12))
sns.heatmap(data=corr_matrix, mask=mask, cmap='Blues', linewidths=1, square=True, linecolor='#fafafa')
plt.title('\nCorrelation matrix\n', fontsize=17)
plt.show()


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Correlation :</b> is basically a mesure of how two or more variables change together either in the same or the opposite direction. For example the more time you spend studying, the higher your grades tend to be - positive correlation (if one increases then other increases). Or the more fast food you consume, the lower your overall health - negative correlation (one increases when other decreases).
#     <br><br>
#     <b>📔 Correlation matrix :</b> dispalys correlation coefficiants between variables. Understanding it can help discover relationships between variables, determine important variables for predictive modeling and uncover underlying patterns in data. Because correlation matrix is symmetrical along its diagonal (<i>A, B</i> are features, <i>corr(A, B)</i> == <i>corr(B, A)</i>) and correlation of a feature with itself is equal to 1 (<i>corr(A, A)</i> == 1) so I plotted it without an upper traingle.
# </div>

# In[11]:


categorical_cols = ['temp_of_extremities', 'peripheral_pulse', 'mucous_membrane','capillary_refill_time','pain','peristalsis','abdominal_distention','nasogastric_tube','nasogastric_reflux','rectal_exam_feces','abdomen','abdomo_appearance','lesion_2','surgery', 'age', 'surgical_lesion', 'lesion_3', 'cp_data']
threshold = .05

print(f'{"Column":<25} | Test result')
print('----------------------------------------')

for column in categorical_cols:
    # Create a contingency table
    contingency_table = pd.crosstab(train[column], train['outcome'])
    
    # Perform the Chi-Square test
    chi2, p, _, _ = stats.chi2_contingency(contingency_table)
    
    print(f'{column:<25} |   ', '\033[32mPassed' if p < threshold else '\033[31mFailed', '\033[0m')


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Chi-square :</b> test of independence of variables is used to assess whether two categorical variables are related to each other. It's often used for feature selection with categorical features in a dataset. We define a threshold against which we will test our feature's p-value and decide whether to drop ones that failed it. In this case setting threshold to 0.05 and dropping <i>lesion_3</i> column that didn't pass the test slightly improves models performance.
# </div>

# # Feature engineering 🛠️
# ***

# ## Drop lesion_3

# In[12]:


train.drop('lesion_3', axis=1, inplace=True)

# train.drop(columns='lesion_3', inplace=True) # Does the same


# # Preprocessing ⚙️
# ***

# ## Splitting dataset

# In[13]:


X_train = train.drop(columns='outcome')
y_train = train[['outcome']]


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Target shape :</b> By using <code>dataframe[[cols...]]</code> notation instead of <code>dataframe[col]</code> <i>y_train</i> will be of shape (n_examples, 1) and not a tuple (n_examples, ). Later in this section we will use <i>OrdinalEncoder</i> which needs its input to be of shape (n, 1).
#     <ul>
#         <li><code>train['outcome']</code> -> <code>pd.Series</code></li>
#         <li><code>train[['outcome']]</code> -> <code>pd.DataFrame</code></li>
#     </ul>
#     Pandas DataFrames are of shape (n, cols) and Series are always (n, ) shape which means they don't have columns..
# </div>

# ## Pipelines

# In[14]:


numerical_pipeline = make_pipeline(
#     SimpleImputer(strategy='mean'), # Tree based models like the LGBM deal with missing values better than SimpleImputer
    QuantileTransformer(output_distribution='normal', random_state=42),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
#     SimpleImputer(strategy='most_frequent'),
    OrdinalEncoder(handle_unknown='use_encoded_value' ,unknown_value=10)
)


# <div style="border-radius: 10px; border: #27374D solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📉 Transformers :</b> 
#     <ul>
#         <li><b>SimpleImputer</b> replaces missing NaN values in every column with either a mean, median or most frequent occurance of that column.</li>
#         <li><b>QuantileTransformer</b> tries to change distribution of each feature to match the normal distribution as closely as possible which is prefered by ML models. Not only that but it will help to lessen the impact of outliers on our model.</li>
#         <li><b>StandardScaler</b> standardizes features by changing mean of each column to 0, standard deviation to 1.
# </li>
#         <li><b>OrdinalEncoder</b> encodes categorical features as integers. Instead of it you can use <code>OneHotEncoder</code> but in this competition <code>OrdnialEncoder</code> gives a beter score.</li>
#     </ul>
# </div>
# 
# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left; margin-top: 10px;">
#     <b>📔 Transformer order :</b> It's very important that <b>StandardScaler is used last</b> otherwise QuantileTransformer will negate the effect of it.
# </div>

# ## Main transformer

# In[15]:


transformer = make_column_transformer(
    (
        numerical_pipeline,
        make_column_selector(dtype_include=np.number) # We want to apply numerical_pipeline only on numerical columns
    ),
    (
        categorical_pipeline,
        make_column_selector(dtype_include=object) # We want to apply categorical_pipeline only on object (string) columns
    ),
    remainder='passthrough', # If any column where missed then don't drop them - we take care of every column so this line is not necessery
    verbose_feature_names_out=False # if False transformer won't add prefixes (name of the transformer that generated specific feature) to column names, column names are shorter that way
)

transformer


# <div style="border-radius: 10px; border: #0ea5e9 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>🖊️ Note :</b> I'm using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html">make_pipeline</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html">make_column_transformer</a> because they are simpler to use than a <a href="https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html">Pipeline</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html">ColumnTransformer</a>. The main difference is that transformer's names are set to their types automatically so you don't have to name each one by hand.
# </div>

# ## Target encoder

# In[16]:


target_encoder = OrdinalEncoder(categories=[['died', 'euthanized', 'lived']])


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 OrdinalEncoder vs LabelEncoder :</b> In this case it's better to use <code>OrdnialEncoder</code> because our target classes have hierarchy. In simple terms, if your categorical data has a specific order (like ‘low’, ‘medium’, ‘high’), use <code>OrdnialEncoder</code>. If it doesn’t (like ‘cat’, ‘dog’, ‘whale’), use <code>LabelEncoder</code>
# </div>

# ## Transforming

# In[17]:


X_train = transformer.fit_transform(X_train)
y_train = target_encoder.fit_transform(y_train).ravel()

# If we saved y_train as pd.Series we would have to use this:
# y_train = target_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# There is even simpler approach without OrdinalEncoder
# y_train = y_train.map({'died':0,'euthanized':1,'lived':2})


# <div style="border-radius: 10px; border: #27374D solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📉 Target's shape :</b> <code>OrdinalEncoder</code>'s inputs must be of shape (n, 1) and if <b>y_train</b> would be a pandas <i>Series</i> with shape (n, ) we'd need to convert it to a numpy array by getting <code>.values</code> and then reshaping it to a desired shape. At the end I'm using <code>ravel()</code> which flattens our (n, 1) numpy array to (n, ) because <code>LGBMClassifier</code> model needs training data to be in a tuple shape.
# </div>

# ## Results

# In[18]:


X_train = pd.DataFrame(data=X_train, columns=transformer.get_feature_names_out(), index=train.index)
X_train.head()


# In[19]:


X_train.describe()


# <div style="border-radius: 10px; border: #00d65c solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>✅ Great :</b> Now columns have zero mean, std equal to one and distribution closer to normal so the model will learn faster and more efficient.
# </div>

# # Modeling 🪄
# ***

# ## Splitting training data

# In[20]:


X_train_optuna, X_val_optuna, y_train_optuna, y_val_optuna = train_test_split(X_train, y_train, train_size=0.9)


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Splitting proportions :</b> Most popular ratio I use in playground is propably 80/20 and 90/10 where 80%/90% goes for training and rest is used in validation/cross-validation. But it's by no means set in stone and will differ a lot when for example you have a very large dataset with over 100k examples. Then you can split the data in 99/1 proportions beacuse then you will still have over a thousand examples to validate your model on which is enough. One other exception is dealing with time series data if you have sales records from a few years and your model will be used for predicting whole next year sales for each month. Then it would be reasonable to assign 1 year of data for validation beacuse you want to validate your model on data similar to the one that it will be used in real world. So remember to always pay attention how to split your dataset.
# </div>

# ## Hyperparameter optimization

# In[21]:


def objective(trial):
#     model = LGBMClassifier(
#         n_estimators = trial.suggest_int('n_estimators', 32, 1024),
#         learning_rate = trial.suggest_float('learning_rate', 0.001, 0.5),
#         max_depth = trial.suggest_int('max_depth', 1, 10),
#         num_leaves = trial.suggest_int('num_leaves', 2, 1024),
#         reg_lambda  = trial.suggest_float('reg_lambda', 0.001, 10),
#         reg_alpha = trial.suggest_float('reg_alpha', 0, 10),
#         subsample = trial.suggest_float('subsample', 0.001, 1),
#         colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1),
#         min_child_samples = trial.suggest_int('min_child_samples', 2, 1024),
#         min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
#         objective = trial.suggest_categorical('objective', ['multiclass']),
#         metric = trial.suggest_categorical('metric', ['multi_logloss']),
#         boosting_type = trial.suggest_categorical('boosting_type', ['gbdt']),
#     )
    
#     model = CatBoostClassifier(
#         iterations = trial.suggest_int('iterations', 32, 1024),
#         learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3),
#         depth = trial.suggest_int('depth', 1, 10),
#         l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0.01, 10),
#         grow_policy = trial.suggest_categorical('grow_policy', ['Depthwise']),
#         bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian']),
#         od_type = trial.suggest_categorical('od_type', ['Iter']),
#         eval_metric = trial.suggest_categorical('eval_metric', ['TotalF1']),
#         loss_function = trial.suggest_categorical('loss_function', ['MultiClass']),
#         random_state = trial.suggest_categorical('random_state', [42]),
#         verbose = trial.suggest_categorical('verbose', [0])
#     )

    model = XGBClassifier(
        eta = trial.suggest_float('eta', 0.001, 0.3),
        n_estimators = trial.suggest_int('n_estimators', 32, 1024),
        max_depth = trial.suggest_int('max_depth', 1, 10),
        reg_lambda = trial.suggest_float('reg_lambda', 0.01, 10),
        subsample = trial.suggest_float('subsample', 0.01, 1),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.01, 1),
        objective = trial.suggest_categorical('objective', ['multi:softmax'])
    )
    
    model.fit(
        X_train_optuna, y_train_optuna,
        eval_set=[(X_train_optuna, y_train_optuna), (X_val_optuna, y_val_optuna)],
        verbose=False
    )
    
    return f1_score(y_val_optuna, model.predict(X_val_optuna), average='micro') # micro F1 is used in this competitons for evaluation so we will use it for hyperparameter optimization

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# best_hyperparams = study.best_params


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Hyperparameter ranges :</b> Machine learning models oftentimes prefer powers of 2 so it's a good practice to set hyperparameter ranges to them but it's by no means necessary and most of the times will not have any impact on models performance.
# </div>

# In[22]:


best_hyperparams_xgb = {'eta': 0.2734096744203229, 'n_estimators': 251, 'max_depth': 1, 'reg_lambda': 1.3536521735953297, 'subsample': 0.9372043032806799, 'min_child_weight': 5, 'colsample_bytree': 0.32973413695986586, 'objective': 'multi:softmax'}
best_hyperparams_lgbm = {'n_estimators': 146, 'learning_rate': 0.09732455260435911, 'max_depth': 8, 'num_leaves': 973, 'reg_lambda': 5.558974411222393, 'reg_alpha': 5.94913795893992, 'subsample': 0.057493821911338956, 'colsample_bytree': 0.7716515051686431, 'min_child_samples': 46, 'min_child_weight': 7, 'objective': 'multiclass', 'metric': 'multi_logloss', 'boosting_type': 'gbdt'}
best_hyperparams_cb = {'iterations': 210, 'learning_rate': 0.21569043805753133, 'depth': 3, 'l2_leaf_reg': 6.171143053175511, 'grow_policy': 'Depthwise', 'bootstrap_type': 'Bayesian', 'od_type': 'Iter', 'eval_metric': 'TotalF1', 'loss_function': 'MultiClass', 'random_state': 42, 'verbose': 0}


# ## Models

# In[23]:


models = [
    XGBClassifier(**best_hyperparams_xgb),
    LGBMClassifier(**best_hyperparams_lgbm),
    CatBoostClassifier(**best_hyperparams_cb)
]


# ## Model evaluation

# In[24]:


for model in models:
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_micro').sum() / 5

    print(f'{model.__class__.__name__} micro F1 cross-validation score: {cv_score:.3f}')


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Cross-validation :</b> is a method giving more accurate performance evaluation of a model compared to normal validation. It involves dividing the original data into several subsets - 5 in this case. Then model is trained on 4 of those and is scored on the remaining one. It's repeated 5 times, each time last fold is different and model is trained on the rest. I haven't used cross-validation in optuna because this method is computationaly expensive. Basically number next to <code>cv</code> attribute is an amount of times our model will be trained. So if optuna runs for 50 iterations it would train our model 250 times witch cross-validation instead of 50.
# </div>

# ## Model training

# In[25]:


for model in models:
    model.fit(X_train, y_train)


# In[26]:


for model in models:
    train_score = f1_score(y_train, model.predict(X_train), average='micro')

    print(f'{model.__class__.__name__} micro F1 training score: {train_score:.3f}')


# <div style="border-radius: 10px; border: #ffac00 solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b> ⚠️ Overfitting:</b> F1 score ranges from [0, 1] so 0.06 to 0.15 difference in cross-validation score to training score is a sign of overfitting to the training set also known as a high variance problem. It means that our model predicts data that it has seen very well but has a trouble generalizing to unseen data. To fix that we will ensamble our models into one.
# </div>

# ## Feature importance

# In[27]:


feature_importance = pd.DataFrame(data = {'feature': train.columns[:-1], 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=True)

feature_importance.plot(kind='barh', x='feature', y='importance', legend=False, color=sns.color_palette('plasma', n_colors=len(feature_importance)), figsize=(16, 12))

plt.xlabel('Importance')
plt.ylabel('Feature name')
plt.title('\nFeature Importance\n', fontsize=15)
plt.show()


# <div style="border-radius: 10px; border: #7c3aed solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📔 Feature importance :</b> assigns a score to input features based on how useful they are at predicting a target variable. It provides an understanding of which features have the most impact on a model’s predictions. This can be particularly useful in feature selection, improving model interpretability, and understanding the data.
# </div>

# # Submission 🏆
# ***
# Now we can apply every transformation on a test set step by step like we did on the training set. In case of this competition there isn't much of it.

# In[28]:


test.drop('lesion_3', axis=1, inplace=True)

X_test = pd.DataFrame(data=transformer.transform(test), columns=transformer.get_feature_names_out(), index=test.index)


# In[29]:


predictions = []
for model in models:
    predictions.append(model.predict(X_test).ravel()) # CatBoostClassifier's predictions are of shape (n ,1) and not (n, ) like other models so we have to use .ravel()

# Mode doesn't work on python lists only on ndarray (numpy arrays)
predictions = np.array(predictions)

# Take the most frequent prediction out of 3 models
final_predictions, _ = stats.mode(predictions, axis=0)

final_predictions.shape


# In[30]:


submission = pd.read_csv('/kaggle/input/playground-series-s3e22/sample_submission.csv', index_col='id')

submission['outcome'] = target_encoder.inverse_transform(final_predictions.reshape(-1, 1)).ravel()

submission.to_csv('/kaggle/working/submission.csv')


# <div style="border-radius: 10px; border: #27374D solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     <b>📉 Submission :</b> Our model predicts classes as integers from 0 to 2 but our submission requires names of classes so we use <code>inverse_transform()</code> to change number to class names. Remember that <code>transform()</code> maps class names (strings) to integers so the <code>inverse_transform()</code> does the opposite.
# </div>

# # Thank you ✨
# 
# I hope you enjoyed this notebook and learned something new. 😊 If you did, please consider upvoting it and leaving a comment. I would love to hear your feedback and suggestions. 💬
# 
# Also, feel free to fork this notebook and experiment with different models, features, and techniques.
# 
# Thank you for reading and happy kaggling! 🚀
