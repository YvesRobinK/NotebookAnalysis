#!/usr/bin/env python
# coding: utf-8

# # 🤝 Intro 

# Rocket League is fun.

# ![Rocket League](https://media2.giphy.com/media/xT0xepBLaRaduNgkne/giphy.gif?cid=ecf05e47c5ru2uhkoa15ekp0l8fm1lz43bwtyehz0tywlwum&rid=giphy.gif&ct=g)

# Let's have some fun predicting the probability of a team scoring, within 10 seconds.

# # 🚚 Import

# ## Packages

# In[1]:


import numpy as np  # linear algebra
import pandas as pd  # data manipulation
import os  # file navigation
import gc  # garbage collection

# visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly import subplots

from sklearn.model_selection import cross_validate  # k-fold Cross Validation
from sklearn.preprocessing import LabelEncoder  # output binary encoding

from xgboost import XGBClassifier  # Gradient Boosted Tree (XGBoost)

from tensorflow.config import list_physical_devices  # check if GPU is available


# ## Config

# In[2]:


# training and cross validation
GPU = list_physical_devices('GPU') != []
N_ESTIMATORS = 2000
MAX_DEPTH = 8
LEARNING_RATE = 0.01
FOLDS = 5

# data loading
DEBUG = False
SAMPLE = 0.2
SEED = 42


# ## Dataset

# Because of memory limitations, we can't use all of the data from the 10 train .csv files.  
# Instead, we'll get a random sample of each file (for now I'm experimenting with 20-33% sample size), and combine these samples into a unique training dataset.

# In[3]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


get_ipython().run_cell_magic('time', '', "col_dtypes = {\n    'game_num': 'int8', 'event_id': 'int8', 'event_time': 'float16',\n    'ball_pos_x': 'float16', 'ball_pos_y': 'float16', 'ball_pos_z': 'float16',\n    'ball_vel_x': 'float16', 'ball_vel_y': 'float16', 'ball_vel_z': 'float16',\n    'p0_pos_x': 'float16', 'p0_pos_y': 'float16', 'p0_pos_z': 'float16',\n    'p0_vel_x': 'float16', 'p0_vel_y': 'float16', 'p0_vel_z': 'float16',\n    'p0_boost': 'float16', 'p1_pos_x': 'float16', 'p1_pos_y': 'float16',\n    'p1_pos_z': 'float16', 'p1_vel_x': 'float16', 'p1_vel_y': 'float16',\n    'p1_vel_z': 'float16', 'p1_boost': 'float16', 'p2_pos_x': 'float16',\n    'p2_pos_y': 'float16', 'p2_pos_z': 'float16', 'p2_vel_x': 'float16',\n    'p2_vel_y': 'float16', 'p2_vel_z': 'float16', 'p2_boost': 'float16',\n    'p3_pos_x': 'float16', 'p3_pos_y': 'float16', 'p3_pos_z': 'float16',\n    'p3_vel_x': 'float16', 'p3_vel_y': 'float16', 'p3_vel_z': 'float16',\n    'p3_boost': 'float16', 'p4_pos_x': 'float16', 'p4_pos_y': 'float16',\n    'p4_pos_z': 'float16', 'p4_vel_x': 'float16', 'p4_vel_y': 'float16',\n    'p4_vel_z': 'float16', 'p4_boost': 'float16', 'p5_pos_x': 'float16',\n    'p5_pos_y': 'float16', 'p5_pos_z': 'float16', 'p5_vel_x': 'float16',\n    'p5_vel_y': 'float16', 'p5_vel_z': 'float16', 'p5_boost': 'float16',\n    'boost0_timer': 'float16', 'boost1_timer': 'float16', 'boost2_timer': 'float16',\n    'boost3_timer': 'float16', 'boost4_timer': 'float16', 'boost5_timer': 'float16',\n    'player_scoring_next': 'O', 'team_scoring_next': 'O', 'team_A_scoring_within_10sec': 'O',\n    'team_B_scoring_within_10sec': 'O'\n}\ncols = list(col_dtypes.keys())\n\npath_to_data = '../input/tabular-playground-series-oct-2022'\ndf = pd.DataFrame({}, columns=cols)\nfor i in range(10):\n    df_tmp = pd.read_csv(f'{path_to_data}/train_{i}.csv', dtype=col_dtypes)\n    if SAMPLE < 1:\n        df_tmp = df_tmp.sample(frac=SAMPLE, random_state=SEED)\n        \n    df = pd.concat([df, df_tmp])\n    del df_tmp\n    gc.collect()\n    if DEBUG:\n        break\n")


# In[5]:


df


# # 👀 Quick EDA

# In[6]:


input_cols = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z', 
    'p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x', 'p0_vel_y', 'p0_vel_z', 
    'p1_pos_x', 'p1_pos_y', 'p1_pos_z', 'p1_vel_x', 'p1_vel_y', 'p1_vel_z',
    'p2_pos_x', 'p2_pos_y', 'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z',
    'p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x', 'p3_vel_y', 'p3_vel_z',
    'p4_pos_x', 'p4_pos_y', 'p4_pos_z', 'p4_vel_x', 'p4_vel_y', 'p4_vel_z',
    'p5_pos_x', 'p5_pos_y', 'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z',
    'p0_boost', 'p1_boost',  'p2_boost', 'p3_boost', 'p4_boost', 'p5_boost',
    'boost0_timer', 'boost1_timer', 'boost2_timer', 'boost3_timer', 'boost4_timer', 'boost5_timer'
]


# In[7]:


output_cols = ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']


# ## Input Variables

# In[8]:


def int_to_grid_coord(k, n):
    return (k // n) + 1, (k % n) + 1


# In[9]:


def plot_distributions(df, row_count, col_count, title, height):
    features = df.columns
    fig = subplots.make_subplots(
        rows=row_count, cols=col_count,
        subplot_titles=features
    )

    for k, col in enumerate(features):
        i, j = int_to_grid_coord(k, col_count)

        fig.add_trace(
            go.Histogram(
                x=df[col].astype('float32'),
                name=col
            ),
            row=i, col=j
        )

    fig.update_layout(
        title=title,
        height=row_count * height,
        showlegend=False
    )

    return fig


# In[10]:


plot_distributions(df[input_cols].sample(frac=0.0005), 9, 6, "Input Variables Distributions", 300)


# ## Output Variables

# In[11]:


plot_distributions(df[output_cols].sample(frac=0.0005), 1, 2, "Output Variables Distributions", height=600)


# # ⚙️ Feature Engineering

# Shoutout [this post by samuelcortinhas](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/discussion/356852) for the idea.

# Let's derive 2 new features:
# * For each player (and the ball) let's get their velocity's magnitude.
# * For each player, let's get their distance from the ball.

# ## Euclidian Norm

# For these 2 new features, we'll need to get the 3D euclidian norm of a vector:
# $$ \| \overrightarrow{v} \| = \sqrt{x^2 + y^2 + z^2} $$
# We'll use numpy's linalg.norm() method for that.

# In[12]:


def euclidian_norm(x):
    return np.linalg.norm(x, axis=1)


# In[13]:


# let's group the x, y and z variables by player and ball categories
# this simplifies the code for the euclidian norm calculation
vel_groups = {
    f"{el}_vel": [f'{el}_vel_x', f'{el}_vel_y', f'{el}_vel_z']
    for el in ['ball'] + [f'p{i}' for i in range(6)]
}
pos_groups = {
    f"{el}_pos": [f'{el}_pos_x', f'{el}_pos_y', f'{el}_pos_z']
    for el in ['ball'] + [f'p{i}' for i in range(6)]
}
pos_groups


# In[14]:


# velocity magnitude
for col, vec in vel_groups.items():
    df[col] = euclidian_norm(df[vec])


# In[15]:


# distance from ball
for col, vec in pos_groups.items():
    df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)


# # 🧹 Cleaning

# We drop the columns below because they should not influence the results.  
# game_num, event_id and event_time are irrelevant.  
# player_scoring_next and team_scoring next are a form of data leakage, as in they're synonymous with the output variable.  
# ball_pos_ball_dist is always 0, the distance between the ball and itself.

# ## Dropping columns

# In[16]:


cols_to_drop = [
    'game_num', 'event_id', 'event_time', 'player_scoring_next', 'team_scoring_next', 'ball_pos_ball_dist'
]


# In[17]:


df = df.drop(columns=cols_to_drop)


# In[18]:


df


# ## Dropping rows containing NaN

# In[19]:


has_na = {}
for col in df.columns:
    has_na[col] = df[col].isnull().values.any()

print("Columns that contain null values:")
for col in has_na:
    if has_na[col]:
        print(col)


# In[20]:


null_p0_pos_x_count = df['p0_pos_x'].isna().sum()
null_p0_pos_x_perc = null_p0_pos_x_count / df.shape[0]
print(f"Missing {null_p0_pos_x_count} values ({null_p0_pos_x_perc:.2%})")


# To keep things simple, let's just drop all null values.

# In[21]:


df = df.dropna(axis=0)


# In[22]:


has_na = {}
for col in df.columns:
    has_na[col] = df[col].isnull().values.any()

print("Columns that contain null values:")
for col in has_na:
    if has_na[col]:
        print(col)


# In[23]:


df


# # 🚀 Model Training

# We'll be predicting the probability of team A scoring and team B scoring with 2 separate models.

# ## Model A

# In[24]:


# used to encode the binary classes
le_a = LabelEncoder()


# In[25]:


model_a = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    objective='binary:logistic',
    tree_method='gpu_hist' if GPU else 'hist'
)


# In[26]:


cv_a = cross_validate(
    model_a, 
    X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
    y=le_a.fit_transform(df['team_A_scoring_within_10sec'].values),
    scoring="neg_log_loss",
    cv=FOLDS,
    verbose=2,
    return_estimator=True
)


# ## Model B

# In[27]:


# used to encode the binary classes
le_b = LabelEncoder()


# In[28]:


model_b = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    objective='binary:logistic',
    tree_method='gpu_hist' if GPU else 'hist'
)


# In[29]:


cv_b = cross_validate(
    model_b, 
    X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
    y=le_b.fit_transform(df['team_B_scoring_within_10sec'].values),
    scoring="neg_log_loss",
    cv=FOLDS,
    verbose=2,
    return_estimator=True
)


# # ✔️ Model Evaluation

# ## Cross Validation Test Score

# Let's visualize the log loss score on each of our folds, for both our models

# In[30]:


df_cv_a = pd.DataFrame(
    {
        "model": "Model A",
        "fold": list(range(FOLDS)),
        "test_log_loss": - cv_a["test_score"]
    }
)
df_cv_b = pd.DataFrame(
    {
        "model": "Model B",
        "fold": list(range(FOLDS)),
        "test_log_loss": - cv_b["test_score"]
    }
)
df_cv = pd.concat([df_cv_a, df_cv_b])

del df_cv_a
del df_cv_b
gc.collect()


# In[31]:


px.bar(
    df_cv, x='fold', y='test_log_loss', color='model', 
    barmode='group', title='Cross Validation Log Loss'
)


# # 💌 Submission

# In[32]:


df_test = pd.read_csv('/kaggle/input/tabular-playground-series-oct-2022/test.csv')
df_test


# In[33]:


def preprocess(df):
    # velocity magnitude
    for col, vec in vel_groups.items():
        df[col] = euclidian_norm(df[vec])
    
    # ball distance
    for col, vec in pos_groups.items():
        df[col + "_ball_dist"] = euclidian_norm(df[vec].values - df[pos_groups["ball_pos"]].values)
    
    df = df.drop(columns=['ball_pos_ball_dist'])
    
    return df


# In[34]:


df_test = preprocess(df_test)


# In[35]:


get_ipython().run_cell_magic('time', '', "# take the mean of the predictions made by the k models gotten out of the k-fold cross validation\npred_a = np.zeros(df_test.shape[0])\nfor estimator in cv_a['estimator']:\n    pred_a += estimator.predict_proba(df_test.drop(columns=['id']).values)[:, 1]\n\npred_a /= FOLDS\n")


# In[36]:


get_ipython().run_cell_magic('time', '', "# take the mean of the predictions made by the k models gotten out of the k-fold cross validation\npred_b = np.zeros(df_test.shape[0])\nfor estimator in cv_b['estimator']:\n    pred_b += estimator.predict_proba(df_test.drop(columns=['id']).values)[:, 1]\n\npred_b /= FOLDS\n")


# In[37]:


df_submission = pd.DataFrame(
    {
        "id": df_test['id'],
        "team_A_scoring_within_10sec": pred_a,
        "team_B_scoring_within_10sec": pred_b
    }
)


# In[38]:


df_submission


# In[39]:


df_submission.to_csv('submission.csv', index=False)


# In[ ]:




