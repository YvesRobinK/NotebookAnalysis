#!/usr/bin/env python
# coding: utf-8

# # ❗️❗️❗️DISCLAIMER: This notebook is a copy of [great notebook from @paddykb](https://www.kaggle.com/code/paddykb/tps-2022-10-fastai) with multistart NN training and test dataset augmentation (TTA). If you like this kernel, you MUST upvote @paddykb's kernel beforehand and don't forget to upvote this one too ❗️❗️❗️

# In[1]:


import random
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from fastai.tabular.all import *
import fastai.losses as loss


# ## Introduction
# 
# In this notebook I'm building a NN using fastai. The "value add" is the augmentation of the training batches with random permutations from the 144 possible:
# * Six ways to order team A players.
# * Six ways to order team B players
# * flip teams A and B
# * horizontal reflection
# 
# I've made no attempt to tune the model or try different architectures. If you do, please make your notebook public so I can learn from your efforts.
# 
# On the shoulders of giants:
# * @pietromaldini1 [data augmentation discussion](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/discussion/357577)
# * @hsuyab [Fast loading & High Compression with Feather](https://www.kaggle.com/code/hsuyab/fast-loading-high-compression-with-feather)
# * @slawekbiel [preloading data into the GPU](https://www.kaggle.com/code/slawekbiel/fast-fastai-training)
# * @spyrow [Mirroring the board](https://www.kaggle.com/code/spyrow/playground-oct-2022-lgbmclassifier?scriptVersionId=107206095)
# 
# **todo**: *Since we have event order in the train set, it would be interesting to model successive positions. I.e. use the current positions as input and the next position of players and ball as the output. (strip off the last layer and use it as an embedding). Take a closer look at borrowing this from @ryancaldwell [Predict next frame](https://www.kaggle.com/code/ryancaldwell/cnn-predict-next-frame)*
# 
# 
# What's new in v3?
# * shuffle the training data between epochs
# * feature engineering ideas @samuelcortinhas, @pietromaldini1 and others. (euclidean distance to the goal, distance of players from the ball, and speeds)
# * still looking at @ryancaldwell [Predict next frame](https://www.kaggle.com/code/ryancaldwell/cnn-predict-next-frame)*
# * Increased the number of training epochs.

# In[2]:


features = [
    'ball_pos_x', 'ball_pos_y','ball_pos_z', 'ball_vel_x', 'ball_vel_y', 'ball_vel_z', 
    'p0_pos_x', 'p0_pos_y', 'p0_pos_z', 'p0_vel_x', 'p0_vel_y', 'p0_vel_z', 'p0_boost', 'p0_na',
    'p1_pos_x', 'p1_pos_y', 'p1_pos_z', 'p1_vel_x', 'p1_vel_y', 'p1_vel_z', 'p1_boost', 'p1_na',
    'p2_pos_x', 'p2_pos_y', 'p2_pos_z', 'p2_vel_x', 'p2_vel_y', 'p2_vel_z', 'p2_boost', 'p2_na',
    'p3_pos_x', 'p3_pos_y', 'p3_pos_z', 'p3_vel_x', 'p3_vel_y', 'p3_vel_z', 'p3_boost', 'p3_na',
    'p4_pos_x', 'p4_pos_y', 'p4_pos_z', 'p4_vel_x', 'p4_vel_y', 'p4_vel_z', 'p4_boost', 'p4_na',
    'p5_pos_x', 'p5_pos_y', 'p5_pos_z', 'p5_vel_x', 'p5_vel_y', 'p5_vel_z', 'p5_boost', 'p5_na',
    'boost0_timer', 'boost1_timer', 
    'boost2_timer', 'boost3_timer',
    'boost4_timer', 'boost5_timer']

features_x_pos = [pos for pos, feature in enumerate(features) if feature.endswith('_x')]
features_y_pos = [pos for pos, feature in enumerate(features) if feature.endswith('_y')]

targets = [
    'team_A_scoring_within_10sec',
    'team_B_scoring_within_10sec']


# ## Load Data

# In[3]:


get_ipython().run_cell_magic('time', '', "DEBUG = False\ninput_path = Path('../input/fast-loading-high-compression-with-feather/feather_data')\n\ndef fe(x):\n    # indicators for respawns...\n    x['p0_na'] = x['p0_pos_x'].isna().astype('int8')\n    x['p1_na'] = x['p1_pos_x'].isna().astype('int8')\n    x['p2_na'] = x['p2_pos_x'].isna().astype('int8')\n    x['p3_na'] = x['p3_pos_x'].isna().astype('int8')\n    x['p4_na'] = x['p4_pos_x'].isna().astype('int8')\n    x['p5_na'] = x['p5_pos_x'].isna().astype('int8')\n    for feature in features:\n        if feature.endswith('_na'):\n            continue\n        # this is just scaling the features to something reasonable\n        # it might make sense to apply a transformation to the z-dimension.\n        if feature.endswith('_x'):\n            x[feature] = (x[feature] / 82).fillna(0).astype('float16')\n        if feature.endswith('_y'):\n            x[feature] = (x[feature] / 120).fillna(0).astype('float16')\n        if feature.endswith('_z'):\n            x[feature] = (x[feature] / 40).fillna(0).astype('float16')\n        if feature.endswith('_boost'):\n            x[feature] = (x[feature] / 100).fillna(0).astype('float16')\n        if feature.endswith('_timer'):\n            x[feature] = (-x[feature] / 100).astype('float16')\n    return x\n\ndef read_train():\n    dfs = []\n    for i in range(10):\n        dfs.append(fe(pd.read_feather(input_path / f'train_{i}_compressed.ftr')))\n    result = pd.concat(dfs)\n    if DEBUG:\n        result = result.sample(frac=0.05)\n    return result\n\ndef read_test():\n    return fe(pd.read_feather(input_path / 'test_compressed.ftr'))\n\ndf_train = read_train()\ngc.collect()\n\nprint(f'Train Rows = {len(df_train):,}  ' \n      f'Memory Usage = {df_train.memory_usage(deep=True).sum() / (1024 * 1024):4.1f} Mb'\n     '\\n')\n")


# ## Move Training Data to the GPU

# In[4]:


# split train & validation

game_nums = df_train['game_num'].unique()
train_game_nums = random.sample(list(game_nums), int(len(game_nums) * 0.80))

train_feature_tensor = torch.tensor(
    df_train.query("game_num in @train_game_nums")[features].to_numpy())
train_target_tensor  = torch.tensor(
    df_train.query("game_num in @train_game_nums")[targets].to_numpy())
valid_feature_tensor = torch.tensor(
    df_train.query("game_num not in @train_game_nums")[features].to_numpy())
valid_target_tensor  = torch.tensor(
    df_train.query("game_num not in @train_game_nums")[targets].to_numpy())

gc.collect()

if torch.cuda.is_available():
    train_feature_tensor = train_feature_tensor.cuda()
    train_target_tensor  = train_target_tensor.cuda()
    valid_feature_tensor = valid_feature_tensor.cuda()
    valid_target_tensor  = valid_target_tensor.cuda()

del df_train
gc.collect()


# In[5]:


# some light feature engineering on each batch

def fe_goal_distance(ball):
    # might need to rescale the vectors...
    dist_a = ((ball[:,0:1] - 0) ** 2 + (ball[:,1:2] - 1) ** 2 + (ball[:,2:3] - 0) ** 2) ** 0.5 / 2
    dist_b = ((ball[:,0:1] - 0) ** 2 + (ball[:,1:2] + 1) ** 2 + (ball[:,2:3] - 0) ** 2) ** 0.5 / 2 
    return dist_a, dist_b
    
def fe_dist_ball_player(ball, player):
    # might need to rescale the vectors...
    dist = ((
        (ball[:,0:1] - player[:,0:1]) ** 2 + 
        (ball[:,1:2] - player[:,1:2]) ** 2 + 
        (ball[:,2:3] - player[:,2:3]) ** 2) ** 0.5) / 12
    return dist

def fe_speed_of_thing(thing):
    # might need to rescale the vectors...
    return (thing[:, 3:4] ** 2 + thing[:, 4:5] ** 2 + thing[:, 5:6] ** 2) ** 0.5

def augment_fe(empty, X, Y):
    ball = X[:, :6]
    p0 = X[:,  6:14]
    p1 = X[:, 14:22]
    p2 = X[:, 22:30]
    p3 = X[:, 30:38]
    p4 = X[:, 38:46]
    p5 = X[:, 46:54]
    boosts = X[:, 54:]
    
    ## distance to goal
    goal_a, goal_b = fe_goal_distance(ball)
    
    ## distance to ball
    p0d = fe_dist_ball_player(ball, p0)
    p1d = fe_dist_ball_player(ball, p1)
    p2d = fe_dist_ball_player(ball, p2)
    p3d = fe_dist_ball_player(ball, p3)
    p4d = fe_dist_ball_player(ball, p4)
    p5d = fe_dist_ball_player(ball, p5)
    
    ## speeds
    ball_s = fe_speed_of_thing(ball)
    p0s = fe_speed_of_thing(p0)
    p1s = fe_speed_of_thing(p1)
    p2s = fe_speed_of_thing(p2)
    p3s = fe_speed_of_thing(p3)
    p4s = fe_speed_of_thing(p4)
    p5s = fe_speed_of_thing(p5)
    
    new_X = torch.cat([
        ball, p0, p1, p2, p3, p4, p5, boosts,
        # 15 new features:
        goal_a, goal_b,
        ball_s,
        p0d, p1d, p2d, p3d, p4d, p5d,
        p0s, p1s, p2s, p3s, p4s, p5s
    ], dim=1)
    
    return empty, new_X, Y


# In[6]:


def augment_mirror(empty, X, Y):
    # mirror the match
    # interchange player 1 and 2
    positions = X[:,:54]
    positions[:, features_x_pos] = -positions[:, features_x_pos]
    positions[:, features_y_pos] = -positions[:, features_y_pos]
    
    ball = positions[:, :6]
    p0 = positions[:,  6:14]
    p1 = positions[:, 14:22]
    p2 = positions[:, 22:30]
    p3 = positions[:, 30:38]
    p4 = positions[:, 38:46]
    p5 = positions[:, 46:54]
    
    players = torch.cat([p3, p4, p5, p0, p1, p2], dim=1)
    # mirror
    boosts = X[:, [59, 58, 57, 56, 55, 54]]
    
    flip_X = torch.cat([ball, players, boosts], dim=1)
    flip_Y = Y[:, :, [1,0]]
    
    return empty, flip_X, flip_Y

def augment_flip_x(empty, X, Y):
    # mirror the match in the Y-axis
    positions = X[:,:54]
    positions[:, features_x_pos] = -positions[:, features_x_pos]
    boosts = X[:, [55, 54, 57, 56, 59, 58]]
    
    flip_X = torch.cat([positions, boosts], dim=1)
    
    return empty, flip_X, Y

def augment_shuffle(empty, X, Y):
    # randomly order players (within teams)
    ball = X[:, :6]
    p0 = X[:,  6:14]
    p1 = X[:, 14:22]
    p2 = X[:, 22:30]
    p3 = X[:, 30:38]
    p4 = X[:, 38:46]
    p5 = X[:, 46:54]
    boosts = X[:, 54:]
    
    # shuffle player positions
    pA = torch.cat(random.sample([p0, p1, p2], 3), dim=1)
    pB = torch.cat(random.sample([p3, p4, p5], 3), dim=1)
    
    # shuffled feats
    shuffled_X = torch.cat([ball, pA, pB, boosts], dim=1)
    
    return empty, shuffled_X, Y

# This bespoke dataset classes are based on the ideas of @slawekbiel 
class BespokeDataset:
    def __init__(self, feature_tensor, targets, augment=False, 
                 augment_coef_mirror = 0.5, augment_coef_flipx = 0.5):
        store_attr()
        self.n_inp = 2
        self.augment_coef_mirror = augment_coef_mirror
        self.augment_coef_flipx = augment_coef_flipx
        
    def __getitem__(self, idx):
        # convert float16 -> float32 during the minibatch
        # and apply any augmentation
        batch = torch.empty(0), self.feature_tensor[idx].float(), self.targets[idx, None]
        if self.augment:
            # shuffle player positions.
            batch = augment_shuffle(*batch)
            if random.random() > self.augment_coef_mirror:
                batch = augment_mirror(*batch)
            if random.random() > self.augment_coef_flipx:
                batch = augment_flip_x(*batch)
        # Apply feature engineering here 
        # (as we'll only use a little memory)
        batch = augment_fe(*batch)
        return batch
    
    def __len__(self):
        return len(self.feature_tensor)
    
class BespokeDL(DataLoader):
    def __iter__(self):
        if self.shuffle:
            self.__idxs = torch.tensor(np.random.permutation(range(0,self.n)))
        else:
            self.__idxs = torch.tensor(range(0,self.n))
        for batch_start in range(0, self.n, self.bs):
            if batch_start + self.bs > self.n and self.drop_last:
                return 
            indices = self.__idxs[batch_start:batch_start+self.bs]
            yield self.dataset[indices]


# ## Fit multiple times using all training data

# In[7]:


# refit on full data - ignore the validation... it is meaningless
ds_train = BespokeDataset(
    torch.cat([train_feature_tensor, valid_feature_tensor]), 
    torch.cat([train_target_tensor, valid_target_tensor]), 
    augment=True, augment_coef_mirror = 0.5, augment_coef_flipx = 0.5)
ds_val   = BespokeDataset(valid_feature_tensor, valid_target_tensor, augment=True)
dls = DataLoaders.from_dsets(ds_train, ds_val, bs=4096, dl_type=BespokeDL, num_workers=0, shuffle=True)

LEARNERS = []

for i in range(5):
    model = TabularModel(
        emb_szs={}, n_cont=len(features) + 15, 
        ps=0.3, out_sz=len(targets), act_cls=Mish(inplace=True),
        layers=[2048, 2048, 1024, 1024, 64], y_range=(0,1))
    if torch.cuda.is_available():
        model = model.cuda()

    learn = Learner(dls, model, loss_func=loss.BCELossFlat())

    # It is still not overfitting...
    fit = learn.fit(5 if DEBUG else 40, 1e-3)
    
    LEARNERS.append(learn)


# ## Prepare Submission

# In[8]:


df_test = read_test()
gc.collect()

PREDS = []

for it, learn in enumerate(LEARNERS):
    print('START LEARNER {}'.format(it))
    # Usual predict
    ds_test = BespokeDataset(torch.tensor(df_test[features].to_numpy()), torch.zeros(len(df_test), 2))
    test_dl = learn.dls.test_dl(ds_test)
    test_dl.shuffle = False  
    preds, _ = learn.get_preds(dl=test_dl)
    PREDS.append(preds.numpy() * 10) # weight usual prediction
    
    # TTA - we DON'T use mirror augmentation because it changes the targets order (augment_coef_mirror = 1.0)
    ds_test_1 = BespokeDataset(torch.tensor(df_test[features].to_numpy()), torch.zeros(len(df_test), 2), 
                             augment = True, augment_coef_mirror = 1.0, augment_coef_flipx = 0.5)
    test_dl_1 = learn.dls.test_dl(ds_test_1)
    test_dl_1.shuffle = False  
    for i in range(5):
        preds_1, _ = learn.get_preds(dl=test_dl_1)
        PREDS.append(preds_1.numpy())
        
    # TTA2 - we use ONLY mirror augmentation because it changes the targets order (augment_coef_mirror = -1.0)
    ds_test_2 = BespokeDataset(torch.tensor(df_test[features].to_numpy()), torch.zeros(len(df_test), 2), 
                             augment = True, augment_coef_mirror = -1.0, augment_coef_flipx = 0.5)
    test_dl_2 = learn.dls.test_dl(ds_test_2)
    test_dl_2.shuffle = False  
    for i in range(5):
        preds_2, _ = learn.get_preds(dl=test_dl_2)
        PREDS.append(preds_2.numpy()[:, ::-1]) # revert the order back


# In[9]:


submission = pd.read_csv('../input/tabular-playground-series-oct-2022/sample_submission.csv')
submission.iloc[:, 1:] = np.array(PREDS).sum(axis = 0) / (20 * len(LEARNERS))
submission.to_csv('model_fastai_multistart_tta.csv', index=False)
submission.head()


# # ❗️❗️❗️DISCLAIMER: This notebook is a copy of [great notebook from @paddykb](https://www.kaggle.com/code/paddykb/tps-2022-10-fastai) with multistart NN training and test dataset augmentation (TTA). If you like this kernel, you MUST upvote @paddykb's kernel beforehand and don't forget to upvote this one too ❗️❗️❗️

# In[ ]:




