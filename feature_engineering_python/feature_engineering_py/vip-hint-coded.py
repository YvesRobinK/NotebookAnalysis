#!/usr/bin/env python
# coding: utf-8

# # Presentation

# * **v_1 update**: as in the reference paper, I created a function that modulates the influence radius of a player as a function of his distance to the ball. Indeed, the farther you're from the ball, the biggest influence you can have in terms of pitch covering. I reimplemented their function but I'm sure you could gain some insight by adapting it with "NFL relevant" values.
# * **v_5 update**: 2 main modifications: (1) I speed up the code by modifying calls to the dataframe (2-3 times faster but there's still room for improvement) ; (2) I created a class wrapping the functions defined in the first part, including the display of the pitch control visually (+ some typo fixes). 

# In this notebook, I code the **VIP hint** given by @MichaelLopez. You can find the paper at : http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf 
# <break>
# 
# I just implemented their version of the algorithm, without any specific attention to the particular context of this competition. I think several thing can be tailored to be "NFL specific". I mainly refer of the influence `radius` of a player, which is set to lie in $[4, 10]$ in football, but that may well be very different here. 
# Once you have the influence of the player and the pitch control, you'll have to create features from that.

# # Imports

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import scipy
from random import choice
from scipy.spatial.distance import euclidean
from scipy.special import expit
from tqdm import tqdm


train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# # Little Standardization Step

# Just a little preprocessing step to ensure we have the X and Y standardized

# In[2]:


def standardize_dataset(train):
    train['ToLeft'] = train.PlayDirection == "left"
    train['IsBallCarrier'] = train.NflId == train.NflIdRusher
    train['TeamOnOffense'] = "home"
    train.loc[train.PossessionTeam != train.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    train['IsOnOffense'] = train.Team == train.TeamOnOffense # Is player on offense?
    train['YardLine_std'] = 100 - train.YardLine
    train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
            'YardLine_std'
             ] = train.loc[train.FieldPosition.fillna('') == train.PossessionTeam,  
              'YardLine']
    train['X_std'] = train.X
    train.loc[train.ToLeft, 'X_std'] = 120 - train.loc[train.ToLeft, 'X'] 
    train['Y_std'] = train.Y
    train.loc[train.ToLeft, 'Y_std'] = 53.3 - train.loc[train.ToLeft, 'Y'] 
    train['Orientation_std'] = train.Orientation
    train.loc[train.ToLeft, 'Orientation_std'] = np.mod(180 + train.loc[train.ToLeft, 'Orientation_std'], 360)
    train['Dir_std'] = train.Dir
    train.loc[train.ToLeft, 'Dir_std'] = np.mod(180 + train.loc[train.ToLeft, 'Dir_std'], 360)
    train.loc[train['Season'] == 2017, 'Orientation'] = np.mod(90 + train.loc[train['Season'] == 2017, 'Orientation'], 360)    
    
    return train


# In[3]:


dominance_df = standardize_dataset(train_df)
dominance_df['Rusher'] = dominance_df['NflIdRusher'] == dominance_df['NflId']

dominance_df.head(3)


# # Player influence Area

# Quoting from the paper : specifically, the player’s influence I at a given location p for a given player i at time t is defined by a bivariate normal distribution with mean $\mu_i(t)$ and covariance matrix $\sum_i(t)$, given the player’s velocity $s$ and angle $\theta$. For a given location in space $p$ at time $t$, the probability density function of player $i$ influence area is defined by a standard multivariate normal distribution. The player’s influence likelihood is then
# defined as the normalization of $f$ at the given location $p$ by the value of $f$ at player’s current location $p_i(t)$, as shown in the following equation :
# <break>
# $I_i(p,t) = \frac{f_i(p,t)}{f_i(p_i(t),t)}$

# In[4]:


def radius_calc(dist_to_ball):
    ''' I know this function is a bit awkward but there is not the exact formula in the paper,
    so I try to find something polynomial resembling
    Please consider this function as a parameter rather than fixed
    I'm sure experts in NFL could find a way better curve for this'''
    return 4 + 6 * (dist_to_ball >= 15) + (dist_to_ball ** 3) / 560 * (dist_to_ball < 15)

# we plot the player influence radius relation with distance to the ball
x = np.linspace(0, 27, 250)
plt.figure(figsize=(8, 5))
plt.plot(x, radius_calc(x), c='g', linewidth=3, alpha=.7, label='Radius in metres')
plt.grid()
plt.ylim(-0.5, 11)
plt.ylabel('Influence radius')
plt.xlabel('Distance to the ball')
plt.title('Influence radius as a function of the distance to the ball')
plt.legend(loc=2)
plt.show()


# In[5]:


@np.vectorize
def compute_influence(x_point, y_point, player_id):
    '''Compute the influence of a certain player over a coordinate (x, y) of the pitch
    '''
    point = np.array([x_point, y_point])
    player_row = my_play.loc[player_id]
    theta = math.radians(player_row[56])
    speed = player_row[5]
    player_coords = player_row[54:56].values
    ball_coords = my_play[my_play['IsBallCarrier']].iloc[:, 54:56].values
    
    dist_to_ball = euclidean(player_coords, ball_coords)

    S_ratio = (speed / 13) ** 2    # we set max_speed to 13 m/s
    RADIUS = radius_calc(dist_to_ball)  # updated

    S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])
    R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))
    
    norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    
    mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2
    
    intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),
                                    np.linalg.inv(COV_matrix)),
                             np.transpose((player_coords - mu_play)))
    player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])
    
    intermed_scalar_point = np.dot(np.dot((point - mu_play), 
                                    np.linalg.inv(COV_matrix)), 
                             np.transpose((point - mu_play)))
    point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])

    return point_influence / player_influence


# # Pitch Control

# Quoting from the paper : based on this, we present a pitch control model that summarizes the level of influence of every player,and outputs a degree of control for any part of the pitch. Equation 2 presents the pitch control level at
# a location $p$ at time $t$, where $i$ and $j$ refers to the index of the player in each opposing team. Here the logistic function transforms the substraction of the accumulated individual influence area of each team into a degree of control within the $[0, 1]$ range. Also, since we are defining a team-oriented pitch control model, a single player without any influence of any other player at its current location only controls
# `logistic(1) = 0.73` of the space. This provides the need of higher density of players near a given area to provide higher level of control in that area.
# We have the following equation : 
# <break>
# $PC(p,t)=\sigma(\sum_i I(p,t) - \sum_j I(p,t))$  ($\sigma$ is the logistic function)

# In[6]:


@np.vectorize
def pitch_control(x_point, y_point):
    '''Compute the pitch control over a coordinate (x, y)'''

    offense_ids = my_play[my_play['IsOnOffense']].index
    offense_control = compute_influence(x_point, y_point, offense_ids)
    offense_score = np.sum(offense_control)

    defense_ids = my_play[~my_play['IsOnOffense']].index
    defense_control = compute_influence(x_point, y_point, defense_ids)
    defense_score = np.sum(defense_control)

    return expit(offense_score - defense_score)


# In[7]:


# select a random play to be plotted
my_play_id = choice(dominance_df['PlayId'].unique())           # if you want a random play_id
my_play = dominance_df[dominance_df['PlayId']==20180916051806]

player_coords = my_play[my_play['Rusher']][['X', 'Y']].values[0]
print('Player coordinates: ', player_coords)


# In[8]:


front = 15
behind = 5
left = right = 20
num_points_meshgr = (30, 15)   # don't make it too large otherwise it'll be long to run


colorm = ['purple'] * 11 + ['orange'] * 11
colorm[np.where(my_play.Rusher.values)[0][0]] = 'black'

X, Y = np.meshgrid(np.linspace(player_coords[0] - behind, 
                               player_coords[0] + front, 
                               num_points_meshgr[0]), 
                   np.linspace(player_coords[1] - left, 
                               player_coords[1] + right, 
                               num_points_meshgr[1]))

# infl is an array of shape num_points with values in [0,1] accounting for the pitch control
infl = pitch_control(X, Y)

plt.figure(figsize=(12, 8))
plt.contourf(X, Y, infl, 12, cmap ='bwr')
plt.scatter(my_play['X'].values, my_play['Y'].values, c=colorm)
plt.title('Yards gained = {}, play_id = {}'.format(my_play['Yards'].values[0], my_play_id))
plt.show()


# ### Class Wrapper

# In[9]:


class Controller:
    '''This class is a wrapper for the two functions written above'''
    def __init__(self, play):
        self.play = play
        self.vec_influence = np.vectorize(self.compute_influence)
        self.vec_control = np.vectorize(self.pitch_control) 
        
    def compute_influence(self, x_point, y_point, player_id):
        '''Compute the influence of a certain player over a coordinate (x, y) of the pitch
        '''
        point = np.array([x_point, y_point])
        player_row = self.play.loc[player_id]
        theta = math.radians(player_row[56])
        speed = player_row[5]
        player_coords = player_row[54:56].values
        ball_coords = self.play[self.play['IsBallCarrier']].iloc[:, 54:56].values

        dist_to_ball = euclidean(player_coords, ball_coords)

        S_ratio = (speed / 13) ** 2         # we set max_speed to 13 m/s
        RADIUS = radius_calc(dist_to_ball)  # updated

        S_matrix = np.matrix([[RADIUS * (1 + S_ratio), 0], [0, RADIUS * (1 - S_ratio)]])
        R_matrix = np.matrix([[np.cos(theta), - np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        COV_matrix = np.dot(np.dot(np.dot(R_matrix, S_matrix), S_matrix), np.linalg.inv(R_matrix))

        norm_fact = (1 / 2 * np.pi) * (1 / np.sqrt(np.linalg.det(COV_matrix)))    
        mu_play = player_coords + speed * np.array([np.cos(theta), np.sin(theta)]) / 2

        intermed_scalar_player = np.dot(np.dot((player_coords - mu_play),
                                        np.linalg.inv(COV_matrix)),
                                 np.transpose((player_coords - mu_play)))
        player_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_player[0, 0])

        intermed_scalar_point = np.dot(np.dot((point - mu_play), 
                                        np.linalg.inv(COV_matrix)), 
                                 np.transpose((point - mu_play)))
        point_influence = norm_fact * np.exp(- 0.5 * intermed_scalar_point[0, 0])

        return point_influence / player_influence
    
    
    def pitch_control(self, x_point, y_point):
        '''Compute the pitch control over a coordinate (x, y)'''

        offense_ids = self.play[self.play['IsOnOffense']].index
        offense_control = self.vec_influence(x_point, y_point, offense_ids)
        offense_score = np.sum(offense_control)

        defense_ids = self.play[~self.play['IsOnOffense']].index
        defense_control = self.vec_influence(x_point, y_point, defense_ids)
        defense_score = np.sum(defense_control)

        return expit(offense_score - defense_score)
    
    def display_control(self, grid_size=(30, 15), figsize=(11, 7)):
        front, behind = 15, 5
        left, right = 20, 20

        colorm = ['purple'] * 11 + ['orange'] * 11
        colorm[np.where(self.play.Rusher.values)[0][0]] = 'black'
        player_coords = self.play[self.play['Rusher']][['X_std', 'Y_std']].values[0]

        X, Y = np.meshgrid(np.linspace(player_coords[0] - behind, 
                                       player_coords[0] + front, 
                                       grid_size[0]), 
                           np.linspace(player_coords[1] - left, 
                                       player_coords[1] + right, 
                                       grid_size[1]))

        # infl is an array of shape num_points with values in [0,1] accounting for the pitch control
        infl = self.vec_control(X, Y)

        plt.figure(figsize=figsize)
        plt.contourf(X, Y, infl, 12, cmap='bwr')
        plt.scatter(self.play['X'].values, self.play['Y'].values, c=colorm)
        plt.title('Yards gained = {}, play_id = {}'.format(self.play['Yards'].values[0], 
                                                           self.play['PlayId'].unique()[0]))
        plt.show()


# In[10]:


# Example: how to use the class?
control = Controller(my_play)
coords = my_play.iloc[1, 54:56].values         # let's compute the influence at the location of the first player
pitch_control = control.vec_control(*coords)
pitch_control


# In[11]:


control.display_control()


# **Don't hesitate if you have any question!**

# In[ ]:




