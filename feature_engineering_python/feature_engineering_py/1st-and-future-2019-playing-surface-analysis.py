#!/usr/bin/env python
# coding: utf-8

# # Playing Surface Analysis
# ![](https://i.imgur.com/ZsERLAT.png)
# ## Lateral movement and increased injury risk.
# 
# This report contains my analysis for the 2019 NFL 1st and Future competition. The competition tasked data scientists to investigate the relationship between the playing surface and the injury and performance of National Football League (NFL) athletes and to examine factors that may contribute to lower extremity injuries.
# 
# I propose a metric for measuring the angle between athlete's body orientation and movement direction. I categorize player movements into three groups: forward, lateral, and backpedaling. I find that increased lateral player movement is strongly correlated with plays involving non-contact (NC) lower limb injuries. I then look to see if we can find a measurable link between increased lateral movement and playing surface.
# I then propose a hypothesis that if injury rate is linked to playing surface, then we may see a link between playing surface and lateral movement. My findings show that there is no link between lateral movement and playing surface.
# 
# Last, I offer three suggestions that I believe may help to identify players at high risk of injuries: 1) Monitor the percentage of lateral movement of players during game play and provide summary statistics on players, and play types. This could be integrated into already existing monitoring protocols.  2) Monitor  the orientation of players’ hips, shoulders and head. I also suggest collecting data that might elucidate the link between playing surface and injuries: player cleat data and dampness of playing surface could be a place to start. 3) Allow strength and conditioning coaches to play a role in injury prevention by educating them about the link between lateral movement and injuries for Linebackers, Wide Receivers and Defensive backs.  They may be able to use  data that I’ve suggested  they develop individualized workout plans.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os
import psutil

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import random
import sklearn
from itertools import cycle, islice

import warnings
warnings.filterwarnings("ignore")

from tqdm.notebook import tqdm
pd.set_option('max_columns', 500)
plt.style.use('fivethirtyeight')


# In[2]:


# Read in data
tracks = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv',
                        dtype={'time':'float64',
                                'x':'float16',
                                'y':'float16',
                                'dir': 'float16',
                                'dis': 'float16',
                                'o':'float16',
                                's':'float16'})

plays = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
injury = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')

# Create injury detailed by merging on play information
injury_detailed = injury.merge(plays, how='left')
injury_detailed = injury_detailed.merge(plays[['PlayerKey','RosterPosition']].drop_duplicates() \
                                            .rename(columns={'RosterPosition':'RosterPosition_notplay'}))
injury_detailed['RosterPosition_notplay'] = injury_detailed['RosterPosition_notplay'] \
    .replace({'Safety':'Defensive Back',
              'Cornerback' : 'Defensive Back'})


# # Background
# There has long been professional and academic interest in the differences regarding athletic performance measures and injury occurrence on various surface types (Meyers and Barnhill, 2004; Powell and Schootman, 1992). Recent studies have specifically examined synthetic turf (Mack et al., 2018; Loughran et al., 2019). These observational studies only show a correlation in injury rate-- they don’t make assertions about the specific mechanisms associated with lower body injury. My analysis attempts to find a specific link between a movement pattern and NC injury. Once I’ve identified a movement pattern link, can I find a measurable difference in that  movement pattern between playing surfaces (synthetic turf vs. natural turf)?
# 
# My analysis is based on data that the NFL has provided with the full player tracking of on-field position for 250 players over two regular season schedules. Of the 250 players provided, one hundred of them sustained one or more injuries during the study period. The other 150 serve as a control group (representing a sample of the non-injured NFL population).
# At first look, it’s provocative to note that, within the data set provided, there are numerically more NC injuries on synthetic turf. I hope to go beyond this purely observational approach, though, by also finding mechanisms associated with these injuries.

# In[3]:


plt.style.use('fivethirtyeight')
# Find Injury Rate by Surface
injury_playkeys = injury['PlayKey'].unique()
plays['counter'] = 1 # Column used when grouping to count
plays['isInjuryPlay'] = False
plays.loc[plays['PlayKey'].isin(injury_playkeys), 'isInjuryPlay'] = True

# Plot Results
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
(plays.groupby('FieldType')[['isInjuryPlay']].mean() * 100000).plot(kind='bar', ax=ax)
ax.get_legend().remove()
ax.set_xlabel('')
ax.set_title('Lower Body Injury Rate by Surface')
ax.set_ylabel('Injury per \n 100,000 plays', rotation=0, fontsize=13, color='darkgrey')
ax.yaxis.set_label_coords(-0.12,0.85)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + 0.20,
                 p.get_height() - 3),
                 fontsize=20,
                color='white')
ax.axhline(0, color='k')
plt.xticks(rotation=0)
plt.show()


# # Focusing on the players and plays at high risk
# 
# It’s intuitive, and quickly reveals itself in the data that certain players are at higher risk than others when it comes to NC lower body injuries. Among  the 100+ injuries from  the 2 seasons of game play, 70% (74/105) were sustained by just three positions (Defensive Backs, Linebackers and Wide Receivers). When thinking about why these positions might be at higher risk, I envisioned the movements commonly made by these players. I was struck by the fact that these men are spending a lot of time moving in open space and making quick changes in direction.

# In[4]:


#### USED FOR SLIDES

# ax = injury_detailed.groupby('RosterPosition_notplay')['PlayerKey'].count().sort_values() \
#     .plot(kind='barh', title='Non-Contact Injuries', figsize=(8, 5))
# count = 0
# for x in ax.patches:
#     if count > 3:
#         x.set_color('orange')
#     count += 1
# # plt.annotate('Three positions account for 2/3 of injuries', xy=(15, 1.5), fontsize=15, color='brown')
# # plt.arrow(15, 2, -0.3, 1, color='brown', head_width=0.2, head_length=0.2, lw=3)
# ax.set_title('All Non-Contact Injuries', fontsize=15)
# ax.grid(b=None, axis='y')
# ax.set_ylabel('')
# ax.set_xlabel('Injury Count', fontsize=15)
# plt.legend(['Excluded from study'])
# ax.axvline(0.05, color='black')

# rects = ax.patches
# # For each bar: Place a label
# for rect in rects:
#     # Get X and Y placement of label from rect.
#     x_value = rect.get_width()
#     y_value = rect.get_y() + rect.get_height() / 2

#     # Number of points between bar and label. Change to your liking.
#     space = -20
#     # Vertical alignment for positive values
#     ha = 'left'

#     # If value of bar is negative: Place label left of bar
#     if x_value < 0:
#         # Invert space to place label to the left
#         space *= -1
#         # Horizontally align label at right
#         ha = 'right'

#     # Use X value as label and format number with one decimal place
#     label = "{:.0f}".format(x_value)

#     # Create annotation
#     ax.annotate(
#         label,                      # Use `label` as label
#         (x_value, y_value),         # Place label at end of the bar
#         xytext=(space, 0),          # Horizontally shift label by `space`
#         textcoords="offset points", # Interpret `xytext` as offset in points
#         va='center',                # Vertically center label
#         ha=ha,
#         fontsize=14,
#         color='white')                      # Horizontally align label differently for
#                                     # positive and negative values.

# plt.show()


# In[5]:


fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ax = axes[0]
injury_detailed.groupby('RosterPosition_notplay')['PlayerKey'].count().sort_values() \
    .plot(kind='barh', title='Non-Contact Injuries', figsize=(10, 5), ax=ax)
count = 0
for x in ax.patches:
    if count > 3:
        x.set_color('orange')
    count += 1
# plt.annotate('Three positions account for 2/3 of injuries', xy=(15, 1.5), fontsize=15, color='brown')
# plt.arrow(15, 2, -0.3, 1, color='brown', head_width=0.2, head_length=0.2, lw=3)
ax.set_title('All Non-Contact Injuries', fontsize=15)
ax.grid(b=None, axis='y')
ax.set_ylabel('')
ax.set_xlabel('Injury Count', fontsize=15)
ax.axvline(0.1, color='black')
ax2 = axes[1]
injury_detailed.query('DM_M7 == 1').groupby('RosterPosition_notplay')['PlayerKey'] \
    .count() \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 5), ax=ax2)
count = 0
for x in ax2.patches:
    if count > 3:
        x.set_color('orange')
    count += 1
ax2.set_title('Non-Contact Injury > 1 week missed', fontsize=15)
ax2.grid(b=None, axis='y')
ax2.set_xlabel('Injury Count', fontsize=15)
ax2.set_ylabel('')
plt.subplots_adjust(wspace = 0.4)
fig.suptitle('3 Positions account for 70% of all injuries', fontsize=20)
plt.subplots_adjust(top=0.83)

rects = ax.patches
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = -20
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,
        fontsize=14,
        color='white')                      # Horizontally align label differently for
                                    # positive and negative values.
        
        
rects = ax2.patches
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = -20
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'center'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    ax2.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,
        fontsize=14,
        color='white')                      # Horizontally align label differently for
                                    # positive and negative values.
plt.legend(['Excluded from study'])
ax2.axvline(0.05, color='black')
plt.show()


# Special teams plays are inherently different from a normal play. As one would expect, most of the injuries (roughly 2/3) in the data were for non-special team plays.

# In[6]:


injury_detailed['PlayType_simple'] = injury_detailed['PlayType'] \
    .replace({'Kickoff Not Returned' : 'Kickoff',
              'Kickoff Returned' : 'Kickoff',
              'Punt Not Returned' : 'Punt',
              'Punt Returned' : 'Punt'})


ax = injury_detailed.groupby('PlayType_simple') \
    .count()['PlayKey'] \
    .sort_values().plot(kind='barh',
                        figsize=(8, 4),
                       title='Non-Contact Injury count by Play Type')
count = 0
#ax.text(16, 1.5, '*Over 70% occured during non-special teams plays', fontsize=12, color='brown')
for x in ax.patches:
    if count > 1:
        x.set_color('orange')
    count += 1
ax.grid(b=None, axis='y')
ax.set_ylabel('')
plt.xlabel('Injury Count', fontsize=15)

rects = ax.patches
# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = -20
    # Vertical alignment for positive values
    ha = 'center'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = "{:.0f}".format(x_value)

    # Create annotation
    ax.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha,
        fontsize=15,
        color='white')                      # Horizontally align label differently for
                                    # positive and negative values.
plt.legend(['Excluded from study'])
ax.axvline(0.05, color='black')
plt.show()


# With these findings in mind, I decided to focus my analysis on **Defensive Backs, Linebackers, and Wide Receivers during rushing and passing plays**. Narrowing the scope of my analysis allows me to clearly identify player movements that involve high risk of injury for these positions and play types while removing the "noise" of plays and positions that have very different movement patterns. 

# # Data Cleaning and Standardizing the Orientation Feature
# 
# It was noted in the data description that the orientation feature may not be completely reliable when considering "geography". I have some experience working with the NGS data in the 2019 Big Data Bowl. During that competition, it was found that the orientation was shifted 90 degrees for one of the seasons' data. To correct for this orientation difference, I determined the plays which appeared to be shifted and standardized them as to be consistent with the direction feature. This approach was first shared by John Miller in his [kernel](https://www.kaggle.com/jpmiller/how-to-adjust-orientation).
# 
# While not perfect, I believe these corrections and standardization techniques are appropriate for the purposes of my analysis. As the NFL continues to collect tracking data, I'm sure the quality of the orientation feature will increase, strengthing the quality of similar analyses.  
# 
# Next, I verified that the orientation was reasonable by plotting orientation at the moment of ball snap for all players. As you would expect, most players were facing towards the line of scrimmage at this time.
# 
# I also did some cleaning to the NGS data in order to hone in on moments of gameplay. I removed the portion of NGS data not associated with play time (0.1 second prior to the ball snap up until the moment of the final "event" in the play). I also capped play length to 25 seconds in order to exclude outliers.
# 
# Lastly, I computed some features to the NGS data later in my analysis, including the time since the snap and acceleration (difference in speed over 0.1 second). I also added binary indicators to the tracking data for plays involving injured players and plays involving an injury.

# In[7]:


# Remove any data for a play 0.1 second before snap
# print(tracks.shape)
tracks_snap = tracks[['PlayKey','x','y','time','event']].query('event == "ball_snap"')
tracks_snap = tracks_snap[['PlayKey','x','y','time']] \
    .rename(columns={'x':'x_snap',
                     'y':'y_snap',
                     'time':'time_snap'}).copy()
tracks = tracks.merge(tracks_snap, on='PlayKey', how='left')
tracks = tracks.query('time >= (time_snap - 0.1)')
# print(tracks.shape)

# Remove any data for a play 0.1 second after last event
# print(tracks.shape)
tracks_max_event = tracks.loc[~tracks['event'].isna()] \
    .groupby('PlayKey')['time'] \
    .max().reset_index()
tracks_max_event = tracks_max_event.rename(columns={'time': 'time_last_event'}).copy()
tracks = tracks.merge(tracks_max_event)
tracks['max_event'] = tracks.loc[tracks['time_last_event'] == tracks['time']]['event'].values[0]
tracks = tracks.query('time <= (time_last_event + 0.1)')
# print(tracks.shape)

# Fix orientation
# Reference: https://www.kaggle.com/jpmiller/how-to-adjust-orientation
# print(tracks.shape)
tough_guys = plays.loc[plays.PlayerDay >= 350, 'PlayerKey'].unique()
playlist_tough = plays[plays.PlayerKey.isin(tough_guys)].copy()
days = playlist_tough.groupby('PlayerDay')['PlayerGamePlay'].mean()

playlist_tough['Season'] = np.where(playlist_tough.PlayerDay<350, 1, 2)
games = playlist_tough.drop_duplicates('GameID')[['GameID', 'Season']]

tracks = tracks.merge(playlist_tough[['GameID', 'PlayKey']], on='PlayKey', how='left')
tracks = tracks.merge(games, on='GameID', how='left')
tracks['Season'] = tracks['Season'].fillna(-999) # Unknown season as -999

# Assume other seasons based on direction at snap - if orientation at snap is outside normal range, shift.
s1 = tracks.query('event == "ball_snap" and o < 50 and Season < 0')['PlayKey'].unique().tolist()
s2 = tracks.query('event == "ball_snap" and o > 325 and Season < 0')['PlayKey'].unique().tolist()
s3 = tracks.query('event == "ball_snap" and o < 225 and o > 125 and Season < 0')['PlayKey'].unique().tolist()

tracks.loc[(tracks['Season'] < 0) &
           (tracks['PlayKey'].isin(s1+s2+s3)), 'Season'] = 1
tracks.loc[(tracks['Season'] < 0) &
           (~tracks['PlayKey'].isin(s1+s2+s3)), 'Season'] = 2

# Change orientation for season 1
tracks['o'] = np.where(tracks.Season == 1,
                            np.mod(tracks.o+90, 360),
                            tracks.o
                            )
# print(tracks.shape)

# Previous speed, acceleration, absolute acceleration
tracks['s_prev1'] = tracks.groupby('PlayKey')['s'].shift(1)
tracks['a'] = tracks['s'] - tracks['s_prev1']
tracks['a_abs'] = np.abs(tracks['a'])

# Add playerkey
tracks = tracks.merge(plays[['PlayKey','PlayerKey']])

# Binary Features for track data
# If tracks is for injured player, play where injury occured, 
tracks = tracks.merge(plays[['PlayKey','RosterPosition','PositionGroup','FieldType','PlayType']], how='left')
tracks.loc[tracks['PositionGroup'].isin(['DB','WR','LB']), 'isInjuryPronePos'] = True
tracks['isInjuryPlay'] = False
tracks.loc[tracks['PlayKey'].isin(injury['PlayKey'].unique()), 'isInjuryPlay'] = True
tracks['isRushPass'] = False
tracks.loc[tracks['PlayType'].isin(['Rush','Pass']), 'isRushPass'] =  True
tracks['isInjuredPlayer'] = False
tracks.loc[tracks['PlayerKey'].isin(injury['PlayerKey'].unique()), 'isInjuredPlayer'] =  True

# Generalized Position groups focus on high injury roles
tracks['Position_inj'] = tracks['PositionGroup']
tracks.loc[~tracks['Position_inj'].isin(['LB','WR','DB']), 'Position_inj'] = 'Other'
tracks['Position_inj'] = tracks['Position_inj'].replace({'LB':'Linebacker',
                                'WR':'Wide Receiver',
                                'DB':'Defensive Back',
                                'Other':'Other Positions'})

# Time since the snap
tracks['time_since_snap'] = tracks['time']- tracks['time_snap']
tracks['time_since_snap'] = tracks['time_since_snap'].round(2)
# print(tracks.shape)
tracks = tracks.loc[tracks['time_since_snap'] < 25]
# print(tracks.shape)
tracks['counter'] = True # Used for aggregating counts


# In[8]:


tracks.query('event == "ball_snap"')['o'] \
    .plot(kind='hist',
          bins=50,
          figsize=(15, 5),
          title='Distribution of Orientation during Snap after Data Cleaning')
plt.show()


# # Computing the Orientation-Movement Angle
# 
# Defensive backs and Linebackers have responsibilities to track offensive players- they’re mirroring quick directional movements. As compared to their offensive opponents, they aren’t putting a lot of forethought into their path. Could this be placing statistically more strain on their knees and ankles? Wide receivers make quick movements down field and then make cuts in their routes to create space between themselves and defenders. Could lateral movements during their “cut” moments be an underlying factor in their NC injuries?
# 
# In this video clip, you can see the movements of Derrick Johnson moments before he tears his achilles tendon. He’s almost exclusively moving sideways before he goes down. This gave me an idea of what to focus on in the data.
# 
# ![](https://media.giphy.com/media/XDErlBCeSQ8TFsRx6n/giphy.gif)

# To quantify these types of lateral movements, I can calculate the angle between the direction a player is moving and the orientation he is facing.
# 
# ![](https://i.imgur.com/OmtVL37.png)
# 
# I created three specific movement groups based on the calculated angle: **Forward Movement** is when a player’s orientation is generally in line with the direction he’s moving; **Lateral Movement** is when a player is moving from side to side; and **Backpedaling** is when a player is moving in the opposite direction of his orientation.
# 
# <table style='font-family:"Courier", Courier, monospace; font-size:120%; boarder=10px'>
#   <tr>
#     <th>Orientation Movement Angle</th>
#     <th>Movement Type </th>
#   </tr>
#   <tr>
#     <td>0°-75°</td>
#     <td>Forward Movement</td>
#   </tr>
#   <tr>
#     <td>75°-105°</td>
#     <td>Lateral Movement</td>
#   </tr>
#   <tr>
#     <td>105°-180°</td>
#     <td>Backpedaling</td>
#   </tr>
# </table>
# 

# In[9]:


# O vs Dir feature
tracks['o_dir_diff1'] = np.abs(tracks['o'] - tracks['dir'])
tracks['o_dir_diff2'] = np.abs(tracks['o'] - (tracks['dir'] - 360))
tracks['o_dir_diff3'] = np.abs(tracks['o'] - (tracks['dir'] + 360))
tracks['o_dir_diff'] = tracks[['o_dir_diff1','o_dir_diff2','o_dir_diff3']].min(axis=1)
tracks = tracks.drop(['o_dir_diff1','o_dir_diff2','o_dir_diff3'], axis=1)

# Create movement groups
tracks['OffsetAngleGroup'] = 'Forward'
tracks.loc[tracks['o_dir_diff'] >= 75, 'OffsetAngleGroup'] = 'Lateral'
tracks.loc[tracks['o_dir_diff'] >= 105, 'OffsetAngleGroup'] = 'Backpedal'
tracks['isLateralMovement'] = False
tracks.loc[tracks['OffsetAngleGroup'] == 'Lateral', 'isLateralMovement'] = True


# # Analysis of movement category over duration of play
# 
# I found it useful to visualize the percentage of time a given position spends using each of these three movement categories during a play. Wide Receivers in our dataset tend to move forward right after the snap use lateral movement and backpedaling 2 to 3 seconds after the play. Defensive Backs spend much of the beginning of a play backpedaling and then quickly change to lateral or forward movement. Linebackers appear to have a mix of all three types of movements with a slight decrease in forward movement around 2-3 seconds after the snap.

# In[10]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# Linebacker
ax=axes[0]
t_group = tracks.query('isRushPass and Position_inj == "Linebacker" and time_since_snap < 5 and s > 0') \
    .groupby(['time_since_snap','OffsetAngleGroup'])['OffsetAngleGroup'] \
    .count() \
    .unstack('OffsetAngleGroup')
t_group.apply(lambda x: 100 * x / float(x.sum()), axis=1) \
    .plot(kind='area', stacked=True, alpha=0.5, ax=ax, title='Linebacker')
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xlabel('seconds after snap', fontsize=14)
ax.set_ylabel('% time in \n movement \n category', rotation=0, fontsize=10, color='darkgrey')
ax.yaxis.set_label_coords(-0.25,0.82)

ax.get_legend().remove()

# Defensive Back
ax=axes[1]
t_group = tracks.query('isRushPass and Position_inj == "Defensive Back" and time_since_snap < 5 and s > 0') \
    .groupby(['time_since_snap','OffsetAngleGroup'])['OffsetAngleGroup'] \
    .count() \
    .unstack('OffsetAngleGroup')
t_group.apply(lambda x: 100 * x / float(x.sum()), axis=1).plot(kind='area', stacked=True, alpha=0.5, ax=ax, title='Defensive Back')
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xlabel('seconds after snap', fontsize=14)
ax.set_ylabel('% time in \n movement \n category', rotation=0, fontsize=10, color='darkgrey')
ax.yaxis.set_label_coords(-0.25,0.82)
ax.get_legend().remove()

# Wide Receiver
ax=axes[2]
t_group = tracks.query('isRushPass and Position_inj == "Wide Receiver" and time_since_snap < 5 and s > 0') \
    .groupby(['time_since_snap','OffsetAngleGroup'])['OffsetAngleGroup'] \
    .count() \
    .unstack('OffsetAngleGroup')
t_group.apply(lambda x: 100 * x / float(x.sum()), axis=1).plot(kind='area', stacked=True, alpha=0.5, ax=ax, title='Wide Receiver')
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.set_xlabel('seconds after snap', fontsize=14)
ax.set_ylabel('% time in \n movement \n category', rotation=0, fontsize=10, color='darkgrey')
ax.yaxis.set_label_coords(-0.25,0.82)
# # Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# # Lateral Movement During Plays is Linked to Injury
# After looking at the data, it was clear to me that one factor closely linked to player injury was the amount of time during a play that he spent in lateral movement.

# In[11]:


t_group = tracks[['s','isRushPass','isInjuryPronePos',
                  'PlayKey','OffsetAngleGroup','isInjuryPlay']] \
    .loc[tracks['isRushPass'] & tracks['isInjuryPronePos']] \
    .groupby(['OffsetAngleGroup','isInjuryPlay'])['PlayKey'] \
    .count() \
    .unstack(['OffsetAngleGroup'])

fig, ax = plt.subplots(1,1, figsize=(8, 8))
t_group.apply(lambda x: 100 * x / float(x.sum()), axis=1)['Lateral'] \
    .plot(kind='bar',
          title='Time spent in Lateral Movement',
          figsize=(10, 4),
         ax=ax)
ax.xaxis.set_label('')
ax.set_ylabel('% of \n play time', rotation=0, color='darkgrey', fontsize=12)
ax.yaxis.set_label_coords(-0.08, 0.85)
ax.set_xlabel('')
for p in ax.patches:
    ax.annotate(f'{round(p.get_height(),1)}%', (p.get_x() + 0.2, p.get_height() - 1.5), color='white')
ax.set_xticklabels(['Non Injury Play', 'Injury Play'], rotation=0)
ax.axhline(0, color='black')
ax.axhline(10.1, linestyle='--', linewidth=2, color='orange')
# fig.annotate('*Linebackers, Defensive Backs, and Cornerbacks, not including special teams', (0, -0.001), fontsize=8)

plt.figtext(0.99, 0.01,
            '*LB, DB, and WRs not including special teams',
            fontsize=6,
            horizontalalignment='right')
plt.annotate(r"$\{$",fontsize=60,
            xy=(0.58, 0.7), xycoords='figure fraction'
            )
plt.annotate('40% increase', xy=(0.3, 12))
plt.grid(b=None, axis='x')


# When we look closer at the lateral movement by position, I found this relationship between lateral movement and injury is found across all three of my focus positions (Wide Receivers, Linebackers and Defensive Backs).

# In[12]:


# Numbers used in powerpoint presentation.
# (tracks.query('isRushPass') \
#     .groupby(['Position_inj','isInjuryPlay'])['isLateralMovement'].mean() * 100) \
#     .unstack('isInjuryPlay')


# In[13]:


injury_prone_pos = ['Wide Receiver', 'Linebacker', 'Defensive Back']
ax = (tracks.query('Position_inj in @injury_prone_pos and isRushPass') \
    .groupby(['Position_inj','isInjuryPlay'])['isLateralMovement'].mean() * 100) \
    .unstack('isInjuryPlay').plot(kind='barh', figsize=(10, 5),
                                  title='Time Spent in Lateral Movement')

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()-0.7, i.get_y()+.08, \
            str(int((i.get_width())))+'%', fontsize=10,
            color='white')

plt.legend(['Non-Injury Play', 'Injury Play'])
plt.xlabel('% of Play')
plt.ylabel('')
plt.grid(b=None, axis='y')


# # No evidence to support a link between playing surface and lateral movement
# 
# Previous observational studies have shown a link between injury and playing surface. My analysis for this report identifies an association between NC injuries and lateral movement during game play. Now I wonder: is there a measurable difference between time spent in lateral movement based  on playing surface? My hypothesis would be that, if players gain more traction when interacting with a type of turf, then I will observe a difference in lateral movement (and therefore a difference in NC injuries) on that turf.
# 
# One of the most straightforward ways to test my hypothesis is to examine the distribution of Orientation-Movement angles over plays on Natural and Synthetic playing surfaces. When I compared these distributions, I couldn't visually see any difference between turf type and the time spent in specific Orientation-Movement angles. To test if these distributions are different I calculated the Kolmogorov-Smirnov statistic on 2 samples. Since this statistic is very small (0.0096) then I cannot reject the hypothesis that the distributions of the two samples are the same.

# In[14]:


fig, ax= plt.subplots(1,1, figsize=(15, 5))
sns.distplot(tracks.query('FieldType == "Natural"')['o_dir_diff'].dropna(),
             hist=False, label='Natural', color='darkgreen')
sns.distplot(tracks.query('FieldType == "Synthetic"')['o_dir_diff'].dropna(),
             hist=False, label='Synthetic', color='mediumseagreen')
ax.set_ylabel('% of play time')
ax.set_xlabel('Orientation-Movement Angle')
ax.set_title('Player movement angle by Turf Type')
ax.legend(['Natural Turf', 'Synthetic Turf'])
plt.show()


# In[15]:


from scipy.stats import ks_2samp

ks_stat = ks_2samp(tracks.query('FieldType == "Natural"')['o_dir_diff'].dropna(),
        tracks.query('FieldType == "Synthetic"')['o_dir_diff'].dropna())[0]
print(f'The Kolmogorov-Smirnov statistic on 2 samples is {ks_stat:0.4f}')


# I also compared the differences of time spent in lateral movement by field type. I broke down the percentages by positions in a similar manner to how I compared injury plays to non-injury plays. I found more commonalities than differences. The difference for each position  were all less than 1%.

# In[16]:


injury_prone_pos = ['Wide Receiver', 'Linebacker', 'Defensive Back']
my_colors = list(islice(cycle(['darkgreen','mediumseagreen']), None, 3))
ax = (tracks.query('Position_inj in @injury_prone_pos and isRushPass') \
    .groupby(['Position_inj','FieldType'])['isLateralMovement'].mean() * 100) \
    .unstack('FieldType').plot(kind='barh', figsize=(10, 5),
                               title='Time Spent in Lateral Movement',
                               color=my_colors)

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()-0.8, i.get_y()+.08, \
            str(round(i.get_width(), 2))+'%', fontsize=10,
            color='white')

plt.xlabel('% of Play')
plt.ylabel('')
plt.show()


# I  created a regression model to capture any difference in the percent of lateral movement by player, based on features about the play. The regression included:
# - Position (one-hot-encoded)
# - PlayerDay (integer sequence reflecting timeline of a players participation in games)
# - PlayerGame (Uniquely identifies player’s games)
# - PlayerGamePlay (Ordered interger denoting the running count of plays the player has participated in during the game)
# - Synthetic (binary indicator if field type is Synthetic)
# - isInjuryPlay (binary indicator)
# 
# The regression coefficients from this model give us a view into which features are most closely tied to lateral movement. These coefficents tell us how much the dependent variable (percentage of lateral movement in a play) is impacted by a change in this feature.

# In[17]:


# Data Prep for linear model

play_movement_dir = tracks.loc[tracks['Position_inj'].isin(injury_prone_pos) &
                               tracks['isRushPass']] \
    .groupby(['PlayKey','PlayerKey','OffsetAngleGroup','Position_inj','FieldType',
              'isInjuredPlayer','isInjuryPlay']) \
    .count()['counter'] \
    .unstack('OffsetAngleGroup') \
    .fillna(0)
play_mov_pct = play_movement_dir.apply(lambda x: 100 * x / float(x.sum()), axis=1)
play_mov_pct = play_mov_pct.reset_index()

# Add play features and make dummy variables for categoricals
play_mov_pct_w_feats = plays.merge(play_mov_pct)
play_mov_pct_w_feats = pd.concat([play_mov_pct_w_feats, pd.get_dummies(play_mov_pct_w_feats['PlayType'])], axis=1)
play_mov_pct_w_feats = pd.concat([play_mov_pct_w_feats, pd.get_dummies(play_mov_pct_w_feats['Position_inj'])], axis=1)
play_mov_pct_w_feats = pd.concat([play_mov_pct_w_feats, pd.get_dummies(play_mov_pct_w_feats['FieldType'])], axis=1)

# Replace -999 Temperature values with average temperature
play_mov_pct_w_feats.loc[play_mov_pct_w_feats['Temperature'] < -100, 'Temperature'] = np.nan
play_mov_pct_w_feats['Temperature'] = play_mov_pct_w_feats['Temperature'].fillna(play_mov_pct_w_feats['Temperature'].mean())


# In[18]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
from scipy import stats


FEATURES = ['PlayerDay','PlayerGame','PlayerGamePlay','Rush','Temperature','Defensive Back',
            'Linebacker', # , DB and WR as 0 is equalt
            'Wide Receiver',
            'Synthetic','isInjuryPlay']
X = play_mov_pct_w_feats[FEATURES]
X['isInjuryPlay'] = X['isInjuryPlay'].astype(int)
y = play_mov_pct_w_feats['Lateral']


# Create and fit OLS Model
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()

# Display results of model
coeff_df = pd.DataFrame(est2.params, columns=['Regression Coefficient'])
# coeff_df['p-value'] = est2.get_robustcov_results().pvalues
def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color
coeff_df.sort_values('Regression Coefficient', ascending=False) \
    .drop('const') \
    .round(4) \
    .style.applymap(color_negative_red) \
    .set_table_attributes('style="font-size: 17px"') \
    .set_precision(4) \
    .apply(lambda x: ['background: lightgreen' if x.name == 'Synthetic' else '' for i in x], axis=1) \
    .apply(lambda x: ['background: lightgreen' if x.name == 'isInjuryPlay' else '' for i in x], axis=1)

plt.style.use('fivethirtyeight')
ax = coeff_df.sort_values('Regression Coefficient') \
     .drop('const') \
    .plot(kind='barh', figsize=(10, 5), title='Regression coefficents modeling Lateral Movement as % of play')
ax.get_legend().remove()
plt.show()


# This model wasn’t created for its predictive power, but rather as  an attempt to quantify the impact of each feature and its statistical significance. The R-squared value is quite low, however we can still gain insights from the model. The main insights are as follows:
# 1. The position type (Defensive Back, Linebacker, or Wide Receiver) has the strongest relationship with lateral movement.
# 2. Injury Play is also a strong indicator, this supports our analysis above.
# 3. Synthetic playing surface  demonstrated a (weakly) negative relationship with amount of lateral movement. Simply put, with all things taken into account synthetic turf decreases the percent of time spent in lateral movement by roughly 0.4%
# 
# To quickly summarize: My analysis shows we are not able to prove that there is a substantial link between playing surface and the amount of lateral movement of an athlete during a play.

# ## Conclusion and Recommendations
# 
# So there you have it. I've shown that high lateral movements have a strong relationship with injury plays. I've also shown that there isn't a relationship between playing surface and lateral movement. Because of that we can't conclude that turf type plays a role in increasing this specific type of high risk movement.
# 
# Given my findings I have the following suggestions for the NFL:
# 1. Monitor the percentage of lateral movement of players during game play and provide summary statistics on players, and play types. This could be integrated into already existing monitoring protocols.
# 2. Increase the quality of the orientation data. If possible include it to monitor the orientation of players’ hips, shoulders and head. I also suggest  collecting data that might elucidate the link between playing surface and injuries: player cleat data and dampness of playing surface could be a place to start.
# 3. Allow strength and conditioning coaches to play a role in injury prevention by educating them about the link between lateral movement and injuries for Linebackers, Wide Receivers and Defensive backs. They may be able to use data that I’ve suggested  they develop individualized workout plans.
# 
# Thanks for taking the time to read my analysis.

# # References
# - Murphy, D F. “Risk Factors for Lower Extremity Injury: a Review of the Literature.” British Journal of Sports Medicine, vol. 37, no. 1, Jan. 2003, pp. 13–29., doi:10.1136/bjsm.37.1.13.
# - Stockman, J.a. “Incidence, Causes, and Severity of High School Football Injuries On FieldTurf Versus Natural Grass: A 5-Year Prospective Study.” Yearbook of Pediatrics, vol. 2006, 2006, pp. 333–335., doi:10.1016/s0084-3954(07)70202-6.
# - Powell, John W., and Mario Schootman. “A Multivariate Risk Analysis of Selected Playing Surfaces in the National Football League: 1980 to 1989.” The American Journal of Sports Medicine, vol. 20, no. 6, 1992, pp. 686–694., doi:10.1177/036354659202000609.
# - Mack, Christina D., et al. “Higher Rates of Lower Extremity Injury on Synthetic Turf Compared With Natural Turf Among National Football League Athletes: Epidemiologic Confirmation of a Biomechanical Hypothesis.” The American Journal of Sports Medicine, vol. 47, no. 1, 2018, pp. 189–196., doi:10.1177/0363546518808499.
# - Mcmurtry, Shea, and Goeran Fiedler. “Comparison of Lower Limb Segment Forces during Running on Artificial Turf and Natural Grass.” Journal of Rehabilitation and Assistive Technologies Engineering, vol. 6, 2019, p. 205566831983570., doi:10.1177/2055668319835701.

# # Appendix
# 
# ## Details of Regression analysis
# 
# Expand the cell below to see the full details of the regression model.

# In[19]:


print(est2.summary())


# ## Machine Learning Model
# Below is an approach I worked on that did not provide conclusive results. I created three machine learning model based on players movement patterns. These models tried to predict using only player movement:
# - What surface type was the player on
# - If it was raining
# - If it was snowing
# 
# I chose these three models because I assumed that players movements would be impacted by rain and snow. If I was able to show similar accuracy when trying to predict playing surface then I could conclude that the NGS data had enough information to show surfaces. Unfortunately I wasn't able to determine any conclusive results.
# 
# Unfortunately, while machine learning models like the one I used in this analysis have very strong predictive power, they are in a sense black boxes where we can't easily understand how the model makes its predictions. Due to the lack of transparency I determined this analysis wouldn't be helpful for my report.
# 
#     Model Features:
#         - From -0.1 second prior to snap until 5 seconds after the snap at 0.1 second intervals:
#             1. Speed of the player
#             2. Acceleration of the player
#             3. Orientation-Direction Angle of the player
#         - If the play is Pass or Rushing
#         - The PlayerID

# In[20]:


# Preparing model training data
tracks_model = tracks.loc[(tracks['time_since_snap'] < 5) &
                           tracks['isRushPass'] &
                           tracks['isInjuryPronePos']]

tracks_model['a'] = tracks_model['a'].astype('float32')
tracks_model['s'] = tracks_model['s'].astype('float32')

# Every Play is a row, - create features for every 10th of a second up until 5 seconds after snap
pp_piv = tracks_model[['PlayKey',
                       'time_since_snap',
                       'o_dir_diff',
                       's',
                       'a',
                       'RosterPosition',
                       'Position_inj',
                       'FieldType']].groupby(['PlayKey','time_since_snap']).sum().unstack('time_since_snap')
pp_piv.columns = [col[0] + str(col[1]) for col in pp_piv.columns.to_flat_index()]
pp_piv = pp_piv.reset_index()
pp_piv = pp_piv.merge(plays[['PlayKey','FieldType','Temperature','PlayType','PlayerDay','PlayerGame','Weather','PlayerKey']])
pp_piv['isSynthetic'] = False
pp_piv.loc[pp_piv['FieldType'] == 'Synthetic', 'isSynthetic'] = True

o_dir_diff_cols = [f for f in pp_piv.columns if 'o_dir_diff' in f]
a_cols = [x for x in pp_piv.columns if 'a' in x and '.' in x]
s_cols = [x for x in pp_piv.columns if 's' in x and '.' in x]

# pp_piv['max_o_dir_diff'] = pp_piv[o_dir_diff_cols].max(axis=1)
# pp_piv['min_o_dir_diff'] = pp_piv[o_dir_diff_cols].min(axis=1)
# pp_piv['avg_o_dir_diff'] = pp_piv[o_dir_diff_cols].mean(axis=1)
# pp_piv['std_o_dir_diff'] = pp_piv[o_dir_diff_cols].std(axis=1)
pp_piv['isPass'] = pd.get_dummies(pp_piv['PlayType'])['Pass']
pp_piv = pp_piv.fillna(0)


# In[21]:


plt.style.use('default')
ax = pp_piv[o_dir_diff_cols].head(50).T \
    .plot(title='Orientation-Movement Angle over first 5 seconds of play',
          figsize=(15, 4), color='grey')
ax.get_legend().remove()
plt.show()
ax = pp_piv[a_cols].head(50).T \
    .plot(title='Acceleration over first 5 seconds of play',
         figsize=(15, 4), color='brown')
ax.get_legend().remove()
plt.show()
ax = pp_piv[s_cols].sample(50).T \
    .plot(title='Speed over first 5 seconds of play',
          figsize=(15, 4), color='lightblue')
ax.get_legend().remove()
plt.show()


# In[22]:


# Add features for percipitation and snow based on weather feature
weather_percip_mapping = {
    'Controlled Climate' : False,
    'Sunny' : False,
    0 : False,
    'Cloudy' : False,
    'Clear' : False,
    'N/A Indoor' : False,
    'Partly sunny' : False,
    'N/A (Indoors)' : False,
    'Sunny and clear' : False,
    'Partly Cloudy' : False,
    'Snow' : True,
    'Indoor' : False,
    'Indoors' : False,
    'Showers' : True,
    'Rain' : True,
    'Clear and warm' : False,
    'Mostly Cloudy' : False,
    'Mostly Sunny' : False,
    'Clear skies' : False,
    'Party Cloudy' : False,
    'Hazy' : False,
    'Partly Clouidy': False,
    'Sunny Skies' : False,
    'Overcast' : False,
    'Rain likely, temps in low 40s.' : True,
    'Cloudy, 50% change of rain' : True,
    'Sunny and warm' : False,
    'Partly cloudy' : False,
    'Clear and Cool' : False,
    'Clear and cold' : False,
    'Sunny and cold' : False,
    'Cloudy, fog started developing in 2nd quarter' : False,
    'Scattered Showers' : True,
    'Heat Index 95' : False,
    'Mostly cloudy' : False,
    'Sunny, highs to upper 80s' : False,
    'Fair' : False,
    'Partly Sunny' : False,
    'Sunny, Windy' : False,
    'Mostly Sunny Skies' : False,
    'Cloudy and Cool' : False,
    'Mostly Coudy' : False,
    'Mostly sunny' : False,
    'Rainy' : True,
    'Cloudy and cold' : False,
    'Rain Chance 40%' : True,
    '30% Chance of Rain' : True,
    '10% Chance of Rain' : True,
    'Cloudy, chance of rain' : True,
    'Light Rain' : True, 
    'cloudy' : False,
    'Clear and Sunny': False,
    'Partly clear' : False,
    'Coudy' : False,
    'Cloudy, Rain' : True,
    'Sun & clouds' : False,
    'Clear to Partly Cloudy' : False,
    'Heavy lake effect snow' : True,
    'Clear and sunny' : False,
    'Rain shower' : True,
    'Cloudy, light snow accumulating 1-3"' : True,
    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' : True,
    'Clear Skies' : False,
    'Cold': False
}

pp_piv['Precipitation'] = pp_piv['Weather'].map(weather_percip_mapping)

weather_snow_mapping = {
    'Controlled Climate' : False,
    'Sunny' : False,
    0 : False,
    'Cloudy' : False,
    'Clear' : False,
    'N/A Indoor' : False,
    'Partly sunny' : False,
    'N/A (Indoors)' : False,
    'Sunny and clear' : False,
    'Partly Cloudy' : False,
    'Snow' : True,
    'Indoor' : False,
    'Indoors' : False,
    'Showers' : False,
    'Rain' : False,
    'Clear and warm' : False,
    'Mostly Cloudy' : False,
    'Mostly Sunny' : False,
    'Clear skies' : False,
    'Party Cloudy' : False,
    'Hazy' : False,
    'Partly Clouidy': False,
    'Sunny Skies' : False,
    'Overcast' : False,
    'Rain likely, temps in low 40s.' : False,
    'Cloudy, 50% change of rain' : False,
    'Sunny and warm' : False,
    'Partly cloudy' : False,
    'Clear and Cool' : False,
    'Clear and cold' : False,
    'Sunny and cold' : False,
    'Cloudy, fog started developing in 2nd quarter' : False,
    'Scattered Showers' : False,
    'Heat Index 95' : False,
    'Mostly cloudy' : False,
    'Sunny, highs to upper 80s' : False,
    'Fair' : False,
    'Partly Sunny' : False,
    'Sunny, Windy' : False,
    'Mostly Sunny Skies' : False,
    'Cloudy and Cool' : False,
    'Mostly Coudy' : False,
    'Mostly sunny' : False,
    'Rainy' : False,
    'Cloudy and cold' : False,
    'Rain Chance 40%' : False,
    '30% Chance of Rain' : False,
    '10% Chance of Rain' : False,
    'Cloudy, chance of rain' : False,
    'Light Rain' : False, 
    'cloudy' : False,
    'Clear and Sunny': False,
    'Partly clear' : False,
    'Coudy' : False,
    'Cloudy, Rain' : False,
    'Sun & clouds' : False,
    'Clear to Partly Cloudy' : False,
    'Heavy lake effect snow' : True,
    'Clear and sunny' : False,
    'Rain shower' : False,
    'Cloudy, light snow accumulating 1-3"' : True,
    'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.' : False,
    'Clear Skies' : False,
    'Cold': False
}

pp_piv['Snow'] = pp_piv['Weather'].map(weather_snow_mapping)


# ## Show the first few rows of features used in the model
# - s -> speed features
# - a -> acceleration features
# - o_dir_diff -> angle difference between orientation and direction

# In[23]:


# Add position
pp_piv = pp_piv.merge(tracks_model[['Position_inj','PlayKey']].drop_duplicates(), on='PlayKey', how='left')
pp_piv = pd.concat([pp_piv, pd.get_dummies(pp_piv['Position_inj'])], axis=1)
pp_piv['Position_inj'] = pp_piv['Position_inj'].astype('category')
# Perform a train / test split
X = pp_piv.drop(['FieldType','PlayKey','isSynthetic','PlayType','Temperature',
                 'PlayerGame','PlayerDay','Weather','PlayerKey',
                 'Precipitation', 'Snow'], axis=1)
y = pp_piv['isSynthetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=529)
X.head()


# In[24]:


params = {}
params['max_bin'] = 50
params['learning_rate'] = 0.01
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'

##################################################
# TRAIN MODEL TO PREDICT FIELD TYPE (isSynthetic)
##################################################

X = pp_piv.drop(['FieldType','PlayKey','isSynthetic','PlayType','Temperature',
                 'PlayerGame','PlayerDay','Weather','PlayerKey',
                 'Precipitation', 'Snow'], axis=1)
y = pp_piv['isSynthetic']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=529)

d_train = lgb.Dataset(X_train, label=y_train) #, categorical_feature=['PlayerKey'])
d_test = lgb.Dataset(X_test, label=y_test)
lgbm_model_FieldType = lgb.train(params,
                       d_train,
                       valid_sets=(d_train, d_test),
                       num_boost_round=5000,
                       verbose_eval=False,
                       early_stopping_rounds=200)

y_pred_train = lgbm_model_FieldType.predict(X_train)
y_pred_test = lgbm_model_FieldType.predict(X_test)

train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train.round())
test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test.round())
train_auc = lgbm_model_FieldType.best_score['training']['auc']
test_auc = lgbm_model_FieldType.best_score['valid_1']['auc']
print('Turf Type Prediction Model:')
print(f'Training accuracy: {train_acc:0.4}')
print(f'Test accuracy:     {test_acc:0.4}')
print(f'Training AUC:      {train_auc:0.4}')
print(f'Test AUC:          {test_auc:0.4}')

fi_df_lgbm_model_FieldType = pd.DataFrame(index=lgbm_model_FieldType.feature_name(),
             data=lgbm_model_FieldType.feature_importance(),
             columns=['importance']
            )


# In[25]:


##################################################
# TRAIN MODEL TO PREDICT PRECIPITATION
##################################################

X = pp_piv.drop(['FieldType','PlayKey','isSynthetic','PlayType','Temperature',
                 'PlayerGame','PlayerDay','Weather','PlayerKey',
                 'Precipitation', 'Snow'], axis=1)
y = pp_piv['Precipitation']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=529)

d_train = lgb.Dataset(X_train, label=y_train) #, categorical_feature=['PlayerKey'])
d_test = lgb.Dataset(X_test, label=y_test)
lgbm_model_PRECIP = lgb.train(params,
                       d_train,
                       valid_sets=(d_train, d_test),
                       num_boost_round=5000,
                       verbose_eval=False,
                       early_stopping_rounds=200)

y_pred_train = lgbm_model_PRECIP.predict(X_train)
y_pred_test = lgbm_model_PRECIP.predict(X_test)

train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train.round())
test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test.round())
train_auc = lgbm_model_PRECIP.best_score['training']['auc']
test_auc = lgbm_model_PRECIP.best_score['valid_1']['auc']
print('Raining Prediction Model:')
print(f'Training accuracy: {train_acc:0.4}')
print(f'Test accuracy:     {test_acc:0.4}')
print(f'Training AUC:      {train_auc:0.4}')
print(f'Test AUC:          {test_auc:0.4}')

fi_df_lgbm_model_PRECIP = pd.DataFrame(index=lgbm_model_PRECIP.feature_name(),
             data=lgbm_model_PRECIP.feature_importance(),
             columns=['importance']
            )


# In[26]:


#####################################
# TRAIN MODEL TO PREDICT IF SNOWING
#####################################

X = pp_piv.drop(['FieldType','PlayKey','isSynthetic','PlayType','Temperature',
                 'PlayerGame','PlayerDay','Weather','PlayerKey',
                 'Precipitation', 'Snow'], axis=1)
y = pp_piv['Snow']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=529)

d_train = lgb.Dataset(X_train, label=y_train)#, categorical_feature=['PlayerKey'])
d_test = lgb.Dataset(X_test, label=y_test)
lgbm_model_SNOW = lgb.train(params,
                       d_train,
                       valid_sets=(d_train, d_test),
                       num_boost_round=5000,
                       verbose_eval=False,
                       early_stopping_rounds=200)

y_pred_train = lgbm_model_SNOW.predict(X_train)
y_pred_test = lgbm_model_SNOW.predict(X_test)

train_acc = sklearn.metrics.accuracy_score(y_train, y_pred_train.round())
test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test.round())
train_auc = lgbm_model_SNOW.best_score['training']['auc']
test_auc = lgbm_model_SNOW.best_score['valid_1']['auc']
print('Snowing Prediction Model:')
print(f'Training accuracy: {train_acc:0.4}')
print(f'Test accuracy:     {test_acc:0.4}')
print(f'Training AUC:      {train_auc:0.4}')
print(f'Test AUC:          {test_auc:0.4}')

fi_df_lgbm_model_SNOW = pd.DataFrame(index=lgbm_model_SNOW.feature_name(),
             data=lgbm_model_SNOW.feature_importance(),
             columns=['importance']
            )


# In[27]:


# Concat results
fi_df = pd.concat([fi_df_lgbm_model_SNOW, fi_df_lgbm_model_FieldType, fi_df_lgbm_model_PRECIP], axis=1)
fi_df.columns = ['snow_importance','fieldtype_importance','precipitation_importance']


# In[28]:


fig, axes = plt.subplots(1,3,figsize=(15, 5))
fi_df.sort_values('fieldtype_importance', ascending=True).tail(10)[['fieldtype_importance']] \
    .plot(kind='barh', ax=axes[0], title='Top 10 Features to predict FieldType')
fi_df.sort_values('precipitation_importance', ascending=True).tail(10)[['precipitation_importance']] \
    .plot(kind='barh', ax=axes[1], title='Top 10 Features to predict precipitation')
fi_df.sort_values('snow_importance', ascending=True).tail(10)[['snow_importance']] \
    .plot(kind='barh', ax=axes[2], title='Top 10 Features to predict snow')
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()
plt.show()

