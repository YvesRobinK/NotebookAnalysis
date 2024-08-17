#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering
# The goal of this notebook is to try to brainstorm from the data given what might help us predict the yards gained on a play

# I'm going to focus on six different kinds of factors:
# 1.  **Formation**:
# Even if you have the best players in the world, if you line them up in the wrong formation your team might lose big yardage. Inversely it doesn't matter how fast your running back is if you don't open a lane for him to run.  
# 2. **Overall Team Attributes**: 
# Having a strong overall team is very important to protecting the ball carrier. It only takes one defender breaking your offensive line to totally ruin a play. A game can be signficantly be influence by a standout individual, but overall strength of the rest of the team is going to greatly affect consistency and can recover botched plays.
# 3. **Individual Player Attributes**: As stated previously an individual can have a huge impact on a play. Either a running back who's exceptionally fast and shifty and can get past everyone even in tight spaces or a great defender breaking through and stopping the runner before he even gets up to speed. 
# 4. **Matchups**: One of the biggest chances for a big play is one where there is a large mismatch in personnel. One team has their fastest and strongest player and they somehow get pitted against someone signficantly undersized and they just get overpowered. Managing it so that the constantly shifting 11 attacking players is matching with equally matched defending 11 players is difficult. 
# 
# 5. **Weather Conditions**:I don't think this is going to be the biggest factor in yardage but it still can play some part. A team might choose to run the ball instead of pass even when they would rather be passing when it's extremely cold or windy or the rushers may run out of steam over time if they get a lot of plays and it's exceptionally hot. 
# 
# 6. **Game Condition**: Position on the field greatly effects which plays are called and also how desparate they are for yards. A team may intentionally go for a 1 yard play just to get a down or they may be desparate and intentionally open up and risk everything. They may be crunched for time and need to just make something happen at the end of the game before the clock ticks down. One additional thing I think some people might miss is that sometimes this can directly limit the distribution possible. It is impossible to get 100+ yards gain when a team is on the 5 yard line and inversely it is impossible to lose 30 yards when a team is on their own 20. 

# In[1]:


import pandas as pd
train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")


# Instead of running this calculation on all of the plays I am going to focus in on just one specific play in order to calculate all of the features and then this can be extended to future plays when actually doing the modeling

# In[2]:


first_play_id = train["PlayId"].unique()[0]


# In[3]:


first_play = train[train["PlayId"]==first_play_id]


# # Formation

# Let's look at what we're actually given here. I'll try to focus purely on the formation data in this section

# In[4]:


formation_cols = ["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "OffenseFormation",
                  "OffensePersonnel", "DefendersInTheBox", "Position"]


# Here are the columns I would say are related to formation
# 
# **X, Y** coordinates define where the player is on the field
# 
# **S** defines what speed they are going and **A** is acceleration
# 
# **Dis** is the Distance they have traveled in the last 1/10 of a second
# 
# **Orientation** is which direction they are facing in degrees
# 
# **Dir** is which direction they were moving in degrees
# 
# 
# The other features are a bit self explanatory, which formation, made up of what type of players. If you aren't familiar with SHOTGUN formation or what a RB is then checkout Tarek Hamdi's write up here: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/111945#latest-646323
# 

# In[5]:


first_play[formation_cols]


# So what can we extract out of these values. We can start with some simple offensive and defensive aggregates. Is the mean of the X,Y for each team overall similar? In theory the defensive team should be trying to match positions with the attackers. A big discrepency in these values might mean the defense is about to get beaten. The difference between these values may be important

# In[6]:


first_play.groupby("Team")[["X", "Y"]].mean()


# In[7]:


xy_diff = first_play.groupby("Team")[["X", "Y"]].mean().diff().iloc[-1]


# These values may need to be corrected to account for the different directions the team can be heading. i.e left and right may flip the X values.

# In[8]:


xy_diff


# One important aspect of having a high yardage rush is finding a way to make the defense spread out and allow pockets for the running back to run through. We can measure this in several different ways. One simple way would be simply measuring the std of the y values as a proxy to measure how spread out the defense and offenses lined up and we could also measure the difference between the defense's width and the offense's width. We could also apply this to the X values but I don't think this will have as signficant of an impact.

# In[9]:


team_widths = first_play.groupby("Team")[["Y"]].std()
team_widths


# We can see that the away team lined up with slightly more spread than the home team. 

# In[10]:


diff_widths = first_play.groupby("Team")[["Y"]].std().diff().iloc[-1]
diff_widths


# One feature we may want to aggregate is the Dis feature. We could look at important statistics like maximum distance traveled of any player. A defense likely wants movement to match the offense so if there is a big difference between distances that may mean an opening may have been created. 

# In[11]:


first_play.groupby("Team").agg(["max", "min", "std", "mean"])["Dis"]


# Looking at these features we can see that the home team had a larger max movement, a player/players who didn't move at all and slightly higher std and a much higher mean. This means the home team had a lot more activity than the away team. It would be interesting to check what these distributions look like comparing offenses and defenses across the whole dataset

# # Overall Team Attributes

# Looking at the overall team attributes we can see things like average **PlayerHeight**, **PlayerWeight** and **PlayerBirthDate** to determine if one team might totally overpower another.

# First we will correct the height measurements to be in inches. Right now it is in 6-0 format. With 6 denoting 6 feet and zero inches tall. We can simply split on the dash and then multiply the first part by 12 to convert to inches and then add them together

# In[12]:


def correct_height(row):
    split_height = row["PlayerHeight"].split("-")
    ft = split_height[0]
    inch = split_height[1]
    height = int(ft)*12 + int(inch)
    return height
first_play["CorrectedHeight"] = first_play.apply(correct_height, axis = 1)


# Now by aggregating on both of these we can see that there is a huge 81 inch tall player on the home team and they have a higher std. As always it is likely useful to measure the difference between these values. We can also see with the PlayerWeight aggregates that the home team has a huge 335lb player and a much higher mean weight.

# In[13]:


first_play.groupby("Team").agg(["max", "min", "mean", "std"])[["CorrectedHeight", "PlayerWeight"]]


# Now looking at the **PlayerBirthDate** we could potentially get the exact age of the player in days but it's probably reasonable enough to just extract their year of birth and compare that to the year of the game for a coarse look at their age.

# In[14]:


def correct_age(row):
    year_handoff = int(row["TimeHandoff"].split("-")[0])
    year_birth = int(row["PlayerBirthDate"].split("/")[-1])
    return year_handoff - year_birth
first_play["PlayerAge"] = first_play.apply(correct_age, axis = 1)


# In[15]:


first_play.groupby("Team")["PlayerAge"].agg(["mean", "min", "std", "max"])


# We can see that the home team is marginally older than the away team with a max age of 40. most likely this will be the QB. 

# In[16]:


first_play[first_play["Position"] == "QB"]["PlayerAge"]


# Yep. It was the QB, not many positions can have people of that age. Only other option is maybe a kicker. 

# # Individual Player Attributes
# Since we are only looking at running plays it is probably useful to look at the running backs stats in more detail. My intention here is to create a vector that sums up the skill of the running back. 

# In[17]:


nflidvectors = train.groupby("NflIdRusher").mean()[["X", "Y", "S", "A", "Dis", "Orientation", "Dir", "Down", "Distance", "Quarter", "YardLine"]]


# In[18]:


nflidvectors


# These vectors represent the average attributes of the runner. We can see where the team typically gives them the ball, how fast they are typically going and how much they accelerate, what orientation they face, what downs they are typically given the ball, the distance and quarter they get the ball and which yardlines they typically start on. These factors might tell you if someone typically grabs the ball very quickly at a lateral angle and they only get the ball when they are close to the goal line. This could be potentially significant. 

# In order to utilize these vectors we can map the nflidrusher to these lookup vectors

# In[19]:


first_play.merge(nflidvectors, left_on = "NflIdRusher", right_index = True)


# If we would like to we could add in our yards gained and that would be like target encoding, but I don't think that's safe in this case when we have relatively limited play data. 

# We can see that in this format things are a bit redundant because we've broadcast the nflidvectors across all of the rows. This is fine for now. It wont be an issue once we separate out the player specific columns and the full team columns

# # Mathcups
# We can look at matchups in a few different ways. One that might be interesting is seeing how closely the players are matched in terms of X, y and then find the closest for each defending player and then compare their weight, height, speed and acceleration

# In[20]:


player_df[comp_features]


# In[21]:


from tqdm import tqdm
import numpy as np
closest_defender = []
comp_features = ["X", "Y", "CorrectedHeight", "PlayerWeight", "S", "A"]
for i in range(22):
    player_df = first_play.iloc[i]
    dist_df = (first_play[['X', 'Y']] - np.array(player_df[["X", "Y"]].values)).pow(2).sum(1).pow(0.5)
    first_play["player_dist"] = dist_df
    opponent_df = first_play[first_play["Team"] != player_df["Team"]]
    closest_df = opponent_df.loc[opponent_df["player_dist"].idxmin()]
    closest_defender.append(player_df[comp_features] - closest_df[comp_features])


# In[22]:


pd.concat(closest_defender, axis = 1).T


# Now for a given play we can see the stats of all of the closest players. How far off they are on the field on each axis, the height difference, weight difference and who is moving faster and who is accelerating faster. These could potentially be iluminating features

# To be continued...

# Loose brainstorming
# * Distance from rb to opposing player
# * Distance from rb to opposing player on defending side (ie. not behind the rb)
# * Offensive team player obstructing rb from forward progress (offensive player with x greater than rb within +/-1 of the rb's y)
# * Distance from rb to greater than a certain threshold of gap in the defensive line

# In[ ]:




