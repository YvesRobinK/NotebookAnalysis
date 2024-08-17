#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch


# In[2]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
days_abb = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
y_ticks = ['Low', 'Medium', 'High']

def create_perf_graph(perf):
    fig, ax = plt.subplots(figsize=(7, 3))

    sns.lineplot(x=days, y=perf, ax=ax, marker='o', markersize=5)

    y_label = ax.set_ylabel('Perfomance', x=0, y=0.86)
    
    ax.set_yticks([1,2,3])
    ax.set_yticklabels(y_ticks)
    
    ax.set_title('Students Perfomance Throughtout The Week', x=0.19, y=1.05)
    
    ax.set_ylim(0.8)

    sns.despine(top=True, right=True)
    
    return ax


# # Overview

# The purpose of this Kaggle notebook is to examine how the **day** and **time of day** when students play a game may impact their **final performance**., which is measured by their quiz score. To achieve this, we will use the **reverse engineering** method described in this [notebook](https://www.kaggle.com/code/pdnartreb/session-id-reverse-engineering) and consider partitioning the data by day of the week and time of day. Our primary hypothesis is that the **mean scores of these groups differ**. To test this, we will use an **ANOVA test**, followed by **multiple t-tests with Bonferroni correction**

# # What are the expecter results?

# Before diving into the analysis, I would like to share my opinion on how the data may appear, firstly for the day of the week partition. These assumptions will be illustrated in a graphs.

# #### Variant 1
# The game is designed for 3-5 grade students. We can assume that they still **may not be as used to studying** as high school students.

# In[3]:


ax = create_perf_graph([2,2,2,2,2,1,1])


# Therefore, it is possible that the graph will show a **medium amount** of effort and performance by students **on weekdays**, with an even more **relaxed performance on weekends**.

# #### Variant 2
# In this scenario, we can observe that the weeks begin with a **high level of performance**, which gradually **decreases** until it reaches its lowest point on Friday.

# In[4]:


ax = create_perf_graph([3,3,2,2,1,2,3])


# However, **performance improves** again over the weekend as the students take a break.

# #### Variant 3
# According to this assumption, the weeks start with a **high level of energy and enthusiasm**, but the **decrease in performance** is more rapid compared to the second assumption.

# In[5]:


ax = create_perf_graph([3,3,2,1,3,2,3])


# Interestingly, I assume that students are filled with joy once the weekdays are over, and therefore, they **do their best to complete any remaining tasks** by the end of the week. On Saturday, they take a **short break**, and then their **performance improves** again on Sunday.

# #### Day Time Assumption

# Regarding the time of day, I assume that perfomance follows the **natural biorhythm of humans**. Typically, our productivity level is **highest** in the morning, followed by a **dip** after lunch, a **slight peak** in the late afternoon, and a **decline** in efficiency towards the evening. The graph reflects this concept.
# 
# <img  src='https://s3.amazonaws.com/wordpress-production/wp-content/uploads/2017/11/time-of-day.jpg' width=50%>

# # Data Loading

# In[6]:


df = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train.csv', 
                 usecols=['session_id'],
                 dtype = {'session_id':'object'})
df.head()


# In[7]:


labels = pd.read_csv('/kaggle/input/predict-student-performance-from-game-play/train_labels.csv')

labels[['session_id', 'question']] = labels['session_id'].str.split('_', expand=True)
labels = labels.pivot(columns='question', index='session_id', values='correct')

scores = labels.sum(axis=1)
scores = scores.rename('score')

scores.head()


# # Data Extraction

# In[8]:


sessions = pd.DataFrame(df['session_id'].unique(), columns=['session_id'])
sessions.head()


# In[9]:


sessions['year'] = sessions['session_id'].str.slice(start=0, stop=2).astype(np.int8)
sessions['month'] = sessions['session_id'].str.slice(start=2, stop=4).astype(np.int8)

sessions['day'] = sessions['session_id'].str.slice(start=4, stop=6).astype(np.int8)

sessions['hour'] = sessions["hour"] = sessions["session_id"].str.slice(start=6, stop=8).astype(np.uint8)

sessions = sessions.set_index('session_id')
sessions.head()


# In[10]:


# Moving Sunday from the beggining of the week to its end
t = sessions['day']
sessions['day'] = sessions['day'].map({0:7}).fillna(t)
sessions.head()


# In[11]:


date_score = pd.concat([sessions, scores], axis=1)
date_score.head()


# ## General Trends EDA

# In this section we explore **general tendencies** for game for present 3 years: 2020, 2021, 2022. In this interval we will consider **game activity** and **mean score** for each month.

# In[12]:


date_score.year.unique()


# ## Game Activity

# In[13]:


game_activity = date_score.groupby(['year', 'month']).size()
game_activity.head()


# In[14]:


fig, ax = plt.subplots(figsize=(13, 5))

sns.lineplot(x=range(9), y=game_activity[:9], ax=ax, color='orange')
sns.lineplot(x=range(8, len(game_activity)), y=game_activity[8:], ax=ax, color='grey')

y_label = ax.set_ylabel('Number Of Sessions', y=0.832)

ax.set_xlim(-0.5)
ax.set_ylim(0, 2250)

ax.set_xticks([-0.5,2.5,5.5,8.5,11.5,14.5,17.5,20.5,23.5,26.5])
ax.set_xticklabels([])

ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22, 25], minor=True)
plt.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
ax.set_xticklabels(['2020 Q4','2021 Q1','2021 Q2','2021 Q3', '2021 Q4','2022 Q1','2022 Q2','2022 Q3', '2022 Q4'],
                   minor=True)

ax.set_title('Total Game Activity In Three Years', x=0.122, y=1.085, size=16)
ax.text(x=-2.25, y=2350, s="The assumed", color='grey')
ax.text(x=0.35, y=2350, s='start of data collection', color='orange')
ax.text(x=4.75, y=2350, s='in 2020 may reveal the popularity in beggining', color='grey')

sns.despine(top=True, right=True)


# Since data collection may not have started precisely at the beginning of first month or ended at the end of last month, the values at the ***tips*** of each month are ***extremely low***.

# ## Quiz Perfomance

# In[15]:


date_score['year_month'] = date_score['year'].astype('str') + '_' + date_score['month'].astype('str')
date_score.head()


# In[16]:


fig, ax = plt.subplots(figsize=(13, 5))
                   
sns.pointplot(data=date_score, x='year_month', y='score', 
              ax=ax, 
              errorbar='se',
              color='grey')

y_label = ax.set_ylabel('Mean Quiz Score', y=0.86)

ax.set_xlim(-0.5)
ax.set_ylim(11, 14)

ax.set_xticks([-0.5, 2.5,5.5,8.5,11.5,14.5,17.5,20.5,23.5,26.5])
ax.set_xticklabels([])

ax.set_xticks([1, 4, 7, 10, 13, 16, 19, 22,25], minor=True)
plt.tick_params(which='minor', bottom=False, top=False, left=False, right=False)
ax.set_xticklabels(['2020 Q4','2021 Q1','2021 Q2','2021 Q3', '2021 Q4','2022 Q1','2022 Q2','2022 Q3', '2022 Q4'],
                   minor=True)

ax.set_xlabel('')

ax.set_title('Quiz Perfomance Activity In Three Years', x=0.16, y=1.085, size=16)
ax.text(x=-2.08, y=14.15, color='grey', s='Over the course of three years, there was only a ')
ax.text(x=7.125, y=14.15, s='slight increase by 0.75 point', color='orange')
ax.text(x=12.6, y=14.15, s='in the quiz score', color='grey')

sns.lineplot(x=[-1,26], y=[11.97,12.76], color='orange', linewidth=3)

sns.despine(top=True, right=True)


# As previously noted, the little number of samples at the **tips** of the data results in a **high mean error**.

# ## Days Of The Week

# In this section, we examine our main hypothesis, which suggests that **performance varies** *throughout the week and can be attributed to **three assumptions** outlined earlier:
# 
# 1. Distribution of **activity** among different groups
# 2. Verification of ANOVA **requirements**
# 3. Implementation of **ANOVA test**
# 4. Conducting multiple **t-tests**
# 5. **Plotting quiz performance** for each group with standard error.

# ## Activity

# In[17]:


game_activity = date_score.groupby(['day']).size()
game_activity.head()


# In[18]:


fig, ax = plt.subplots(figsize=(7, 5))

sns.lineplot(x=game_activity.index[4:], y=game_activity.iloc[4:],
             ax=ax, color='grey', marker='o', markerfacecolor='c')
sns.lineplot(x=game_activity.index[:5], y=game_activity.iloc[:5],
             ax=ax, color='grey', marker='o', markerfacecolor='orange')

ax.set_xlabel('')

ax.set_ylabel('Number Of Sessions', y=0.84)
ax.set_xticks(range(1,8))
ax.set_xticklabels(days)

ax.set_ylim(1000,4500)

ax.set_title('Game Activity Throughtout The Week', x=0.215, y=1.085, size=14)
ax.text(-1.09+1, 4675, 'Work hard on', color='grey')
ax.text(0.1+1, 4675, 'weekday', color='orange')
ax.text(0.88+1, 4675, '- rest on', color='grey')
ax.text(1.65+1, 4675, 'weekends', color='c')

sns.despine(top=True, right=True)


# If we measure students' performance based on the **number of sessions** throughout the week, it appears similar to our **second model**, but with a slightly **higher performance** observed in the middle of the week.

# ## Hypothesis Testing: Day Of Week

# ### ANOVA Reqierements

# 1. **Homogeneity of variances** - the variances are equal across all groups.

# In[19]:


ax = sns.boxplot(y=date_score.day, x=date_score.score, orient='h', color='#ADD8E6')

ax.set_title('Variance Homogeneity', x=0.01)
ax.set_ylabel('')
# ax.set_xticks(range(1,8))
ax.set_yticklabels(days)
sns.despine(top=True, right=True)


# As observed, the variances are relatively similar. What is particularly interesing is the pattern observed in the medians of the scores, indicating moderate performance during weekdays and higher performance during weekends. It does not align with any of our initial assumptions.

# 2. **Independence of observations** - each subject must belong to only one group

# 3. **Categorical independent** variable - day of the week, and a **continuous dependent** variable - score. It is important to note that while the score is a discrete numerical variable, we can treat it as continuous for the purpose of this analysis.

# 4. **Equal group sisez** - group sizes within weekdays (Monday to Friday) and weekends (Saturday and Sunday) are approximately equal. However, there are differences between the group sizes of weekdays and weekends in 4 times!

# Now that we have checked the relevant conditions, although not all of them are fully met, we can proceed with running the **ANOVA test**.

# In[20]:


from scipy.stats import f_oneway

groups = []
for day in range(1,8):
    groups.append(date_score.loc[date_score['day'] == day, 'score'].values)
f_oneway(*groups)


# # Pairwise comparisons

# The results of the ANOVA test indicate **statistical significance**! However, it only confirms the presence of differences between **at least two groups** and does not specify the exact nature of those differences. 
# 
# To determine the specific group differences, we will conduct **multiple t-tests with Bonferroni correction**. This correction is necessary as we will be conducting a total of **21 tests,** and using the standard threshold may lead to obtaining significant results by chance where there are no true differences. **The Bonferroni correction** is a **simple** yet **conservative** approach, where the threshold p-value is divided by the number of tests conducted.

# In[21]:


# Performing multiple t-tests
from scipy.stats import ttest_ind

p_value = 0.05
comparisons = 7*6/2
p_value = p_value/comparisons

p_matrix = pd.DataFrame(index=days_abb, columns=days_abb, dtype=float)
f_matrix = pd.DataFrame(index=days_abb, columns=days_abb, dtype=float)

for i in range(len(groups)):
    for j in range(i+1, len(groups)):
        results = ttest_ind(groups[i], groups[j])
        p_matrix.iloc[i,j] = results[1]
        f_matrix.iloc[i,j] = results[0]
            
p_matrix = p_matrix.fillna(0)
p_matrix = p_matrix.transpose() + p_matrix
np.fill_diagonal(p_matrix.values, 1)
signif_matrix = p_matrix < p_value


# In[22]:


cyan = (0, 100/256, 100/256,1)
grey = (0.9,0.9,0.9,1)
cmap = ListedColormap([grey, cyan])

ax = sns.heatmap(signif_matrix, cbar=False, cmap=cmap)

plt.yticks(rotation=0)
ax.tick_params(left=False, bottom=False)

ax.set_title('Significant Difference', color=cyan, x=0.11)


# Significant differences in quiz scores were found between certain days of the week. Tuesday differed from Monday, and Friday differed from Wednesday. Additionally, weekends differed from weekdays, but there was no significant difference between Saturday and Sunday.

# # Plotting Relationship

# Let's now explore the mean quiz scores graph for more clarity.

# In[23]:


fig, ax = plt.subplots(figsize=(7, 5))

sns.pointplot(data=date_score, x='day', y='score', ax=ax, errorbar='se')

ax.set_xlabel('')
ax.set_xticks(range(7))
ax.set_xticklabels(days_abb)

ax.set_title('Quiz Perfomance Throughout the day', x=0.179, y=1.05)

ax.set_ylim(12.4,13.4)
ax.set_ylabel('score',y=0.967)

sns.despine(right=True, top=True)


# When measuring performance based on scores, the observed pattern appears to be a combination of the **first and third variants**. Both variants suggest a **decline** in performance throughout the week and an **increase** in performance during weekends or holidays. However, it should be noted that the **mean error** for weekdays is higher due to **smaller sample sizes** (1,000 versus 4,000 for weekdays).

# As a final note, we could consider **PROJECTING** the data specifically for **weekdays** and **weekends** for further analysis of performance patterns and feature engineering.

# # Day Time

# In this section, we will examine **performance** throughout the day. This chapter will be much shorter, as the ANOVA test will be left as an **exercise for YOU** to perform!

# # Activity

# In[24]:


game_activity = date_score.groupby('hour').size()
game_activity.head()


# In[25]:


fig, ax = plt.subplots(figsize=(12,4))

sns.lineplot(x=game_activity.index, y=game_activity, ax=ax, color='grey')

ax.set_ylim(0,3000)
ax.set_ylabel('Number Of Sessions',x=0,y=0.79)

ax.set_xticks(range(24))
ax.set_xlim(0,23)

ax.set_title('Game Activity Throughout the Day', x=0.112, y=1.1, size=14)
ax.text(x=-1.59, y=3170, s="Players' activity aligns with the", color='grey')
ax.text(x=3.78, y=3170, s=" human biological rhythm", color='orange')

sns.despine(right=True, top=True)


# ## Data Projection

# There are a few issues with performing ANOVA on these 24 groups:
# 
# 1. The **large number of groups** results in a significant number of comparisons (24*23/2=276)
# 2. The **night data** is particularly **noisy** due to the limited sample size of night players 🦉
# 
# I suggest categorizing our data into broader groups based on time periods: Night (0-5), Morning (6-11), Day (12-17), and Evening (18-23). This division can be visually represented on the graph.

# In[26]:


fig, ax = plt.subplots(figsize=(12,4))

sns.pointplot(data=date_score, x='hour', y='score', 
              errorbar='se',
              color='grey')

ax.set_title('Quiz Perfomance throughout the day', size=14, x=0.128, y=1.1)
ax.text(x=-2.05, y=15.2,s='For simplicity of our analysis we do categorical', color='grey')
ax.text(x=6.52, y=15.2,s='projection', color='black')
ax.set_ylabel('score', y=0.962)
ax.set_ylim(11.5,15)

plt.axvline(x = 5.5, color = 'black', linestyle='--')
plt.axvline(x = 11.5, color = 'black', linestyle='--')
plt.axvline(x = 17.5, color = 'black', linestyle='--')

ax.text(x=0.5,y=14.5,s='Night')
ax.text(x=6.5,y=14.5,s='Morning')
ax.text(x=12.5,y=14.5,s='Day')
ax.text(18.5,y=14.5,s='Evening')

sns.despine(right=True, top=True)


# Upon closer examination, it becomes clear that the observed pattern does **not align** with our **initial assumption** regarding the impact of time of day. Additionally, the data from players during the **night** period appears to be quite **messy and inconsistent**.

# In[27]:


day_time = {}

for i in range(24):
    if i < 6:
        day_time[i] = 0
    elif i < 12:
        day_time[i] = 1
    elif i < 18:
        day_time[i] = 2
    else:
        day_time[i] = 3

date_score['day_time'] = date_score['hour'].map(day_time)
date_score.head()


# To speed up the process of writing this notebook and provide an **opportunity for YOU to practice** the steps involved in applying the **ANOVA test**, I will proceed with directly plotting the data without performing any statistical tests. This way, you can observe the visual representation and undertake the mutlitple comparisons analysis yourself, consolidating your knowledge through hands-on practice.

# In[28]:


performance = date_score.groupby('day_time')['score'].mean()
performance


# In[29]:


fig, ax = plt.subplots(figsize=(9,4))

sns.pointplot(data=date_score, x='day_time', y='score', 
              errorbar='se',
              color='grey')

ax.set_title('Quiz Perfomance throughout the day', size=14, x=0.172, y=1.1)
ax.text(x=-0.83,y=13.45,s='Statistically significant', color='grey')
ax.text(x=0.1,y=13.45,s='difference', color='orange')
ax.text(x=0.53,y=13.45,s='observed in 2 pairs', color='grey')

ax.set_ylabel('score', y=0.96)
ax.set_ylim(12.5,13.4)

ax.set_xticklabels(['Night', 'Morning', 'Day', 'Evening'])
ax.set_xlabel('')

arrow = ConnectionPatch(xyA=(1, performance[1]),
                        xyB=(3, performance[3]),
                        coordsA="data",
                        coordsB="data",
                        arrowstyle='<|-|>,head_width=0.4,head_length=0.8',
                        linewidth=1.5,
                        edgecolor='orange',
                        facecolor='orange',
                        connectionstyle='arc3,rad=.4')
ax.add_artist(arrow)
                        
arrow = ConnectionPatch(xyA=(2, performance[2]),
                        xyB=(3, performance[3]),
                        coordsA="data",
                        coordsB="data",
                        arrowstyle='<|-|>,head_width=0.4,head_length=0.8',
                        linewidth=1.5,
                        edgecolor='orange',
                        facecolor='orange',
                        connectionstyle='arc3,rad=.3')
ax.add_artist(arrow)

sns.despine(right=True, top=True)


# # Wind up 👨‍💻

# ### So, what do we have in the end?
# 
# - We generated **2 features** - day of the week and hour of the day.
# - Applied **ANOVA test** for multiple groups comparison to identify difference in quiz scores among all groups created by our 2 independent variable. It required us to check all the **requirements for ANOVA test**.
# - Performed **multiple t-tests** to further investigate the observed differences. To reduce the risk of false positive results, we employed the **Bonferroni correction**.
# - We discussed how number of groups may be lowered using **categorical projection** to broader groups: weekdays and weekends; morning, afternoon, day and evening.
# - Tried to draw pretty graphs! 😀
# 
# I hope you enjoyed this long journey and find something new for you. Thank you for reading my notebook! If you have any questions, suggestions or ideas on this topic - leave them in the comments.
