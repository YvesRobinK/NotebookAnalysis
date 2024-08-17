#!/usr/bin/env python
# coding: utf-8

# <div style="color:#7b6b59;margin:0;font-size:32px;font-family:Georgia;text-align:center;display:fill;border-radius:5px;overflow:hidden;font-weight:600;">From A to Z: EDA, Feature Engineering & Modeling</div>
# 
# <div style="text-align:center">
#     <img width="1065" alt="image" src="https://github.com/eraikakou/LLMs-News/assets/28102493/dbf69088-a2c2-4db3-b545-d2b5ad0ea457">
# </div>
# <div style="text-align:center">
#     <a href="https://unsplash.com/photos/silver-iphone-6-on-white-paper-mHjvJqvj1XE">Photo from Unsplash</a>
# </div>
# 
# # <div style="padding:20px;color:white;margin:0;font-size:30px;font-family:Georgia;text-align:left;display:fill;border-radius:5px;background-color:#7b6b59;overflow:hidden">1. Introduction</div>
# 
# Welcome to a practical journey through the world of data science as it meets the skill of writing. This Kaggle notebook is more than just a contest entry; it's a resource for learning and inspiration. Here, you'll find a thorough exploration of data, starting with Exploratory Data Analysis (EDA) and moving through the steps of building and refining a predictive model. My goal is to understand how the way we write can affect the final quality of an essay.
# 
# By analyzing keystroke logs with careful attention to detail, we aim to uncover the subtle factors that make writing great. This notebook is designed to be a clear and easy-to-follow guide that will inspire and equip you with the know-how to conduct comprehensive analyses and develop predictive models confidently.
# 
# Join me as I dive into the keystrokes that tell the story of writing quality. Let’s start decoding each keystroke's story and transforming raw data into meaningful insights.

# ## <span style="color: #7b6b59;">1.1 Import Python Libraries</span>

# In[1]:


import pandas as pd
import numpy as np

from functools import reduce
from typing import Tuple, List
from collections import Counter


import seaborn as sns
import matplotlib.colors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error


# ## <span style="color: #7b6b59;">1.2 Load the Dataset</span>

# In[2]:


train_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
train_scores = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
test_logs = pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")


# In[3]:


train_logs.head(10)


# # <div style="padding:20px;color:white;margin:0;font-size:24px;text-align:left;display:fill;border-radius:5px;background-color:#7b6b59;overflow:hidden">2. Exploratory Data Analysis</div>

# ## <span style="color: #7b6b59;">Analyzing Event Counts in Essays</span>

# In[4]:


essay_event_counts = train_logs.groupby("id")["event_id"].count().reset_index(name="event_count")

# Using Plotly for a violin plot
fig = px.violin(essay_event_counts, y='event_count', box=True, points="all", title='Event Count per Essay', color_discrete_sequence=["#c07156"])

# Update layout to customize
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=f"<span style='font-size:26px; font-family:Times New Roman'>Number of Events per Essay</span>",
    font = dict(color = '#7b6b59'),
    xaxis_title='', 
    yaxis_title='Number of Events'
)

fig.show()


# ## <span style="color: #7b6b59;">Activity Types Analysis</span>

# In[5]:


#train_logs["activity"].value_counts(dropna=False)
print(f"Total distinct activity types: {train_logs['activity'].nunique()}")
train_logs["activity_type"] = train_logs["activity"].apply(lambda x: "Move" if x.startswith('Move') else x)


# In[6]:


# Let's assume 'activity_counts' is a dictionary with your activity types as keys and their counts as values.
# Example: activity_counts = {'Walking': 150, 'Running': 100, 'Swimming': 50, ...}
activity_counts = train_logs["activity_type"].value_counts(dropna=False).to_dict()

# Calculate total counts and percentages
total_counts = sum(activity_counts.values())
activity_percentages = {activity: f"{(count / total_counts) * 100:.2f}%" for activity, count in activity_counts.items()}

# Sort activities by counts
sorted_activities = sorted(activity_counts.items(), key=lambda item: item[1], reverse=True)

# Separate activities and their counts into two lists for plotting
activities, counts = zip(*sorted_activities)

# Create horizontal bar plot
fig = go.Figure(go.Bar(
    x=counts,
    y=activities,
    text=[f"{activity_percentages[activity]}" for activity, count in sorted_activities],
    textposition='auto',
    orientation='h',
    marker_color=["#E6b6a4"]*len(activity_counts)
))


# Update layout to customize
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=f"<span style='font-size:26px; font-family:Times New Roman'>Most Common Activities by Count and Percentage</span>",
    font=dict(color='#7b6b59'),
    xaxis_title='Count',
    yaxis_title='Activity',
    bargap=0.2,  # Gap between bars can be adjusted as needed
    yaxis={'categoryorder':'total ascending'},  # This will order the bars by count
)

# Show the plot
fig.show()


# ## <span style="color: #7b6b59;">Essay Score Distribution Analysis</span>
# 
# - The most common scores are 3.5 and 4, with approximately 19.67% and 20.28% of the essays receiving these scores, respectively.
# - Scores 0.5 and 6 are the least frequent, with only 0.20% and 1.50% of essays receiving these scores, respectively. This indicates that very few essays are rated as poor or excellent.
# 
# In conclusion, the score distribution points towards a central tendency in grading, with most essays falling into the mid-range of the scoring scale and only a small fraction receiving the highest or lowest possible scores.

# In[7]:


# Count the frequency of each score
score_counts = Counter(train_scores["score"])

# Sorting the scores and counts so they are plotted in order
sorted_scores = sorted(score_counts.items())

# Separate the scores and their counts into two lists for plotting
x, y = zip(*sorted_scores)
total = sum(y)

# Calculate percentages after sorting
percentages = [f'{(count/total)*100:.2f}%' for count in y]

# Create bar plot with the sorted scores and their corresponding percentages
fig = go.Figure([go.Bar(x=x, y=y, text=percentages, textposition='outside', marker_color=["#c07156"]*len(x))])

# Update layout to customize
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=f"<span style='font-size:26px; font-family:Times New Roman'>Essays Score Distribution</span>",
    font=dict(color='#7b6b59'),
    xaxis_title='Scores',
    yaxis_title='Frequency'
)

# Show the plot
fig.show()


# ## <span style="color: #7b6b59;">Assessment of Writing Duration per Essay</span>
# 
# 
# To find out how long it takes to complete an essay in the training dataset, we will typically look for the timestamp of the first and last actions recorded for each essay. However, since the data fields provided do not include direct timestamps but rather **`down_time`** and **`up_time`** for events, we'll need to infer the start and end times of each essay based on these.
# 
# Here's a step-by-step approach on how to calculate the duration for each essay:
# 
# 1. **Identify Start and End Events:** The `event_id` starts at 0 for each essay and increments with each action, the first and last events for each essay can serve as proxies for the start and end times.
# 1. **Calculate Duration for Each Essay:**
#     - Group the data by the `id` to segregate events for each essay.
#     - For each group (essay), find the **minimum** `down_time` to estimate the **start**.
#     - Then, find the **maximum** `up_time` to estimate the **end**.
#     - The duration for the essay can be estimated by subtracting the start `down_time` from the end `up_time`.
# 1. **Aggregate Data:** Once we have the duration for each essay, we can calculate statistics such as the average, median, and range of essay completion times

# In[8]:


# Group by essay id and calculate the start and end times
grouped_logs = train_logs.groupby("id").agg(start_time=("down_time", "min"), end_time=("up_time", "max"))

# Calculate the duration for each essay
grouped_logs["duration_ms"] = grouped_logs["end_time"] - grouped_logs["start_time"]

# Convert duration from milliseconds to a more readable format if desired, e.g., minutes
grouped_logs["duration_min"] = grouped_logs["duration_ms"] / 60000

# You can calculate summary statistics for these durations
duration_stats = grouped_logs["duration_min"].describe()
grouped_logs.head(10)


# In[9]:


# Assuming 'grouped_logs' is your DataFrame and it has a column named 'duration_min'.

# Create a subplot with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2, subplot_titles=('Boxplot of Essay Completion Times', 'Density Plot of Essay Completion Times'))

# Boxplot
box_trace = go.Box(y=grouped_logs['duration_min'], name='',
                   marker_color='#e5b01c', line_color='#e5b01c')

# Density Plot
density_trace = go.Histogram(x=grouped_logs['duration_min'], name='', 
                             histnorm='probability density', 
                             marker_color='#e5b01c')

# Adding box trace to the first column
fig.add_trace(box_trace, row=1, col=1)

# Adding density trace to the second column
fig.add_trace(density_trace, row=1, col=2)

# Update layout to customize
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=f"<span style='font-size:26px; font-family:Times New Roman'>Essay Completion Time Analysis</span>",
    font = dict(color = '#7b6b59'),
)

# Customize x-axis and y-axis of the density plot
fig.update_xaxes(title_text="Duration (minutes)", row=1, col=2)

# Customize y-axis of the boxplot
fig.update_yaxes(title_text="Duration (minutes)", row=1, col=1)


# Remove gridlines
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

# Show the figure
fig.show()


# **Insights:**
# 
# 1. **Boxplot Analysis**:
# 
#     - The boxplot shows a median completion time that appears to be around 20 minutes.
#     - The interquartile range (IQR), which represents the middle 50% of the data, is quite narrow, indicating that half of the essay completion times are clustered closely around the median.
#     - The whiskers (lines extending from the box) suggest that the bulk of the data lies between roughly 15 and 25 minutes.
#     - There are several points above the upper whisker that are considered outliers, indicating that there are some essays which took significantly longer to complete than the majority.
#     - The range of completion times, excluding outliers, appears to be from slightly less than 15 minutes to a little over 25 minutes.
#     - The presence of outliers suggests that there could be factors causing certain essays to take much longer to complete.
# 
# 1. **Density Plot Analysis**:
# 
#     - The density plot shows the distribution of essay completion times in a continuous curve, providing a sense of the probability density of the times.
#     - It confirms the concentration of completion times around the 20-minute mark, with a sharp peak at this value indicating the mode of the dataset.
#     - The plot is right-skewed, meaning that there are a tail of values stretching towards the longer completion times.
#     - This skewness is also indicative of the outliers we observed in the boxplot, confirming that while most essays are completed in a shorter time, there are a few essays that take much longer.
#     - There is a very steep drop-off in density after the mode, reinforcing that times longer than the mode are less common.
# 
# Overall, from the text and the graphs we can conclude that most essays are completed within a fairly consistent, short time frame, with a median and mode around 20 minutes, but there are exceptions that take significantly longer, which are highlighted as outliers in the boxplot and contribute to the right skew in the density plot.
# 
# 
# 
# 
# 
# 

# ## <span style="color: #7b6b59;">Assessment of Essay Lengths</span>
# 
# If we want to determine the total number of words per essay after the last event, we would proceed as follows:
# 
# 1. Group the data by `id` (which represents each essay).
# 1. Within each group, find the entry with the highest `event_id` (as it's ordered chronologically, the highest event_id would represent the last event).
# 1. Extract the `word_count` from this entry, as it represents the word count after the last event for that essay.
# 

# In[10]:


# Sort the DataFrame based on the essay 'id' and 'event_id'
df_sorted = train_logs.sort_values(by=['id', 'event_id'])

# Group by 'id' and get the last entry for each essay
df_last_event = df_sorted.groupby('id').last().reset_index()

# The 'word_count' column of this DataFrame now holds the word count after the last event for each essay
final_word_counts = df_last_event['word_count']

# If you need to calculate the average word count across all essays:
average_word_count = final_word_counts.mean()

print(f"The average word count per essay after the last event is: {average_word_count}")


# In[11]:


# Create the density plot
fig = ff.create_distplot([df_last_event['word_count']], ['Word Count'], bin_size=30, colors=['#e5b01c'])

# Update layout to customize
fig.update_layout(
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    title=f"<span style='font-size:26px; font-family:Times New Roman'>Density Plot of Essay Word Counts</span>",
    font = dict(color = '#7b6b59'),
    xaxis=dict(title='Word Count'),
    yaxis=dict(title='Density')
)

# Show the figure
fig.show()


# **Insights:** 
# 
# 1. **Central Tendency:** The peak of the density curve indicates the most common range of word counts for the essays. It appears that the majority of essays have a word count in the lower range of the x-axis, which seems to be around the 200–400 word mark, indicating that most of the essays are relatively short.
# 
# 1. **Skewness:** The plot is right-skewed (or positively skewed), as indicated by the long tail extending to the right. This suggests that while most essays are shorter, there is a smaller number of essays that are much longer.
# 
# 1. **Variability:** There is some variability in the lengths of essays, but the declining density as word count increases shows that essays of greater length are less common.
# 
# 1. **Outliers/Extremes:** The 'spikes' or individual vertical lines towards the higher end of the word count axis may indicate outliers or essays that are significantly longer than the rest. This could represent a few essays that are exceptionally lengthy.
# 
# 1. **Distribution Shape:** The bulk of the data is concentrated in a specific region, and the distribution is unimodal (one peak), indicating that there is one predominant essay length category.
# 
# 1. **Range:** The range of word counts spans from 0 up to what appears to be around 1400 words. This wide range suggests that there is considerable diversity in essay lengths within the dataset.
# 
# 1. **Absence of Bimodality:** There does not seem to be a significant secondary peak, which would suggest another common word count range. Hence, there is likely a single common essay length rather than multiple common lengths.
# 
# In summary, the plot suggests that while there is some variation in essay lengths, there is a clear tendency toward shorter essays in the dataset with fewer essays having a higher word count. Analyzing these patterns can be important for understanding the dataset's characteristics, and could influence decisions about data preprocessing or analysis, such as setting word count limits or handling outliers in further analyses.
# 
# 
# 
# 
# 
# 

# ## <span style="color: #7b6b59;">Pause Analysis in Detail</span>
# 
# ### <span style="color: #7b6b59;">Definition of Pauses</span>
# 
# Pauses are generally defined as inter-keystroke intervals (IKI) above a certain threshold (e.g., 2000 milliseconds). The IKI refers to the gap time between two consecutive key presses typically expressed in milliseconds. To illustrate, suppose a writer types a character "A" at time 1 and then a character "B" at time 2. One can obtain the IKI between the two characters simply using the formula: IKI = Time2 - Time1. Global measures of pausing are usually associated with the duration and frequency of pauses calculated from different dimensions. Below are some typical pause measures.
# 
# - number of pauses (in total or per minute)
# - proportion of pause time (as a % of total writing time)
# - pause length (usually the mean duration of all pauses in text production)
# - pause lengths or frequencies within words, between words, between sentences, between paragraphs, etc.
# 
# ### <span style="color: #7b6b59;">Pause Features</span>
# 
# Deriving features from pauses in typing:
# * Number of pauses, proportion of pause time, mean pause length, etc.
# 
# 
# 1. **Step 1 - Calculating Inter-Keystroke Interval (IKI):** We need to calculate the IKI for each event. IKI is the time difference between consecutive keystrokes. Since our data includes timestamps for key down events (down_time), we can use these to calculate IKIs. When we call `diff()` on the `down_time` column, it computes the difference between each row and the row before it. This includes rows that are the first occurrence of a new essay ID, which will have a large `IKI` since they're subtracted from the last `down_time` of the previous essay. To address this, we need to ensure that the first row of each essay ID group is treated as having an `IKI` of zero.
# 
# 1. **Step 2 - Identifying Pauses:** Next, we need to define what constitutes a pause. A common approach is to consider any IKI greater than a certain threshold (e.g., 2000 milliseconds) as a pause.
# 
# 1. **Step 3 - Feature Engineering:** Now, we can create various features based on these pauses.
#     - **Number of Pauses:** Count how many times is_pause is True for each essay.
#     - **Proportion of Pause Time:** Calculate the total pause time and divide it by the total writing time for each essay.
#     - **Mean Pause Length:** Find the average length of pauses for each essay.
# 
# ### <span style="color: #7b6b59;">Time Series Data Visualization</span>
# 
# A time series is a series of data points indexed in time order, often consisting of sequences of data points recorded at successive times, spaced at uniform time intervals.
# 
# Plotting this graph for both a high-scoring and a low-scoring essay could reveal differences in writing behavior that might correlate with essay quality. For example, consistent writing with fewer long pauses might indicate a fluid thought process and organization in a high-scoring essay, whereas erratic intervals and longer pauses in a low-scoring essay could suggest difficulties in articulating thoughts or less effective writing strategies. Such comparisons could provide valuable insights for educators and researchers into the writing habits that contribute to a successful essay.
# 

# In[12]:


train_logs = train_logs.sort_values(by=['id', 'event_id'])

# Calculate IKI with a groupby and transform, which will reset the diff() calculation for each group
train_logs["IKI"] = train_logs.groupby('id')['down_time'].diff().fillna(0)

# Define a pause threshold (in milliseconds)
pause_threshold = 2000  # This can be adjusted based on your needs

# Identify pauses
train_logs["is_pause"] = train_logs["IKI"] > pause_threshold

# Group data by essay ID
grouped_data = train_logs.groupby("id")

# Calculate features
pause_features = grouped_data.apply(lambda x: pd.Series({
    "number_of_pauses": x["is_pause"].sum(),
    "total_pause_time": x[x["is_pause"]]['IKI'].sum(),
    "mean_pause_length": x[x["is_pause"]]['IKI'].mean()
}))

# Calculate total writing time for each essay
pause_features["total_writing_time"] = grouped_data["action_time"].sum()

# Calculate the proportion of pause time
pause_features["proportion_of_pause_time"] = pause_features["total_pause_time"] / pause_features["total_writing_time"]

# Fill NaN values with appropriate values for mean_pause_length
pause_features['mean_pause_length'] = pause_features['mean_pause_length'].fillna(0)

# Reset index to flatten the DataFrame after groupby apply
pause_features.reset_index(inplace=True)


# In[13]:


def plot_event_intervals(sample_data, essay_id, score):
    """Plots the intervals between events over time for a specified essay.
    
    Args:
        sample_data: A pandas DataFrame containing the event data.
        essay_id: A string representing the ID of the essay to plot.
        score: The score of the essay to include in the plot title.
    
    Returns:
        A Plotly graph object.
    """
    # Sort the sample data by event_id
    sample_data.sort_values('event_id', inplace=True)

    # Calculate the intervals (IKI)
    sample_data['IKI'] = sample_data['IKI'] / 1000  # Convert milliseconds to seconds

    # Convert 'down_time' to a datetime format for plotting
    sample_data['down_time'] = pd.to_datetime(sample_data['down_time'], unit='ms')

    # Create the plot using Plotly
    fig = px.line(sample_data, x='down_time', y='IKI', title=f"Time between events for Essay ID: {essay_id} with score {score}")

    # Format the x-axis to show time nicely
    fig.update_xaxes(
        tickformat='%H:%M:%S',
        dtick=60000  # 60-second interval; adjust as needed
    )
    fig.update_traces(line_color='#c07156')  # You can use any valid CSS color.

    # Label your axes and update the layout
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis_title='Time',
        yaxis_title='Interval between events (seconds)',
        xaxis_tickangle=-45,
        font=dict(color='#7b6b59'),
        title=f"<span style='font-size:26px; font-family:Times New Roman'>Time between events for Essay ID: {essay_id} with score {score}</span>",
    )

    # Show the plot
    fig.show()


# In[14]:


import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_essay_activities(dataframe, essay_id, title):
    """
    Creates a Plotly subplot visualization for different activity types over time for a given essay.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame containing the essay activities data.
        essay_id (str): The ID of the essay to visualize.
        title (str): The title for the plot.
    
    Returns:
        None: Displays a Plotly figure.
    """
    # Define specific colors for each activity type
    activity_colors = {
        'Nonproduction': '#edc860',
        'Input': '#c07156',
        'Remove/Cut': '#beb29e',
        'Replace': '#b39a74',
        'Move': '#beb29e',
        'Paste': '#a43725'
    }

    # Select the data for a single essay
    single_essay_df = dataframe[dataframe['id'] == essay_id]
    # Convert 'down_time' to a datetime format for plotting
    single_essay_df['down_time'] = pd.to_datetime(single_essay_df['down_time'], unit='ms')

    # Create a subplot for each activity type
    activities = single_essay_df['activity_type'].unique()
    subplot_titles = [f"Activity: {activity}" for activity in activities]
    fig = make_subplots(rows=len(activities), cols=1, subplot_titles=subplot_titles)

    # Add a scatter plot for each activity type with its specific color
    for i, activity in enumerate(activities, start=1):
        activity_df = single_essay_df[single_essay_df['activity_type'] == activity]
        # Use the specific color for the activity if it's defined, else default to black
        color = activity_colors.get(activity, '#000000')
        fig.add_trace(
            go.Scatter(x=activity_df['down_time'], y=activity_df['word_count'],
                       mode='lines+markers', name=activity, line=dict(color=color)),
            row=i, col=1
        )
    
    # Format the x-axis to show time nicely
    fig.update_xaxes(
        tickformat='%H:%M:%S',
        dtick=60000  # 60-second interval; adjust as needed
    )
    # Update layout
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=300*len(activities),
        title_text=title,
        font=dict(color='#7b6b59'),
        xaxis_tickangle=-45
    )

    # Show the plot
    fig.show()


# In[15]:


# Filter the data for the chosen essay ID
#essay_id = "40b28508"
essay_id = "c3663a2d"
#essay_id = train_scores[train_scores["score"] == 0.5]["id"].sample(1).iloc[0]
sample_data = train_logs[
    train_logs["id"] == essay_id
]
plot_event_intervals(sample_data, "c3663a2d", "0.5")


# In[16]:


# Normalize 'down_time' to the start of each essay
train_logs['time_since_start'] = train_logs.groupby('id')['down_time'].transform(lambda x: x - x.min())

# Convert 'time_since_start' to seconds for plotting
train_logs['time_since_start'] = train_logs['time_since_start'] / 1000  # convert milliseconds to seconds


# In[17]:


essay_id = "c3663a2d"
plot_essay_activities(train_logs, essay_id, f"Word Count and Activities Over Time for Essay ID {essay_id} with score 0.5")


# In[18]:


# Filter the data for the chosen essay ID
#essay_id = "e01ec054"
#essay_id = "f9fd3268"
#print(essay_id)
essay_id = "1343a4d2"
#essay_id = train_scores[train_scores["score"] == 6.0]["id"].sample(1).iloc[0]
sample_data = train_logs[
    train_logs["id"] == essay_id
]
plot_event_intervals(sample_data, essay_id, "6.0")


# The visualization appears to be a line chart representing the intervals between events (such as keystrokes or mouse clicks) over time for a specific essay. Each point on the x-axis corresponds to a specific time, formatted as `hours:minutes:seconds`, and the y-axis shows the interval between successive events in seconds. From the provided visualization, which plots the intervals between events over time for a specific essay, several insights can be drawn:
# 
# **Insights:**
# 
# 1. **Writing Rhythm Analysis:** It helps in analyzing the writing rhythm of the person who composed the essay. Regular short intervals may indicate a steady flow of writing, while longer intervals could suggest pauses for thinking, researching, or interruptions. ***The majority of the data points are at the lower end of the y-axis, suggesting a generally consistent writing flow with frequent, short intervals that likely correspond to regular typing.***
# 
# 1. **Pause Patterns:** The spikes represent longer pauses, which could be significant. For example, if these longer pauses regularly occur before new paragraphs or sections, this might indicate the writer's planning or reflective moments.
#     - The peaks in the graph, where the intervals between events are longer, may indicate moments where the writer paused to think, plan, or review their work. These could correspond to cognitive processes such as reflection, problem-solving, or information processing.
#     - The highest peaks, especially if they are significantly higher than the average typing interval, could suggest that the writer was distracted, took a break, or was otherwise interrupted in their writing process.
# 
# 1. **Productivity Insight:** By examining the frequency and duration of pauses, one can infer the writer's productivity and engagement levels throughout the writing process.
# 
# 1. **Predictive Modeling:** Features derived from this visualization, such as the mean interval length or the frequency of long pauses, can be used to predict the essay's quality or the writer's proficiency.
# 
# 1. **Behavioral Analysis:** Understanding the timing and context of pauses can provide insights into the writer's typing behavior, potentially revealing habits or strategies in writing, like frequent reviewing or editing.
# 
# In essence, this plot serves as a detailed temporal analysis of the writing process for an individual essay, and can be a valuable tool in educational settings, user experience research, and writing tool development.

# In[19]:


essay_id = "1343a4d2"
plot_essay_activities(train_logs, essay_id, f"Word Count and Activities Over Time for Essay ID {essay_id} with score 6.0")


# In[20]:


# # Assuming 'train_logs.csv' has been previously loaded into a DataFrame named train_logs_df

# # For illustration, we'll create a cumulative count of each activity over time for one essay.
# # You'll need to ensure that train_logs_df is the name of your DataFrame containing the data.

# # First, we'll filter out the data for one essay.
# # Replace 'id_of_interest' with the actual essay id you want to analyze.
# id_of_interest = 'c3663a2d'
# essay_df = train_logs[train_logs['id'] == id_of_interest]

# # Now, we'll get the unique list of activities for the subplots.
# activities = essay_df['activity'].unique()
# subplot_titles = [f"Activity: {activity}" for activity in activities]

# # Initialize a subplot for each activity.
# fig = make_subplots(rows=len(activities), cols=1, subplot_titles=subplot_titles)

# # Convert 'down_time' to seconds since the start of the essay.
# essay_df['seconds_since_start'] = (essay_df['down_time'] - essay_df['down_time'].min()) / 1000

# # Create a cumulative count for each activity and plot.
# for i, activity in enumerate(activities, 1):
#     # Get the subset of the DataFrame for the activity
#     activity_df = essay_df[essay_df['activity'] == activity]
    
#     # Calculate the cumulative count
#     activity_df = activity_df.sort_values('seconds_since_start')
#     activity_df['cumulative_count'] = range(1, len(activity_df) + 1)
    
#     # Add the trace to the subplot.
#     fig.add_trace(
#         go.Scatter(
#             x=activity_df['seconds_since_start'],
#             y=activity_df['cumulative_count'],
#             mode='lines+markers',
#             name=activity
#         ),
#         row=i,
#         col=1
#     )

# # Update the layout.
# fig.update_layout(height=300*len(activities), title_text="Cumulative Activity Counts Over Time")

# # Show the figure.
# fig.show()


# 
# # <div style="padding:20px;color:white;margin:0;font-size:30px;font-family:Georgia;text-align:left;display:fill;border-radius:5px;background-color:#7b6b59;overflow:hidden">3. Feature Engineering</div>
# 
# ## <span style="color: #7b6b59;">3.1 Behavioral and Temporal Text Composition Metrics</span>
# 
# Since we have multiple keystroke logs for each essay and need to aggregate them for our machine learning model, our feature engineering approach should focus on summarizing the information across all logs per essay. Here are some aggregated feature suggestions:
# 
# 1. **Basic Statistical Features:**
#     - ✅ Calculate basic statistics for time-related fields (`action_time`) for each user/essay like `mean`, `median`, `std`, `min`, `max`, `skew`, etc. These features can provide insights into the user's typing behavior and speed.
#     
# 1. **Text Composition Features:**
#     - ✅ **Count-Based Features: Frequency of Different Actions: Count the occurrences of each type of action (e.g., Input, Remove/Cut, Paste, Replace, Move).** It shows how much editing, writing, and correcting happened.
#     - ✅ **Total number of activities for each essay:** which might capture how much effort was put into the essay.
#  
# 1. **Temporal (Time-Based) Features:**
#     - ✅ **Session Duration: Total time taken to write the essay** (difference between the first and last event time). Total Writing Time: Total time spent on writing the essay = Sum of all session durations per essay or The time difference between the first and last event.
#     - ✅ **Average Time per Event or Specific Activity Category**: Average time per event or specific activity category, which might indicate the thought process involved. For each essay and each activity category, calculate the average `action_time`. This involves grouping the data by `id` and `activity_type`, and then taking the `mean` of `action_time`.
# 
# 1. **Text-Based Features:**
#     - ✅ **Final word count:** The `word_count` from the last log entry of an essay, indicative of essay length.
#     
# 1. **Complex Behavioral Features:**
#     - ✅ **Error Correction Analysis:** Deletion to insertion ratio = Ratio of 'Remove/Cut' events to 'Input' events, which might indicate how often a participant rephrases or corrects their essay.
#     -  ✅ **Overall Average Typing Speed:** Compute the average typing speed across all logs. Typing speed is typically measured in words per minute (WPM).

# In[21]:


# Define a global list of all possible activity types
ALL_ACTIVITY_TYPES = ["Nonproduction", "Input", "Remove/Cut", "Paste", "Replace", "Move"]

def generate_behavioral_features(df):
    """Generates aggregated features from a log dataset of essay compositions.

    Args:
        df (pandas.DataFrame): A dataframe containing keystroke and mouse click logs of essay compositions.

    Returns:
        pandas.DataFrame: A dataframe with aggregated features for each essay.
    """
    df = df.sort_values(by=["id", "event_id"])
    
    # Calculate the number of activities per category for each essay
    events_per_category = df.groupby(["id", "activity_type"]).size().unstack(fill_value=0)
    # Ensure all activity types are represented
    events_per_category = events_per_category.reindex(columns=ALL_ACTIVITY_TYPES, fill_value=0)

    # Calculate the total number of activities for each essay
    total_events = df.groupby("id").size().rename("total_activities")

    # Merge the DataFrames
    final_df = pd.merge(events_per_category, total_events.to_frame(), left_index=True, right_index=True)

    # Calculate the average time per event for each essay and activity
    average_time_per_activity = df.groupby(["id", "activity_type"])["action_time"].mean().unstack(fill_value=0)
    average_time_per_activity = average_time_per_activity.reindex(columns=ALL_ACTIVITY_TYPES, fill_value=0)
    final_df = pd.merge(final_df, average_time_per_activity, left_index=True, right_index=True, suffixes=('', '_avg_time'))

    # Group by essay id and calculate the start and end times
    grouped_logs = df.groupby("id").agg(start_time=("down_time", "min"), end_time=("up_time", "max"))

    # Calculate the duration for each essay
    grouped_logs["duration_ms"] = grouped_logs["end_time"] - grouped_logs["start_time"]

    # Convert duration from milliseconds to a more readable format if desired, e.g., minutes
    grouped_logs["duration_min"] = grouped_logs["duration_ms"] / 60000

    final_df = pd.merge(final_df, grouped_logs, left_index=True, right_index=True)

    # Get the final word count for each essay
    final_word_count = df.groupby("id")["word_count"].last()
    final_df = pd.merge(final_df, final_word_count, left_index=True, right_index=True)

    # Calculate the deletion to insertion ratio
    deletion_to_insertion_ratio = final_df["Remove/Cut"] / final_df["Input"]

    # Handle division by zero if there are no 'Input' events
    deletion_to_insertion_ratio = deletion_to_insertion_ratio.fillna(0)

    # Add this as a new column to your DataFrame
    final_df["deletion_to_insertion_ratio"] = deletion_to_insertion_ratio
    
    # Calculate basic statistics for action_time for each essay
    action_time_stats = df.groupby("id")["action_time"].agg(["mean", "median", "std", "min", "max", "skew"])

    # Rename the columns to include '_action_time' suffix
    action_time_stats.columns = [f"{col}_action_time" for col in action_time_stats.columns]

    # Merge these statistics with your existing DataFrame
    # Make sure that 'id' column exists in both DataFrames
    final_df = pd.merge(final_df, action_time_stats, on="id")

    # Calculate words per minute for each essay
    final_df["WPM"] = final_df["word_count"] / final_df["duration_min"]
    final_df = final_df.reset_index()
    final_df = final_df[[
        "id",
        "Nonproduction",
        "Input",
        "Remove/Cut",
        "Paste",
        "Replace",
        "Move",
        "total_activities",
        "duration_min",
        "Nonproduction_avg_time",
        "Input_avg_time",
        "Remove/Cut_avg_time",
        "Paste_avg_time",
        "Replace_avg_time",
        "Move_avg_time",
        "word_count",
        "deletion_to_insertion_ratio",
        "WPM",
        "mean_action_time", 
        "median_action_time", 
        "std_action_time",
        "min_action_time",
        "max_action_time",
        "skew_action_time"
    ]]

    return final_df


# In[22]:


train_features = generate_behavioral_features(train_logs)


# ## <span style="color: #7b6b59;">3.2 Pause Features</span>
# 
# 1. **Pause: Definition -** Pauses are generally defined as inter-keystroke intervals (IKI) above a certain threshold (e.g., 2000 milliseconds). The IKI refers to the gap time between two consecutive key presses typically expressed in milliseconds. To illustrate, suppose a writer types a character "A" at time 1 and then a character "B" at time 2. One can obtain the IKI between the two characters simply using the formula: IKI = Time2 - Time1. 
# 1. **Features:** Global measures of pausing are usually associated with the duration and frequency of pauses calculated from different dimensions. Below are some typical pause measures:
#     - **number of pauses** (in total or per minute)
#     - **proportion of pause time** (as a % of total writing time)
#     - **pause length** (usually the mean duration of all pauses in text production)
# 
# - **Understanding Key Events:** Each keystroke has a "down" event (when the key is pressed) and an "up" event (when the key is released). In our dataset, these are represented by `down_time` and `up_time` respectively. `Down Time` denotes the time (in milliseconds) when a key or the mouse was pressed while `Up Time` indicates the release time of the event. `Action Time` represents the duration of the operation (i.e., Up Time - Down Time)
#  
# - **Calculate Inter-Keystroke Intervals (IKIs) for each essay:** There are two ways.
#     - **Solution 1:** For each essay, track the `down_time` and `up_time` of each keystroke to calculate the IKIs. We'll then apply our pause criteria (e.g., IKI > 2000 ms) to identify pauses. Calculating IKI for Each Keystroke: a) IKI is essentially the time gap between the release of one key and the press of the next key. b) To calculate this, you subtract the `up_time` of a keystroke from the `down_time` of the subsequent keystroke. **Applying to the Dataset:**
#         - For each essay, you go through the sequence of keystrokes.
#         - For each keystroke (except the last one), calculate the IKI by taking the down_time of the next keystroke and subtracting the up_time of the current keystroke. For the last keystroke in an essay, you cannot calculate the IKI in the same way since there is no subsequent keystroke. This keystroke is typically excluded from the IKI calculation.
#         - This gives you the time interval between the end of one keystroke and the beginning of the next.
#         - **IKI Calculation Formula:** `IKI = down_time of next keystroke - up_time of current keystroke`
# 
#     - **Solution 2:** For each keystroke event, calculate the time difference between the current `down_time `and the previous `down_time`. For the first keystroke of each essay, the IKI is undefined or can be set to a default value (like 0).
# 
# 

# In[23]:


def calculate_pause_features(essay_df, pause_threshold=2000):
    """Calculate various pause-related features for an essay.

    This function computes the Inter-Keystroke Intervals (IKIs) for each keystroke
    in the given dataframe, identifies pauses based on a specified threshold,
    calculates features such as the number of pauses, total pause time, mean pause length,
    and the proportion of pause time.

    Args:
        essay_df (pd.DataFrame): A dataframe representing the keystroke log of a single essay. 
                                 It should contain at least 'down_time' and 'up_time' columns.
        pause_threshold (int, optional): The threshold (in milliseconds) to define a pause. 
                                         A pause is considered when the IKI is greater than this value. 
                                         Defaults to 2000 milliseconds.

    Returns:
        pd.Series: A Series containing the calculated pause features.
    """
    # Calculate IKIs
    # This line shifts the down_time column up by one row, so that each down_time aligns with the up_time of the previous keystroke.
    # The subtraction then gives the IKI for each pair of consecutive keystrokes.
    #essay_df["IKI"] = essay_df["down_time"].shift(-1) - essay_df["up_time"]    
    # Calculate IKI with a groupby and transform, which will reset the diff() calculation for each group
    essay_df["IKI"] = essay_df["down_time"].diff().fillna(0)

    # Identify pauses (IKI > 2000 milliseconds)
    pauses = essay_df[essay_df["IKI"] > pause_threshold]

    # Calculate pause features
    num_pauses = len(pauses)

    total_pause_time = pauses["IKI"].sum()
    total_writing_time = essay_df["up_time"].max() - essay_df["down_time"].min()
    proportion_pause_time = (total_pause_time / total_writing_time) * 100 if total_writing_time != 0 else 0
    mean_pause_length = pauses["IKI"].mean() if num_pauses != 0 else 0
    mean_pause_length = mean_pause_length / 60000
    total_pause_time = total_pause_time / 60000

    return pd.Series({
        "num_pauses": num_pauses,
        "total_pause_time": total_pause_time,
        "proportion_pause_time": proportion_pause_time,
        "mean_pause_length": mean_pause_length
    })


# In[24]:


# Sort the DataFrame based on the essay 'id' and 'event_id'
train_logs = train_logs.sort_values(by=["id", "event_id"])

# Group by essay ID and calculate features for each essay
# This calculation is done for each essay separately to ensure
# that IKIs are only calculated within the boundaries of a single essay.
pause_features = train_logs.groupby("id").apply(calculate_pause_features).reset_index()

# Display the results
pause_features.head()


# ## <span style="color: #7b6b59;">3.3 Burst Features</span>
# 
# ***Burst Features & Burst Writing Intervals***: 
# 
# - **Definition:** Bursts refer to the periods in text production in which stretches of texts were continuously produced with no pauses and/or revisions. 
# - **Goal:** Identify and measure intervals of continuous writing (long sequences of 'Input' events without interruptions). The duration and frequency of these bursts might relate to the writer's focus and planning ability.
# - There are mainly two types of bursts: 
#     - **P-bursts** that refer to the written segments terminated by pauses, and 
#     - **R-bursts** that describe the segments terminated by an evaluation, revision or other grammatical discontinuity.
# - **Identifying P-bursts and R-bursts:**
#     - P-bursts: These are continuous writing segments without pauses. You can identify them by looking for sequences of `Input` events without significant time gaps (i.e., the time between the `up_time` of one event and the `down_time` of the next event is minimal).
#     - R-bursts: These are segments terminated by revisions, evaluations, or grammatical discontinuities. Look for sequences of `Input` events followed by `Remove/Cut`, `Paste`, `Replace`, or significant pauses.

# # <div style="padding:20px;color:white;margin:0;font-size:30px;font-family:Georgia;text-align:left;display:fill;border-radius:5px;background-color:#7b6b59;overflow:hidden">4. Machine Learning Model</div>
# 

# In[25]:


final_df = pd.merge(train_features, pause_features, on="id")


# In[26]:


final_df = final_df.merge(train_scores, on="id")

# Separate features and target variable
X = final_df.drop(["id", "score"], axis=1)
y = final_df["score"]


# In[27]:


# import lightgbm as lgb
# import numpy as np
# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.metrics import mean_squared_error


# # Define the parameter grid
# param_grid = {
#     'num_leaves': [31, 50, 70],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'n_estimators': [100, 200, 500],
#     'max_depth': [5, 10, 15],
#     # Add other parameters here
# }

# # Create LightGBM model
# model = lgb.LGBMRegressor()

# # Setup K-Fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Setup GridSearchCV
# grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=2)

# # Fit the model
# grid_search.fit(X, y)

# # Best parameters and best score
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", np.sqrt(-grid_search.best_score_))



# In[28]:


# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


# Initialize the XGBoost regressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def RMSE(y_true,y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

y=final_df["score"].values
X=final_df.drop(["id", "score"], axis=1).values
# model=XGBRegressor(base_score=None, booster=None, callbacks=[],
#              colsample_bylevel=0.4893144868569437, colsample_bynode=None,
#              colsample_bytree=1.0, early_stopping_rounds=None,
#              enable_categorical=False, eval_metric=None, feature_types=None,
#              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#              interaction_constraints=None, learning_rate=0.030367573025864663,
#              max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
#              max_delta_step=None, max_depth=2, max_leaves=None,
#              min_child_weight=4.085961297024359,
#              monotone_constraints=None, n_estimators=739, n_jobs=-1,
#              num_parallel_tree=None, predictor=None, random_state=2023)
# model = CatBoostRegressor(
#     iterations=200,         
#     learning_rate=0.05,       
#     depth=6,                
#     loss_function='RMSE',    
#     logging_level='Silent', 
#     random_seed=42
# )

best_params = {'boosting_type': 'gbdt', 
               'metric': 'rmse',
              # 'reg_alpha': 0.003188447814669599, 
              # 'reg_lambda': 0.0010228604507564066, 
               #'colsample_bytree': 0.5420247656839267, 
               #'subsample': 0.9778252382803456, 
               #'feature_fraction': 0.8,
               'bagging_freq': 1,
               'bagging_fraction': 0.75,
               'learning_rate': 0.05, 
               'num_leaves': 19, 
               'min_child_samples': 46,
               'verbosity': -1,
               'random_state': 42,
               'n_estimators': 500,
               'device_type': 'cpu'}

params = {
            "objective": "regression",
            "metric": "rmse",
            'random_state': 42,
            "n_estimators" : 12001,
            "verbosity": -1,
            "device_type": "cpu",
            **best_params
}

model = lgb.LGBMRegressor(**params)
model.fit(X,y)
train_pred=model.predict(X)
train_pred=np.where(train_pred<=0,0,train_pred)
train_pred=np.where(train_pred>=6,6,train_pred)
print(f"RMSE:{RMSE(y,train_pred)}")


# In[30]:


test_logs["activity_type"] = test_logs["activity"].apply(lambda x: "Move" if x.startswith('Move') else x)
test_features = generate_behavioral_features(test_logs)
# Sort the DataFrame based on the essay 'id' and 'event_id'
test_logs = test_logs.sort_values(by=["id", "event_id"])

# Group by essay ID and calculate features for each essay
# This calculation is done for each essay separately to ensure
# that IKIs are only calculated within the boundaries of a single essay.
pause_features_test = test_logs.groupby("id").apply(calculate_pause_features).reset_index()
test_final_df = pd.merge(test_features, pause_features_test, on="id")
test_final_df


# In[31]:


test_X=test_final_df.drop(["id"], axis=1).values
test_pred=model.predict(test_X)
test_pred=np.where(test_pred<=0,0,test_pred)
test_pred=np.where(test_pred>=6,6,test_pred)


# In[32]:


test_pred


# In[33]:


test_final_df['score'] = test_pred
test_final_df[['id', 'score']].to_csv("submission.csv", index=False)


# In[34]:


test_final_df[['id', 'score']]


# # <span style="color: #7b6b59;">Work In Progress...</span>
# 

# In[ ]:





# In[ ]:




