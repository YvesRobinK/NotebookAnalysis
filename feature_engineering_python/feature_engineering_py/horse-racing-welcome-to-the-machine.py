#!/usr/bin/env python
# coding: utf-8

# # Horse Racing &mdash; Welcome to the Machine
# ## An analysis of equine performance and health
# 
# 
# 
# <img src="https://i.imgur.com/SYKb6Um.jpg" width="520">
# 
# *A historic moment at Saratoga as Arrogate wins the 2016 Travers Stakes in track record time*

# In[1]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\ndiv.h1 {\n    font-size: 32px; \n    margin-bottom:2px;\n}\ndiv.h2 {\n    background-color: steelblue; \n    color: white; \n    padding: 8px; \n    padding-right: 300px; \n    font-size: 24px; \n    max-width: 1500px; \n    margin-top: 50px;\n    margin-bottom:4px;\n}\ndiv.h3 {\n    color: steelblue; \n    font-size: 18px; \n    margin-top: 4px; \n    margin-bottom:8px;\n}\ndiv.h4 {\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\nspan.note {\n    font-size: 5; \n    color: gray; \n    font-style: italic;\n}\nspan.lists {\n    font-size: 16; \n    color: dimgray; \n    font-style: bold;\n    vertical-align: top;\n}\nspan.captions {\n    font-size: 5; \n    color: dimgray; \n    font-style: italic;\n    margin-left: 130px;\n    vertical-align: top;\n}\nhr {\n    display: block; \n    color: gray;\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\nhr.light {\n    display: block; \n    color: lightgray;\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n    text-align: center;\n} \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 14px;\n}\ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \ntable.rules tr.best\n{\n    color: green;\n}\n\n</style>\n')


# The NYRA and NYTHA have asked the data science community to join them in improving equine welfare and performance. We accepted the challenge and teamed up to combine data science skills and racing knowledge. Our approach was to identify the factors affecting racing performance and provide practical applications to assist racing teams. We used a series of state-of-the-art Machine Learning models as the basis for our analysis.
# 
# Six factors surfaced as being the most predictive of teams’ finishing positions. We explore the first two factors in greater detail later in the report.
# 
# - **Lateral acceleration**
# - **Pedigree**
# - **Jockey**
# - **Path efficiency**
# - **Post position**
# - **Race distance and course type**
# 
# Many of these factors are correlated with each other; for instance, a jockey’s style and decision-making can affect path efficiency and acceleration. One opportunity then is for a jockey to modify their actions to be more consistent with the patterns of jockeys who win a higher percentage of races. This report includes an augmented racing video designed to supplement the video with metrics based on the tracking data.
# 
# <img src="https://i.imgur.com/zFImBeP.png" width="550">
# 
# Also, we chose these factors because they can be extended to predictive modeling for horse injuries. Accelerative forces on the joints, breeding characteristics, jockey style, course condition, etc. have all been linked to thoroughbred health over time. *(1)*
# 
# Our report contains the following sections:
# - Background
# - Methodology
# - Analysis and Insights
# - Practical Application
# - Appendix
# 
# Thank you for this opportunity! -- John Miller and Steve Ragone
# 

# # Background
# 

# Horse racing in New York has a long and distinguished history, starting with America's first recognized racing event in 1665 at the Newmarket course in Salisbury, New York. Today, the New York Racing Association (NYRA) oversees and operates the pre-eminent thoroughbred horse racing circuit in North America. Races are conducted at three historic venues: Aqueduct Race Track, Belmont Park, and Saratoga Race Course.
# 
# - **Aqueduct Race Track** is a 1-1/8 mile oval track located in New York City. It operates primarily from fall to spring each year. 
# - **Belmont Park**, at 1-1/2 miles, is the largest track in the United States, and is located on Long Island in New York. It is home to the third jewel of racing’s Triple Crown, the Belmont Stakes.
# - **Saratoga Race Course** is a 1-1/8 mile track in Saratoga Springs, New York. Established in 1863, it is one of the oldest, continually run major sports venues in the United States. The 40-day summer meet is considered to be the highest quality race meeting in the country. The Travers Stakes for three-year-olds highlights the racing each year.
# 
# Like other sports, the focus on safety in thoroughbred racing has increased over the last several years. Thoroughbred injuries are costly in many ways and can carry a heavy toll. Jockey injury, veterinary care for the horse, lost prize winnings, and stress for the horses, trainers, and owners are among the negative effects. But we can all agree, most recently, public perception has also made its way to the forefront in this regard. As the sport seeks to attract a new generation of fans, higher safety will translate to more fans and more revenue.

# # Methodology

# We used the following methodology for our analysis. 
# - **Data Collection.** We relied mostly on the racing and coordinate data provided by the hosts. We also collected external data, including pedigree data available online.
# - **Analysis and Insight.** We created a Machine Learning model as the basis for our analysis. The model is designed to detect patterns across multiple variables and correlate the patterns with the finishing positions of the horses. We also conducted deeper analysis into two factors where we thought we could be the most helpful. 
# - **Practical Application.** We considered several ideas and created an overlay for racing videos with calculated metrics. The augmented video could help trainers and jockeys as they review their performance.

# # Analysis and Insights

# In[2]:


get_ipython().system('pip install hvplot')

import re

import colorcet as cc
import holoviews as hv
import hvplot.pandas
import numpy as np
import pandas as pd


# ### Predictive Modeling
# 
# The first part of our analysis focused on the creation of a predictive machine learning model to identify important factors. We used Driverless AI to perform our machine learning experiments. Driverless AI is a software tool that creates best-in-class models using a point-and-click interface. It uses advanced techniques under the hood first developed on Kaggle by data scientists who have won countless competitions. The winning techniques are embedded and automatically executed during model building. 
# 
# The appendix contains details on the final model architecture, feature engineering, and parameters. The model can be completely replicated using open-source software and the information specified in the model’s document.
# 
# The combined model identified several key factors contributing to racing performance as seen in the chart below. Note we did not use velocity as a factor. Mean velocity, aka average speed, is the number one determinant in finishing position. However, using it in a model would not add meaningful insight and would overshadow the contributions of other factors.
# 
# The highlighted bars in the chart show which factors we chose for a deeper analysis.
# 

# In[3]:


shaps = pd.read_csv("../input/equine-safety/shaps.csv")

old_labels = [
    "jockey",
    "dist_from_center_mean",
    "accel_lateral_var",
    "distance_id",
    "program_number",
    "course_type",
    "post_time",
    "accel_lateral_mean",
    "ggs",
    "gggs",
]
new_labels = [
    "Jockey",
    "Path Efficiency",
    "Variance in Lateral Acceleration",
    "Race Distance",
    "Post Position",
    "Course Type",
    "Post Time",
    "Mean Lateral Acceleration",
    "3rd Generation Sires",
    "4th Generation Sires",
]
label_change = dict(zip(old_labels, new_labels))
shaps["label"] = shaps["label"].map(label_change)

shaps = shaps[:10]
shaps_in = shaps.iloc[np.r_[2, 7:10], :]

grid_style = {
    "xgrid_line_color": "gray",
    "ygrid_line_color": "white",
    "grid_line_width": 0.5,
}

def make_bar(df, color):
    lb = df.hvplot.bar(
        x="label",
        y="value",
        c=color,
    #     title="",
        invert=True,
        flip_yaxis=True,
        xlabel="",
        ylabel="Relative Importance",
        xticks=np.arange(0, 0.36, 0.05).tolist(),
        line_alpha=0,
        bar_width=0.2,
    ).opts(
        show_grid=True,
        gridstyle=grid_style,
        show_frame=False,
        width=800,
        height=400,
        fontscale=1.3
        # padding=0.5
    )
    return lb

def make_scatter(df, color):
    rd = df.hvplot.scatter(
        x="label",
        y="value",
        c=color,
        title="Feature Importance",
        invert=True,
        flip_yaxis=True,
        xlabel="",
        ylabel="Relative Importance",
        xticks=np.arange(0, 0.36, 0.05).tolist(),
        size=100,
    ).opts(padding=0.5)
    return rd

bars = make_bar(shaps, "#888888") * make_bar(shaps_in, "#ffa43d")
ends = make_scatter(shaps, "#888888") * make_scatter(shaps_in, "#ffa43d")
display(bars * ends)


# Most of the factors here make intuitive sense although their relative importance may challenge common knowledge. Sometimes though a machine learning model latches onto features that help segregate data without being of individual importance. Post Time may be such a factor as the time of day for a race doesn't seem like it would matter. On the other hand, it could be correlated with track condition - a deeper analysis would be needed.
# 
# The model is directionally correct and does relatively well at the high and low ends. In other words, it typically  identifies winning horses as being in the top 3 and the bottom few as being at the bottom. The model predicts the actual position to +- 2 places. Although it wouldn't be useful for placing winning bets, models at this level are often good at indicating relevant factors. We expect that more data over a longer period of time would improve results.
# 
# Additional data could also allow extension of the model to assess health and safety. Looking at racing data and also aggregate data for each horse would bring to light both acute and chronic conditions that contribute to injuries. 
# 
# After completing our predictive model and identifying key factors we performed deeper analysis in two areas: Lateral Acceleration and Pedigree.
# 
# 
# ### Lateral Acceleration
# 
# We first looked at the lateral acceleration experienced by horses, particularly through the turns. Lateral acceleration in a race is a function of the horse’s speed and change in direction. Increasing speed or tightening the turn radius results in higher lateral acceleration. The picture below shows the geometry of lateral acceleration.
# 
# 
# <img src="https://i.imgur.com/hBIbWlP.png" width="500">
# 
# *Justify, winning the 2018 Belmont Stakes to become the 13th Triple Crown champion*
# 
# The second chart below confirms the importance of the mean and variance of lateral acceleration. It  shows the distribution of lateral acceleration for the 5th race on 9/2/2019 at Saratoga, which was a 9 furlong race on the dirt track. Only the top 2 and bottom 2 finishers are shown for clarity.
# 
# These numbers are for the final turn as shown in the first chart below *(added 11/16)*. 

# In[4]:


df = pd.read_parquet("../input/equine-safety/trakus_accel.parquet")
horses = pd.read_csv("../input/equine-safety/horse_ids.csv")

merge_cols = ["track_id", "race_date", "race_number", "program_number"]

matched = horses.merge(df, on=merge_cols, how="left")
matched["dist_from_center"] = matched.dist_from_center**0.5
matched["time_index"] = matched.trakus_index * 3 / 10

base = matched[
    matched.track_id.eq("SAR")
    & matched.race_date.eq("2019-09-02")
    & matched.race_number.eq(4)
]
base_mean = base.groupby("trakus_index", as_index=False, sort=False).mean(
    numeric_only=True
)
one = matched[
    matched.track_id.eq("SAR")
    & matched.race_date.eq("2019-09-02")
    & matched.race_number.eq(5)
]
race_mean = (
    one[["x_coord", "y_coord", "accel_lateral", "trakus_index"]]
    .groupby("trakus_index")
    .mean()
)
race_mean_chart = race_mean[race_mean.x_coord<=800]

r = race_mean_chart.hvplot.scatter(
    x="x_coord",
    y="y_coord",
    c="accel_lateral",
    cmap=cc.blues[40:],
    xticks = list(range(0, 2600, 200)),
    yticks=list(range(0, 1100, 200)),
    xlabel="X Distance (ft)",
    ylabel="Y Distance (ft)",
    size=10,
    clabel="Lateral Acceleration (ft / s^2)",
    title="Mean Lateral Acceleration at Final Turn",
)
b = base_mean.hvplot.line(
    x="x_coord", y="y_coord", color="gray", alpha=0.2, 
        line_width=18, padding=0.2
).opts(show_frame=False, fontscale=1.2,
         )

display(r * b)


# In[5]:


get_ipython().run_cell_magic('opts', "Distribution [tools=['save', 'pan', 'wheel_zoom', 'box_zoom', 'reset']]", '\nhv.renderer(\'bokeh\').theme = None\none = pd.read_csv("../input/equine-safety/one.csv")\n\ngrid_style = {\n    "xgrid_line_color": "white",\n    "ygrid_line_color": "gray",\n    "grid_line_width": 0.5,\n    "grid_line_alpha": 0.5,\n}\nlateral = (\n    one[one.accel_lateral.gt(3) & one.finishing_place.isin([1, 2, 5, 6])]\n    .sort_values("finishing_place")\n    .hvplot.kde(\n        y="accel_lateral",\n        by="finishing_place",\n        alpha=0.1,\n        xlabel="Lateral Acceleration (ft / s^2)",\n        ylabel="Relative Frequency",\n        title="Distribution of Lateral Acceleration for Final Turn",\n        yticks=np.arange(0, 1.1, 0.2).tolist(),\n        height=400\n    )\n    .opts(show_grid=True, gridstyle=grid_style, show_frame=False, fontscale=1.2,\n         )\n)\nlateral.get_dimension(\'finishing_place\').label=\'Position at Finish\'\n\ndisplay(lateral)\n')


# The chart shows the differences in the lateral acceleration profile between the top two and bottom two finishers. 
# 
# - **Mean level of acceleration** The top finishes have a higher average level of acceleration. This can come from higher speed through the turn, or from a tighter turn radius and shorter path through the turn. Both are advantageous.
# 
# - **Variance, or spread, of acceleration** The top finishers we looked at often had a tighter range, indicating a smoother passage through the turn. Abrupt changes to the high side typically indicate a decrease in forward velocity as a jockey attempts to cut into the turn. Changes to the low side typically indicate a decrease in velocity or a widening of the turn radius. Changes in either direction can reduce overall performance through a turn.
# 
# The chart above is typical for the races we examined, although not every race shows a clear difference. Horses win or lose for any number of reasons, including average velocity as we mentioned before. In the case shown here, it might indicate to the jockey in 6th place that execution during the final turn contributed in part to their loss.
# 
# The next chart shows the profile of lateral acceleration during the same race. As with any high-frequency sensor data, the raw numbers contain a certain amount of noise. We smoothed the raw data using a combination of short-term rolling mean and a function called a Kalman filter. This smoothing method reduced a fair amount of noise while still capturing the variations.
# 

# In[6]:


onet = one[one.finishing_place.isin([1,2,5,6]) & one.time_index.between(70, 87)]

line_plot = onet.hvplot.line(
    x="time_index",
    y="accel_lateral",
    by="finishing_place",
    alpha=1,
    xlabel="Time (s)",
    ylabel="Lateral Acceleration (ft / s^2)",
    title="Lateral Acceleration through the Final Turn",
    xticks = list(range(70,87, 2)),
    yticks = np.arange(5.5, 8.2, 0.5).tolist(),
    grid=True,
    height=400,
    width=800
).opts(fontscale=1.2)

line_plot.get_dimension('finishing_place').label='Position at Finish'
display(line_plot)


# Here it becomes clear that horse 1 (Rucksack) and horse 2 (Super Silver) maintained higher acceleration through the turn than the other horses. Super Silver accelerated faster out of the turn than the other horses, which actually might have helped. A review of the video along with velocity and acceleration charts are needed to draw a conclusion for this specific race.

# ### Pedigree
# 
# We next took a closer look at pedigreee, the second of our two deep dives. The pedigree of a thoroughbred receives a great deal of attention, both from the perspective of the foal’s potential and the sire’s previous success. We analyzed sire lines for each horse from the 2nd generation through the 5th generation. For this analysis, we used Natural Language Processing (NLP) to look directly at the names of the sires and identified which sires appeared to have the most influence on racing performance. We used Bidirectional Encoder Representations from Transformers (BERT) in combination with a tree-based model. BERT was developed by Google in 2018 and, along with derivative transformers, has proven to be the highest performing method for NLP tasks.
# 
# We included sires from both the paternal and maternal sides of each generation. A 2015 study on the role of maternal lineage found that the heritability of performance between dams and their foals (r=0.141) was significantly higher than between sires and their foals (r=0.035). They further found that a good stallion could generally not negate the heredity effect of a poor race mare but that a good race mare could negate the hereditary effect of a poor sire. *[2]*
# 
# The chart below shows the most influential sires relative to a horse’s racing results. Overall, the 3rd generation sires on both sides show the most influence. The result is expected because the lineage is less diluted at that level, and more unique than older generations. First and 2nd generation sires would intuitively be even more important, but the model is limited by the small sample size. There is not enough commonality in those generations across the horses in our sample to discover a pattern.
# 

# In[7]:


locos = pd.read_csv("../input/equine-safety/locotable.csv")
locos = locos[:8]

locos.label.tolist()
pretty_labels = [
    "3rd gen: Seattle Slew",
    "3rd gen: Mr Prospector",
    "3rd gen: A P Indy",
    "3rd gen: Unbridled",
    "3rd gen: Stormcat",
    "4th gen: Mr Prospector",
    "4th gen: Raise a Native",
    "4th gen: Northern Dancer"
]
locos["label"] = pretty_labels

grid_style = {
    "xgrid_line_color": "gray",
    "ygrid_line_color": "white",
    "grid_line_width": 0.5,
}

lb = locos.hvplot.bar(
    x="label",
    y="value",
#     title="",
    invert=True,
    flip_yaxis=True,
    xlabel="",
    ylabel="Relative Importance",
    xticks=np.arange(0, 0.36, 0.05).tolist(),
    line_alpha=0,
    bar_width=0.2,
).opts(
    show_grid=True,
    gridstyle=grid_style,
    show_frame=False,
    width=800,
    height=400,
    fontscale=1.3
    # padding=0.5
)

s = locos.hvplot.scatter(
    x="label",
    y="value",
    title="Sires of High Performing Horses",
    invert=True,
    flip_yaxis=True,
    xlabel="",
    ylabel="Relative Importance",
    xticks=np.arange(0, 0.36, 0.05).tolist(),
    size=150,
).opts(padding=0.5)

display(lb * s)


# Extending this study to all current thoroughbreds and including detailed maternal lineage could likely be used also for identifying correlations between pedigree and horses' susceptibility to chronic health conditions. The findings could supplement research in equine genetics such as that underway at The School of Veterinary Medicine at UC Davis.*[4]*
# 
# Raw data for this part of the analysis was gathered from the Pedigree Online Thoroughbred Database. The linked dataset contains the file used.

# # Practical Application

# There are several ways the acceleration, velocities, and positions from sensor data can be used together. Televised horse racing already uses the data in several ways to provide on-screen information throughout a race. 
# 
# We propose an on-screen overlay that trainers, jockeys, and other stakeholders can use to assess performance in certain areas. As mentioned before, videos of the races provide context for charts of the metrics. And the charts provide a quantitative view, based on data, to supplement the video. It's likely that a seasoned jockey or horse enthusiast will be able to assess most of this information most of the time. For younger jockeys and new fans to racing, it would be an addition to interpreting one aspect of a race's outcome. The cumulative view showing distributions of acceleration can also be useful in showing the characteristics associated with high performance of jockey and horse. 
# 
# Here is one version of combining key metrics with video replays. Here we show lateral and longitudinal acceleration for a single horse. We chose Super Silver in the race mentioned previously due to his strong exit from the final turn and the final push at the finish line.You can see how longitudinal and lateral acceleration typically move in opposing directions.

# The video shown here is meant to be a proof of concept. We would expect that further evolution from the horse racing community would lead to new and exciting ways to build on the concept.
# 
# 
# 
# 
# The blue line shows lateral acceleration for Super Silver.
# The red line shows his longitudinal acceleration.
# 
# **Follow the link on the video to YouTube to watch full screen and hear the announcer.**

# In[8]:


from IPython.display import YouTubeVideo

YouTubeVideo('NuOlFT6CdV0', width=600)


# # Conclusion

# This is the first time to our knowledge that Trakus coordinate data has been made publicly available. We utilized that data as well as other publicly available information for our analysis. The limitations of the data were apparent. A larger, multi-year dataset would allow for more accurate diagnosis of racing best practices and would better detect telltale signs of impending racehorse injury.
# 
# 
# 
# Even so,we believe that additional metrics and insights based on current sensor data are a useful addition to the NYRA's current efforts. It would provide jockeys and trainers  with additional information in context when reviewing game video, and also reinforce the importance of choices made during turns. The information can also contribute to fan engagement and provide additional understanding when viewing on-air broadcasts.
# 
# 
# 
# Finally, further analysis of key factors based on sensor data and factors such as pedigree will likely provide insights beyond our findings. Such analysis will provide data-based evidence that will either reinforce or challenge the collective body of knowledge gained from over 300 years of horse racing in New York and elsewhere.
# 

# <img src="https://i.imgur.com/p05TMgT.png" width="550">
# 
# *One of the rarest occurrences in racing, a triple dead heat. The most famous one took place at Aqueduct Race Track, in the 1944 Carter Handicap.*

# # Appendix

# #### About the authors
# 
# John Miller works as a Principal Data Scientist at H2O.ai. He is currently helping a large hospital system use AI to provide better healthcare. John has also contributed in the past to sports safety for football players. The NFL used his recommendations to support their efforts for reducing player concussions and leg injuries.
# 
# Steve Ragone is retired from a career in the public sector, most recently serving as a Health Program Director. A lifelong horse racing aficionado, he has attended the races at Saratoga for over 50 years. Steve has seen Triple Crown champions Secretariat, Seattle Slew, Affirmed, and American Pharoah compete in person.
# 

# #### Model Details
# 
# The final machine learning model is a combination of two tree-based models. The autodoc.docx word document in the attached dataset contains 75 pages describing details of the model, including the full set of factors used.

# #### Other Ideas that might improve safety
# 
# We considered a few other ideas during the course of our analysis based on one author's knowledge of horse racing gained over many years. Data in these areas was not available so we left them as potential areas of interest.
# 
# **Temporary Rail for Dirt Courses**
# 
# Admittedly an outside the box suggestion - would be a temporary rail on dirt courses, especially during stretches of inclement weather. This could allow for a more fair and safe racetrack by taking the inside, oversaturated portion of the track out of the equation. Sedimentation during and after heavy rains no doubt occurs, and alters the makeup of the track, particularly that part closest to the rail. A temporary rail may provide a good opportunity for the track superintendent to restore the lowest portion of the track to proper condition while allowing for the continuation of racing on a more uniform track.
# 
# <img src="https://i.imgur.com/JqIOs0C.jpg" width="550">
# 
# 
# **Analyzing Warm-ups**
# 
# Trainers rightfully often wonder, especially before big races, if a horse’s final workout was too slow or too fast. Did a blazing work take too much out of the runner? Did a slow work not ensure ultimate fitness? 
# 
# These workouts usually occur a week before the race. If there is concern about taking too much out of a horse a week in advance, what about aggressive pre-race warm-ups? Should there be any consideration given for effects on endurance in the moments preceding a race? What effect, if any, do aggressive or non-aggressive warm-ups have on performance?
# 
# Per its literature, Trakus monitors horses as soon as they enter the track. It might be beneficial to examine the impact warm-ups have on performance and injury. By analyzing the distance and speed of pre-race activity, one could identify the positive and negative aspects of various types of warm-ups. Is there a relationship between warm-ups and performance? Is there a relationship between warm-ups and injury? 
# 
# 
# **Enhanced Presentation / Identifying Prospective Injuries**
# 
# We believe the Trakus data can be used further to enhance the presentation of the NYRA racing product, and at the same time identify horses that may tipping off lameness or some other ailment. By using a model to factor in, among other things, pace, distance, surface, class level, position and velocity, the NYRA could develop something similar to what the NFL uses to show Win Probability during games. However, for horse races, the percentages will be in a state of flux as the race progresses and a horse’s position and speed is measured against competitor’s positions and speed, as well as distance remaining in the race. 
# 
# We can also use the data to identify, in real-time, idiosyncrasies
# in a horse’s deceleration when viewed against what is typical for the class
# level of the race for a particular surface and distance. Any horse that is
# identified to be significantly outside the expected values of deceleration,
# which may be the harbinger of an impending health issue, can be immediately
# sent to the post-race test barn for veterinary examination or assigned for
# further evaluation prior to entry in another race.
# 
# Product improvements gained through technological innovation
# is a hallmark of sporting event presentation, and these enhanced graphics
# should appeal to NYRA. Injury prevention/identification is certainly first and
# foremost on the minds of horse racing’s curators, so any tool that can aid in
# this endeavor should be welcome assistance.
# 
# The following series of graphics are from the 2019 Diana Stakes at Saratoga Win percentages change as the race develops and converge toward the end of the race. Note: this is a prototype and is not based on any AI currently available.
# 
# 
# <img src="https://i.imgur.com/xwLuFjx.jpg" width="550">
# 
# 

# *A 5-second glimpse into the future...*

# In[9]:


YouTubeVideo('Fi9UVbf6pz8', width=600, height=300)


# **If the Shoe Doesn’t Fit...**
# 
# Is it inconceivable that shoes lost on the track are responsible for some racing injuries? Shoes are lost all the time, along with nails. Are competitors checking for lost shoes after every race and initiating recovery prior to subsequent races?
# 
# <img src="https://i.imgur.com/nPCXmR7.jpg" width="580">
# 
# 
# *This horse will need help finding his shoe.*

# #### Citations
# 
# [1] Meta-analysis of risk factors for racehorse catastrophic musculoskeletal injury in flat racing. The Veterinary Journal 245 (2019) 29–40
# 
# [2] Potential role of maternal lineage in the thoroughbred breeding strategy. Reproduction, Fertility, and Development
#  (2015 May 5). doi: 10.1071/RD15063
# 
# [3] Pedigree Online Thoroughbred database.
# 
# [4] Horse Report, Summer 2020. Center for Equine Health. School of Veterinary Medicine, University of California, Davis
