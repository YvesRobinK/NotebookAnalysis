#!/usr/bin/env python
# coding: utf-8

# ## Tips for making the Informative Visualization
# 
# I've done a lot of visualizations at Kaggle, and I've gained a lot of insight into visualizations in the process.
# 
# Share some visualization tips in data analysis, hoping you won't make the same mistake as me.
# 
# ![](https://scontent-ssn1-1.xx.fbcdn.net/v/t1.0-9/s960x960/119476347_3252251351526507_8906565759837447200_o.jpg?_nc_cat=106&ccb=2&_nc_sid=825194&_nc_eui2=AeFs1jMO0uWBFhU86RESyXDW4Uz4Ja3Ean3hTPglrcRqfUeoe3F2-OqI1ghXoL0chYIcvpIrLBo7WrH2hl5aZZFG&_nc_ohc=0JY5477ksqUAX_f1R0b&_nc_ht=scontent-ssn1-1.xx&tp=7&oh=adbba25fe0fdaf4da44e3e2a8dd87e60&oe=5FCE2AB2)
# 
# > Small data visualization community in Korea.

# ## Data visualization is "Data" and "Visualization"
# 
# Literally. Data visualization is a task that requires both data and visualization to be understood at the same time.
# 
# First, you have to understand the type of the variable. 
# 
# - **Numerical**
#     - **Discrete**: Discrete data such as number of people and dice scale.
#         - 0, 1, 2
#     - **Continuous**: Data such as age, score, temperature, etc.
#         - 10.2, 20.3
# - **Categorical**
#     - **Ordered**: Categorical data, such as high school, university, and graduate school.
#     - **Nominal**: There is no order, such as country, blood type, race, but different categorical data.
# 
# Then, examples of preattentive(visual) attributes are as follows:
# 
# - Location
# - Form
# - Size
# - Color
# - ETC
# 
# <img src="https://miro.medium.com/proxy/0*0Mil7au6biskTv6e" width=600> 
# 
# You need to understand the proportional elements of data and visualization elements.

# ## Let's proceed with visualizations that fit the purpose.
# 
# Visualization is a targeted act.
# 
# Information visualization is a process to find 4 main things.
# 
# - **Composition**: What does the data consist of?
# - **Distribution**: What distribution does the data have?
# - **Comparison**: What distribution are features in?
# - **Relationship**: What about the relationship between two or more features?
# 
# <img src="https://about.infogr.am/wp-content/uploads/2016/01/types-of-charts-1024x767.jpg" width=700>
# 
# And let's think about this.
# 
# - What content can we use after this?
# - What process will it take to do Feature Engineering after EDA?
# - Did they create visualizations that are more understandable than text?
# 
# Visualization does not necessarily have to be aesthetically pleasing if it serves its purpose and appropriately helps to understand.
# 
# However, ***pretty visualizations are better than ugly ones.***

# ## Colorful isn't necessarily a good thing.
# 
# <img src="https://64.media.tumblr.com/6afe14d0677fd565808b3faa2be44036/tumblr_q9gpy4XYFW1sgh0voo1_1280.png" width=700>
# 
# > ref : https://viz.wtf/
# ### Stop using the rainbow color palette
# 
# Colors can be used in two main ways.
# 
# 1. Distinguish: Data are different from each other.
# 2. Emphasis: Emphasize subdata.
# 
# There are three main types of color palette if you look at the classification first.
# 
# - **Qualitative palettes**
# - **Sequential palettes**
# - **Diverging palettes**
# 
# I think I can refer to [Simple Matplotlib & Visualization Tips 💡](https://www.kaggle.com/subinium/simple-matplotlib-visualization-tips)
# 
# ### Contrast
# 
# For emphasis or effective differentiation, color contrast should be well utilized. The types of contrast are as follows.
# 
# ![](https://libdiz.com/wp-content/uploads/2016/01/ittens-color-contrast-natural-colourslive.jpg)
# 
# ### Meaning of Colors
# 
# Let's make good use of the overall concept of color. Colors all have meanings and impressions.
# 
# ![](https://blog.presentation-company.com/hs-fs/hubfs/Imported_Blog_Media/MeaningOfColor_v3-1.png?width=600&height=306&name=MeaningOfColor_v3-1.png)
# 
# In addition to this, think about the meaning of brand color.
# 
# ### Color blindness
# 
# There is a greater percentage of color blindness than you thought. 
# Consider also color to prevent color blindness.
# 
# ### Color Scheme
# 
# Color always exists in combination. The combination of good color and good color does not make a good color scheme.
# 
# Always refer to different color combinations.
# 
# - [Color Hunt](https://colorhunt.co/)
# - [Visme : 50 Beautiful Color Combinations](https://visme.co/blog/color-combinations/)

# ### A lot of charts don't mean a lot of analysis.
# 
# Drawing multiple charts is inconvenient for many reasons.
# 
# > In particular, it would be nice to use the facet grid well.
# 
# Very few people can read multiple charts at the same time.
# 
# The web environment is quite limited in size, so one neat visualization may be better.

# ### Interactive, understand and use.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/Plotly-logo-01-square.png" width=300>
# 
# Let's understand the difference between interactive and static visualizations.
# 
# Many of the interactive visualizations used in the Cagle mean no more than static visualizations. 
# 
# Rather, there may be only negative effects that slow down the rendering speed. 
# 
# The key reason for interactive is not glamour but to allow users to look at data autonomously. 
# 
# Let's use meaningful interactive. The representative features of Python Interactive Visualization Library include:
# 
# - **Explore**
#     - A typical function is panning, which allows you to view the chart in various places while moving the right left side with the mouse pressed.
#     - It can be useful on a large scale, such as map data, but is there a reason to use it in the bar chart?
# - **Select**
# - **Abstract** : mouse hover and tooltip
#     - Except for information on color and coordinate axes, insufficient information can be filled.
#     - However, most of the contents contain color and coordinate information. Of course, you can show more precise values, but there aren't many cases that take advantage of them. Let's put the necessary information.
# - **Animation**
#     - It seems useful for time series data. However, when expressing heat maps, etc., let's fix the color scale and add animations that are less confusing meaningfully.

# ## The detail completes the visualization.
# 
# ### Arrangement(Layout), margin, and ratio
# 
# - Direction and readability.
#     - In many cultures, texts and charts are read left and right, and from top to right.
#     - But one looks at the color and the overall structure first, and then again from the beginning.
# - Let's not do all the charts at the same rate.  The aspect ratio is important.
#     - Even charts of the same kind have different ratios if the number of features is different.
#     - If you want to draw it at once, let's vary the size of the chart, such as grid spec.
# - Avoid visual frustration by setting margins or spacing between charts.
#     - Let's leave it a little more relaxed than just hitting the x-axis and y-axis limits.
#     - Let's also leave a gap between axes.
# 
# ### Text Utilization
# 

# In[1]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,10))
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

ax.text(3, 2, 'unicode: Institut für Festkörperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

plt.show()


# 
# - Text based on location
#     
#     - Title : The title is a summary of the entire subject of the visualization.
#     - Axis
#     - Legend
#     - Explanation
#         - This could be a description of each data point.
#         - It could be an explanation of the whole visualization.
# 
# 
# 
# - The setting of the text component.
# 
#     - fontfamily : I prefer 'serif'
#     - fontweight
#     - fontsize 
#     - color 
#     - bbox : To decorate the border of text, it helps to highlight text and distinguish it from charts.
# 
# ### Grids, Axis, and Borders
# 
# - **Grid**: When there is more data, it helps to understand the scale.
# - **Axis**: Minor tags can be set to show detailed scale, and if the axis is meaningless, remove it.
#     - ex) Dimension Reduction
# - **Border**: Let's try various things, such as removing spines or applying a border to data points.

# ## You don't have to look at all the data.
# 
# While data analysis looks at the entire data, it also selects the appropriate subset of the data.

# ## Reference
# 
# - Visualization Analysis & Design, Tamara Munzner
# - Fundamentals of Data Visualization, Claus O.Wilke
# - A Gentle Introduction to Exploratory Data Analysis : [https://towardsdatascience.com/a-gentle-introduction-to-exploratory-data-analysis-f11d843b8184](https://towardsdatascience.com/a-gentle-introduction-to-exploratory-data-analysis-f11d843b8184)
# - Finding the Right Color Palettes for Data Visualizations : [https://blog.graphiq.com/finding-the-right-color-palettes-for-data-visualizations-fcd4e707a283#.wf7bgbdq2](https://blog.graphiq.com/finding-the-right-color-palettes-for-data-visualizations-fcd4e707a283#.wf7bgbdq2)
# - How to Choose Colors for Your Data Visualizations : [https://medium.com/nightingale/how-to-choose-the-colors-for-your-data-visualizations-50b2557fa335](https://medium.com/nightingale/how-to-choose-the-colors-for-your-data-visualizations-50b2557fa335)
# - The Architecture of a DataVisualization [https://medium.com/accurat-studio/the-architecture-of-a-data-visualization-470b807799b4](https://medium.com/accurat-studio/the-architecture-of-a-data-visualization-470b807799b4)
# - Mistakes, we’ve drawn a few : [https://medium.economist.com/mistakes-weve-drawn-a-few-8cdd8a42d368](https://medium.economist.com/mistakes-weve-drawn-a-few-8cdd8a42d368)
# - [Toward a Deeper Understanding of the Role of Interaction in Information Visualization(2007)](https://www.cc.gatech.edu/~stasko/papers/infovis07-interaction.pdf)
