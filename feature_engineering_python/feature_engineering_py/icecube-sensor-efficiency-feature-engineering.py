#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# # IceCube Sensor Sensitivity: Feature Engineering

# This project is based off [Paper Overview: Graph Neural Networks in IceCube notebook](https://www.kaggle.com/code/antonsevostianov/paper-overview-graph-neural-networks-in-icecube). We aren't going to discuss GNNs in this work, but concentrate on a few findings of the paper, which can, hopefully, help improve the models for the IceCube competition.
# 
# The purpose of this notebook is:
# 
# 1. Check how sensors of IceCube differ in performance.
# 2. Create a **new feature** for future model - quantum efficiency (QE) aka sensor sensitivity/precision. 
# 
# ### Contents:
# 
# 1. Brief IceCube Detectors Overview
# 2. Breaking down detectors into groups
# 3. Assigning numerical values of relative performance to the groups
# 4. Conclusion

# ## IceCube Detectors
# 
# As the paper by R. Abbasi et al and the previously linked notebook discussed, IceCube Observatory isn't a monolith with homogenious sensors. All **detectors vary in their precision** and efficiency, therefore, this data can be used as a feature for future models.
# 
# Let's have a look again at the diagram.
# 
# <img src="https://i.postimg.cc/y8Cmr7Kg/IceCube.png" width="400px" height="500px">
# 
# Main takeaways are:
# 
# * Ice around detectors aren't homogeneous - their absorption properties differs, depending on the depth. 
# * [As a result] Different detectors have different performance/precision
# * *DeepCore enhanced quantum efficiency detectors* (marked green on the picture) are **the most precise**, according to the authors, they're at least 1.35 times more precise than standard detectors 
# * Other *detectors beneath the Dust layer* [layer of ice with dust impurities] perform well, but worse than DeepCore ones
# * *Detectors above the Dust layer* perform worse than those beneath it
# * *Detectors inside the Dust layer* are **the worst**, precision wise

# ## Detector break down by groups
# 
# As described above, all sensors can be broken down into several groups by their relative performance.

# In[2]:


sensor_geometry = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')


# Based on the information available, we can divide all sensors into following groups:
# 
# * **DeepCore** sensors - the best quantum efficiency.
# * Detectors **under** the dust layer - second best QE
# * Detectors **above** the dust layer - third best QE
# * Detectors **inside** the dust layer - worst QE
# 
# Now it's time to create  new feature, for now categorical.

# In[3]:


def detector_group(x, z):
    """
    Assigns values - deepcore, dustlayer, abovedust, underdust -
    depending on sensor coordinates x and z
    """
    
    # Define functions for each sensor category
    def is_deepcore(x, z):
        return x in {57.2, -9.68, 31.25, 72.37, 113.19, 106.94, 41.6, -10.97} or \
               (x in {46.29, 194.34, 90.49, -32.96, -77.8, 1.71, 124.97} and 
                ((z <= 186.02 and z >= 95.91) or 
                 (z <= -157 and z >= -511)))
    
    def is_dustlayer(z):
        return z <= 0 and z >= -155
    
    def is_abovedust(x, z):
        return z > 0 and not is_deepcore(x, z) and not is_dustlayer(z)
    
    def is_underdust(x, z):
        return not is_deepcore(x, z) and not is_dustlayer(z) and not is_abovedust(x, z)
    
    # Check which sensor category the coordinates belong to
    if is_deepcore(x, z):
        return "deepcore"
    
    if is_dustlayer(z):
        return "dustlayer"
    
    if is_abovedust(x, z):
        return "abovedust"
    
    return "underdust"


# In[4]:


sensor_geometry['sensor_group'] = sensor_geometry.apply(lambda row: detector_group(row['x'], row['z']), axis=1)


# In[5]:


sensor_geometry.head()


# In[6]:


scatter3d = px.scatter_3d(sensor_geometry, x='x', y='y', z='z',color='sensor_group', opacity=0.7)
scatter3d.update_traces(marker = dict(size = 2, symbol = "diamond-open"))
scatter3d.update_coloraxes(showscale = False)
scatter3d.update_layout(template = "plotly", font = dict(family = "Arial", size = 12, color = "#9e97ff"))
scatter3d.show()


# Et voilà! We've broken down all sensors into groups, depending on their quantum efficiency. Though, there is more to sensor QE, than just it's location within designated groups, even this rough division can give us additional information that can, probably, increase the score in the competition. 

# ### Detectors relative performance
# 
# Each detector is unique and has slightly different performance comparing to others. Unfortunately there is little available information to assaign a certain value to every detector. However, albeit roughly, we still can give an estimate of relative performance of groups of sensors we created.
# 
# According to R. Abbasi et al, DeepCore sensors are roughly 1.35 times more sensitive than "normal" detectors. We also know that dust layer detectors are roughly 0.5 - 0.6 time less sensitive. Based on this information we can engineer an additional feature - DOM's relative quantum efficiency.

# In[7]:


def relative_qe(group):
    """
    Returns a relative quantum efficiency of a sensor group according to the following rules:
    - 1.35, if sensor is deepcore;
    - 0.95, if sensor is abovedust;
    - 1.05 if sensor is underdust;
    - 0.6 if sensor is dustlayer
    """
    if group == 'deepcore':
        return 1.35
    if group == 'abovedust':
        return 0.95
    if group == 'underdust':
        return 1.05
    else:
        return 0.6


# In[8]:


sensor_geometry['relative_qe'] = sensor_geometry['sensor_group'].apply(relative_qe)


# In[9]:


sensor_geometry.head()


# Our dataset is ready! We can use one hot encoding to break down the categorical variable, if we want to, or just go with the numerical variable.

# ## Conclusion
# 
# In this notebook we were feature engineering a **sensor efficiency feature** that might come handy in training models for IceCube competiton. We have assigned a categorical and a numerical value to 4 different groups of detectors based on their relative performance.
# 
# If you're going to use this feature, please beware that numerical values are only rough estimates - so use your discretion while incorporating it into your models. While categorical column is probably fine, those numerical values might be hit or miss. However - this is the beauty of Kaggle competitions - you can try and see if it works in your model!
# 
# PS. Please let me know if there are any mistakes. You most welcome to **share your experience - if this feature works or not for you**. 
