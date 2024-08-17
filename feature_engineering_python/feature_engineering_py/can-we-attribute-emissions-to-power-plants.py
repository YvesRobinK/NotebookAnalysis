#!/usr/bin/env python
# coding: utf-8

# ## <u>About this kernel</u>
# 
# To calculate marginal emissions factor, I want to attribute emissions to each power plants.
# 
# ※For marginal emissions factor, refer https://www.tmrow.com/blog/marginal-emissions-what-they-are-and-when-to-use-them
# 
# 
# For calculate reasonable emissions of each power plants, I try to apply k-means to satellite data as one of the typical clustering methods.
# 
# If we devide into small area, we can detect its area(m^2) and gas emission by following fomula:
# 
# ## Emissions(mol) = NO2_column_number_density(mol/m^2) * area(m^2)
# 
# When we get mol of emited gas, we can calculate its weight (such as gram).
# 

# In[1]:


from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import folium
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio as rio
import seaborn as sns
from sklearn.cluster import KMeans
import tifffile as tiff 


# ## Snippts

# In[2]:


def overlay_image_on_puerto_rico_df(df, img, zoom):
    lat_map=df.iloc[[0]].loc[:,["latitude"]].iat[0,0]
    lon_map=df.iloc[[0]].loc[:,["longitude"]].iat[0,0]
    m = folium.Map([lat_map, lon_map], zoom_start=zoom)
    color={ 'Hydro' : 'lightblue', 'Solar' : 'orange', 'Oil' : 'darkblue', 'Coal' : 'black', 'Gas' : 'lightgray', 'Wind' : 'green' }
    folium.raster_layers.ImageOverlay(
        image=img,
        bounds = [[18.56,-67.32,],[17.90,-65.194]],
        colormap=lambda x: (1, 0, 0, x),
    ).add_to(m)
    
    for i in range(0,len(df)):
        popup = folium.Popup(str(df.primary_fuel[i:i+1]))
        folium.Marker([df["latitude"].iloc[i],df["longitude"].iloc[i]],
                     icon=folium.Icon(icon_color='red',icon ='bolt',prefix='fa',color=color[df.primary_fuel.iloc[i]])).add_to(m)
        
    return m


# In[3]:


def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):
    for i in range(0, len(dataframe)):
        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]
    return dataframe


# ## Data overview

# In[4]:


power_plants = pd.read_csv('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
power_plants = split_column_into_new_columns(power_plants,'.geo','latitude',50,66)
power_plants = split_column_into_new_columns(power_plants,'.geo','longitude',31,48)
power_plants['latitude'] = power_plants['latitude'].astype(float)
a = np.array(power_plants['latitude'].values.tolist()) # 18 instead of 8
power_plants['latitude'] = np.where(a < 10, a+10, a).tolist() 
power_plants_df = power_plants.sort_values('capacity_mw',ascending=False).reset_index()


# In[5]:


power_plants_df.head()


# In[6]:


#From https://www.kaggle.com/ajulian/capacity-factor-in-power-plants

total_capacity_mw = power_plants_df['capacity_mw'].sum()
print('Total Installed Capacity: '+'{:.2f}'.format(total_capacity_mw) + ' MW')
capacity = (power_plants_df.groupby(['primary_fuel'])['capacity_mw'].sum()).to_frame()
capacity = capacity.sort_values('capacity_mw',ascending=False)
capacity['percentage_of_total'] = (capacity['capacity_mw']/total_capacity_mw)*100
capacity.sort_values(by='percentage_of_total', ascending=True)['percentage_of_total'].plot(kind='bar',color=['lightblue', 'green', 'orange', 'black','lightgray','darkblue'])


# **Power plants and NO2_column_number_density of 20180701-20180707 in Puerto Rico**

# In[7]:


image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')
overlay_image_on_puerto_rico_df(power_plants_df,image[:,:,0],8)

#https://www.kaggle.com/paultimothymooney/explore-image-metadata-s5p-gfs-gldas
#band1: NO2_column_number_density


# Especially, Coal, Gas and Oil power plants emit NO2 gas. So it may be nice idea that whole data divide into some areas. 
# 
# Ideally I want to devide whole data into areas as many as power plants in Puerto Rico, but it is difficult. One of the difficulty is that there are some power plants very near each other. 
# 
# 
# In small divided areas, it would be acceptable to divide the emissions by the amount of electricity generated.

# # Devide NO2 map by k-means
# 
# I try to generate devided areas by longitude, latitude and average NO2 density in a week.

# In[8]:


lon = []
lat = []
NO2 = []

for i in range(image[:,:,0].shape[0]):
    for j in range(image[:,:,0].shape[1]):
        #print(image[:,:,0][i,j])
        NO2.append(image[:,:,0][i,j])
        lon.append(i)
        lat.append(j)
        
NO2 = np.array(NO2)
lon = np.array(lon)
lat = np.array(lat)


# In[9]:


results = pd.DataFrame(columns=['NO2', 'lat', 'lon'])
results = pd.DataFrame({'NO2': NO2/max(NO2),
                    'lat': lat/max(lat),
                    'lon': lon/max(lon)})


# In[10]:


sns.distplot(results["NO2"])


# In[11]:


pred = KMeans(n_clusters=11).fit_predict(results)


# In[12]:


plt.figure()
sns.heatmap(pred.reshape((148, 475)))


# In[13]:


overlay_image_on_puerto_rico_df(power_plants_df, pred.reshape((148, 475)), 8)


# ## Consideration
# 
# Considering the area made, there are some characteristics like below:
# 
# ・The devided areas seems be stronglly  constrained by coordinates. 
# 
# ・ We have to decide the number of cluster manually.
# 
# It is nice that model more reasonablly devide while data into area if we give the data and coodinates of power plants.
# 
# If we input terrain and weather data to k-means, I think we may get more apposite attribute.

# # Classification including wind data.

# ### Can we devide only by NO2 density?
# 
# In the first place, NO2 gas is distributes as we image? For example, if NO2 gas is distributed in a circle or a band from the power plant, it seems good to simply cluster at the density of NO2. But if not so, we have to use other data they seems efffect NO2 gas distribution.
# 
# Fitst, simply we can use k-means to convert no2 density to more monotonous classes. 

# In[14]:


monotonous = KMeans(n_clusters=3).fit_predict(image[:,:,0].reshape(-1, 1))
#pred = KMeans(n_clusters=2).fit_predict(results["NO2"])
plt.figure()
sns.heatmap(monotonous.reshape((148, 475)))


# In[15]:


#Note that the color intensity and the high NO2 concentration do not always match.
overlay_image_on_puerto_rico_df(power_plants_df, monotonous.reshape((148, 475)), 8)


# 
# As you can see from this figure, simply clustering seems to be difficult because the NO2 gas concentration is not distribute in a circle or a band from the power plant.　😅

# ## Including Wind Data
# 
# I tried to make wind feature, and make cluster by k-means.
# 
# NO2 gases flowed by wind, and accumulate place where no wind.

# In[16]:


import os 
import glob


# In[17]:


gldas_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gldas/*')
gldas_files = sorted(gldas_files)
gfs_files = glob.glob('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*')
gfs_files = sorted(gfs_files)


# Let's retrieve the data from 7/1 to 7/7. And I put data together each day.

# In[18]:


gldas_files_par_day = []
for i in range(0,len(gldas_files[6:54]),8):
    #print(gldas_files[i:i+8])
    gldas_files_par_day.append(gldas_files[i:i+8])


# In[19]:


gfs_files_par_day = []
for i in range(0,len(gfs_files[3:27]),4):
    #print(gfs_files[i:i+4])
    gfs_files_par_day.append(gfs_files[i:i+4])


# ### future engineering
# 
# I tried flooowing 2 way.
# 
# We can use following three wind propaties: wind_u, wind_v and wind speed. They are given smaller time scale than NO2 gas.
# 
# 1. just calculate average of wind_u, wind_v and wind speed.
# 
# 2. make feature using wind_u, wind_v and wind speed. These are calculated by multipling wind speed to wind_u and wind_v.

# ### <u>Way1</u>

# In[20]:


image_reglession_u = []
image_reglession_v = []
image_reglession_speed = []

for i in range(len(gfs_files_par_day)):
    gfs_tmp = gfs_files_par_day[i]
    gldas_tmp = gldas_files_par_day[i]
    array_wind_u = []
    array_wind_v = []
    array_wind_speed = []
    for j in range(len(gfs_tmp)):
        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]
        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]
        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]
        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]

        #fill na by mean
        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))
        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))
        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))
        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))
        
        
        gldas_image = (gldas_image1 + gldas_image2)/2
        
        array_wind_u.append(gfs_image_u)
        array_wind_v.append(gfs_image_v)
        array_wind_speed.append(gldas_image)
        
        image_reglession_u.append(np.nanmean(np.array(array_wind_u), axis=0))
        image_reglession_v.append(np.nanmean(np.array(array_wind_v), axis=0))
        image_reglession_speed.append(np.nanmean(np.array(array_wind_speed), axis=0))
       
image_reglession_u = np.nanmean(np.array(image_reglession_u), axis=0)
image_reglession_v = np.nanmean(np.array(image_reglession_v), axis=0)
image_reglession_speed = np.nanmean(np.array(image_reglession_speed), axis=0)


# In[21]:


sns.heatmap(image_reglession_u.reshape((148, 475)))


# In[22]:


sns.heatmap(image_reglession_v.reshape((148, 475)))


# In[23]:


sns.heatmap(image_reglession_speed.reshape((148, 475)))


# In[24]:


image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')
lon = []
lat = []
NO2 = []
wind_u = []
wind_v = []
wind_speed = []

for i in range(image[:,:,0].shape[0]):
    for j in range(image[:,:,0].shape[1]):
        #print(image[:,:,0][i,j])
        NO2.append(image[:,:,0][i,j])
        lon.append(i)
        lat.append(j)
        wind_u.append(image_reglession_u.reshape((148, 475))[i,j])
        wind_v.append(image_reglession_v.reshape((148, 475))[i,j])
        wind_speed.append(image_reglession_speed.reshape((148, 475))[i,j])
        
NO2 = np.array(NO2)
lon = np.array(lon)
lat = np.array(lat)
wind_u = np.array(wind_u)
wind_v = np.array(wind_v)
wind_spped = np.array(wind_speed)
        
results_wind = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v', 'wind_speed'])
results_wind = pd.DataFrame({
                    'NO2': NO2/max(NO2),
                    'lat': lat/max(lat),
                    'lon': lon/max(lon),
                    'wind_u' : wind_u/(- min(wind_u)),
                    'wind_v' : wind_v/(- min(wind_v)),
                    'wind_speed': wind_speed/max(wind_speed)})


# In[25]:


pred_wind1 = KMeans(n_clusters=11).fit_predict(results_wind)
plt.figure()
sns.heatmap(pred_wind1.reshape((148, 475)))


# In[26]:


overlay_image_on_puerto_rico_df(power_plants_df, pred_wind1.reshape((148, 475)), 8)


# ### <u>Way2</u>

# In[27]:


image_reglession_u = []
image_reglession_v = []

for i in range(len(gfs_files_par_day)):
    gfs_tmp = gfs_files_par_day[i]
    gldas_tmp = gldas_files_par_day[i]
    array_wind_u = []
    array_wind_v = []
    for j in range(len(gfs_tmp)):
        gfs_image_u = tiff.imread(gfs_tmp[j])[:,:,3]
        gfs_image_v = tiff.imread(gfs_tmp[j])[:,:,4]
        gldas_image1 = tiff.imread(gldas_tmp[2*j])[:,:,11]
        gldas_image2 = tiff.imread(gldas_tmp[2*j + 1])[:,:,11]

        #fill na by mean
        gfs_image_u = np.nan_to_num(gfs_image_u, nan=np.nanmean(gfs_image_u))
        gfs_image_v = np.nan_to_num(gfs_image_v, nan=np.nanmean(gfs_image_v))
        gldas_image1 = np.nan_to_num(gldas_image1, nan=np.nanmean(gldas_image1))
        gldas_image2 = np.nan_to_num(gldas_image2, nan=np.nanmean(gldas_image2))
        
        
        gldas_image = (gldas_image1 + gldas_image2)/2
        wind_u = gfs_image_u * gldas_image
        wind_v = gfs_image_v * gldas_image
        
        array_wind_u.append(wind_u)
        array_wind_v.append(wind_v)
        
        image_reglession_u.append(np.nanmean(np.array(array_wind_u), axis=0))
        image_reglession_v.append(np.nanmean(np.array(array_wind_v), axis=0))
       
image_reglession_u = np.nanmean(np.array(image_reglession_u), axis=0)
image_reglession_v = np.nanmean(np.array(image_reglession_v), axis=0)


# In[28]:


sns.heatmap(image_reglession_u.reshape((148, 475)))


# In[29]:


sns.heatmap(image_reglession_v.reshape((148, 475)))


# In[30]:


image = tiff.imread('/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif')
lon = []
lat = []
NO2 = []
wind_u = []
wind_v = []

for i in range(image[:,:,0].shape[0]):
    for j in range(image[:,:,0].shape[1]):
        #print(image[:,:,0][i,j])
        NO2.append(image[:,:,0][i,j])
        lon.append(i)
        lat.append(j)
        wind_u.append(image_reglession_u.reshape((148, 475))[i,j])
        wind_v.append(image_reglession_v.reshape((148, 475))[i,j])
        
NO2 = np.array(NO2)
lon = np.array(lon)
lat = np.array(lat)
wind_u = np.array(wind_u)
wind_v = np.array(wind_v)
        
results_wind = pd.DataFrame(columns=['NO2', 'lat', 'lon', 'wind_u', 'wind_v'])
results_wind = pd.DataFrame({
                    'NO2': NO2/max(NO2),
                    'lat': lat/max(lat),
                    'lon': lon/max(lon),
                    'wind_u' : wind_u/(- min(wind_u)),
                    'wind_v' : wind_v/(- min(wind_v))})



# In[31]:


pred_wind2 = KMeans(n_clusters=11).fit_predict(results_wind)
plt.figure()
sns.heatmap(pred_wind2.reshape((148, 475)))


# In[32]:


overlay_image_on_puerto_rico_df(power_plants_df, pred_wind2.reshape((148, 475)), 8)


# ### Conparison

# In[33]:


plt.figure()
sns.heatmap(pred.reshape((148, 475)))


# In[34]:


plt.figure()
sns.heatmap(pred_wind1.reshape((148, 475)))


# In[35]:


plt.figure()
sns.heatmap(pred_wind2.reshape((148, 475)))


# In[ ]:




