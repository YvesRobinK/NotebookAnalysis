#!/usr/bin/env python
# coding: utf-8

# First simple EDA and feature engineering:
# - distance from one point
# - zip code and adress analysis
# - first try to match localtion - simple but works

# In[1]:


import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

pd.set_option('display.float_format', lambda x: '%.5f' % x)


# In[2]:


train_df = pd.read_csv("../input/foursquare-location-matching/train.csv")
pairs_df = pd.read_csv("../input/foursquare-location-matching/pairs.csv")


# In[3]:


print(f'Train dataset lenght: {len(train_df)}')
print(f'Pairs dataset lenght: {len(pairs_df)}')


# In[4]:


train_df.head(5)


# In[5]:


train_df.nunique()


# In[6]:


train_df[['latitude', 'longitude']].describe()


# In[7]:


def draw_lon_lat(df, world):
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
    gdf = GeoDataFrame(df, geometry=geometry)   
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);


# In[8]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
draw_lon_lat(train_df, world)


# ## NEW FEATURES
# 
# ### DISTANCE
# 
# Let's make some test to find closest location to each other:
# 1. Calculate distance from "center" of Earth ;) lon: 0, lat: 0
# 2. Find some closest representation

# In[9]:


from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371* c
    return km


# In[10]:


train_df['distance'] = [haversine(0,0,train_df.longitude[i],train_df.latitude[i]) for i in range(len(train_df))]
train_df['distance'] = train_df['distance'].round(decimals=3)

train_df.head()


# In[11]:


# Let's find object 400m far from each other

train_df.query("distance>5000 and distance<5000.4")


# ### ZIP CODE, COUNTRY, ADRESS
# Let's take adress and ZIP code 
# - I tool for only 10 observations only to limit API query 

# In[12]:


# Even this is syntetic data ... zip code is real (I checked it on google)

import geopy
import pandas as pd


def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location


geolocator = geopy.Nominatim(user_agent='1234')

loc = train_df[0:10].apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude')

for idx, lo in enumerate(loc):
    loc_text = loc[idx].address.split(',')
    print(f'{idx} - Country: {loc_text[-1]} - ZIP: {loc_text[-2]} - Adress: {loc[idx].address} - LAT: {loc[idx].latitude} - LON: {loc[idx].longitude} \n')


# ## FIND MATCHING LOCATION - SIMPLE APPROACH
# 
# - take random distance eg. 5000km far from lat:0 long:0
# - take POIs 400m far from location
# - find real zip and adress 

# In[13]:


idx = train_df.query("distance>5000 and distance<5000.4").index
dxx = train_df.loc[idx].reset_index(drop = True)

dxx['new_adress'] = dxx.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='latitude', lon_field='longitude')


# In[14]:


dxx


# In[15]:


def get_adress(df):
    if df.new_adress is not None:
        loc_text = df.new_adress.address.split(',')
        zipcode = loc_text[-2]
        country = loc_text[-1]
    return country, zipcode

dxx[['new_country','new_zip']] = dxx.apply(get_adress, axis=1, result_type="expand")
dxx[['name', 'new_country','new_zip']]


# In[16]:


features = ['id', 'name', 'address', 'new_country','new_zip', 'distance', 'categories']
dxx[features]


# We can see matching location - are they the same?

# In[17]:


features.extend(['latitude','longitude'])
dxx.query('new_zip == " 48450"')[features]


# In[18]:


dxx.query('new_zip == " 50041"')[features]


# In[19]:


dxx.query('new_zip == " 19100"')[features]


# In[20]:


train_df.query('address == "Molo Italia"')


# # ROI

# In[21]:


train_df.query('latitude > 37.085 and latitude < 37.105 and longitude> 27.485 and longitude < 27.505')


# ## PAIRS DATASET

# In[22]:


pairs_df.head(1).transpose()


# In[23]:


train_df.query('id == "E_da7fa3963561f8" or id =="E_000001272c6c5d"')

