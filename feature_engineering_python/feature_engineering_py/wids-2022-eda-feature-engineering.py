#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Part 1: EDA & feature engineering

# 

# ## 1. Introduction
# 
# 
# ### 1.1 Problem statement     
# 
# Climate change has significant impact on individuals, business, and our environment. Thus, it is important that we understand the factors driving energy consumption, be able to predict energy usage and actively manage it to slow down the impact on climate changes.   
# 
# ### 1.2 Goal 
# 
# The goal of this competition is to predict the energy consumption using building characteristics and climate and weather variables .
# 
# ### 1.3 Dataset 
# ``` train.csv``` - the training dataset where the observed values of the Site EUI for each row is provided   
# ```test.csv``` - the test dataset where we withhold the observed values of the Site EUI for each row.    
# ```sample_submission.csv``` - a sample submission file in the correct format   
# 
# **Columns**
# 
# ```id:``` building id    
# 
# ```Year_Factor:``` anonymized year in which the weather and energy usage factors were observed   
# 
# ```State_Factor:``` anonymized state in which the building is located  
# 
# ```building_class:``` building classification   
# 
# ```facility_type:``` building usage type   
# 
# ```floor_area:``` floor area (in square feet) of the building   
# 
# ```year_built:``` year in which the building was constructed   
# 
# ```energy_star_rating:``` the energy star rating of the building   
# 
# ```ELEVATION:``` elevation of the building location   
# 
# ```january_min_temp:``` minimum temperature in January (in Fahrenheit) at the location of the building  
# 
# ```january_avg_temp:``` average temperature in January (in Fahrenheit) at the location of the building   
# 
# ```january_max_temp:``` maximum temperature in January (in Fahrenheit) at the location of the building   
# 
# ```cooling_degree_days:``` cooling degree day for a given day is the number of degrees where the daily average temperature exceeds 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.   
# 
# ```heating_degree_days:``` heating degree day for a given day is the number of degrees where the daily average temperature falls under 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.   
# 
# ```precipitation_inches:``` annual precipitation in inches at the location of the building    
# 
# ```snowfall_inches:``` annual snowfall in inches at the location of the building   
# 
# ```snowdepth_inches:``` annual snow depth in inches at the location of the building  
# 
# ```avg_temp:``` average temperature over a year at the location of the building  
# 
# ```days_below_30F:``` total number of days below 30 degrees Fahrenheit at the location of the building  
# 
# ```days_below_20F:``` total number of days below 20 degrees Fahrenheit at the location of the building  
# 
# ```days_below_10F:``` total number of days below 10 degrees Fahrenheit at the location of the building  
# 
# ```days_below_0F:``` total number of days below 0 degrees Fahrenheit at the location of the building  
# 
# ```days_above_80F:``` total number of days above 80 degrees Fahrenheit at the location of the building   
# 
# ```days_above_90F:``` total number of days above 90 degrees Fahrenheit at the location of the building   
# 
# ```days_above_100F:``` total number of days above 100 degrees Fahrenheit at the location of the building   
# 
# ```days_above_110F:``` total number of days above 110 degrees Fahrenheit at the location of the building   
# 
# ```direction_max_wind_speed:``` wind direction for maximum wind speed at the location of the building. Given in 360-degree compass point directions (e.g. 360 = north, 180 = south, etc.).   
# 
# ```direction_peak_wind_speed:``` wind direction for peak wind gust speed at the location of the building. Given in 360-degree compass point directions (e.g. 360 = north, 180 = south, etc.).   
# 
# ```max_wind_speed:``` maximum wind speed at the location of the building   
# 
# ```days_with_fog:``` number of days with fog at the location of the building       
# 
# 
# 
# Note:     
# EUI is expressed as energy per square foot per year. It’s calculated by dividing the total energy consumed by the building in one year (measured in kBtu or GJ) by the total gross floor area of the building (measured in square feet or square meters). (Energy star, 2021)    
# 
# 
# ### 1.4 Addition of new features       
# 
# Our new features are built on the following hypothesis:    
# 1. Older buildings are less energy savvy due to their poor conditions    
# 2. Buildings with larger floor area has lesser ability to retain heat, thus lower EUI    
# 3. Residential buildings has higher EUI compared to commercial buildings which are operational only during working hours    
# In the next sections, we will perform detailed EDA to test out the hypothesis and fine tune the potential features we will use for the model.    
# 
# **New feature description**  
# 1. Age of building       
# The general intuition is that the older buildings, especially those which are not as well maintained might have poorer energy saving features. Thus, we will create this feature to examine if there is any likely relationship between energy consumption and the age of the building.  
# 
#      

# In[2]:


# Import library
import os
import gc
import copy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold ,RepeatedKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
plt.style.use('ggplot')
import seaborn as sns
from scipy import stats

import shap

from sklearn.preprocessing import StandardScaler

import optuna
import optuna.integration.lightgbm as lgbm

import warnings
warnings.filterwarnings('ignore')

import wandb
os.environ["WANDB_SILENT"] = "true"

#pd.set_option('display.max_columns', 100)
#pd.set_option('display.width', 1000)

# Others
get_ipython().run_line_magic('matplotlib', 'inline')
#cmap=sns.color_palette('Blues_r')
cmap=sns.color_palette("Spectral")


# In[3]:


# Import data
train=pd.read_csv('/kaggle/input/widsdatathon2022/train.csv')
test=pd.read_csv('/kaggle/input/widsdatathon2022/test.csv')
print("No. of train samples are",train.shape)
print("No. of test samples are",test.shape)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


# Age of building 
def age(df):
    if type(df['year_built']) == float and pd.isna(df['year_built']):
        return 0
    else:
        return (2022 - df['year_built'])


# In[7]:


# Create new features
train['age'] = train.apply(age, axis=1)
test['age'] = test.apply(age, axis=1)


# ## 2. Descriptive statistics   

# In[8]:


# List all numeric attributes
num = list(train.select_dtypes(include=[np.number]).columns)
print(f'There are {len(num)} numerical columns in the dataset')
num


# In[9]:


# List all non-numeric attributes
non_num = list(train.select_dtypes(exclude=[np.number]).columns)
print(f'There are {len(non_num)} categorial columns in the dataset')
non_num


# ### 2.1 Missing values

# In[10]:


tmp = pd.DataFrame((train.isna().mean() * 100), columns=['Percentage of Nan values'])
tmp[tmp['Percentage of Nan values']>0]


# From the quick scan above, there are missing values in some of the columns. We will study the distribution of each feature in the next section before we decide on logical values to use as replacements.   
# 
# Also, we must be cautious that there might be outliers which might negatively skew our data/ results.
# We will study this further in the next section.

# ### 2.2 Numerical data   
# 
# In this section, we study the distribution of the numerical features in detail.      
# By doing so, we want to detect the presence of missing values, skewed data and outliers/ anomly which might negatively impact out modelling.     
#     
# ### a. Statistics

# In[11]:


# Numerical features
train[num].describe()


# From the data above, we see that:    
# - Months with lowest average temperature: Jan & Feb    
# This is expected since these are the cooler winter months
# - Months with highest average temperature: Jul & Aug    
# This is expected since these are the hotter summer months

# ### b. Distribution   
# Firstly, we will temporarily replace these missing values with the mean and review the distribution in the next section before we decide on logical values to use as replacements. 

# In[12]:


train['year_built'].min()


# In[13]:


train['year_built'].max()


# In[14]:


# Train - replace nan mean
train['year_built'] = train['year_built'].fillna(train['year_built'].mean())
train['energy_star_rating'] = train['energy_star_rating'].fillna(train['energy_star_rating'].mean())
train['direction_max_wind_speed'] = train['direction_max_wind_speed'].fillna(train['direction_max_wind_speed'].mean())
train['direction_peak_wind_speed'] = train['direction_peak_wind_speed'].fillna(train['direction_peak_wind_speed'].mean())
train['max_wind_speed'] = train['max_wind_speed'].fillna(train['max_wind_speed'].mean())
train['days_with_fog'] = train['days_with_fog'].fillna(train['days_with_fog'].mean())


# In[15]:


# Test - replace nan mean
test['year_built'] = test['year_built'].fillna(test['year_built'].mean())
test['energy_star_rating'] = test['energy_star_rating'].fillna(test['energy_star_rating'].mean())
test['direction_max_wind_speed'] = test['direction_max_wind_speed'].fillna(test['direction_max_wind_speed'].mean())
test['direction_peak_wind_speed'] = test['direction_peak_wind_speed'].fillna(test['direction_peak_wind_speed'].mean())
test['max_wind_speed'] = test['max_wind_speed'].fillna(test['max_wind_speed'].mean())
test['days_with_fog'] = test['days_with_fog'].fillna(test['days_with_fog'].mean())


# In[16]:


# Check distribution (num)
plt.figure(figsize=(15, 80))
for i, col in enumerate(num):
    # Plot distribution 
    plt.subplot(32,2,i+1); sns.distplot(train[col], color='blue')
    plt.title(f'Distribution of {col}')
# Show the plot
plt.tight_layout()
plt.show()


# From the distribution plots above, we observed the following:   
# 
# - 30% of the buildings are built in the 1920s    
# Nearly 30% of the houses in the dataset are built in the 1920s i.e. >80-year-old. This coincides with the historic housing boom after World War II. Such old buildings, if not upgraded, are likely to be less energy savvy. Thus, we will study further if there's any relationship between the age of the house and the EUI.  
# 
# 
# - Most of the buildings reside in subtropical climate - Temperatures are mostly low with short summers    
# The average temperatures are concentrated around the lower spectrum of 50F-60F. In addition, there are hardly days above 100F, the hotter days are above 90F but typically do not last for >25 days. On the other hand, the colder months are typically between 20-30F. Thus. it is likely that the data are mostly collected from buildings in the sub-tropical climate. (NWS, 2022)
# 
# 
# - Huge proportion of houses with no energy star rating     
# Many buildings (approx. 30%) do not have energy star rating. Such building might not meet the minimum standards to be granted a rating, thus, we will delve deeper into these building to determine if EUI are higher in these building. 
# 
# 
# - Huge proportion of houses with no wind & fog data    
# Most of the buildings do not have wind & fog data. (>50%). Thus, in order not to skew the dataset, we will replace the missing values with the average wind/ fog value.    
# 
# 
# - Most numerical features in the dataset are highly skewed     
# This can pose a problem when modelling as most machine learning models require features to be normally distributed.
# Hence, we will keep this observation in mind and possibly look at techniques such as applying log to these numerical data before using them as features for the models.   
# 

# ### c. Relationship between numerical features (all)   

# In[17]:


def plt_corr(df, figsize ):
    # Create the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle 
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # Add the mask to the heatmap
    plt.figure(figsize=(figsize))
    sns.set(font_scale=1)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, fmt=".2f")

    plt.title('Correlation between numerical features')
    plt.show()


# In[18]:


figsize=(100,150)
plt_corr(train[num], figsize)


# As expected, the temperature related features are highly correlated. e.g. average temperature and days below/ above a given temperature.          
# 
# Due to the large number of features (>60), we will next do more detailed analysis on the smaller subsets of these features to gain better insights.    
# 

# ### d. Relationship between numerical features (non-temperature related features)      
# 
# From the previous section, we see the temperature related features are highly correlated as expected.    
# In this section, we focus on the non-temperature related features to gain better insights.    

# In[19]:


num_nontemp = ['Year_Factor',
 'floor_area',
 'year_built',
 'energy_star_rating',
 'ELEVATION',
 'cooling_degree_days',
 'heating_degree_days',
 'precipitation_inches',
 'snowfall_inches',
 'snowdepth_inches',
 'avg_temp',
 'days_below_30F',
 'days_below_20F',
 'days_below_10F',
 'days_below_0F',
 'days_above_80F',
 'days_above_90F',
 'days_above_100F',
 'days_above_110F',
 'direction_max_wind_speed',
 'direction_peak_wind_speed',
 'max_wind_speed',
 'days_with_fog',
 'age',
 'site_eui']


# In[20]:


figsize=(18,18)
plt_corr(train[num_nontemp], figsize)


# From the correlation heatmap above, we see that:    
# - Lower EUI on buildings with more hotter days, but higher EUI for building with more cooler days    
# Since temperature >=90F are summer temperatures, thus, it makes sense that the more summer days there are, the lower the energy consumed since no building heating is required. On the other hand, for temperatures <=80F, the more the number of days of such colder days, the higher the energy consumed possibly for heating the building. This aligns with our research which shows that generally, households spend >50% of the energy consumption on space & water heating and <10% on air conditioning (EIA, 2015). This also aligns with our previous findings that most of the buildings reside in sub-tropical climates, thus, it is likely that less air conditioning and thus lower EUI are needed even during summer.   
# 
# 
# - Lower EUI on buildings with higher energy star rating    
# When the building has high energy star rating, the energy consumed is lower. This is expected since buildings with high energy star ratings likely have more eco-friendly installations (e.g. energy saving heating), thus, lower energy consumed. 
# 
# 
# - Older buildings have lower EUI, higher energy star ratings          
# This goes against our intuition that older buildings might not be equipped with energy saving facilities, consuming more energy. This might be likely because such old buildings have been actively upgraded with better energy savvy features as seen in the positive correlation with the energy star ratings.     
# 
# 
# - Buildings with larger floor area have higher EUI     
# Buildings with larger floor area have higher EUI (energy per square foot per year, Energy star, 2021). This is likely because reduced surface area results in lesser heat loss/ gain, thus requiring lesser energy for heating, cooling, and illumination (Izzet, 2016).  
# 

# In[21]:


def plt_matrix(df, figsize):
    # Create the correlation matrix
    #corr = df.corr()

    # Generate a mask for the upper triangle 
    #mask = np.triu(np.ones_like(corr, dtype=bool))


    # Add the mask to the heatmap
    plt.figure(figsize=(figsize))
    sns.pairplot(df)

    plt.title('Scatter plot matrix')
    plt.show()


# ### 2.2 Categorial data
# 
# ### a. Statistics

# In[22]:


# Categorial features 
train[non_num].describe()


# From the statistics above, most of the buildings are residential, multi-family properties residing in state 6.      
# 
# ### b. Distribution

# In[23]:


# Check distribution (non-num)
plt.figure(figsize=(15, 27))
# Plot distribution 
for i, col in enumerate(non_num):    
    if col == 'day' or col == 'month':
        order = None
    else: 
        order = train[col].value_counts().index   
    plt.subplot(3,1,i+1); sns.countplot(train[col],palette=cmap, order = order ) 
    plt.title(f'Distribution of {col}')
    plt.ylabel('No. of buildings')
    plt.xticks(rotation=90)
# Show the plot
plt.tight_layout()
plt.show()


# According to our research (US department of energy, 2008), residential buildings generally consume more energy than commercial buildings. Thus, we will keep this observation in mind when studying the relationship against the target variable.
# 

# In[24]:


# Check distribution (num)
plt.figure(figsize=(10, 30))
for i, col in enumerate(non_num):
    # Plot distribution    
    plt.subplot(3,1,1+i); sns.violinplot(x='site_eui', y=col, data=train, inner=None, color='lightgray');sns.stripplot(x='site_eui', y=col, data=train, size=0.8,jitter=True);
    plt.title(f'EUI');
    plt.ylabel(f'{col}')    
# Show the plot
plt.tight_layout()
plt.show()


# In[25]:


avg_comm_eui = round(train[train['building_class']=='Commercial']['site_eui'].mean(),2)
print(f'Commercial building has average EUI at {avg_comm_eui}')


# In[26]:


avg_resid_eui = round(train[train['building_class']=='Residential']['site_eui'].mean(),2)
print(f'Residential building has average EUI at {avg_resid_eui}')


# 

# Overall, we observed the following from the categorial features:    
# 
# - Commercial building have higher EUI compared to Residential buildings    
# Contrary to our literature research, commercial buildings have higher average EUI compared to residential building . We will keep this in view and investigate the trend further in the next section.     
# 
# 
# - Large spread & more outlier EUI for commercial buildings    
# Also, we see from the plot that there is a larger spread of commercial buildings at the higher end of EUI (>200). These outliers are likely due to the facilities which operate intensively 24/7 e.g. Data centres, health care etc. We will attempt to confirm this observation in the next section.         
# 
# ### 2.3 Target variable 
# 

# In[27]:


# Calc conversion rate per group
def calc_EUI(dataframe, column_names=None):
    #print('test')
    if column_names != None:
        # Calc mean EUI
        #print('test')
        mean_EUI = dataframe.groupby(column_names)['site_eui'].mean()  
        #print(mean_EUI)
        # Fill missing values with 0
        mean_EUI = mean_EUI.fillna(0) 
    else:       
        # Conversion rate 
        mean_EUI = dataframe['site_eui'].mean()  

    return round(mean_EUI,2)


# In[28]:


original_EUI = calc_EUI(train)
print(f'The average EUI for the full dataset is : {original_EUI}')


# #### a. Target variable vs numeric features
# In this section, we try to understand if there is any trend between the numeric feature vs EUI.   

# In[29]:


# Check distribution (num)
plt.figure(figsize=(15, 66), dpi=80)

for i, col in enumerate(num):
    # Calc conv rate 
    eui = calc_EUI(train, [col])
    #print(eui)
    plt.subplot(22,3,i+1);plt.plot(eui.index, eui);plt.ylabel('EUI') 
    plt.title(f'EUI vs {col}')
    # plot the mean 
    plt.axhline(y=original_EUI, color='orange', linestyle='--', alpha=0.7)
# Show the plot
plt.tight_layout()
plt.show()    


# From above, we see that:    
# - EUI drops significantly when energy star rating <50    
# This is expected since buildings with energy star ratings theoretically have more energy efficient features.    
# 
# - EUI drops when year factor >5   
# It's likely that the energy consciousness of the public had increased significantly over the years. Thus, buildings surveyed in more recent years are likely to be more energy efficient.     
# 
# - EUI decrease with cooling degree days but increases with heating degree days    
# There is a step drop in EUI when cooling degree days >2000 and a step increase in EUI when heating degree days is >4000. Thus, it's likely that prolonged cold weather caused even higher energy consumption. This aligns with our previous observation that higher temperature generally results in lower EUI while lower temperature results in higher EUI.    
# 
# - EUI drop significantly when snow depth <700 inches and snowfall >70 inches     
# Interestingly, when snowfall increases, EUI drops significantly. Based on our research, this is likely due to the insulating property of snow which protects the area below snow from changes in atmospheric temperature.  (NSIDC, 2020)   
# Nevertheless, we also note that at extremely high snow fall, EUI starts trending upwards again.    
# 
# Also, aligned with previous observations, EUI increase sharply when floor area increases.    
# On the other hand, there seem to be a rather flat and sometimes noisy relationship between building age and EUI. It is likely that the older buildings in the dataset might have undergone upgrading to be more energy efficient. Thus, our hypothesis that older building consumes more EUI might not be valid after all.     
# 

# #### b. Target variable vs categorial features
# In this section, we try to understand if there is any trend between the categorial feature vs EUI.   

# In[30]:


# Check distribution (non-num)
plt.figure(figsize=(15, 12), dpi=80)
for i, col in enumerate(non_num):   
    # Calc conv rate 
    eui = calc_EUI(train, [col])   
    # Plot distribution 
    plt.subplot(3,1,i+1);eui.plot(kind='bar', color=cmap );plt.ylabel('EUI') 
    plt.title('EUI per category')
    # plot the mean 
    plt.axhline(y=original_EUI, color='orange', linestyle='--', alpha=0.9)
# Show the plot
plt.tight_layout()
plt.show()    


# In[31]:


pd.DataFrame( train.groupby(['building_class'])['site_eui', 'energy_star_rating'].mean() )


# We see again that the EUI is higher in commercial building than residential buildings.         
# Moreover, EUI is higher in data centres, laboratories, food services and groceries/ food markets. Aligned with our previous hypothesis, EUI are higher in these buildings since they are likely to operate 24/7. We will confirm this by creating a new feature - 'building_class_group' which group all commercial building most likely to operate 24/7 in 1 category 'commercial_24_7'. The other 2 categories will be 'commercial_other' & 'residential'. Our hypothesis is that building under 'commercial_24_7' will have high EUI.        
# 

# In[32]:


def building_class_group(d):
    if (d['building_class'] == 'Residential'):
        return 'Residential'
    elif (d['facility_type'] in ('Data_Center', 'Laboratory', 'Grocery_store_or_food_market', 'Health_Care_Inpatient', 'Health_Care_Uncategorized', 'Health_Care_Outpatient_Uncategorized', 'Food_Service_Restaurant_or_cafeteria')):
        return 'Commercial_24_7'
    else:
        return 'Commercial_others'


# In[33]:


train['building_class_group'] = train.apply(building_class_group, axis=1)
test['building_class_group'] = test.apply(building_class_group, axis=1)
train.head()


# In[34]:


# List all non-numeric attributes
non_num = list(train.select_dtypes(exclude=[np.number]).columns)
print(f'There are {len(non_num)} categorial columns in the dataset')
non_num


# In[35]:


# Check distribution (non-num)
plt.figure(figsize=(15, 20), dpi=80)
for i, col in enumerate(non_num):   
    # Calc conv rate 
    eui = calc_EUI(train, [col])   
    # Plot distribution 
    plt.subplot(4,1,i+1);eui.plot(kind='bar', color=cmap );plt.ylabel('EUI') 
    plt.title('EUI per category')
    # plot the mean 
    plt.axhline(y=original_EUI, color='orange', linestyle='--', alpha=0.9)
# Show the plot
plt.tight_layout()
plt.show() 


# In[36]:


pd.DataFrame( train.groupby(['building_class_group'])['site_eui', 'energy_star_rating'].mean() )
#mean_EUI = dataframe.groupby(column_names)['site_eui'].mean() 


# As expected, the selected facility types buildings which operates 24/7 now has a significantly higher EUI compared to the residential buildings. On the other hand, the normal commercial buildings which likely operate only during weekdays & office hours consume similar EUI as the residential buildings.     

# In[37]:


# Export for further feature selection
train.to_csv('train_new1.csv', encoding='utf-8', header=True) 
test.to_csv('test_new1.csv', encoding='utf-8', header=True) 


# ## 3. Summary   
# 
# While performing the EDA, we noticed some consistent trends and fine-tuned some new features.    
# In general, we saw that EUI exhibits distinctive relationships with the following features:     
# 1. energy_star_rating     
# 2. building_class_group     
# 3. Year_Factor    
# 4. floor_area      
# 5. cooling_degree_days   
# 6. heating_degree_days   
# 7. snowfall_inches   
# 8. snowdepth_inches   
# 9. avg_temp    
# 
# On the other hand, features like building's age does not seem to have a strong relationship with EUI.    
# We will deep delve into these observations in the section 'Feature Selection'.    
# 

# ## Reference
# 1. US department of energy (2008), Energy Efficiency Trends in Residential and Commercial Buildings. Available from:  https://www1.eere.energy.gov/buildings/publications/pdfs/corporate/bt_stateindustry.pdf [Accessed on 6 Jan 2022]    
# 2. Energy star (2021), Does building age affect energy use? Available from: https://energystar-mesa.force.com/PortfolioManager/s/article/Does-building-age-affect-energy-use-1600088543994 [Accessed on 7 Jan 2022]
# 3. EIA, US energy information adminstration (2015), Use of energy explained. Available from: https://www.eia.gov/energyexplained/use-of-energy/homes.php [Accessed 8 Jan 2022]    
# 4. NWS, National Weather Service (2022), Climate zones. Available from: https://www.weather.gov/jetstream/climates [Accessed 9 Jan 2022]   
# 5. Energy star (2022), What is Energy Use Intensity (EUI)? Available from: https://www.energystar.gov/buildings/benchmark/understand_metrics/what_eui [Accessed on 10 Jan 2022]    
# 6. Izzet Yüksek and Tülay Tikansak Karadayi, Energy-Efficient Building Design in the Context of Building Life Cycle. Available from: https://www.intechopen.com/chapters/53557. [Accessed on 13 Jan 2022]       
# 7. NSICD (2020), Snow and Climate. Available from: https://nsidc.org/cryosphere/snow/climate.html. [Accessed on 15 Jan 2022]
