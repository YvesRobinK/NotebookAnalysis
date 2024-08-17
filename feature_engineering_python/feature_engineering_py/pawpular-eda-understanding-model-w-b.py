#!/usr/bin/env python
# coding: utf-8

# <h1><center>Pawpular : InDepth EDA + Understanding + Model + W&B</center></h1>
#                                                       
# <center><img src = "https://www.petfinder.my/images/logo-575x100.png" width = "750" height = "500"/></center>                                                                                               

# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Contents</center></h2>

# 1. [Competition Overview](#competition-overview)  
# 2. [Libraries](#libraries)  
# 3. [Weights and Biases](#weights-and-biases)   
# 4.[Global Config](#global-config)
# 5. [Load Datasets](#load-datasets)  
# 6. [Tabular Exploration](#tabular-exploration)  
# 7. [Distribution Plots](#distribution-plots)
# 8. [Feature Wise Analysis](#feature-wise-analysis)
# 9. [Pawpularity Score Wise Images](#pawpularity-score-wise-images)  
# 10. [YOLO V5 Object Detection](#yolo-v5-object-detection)
# 10. [Dataset and Augmentations](#dataset-and-augmentations)
# 11. [Efficientnet Model and Understanding](#efficientnet-model-and-understanding)  
# 12. [WandB Dashboard](#wandb-dashboard)  
# 13. [References](#references)

# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list" style='background:maroon; border:0; color:white' role="tab" aria-controls="home"><center>If you find this notebook useful, do give me an upvote, it helps to keep up my motivation. This notebook will be updated frequently so keep checking for furthur developments.</center></h3>

# <a id="competition-overview"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Competition Overview</center></h2>

# ## **<span style="color:orange;">Description</span>**
# 
# In this competition, you’ll analyze raw images and metadata to predict the “Pawpularity” of pet photos.   
#   
# You'll train and test your model on PetFinder.my's thousands of pet profiles. Winning versions will offer accurate recommendations that will improve animal welfare.
#   
# If successful, your solution will be adapted into AI tools that will guide shelters and rescuers around the world to improve the appeal of their pet profiles, automatically enhancing photo quality and recommending composition improvements.   
#   
# As a result, stray dogs and cats can find their "furever" homes much faster. With a little assistance from the Kaggle community, many precious lives could be saved and more happy families created.
# 
# ---

# ## **<span style="color:orange;">Evaluation Criteria</span>**
# 
# Submissions are scored on the **Root mean squared error**. 
#   
# - Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
# - Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. 
# - In other words, it tells you how concentrated the data is around the line of best fit. - Root mean square error is commonly used in climatology, forecasting, and regression analysis to verify experimental results.
# 
# **<span style="color:orange;">Resources to learn and understand RMSE:</span>**
# - [Root-mean-square deviation](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
# - [Root-Mean-Squared Error](https://www.sciencedirect.com/topics/engineering/root-mean-squared-error)
# - [What does RMSE really mean?](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)
# - [
# Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)](http://www.eumetrain.org/data/4/451/english/msg/ver_cont_var/uos3/uos3_ko1.htm)
# - [What is Root Mean Square Error (RMSE)?](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/)

# <a id="libraries"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Libraries</center></h2>

# In[2]:


get_ipython().run_cell_magic('sh', '', 'pip install -q pytorch-lightning==1.1.8\npip install -q timm\npip install -q albumentations\npip install -q --upgrade wandb\n')


# In[4]:


import gc
import os
import glob
import sys
import cv2
import imageio
import joblib
import math
import random
import wandb
import math

import numpy as np
import pandas as pd

from scipy.stats import kstest

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.graphics.gofplots import qqplot

plt.rcParams.update({'font.size': 18})
plt.style.use('fivethirtyeight')

import seaborn as sns
import matplotlib

from termcolor import colored

from multiprocessing import cpu_count
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

import timm
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2

import warnings
warnings.simplefilter('ignore')

# Activate pandas progress apply bar
tqdm.pandas()


# In[3]:


# Wandb Login
import wandb
wandb.login()


# <a id="weights-and-biases"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Weights and Biases</center></h2>

# <center><img src = "https://i.imgur.com/1sm6x8P.png" width = "750" height = "500"/></center>  

# **Weights & Biases** is the machine learning platform for developers to build better models faster. 
# 
# You can use W&B's lightweight, interoperable tools to 
# - quickly track experiments, 
# - version and iterate on datasets, 
# - evaluate model performance, 
# - reproduce models, 
# - visualize results and spot regressions, 
# - and share findings with colleagues. 
# 
# Set up W&B in 5 minutes, then quickly iterate on your machine learning pipeline with the confidence that your datasets and models are tracked and versioned in a reliable system of record.
# 
# In this notebook I will use Weights and Biases's amazing features to perform wonderful visualizations and logging seamlessly. 

# <a id="global-config"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Global Config</center></h2>

# In[5]:


class config:
    DIRECTORY_PATH = "../input/petfinder-pawpularity-score"
    TRAIN_CSV_PATH = DIRECTORY_PATH + "/train.csv"
    TEST_CSV_PATH = DIRECTORY_PATH + "/test.csv"
    
    SEED = 42
    
Config = dict(
    NFOLDS = 5,
    EPOCHS = 1,
    LR = 2e-4,
    IMG_SIZE = (224, 224),
    MODEL_NAME = 'tf_efficientnet_b6_ns',
    DR_RATE = 0.35,
    NUM_LABELS = 1,
    TRAIN_BS = 32,
    VALID_BS = 16,
    min_lr = 1e-6,
    T_max = 20,
    T_0 = 25,
    NUM_WORKERS = 4,
    infra = "Kaggle",
    competition = 'petfinder',
    _wandb_kernel = 'neuracort',
    wandb = False
)


# In[5]:


# wandb config
WANDB_CONFIG = {
     'competition': 'PetFinder', 
              '_wandb_kernel': 'neuracort'
    }


# In[6]:


wandb_logger = WandbLogger(project='pytorchlightning', group='vision', job_type='train', 
                           anonymous='allow', config=Config)


# In[7]:


def set_seed(seed=config.SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed()


# <a id="load-datasets"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Load Datasets</center></h2>

# ## **<span style="color:orange;">Understanding the Structure of the Dataset</span>**

# > ### **<span style="color:orange;">Goal of Competition</span>**
# > In this competition, your task is to predict engagement with a pet's profile based on the photograph for that profile. You are also provided with hand-labelled metadata for each photo. The dataset for this competition therefore comprises both images and tabular data.
# > 
# > ### **<span style="color:orange;">How Pawpularity Score Is Derived</span>**
# > The Pawpularity Score is derived from each pet profile's page view statistics at the listing pages, using an algorithm that normalizes the traffic data across different pages, platforms (web & mobile) and various metrics.  
# > Duplicate clicks, crawler bot accesses and sponsored profiles are excluded from the analysis.
# >
# >---

# > ### **<span style="color:orange;">Training Data</span>**
# > - train/ - Folder containing training set photos of the form {id}.jpg, where {id} is a unique Pet Profile ID.
# > - train.csv - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.
# > 
# > ### **<span style="color:orange;">Example Test Data</span>**
# > In addition to the training data, we include some randomly generated example test data to help you author submission code. When your submitted notebook is scored, this example data will be replaced by the actual test data (including the sample submission).
# > 
# > - test/ - Folder containing randomly generated images in a format similar to the training set photos. The actual test data comprises about 6800 pet photos similar to the training set photos.
# > - test.csv - Randomly generated metadata similar to the training set metadata.
# > - sample_submission.csv - A sample submission file in the correct format.
# >
# >---
# 
# > ### **<span style="color:orange;">Photo Metadata</span>**
# > The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:
# > 
# > - Focus - Pet stands out against uncluttered background, not too close / far.
# > - Eyes - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
# > - Face - Decently clear face, facing front or near-front.
# > Near - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
# > - Action - Pet in the middle of an action (e.g., jumping).
# > - Accessory - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
# > - Group - More than 1 pet in the photo.
# > - Collage - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
# > - Human - Human in the photo.
# > - Occlusion - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
# > - Info - Custom-added text or labels (i.e. pet name, description).
# > - Blur - Noticeably out of focus or noisy, especially for the pet’s eyes and face. For Blur entries, “Eyes” column is always set to 0.
# >
# >---

# In[6]:


# Efficient Data Types
dtype = {
    'Id': 'string',
    'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
    'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
    'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
    'Pawpularity': np.uint8,
}

train = pd.read_csv(config.TRAIN_CSV_PATH, dtype=dtype)
test = pd.read_csv(config.TEST_CSV_PATH, dtype=dtype)


# <a id="tabular-exploration"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Tabular Exploration</center></h2>

# ## **<span style="color:orange;">Train Head</span>**

# In[9]:


train.head()


# ## **<span style="color:orange;">Test Head</span>**

# In[10]:


test.head()


# ### **<span style="color:orange;">Train Info</span>**

# In[11]:


train.info()


# ## **<span style="color:orange;">Test Info</span>**

# In[12]:


test.info()


# ## **<span style="color:orange;">Dataset Size</span>**

# In[13]:


print(f"Training Dataset Shape: {colored(train.shape, 'yellow')}")
print(f"Test Dataset Shape: {colored(test.shape, 'yellow')}")


# <a id="distribution-plots"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Distribution Plots</center></h2>

# In[7]:


# Add File path to Train
def get_image_file_path(image_id):
    return f'/kaggle/input/petfinder-pawpularity-score/train/{image_id}.jpg'

train['file_path'] = train['Id'].apply(get_image_file_path)


# In[15]:


widths = []
heights = []
ratios = []
for file_path in tqdm(train['file_path']):
    image = imageio.imread(file_path)
    h, w, _ = image.shape
    heights.append(h)
    widths.append(w)
    ratios.append(w / h)


# ## **<span style="color:orange;"> Images Height and Width Distribution</span>**

# In[16]:


# Images Height and Width Distribution
print(colored('Width Statistics', 'yellow'))
display(pd.Series(widths).describe())
print()

print(colored('Height Statistics', 'yellow'))
display(pd.Series(heights).describe())
print()

plt.figure(figsize=(15,8))
plt.title(f'Images Height and Width Distribution', size=24)
plt.hist(heights, bins=32, label='Image Heights')
plt.hist(widths, bins=32, label='Image Widths')
plt.legend(prop={'size': 16})
plt.show()


# The image width to height ratio have a mean below zero and a peak on 0.75, pictures thus tend to be taken vertically, not horizontally.

# ## **<span style="color:orange;"> Images Ratio Distribution</span>**

# In[17]:


# Images Ratio Distribution
print(colored('Ratio Statistics', 'yellow'))
display(pd.Series(ratios).describe())
print()

plt.figure(figsize=(15,8))
plt.title(f'Images Ratio Distribution', size=24)
plt.hist(ratios, bins=16, label='Image Heights')
plt.legend(prop={'size': 16})
plt.show()


# ## **<span style="color:orange;"> Pawpularity Score Distribution</span>**
# The pawpularity score is centered around 40 and has a peak on 0 and 100.

# In[18]:


# Pawpularity Score Distribution
print(colored('Pawpularity Statistics', 'yellow'))
display(train['Pawpularity'].describe())
print()

plt.figure(figsize=(15,8))
plt.title('Train Data Pawpularity Score Distribution', size=24)
plt.hist(train['Pawpularity'], bins=32)
plt.show()


# ## **<span style="color:orange;">Quantile-Quantile plot of Pawpularity distribution</span>**

# In[19]:


fig = plt.figure()
qqplot(train['Pawpularity'], line='s')
plt.title('Quantile-Quantile plot of Pawpularity distribution', 
          fontsize=20, fontweight='bold')
plt.show()


# We notice the deviation at this QQPlot which seems to indicate a non-Gaussian distribution. We will check with the Kolmogorov-Smirnov test (Shapiro-Wilks is not suitable for a dataset greater than 5000 items).

# In[20]:


# Kolmogorov-Smirnov test with Scipy
stat, p = kstest(train['Pawpularity'],'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print(f'Sample looks Gaussian (fail to reject H0 at {int(alpha*100)}% test level)')
else:
    print(f'Sample does not look Gaussian (reject H0 at {int(alpha*100)}% test level)')


# The test clearly indicates that the distribution does not follow a Gaussian law. It will therefore be important to normalize the data according to the modeling chosen.

# <a id="feature-wise-analysis"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Feature Wise Analysis</center></h2>

# ## **<span style="color:orange;">Subject Focus</span>**

# In[21]:


sns.boxplot(data=train, x='Subject Focus', y='Pawpularity')
plt.show()


# In[22]:


sns.histplot(train, x="Pawpularity", hue="Subject Focus", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Eyes</span>**
# 

# In[23]:


sns.boxplot(data=train, x='Eyes', y='Pawpularity')
plt.show()


# In[24]:


sns.histplot(train, x="Pawpularity", hue="Eyes", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Face</span>**

# In[25]:


sns.boxplot(data=train, x='Face', y='Pawpularity')
plt.show()


# In[26]:


sns.histplot(train, x="Pawpularity", hue="Face", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Near</span>**

# In[27]:


sns.boxplot(data=train, x='Near', y='Pawpularity')
plt.show()


# In[28]:


sns.histplot(train, x="Pawpularity", hue="Near", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Action</span>**

# In[29]:


sns.boxplot(data=train, x='Action', y='Pawpularity')
plt.show()


# In[30]:


sns.histplot(train, x="Pawpularity", hue="Action", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Accessory</span>**

# In[31]:


sns.boxplot(data=train, x='Accessory', y='Pawpularity')
plt.show()


# In[32]:


sns.histplot(train, x="Pawpularity", hue="Accessory", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Group</span>**

# In[33]:


sns.boxplot(data=train, x='Group', y='Pawpularity')
plt.show()


# In[34]:


sns.histplot(train, x="Pawpularity", hue="Group", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Collage</span>**

# In[35]:


sns.boxplot(data=train, x='Collage', y='Pawpularity')
plt.show()


# In[36]:


sns.histplot(train, x="Pawpularity", hue="Collage", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Human</span>**

# In[37]:


sns.boxplot(data=train, x='Human', y='Pawpularity')
plt.show()


# In[38]:


sns.histplot(train, x="Pawpularity", hue="Human", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Occlusion</span>**

# In[39]:


sns.boxplot(data=train, x='Occlusion', y='Pawpularity')
plt.show()


# In[40]:


sns.histplot(train, x="Pawpularity", hue="Occlusion", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Info</span>**

# In[41]:


sns.boxplot(data=train, x='Info', y='Pawpularity')
plt.show()


# In[42]:


sns.histplot(train, x="Pawpularity", hue="Info", kde=True)
plt.show()


# ---

# ## **<span style="color:orange;">Blur</span>**

# In[43]:


sns.boxplot(data=train, x='Blur', y='Pawpularity')
plt.show()


# In[44]:


sns.histplot(train, x="Pawpularity", hue="Blur", kde=True)
plt.show()


# ---

# <a id="pawpularity-score-wise-images"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Pawpularity Score Wise Images</center></h2>

# In[45]:


def pawpularity_pics(df, num_images, desired_pawpularity, random_state):
    
    '''The pawpularity_pics() function accepts 4 parameters: df is a dataframe, 
    num_images is the number of images you want displayed, desired_pawpularity 
    is the pawpularity score of pics you want to see, and random state ensures reproducibility.'''
    
    #how many images to display
    num_images = num_images
    
    #set the rample state for the sampling for reproducibility
    random_state = random_state
    
    #filter the train_df on the desired_pawpularity and use .sample() to get a sample
    random_sample = df[df["Pawpularity"] == desired_pawpularity].sample(num_images, random_state=random_state).reset_index(drop=True)
    
    #The for loop goes as many loops as specified by the num_images
    for x in range(num_images):
        #start from the id in the dataframe
        image_path_stem = random_sample.iloc[x]['Id']
        root = '../input/petfinder-pawpularity-score/train/'
        extension = '.jpg'
        image_path = root + str(image_path_stem) + extension
         
        #get the pawpularity to confirm it worked
        pawpularity_by_id = random_sample.iloc[x]['Pawpularity']
    
        #use plt.imread() to read in the image file
        image_array = plt.imread(image_path)
        
        #make a subplot space that is 1 down and num_images across
        plt.subplot(1, num_images, x+1)
        #title is the pawpularity score from the id
        title = pawpularity_by_id
        plt.title(title) 
        #turn off gridlines
        plt.axis('off')
        
        #then plt.imshow() can display it for you
        plt.imshow(image_array)

    plt.show()
    plt.close()


# ## **<span style="color:orange;">Pawpularity = 10</span>**

# In[46]:


df = train
num_images = 5
desired_pawpularity = 10
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 20</span>**

# In[47]:


df = train
num_images = 5
desired_pawpularity = 20
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 30</span>**

# In[48]:


df = train
num_images = 5
desired_pawpularity = 30
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 40</span>**

# In[49]:


df = train
num_images = 5
desired_pawpularity = 40
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 50</span>**

# In[50]:


df = train
num_images = 5
desired_pawpularity = 50
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 60</span>**

# In[51]:


df = train
num_images = 5
desired_pawpularity = 60
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 70</span>**

# In[52]:


df = train
num_images = 5
desired_pawpularity = 70
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 80</span>**

# In[53]:


df = train
num_images = 5
desired_pawpularity = 80
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 90</span>**

# In[54]:


df = train
num_images = 5
desired_pawpularity = 90
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ## **<span style="color:orange;">Pawpularity = 100</span>**

# In[55]:


df = train
num_images = 5
desired_pawpularity = 100
random_state = 1
pawpularity_pics(df, num_images, desired_pawpularity, random_state)


# ---

# In[80]:


# Shows a batch of images
def show_batch_df(df, rows=8, cols=4):
    df = df.copy().reset_index()
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            img = imageio.imread(df.loc[idx, 'file_path'])
            axes[r, c].imshow(img)
            axes[r, c].set_title(f'{idx}, label: {df.loc[idx, "Pawpularity"]}')


# ## **<span style="color:orange;">Least Popular Pets</span>**

# In[81]:


show_batch_df(train.sort_values('Pawpularity'))


# ## **<span style="color:orange;">Most Popular Pets</span>**

# In[74]:


show_batch_df(train.sort_values('Pawpularity', ascending=False))


# <a id="yolo-v5-object-detection"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>YOLO V5 Object Detection</center></h2>

# YOLOv5 🚀 is a family of compound-scaled object detection models trained on the COCO dataset, and includes simple functionality for Test Time Augmentation (TTA), model ensembling, hyperparameter evolution, and export to ONNX, CoreML and TFLite.
#   
# [YOLOV5](https://github.com/ultralytics/yolov5) is the fifth iteration of the Yo Only Look Once object detection famility, which is quite controversial as no official paper has been published. It is however freely available, easy to use and scores fairly high in benchmarks.
#   
# Using object detection the images can be classified as either cat or dog, the contours of the pets can be determined and the amount of pets in the images can be counted.
#   
# This object detection can be a source of features and a fundamental tool for preprocessing.|

# ![](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_comparison.png)

# ---

# ![](https://github.com/ultralytics/yolov5/releases/download/v1.0/model_plot.png)

# ---

# In[8]:


# Download YOLOV5 GitHub Repo
get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[9]:


# Load Best Performing YOLOV5X Model
yolov5x6_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')


# In[10]:


# Get Image Info
def get_image_info(file_path, plot=False):
    # Read Image
    image = imageio.imread(file_path)
    h, w, c = image.shape
    
    if plot: # Debug Plots
        fig, ax = plt.subplots(1, 2, figsize=(8,8))
        ax[0].set_title('Pets detected in Image', size=16)
        ax[0].imshow(image)
        
    # Get YOLOV5 results using Test Time Augmentation for better result
    results = yolov5x6_model(image, augment=True)
    
    # Mask for pixels containing pets, initially all set to zero
    pet_pixels = np.zeros(shape=[h, w], dtype=np.uint8)
    
    # Dictionary to Save Image Info
    h, w, _ = image.shape
    image_info = { 
        'n_pets': 0, # Number of pets in the image
        'labels': [], # Label assigned to found objects
        'thresholds': [], # confidence score
        'coords': [], # coordinates of bounding boxes
        'x_min': 0, # minimum x coordinate of pet bounding box
        'x_max': w - 1, # maximum x coordinate of pet bounding box
        'y_min': 0, # minimum y coordinate of pet bounding box
        'y_max': h - 1, # maximum x coordinate of pet bounding box
    }
    
    # Save found pets to draw bounding boxes
    pets_found = []
    
    # Save info for each pet
    for x1, y1, x2, y2, treshold, label in results.xyxy[0].cpu().detach().numpy():
        label = results.names[int(label)]
        if label in ['cat', 'dog']:
            image_info['n_pets'] += 1
            image_info['labels'].append(label)
            image_info['thresholds'].append(treshold)
            image_info['coords'].append(tuple([x1, y1, x2, y2]))
            image_info['x_min'] = max(x1, image_info['x_min'])
            image_info['x_max'] = min(x2, image_info['x_max'])
            image_info['y_min'] = max(y1, image_info['y_min'])
            image_info['y_max'] = min(y2, image_info['y_max'])
            
            # Set pixels containing pets to 1
            pet_pixels[int(y1):int(y2), int(x1):int(x2)] = 1
            
            # Add found pet
            pets_found.append([x1, x2, y1, y2, label])

    if plot:
        for x1, x2, y1, y2, label in pets_found:
            c = 'red' if label == 'dog' else 'blue'
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=c, facecolor='none')
            # Add the patch to the Axes
            ax[0].add_patch(rect)
            ax[0].text(max(25, (x2+x1)/2), max(25, y1-h*0.02), label, c=c, ha='center', size=14)
                
    # Add Pet Ratio in Image
    image_info['pet_ratio'] = pet_pixels.sum() / (h*w)

    if plot:
        # Show pet pixels
        ax[1].set_title('Pixels Containing Pets', size=16)
        ax[1].imshow(pet_pixels)
        plt.show()
        
    return image_info


# <a id="dataset-and-augmentations"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Dataset and Augmentations</center></h2>

# This part of the notebook has been referred from [here](https://www.kaggle.com/heyytanay/train-baseline-torch-lightning-gpu-tpu-w-b/notebook)

# ## **<span style="color:orange;">Lightning Dataset</span>**

# In[56]:


class PetfinderData(Dataset):
    def __init__(self, df, is_test=False, augments=None):
        self.df = df
        self.is_test = is_test
        self.augments = augments
        
        self.images, self.meta_features, self.targets = self._process_df(self.df)
    
    def __getitem__(self, index):
        img = self.images[index]
        meta_feats = self.meta_features[index]
        meta_feats = torch.tensor(meta_feats, dtype=torch.float32)
        
        img = cv2.imread(img)
        img = img[:, :, ::-1]
        img = cv2.resize(img, Config['IMG_SIZE'])
        
        if self.augments:
            img = self.augments(image=img)['image']
        
        if not self.is_test:
            target = torch.tensor(self.targets[index], dtype=torch.float32)
            return img, meta_feats, target
        else:
            return img, meta_feats
    
    def __len__(self):
        return len(self.df)
    
    def _process_df(self, df):
        TRAIN = "../input/petfinder-pawpularity-score/train"
        TEST = "../input/petfinder-pawpularity-score/test"
        
        if not self.is_test:
            df['Id'] = df['Id'].apply(lambda x: os.path.join(TRAIN, x+".jpg"))
        else:
            df['Id'] = df['Id'].apply(lambda x: os.path.join(TEST, x+".jpg"))
            
        meta_features = df.drop(['Id', 'Pawpularity'], axis=1).values
        
        return df['Id'].tolist(), meta_features, df['Pawpularity'].tolist()


# ## **<span style="color:orange;">Augmentations</span>**

# In[57]:


class Augments:
    """
    Contains Train, Validation Augments
    """
    train_augments = Compose([
        Resize(*Config['IMG_SIZE'], p=1.0),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ],p=1.)
    
    valid_augments = Compose([
        Resize(*Config['IMG_SIZE'], p=1.0),
        Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], p=1.)


# <a id="efficientnet-model-and-understanding"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>Efficientnet Model and Understanding</center></h2>

# ## **<span style="color:orange;">What is Scaling?</span>**
# 
# Scaling is generally done to improve the model’s accuracy on a certain task, for example, ImageNet classification. Although sometimes researchers don’t care much about efficient models as the competition is to beat the SOTA, scaling, if done correctly, can also help in improving the efficiency of a model.

# ## **<span style="color:orange;">Types of Scaling for CNNs</span>**
# 
# There are three scaling dimensions of a CNN: 
# 1) **Depth** - Depth simply means how deep the networks is which is equivalent to the number of layers in it.  
# 2) **Width** - Width simply means how wide the network is. One measure of width, for example, is the number of channels in a Conv layer whereas Resolution is simply the image resolution that is being passed to a CNN.  
# 3) **Resolution**     
#   
# The figure below(from the paper itself) will give you a clear idea of what scaling means across different dimensions. We will discuss these in detail as well.
# 
# <center><img src = "https://miro.medium.com/max/875/1*xQCVt1tFWe7XNWVEmC6hGQ.png" width = "750" height = "500"/></center>  
# 
# ---

# ## **<span style="color:orange;">Depth Scaling (d)</span>**
# 
# Scaling a network by depth is the most common way of scaling. Depth can be scaled up as well as scaled down by adding/removing layers respectively. For example, ResNets can be scaled up from ResNet-50 to ResNet-200 as well as they can be scaled down from ResNet-50 to ResNet-18.   
#   
# But why depth scaling? The intuition is that a deeper network can capture richer and more complex features, and generalizes well on new tasks.  
#   
# *Fair enough. Well, let’s make our network 1000 layers deep then? We don’t mind adding extra layers if we have the resources and a chance to improve on this task.*  
#   
# Easier said than done! Theoretically, with more layers, the network performance should improve but practically it doesn’t follow. Vanishing gradients is one of the most common problems that arises as we go deep.   
#   
# Even if you avoid the gradients to vanish, as well as use some techniques to make the training smooth, adding more layers doesn’t always help. For example, ResNet-1000 has similar accuracy as ResNet-101.
# 
# ---

# ## **<span style="color:orange;">Width Scaling (w)</span>**
# 
# This is commonly used when we want to keep our model small. Wider networks tend to be able to capture more fine-grained features. Also, smaller models are easier to train.  
#   
# What is the problem now?
# The problem is that even though you can make your network extremely wide, with shallow models (less deep but wider) accuracy saturates quickly with larger width.
# 
# ---

# ## **<span style="color:orange;">Resolution (r)</span>**
# 
# Intuitively, we can say that in a high-resolution image, the features are more fine-grained and hence high-res images should work better. This is also one of the reasons that in complex tasks, like Object detection, we use image resolutions like 300x300, or 512x512, or 600x600.   
#   
# But this doesn’t scale linearly. The accuracy gain diminishes very quickly. For example, increasing resolution from 500x500 to 560x560 doesn’t yield significant improvements.
#   
# The above three points lead to our first observation: Scaling up any dimension of network (width, depth or resolution) improves accuracy, but the accuracy gain diminishes for bigger models.
# 
# ![](https://miro.medium.com/max/875/1*yMCuuf5qzOVbYIJWmvW6Tg.png)
# 
# Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficients. Bigger networks with larger width, depth, or resolution tend to achieve higher accuracy, but the accuracy gain quickly saturate after reaching 80%, demonstrating the limitation of single dimension scaling.
# 
# ---

# ## **<span style="color:orange;">Combined Scaling</span>**
# 
# Yes, we can combine the scalings for different dimensions but there are some points that the authors have made:  
# - Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling is a tedious task.
# - Most of the times, manual scaling results in sub-optimal accuracy and efficiency.
#   
# Intuition says that as the resolution of the images is increased, depth and width of the network should be increased as well. As the depth is increased, larger receptive fields can capture similar features that include more pixels in an image. Also, as the width is increased, more fine-grained features will be captured. To validate this intuition, the authors ran a number of experiments with different scaling values for each dimension. For example, as shown in the figure below from the paper, with deeper and higher resolution, width scaling achieves much better accuracy under the same FLOPS cost.
# 
# ![](https://miro.medium.com/max/875/1*99pp7-l0392l57TvpxHS9g.png)
# 
# Scaling Network Width for Different Baseline Networks. Each dot in a line denotes a model with different width coefficient (w). All baseline networks are from Table 1. The first baseline network (d=1.0, r=1.0) has 18 convolutional layers with resolution 224x224, while the last baseline (d=2.0, r=1.3) has 36 layers with resolution 299x299  
#   
# These results lead to our second observation: It is critical to balance all dimensions of a network (width, depth, and resolution) during CNNs scaling for getting improved accuracy and efficiency.
# 
# ---

# ## **<span style="color:orange;">Proposed Compound Scaling<span>**
# 
# The authors proposed a simple yet very effective scaling technique which uses a compound coefficient ɸ to uniformly scale network width, depth, and resolution in a principled way:  
# 
# ![](https://miro.medium.com/max/705/1*iYn6_BvI2mFk6rls8LopVA.png)
#   
# `ɸ` is a user-specified coefficient that controls how many resources are available whereas `α`, `β`, and `γ` specify how to assign these resources to network depth, width, and resolution respectively.  
#       
# In a CNN, Conv layers are the most compute expensive part of the network. Also, FLOPS of a regular convolution op is almost proportional to `d`, `w²`, `r²`, i.e. doubling the depth will double the FLOPS while doubling width or resolution increases FLOPS almost by four times. Hence, in order to make sure that the total FLOPS don’t exceed `2^ϕ`, the constraint applied is that `(α * β² * γ²) ≈ 2`
#     
#  ---

# ## **<span style="color:orange;">EfficientNet Architecture<span>**
# 
# Scaling doesn’t change the layer operations, hence it is better to first have a good baseline network and then scale it along different dimensions using the proposed compound scaling. The authors obtained their base network by doing a Neural Architecture Search (NAS) that optimizes for both accuracy and FLOPS. The architecture is similar to M-NASNet as it has been found using the similar search space. The network layers/blocks are as shown below:  
# 
# ![](https://miro.medium.com/max/1400/1*OpvSpqMP61IO_9cp4mAXnA.png)
#   
# The MBConv block is nothing fancy but an Inverted Residual Block (used in MobileNetV2) with a Squeeze and Excite block injected sometimes.  
#       
# Now we have the base network, we can search for optimal values for our scaling parameters. If you revisit the equation, you will quickly realize that we have a total of four parameters to search for: `α`, `β`, `γ`, and `ϕ`.   
#       
# In order to make the search space smaller and making the search operation less costly, the search for these parameters can be completed in two steps.  
#     
# 1) Fix `ϕ =1`, assuming that twice more resources are available, and do a small grid search for `α`, `β`, and `γ`. For baseline network B0, it turned out the optimal values are `α =1.2`, `β = 1.1`, and `γ = 1.15` such that `α * β² * γ² ≈ 2`  
#     
# 2) Now fix `α`, `β`, and `γ` as constants (with values found in above step) and experiment with different values of `ϕ`. The different values of `ϕ` produce EfficientNets `B1-B7`.
# 
#  ---

# [](!https://miro.medium.com/max/875/1*yMCuuf5qzOVbYIJWmvW6Tg.png)

# In[58]:


class PetFinderModel(pl.LightningModule):
    def __init__(self, pretrained=True):
        super(PetFinderModel, self).__init__()
        self.model = timm.create_model(Config['MODEL_NAME'], pretrained=pretrained)
        
        self.n_features = self.model.classifier.in_features
        self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features + 12, Config['NUM_LABELS'])
        
        self.train_loss = nn.MSELoss()
        self.valid_loss = nn.MSELoss()

    def forward(self, images, meta):
        features = self.model(images)
        features = torch.cat([features, meta], dim=1)
        output = self.fc(features)
        return output
    
    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        meta = batch[1]
        target = batch[2]
        
        out = self(imgs, meta)
        train_loss = torch.sqrt(self.train_loss(out, target))
        
        logs = {'train_loss': train_loss}
        
        return {'loss': train_loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        meta = batch[1]
        target = batch[2]
        
        out = self(imgs, meta)
        valid_loss = torch.sqrt(self.valid_loss(out, target))
        
        return {'val_loss': valid_loss}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        
        print(f"val_loss: {avg_loss}")
        return {'avg_val_loss': avg_loss, 'log': logs}
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=Config['LR'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=Config['T_max'],
            eta_min=Config['min_lr']
        )
        
        return [opt], [sch]


# ## **<span style="color:orange;">Data Folds</span>**

# In[59]:


# Run the Kfolds training loop
kf = StratifiedKFold(n_splits=Config['NFOLDS'])
train_file = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")

for fold_, (train_idx, valid_idx) in enumerate(kf.split(X=train_file, y=train_file['Pawpularity'])):
    print(f"{'='*20} Fold: {fold_} {'='*20}")
    
    train_df = train_file.loc[train_idx]
    valid_df = train_file.loc[valid_idx]
    
    train_set = PetfinderData(
        train_df,
        augments = Augments.train_augments
    )

    valid_set = PetfinderData(
        valid_df,
        augments = Augments.valid_augments
    )
    
    train = DataLoader(
        train_set,
        batch_size=Config['TRAIN_BS'],
        shuffle=True,
        num_workers=Config['NUM_WORKERS'],
        pin_memory=True
    )
    valid = DataLoader(
        valid_set,
        batch_size=Config['VALID_BS'],
        shuffle=False,
        num_workers=Config['NUM_WORKERS']
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./",
        filename=f"fold_{fold_}_{Config['MODEL_NAME']}",
        save_top_k=1,
        mode="min",
    )
    
    model = PetFinderModel()
    trainer = pl.Trainer(
        max_epochs=Config['EPOCHS'], 
        gpus=1, 
        callbacks=[checkpoint_callback], 
        logger= wandb_logger
    )
    trainer.fit(model, train, valid)


# <a id="wandb-dashboard"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>WandB Dashboard</center></h2>

# > ### [Link to Dashboard](https://wandb.ai/ishandutta/pytorchlightning?workspace=user-ishandutta)

# In[60]:


# Store all wandb image paths in a list

wandb_img_paths = []

for i in range(1, 5):
    path = "../input/pawpularitywandb/wandb-" + str(i) + ".png"
    wandb_img_paths.append(path)


# In[61]:


def display_img(img_path):
    """
    Function which takes an image path and displays it.
    
    params: img_path(str): Path of Image to be displayed
    """

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(25.5, 17.5)

    img = cv2.imread(img_path)

    plt.axis('off')
    plt.imshow(img)


# In[62]:


display_img(wandb_img_paths[0])


# In[63]:


display_img(wandb_img_paths[1])


# In[64]:


display_img(wandb_img_paths[2])


# In[65]:


display_img(wandb_img_paths[3])


# <a id="references"></a>
# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:orange; border:0; color:white' role="tab" aria-controls="home"><center>References</center></h2>

# >- [PetFinder EDA + YOLOV5 Obj Detection + TFRecords](https://www.kaggle.com/markwijkhuizen/petfinder-eda-yolov5-obj-detection-tfrecords)
# >- [🔥TensorFlow Probability😺🐶+NGBoost+W&B](https://www.kaggle.com/usharengaraju/tensorflow-probability-ngboost-w-b)
# >- [Tutorial Part 1: EDA for Beginners](https://www.kaggle.com/alexteboul/tutorial-part-1-eda-for-beginners)
# >- [Pawpularity - EDA - Feature Engineering - Baseline](https://www.kaggle.com/michaelfumery/pawpularity-eda-feature-engineering-baseline)
# >- [[🐾Train Baseline] Torch Lightning + GPU&TPU + W&B](https://www.kaggle.com/heyytanay/train-baseline-torch-lightning-gpu-tpu-w-b/notebook)
# >- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://medium.com/@nainaakash012/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks-92941c5bfb95)
# >- [Efficientnet Paper](https://arxiv.org/abs/1905.11946)
# >- [Efficientnet Official Released Code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
# >- [PetFinder EDA + YOLOV5 Obj Detection + TFRecords](https://www.kaggle.com/markwijkhuizen/petfinder-eda-yolov5-obj-detection-tfrecords)
# >
# >---

# <h1><center>More Plots and Models coming soon!</center></h1>
#                                                       
# <center><img src = "https://static.wixstatic.com/media/5f8fae_7581e21a24a1483085024f88b0949a9d~mv2.jpg/v1/fill/w_934,h_379,al_c,q_90/5f8fae_7581e21a24a1483085024f88b0949a9d~mv2.jpg" width = "750" height = "500"/></center> 

# --- 
# 
# ## **<span style="color:orange;">Let's have a Talk!</span>**
# > ### Reach out to me on [LinkedIn](https://www.linkedin.com/in/ishandutta0098)
# 
# ---
