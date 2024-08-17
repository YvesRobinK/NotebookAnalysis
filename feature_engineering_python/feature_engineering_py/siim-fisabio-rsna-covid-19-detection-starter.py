#!/usr/bin/env python
# coding: utf-8

# # Intro
# Welcome to the [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/data) compedition.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/26680/logos/header.png)
# 
# For handling chest-x-ray data we recommend [this notebook](https://www.kaggle.com/drcapa/chest-x-ray-starter).
# 
# <span style="color: royalblue;">Please vote the notebook up if it helps you. Feel free to leave a comment above the notebook. Thank you. </span>

# # Libraries

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pydicom as dicom
import cv2
import ast

import warnings
warnings.filterwarnings("ignore")


# # Path

# In[2]:


path = '/kaggle/input/siim-covid19-detection/'
os.listdir(path)


# # Load Data

# In[3]:


train_image = pd.read_csv(path+'train_image_level.csv')
train_study = pd.read_csv(path+'train_study_level.csv')
samp_subm = pd.read_csv(path+'sample_submission.csv')


# # Overview

# In[4]:


print('Number train images samples:', len(train_image))
print('Number train study samples:', len(train_study))
print('Number test samples:', len(samp_subm))


# The train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format.

# In[5]:


train_image.head()


# The train study-level metadata, with one row for each study, including correct labels.

# In[6]:


train_study.head()


# # Read DCM File
# We consider the first train sample.
# 
# All images are stored in paths with the form **study/series/image**.

# In[7]:


# Define image path of the example
path_train = path+'train/'+train_image.loc[0, 'StudyInstanceUID']+'/'+'81456c9c5423'+'/'
# Extract image name of the example
img_id = train_image.loc[0, 'id'].replace('_image', '.dcm')
# Load dicom file
data_file = dicom.dcmread(path_train+img_id)
# Extract image data of the dicom file
img = data_file.pixel_array


# Print meta data of the image:

# In[8]:


print(data_file)


# Image shape:

# In[9]:


print('Image shape:', img.shape)


# Bounding Boxes:

# In[10]:


boxes = ast.literal_eval(train_image.loc[0, 'boxes'])
boxes


# Plot the image of the chest-x-ray with the bounding boxes:

# In[11]:


fig, ax = plt.subplots(1, 1, figsize=(20, 4))

for box in boxes:
    p = matplotlib.patches.Rectangle((box['x'], box['y']), box['width'], box['height'],
                                     ec='r', fc='none', lw=2.)
    ax.add_patch(p)
ax.imshow(img, cmap='gray')
plt.show()


# # Show Examples
# We plot some examples with the chest-x-ray image, the bounding boxes and the label:

# In[12]:


fig, axs = plt.subplots(3, 3, figsize=(20, 20))
fig.subplots_adjust(hspace = .1, wspace=.1)
axs = axs.ravel()

for row in range(9):
    study = train_image.loc[row, 'StudyInstanceUID']
    path_in = path+'train/'+study+'/'
    folder = os.listdir(path_in)
    path_file = path_in+folder[0]
    filename = os.listdir(path_file)[0]
    file_id = filename.split('.')[0]
    
    data_file = dicom.dcmread(path_file+'/'+file_id+'.dcm')
    img = data_file.pixel_array
    if (train_image.loc[row, 'boxes']!=train_image.loc[row, 'boxes']) == False:
        boxes = ast.literal_eval(train_image.loc[row, 'boxes'])
    
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']), box['width'], box['height'],
                                     ec='r', fc='none', lw=2.)
            axs[row].add_patch(p)
    axs[row].imshow(img, cmap='gray')
    axs[row].set_title(train_image.loc[row, 'label'].split(' ')[0])
    axs[row].set_xticklabels([])
    axs[row].set_yticklabels([])


# # Feature Engineering
# There are 3 labels possible:
# * none: no abnormalities on chest radiographs
# * simple opacity: abnormalities on one side
# * double opacity: abnormalities on both sides
# 
# So we can define 3 catgories:

# In[13]:


label_dict = {0: 'none', 1: 'simple_opacity', 2: 'double_opacity'}


# In[14]:


def split_label(s):
    split_string = s.split(' ')
    if len(split_string)==6 and 'none' in split_string:
        return 0
    elif len(split_string)==6 and 'opacity' in split_string:
        return 1
    else:
        return 2


# In[15]:


train_image['category'] = train_image['label'].apply(split_label)


# In[16]:


train_image.head()


# # EDA

# We consider on the distribution of the three categories:

# In[17]:


train_image['category'].value_counts().sort_index().rename(label_dict).plot.bar(rot=0, color='orange', alpha=0.6, grid=True, figsize=(8,4), fontsize=16)
plt.show()


# Next we have a look on the study-level metadata:
# * Negative for Pneumonia - 1 if the study is negative for pneumonia, 0 otherwise
# * Typical Appearance - 1 if the study has this appearance, 0 otherwise
# * Indeterminate Appearance  - 1 if the study has this appearance, 0 otherwise
# * Atypical Appearance  - 1 if the study has this appearance, 0 otherwise

# In[18]:


train_study.sum()[1:].plot.bar(rot=45, color='orange', alpha=0.6, grid=True, figsize=(8,4), fontsize=12)
plt.show()


# # Data Generator
# To load the data on demand we use a data generator.
# 
# *Coming soon*

# # Export

# In[19]:


samp_subm.to_csv('submission.csv', index=False)

