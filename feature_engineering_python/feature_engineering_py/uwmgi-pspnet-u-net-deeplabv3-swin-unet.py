#!/usr/bin/env python
# coding: utf-8

# ---
# # [UW-Madison GI Tract Image Segmentation][1]
# 
# - The goal of this competition is to create a model to automatically segment stomach and intestines on MRI scans.
# 
# ---
# #### **The aim of this notebook is to**
# - **1. Conduct Exploratory Data Analysis (EDA).**
# - **2. Build the PSPNet with ResNet18 (pretrained on ImageNet) as encoder, and other modules from scratch.**
# - **3. Build the U-Net model with ResNet18 as encoder, and decoder from scratch.**
# - **4. Build the Attention U-Net model with ResNet18 as encoder, and decoder from scratch.**
# - **5. Build the DeepLabv3 model with using [segmentation_models_pytorch][2] library.**
# - **6. Build the Swin-UNET model with using [keras-unet-collection][3] library.**
# - **7. Train the model with focal loss function for the unbalanced multi label semantic segmentation problem.**
# 
# ---
# #### **Please note**
# - **We need Internet access to run this code. Thus, this notebook doesn't fulfill the conditions for submitting (Internet access disabled).**
# 
# ---
# #### **References:** 
# Thanks to previous great blogs, codes and notebooks.
# - [YutaroOgawa/pytorch_advanced: 3_semantic_segmentation][4]
# - [Review: DeepLabv3 — Atrous Convolution (Semantic Segmentation)][5]
# - [Review: Swin Transformer][6]
# 
# ---
# #### **If you find this notebook useful, please do give me an upvote. It helps me keep up my motivation.**
# #### **Also, I would appreciate it if you find any mistakes and help me correct them.**
# 
# ---
# 
# [1]: https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview
# [2]: https://github.com/qubvel/segmentation_models.pytorch
# [3]: https://github.com/yingkaisha/keras-unet-collection
# [4]: https://github.com/YutaroOgawa/pytorch_advanced/tree/master/3_semantic_segmentation
# [5]: https://towardsdatascience.com/review-deeplabv3-atrous-convolution-semantic-segmentation-6d818bfd1d74
# [6]: https://sh-tsang.medium.com/review-swin-transformer-3438ea335585
# 

# <h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>TABLE OF CONTENTS</center></h1>
# 
# <ul class="list-group" style="list-style-type:none;">
#     <li><a href="#0" class="list-group-item list-group-item-action">0. Settings</a></li>
#     <li><a href="#1" class="list-group-item list-group-item-action">1. Data Loading</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#1.1" class="list-group-item list-group-item-action">1.1 Feature Engineering </a></li>
#         </ul>
#     </li>
#     <li><a href="#2" class="list-group-item list-group-item-action">2. Exploratory Data Analysis</a></li>
#     <li><a href="#3" class="list-group-item list-group-item-action">3. Dataset & DataLoader</a></li>
#     <li><a href="#4" class="list-group-item list-group-item-action">4. Model Building</a>
#         <ul class="list-group" style="list-style-type:none;">
#             <li><a href="#4.1" class="list-group-item list-group-item-action">4.1 PSPNet </a></li>
#             <li><a href="#4.2" class="list-group-item list-group-item-action">4.2 U-Net (decoder from scratch) </a></li>
#             <li><a href="#4.3" class="list-group-item list-group-item-action">4.3 Attention U-Net (decoder from scratch) </a></li>
#             <li><a href="#4.4" class="list-group-item list-group-item-action">4.4 DeepLabv3 </a></li>
#             <li><a href="#4.5" class="list-group-item list-group-item-action">4.5 Swin-UNET </a></li>
#         </ul>
#     </li>
#     <li><a href="#5" class="list-group-item list-group-item-action">5. Training</a></li>
#     <li><a href="#6" class="list-group-item list-group-item-action">6. Prediction</a></li>
# </ul>
# 

# <a id ="0"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>0. Settings</center></h1>

# In[1]:


## Import dependencies 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.offline as offline

import os
import pathlib
import gc
import sys
import re
import math 
import random
import time 
import datetime as dt
from tqdm import tqdm 
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
get_ipython().system('pip install torchinfo -q --user')
from torchinfo import summary

from PIL import Image

print('import done!')


# In[2]:


## For reproducible results    
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


# <a id ="1"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>1. Data Loading</center></h1>

# ---
# ### [Files Descriptions](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data)
# 
# - **train.csv** - IDs and masks for all training objects.
# 
# - **train** - A folder of case/day folders, each containing slice images for a particular case on a given day.
# 
# - **test** - The test set is entirely unseen. It is roughly 50 cases, with a varying number of days and slices, as seen in the training set.
# 
# - **sample_submission.csv** - A sample submission file in the correct format.
# 
# ---
# ### [Field Descriptions](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data)
# 
# - **train.csv**
#  - `id` - unique identifier for object
#  - `class` - the predicted class for the object
#  - `EncodedPixels` - RLE-encoded pixels for the identified object
#  
# --- 
# 
# ### [Submission File](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/overview/evaluation)
# In order to reduce the submission file size, our metric uses run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
# 
# Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.
# 
# The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

# In[3]:


## Data Loading
data_config = {'train_csv_path': '../input/uw-madison-gi-tract-image-segmentation/train.csv',
               'train_folder_path': '../input/uw-madison-gi-tract-image-segmentation/train',
               'test_folder_path': '../input/uw-madison-gi-tract-image-segmentation/test',
               'sample_submission_path': '../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv',
              }

train_df = pd.read_csv(data_config['train_csv_path'])
submission_df = pd.read_csv(data_config['sample_submission_path'])

print(f'train_length: {len(train_df)}')
print(f'submission_length: {len(submission_df)}')


# In[4]:


## Null Value Check
print('train_df.info()'); print(train_df.info(), '\n')

train_df.head()


# <a id ="1.1"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>1.1 Feature Engineering</center></h2>

# In[5]:


## Separate 'id' columns' texts, and create new id columns.
## This code takes about 2 minutets to execute.

def create_id_list(text, p_train = pathlib.Path(data_config['train_folder_path'])):
    t = text.split('_')
    
    case_id = t[0][4:]
    day_id = t[1][3:]
    slice_id = t[3]
    
    case_folder = t[0]
    day_folder = ('_').join([t[0], t[1]])
    slice_file = ('_').join([t[2], t[3]])
    
    p_folder = p_train / case_folder / day_folder / 'scans'
    file_name = [p.name for p in p_folder.iterdir() if p.name[6:10] == slice_id]
    id_list = [case_id, day_id, slice_id, case_folder, day_folder, slice_file]
    id_list.extend(file_name)    
    return id_list

def create_new_ids(dataframe, new_ids = ['case_id', 'day_id', 'slice_id', 'case_folder', 'day_folder', 'slice_file', 'file_name']):
    dataframe['id_list'] = dataframe['id'].map(create_id_list)   
    for i, item in enumerate(new_ids):
        dataframe[item] = dataframe['id_list'].map(lambda x: x[i])
    dataframe = dataframe.drop(['id_list'], axis=1)
    return dataframe

train_df = create_new_ids(train_df)
train_df.head()


# In[6]:


## Create detection column (1: non NaN segmentation, 0: NaN segmentation).
train_df['detection'] = train_df['segmentation'].notna() * 1
train_df.head()


# In[7]:


total_img_n = int(len(train_df) / 3)
print('The number of imgs: ', total_img_n)


# In[8]:


## Calculate segmentation areas and img size.
def cal_pos_area(segmentation):
    pos_area = 0
    if type(segmentation) is str:
        seg_list = segmentation.split(' ')
        for i in range(len(seg_list)//2):
            pos_area += int(seg_list[i*2 + 1])
    return pos_area

def cal_total_area(file_name):
    img_h = int(file_name[11:14])
    img_w = int(file_name[15:18])
    total_area = img_h * img_w
    return total_area

train_df['pos_area'] = train_df['segmentation'].map(cal_pos_area)
train_df['total_area'] = train_df['file_name'].map(cal_total_area)
train_df['pos_area_percentage'] = train_df['pos_area'] / train_df['total_area'] * 100

## Check
train_df[1920:1930]


# In[9]:


## Split the samples based on the 'class'.
train_lb_df = train_df[train_df['class']=='large_bowel'].reset_index(drop=True)
train_sb_df = train_df[train_df['class']=='small_bowel'].reset_index(drop=True)
train_st_df = train_df[train_df['class']=='stomach'].reset_index(drop=True)

## Calculate each segmentation pixels' ratio to the total img pixels.
lb_area_ratio = train_lb_df['pos_area'].sum() / train_lb_df['total_area'].sum()
sb_area_ratio = train_sb_df['pos_area'].sum() / train_sb_df['total_area'].sum()
st_area_ratio = train_st_df['pos_area'].sum() / train_st_df['total_area'].sum()
bg_area_ratio = 1 - (lb_area_ratio + sb_area_ratio + st_area_ratio)

print(lb_area_ratio, sb_area_ratio, st_area_ratio, bg_area_ratio)


# In[10]:


## Split the samples which have non-null values in 'segmentation' as positive ones.
train_positive_df = train_df.dropna(subset=['segmentation']).reset_index(drop=True)
train_negative_df = train_df[train_df['segmentation'].isna()].reset_index(drop=True)

pos_lb_df = train_positive_df[train_positive_df['class']=='large_bowel'].reset_index(drop=True)
pos_sb_df = train_positive_df[train_positive_df['class']=='small_bowel'].reset_index(drop=True)
pos_st_df = train_positive_df[train_positive_df['class']=='stomach'].reset_index(drop=True)


# <a id ="2"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>2. Exploratory Data Analysis</center></h1>

# In[11]:


## Plot the bar graph of the detection percentages (per total number of images) of each classes.
class_group = train_df.groupby(['class'])['detection'].mean() * 100

fig = px.bar(class_group)
fig.update_layout(title = "<span style='font-size:36px;>Detection Percentages (per total number of images) of Each Classes</span>", 
                  yaxis_title = 'detection percentage')  


# In[12]:


## Plot the histogram of the detection percentage of large_bowel class in each case_ids
lb_detection_mean = train_lb_df.groupby(['case_id'])['detection'].mean() * 100
fig = px.histogram(lb_detection_mean, nbins=25, marginal='box')
fig.update_layout(title = "<span style='font-size:36px;>Detection Percentage of 'large_bowel' in Each Case_ids</span>",
                  xaxis_title = 'detection percentage')  


# In[13]:


## Plot the histogram of the detection percentage of small_bowel class in each case_ids
sb_detection_mean = train_sb_df.groupby(['case_id'])['detection'].mean() * 100
fig = px.histogram(sb_detection_mean, nbins=25, marginal='box')
fig.update_layout(title = "<span style='font-size:36px;>Detection Percentage of 'small_bowel' in Each Case_ids</span>", 
                  xaxis_title = 'detection percentage')  


# In[14]:


## Plot the histogram of the detection percentage of stomach class in each case_ids
st_detection_mean = train_st_df.groupby(['case_id'])['detection'].mean() * 100
fig = px.histogram(st_detection_mean, nbins=25, marginal='box')
fig.update_layout(title = "<span style='font-size:36px;>Histogram of Detection Percentage of 'stomach' in Each Case_ids</span>", 
                  xaxis_title = 'detection percentage')  


# In[15]:


## Compare the above three histograms in one figure.
fig = go.Figure()
lb_detection_mean = train_lb_df.groupby(['case_id'])['detection'].mean() * 100
fig.add_trace(go.Histogram(x=lb_detection_mean.values, nbinsx=25, 
                           opacity=0.5, name='large_bowel',
                           histnorm='probability'))

sb_detection_mean = train_sb_df.groupby(['case_id'])['detection'].mean() * 100
fig.add_trace(go.Histogram(x=sb_detection_mean.values, nbinsx=25, 
                           opacity=0.5, name='small_bowel',
                           histnorm='probability'))

st_detection_mean = train_st_df.groupby(['case_id'])['detection'].mean() * 100
fig.add_trace(go.Histogram(x=st_detection_mean.values, nbinsx=25, 
                           opacity=0.5, name='stomach',
                           histnorm='probability'))
fig.update_layout(barmode='overlay', 
                  title = "<span style='font-size:36px;>Comparison of Detection Percentages in Each Case_ids</span>",
                  xaxis_title = 'detection percentage',
                  yaxis_title = 'n_cases / n_total_cases')


# In[16]:


## Plot the bar graph of the detection area percentages of three classes.
class_group = train_positive_df.groupby(['class'])['pos_area_percentage']

fig = px.bar(class_group.mean(), error_y=class_group.std())
fig.update_layout(title = "<span style='font-size:36px;>Mean Area Percentages (for the area of an image) of Eacg Classes</span>", 
                  yaxis_title = 'detection area percentage') 


# In[17]:


## Plot the histogram of the detection area percentages of three classes.
fig = go.Figure()
fig.add_trace(go.Histogram(x=pos_lb_df['pos_area_percentage'], nbinsx=100, 
                           opacity=0.5, name='large_bowel',
                           histnorm='probability'))
fig.add_trace(go.Histogram(x=pos_sb_df['pos_area_percentage'], nbinsx=100, 
                           opacity=0.5, name='small_bowel',
                           histnorm='probability'))
fig.add_trace(go.Histogram(x=pos_st_df['pos_area_percentage'], nbinsx=100, 
                           opacity=0.5, name='stomach',
                           histnorm='probability'))
fig.update_layout(barmode='overlay', 
                  title = "<span style='font-size:36px;>Comparison of Detection Area Percentage </span>",
                  xaxis_title = 'detection area percentage',
                  yaxis_title = 'n_images / n_total_detected_imgs')


# <a id ="3"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>3. Dataset & DataLoader</center></h1>

# In[18]:


## Train - Valid - Test split
## I split the train, valid, test data based on the case_id (imgs that have the same case_id are assigned in the same set).

train_ratio = 0.85
valid_ratio = 0.10
test_ratio = 0.05

case_ids = train_df['case_id'].unique()
idxs = np.random.permutation(range(len(case_ids)))
cut_1 = int(train_ratio * len(idxs))
cut_2 = int((train_ratio + valid_ratio) * len(idxs))

train_case_ids = case_ids[idxs[:cut_1]]
valid_case_ids = case_ids[idxs[cut_1:cut_2]]
test_case_ids = case_ids[idxs[cut_2:]]

train = train_df.query('case_id in @train_case_ids')
valid = train_df.query('case_id in @valid_case_ids')
test = train_df.query('case_id in @test_case_ids')

print(len(train), len(valid), len(test), len(train_df))


# In[19]:


train_case_folders = train['case_folder'].unique()
train_files = []
for case_folder in train_case_folders:
    p_train = pathlib.Path(data_config['train_folder_path'])
    p_folder = p_train / case_folder
    tmp_files = list(p_folder.glob('**/scans/*.png'))
    train_files.extend(tmp_files)
    
valid_case_folders = valid['case_folder'].unique()
valid_files = []
for case_folder in valid_case_folders:
    p_train = pathlib.Path(data_config['train_folder_path'])
    p_folder = p_train / case_folder
    tmp_files = list(p_folder.glob('**/scans/*.png'))
    valid_files.extend(tmp_files)
    
test_case_folders = test['case_folder'].unique()
test_files = []
for case_folder in test_case_folders:
    p_train = pathlib.Path(data_config['train_folder_path'])
    p_folder = p_train / case_folder
    tmp_files = list(p_folder.glob('**/scans/*.png'))
    test_files.extend(tmp_files)
    
print(len(train_files), len(valid_files), len(test_files))


# In[20]:


## Building Dataset and DataLoader
class UWMadison2022Dataset(torch.utils.data.Dataset):
    def __init__(self, files, dataframe=None, input_shape=256,):
        self.files = files
        self.df = dataframe
        self.input_shape = input_shape
        self.transforms = transforms.Compose([
            transforms.CenterCrop(self.input_shape),
            transforms.Normalize(mean=[(0.485+0.456+0.406)/3], std=[(0.229+0.224+0.225)/3]),
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        p_file = self.files[idx]
        #img = torchvision.io.read_image(p_file)
        img = np.array(Image.open(p_file))
        img_shape = torch.tensor(img.shape)
        img = transforms.functional.to_tensor(img) / 255.
        img = self.transforms(img)
        #img = torch.cat([img, img, img], dim=0)
        
        if self.df is not None:
            f_name = str(p_file).split('/')
            case_day_id = f_name[5]
            slice_id = f_name[7][:10]
            f_id = '_'.join([case_day_id, slice_id])
            labels_df = self.df.query('id == @f_id')
            
            label = torch.zeros([img_shape[0]*img_shape[1]])
            for i, organ in enumerate(['large_bowel', 'small_bowel', 'stomach']):
                segmentation = labels_df[labels_df['class'] == organ]['segmentation'].item()
                if type(segmentation) is str:
                    segmentation = segmentation.split(' ')
                    for j in range(len(segmentation)//2):
                        start_idx = int(segmentation[j*2])
                        span = int(segmentation[j*2 + 1])
                        label[start_idx:(start_idx+span)] = (i+1)
            label = torch.reshape(label, (img_shape[0], img_shape[1]))
            label = transforms.CenterCrop(self.input_shape)(label)
            label = torch.nn.functional.one_hot(label.to(torch.int64), num_classes=4)
            label = label.permute(2, 0, 1)
            return img, label, img_shape
        
        else: return img, img_shape
        
train_ds = UWMadison2022Dataset(train_files, train, input_shape=256)
valid_ds = UWMadison2022Dataset(valid_files, valid, input_shape=256)
test_ds = UWMadison2022Dataset(test_files, test, input_shape=256)

BATCH_SIZE = 32

## Checking dataset and dataloder  
print('------ train_dl ------')
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
tmp = train_dl.__iter__()
x, y, shape = tmp.next()
print(f"x : {x.shape}")
print(f"labels: {y.shape}")
print(f"img_shapes: {shape.shape}")
print(f"n_samples: {len(train_ds)}")
print(f"n_batches: {len(tmp)}")
print()

print('------ valid_dl ------')
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
tmp = valid_dl.__iter__()
x, y, shape = tmp.next()
print(f"x : {x.shape}")
print(f"labels: {y.shape}")
print(f"img_shapes: {shape.shape}")
print(f"n_samples: {len(valid_ds)}")
print(f"n_batches: {len(tmp)}")
print()

print('------ test_dl ------')
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
tmp = test_dl.__iter__()
x, y, shape = tmp.next()
print(f"x : {x.shape}")
print(f"labels: {y.shape}")
print(f"img_shapes: {shape.shape}")
print(f"n_samples: {len(test_ds)}")
print(f"n_batches: {len(tmp)}")
print()


# <a id ="4"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>4. Model Building</center></h1>

# <a id ="4.1"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.1 PSPNet</center></h2>

# - **[PSPNet](https://arxiv.org/abs/1612.01105), or Pyramid Scene Parsing Network, is a semantic segmentation model that utilises a pyramid pooling module that exploits global context information by different-region based context aggregation.**
# - **We use ResNet18 (pretrained on Imagenet) for the encoder, and build pyramid pooling module and decoder from scratch.**
# 
# <img src="https://qiita-user-contents.imgix.net/https%3A%2F%2Fqiita-image-store.s3.ap-northeast-1.amazonaws.com%2F0%2F482094%2F176d3ce5-8e3a-cd40-efd4-526f1702f343.png?ixlib=rb-4.0.0&auto=format&gif-q=60&q=75&w=1400&fit=max&s=0fa30aa09b4880ee27a38839bd007bc4" width="750"/>
# 
# - **The feature map size is 1/8 of the input image.**
# - **We use the 4-level pyramid pooling module to gather various context information.**
# - **Then, it is followed by a convolution and upsampling layer to generate the final prediction.**
# - **For the auxiliary of training, auxiliary (Aux) loss is added to the final loss of the network.**
# - **The idea of Aux loss comes from [GoogLeNet](https://arxiv.org/abs/1409.4842) paper. Auxiliary network's task is to predict pixel's labels using the intermediate output of encoder module. When the encoder generates big loss, that helps to improve the encoder directly.**

# In[21]:


## Structure of Resnet18
resnet = resnet18(pretrained=True)
batch_size = 16

summary(
    resnet,
    input_size=(batch_size, 3, 256, 256),
    col_names=["output_size", "num_params"],
)


# We will use the feature outputs of ResNet18 at **Sequential: 1-5 [batch_size, 64, 64, 64] ('conv2_x' in the figure below)** for the inputs of Auxiliary network, and **Sequential: 1-6 [batch_size, 128, 32, 32] ('conv3_x' in the figure below)** for the PyramidPooling module.
# 
# 
# - **Reference: Comparison of various ResNets' architecture**
# 
# <img src="https://pystyle.info/wp/wp-content/uploads/2021/11/pytorch-resnet_06.jpg" width="750"/>

# In[22]:


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, bias, activation=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, 
                              padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        if self.activation:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation:
            outputs = self.relu(x)
        else:
            outputs = x
            
        return outputs


# In[23]:


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        # Change first conv layer to accept single-channel (grayscale) input
        self.resnet.conv1.weight = torch.nn.Parameter(self.resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        
    def forward(self, x):
        for i in range(6):
            x = list(self.resnet.children())[i](x)
            if i == 4:
                aux_inputs = x
        encoder_outputs = x
        
        return encoder_outputs, aux_inputs


# In[24]:


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super().__init__()
        
        self.height = height
        self.width = width
        
        out_channels = int(in_channels / len(pool_sizes))
        
        ## pool__sizes: [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=False, activation=True)
        
        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=False, activation=True)
        
        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=False, activation=True)
        
        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, dilation=1, bias=False, activation=True)
        
    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(
            self.height, self.width), mode='bilinear', align_corners=True)
        
        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(
            self.height, self.width), mode='bilinear', align_corners=True)
        
        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(
            self.height, self.width), mode='bilinear', align_corners=True)
        
        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(
            self.height, self.width), mode='bilinear', align_corners=True)
        
        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        
        return output


# In[25]:


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super().__init__()
        
        self.height = height
        self.width = width
        
        self.cbr = conv2DBatchNorm(
            in_channels=256, out_channels=64, kernel_size=3, stride=1, 
            padding=1, dilation=1,  bias=False, activation=True)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=64, out_channels=n_classes, kernel_size=1,
            stride=1, padding=0)
        
    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), 
            mode='bilinear', align_corners=True)
        
        return output


# In[26]:


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super().__init__()
        
        self.height = height
        self.width = width
        
        self.cbr = conv2DBatchNorm(
            in_channels=in_channels, out_channels=64,
            kernel_size=3, stride=1, padding=1, 
            dilation=1, bias=False, activation=True)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=64, out_channels=n_classes, 
            kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), 
            mode='bilinear', align_corners=True)
        
        return output


# In[27]:


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        full_img_size = 256
        feature_map_size = 32
        
        self.feature_extractor = FeatureExtractor()
        self.pyramid_pooling = PyramidPooling(
            in_channels=128, pool_sizes=[6, 3, 2, 1],
            height=feature_map_size, width=feature_map_size)
        self.decode_feature = DecodePSPFeature(
            height=full_img_size, width=full_img_size,
            n_classes=n_classes)
        self.aux = AuxiliaryPSPlayers(
            in_channels=64, n_classes=n_classes,
            height=full_img_size, width=full_img_size)
        
    def forward(self, x):
        encoder_outputs, aux_inputs = self.feature_extractor(x)
        pyramid_outputs = self.pyramid_pooling(encoder_outputs)
        docoder_outputs = self.decode_feature(pyramid_outputs)
        aux_outputs = self.aux(aux_inputs)
        
        return (docoder_outputs, aux_outputs)


# In[28]:


model = PSPNet(n_classes=4)

summary(
    model,
    input_size=(batch_size, 1, 256, 256),
    col_names=["output_size", "num_params"],
)


# In[29]:


## We calculate the final loss and Aux loss each other, and mix them up.

class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super().__init__()
        self.aux_weight = aux_weight
        
    def forward(self, outputs, targets):
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')
        
        return loss + self.aux_weight * loss_aux
    
criterion = PSPLoss(aux_weight=0.4)


# In[30]:


## In the inference phase, we don't use Aux loss.

#model.eval()
#outputs = model(x)
#pred = outputs[0]


# <a id ="4.2"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.2 U-Net (decoder from scratch)</center></h2>

# - **The U-Net consists of encoder - decoder network architecture.**
# - **We use ResNet18 (pretrained on Imagenet) for the encoder, and build decoder from scratch.**
# - **We have to make skip connections from encoder to decoder.**
# 
# <img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="500"/>

# We will use the feature outputs of ResNet18 at 
# - **ReLU: 1-3 [batch_size, 64, 128, 128]**
# - **Sequential: 1-5 [batch_size, 64, 64, 64]**
# - **Sequential: 1-6 [batch_size, 128, 32, 32]**
# - **Sequential: 1-7 [batch_size, 256, 16, 16]**
# 
#  for the skip connections. And, outputs at 
# 
# - **Sequential: 1-8 [16, 512, 8, 8]**
# 
#  is for the bottleneck features (first inputs for the decoder).

# In[31]:


## The Extractor of intermediate features of ResNet (encoder) for the skip connections to the decoder.
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        # Change first conv layer to accept single-channel (grayscale) input
        self.resnet.conv1.weight = torch.nn.Parameter(self.resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
        
    def forward(self, x):
        skip_connections = []
        for i in range(8):
            x = list(self.resnet.children())[i](x)
            if i in [2, 4, 5, 6, 7]:
                skip_connections.append(x)
        encoder_outputs = skip_connections.pop(-1)
        skip_connections = skip_connections[::-1]
        
        return encoder_outputs, skip_connections


# In[32]:


## The modules used for building the decoder architecture.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        return  self.dconv(x)
    
    
class UnetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.dconv = DoubleConv(out_channels*2, out_channels)
         
    def forward(self, layer_input, skip_input):
        u = self.convt(layer_input)
        u = self.norm1(u)
        u = self.act1(u)
        u = torch.cat((u, skip_input), dim=1)
        u = self.dconv(u)
        return u


# In[33]:


## U-Net architechture.
class UW2022Unet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.encoder = FeatureExtractor()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.upsample1 = UnetUpSample(512, 256)
        self.upsample2 = UnetUpSample(256, 128)
        self.upsample3 = UnetUpSample(128, 64)
        self.upsample4 = UnetUpSample(64, 64)
        
        self.final_convt = nn.ConvTranspose2d(
            64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1, skip_connections = self.encoder(x)
        x2 = self.upsample1(x1, skip_connections[0])
        x3 = self.upsample2(x2, skip_connections[1])
        x4 = self.upsample3(x3, skip_connections[2])
        x5 = self.upsample4(x4, skip_connections[3])
        x6 = self.final_convt(x5)
        
        return self.final_conv(x6)
    
model = UW2022Unet(out_channels=4)

summary(
    model,
    input_size=(batch_size, 1, 256, 256),
    col_names=["output_size", "num_params"],
)


# <a id ="4.3"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.3 Attention U-Net (decoder from scratch)</center></h2>

# - **The [Attention U-Net](https://arxiv.org/abs/1804.03999) has attention architecture (the figure below) in the skip connections between encoder and decoder.**
# 
# <img src="https://cdn-ak.f.st-hatena.com/images/fotolife/y/y_kurashina/20190429/20190429231337.jpg" width="500"/>
# <img src="https://cdn-ak.f.st-hatena.com/images/fotolife/y/y_kurashina/20190429/20190429233922.jpg" width="500"/>

# In[34]:


## The Modules used for building Attention architecture.
class GateSignal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ) 
        
    def forward(self, x):
        return self.gate_conv(x)
        
class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, x_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv2d(x_channels, inter_channels, kernel_size=2, stride=2)
        self.phi = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1)
        self.act1 = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1)
        self.act2 = nn.Sigmoid()
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(inter_channels, x_channels, kernel_size=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(x_channels)
        
    def forward(self, x, g):
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        xg = torch.add(theta_x, phi_g)
        xg = self.act1(xg)
        score = self.psi(xg)
        score = self.act2(score)
        score = self.upsample(score)
        score = score.expand(x.shape)        
        att_x = torch.mul(score, x)
        att_x = self.final_conv(att_x) 
        att_x = self.norm(att_x)
        return att_x


# In[35]:


## Attention U-Net architechture.
class UW2022AttentionUnet(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.encoder = FeatureExtractor()
        
        self.upsample1 = UnetUpSample(512, 256)
        self.upsample2 = UnetUpSample(256, 128)
        self.upsample3 = UnetUpSample(128, 64)
        self.upsample4 = UnetUpSample(64, 64)
        
        self.gate_signal1 = GateSignal(512, 256)
        self.gate_signal2 = GateSignal(256, 128)
        self.gate_signal3 = GateSignal(128, 64)
        self.gate_signal4 = GateSignal(64, 64)
        
        self.attention1 = AttentionBlock(256, 256, 256)
        self.attention2 = AttentionBlock(128, 128, 128)
        self.attention3 = AttentionBlock(64, 64, 64)
        self.attention4 = AttentionBlock(64, 64, 64)
        
        self.final_convt = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x1, skip_connections = self.encoder(x)
        
        g1 = self.gate_signal1(x1)
        attention_skip1 = self.attention1(skip_connections[0], g1)
        x2 = self.upsample1(x1, attention_skip1)
        
        g2 = self.gate_signal2(x2)
        attention_skip2 = self.attention2(skip_connections[1], g2)
        x3 = self.upsample2(x2, attention_skip2)
        
        g3 = self.gate_signal3(x3)
        attention_skip3 = self.attention3(skip_connections[2], g3)
        x4 = self.upsample3(x3, attention_skip3)
        
        g4 = self.gate_signal4(x4)
        attention_skip4 = self.attention4(skip_connections[3], g4)
        x5 = self.upsample4(x4, attention_skip4)

        x6 = self.final_convt(x5)       
        return self.final_conv(x6)

model = UW2022AttentionUnet(out_channels=4)

summary(
    model,
    input_size=(batch_size, 1, 256, 256),
    col_names=["output_size", "num_params"],
)


# <a id ="4.4"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.4 DeepLabv3</center></h2>

# - **In [DeepLabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/), Atrous Convolution (or Dilated Convolution) block is used to enlarge the field of view of filters to incorporate larger context.**
# 
# <img src="https://miro.medium.com/max/1130/1*-r7CL0AkeO72MIDpjRxfog.png" width="500"/>
# <img src="https://miro.medium.com/max/1400/1*nFJ_GqK1D3zKCRgtnRfrcw.png" width="750"/>
# 
# ---
# - **In Atrous Spatial Pyramid Pooling (ASPP), parallel atrous convolution with different rate applied in the input feature map, and fuse together. As objects of the same class can have different scales in the image, ASPP helps to account for different object scales which can improve the accuracy.**
# 
# <img src="https://miro.medium.com/max/1400/1*8Lg66z7e7ijuLmSkOzhYvA.png" width="750"/>

# In[36]:


get_ipython().system('pip install -q -U segmentation_models_pytorch')
import segmentation_models_pytorch as smp

model = smp.DeepLabV3(encoder_name='resnet34',
                      encoder_depth=5,
                      encoder_weights='imagenet',
                      in_channels=1,
                      classes=4,
                      aux_params=None)


# In[37]:


summary(
    model,
    input_size=(batch_size, 1, 256, 256),
    col_names=["output_size", "num_params"],
)


# <a id ="4.5"></a><h2 style="background:#d9afed; border:0; border-radius: 8px; color:black"><center>4.5 Swin-UNET</center></h2>

# - **[Swin Transformer](https://github.com/microsoft/Swin-Transformer), is an applications of Transformers in vision problems.**
# 
# <img src="https://miro.medium.com/max/1400/1*x50n-zyiU6TMHamEPM2HeA.png" width="750"/>
# 
# ---
# - **One of the biggest difference from ViT (Vision Transformer) is Swin Transformer block, which consists of a shifted window based Multi-Headed Self-Attention (MSA) module, and its efficient computation by shifted configuration (cyclic shift) method.**
# 
# <img src="https://miro.medium.com/max/924/1*wc1e3QQu2AIhRWHvfId8pw.png" width="500"/>
# <img src="https://miro.medium.com/max/930/1*Z8MJcU0p2G3NayYmNV_dnQ.png" width="500"/>
# 
# ---
# - **The Architecture of Swin-Unet.**
# 
# <img src="https://aisholar.s3.ap-northeast-1.amazonaws.com/media/April2022/2022_04_08_1f7df21e7573181e3115g-05.jpeg" width="750"/>

# In[38]:


import tensorflow as tf
from tensorflow import keras

## Limit GPU Memory in TensorFlow
## Because TensorFlow, by default, allocates the full amount of available GPU memory when it is launched. 
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")


# In[39]:


get_ipython().system('pip install keras-unet-collection -q -U')
from keras_unet_collection import models, losses

tf_model = models.swin_unet_2d((256, 256, 1), filter_num_begin=64,
                               n_labels=4, depth=4, stack_num_down=2, stack_num_up=2,
                               patch_size=(4, 4), num_heads=[4, 8, 8, 8],
                               window_size=[4, 2, 2, 2], num_mlp=512, 
                               output_activation='Softmax', shift_window=True,
                               name='swin_unet')


# In[40]:


tf_model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy', losses.dice_coef])
tf_model.summary()
## To train this tf_model, we have to create TensorFlow Datasets.


# In[41]:


del tf_model
keras.backend.clear_session()


# <a id ="5"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>5. Training</center></h1>

# Because this competition's label data (mask) is unbalanced (positive areas are smaller than negative areas), I used **"Focal Loss"** instead of normal cross entropy loss for the training loss function. It is said that focal loss function is good for multiclass classification where some classes are difficult to classify, and others are easy. **"gamma"** is a parameter of focal loss function that decides how emphasise the minor labels (the bigger gamma is, the more emphasis on minors we put).

# In[42]:


## Focal Loss Function
class SegmentationFocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        if torch.cuda.is_available():
            self.loss = torch.nn.CrossEntropyLoss(weight=weight).cuda()
        else:
            self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, target):
        ce_loss = self.loss(pred, target)
        #ce_loss = torch.nn.functional.cross_entropy(pred, target, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)

##Setting the weight parameter of CrossEntropyLoss.
lb_weight = 1 / lb_area_ratio
sb_weight = 1 / sb_area_ratio
st_weight = 1 / st_area_ratio
bg_weight = 1 / bg_area_ratio
total_weight = lb_weight + sb_weight + st_weight + bg_weight

lb_weight = lb_weight / total_weight * 5
sb_weight = sb_weight / total_weight * 5 
st_weight = st_weight / total_weight * 5
bg_weight = bg_weight / total_weight * 5
weight = torch.tensor([bg_weight, lb_weight, sb_weight, st_weight], dtype=torch.float)
print(f'bg:{bg_weight}, lb:{lb_weight}, sb:{sb_weight}, st{st_weight}')

loss_fn = SegmentationFocalLoss(gamma=3, weight=weight)


# In[43]:


LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[44]:


## For the model training loop.
if torch.cuda.is_available():
    DEVICE = 'cuda'
else: DEVICE = 'cpu'

def train_fn(loader, model, optimizer, loss_fn, device=DEVICE):
    model.train()
    train_loss = 0.
    loop = tqdm(loader)
    
    for batch_idx, (data, targets, img_size) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        predictions = model(data)
        targets = torch.argmax(targets, dim=1)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())
        train_loss += loss.detach().cpu().numpy() * BATCH_SIZE
        
    train_loss = train_loss / (BATCH_SIZE * len(train_dl))
    return train_loss

## For the model validation loop.
def valid_fn(loader, model, loss_fn, device=DEVICE):
    model.eval()
    valid_loss = 0.
    loop = tqdm(loader)
    
    with torch.no_grad():
        for batch_idx, (data, targets, img_size) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.to(device=device)
            
            predictions = model(data)
            targets = torch.argmax(targets, dim=1)
            loss = loss_fn(predictions, targets)
            valid_loss += loss * BATCH_SIZE
            
            loop.set_postfix(loss=loss.item())
            
        valid_loss = valid_loss / (BATCH_SIZE * len(valid_dl))
    return valid_loss


# In[45]:


## For the train & validation loop.
NUM_EPOCHS = 1

## DeepLabv3 model
model.to(device=DEVICE)

best_loss = 100
for epoch in range(NUM_EPOCHS):
    print('-------------')
    print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
    print('-------------')
    
    train_loss = train_fn(train_dl, model, optimizer, loss_fn, DEVICE)
    valid_loss = valid_fn(valid_dl, model, loss_fn, DEVICE)
    
    if valid_loss < best_loss:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, "./checkpoint.pth")
        print('best model saved!')
        best_loss = valid_loss
    
    print(f'Train Loss: {train_loss},  Valid Loss: {valid_loss}')


# <a id ="6"></a><h1 style="background:#a1a8f0; border:0; border-radius: 10px; color:black"><center>6. Prediction</center></h1>

# In[46]:


checkpoint = torch.load("./checkpoint.pth")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
        
model.eval()
predictions = []

with torch.no_grad():
    for batch in tqdm(test_dl):
        x = batch[0].to(DEVICE)
        test_pred = model(x)
        test_pred = torch.argmax(test_pred, dim=1)
        test_pred = torch.nn.functional.one_hot(test_pred, num_classes=4)
        test_pred = torch.permute(test_pred, dims=[0, 3, 1, 2])
        test_pred = test_pred[:, 1:, ...] ## We don't need background predictions.
        test_pred = test_pred.detach().cpu().numpy()
        predictions.append(test_pred)
    
predictions = np.concatenate(predictions, axis=0)
predictions = predictions.reshape([-1, 256, 256])
print(predictions.shape)


# In[47]:


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    encodes = ' '.join(str(x) for x in runs)
    if encodes == '':
        encodes = np.nan
    return encodes

predictions_rle = []

for pred in predictions:
    pred_rle = rle_encode(pred)
    predictions_rle.append(pred_rle)
    
predictions_rle = np.concatenate([predictions_rle], axis=0)
test['prediction'] = predictions_rle

test.head(20)


# In[ ]:




