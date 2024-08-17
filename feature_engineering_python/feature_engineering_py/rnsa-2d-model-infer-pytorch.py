#!/usr/bin/env python
# coding: utf-8

# ### Model Architecture
# 
# <center>
# <img src='https://i.postimg.cc/ZYV9RwXt/2-D-model-architecture.png' width=800>
# </center>
# 
# <br>
# 
# ### Links
# 
# 1. [RSNA Fracture Detection - in-depth EDA](https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda)
# 2. [RSNA 2022 Spine Fracture Detection - Metadata](https://www.kaggle.com/datasets/samuelcortinhas/rsna-2022-spine-fracture-detection-metadata)
# 3. [RNSA - 2D model [Train] [PyTorch]](https://www.kaggle.com/code/samuelcortinhas/rnsa-2d-model-train-pytorch)
# 4. [RNSA - 2D model [Re-Train] [PyTorch]](https://www.kaggle.com/code/samuelcortinhas/rnsa-2d-model-re-train-pytorch)
# 5. [RSNA - Trained 2D models [PyTorch]](https://www.kaggle.com/datasets/samuelcortinhas/rsna-trained-2d-models-pytorch)
# 6. [RNSA - 2D model [Validate] [PyTorch]](https://www.kaggle.com/code/samuelcortinhas/rnsa-2d-model-validate-pytorch)
# 7. [RNSA - 2D model [Infer] [PyTorch]](https://www.kaggle.com/code/samuelcortinhas/rnsa-2d-model-infer-pytorch)
# 
# <hr>
# 
# Much of my work is inspired by https://www.kaggle.com/code/vslaykovsky/train-pytorch-effnetv2-baseline-cv-0-49

# # Libraries

# In[1]:


get_ipython().system('pip install -qU ../input/for-pydicom/python_gdcm-3.0.14-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ../input/for-pydicom/pylibjpeg-1.4.0-py3-none-any.whl --find-links frozen_packages --no-index')


# In[2]:


get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints/')
get_ipython().system('cp ../input/rsna-2022-whl/efficientnet_v2_s-dd5fe13b.pth  /root/.cache/torch/hub/checkpoints/')
get_ipython().system('pip install /kaggle/input/rsna-2022-whl/{pydicom-2.3.0-py3-none-any.whl,pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}')
get_ipython().system('pip install /kaggle/input/rsna-2022-whl/{torch-1.12.1-cp37-cp37m-manylinux1_x86_64.whl,torchvision-0.13.1-cp37-cp37m-manylinux1_x86_64.whl}')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.6)
import cv2
import os
from os import listdir
import re
import gc
import random
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm.auto import tqdm
from pprint import pprint
from time import time
import itertools
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
from glob import glob
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=UserWarning)
#warnings.filterwarnings("ignore", category=FutureWarning)
import zipfile
from scipy import ndimage
from sklearn.model_selection import train_test_split, GroupKFold
from joblib import Parallel, delayed
from PIL import Image
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
#from kaggle_volclassif.utils import interpolate_volume
from skimage import exposure

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn.functional as F
import kornia
import kornia.augmentation as augmentation
import albumentations as A

from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
#from tqdm.notebook import tqdm
#import wandb


# ### Reproducibility

# In[4]:


# Set random seeds
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
set_seed()


# # Config

# In[5]:


# Hyperparameters
if torch.cuda.is_available():
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 4

IMG_SIZE = (256,256)

# Config device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# # Data

# ### Load tables

# In[6]:


# Load tables
train_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train.csv")
train_bbox = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv")
test_df = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/test.csv")
ss = pd.read_csv("../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv")

# Print dataframe shapes
print('train shape:', train_df.shape)
print('train bbox shape:', train_bbox.shape)
print('test shape:', test_df.shape)
print('ss shape:', ss.shape)
print('')

# Show first few entries
train_df.head(3)


# ### Debug

# In[7]:


if len(ss)==3:
    # Fix mismatch with test_images folder
    test_df = pd.DataFrame(columns = ['row_id','StudyInstanceUID','prediction_type'])
    for i in ['1.2.826.0.1.3680043.22327','1.2.826.0.1.3680043.25399','1.2.826.0.1.3680043.5876']:
        for j in ['C1','C2','C3','C4','C5','C6','C7','patient_overall']:
            test_df = test_df.append({'row_id':i+'_'+j,'StudyInstanceUID':i,'prediction_type':j},ignore_index=True)
    
    # Sample submission
    ss = pd.DataFrame(test_df['row_id'])
    ss['fractured'] = 0.5
    
    display(test_df.head(3))


# ### Test table

# In[8]:


# Test table
test_table = pd.DataFrame(pd.unique(test_df['StudyInstanceUID']), columns=['StudyInstanceUID'])
test_table.head()


# ### Extract metadata

# In[9]:


def get_observation_data(path):
    '''
    Get information from the .dcm files
    '''
    
    dataset = pydicom.read_file(path)
    
    # Dictionary to store the information from the image
    observation_data = {
        "SOPInstanceUID" : dataset.get("SOPInstanceUID"),
        "InstanceNumber" : dataset.get("InstanceNumber"),
        "SliceThickness" : dataset.get("SliceThickness"),
        "ImagePositionPatient" : dataset.get("ImagePositionPatient"),
    }

    return observation_data

def get_metadata():
    '''
    Retrieves the desired metadata from the .dcm files and saves it into dataframe.
    '''
    
    dicts = []
    
    for k in tqdm(range(len(test_table))):
        patient = test_table.loc[k,'StudyInstanceUID']

        # Get all .dcm paths for this Instance
        dcm_paths = glob(f"../input/rsna-2022-cervical-spine-fracture-detection/test_images/{patient}/*")

        for path in dcm_paths:
            # Get datasets
            dataset = get_observation_data(path)
            dicts.append(dataset)
    
    return pd.DataFrame(data=dicts, columns=md_example.keys())


# In[10]:


# Example
ex_path = "../input/rsna-2022-cervical-spine-fracture-detection/train_images/1.2.826.0.1.3680043.10001/101.dcm"
md_example = get_observation_data(ex_path)
pprint(md_example)


# In[11]:


# Get metadata
test_meta = get_metadata()
test_meta.head(3)


# ### Clean metadata

# In[12]:


# Change data types
test_meta['SOPInstanceUID'] = test_meta['SOPInstanceUID'].astype('str')
test_meta['InstanceNumber'] = test_meta['InstanceNumber'].astype('int32')
test_meta['SliceThickness'] = test_meta['SliceThickness'].astype('float32')
test_meta['ImagePositionPatient'] = test_meta['ImagePositionPatient'].astype('str')

# Patient id
test_meta["StudyInstanceUID"] = test_meta["SOPInstanceUID"].apply(lambda x: ".".join(x.split(".")[:-2]))

# Extract x, y, z coordinates of position vector
test_meta['ImagePositionPatient_x'] = test_meta['ImagePositionPatient'].apply(lambda x: float(x.replace(',','').replace(']','').replace('[','').split()[0]))
test_meta['ImagePositionPatient_y'] = test_meta['ImagePositionPatient'].apply(lambda x: float(x.replace(',','').replace(']','').replace('[','').split()[1]))
test_meta['ImagePositionPatient_z'] = test_meta['ImagePositionPatient'].apply(lambda x: float(x.replace(',','').replace(']','').replace('[','').split()[2]))

# Clean metadata
test_meta.drop(['SOPInstanceUID','ImagePositionPatient'], axis=1, inplace=True)
test_meta.rename(columns={"InstanceNumber": "Slice"}, inplace=True)
test_meta = test_meta[['StudyInstanceUID','Slice','SliceThickness','ImagePositionPatient_x','ImagePositionPatient_y','ImagePositionPatient_z']]
test_meta.sort_values(by=['StudyInstanceUID','Slice'], inplace=True)
test_meta.reset_index(drop=True, inplace=True)


# ### Feature engineering

# In[13]:


# Calculate slice ratio
slice_max = test_meta.groupby('StudyInstanceUID')['Slice'].max().to_dict()
test_meta['SliceRatio'] = 0
test_meta['SliceRatio'] = test_meta['Slice']/test_meta['StudyInstanceUID'].map(slice_max)

# Number of slices
test_meta['SliceTotal'] = 0
test_meta['SliceTotal'] = test_meta['StudyInstanceUID'].map(slice_max)

# Reversed indicator
z_reversed = ((test_meta.groupby('StudyInstanceUID')['ImagePositionPatient_z'].first()-test_meta.groupby('StudyInstanceUID')['ImagePositionPatient_z'].last())<0).astype('int')
test_meta['Reversed'] = 0
test_meta['Reversed'] = test_meta['StudyInstanceUID'].map(z_reversed)

# Preview
test_meta.head(3)


# # Helper functions

# In[14]:


def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = pydicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img


# # Torch dataset

# In[15]:


# Dataset for test set only
class RSNADataset_test(Dataset):
    # Initialise
    def __init__(self, df_table = test_meta, transform=None):
        super().__init__()
        self.df_table = df_table.reset_index(drop=True)
        self.transform = transform
        self.test_path = '../input/rsna-2022-cervical-spine-fracture-detection/test_images'
        self.meta_cols = ['SliceRatio','SliceTotal','SliceThickness','ImagePositionPatient_x','ImagePositionPatient_y','ImagePositionPatient_z','Reversed']
        
    # Get item in position given by index
    def __getitem__(self, index):
        # Load image
        path = os.path.join(self.test_path, self.df_table.iloc[index].StudyInstanceUID, f'{self.df_table.iloc[index].Slice}.dcm')
        img = load_dicom(path)[0]
        
        # Data augmentations
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor
        img = torch.from_numpy(img.astype('float32'))
        
        # Metadata
        meta = torch.as_tensor(self.df_table.iloc[index][self.meta_cols].astype('float32').values)
        
        return img, meta

    # Length of dataset
    def __len__(self):
        return len(self.df_table)


# In[16]:


test_dataset = RSNADataset_test(df_table = test_meta, transform=A.Resize(*IMG_SIZE, interpolation=cv2.INTER_LINEAR))


# # Torch dataloaders

# In[17]:


# Dataloader
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# # Model

# In[18]:


class ImageMetaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Image model layers
        effnet = tv.models.efficientnet_v2_s()
        self.effnet = create_feature_extractor(effnet, ['flatten'])
        self.img_linear = nn.Linear(1280, 64)
        self.swish = nn.SiLU()
        
        # Metadata model layers
        self.tab_linear1 = nn.Linear(7, 128)
        self.tab_linear2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        
        # Combined model layers
        self.drop = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 7)
        self.linear3 = nn.Linear(256, 7)
        
        
    # Forward pass
    def forward(self, img, meta):
        # Image model
        x_img = self.effnet(img)['flatten']
        x_img = self.img_linear(x_img)
        x_img = self.swish(x_img)
        
        # Metadata model
        x_meta = self.tab_linear1(meta)
        x_meta = self.relu(x_meta)
        x_meta = self.tab_linear2(x_meta)
        x_meta = self.relu(x_meta)
        
        # Concatenate
        x = torch.cat([x_img, x_meta], dim=1)
        
        # Combined model
        x = self.drop(x)
        x = self.linear1(x)
        #x = self.relu(x)
        
        # Split
        x_frac = self.linear2(x)
        x_vert = self.linear3(x)
        
        return x_frac, x_vert

    def predict(self, img, meta):
        frac, vert = self.forward(img, meta)
        return torch.sigmoid(frac), torch.sigmoid(vert)

model = ImageMetaModel().to(device)


# ### Load model

# In[19]:


# Load checkpoint
PATH='../input/rsna-trained-2d-models-pytorch/EffNet_model_fold0_epoch6g.pt'
checkpoint = torch.load(PATH, map_location=torch.device(device))

# Load states
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Evaluation mode
model.eval()
print('')


# In[20]:


# Print final loss and epoch
print('Final epoch:', epoch)
print('Final loss:', loss)


# # Make predictions

# In[21]:


def get_predictions(model, data_loader):
    '''Make model predictions'''
    with torch.no_grad():
        predictions = []
        for idx, (img, meta) in enumerate(tqdm(data_loader)):
            y1, y2 = model.predict(img.to(device), meta.to(device))
            pred = torch.cat([y1, y2], dim=1)
            predictions.append(pred)
        return torch.cat(predictions).cpu().numpy()


# In[22]:


preds = get_predictions(model, test_loader)


# In[23]:


df_preds = pd.DataFrame(data=preds, columns=[f'C{i}_frac' for i in range(1, 8)] + [f'C{i}_vert' for i in range(1, 8)])
df_preds_concat = pd.concat([test_meta.reset_index(drop=True).loc[:,['StudyInstanceUID','Slice']], df_preds], axis=1)
df_preds_concat.head(3)


# ### Visualise predictions

# In[24]:


def plot_sample_patient(df_pred):
    patient = '1.2.826.0.1.3680043.22327'
    df = df_pred.query('StudyInstanceUID == @patient').reset_index(drop=True)
    
    plt.figure(figsize=(24,5))
    df[[f'C{i}_frac' for i in range(1, 8)]].plot(
        title=f'{patient}, fracture prediction',
        ax=(plt.subplot(1, 2, 1)))

    df[[f'C{i}_vert' for i in range(1, 8)]].plot(
        title=f'{patient}, vertebrae prediction',
        ax=plt.subplot(1, 2, 2)
    )

plot_sample_patient(df_preds_concat)


# ### Patient overall

# In[25]:


def patient_prediction(df):
    c1c7 = np.average(df[[f'C{i}_frac' for i in range(1, 8)]].values, axis=0, weights=df[[f'C{i}_vert' for i in range(1, 8)]].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return pd.Series(data=np.concatenate([[pred_patient_overall], c1c7]), index=['patient_overall'] + [f'C{i}' for i in range(1, 8)])


# In[26]:


df_patient_pred = df_preds_concat.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df))
df_patient_pred.head()


# ### Post-processing

# In[27]:


# https://stats.stackexchange.com/questions/214877/is-there-a-formula-for-an-s-shaped-curve-with-domain-and-range-0-1
def squish(x, beta=3):
    return 1/(1+(x/(1-x))**(-beta))


# In[28]:


BETA = 0.8
#df_patient_pred['patient_overall'] = df_patient_pred['patient_overall'].apply(lambda x : squish(x, beta=BETA))
#df_patient_pred.head()


# # Submission

# In[29]:


# Melt table
pred_melt = df_patient_pred.reset_index().melt(id_vars='StudyInstanceUID', value_vars=['C1','C2','C3','C4','C5','C6','C7','patient_overall'], value_name='fractured', var_name='prediction_type')

# Merge predictions
submission_df = test_df.merge(pred_melt, how='inner', on=['StudyInstanceUID','prediction_type'])

# Save to csv
submission = submission_df[['row_id','fractured']]
submission.to_csv('submission.csv', index=False)
submission.head(3)

