#!/usr/bin/env python
# coding: utf-8

# This notebook illustrates how to parallelize the process of meta-feature extraction using `joblib` to get meta-data and some pixel-based features from the training images.
# 
# The extracted features are used in [this notebook](https://www.kaggle.com/kozodoi/lightgbm-on-meta-features) that develops a LightGBM pipeline on top of extracted features.
# 
# The pipeline is inspired by [this notebook](https://www.kaggle.com/swarajshinde/rsna-pulmonary-embolism-analysis-eda-meta-data#Extracting-Meta-Data-from-Dicom-Data-and-Storing-in-CSV-Format). You can also check out [this notebook](https://www.kaggle.com/teeyee314/rsna-pe-metadata-with-multithreading?select=test_meta.csv) for an alternative approach to meta-feature extraction with `multiprocessing`.

# In[1]:


##### PACKAGES

import os
import numpy as np
import pandas as pd

import pydicom as dcm
import PIL

from tqdm import tqdm
from joblib import Parallel, delayed

import cv2
import vtk
from vtk.util import numpy_support


# In[2]:


##### IMAGE PATH

im_path = []
train_path = '../input/rsna-str-pulmonary-embolism-detection/train/'
for i in tqdm(os.listdir(train_path)): 
    for j in os.listdir(train_path + i):
        for k in os.listdir(train_path + i + '/' + j):
            x = i + '/' + j + '/' + k
            im_path.append(x)
len(im_path)


# In[3]:


##### EXTRACT META-FEATURES

def window(img, WL = 50, WW = 350):
    upper, lower = WL + WW//2, WL - WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    return X

def extract_meta_feats(img):

    img_id = img.split('/')[2].replace('.dcm', '')
    image  = dcm.dcmread(train_path + img)
    
    ### META-FEATURES
    
    pixelspacing      = image.PixelSpacing[0]
    slice_thicknesses = image.SliceThickness
    kvp               = image.KVP
    table_height      = image.TableHeight
    x_ray             = image.XRayTubeCurrent
    exposure          = image.Exposure
    modality          = image.Modality
    rot_direction     = image.RotationDirection 
    instance_number   = image.InstanceNumber
    
    
    ### PIXEL-BASED FEATURES
    
    reader = vtk.vtkDICOMImageReader()
    reader.SetFileName(train_path + img)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()
    imageData  = reader.GetOutput()
    pointData  = imageData.GetPointData()
    arrayData  = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order = 'F')
    ArrayDicom = cv2.resize(ArrayDicom, (512,512))

    img = ArrayDicom.astype(np.int16)
    img[img <= -1000] = 0

    intercept = reader.GetRescaleOffset()
    slope     = reader.GetRescaleSlope()
    if slope != 1:
        img = slope * img.astype(np.float64)
        img = img.astype(np.int16)
    img += np.int16(intercept)

    hu_min  = np.min(img)
    hu_mean = np.mean(img)
    hu_max  = np.max(img)
    hu_std  = np.std(img)
    
    
    ### WINDOW-BASED FEATURES
    
    img_lung = window(img, WL = -600, WW = 1500)
    img_medi = window(img, WL = 40,   WW = 400)
    img_pesp = window(img, WL = 100,  WW = 700)
    
    lung_mean = np.mean(img_lung)
    lung_std  = np.std(img_lung)
    
    medi_mean = np.mean(img_medi)
    medi_std = np.std(img_medi)
    
    pesp_mean = np.mean(img_pesp)
    pesp_std  = np.std(img_pesp)
    
    
    return [img_id, 
            pixelspacing, slice_thicknesses, kvp, table_height, x_ray, exposure, modality, rot_direction, instance_number,
            hu_min, hu_mean, hu_max, hu_std,
            lung_mean, lung_std, medi_mean, medi_std, pesp_mean, pesp_std
           ]

results = Parallel(n_jobs = -1, verbose = 1)(map(delayed(extract_meta_feats), im_path))


# In[4]:


##### SAVE FEATURES

df = pd.DataFrame(results, columns = ['SOPInstanceUID', 
                                      'pixelspacing',
                                      'slice_thicknesses',
                                      'kvp',
                                      'table_height',
                                      'x_ray_tube_current',
                                      'exposure',
                                      'modality',
                                      'rotation_direction',
                                      'instance_number',
                                      'hu_min',
                                      'hu_mean',
                                      'hu_max',
                                      'hu_std',
                                      'lung_mean',
                                      'lung_std',
                                      'medi_mean',
                                      'medi_std',
                                      'pesp_mean',
                                      'pesp_std'])
df.to_csv('train_meta.csv', index = False)
df.head()

