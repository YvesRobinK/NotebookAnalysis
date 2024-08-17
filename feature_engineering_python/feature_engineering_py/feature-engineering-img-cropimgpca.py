#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd
from glob import glob
import shutil, os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import IncrementalPCA
from tqdm.notebook import tqdm
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


# In[2]:


df = pd.read_csv("../input/yolopred/objectdetection.csv",index_col=0)
df = df[df.Index==1].drop("Index",axis=1).reset_index(drop=True)
df["xmin"] = df["x"] - df["w"]/2
df["xmax"] = df["x"] + df["w"]/2
df["ymin"] = df["y"] - df["h"]/2
df["ymax"] = df["y"] + df["h"]/2
df.loc[df.xmin<0,"xmin"] = 0
df.loc[df.ymin<0,"ymin"] = 0
df.loc[df.xmax>1,"xmax"] = 1.0
df.loc[df.ymax>1,"ymax"] = 1.0
df


# In[ ]:





# # Crop test

# In[3]:


for index,row in df.sample(n=3).iterrows():
    image = cv2.imread(f"/kaggle/input/petfinder-pawpularity-score/train/{row.Id}.jpg")
    h,w,_ = image.shape
    xmin, ymin, xmax, ymax = int((row.xmin)*w), int((row.ymin)*h), int((row.xmax)*w), int((row.ymax)*h)
    plt.imshow(image)
    plt.show()
    plt.imshow(image[ymin:ymax,xmin:xmax,:])
    plt.show()


# In[4]:


#PCA part

#define dataset
class dataset(Dataset):
    def __init__(self,df):
        self.df = df
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = df.iloc[idx]
        path = f"/kaggle/input/petfinder-pawpularity-score/train/{row.Id}.jpg"
        image = cv2.imread(path)
        h,w,_ = image.shape
        xmin, ymin, xmax, ymax = int((row.xmin)*w), int((row.ymin)*h), int((row.xmax)*w), int((row.ymax)*h)
        cropimage = image[ymin:ymax,xmin:xmax,:]
        image = cv2.resize(image,(224,224))
        image = image.flatten()
        
        cropimage = cv2.resize(cropimage,(128,128))
        cropimage = cropimage.flatten()
        
        return image, cropimage
        
def get_loader(df):                   
    datasets = dataset(df=df)
    loader = torch.utils.data.DataLoader(datasets, batch_size=batchsize,shuffle=False, num_workers=2)
    return loader

def make_model(df):
    loader = get_loader(df)
    transform = IncrementalPCA(n_components=PCAcomp, batch_size=batchsize)
    transform_crop = IncrementalPCA(n_components=PCAcomp, batch_size=batchsize)
    for idx,(image,cropimage) in enumerate(loader):
        transform.partial_fit(image)
        transform_crop.partial_fit(cropimage)
        print(idx)
        
    
    with open('model.pickle', mode='wb') as fp:
        pickle.dump(transform, fp)
    
    with open('cropmodel.pickle', mode='wb') as fp:
        pickle.dump(transform_crop, fp)
    
    return transform, transform_crop
    
def data_transform(df,model,cropmodel):
    loader = get_loader(df)
    arr = np.empty((0,PCAcomp))
    arr_crop = np.empty((0,PCAcomp))
    labellist = []
    for idx,(image, cropimage) in enumerate(loader):
        com_img = model.transform(image)
        com_img = cropmodel.transform(cropimage)
        arr = np.append(arr,com_img,axis=0)
        arr_crop = np.append(arr_crop,com_img,axis=0)
        if idx%5==0:
            print(idx)
    return arr,arr_crop


# In[5]:


batchsize = 512
PCAcomp = 30


# In[6]:


model,cmodel = make_model(df)


# In[7]:


arr,arr_crop = data_transform(df,model,cmodel)


# In[8]:


croppcadf = pd.DataFrame(arr_crop)
croppcadf["Id"] = df.Id.values

pcadf = pd.DataFrame(arr)
pcadf["Id"] = df.Id.values


# In[9]:


display(croppcadf,pcadf)


# In[10]:


pcadf = pd.merge(pcadf,croppcadf,on=["Id"],suffixes=("","_crop"))
display(pcadf)


# In[11]:


df = pd.merge(df,pcadf,on=["Id"],suffixes=("",""))
display(df)


# In[12]:


df.to_csv("train.csv")


# In[ ]:




