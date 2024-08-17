#!/usr/bin/env python
# coding: utf-8

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; border:0; color:red' role="tab" aria-controls="home"><center>MNIST Pytorch: Convoluton Neural Networks</center></h2>
# 

# `We are using Mnist kaggle training set as our training dataset. It consists of 28px by 28px grayscale images of handwritten digits (0 to 9), along with labels for each image indicating which digit it represents. Here are some sample images from the dataset:`
# 
# ![mnist-sample](https://i.imgur.com/CAYnuo1.jpg)
# 
# `Our goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.`

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:lightgray; border:0; color:black' role="tab" aria-controls="home"><center>Table of Contents</center></h2>
# 
#     
# - [Import Libaries](#2)
# - [Data Augmentation](#3)     
# - [Pytorch Dataset Classes](#4)
# - [Feature Engineering](#5)
# - [CNN(LeNet5)](#6)
# - [Training and Validation losses](#7)
# - [Prediction & Submission](#8)
# 

# <a id="2"></a>
# # Import Libraries

# In[1]:


import warnings 
warnings.filterwarnings('ignore')
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper", font_scale = 1, rc={"grid.linewidth": 3})
pd.set_option('display.max_rows', 100, 'display.max_columns', 400)
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from torchvision import transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn as nn


# In[2]:


train_data=pd.read_csv('../input/digit-recognizer/train.csv')
test_data=pd.read_csv('../input/digit-recognizer/test.csv')
sample_data = pd.read_csv('../input/digit-recognizer/sample_submission.csv')


# In[3]:


print(train_data.info())
print('\n')
print(test_data.info())
print('\n')
print(sample_data.info())


# The dataset has 42,000 images for train data which can be used to train the model. 28,000 images for test set.

# In[4]:


train_df = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values
test_df = test_data.values


# In[5]:


image_1 = train_df.reshape(train_df.shape[0], 28, 28) #rehsaping it to plot image
plt.figure(figsize=(20,8))
for i in range(10,18):
    plt.subplot(231 + (i))
    plt.imshow(image_1[i], cmap="gray")
    plt.title('Label:'+str(y_train[i]),fontweight='bold',size=20)


# It's evident that these images are quite small in size, and recognizing the digits can sometimes be hard even for the human eye. While it's useful to look at these images, there's just one problem here: PyTorch doesn't know how to work with images. We need to convert the images into tensors. We can do this by specifying a transform while creating our dataset.

# PyTorch datasets allow us to specify one or more transformation functions which are applied to the images as they are loaded. `torchvision.transforms` contains many such predefined functions, and we'll use the `ToTensor` transform to convert images into PyTorch tensors.

# <a id="3"></a>
# # Data Augmentation 

# In[6]:


img_tform_1 = transforms.Compose([
    transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_2 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomRotation(10),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_3 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomRotation(20),transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_4 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.85,0.85)),\
    transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_5 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomAffine(0,shear=30,scale=[1.15,1.15]),\
    transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_6 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomAffine(0,shear=20,scale=[0.8,0.8]),\
    transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

img_tform_7 = transforms.Compose([
    transforms.ToPILImage(),transforms.RandomAffine(degrees=30, scale=(1.2,1.2)),\
    transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])


# <a id="4"></a>
# # Pytorch Dataset Classes

# In[7]:


class MnistDataset(Dataset):
    #it takes whatever arguments needed to build a list of tuples — it may be the name of a CSV file that will be loaded and processed; it may be two tensors, one for features, another one for labels; or anything else, depending on the task at hand.
    def __init__(self, features,transform=img_tform_1): 
        self.features = features.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)
        self.targets = torch.from_numpy(features.label.values)
        self.transform=transform
        
   #it should simply return the size of the whole dataset so, whenever it is sampled, its indexing is limited to the actual size.
    def __len__(self):
        return (self.features.shape[0])
    #There is no need to load the whole dataset in the constructor method (__init__). If your dataset is big (tens of thousands of image files, for instance), loading it at once would not be memory efficient. It is recommended to load them on demand (whenever __get_item__ is called).
    # it allows the dataset to be indexed, so it can work like a list (dataset[i]) — it must return a tuple (features, label) corresponding to the requested data point. 
    def __getitem__(self, idx):
        return self.transform(self.features[idx]),self.targets[idx]


# Checking this class    
class TestDataset(Dataset):
    def __init__(self, features,transform=img_tform_1):
        self.features = features.values.reshape((-1,28,28)).astype(np.uint8)
        self.targets = None
        self.transform=transform
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        return self.transform(self.features[idx])


# <a id="5"></a>
# # Feature Engineering

# In[8]:


def create_dataloaders(seed, test_size=0.1, df=train_data, batch_size=32):
    # Create training set and validation set
    train_df, val_df = train_test_split(df,test_size=test_size,random_state=seed)
    
    # Create Datasets
    train_data_1 = MnistDataset(train_df)
    train_data_2 = MnistDataset(train_df, img_tform_2)
    train_data_3 = MnistDataset(train_df, img_tform_3)
    train_data_4 = MnistDataset(train_df, img_tform_4)
    train_data_5 = MnistDataset(train_df, img_tform_5)
    train_data_6 = MnistDataset(train_df, img_tform_6)
    train_data_7 = MnistDataset(train_df, img_tform_7)
    train_final = ConcatDataset([train_data_1, train_data_2, train_data_3, train_data_4, train_data_5,\
                                   train_data_6,train_data_7])

    val_data = MnistDataset(val_df)
    
    # Create Dataloaders
    train_loader = torch.utils.data.DataLoader(train_final, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


# <a id="6"></a>
# # CNN (Custom LeNet5) Model

# In[9]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  #26x26x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True), # inplace=True helps to save some memory
            
            nn.Conv2d(32, 32, kernel_size=3), # 24x24x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=14), # 24x24x32 (same padding)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.MaxPool2d(2, 2), #12x12x32
            nn.Dropout2d(0.25),
        
            nn.Conv2d(32, 64, kernel_size=3), #10x10x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), 
            
            nn.Conv2d(64, 64, kernel_size=3), # 8x8x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), 
            
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=6), # 8x8x64(same padding)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.MaxPool2d(2, 2),# 4x4x64 (half)
            nn.Dropout2d(0.25),
    
            nn.Conv2d(64, 128, kernel_size=4), # 1x128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(0.25)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128*1*1, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 128*1*1)
        x = self.fc(x)
        
        return x


# <a id="7"></a>
# # Training and Validation losses

# In[10]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train() #  set the model to training mode
    final_loss = 0  # Initialise final loss to zero
    train_acc=0
    total=0
    train_preds=[]
    
    for features,labels in dataloader:
        optimizer.zero_grad() #every time we use the gradients to update the parameters, we need to zero the gradients afterwards
        inputs, targets = features.to(device), labels.to(device) #Sending data to GPU(cuda) if gpu is available otherwise CPU
        outputs = model(inputs) #output 
        loss = loss_fn(outputs, targets) #loss function
        loss.backward() #compute gradients(work its way BACKWARDS from the specified loss)
        optimizer.step()  #gradient optimisation
        scheduler.step() #scheduler optimisation
        total+=len(targets)
        final_loss += loss.item() #Final loss
        train_preds.append(outputs.sigmoid().detach().cpu().numpy()) # get CPU tensor as numpy array # cannot get GPU tensor as numpy array directly
        _, predicted = torch.max(outputs, 1)
        train_acc+=((predicted == targets).sum().item())
    final_loss /= len(dataloader) #average loss
    train_preds = np.concatenate(train_preds)#concatenating predictions under train_pred
    train_acc=(train_acc/total)*100
    
    return final_loss,train_acc


def valid_fn(model, loss_fn, dataloader, device):
    model.eval() #  set the model to evaluation/validation mode
    final_loss = 0 # Initialise validation final loss to zero
    valid_preds = [] #Empty list for appending prediction
    val_acc=0
    total=0
    for features,labels in dataloader:
        inputs, targets = features.to(device), labels.to(device) #Sending data to GPU(cuda) if gpu is available otherwise CPU
        outputs = model(inputs) #output
        loss = loss_fn(outputs, targets) #loss calculation
        total+=len(targets)
        final_loss += loss.item() #final validation loss
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy()) # get CPU tensor as numpy array # cannot get GPU tensor as numpy array directly
        _, predicted = torch.max(outputs, 1)
        val_acc+=((predicted == targets).sum().item())
              
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds) #concatenating predictions under valid_preds
    val_acc=(val_acc/total)*100
    
    return final_loss, valid_preds,val_acc


# In[11]:


# HyperParameters

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-8
seed=42
#EARLY_STOPPING_STEPS = 10
#EARLY_STOP = False
#Dropout_model_val=0.2619422201258426


# In[12]:


def run_training(seed):
    # train and data val dataloaders
    train_loader, valid_loader= create_dataloaders(seed=seed)
    model=Model()
    model.to(DEVICE)
    #using adam optimizer for optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e2, 
                                              max_lr=1e-2, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        train_loss,train_acc = train_fn(model, optimizer,scheduler, loss_fn, train_loader, DEVICE) #training loss and accuracy
        print(f"EPOCH: {epoch}, train_loss: {train_loss},, train_accuracy:{train_acc}")
        val_loss, val_preds, val_acc = valid_fn(model, loss_fn, valid_loader, DEVICE) #validation loss and accuracy
        print(f"EPOCH: {epoch}, valid_loss: {val_loss}, val_accuracy:{val_acc}")
        
    test_pred = torch.LongTensor()        
    testdataset = TestDataset(test_data)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    for features in testloader:
        features=features.to(DEVICE)
        outputs=model(features)
        _, predicted = torch.max(outputs, 1)
        test_pred = torch.cat((test_pred.to(DEVICE), predicted.to(DEVICE)), dim=0)
    pred_df['predict'] = test_pred.cpu().numpy()


# In[13]:


pred_df = sample_data.copy()
run_training(seed)


# <a id="8"></a>
# # Prediction & Submission

# In[14]:


#Prediction
final_pred = pred_df['predict']
sample_data.Label = final_pred.astype(int)
sample_data.head()


# In[15]:


sample_data.to_csv('./submission.csv', index=False) # submission file


# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; border:0; color:red' role="tab" aria-controls="home"><center>If you found this notebook helpful , some upvotes would be very much appreciated - That will keep me motivated 😊</center></h2>
# 

# <div class="list-group" id="list-tab" role="tablist">
# <h2 class="list-group-item list-group-item-action active" data-toggle="list" style='background:black; border:0; color:red' role="tab" aria-controls="home"><center>Thank You 😊🙏</center></h2>
# 
