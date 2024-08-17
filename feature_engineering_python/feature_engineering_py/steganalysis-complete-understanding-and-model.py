#!/usr/bin/env python
# coding: utf-8

# # About this Competition
# 
# "Wait!What? Another one , No man this can't be happening", Calm down there Pal! , it's just the quarantine , It's messing with Kaggle.
# Kaggle is going wild with competitons releasing 4th competiton this week, but we need to support Kaggle as good community members right? I mean these are tough times 😛 .
# This is an interesting competition though,I mean Stegnography, now we are talking, it's time to give all those suppressed spy fantasies we have had for a long time , some air  😍 and dive right in .
# 
# This competition wants us to create an efficient and reliable method to detect secret data hidden within innocuous-seeming digital images.

# # About this Notebook
# 
# What most people might think : This competition is not for me ,I'm having zero understanding and knowledge of stegnography and stegnanalysis, but we are not like them , are we? <br>
# Well I leave it for you to decide , as for me I am not , I am eager to learn,explore and most of all get the feel of an NSA agent
# Thus I will dive right in and in this notebook I will build on the concept from zero to publishing a solution. We can do this together if you want to , just come along,because all it requires is willingness to learn.<br><br>
# 
# We will see the following in this notebook :-
# 
# * What's Steganagraphy and Steganalysis? : Basic Definitions
# * Knaive approach for Steganalysis
# * Had we been doing it all wrong?
# * Steganography : Complete Understandind of what it is and why we were doing it all wrong?
# * New and Improved approach of solution
# 
# By the end , You will have all the understanding of the data and the competition and would be well equipped to do well in this competition, just stay patient
# 
# This notebook is a work in progress , if this notebook receives support , I will keep on updating it with new and latest developments which I make in the this competition
# 
# **<span style="color: red;">If you like my effort and contribution please show token of gratitude by upvoting my kernel</span>**

# # PRELIMINARIES

# In[1]:


get_ipython().system('pip install stegano')


# In[2]:


# PRELIMINARIES
import os
from stegano import lsb #USED FOR PNG IMAGE
import skimage.io as sk
import matplotlib.pyplot as plt
from scipy import spatial
from tqdm import tqdm

from PIL import Image
from random import shuffle


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# # Q1) What is Steganography?
# 
# Let's dive in by understanding our first point
# 
# Well in pure english it has the following meaning : the practice of concealing messages or information within other non-secret text or data.
# 
# Here is how one research paper describes it : "Steganography is usually referred to as the techniques and methods that allow to hide information within an innocuous-like cover object. The resulting stego-object resembles, as much as possible, the original cover object. Therefore it can be sent over an unsecured communication channel that may be subject to wiretapping by an eavesdropper"
# 
# In context to this competition it mean hiding information in the digital image.The same technology employed for digital watermarking and for digital markings/copyrights .But it's not limited to it , anyone can use it hide information , to bypass securities , contact specific person, send malicious content ,etc .Now you see right , why I was talking about all the spy work.
# 
# Nowadays, steganography has been mostly developed for digital images because of their massive presence over the
# Internet, the universally adopted JPEG compression scheme and
# its relative simplicity to be modified
# 
# ## Difference B/w steganalysis and steganography
# 
# Steganography is the encoding of information on a viable object source and steganography is the decoding of information from the encoded object.

# ## Fun Exercise
# Before we begin with the original dataset and the problem of steganoanalysis, let's first try to perform steganography just for fun, using a python module called stegano and after doing steganography, we will understand how it was done.
# 
# More can be read here from docs [here](https://stegano.readthedocs.io/en/latest/module/)

# I have added few png images , you can add yours and play around

# In[3]:


image = sk.imread("/kaggle/input/png-for-steg/bald-eagle-png-transparent-image-pngpix-7.png")


# In[4]:


secret = lsb.hide("/kaggle/input/png-for-steg/bald-eagle-png-transparent-image-pngpix-7.png", "I will be there but you can't find me even if I'm a very very very long sentence")
secret.save("encoded.png")


# We just hid a message in our image with two lines of code , Cool huh? <br>
# Let's try looking at the image to observe if there is any difference

# In[5]:


img1 = sk.imread("/kaggle/input/png-for-steg/bald-eagle-png-transparent-image-pngpix-7.png")
img2 = sk.imread("/kaggle/working/encoded.png")

fig,ax = plt.subplots(1,2,figsize=(18,8))
    
ax[0].imshow(img1)
ax[1].imshow(img2)


# **Surpised? Can't tell the difference can we? This is exceptional , Now we why it is so difficult**  <br>
# Steagano also provides a function for decoding the message hidden in our image , Let's Try that

# In[6]:


print(lsb.reveal("/kaggle/working/encoded.png"))


# ### So we can see the text that we hid, But What's going on under the hood?
# 
# The module uses a technique that create a covert channel in the parts of the cover image in which changes are likely to be a bit scant when compared to the human visual system (HVS).It hides the information in the least significant bit (LSB) of the image data . This embedding method is basically based on the fact that the least significant bits in an image can be thought of as random noise, and consequently they become not responsive to any changes on the image <br>
# 
# This is one of the first and classical ways of doing stegnagraphy on images .The well-known steganographic tools based on LSB embedding are different as far as the way they hide information is concerned. Some of them change the LSB of pixels randomly, others modify pixels not in the whole image but in selected areas of it, and still others increase or decrease the pixel value of the LSB, rather than change the value

# # Knaive Approach
# 
# So now that we understand how stegano works and how one of the first and oldest way of stegnanalyis work , we can use a knaive approach for detecting whether a message has been encoded in a image file or not. The following presents a knaive approach :
# 
# 
# * An RGB image is stored in a form of a 3-D numpy array where each dimensions contains pixels of Red,Green and blue colors.The values in this vary between 0 and 255. We can flatten the numpy array into a vaactor of rdim x gdim x bdim . This will get us a feature vector for an img
# * Now we can find the cosine similarity between the two vectors of our normal image and encoded image, if they are alike then cosine dissimilarity(1-similarity) must be one else it will be less than 1 and we can get an idea that's something is wrong
# 
# Let's do this exercise and see the results

# In[7]:


vec1 = np.reshape(img1,(1151*1665*4))
vec2 = np.reshape(img2,(1151*1665*4))


# Let's first verify our argument that cosine similarity is 1 for exact same vectors

# In[8]:


print(1 - spatial.distance.cosine(vec1,vec1))


# In[9]:


print(1 - spatial.distance.cosine(vec1,vec2)) #Cosine Similarity


# In[10]:


print(spatial.distance.cosine(vec2,vec1)) #Cosine Dissimilarity


# * So we can see that we can find whether an information is hidden or not using a naive approach without much efforts .
# * **I  tried to use this knaive with the main dataset , but was made aware by the organizers that this knaive approach is not wrong but not exactly right because the information is hidden in the DCT coefficients of the image rather than the RGB pixels** 
# 
# * Don't worry if all of this does not make sense, we will look into everything , but first look at some visualizations

# # Data Exploration
# 
# Now that we are familiar to the techniques of steganograhy , we can now move to exploration of data and steganalysis part

# In[11]:


BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)
test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)


# In[12]:


cover_images_path = pd.Series(BASE_PATH + '/Cover/' + train_imageids ).sort_values(ascending=True)
JMIPOD_images_path = pd.Series(BASE_PATH + '/JMiPOD/'+train_imageids).sort_values(ascending=True)
JUNIWARD_images_path = pd.Series(BASE_PATH + '/JUNIWARD/'+train_imageids).sort_values(ascending=True)
UERD_images_path = pd.Series(BASE_PATH + '/UERD/'+train_imageids).sort_values(ascending=True)
test_images_path = pd.Series(BASE_PATH + '/Test/'+test_imageids).sort_values(ascending=True)
ss = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


# So the organizers have used four algorithms for encoding data into the cover images, they are [JUNIWARD , JMiPOD, UERD] 
# * UNIWARD -  Universal Wavelet Relative Distortion<br>
# Paper describing CNN for stegnalysis of [JUNIWARD images](https://arxiv.org/ftp/arxiv/papers/1704/1704.08378.pdf)
#  
# * JMiPOD -
#  Paper describing CNN for stegnalysis of [JMiPOD images](https://hal-utt.archives-ouvertes.fr/hal-02542075/file/J_MiPOD_vPub.pdf)
#  
# * UERD - Uniform Embedding Revisited Distortion
# <br> No resources found yet , I will update as soon as I find something
# 
# For now just assume they are certain algorithms that are used for encoding data into image we will understand everything in a bit

# In[13]:


#VISUALIZING SOME IMAGES FROM COVER SECTION
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(30, 15))
k=0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        img = sk.imread(cover_images_path[k])
        col.imshow(img)
        col.set_title(cover_images_path[k])
        k=k+1
plt.suptitle('Samples from Cover Images', fontsize=14)
plt.show()


# ## Visualizing Cover and Encoded side by side
# 
# Let's visualize cover image and encoded images side by side

# In[14]:


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(30, 15))
for i in range(3):
    '''
    If you want to print more images just change the values in range and ncols in subplot
    
    '''
    cvimg = sk.imread(cover_images_path[i])
    uniimg = sk.imread(JUNIWARD_images_path[i])
    jpodimg = sk.imread(JMIPOD_images_path[i])
    uerdimg = sk.imread(UERD_images_path[i])
    
    ax[i,0].imshow(cvimg)
    ax[i,0].set_title('Cover_IMG'+train_imageids[i])
    ax[i,1].imshow(uniimg)
    ax[i,1].set_title('JNIWARD_IMG'+train_imageids[i])
    ax[i,2].imshow(jpodimg)
    ax[i,2].set_title('JMiPOD_IMG'+train_imageids[i])
    ax[i,3].imshow(uerdimg)
    ax[i,3].set_title('UERD_IMG'+train_imageids[i])


# **As expected we can't see any difference with naked eye ,We know that information is hidden in somewhere using distortion function**<br>
# <br>
# I will try to do one final visualization of pixel deviation in the image channels and see if it works 
# Idea taken shamelessly from: https://www.kaggle.com/iamleonie/alaska2-first-look-into-the-wild-data

# In[15]:


img_cover = sk.imread(cover_images_path[0])
img_jmipod = sk.imread(JMIPOD_images_path[0])
img_juniward = sk.imread(JUNIWARD_images_path[0])
img_uerd = sk.imread(UERD_images_path[0])


fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
ax[0,0].imshow(img_jmipod)
ax[0,1].imshow((img_cover == img_jmipod).astype(int)[:,:,0])
ax[0,1].set_title(f'{train_imageids[k]} Channel 0')

ax[0,2].imshow((img_cover == img_jmipod).astype(int)[:,:,1])
ax[0,2].set_title(f'{train_imageids[k]} Channel 1')
ax[0,3].imshow((img_cover == img_jmipod).astype(int)[:,:,2])
ax[0,3].set_title(f'{train_imageids[k]} Channel 2')
ax[0,0].set_ylabel('JMiPOD', rotation=90, size='large', fontsize=14)


ax[1,0].imshow(img_juniward)
ax[1,1].imshow((img_cover == img_juniward).astype(int)[:,:,0])
ax[1,2].imshow((img_cover == img_juniward).astype(int)[:,:,1])
ax[1,3].imshow((img_cover == img_juniward).astype(int)[:,:,2])
ax[1,0].set_ylabel('JUNIWARD', rotation=90, size='large', fontsize=14)

ax[2,0].imshow(img_uerd)
ax[2,1].imshow((img_cover == img_uerd).astype(int)[:,:,0])
ax[2,2].imshow((img_cover == img_uerd).astype(int)[:,:,1])
ax[2,3].imshow((img_cover == img_uerd).astype(int)[:,:,2])
ax[2,0].set_ylabel('UERD', rotation=90, size='large', fontsize=14)

plt.suptitle('Pixel Deviation from Cover Image', fontsize=14)

plt.show()


# * So Again, on doing these visualizations , the organizers again made the point that comparing the pixel values of RGB channels of normal and stego images is not exactly right.
# https://www.kaggle.com/c/alaska2-image-steganalysis/discussion/147494 ---> Please read this before Continuing , don't worry if you don't understand it , we will dig deeper
# 
# To understand everything we will have to dive deeper into the underlying principles of JPEG and Stregnography

# # Had we been doing it all wrong?
# 
# 1)**We will understand everything step by step one at a time , so let's start with what is 'JPEG'?**
# 
# So JPEG is not a file format , it's a compressing algorithm for an image file to reduce it's size , without losing a lot of information. So let's look at the steps invloved in compressing an image using JPEG image :
# 
# ![image.png](attachment:image.png)

# * Firstly the image is converted into YCbCr from RGB channels. YCbCr and RGB are both colorspaces having different channels where YCbCr consists three channels as Luminance(Y) , Cb(Cb is blue minus luma (B-Y)) ,  Cr(Cr is red minus luma (R-Y)). To know more about color space please watch this [video][1]
# [1]: https://www.youtube.com/watch?v=LFXN9PiOGtY&list=PLzH6n4zXuckoAod3z31QEST1ZaizBuNHh&index=1
# 
# * Then DCT is applied on the pixels of these channels , using DCT coeff , To understand DCT please watch this [video][2]
# [2]: https://www.youtube.com/watch?v=Q2aEzeMDHMA&list=PLzH6n4zXuckoAod3z31QEST1ZaizBuNHh&index=3
# 
# * The image encoded using JPEG algorithm stays in YCbCr colorspace untill it is decoded by an Image viewer software. When a JPEG is read it is decoded and converted back to RGB colorspace to be rendered on screen using the techniques described in the video above

# # Stenography : Complete Understanding
# 
# Now that we know  how JPEG works , let's understand how stegnography works and how techniques used have evolved over the years
# 
# So what do you think can be the potential hiding spots of information in a JPEG File ?
# * Channels of the respective color Spaces like RGB and YCbCr.<br>
#   These were the techniques used in the starting algorithms but as we have seen it is easily recognizeable even using knaive approach , so people started to find a new technique for hiding information 
# <br>
# * DCT coeffs of different channels<br>
#   The more newer approaches hides information in the DCT coeffs of different channels of JPEG image and that too the payload(secret data) is randomly distributed among them taking into the statistics of DCT coeffs
# 
# After watching this [video](https://www.youtube.com/watch?v=TWEXCYQKyDc&t=342s) you will have a clear understanding of everything that has been discussed uptill and now  

# # What happened to the Knaive approach
# I have made this notebook new again from scratch , as for my knaive approach to this problem i.e approaching this as regreesion problem based on Pixel Difference, I have added it here if you want to know about it here: https://www.kaggle.com/tanulsingh077/creating-labels-for-steganalysis/
# 
# # New and Improved Approach
# So now we have got to approache for model building :-
# * Use pixels as input of our neural nets as we were doing uptill now and we can also use naive method
#                                              or
# * Use DCT coeff's instead of pixel values as an input to our neural networks for prediction of labels
# 
# 
# In this section we will and try and determine what would be a good idea (What you think about it, let's discuss in the comments)

# ### Let'se First go and see how we can visualize different Channels of a YCbCr colorspace Image

# In[16]:


fig,ax = plt.subplots(3,4,figsize=(20,12))

for i,paths in enumerate(cover_images_path[:3]):
    image = Image.open(paths)
    ycbcr = image.convert('YCbCr')
    (y, cb, cr) = ycbcr.split()

    ax[i,0].imshow(image)
    ax[i,0].set_title('Cover'+train_imageids[i])
    ax[i,1].imshow(y)
    ax[i,1].set_title('Luminance')
    ax[i,2].imshow(cb)
    ax[i,2].set_title('Cb:Chroma Blue')
    ax[i,3].imshow(cr)
    ax[i,3].set_title('Cr:Chroma Red')


# Now let's explore the Different channels of YCbCr of Cover and Encoded Image side by side

# In[17]:


fig,ax = plt.subplots(4,4,figsize=(20,16))
plt.tight_layout()


im1 = Image.open(cover_images_path[0])
im2 = Image.open(JUNIWARD_images_path[0])
im3 = Image.open(JMIPOD_images_path[0])
im4 = Image.open(UERD_images_path[0])

for i,image in enumerate([im1,im2,im3,im4]):
    ycbcr = image.convert('YCbCr')
    (y, cb, cr) = ycbcr.split()

    ax[i,0].imshow(image)
    ax[i,0].set_title('Image')
    ax[i,1].imshow(y)
    ax[i,1].set_title('Luminance')
    ax[i,2].imshow(cb)
    ax[i,2].set_title('Cb:Chroma Blue')
    ax[i,3].imshow(cr)
    ax[i,3].set_title('Cr:Chroma Red')


# After covering all the videos in the above sections , we know how an Image is compressed using JPEG and how DCT coeffs are calculated using Quantization and Encoding
# 
# Now we can go deeper and analyze for which model would be better . We will start this by getting the DCT coeffs  of images and visualizing them

# In[18]:


get_ipython().system(' git clone https://github.com/dwgoon/jpegio')


# In[19]:


get_ipython().system('pip install jpegio/.')
import jpegio as jio


# Usage and review of the Library can be viewed [here](https://github.com/dwgoon/jpegio)
# * The attribute coeff_arrays gives the DCT coeffs of all the three channels of YCbCr of the images , Let's go and plot DCT coeffs of Cover and Encoded Image and try to see if there is any difference

# In[20]:


fig,ax = plt.subplots(4,4,figsize=(20,16))
plt.tight_layout()

for i,path in enumerate([cover_images_path[0],JUNIWARD_images_path[0],JMIPOD_images_path[0],UERD_images_path[0]]):
    
    image = Image.open(path)
    jpeg = jio.read(path)
    DCT_Y = jpeg.coef_arrays[0]
    DCT_Cr = jpeg.coef_arrays[1]
    DCT_Cb = jpeg.coef_arrays[2]
    
    
    ax[i,0].imshow(image)
    ax[i,0].set_title('Image')
    ax[i,1].imshow(DCT_Y)
    ax[i,1].set_title('Luminance')
    ax[i,2].imshow(DCT_Cb)
    ax[i,2].set_title('Cb:Chroma Blue')
    ax[i,3].imshow(DCT_Cr)
    ax[i,3].set_title('Cr:Chroma Red')


# Up untill now we were using difference in pixel values and normal pixel values for analysis and it has been argued that's it is not the right way 
# Lets now look at difference in DCT values and difference in pixels and see the differences, this will help us to make better  decision 

# In[21]:


coverDCT = np.zeros([512,512,3])
stegoDCT = np.zeros([512,512,3])
jpeg = jio.read(cover_images_path[0])
stego_juni = jio.read(JUNIWARD_images_path[0])


# In[22]:


coverDCT[:,:,0] = jpeg.coef_arrays[0] ; coverDCT[:,:,1] = jpeg.coef_arrays[1] ; coverDCT[:,:,2] = jpeg.coef_arrays[2]
stegoDCT[:,:,0] = stego_juni.coef_arrays[0] ; stegoDCT[:,:,1] = stego_juni.coef_arrays[1] ; stegoDCT[:,:,2] = stego_juni.coef_arrays[2]

DCT_diff = coverDCT - stegoDCT
# So since they are not the same Images the DCT_diff would not be zero
print(len(DCT_diff[np.where(DCT_diff!=0)]))
print(np.unique(DCT_diff))
plt.figure(figsize=(16,10))
plt.imshow( abs(DCT_diff) )
plt.show()


# In[23]:


coverPixels = np.array(Image.open(cover_images_path[0])).astype('float')
stegoPixels = np.array(Image.open(JUNIWARD_images_path[0])).astype('float')

pixelsDiff = coverPixels - stegoPixels

# So since they are not the same Images the pixels_diff would not be zero
print(len(pixelsDiff[np.where(pixelsDiff!=0)]))
print(np.unique(pixelsDiff))
plt.figure(figsize=(16,10))
plt.imshow( abs(pixelsDiff) )
plt.show()


# * From the images it is hard to conclude which one works better, as images of both DCT and Pixel pattern give us the same underlying pattern. But one can argue that pixel values contain a lot of noise whereas DCT values do not and can get you a better model
# 
# * But from the number of non-zero difference values of both DCT and pixels we can see that Pixels have more non-zero values than DCT and hence are more distintive for differentiation between cover and stego images
# 
# * Let's also look the images side by side to get a better unders

# In[24]:


fig,ax = plt.subplots(1,2,figsize=(16,12))
ax[0].imshow(abs(DCT_diff))
ax[1].imshow(abs(pixelsDiff))


# # What's Next
# 
# * Building a baseline with pixel difference and my naive way as I have discussed in 8th version of my notebook 
# * Building a baseline with pixel value only and with DCT values and seeing which works better
# * After deciding the approach using SOTA models for perfections
# * More insights that would help us
# 
# If you have read this kernel , do share your  thoughts on which would be a better way to model in the comments below
# 
# STAY TUNED !!!

# In[ ]:




