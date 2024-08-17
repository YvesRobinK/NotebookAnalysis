#!/usr/bin/env python
# coding: utf-8

# <a id="imports"></a>
# 
# <h1 style="font-family: Verdana; font-size: 30px; font-style: normal; font-weight: bold; text-decoration: none; text-transform: none; letter-spacing: 1px; background-color: #ffffff; color: #03045e;" id="imports">Start Your CV Competition 🎯&nbsp;&nbsp;&nbsp;&nbsp;<a href="#toc">&#10514;</a></h1>

# ![](http://ammarjaved.com/storage/posts/what-is-computer-vision-applications-of-computer-vision.png)

# ### **First You Need to Know What is CV (Computer Vision)?**
# 
# * Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information. If AI enables computers to think, computer vision enables them to see, observe and understand.
#  
# * Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image.
#  
# * Computer vision trains machines to perform these functions, but it has to do it in much less time with cameras, data and algorithms rather than retinas, optic nerves and a visual cortex. Because a system trained to inspect products or watch a production asset can analyze thousands of products or processes a minute, noticing imperceptible defects or issues, it can quickly surpass human capabilities.
# 
# Link: https://www.ibm.com/topics/computer-vision

# ### **Interpretation:**
# 
# You wonder how we use these images to predict by using Model?
# The Answer you will get is:
# 
# * First we load the data and check the data by visualizing it, because visualization will give us a clear picture of every single data it contain.
# * Second one is that, we need to clean the data and cleaning the data means preprocessing the images that we have. I'll explain the preprocessing of the images. So, don't worry :)
# * After preprocessing we do feature engineering but we are on basics, so we will not do this part.
# * At last, we Create a Deep Learning Model from Scratch and train it and then validate some hidden part of the data, and also we predict the test data that we have by using our Deep Learning Model.

# 
# A Kernel by Siti Khodijah on image preprocessing where she explained the basics Preprocessing of Images very well explained and with great visualization, it will helps you alot, must check this out: [Image Preprocessing](https://www.kaggle.com/code/khotijahs1/cv-image-preprocessing)

# ### **Now, we first get to the theory of Image Processing:**
# 
# Image processing could be simple tasks like image resizing.In order to feed a dataset of images to a convolutional network, they must all be the same size. Other processing tasks can take place like geometric and color transformation or converting color to grayscale and many more.

# #### **We use Convolutional Neural Network:**
# 
# Neural networks are a type of machine learning models which are designed to operate similar to biological neurons and human nervous system. These models are used to recognize complex patterns and relationships that exists within a labelled dataset. They have following properties:
# 
# * The core architecture of a Neural Network model is comprised of a large number of simple processing nodes called Neurons which are interconnected and organized in different layers.
# 
# * An individual node in a layer is connected to several other nodes in the previous and the next layer. The inputs form one layer are received and processed to generate the output which is passed to the next layer.
# 
# * The first layer of this architecture is often named as input layer which accepts the inputs, the last layer is named as the output layer which produces the output and every other layer between input and output layer is named is hidden layers.
# 
# ### **Key concepts in a Neural Network**
# 
# #### 1. Neuron:
# A Neuron is a single processing unit of a Neural Network which are connected to different other neurons in the network. These connections repersents inputs and ouputs from a neuron. To each of its connections, the neuron assigns a “weight” (W) which signifies the importance the input and adds a bias (b) term.
# 
# #### 2. Activation Functions:
# The activation functions are used to apply non-linear transformation on input to map it to output. The aim of activation functions is to predict the right class of the target variable based on the input combination of variables. Some of the popular activation functions are Relu, Sigmoid, and TanH.
# 
# #### 3. Forward Propagation:
# 
# Neural Network model goes through the process called forward propagation in which it passes the computed activation outputs in the forward direction.
# 
# Z = W*X + b
# A = g(Z)
# 
# * g is the activation function
# * A is the activation using the input
# * W is the weight associated with the input
# * B is the bias associated with the node
# 
# #### 4. Error Computation:
# The neural network learns by improving the values of weights and bias. The model computes the error in the predicted output in the final layer which is then used to make small adjustments the weights and bias. The adjustments are made such that the total error is minimized. Loss function measures the error in the final layer and cost function measures the total error of the network.
# 
# Loss = Actual_Value - Predicted_Value
# 
# Cost = Summation (Loss)
# 
# #### 5. Backward Propagation:
# Neural Network model undergoes the process called backpropagation in which the error is passed to backward layers so that those layers can also improve the associated values of weights and bias. It uses the algorithm called Gradient Descent in which the error is minimized and optimal values of weights and bias are obtained. This weights and bias adjustment is done by computing the derivative of error, derivative of weights, bias and subtracting them from the original values.
# 
# 
# 
# 
# 

# <div class="alert alert-info">  
# <h3><strong>Import Libraries</strong></h3>
# </div>

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Dropout, Flatten, BatchNormalization, InputLayer
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import cv2


# <div class="alert alert-info">  
# <h3><strong>Import Dataset 📃</strong></h3>
# </div>

# In[ ]:


train_data = pd.read_csv('../input/digit-recognizer/train.csv')
test_data = pd.read_csv('../input/digit-recognizer/test.csv')


# In[ ]:


train_data.head()


# Now I will explore the data and gain insight from the data.
# 
# #### **View the Dimensions of the train and test dataset.**
# 
# 
# Using function name **.shape** to find the rows and columns of the dataset, it returns the tuple (nrows,ncols) => so nrows shows the total data of the file and ncols shows the features of the file that we will use in future.

# In[ ]:


# Printing the Shape of the Train and Tet data to find data's instances and features.

print(f'The Shpae of the Train data : {train_data.shape}')
print(f'The Shpae of the Test data : {test_data.shape}')


# * In Train data we have *42000* instances and *785* features.
# * In Test data we have *28000* instances and *784* features.
# 
# Now we see that we have 785 features in train data which means that there are 784 pixel valuess in the image and one column is about label that which digit it is.
# 

# In[ ]:


nb_rows = 3
nb_cols = 5
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5));
plt.suptitle('SAMPLE IMAGES');
k = 0 
for i in range(0, nb_rows):
    for j in range(0, nb_cols):
        axs[i, j].xaxis.set_ticklabels([]);
        axs[i, j].yaxis.set_ticklabels([]);
        axs[i, j].imshow(train_data.iloc[k,1:].to_numpy().reshape(28,28));
        k = k + 3
plt.show();


# ### **Checking the Values in Image Form.**

# In[ ]:


fig,ax = plt.subplots(4,3,figsize=(6,7))
arr = [i for i in range(len(train_data.loc[:,'label'].unique()))]
ax = ax.flatten()
k = 0 
for i in range(12):
    ax[i].imshow(train_data.iloc[i,1:].to_numpy().reshape(28,28),cmap='gray')
    ax[i].set_title(f'Digit : {train_data.iloc[k,0]}')


# ### **Checking the Images One by One:**

# In[ ]:


def show_full_image(i):
    plt.figure(figsize=(12,8))
    plt.imshow(train_data.iloc[i,1:].to_numpy().reshape(28,28),cmap='gray')
    plt.title(f"Digit no : {train_data.iloc[i,0]}")
    
    plt.xticks([])
    plt.yticks([])


# In[ ]:


show_full_image(2)


# In[ ]:


show_full_image(3)


# In[ ]:


show_full_image(19)


# #### **Interpretation:**
# 
# ![](https://cdn.analyticsvidhya.com/wp-content/uploads/2021/03/Screenshot-from-2021-03-16-10-58-08.png)
# 
# 
# * Like you see the picture is zoomed and we can easily analyze the picture.
# * It contain pixel values in grid of 255 x 255.

# In[ ]:


input_data = train_data.drop(['label'],axis=1)
target = train_data.label


# In[ ]:


plt.figure(figsize=(16,8))
sns.countplot(x=train_data.loc[:,'label'] ,data=train_data.loc[:,'label'].dropna())


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(input_data,target,test_size=0.25)


# ### **We must encode the Output by using encoding technique, we use one-hot encoding**

# In[ ]:


y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)


# In[ ]:


y_train_enc[0]


# ### **Just reshaped to fit this in Model**
# 

# In[ ]:


x_train = x_train.values.reshape(-1,28,28,1)
x_test = x_test.values.reshape(-1,28,28,1)


# we have to make our Model's neuron's output non-linear so for achieving Non-Linearity we used activation function and used relu activation function which convert negative to zero and positive to 1.

# ### **I applied to DropOut to achieve Reguralization, this will tackle the overfitting.**

# In[ ]:


def model():
    
    nn = tf.keras.models.Sequential()
    
    nn.add(Conv2D(64,kernel_size=(3,3),kernel_initializer='he_normal',input_shape=(28,28,1),activation='relu'))
    nn.add(Conv2D(64,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))
    nn.add(MaxPooling2D(pool_size=(2,2)))
#     nn.add(Dropout(0.2))
    nn.add(BatchNormalization())
    nn.add(Conv2D(128,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))
    nn.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
    nn.add(MaxPooling2D(pool_size=(2,2)))
    nn.add(Dropout(0.2))
    nn.add(Conv2D(256,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))
   # nn.add(Conv2D(256,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))
 #   nn.add(MaxPooling2D(pool_size=(2,2)))
    
#     nn.add(Conv2D(512,kernel_size=(3,3),activation='relu',kernel_initializer='he_normal'))
    nn.add(MaxPooling2D(pool_size=(2,2)))
    
    nn.add(Flatten())
    nn.add(Dense(2048,activation='relu'))
    nn.add(Dense(10,activation='softmax'))
    
    nn.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return nn


# ### **Model.summary function prints the whole structure with details that we created Above.**

# In[ ]:


model = model()
model.summary()


# ## **Training the Model**

# In[ ]:


model.fit(x_train,y_train_enc,batch_size=5,epochs=20)


# ## **Validation of Model**

# In[ ]:


loss,acc = model.evaluate(x_test,y_test_enc)


# In[ ]:


print('Loss = ',loss)
print('Accuracy = ',acc*100)


# In[ ]:


pred = model.predict(test_data.values.reshape(-1,28,28,1))
pred.shape


# In[ ]:


preds = []
for i in range(28000):
    preds.append(np.argmax(pred[i]))


# In[ ]:


sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
sub['Label'] = preds


# In[ ]:


sub.to_csv('submission.csv',index=False)

