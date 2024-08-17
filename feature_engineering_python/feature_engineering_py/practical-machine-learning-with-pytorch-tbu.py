#!/usr/bin/env python
# coding: utf-8

# # Practical Machine Learning ML with PyTorch (TBU)
# 
# This kernel is empowering you to use machine learning to get valuable insights from data.
# - 🔥 Implement basic ML algorithms and deep neural networks with <a href="https://pytorch.org/" target="_blank" style="color:#ee4c2c">PyTorch</a>.
# - 📦 Learn object-oriented ML to code for products, not just tutorials.
# 
# > #### **Credits**: Thanks to **Practical AI - Goku Mohandas** and other contributers for such wonderful work!
# 
# ### Here are some of *my kernel notebooks* for **Machine Learning and Data Science** as follows, ***Upvote*** them if you *like* them
# 
# > * [Awesome Deep Learning Basics and Resources](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-resources)
# > * [Data Science with R - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-r-awesome-tutorials)
# > * [Data Science and Machine Learning Cheetcheets](https://www.kaggle.com/arunkumarramanan/data-science-and-machine-learning-cheatsheets)
# > * [Awesome ML Frameworks and MNIST Classification](https://www.kaggle.com/arunkumarramanan/awesome-machine-learning-ml-frameworks)
# > * [Awesome Data Science for Beginners with Titanic Exploration](https://kaggle.com/arunkumarramanan/awesome-data-science-for-beginners)
# > * [Tensorflow Tutorial and House Price Prediction](https://www.kaggle.com/arunkumarramanan/tensorflow-tutorial-and-examples)
# > * [Data Scientist's Toolkits - Awesome Data Science Resources](https://www.kaggle.com/arunkumarramanan/data-scientist-s-toolkits-awesome-ds-resources)
# > * [Awesome Computer Vision Resources (TBU)](https://www.kaggle.com/arunkumarramanan/awesome-computer-vision-resources-to-be-updated)
# > * [Machine Learning and Deep Learning - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/awesome-deep-learning-ml-tutorials)
# > * [Data Science with Python - Awesome Tutorials](https://www.kaggle.com/arunkumarramanan/data-science-with-python-awesome-tutorials)
# > * [Awesome TensorFlow and PyTorch Resources](https://www.kaggle.com/arunkumarramanan/awesome-tensorflow-and-pytorch-resources)
# > * [Awesome Data Science IPython Notebooks](https://www.kaggle.com/arunkumarramanan/awesome-data-science-ipython-notebooks)
# > * [Machine Learning Engineer's Toolkit with Roadmap](https://www.kaggle.com/arunkumarramanan/machine-learning-engineer-s-toolkit-with-roadmap) 
# > * [Hands-on ML with scikit-learn and TensorFlow](https://www.kaggle.com/arunkumarramanan/hands-on-ml-with-scikit-learn-and-tensorflow)
# > * [Practical Machine Learning with PyTorch](https://www.kaggle.com/arunkumarramanan/practical-machine-learning-with-pytorch)
# 
# > ***Practical Machine Learning ML with TensorFlow***
# 
# > The above highlighted work will soon be released and also be based on the below coursework;
# 
# * [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) - Machine Learning ML Crash Course with TensorFlow APIs (Google's fast-paced, practical introduction to machine learning, A self-study guide for aspiring machine learning practitioners: Machine Learning Crash Course features a series of lessons with video lectures, real-world case studies, and hands-on practice exercises.) is highly recommended by Google as it's developed by googlers along with Notebooks for exercises.
# 
#  <a id="top"></a> <br>
#  
# ## Kernel Notebook Content
#  
# ## [Basics](#1)                  
# - 🐍 [Python](#1)                                                                           
# - 🔢 [NumPy](#1)                                                                                      
# - 🐼[Pandas](#1)                            
# - 📈 [Linear Regression](#1) 
# - 📊 [Logistic Regression](#1)
# - 🌳 [Random Forests](#1)
# - 💥 KMeans Clustering
#  
# ## [Deep Learning](#2)
# - 🔥 [PyTorch](#2)
# - 🎛️ [Multilayer Perceptrons](#2)
# - 🔎 [Data & Models](#2)
# - 📦 [Object-Oriented ML](#2)
# - 🖼️ [Convolutional Neural Networks](#2)
# - 📝 [Embeddings](#2)
# - 📗 [Recurrent Neural Networks](#2)
# 
# ## [Advanced](#3)
# - 📚 [Advanced RNNs](#3)
# - 🏎️ Highway and Residual Networks
# - 🔮 Autoencoders
# - 🎭 Generative Adversarial Networks
# - 🐝 Spatial Transformer Networks
# 
# ## [Topics](#4)
# - 📸 [Computer Vision](#4)
# - ⏰ Time Series Analysis
# - 🏘️ Topic Modeling
# - 🛒 Recommendation Systems
# - 🗣️ Pretrained Language Modeling
# - 🤷 Multitask Learning
# - 🎯 Low Shot Learning|
# - 🍒 Reinforcement Learning|
# 
# 

#  <a id="1"></a> <br>
# ## 1. Basics    
# 
# <a id="1.1"></a> <br>
# ## 1.1 Introduction to Python
# In this lesson we will learn the basics of the Python programming language (version 3). We won't learn everything about Python but enough to do some basic machine learning.
# 
# <img src="https://www.python.org/static/community_logos/python-logo-master-v3-TM.png" width=350>

# ###  Variables
# Variables are objects in python that can hold anything with numbers or text. Let's look at how to make some variables.
# 

# In[1]:


# Numerical example
x = 5
print (x)


# In[2]:


# Text example
x = "hello"
print (x)


# In[3]:


# int variable
x = 5
print (x)
print (type(x))

# float variable
x = 5.0
print (x)
print (type(x))

# text variable
x = "5" 
print (x)
print (type(x))

# boolean variable
x = True
print (x)
print (type(x))


# It's good practice to know what types your variables are. When you want to use numerical operations on then, they need to be compatible. 

# In[4]:


# int variables
a = 5
b = 3
print (a + b)

# string variables
a = "5"
b = "3"
print (a + b)


# ###  Lists
# Lists are objects in python that can hold a ordered sequence of numbers **and** text.

# In[5]:


# Making a list
list_x = [3, "hello", 1]
print (list_x)

# Adding to a list
list_x.append(7)
print (list_x)

# Accessing items at specific location in a list
print ("list_x[0]: ", list_x[0])
print ("list_x[1]: ", list_x[1])
print ("list_x[2]: ", list_x[2])
print ("list_x[-1]: ", list_x[-1]) # the last item
print ("list_x[-2]: ", list_x[-2]) # the second to last item

# Slicing
print ("list_x[:]: ", list_x[:])
print ("list_x[2:]: ", list_x[2:])
print ("list_x[1:3]: ", list_x[1:3])
print ("list_x[:-1]: ", list_x[:-1])

# Length of a list
len(list_x)

# Replacing items in a list
list_x[1] = "hi"
print (list_x)

# Combining lists
list_y = [2.4, "world"]
list_z = list_x + list_y
print (list_z)


# ### Tuples
# Tuples are also objects in python that can hold data but you cannot replace values (for this reason, tuples are called immutable, whereas lists are known as mutable).

# In[6]:


# Creating a tuple
tuple_x = (3.0, "hello")
print (tuple_x)

# Adding values to a tuple
tuple_x = tuple_x + (5.6,)
print (tuple_x)

# Trying to change a tuples value (you can't)
tuple_x[1] = "world"


# ### Dictionaries
# Dictionaries are python objects that hold key-value pairs. In the example dictionary below, the keys are the "name" and "eye_color" variables. They each have a value associated with them. A dictionary cannot have two of the same keys. 

# In[7]:


# Creating a dictionary
arun = {"name": "Arun",
        "eye_color": "brown"}
print (arun)
print (arun["name"])
print (arun["eye_color"])

# Changing the value for a key
arun["eye_color"] = "black"
print (arun)

# Adding new key-value pairs
arun["age"] = 24
print (arun)

# Length of a dictionary
print (len(arun))


# ### If statements
# You can use if statements to conditionally do something.

# In[8]:


# If statement
x = 4
if x < 1:
    score = "low"
elif x <= 4:
    score = "medium"
else:
    score = "high"
print (score)

# If statment with a boolean
x = True
if x:
    print ("it worked")


# ### Loops
# You can use for or while loops in python to do something repeatedly until a condition is met.

# In[9]:


# For loop
x = 1
for i in range(3): # goes from i=0 to i=2
    x += 1 # same as x = x + 1
    print ("i={0}, x={1}".format(i, x)) # printing with multiple variables


# In[10]:


# While loop
x = 3
while x > 0:
    x -= 1 # same as x = x - 1
    print (x)


# ### Functions
# Functions are a way to modularize reusable pieces of code. 

# In[11]:


# Create a function
def add_two(x):
    x += 2
    return x

# Use the function
score = 0
score = add_two(x=score)
print (score)


# In[12]:


# Function with multiple inputs
def join_name(first_name, last_name):
    joined_name = first_name + " " + last_name
    return joined_name

# Use the function
first_name = "Arunkumar"
last_name = "Venkataramanan"
joined_name = join_name(first_name=first_name, last_name=last_name)
print (joined_name)


# ### Classes
# Classes are a fundamental piece of object oriented Python programming.

# In[13]:


# Create the function
class Pets(object):
  
    # Initialize the class
    def __init__(self, species, color, name):
        self.species = species
        self.color = color
        self.name = name

    # For printing  
    def __str__(self):
        return "{0} {1} named {2}.".format(self.color, self.species, self.name)

    # Example function
    def change_name(self, new_name):
        self.name = new_name


# In[14]:


# Making an instance of a class
my_dog = Pets(species="dog", color="orange", name="Guiness",)
print (my_dog)
print (my_dog.name)


# In[15]:


# Using a class's function
my_dog.change_name(new_name="Johny Kutty")
print (my_dog)
print (my_dog.name)


# ### Additional resources
# This was a very quick look at python and we'll be learning more in future lessons. If you want to learn more right now before diving into machine learning, check out this free course: [Free Python Course](https://www.codecademy.com/learn/learn-python) and [Kaggle Learn](https://www.kaggle.com/learn/python)
# 
# ###### [Go to top](#top)

# <a id="1.2"></a> <br>
# ## 1.2 NumPy
# 
# In this lesson we will learn the basics of numerical analysis using the NumPy package.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/1/1a/NumPy_logo.svg" width=300>
# 
# 

# ### NumPy basics

# In[16]:


import numpy as np


# In[17]:


# Set seed for reproducability
np.random.seed(seed=1234)


# In[18]:


# Scalars
x = np.array(6) # scalar
print ("x: ", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print ("x dtype: ", x.dtype)


# In[19]:


# 1-D Array
x = np.array([1.3 , 2.2 , 1.7])
print ("x: ", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print ("x dtype: ", x.dtype) # notice the float datatype


# In[20]:


# 3-D array (matrix)
x = np.array([[1,2,3], [4,5,6], [7,8,9]])
print ("x:\n", x)
print("x ndim: ", x.ndim)
print("x shape:", x.shape)
print("x size: ", x.size)
print ("x dtype: ", x.dtype)


# In[21]:


# Functions
print ("np.zeros((2,2)):\n", np.zeros((2,2)))
print ("np.ones((2,2)):\n", np.ones((2,2)))
print ("np.eye((2)):\n", np.eye((2)))
print ("np.random.random((2,2)):\n", np.random.random((2,2)))


# ### Indexing

# In[22]:


# Indexing
x = np.array([1, 2, 3])
print ("x[0]: ", x[0])
x[0] = 0
print ("x: ", x)


# In[23]:


# Slicing
x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print (x)
print ("x column 1: ", x[:, 1]) 
print ("x row 0: ", x[0, :]) 
print ("x rows 0,1,2 & cols 1,2: \n", x[:3, 1:3]) 


# In[24]:


# Integer array indexing
print (x)
rows_to_get = np.arange(len(x))
print ("rows_to_get: ", rows_to_get)
cols_to_get = np.array([0, 2, 1])
print ("cols_to_get: ", cols_to_get)
print ("indexed values: ", x[rows_to_get, cols_to_get])


# In[25]:


# Boolean array indexing
x = np.array([[1,2], [3, 4], [5, 6]])
print ("x:\n", x)
print ("x > 2:\n", x > 2)
print ("x[x > 2]:\n", x[x > 2])


# ### Array math

# In[26]:


# Basic math
x = np.array([[1,2], [3,4]], dtype=np.float64)
y = np.array([[1,2], [3,4]], dtype=np.float64)
print ("x + y:\n", np.add(x, y)) # or x + y
print ("x - y:\n", np.subtract(x, y)) # or x - y
print ("x * y:\n", np.multiply(x, y)) # or x * y


# <img src="https://blog.studygate.com/wp-content/uploads/2017/11/Matrix-Multiplication-dot-product.png" width=400>
# 

# In[27]:


# Dot product
a = np.array([[1,2,3], [4,5,6]], dtype=np.float64) # we can specify dtype
b = np.array([[7,8], [9,10], [11, 12]], dtype=np.float64)
print (a.dot(b))


# In[28]:


# Sum across a dimension
x = np.array([[1,2],[3,4]])
print (x)
print ("sum all: ", np.sum(x)) # adds all elements
print ("sum by col: ", np.sum(x, axis=0)) # add numbers in each column
print ("sum by row: ", np.sum(x, axis=1)) # add numbers in each row


# In[29]:


# Transposing
print ("x:\n", x)
print ("x.T:\n", x.T)


# ### Advanced

# In[30]:


# Tile
x = np.array([[1,2], [3,4]])
y = np.array([5, 6])
addent = np.tile(y, (len(x), 1))
print ("addent: \n", addent)
z = x + addent
print ("z:\n", z)


# In[31]:


# Broadcasting
x = np.array([[1,2], [3,4]])
y = np.array([5, 6])
z = x + y
print ("z:\n", z)


# In[32]:


# Reshaping
x = np.array([[1,2], [3,4], [5,6]])
print (x)
print ("x.shape: ", x.shape)
y = np.reshape(x, (2, 3))
print ("y.shape: ", y.shape)
print ("y: \n", y)


# In[33]:


# Removing dimensions
x = np.array([[[1,2,1]],[[2,2,3]]])
print ("x.shape: ", x.shape)
y = np.squeeze(x, 1) # squeeze dim 1
print ("y.shape: ", y.shape) 
print ("y: \n", y)


# In[34]:


# Adding dimensions
x = np.array([[1,2,1],[2,2,3]])
print ("x.shape: ", x.shape)
y = np.expand_dims(x, 1) # expand dim 1
print ("y.shape: ", y.shape) 
print ("y: \n", y)


# ### Additional resources
# 
# You don't to memorize anything here and we will be taking a closer look at NumPy in the later lessons. If you are curious about more checkout the [NumPy reference manual](https://docs.scipy.org/doc/numpy-1.15.1/reference/).
# 
# ###### [Go to top](#top)

# <a id="1.3"></a> <br>
# ## 1.3 Pandas

# In this notebook, we'll learn the basics of data analysis with the Python Pandas library.
# 
# <img src="https://raw.githubusercontent.com/ArunkumarRamanan/practicalAI/master/images/pandas.png" width=500>
# 

# ### Uploading the data
# 
# We're first going to get some data to play with. We're going to load the titanic dataset from the public link below.

# ### Loading the data
# 
# Now that we have some data to play with, let's load into a Pandas dataframe. Pandas is a great python library for data analysis.

# In[35]:


import pandas as pd


# In[36]:


# Read from CSV to Pandas DataFrame
df = pd.read_csv("../input/train.csv", header=0)


# In[37]:


# First five items
df.head()


# These are the diferent features: 
# * pclass: class of travel
# * name: full name of the passenger
# * sex: gender
# * age: numerical age
# * sibsp: # of siblings/spouse aboard
# * parch: number of parents/child aboard
# * ticket: ticket number
# * fare: cost of the ticket
# * cabin: location of room
# * emarked: port that the passenger embarked at (C - Cherbourg, S - Southampton, Q = Queenstown)
# * survived: survial metric (0 - died, 1 - survived)

# ### Exploratory Dats Analysis EDA
# 
# We're going to explore the Pandas library and see how we can explore and process our data.

# In[38]:


# Describe features
df.describe()


# In[39]:


# Histograms
df["Age"].hist()


# In[40]:


# Unique values
df["Embarked"].unique()


# In[41]:


# Selecting data by feature
df["Name"].head()


# In[42]:


# Filtering
df[df["Sex"]=="female"].head() # only the female data appear


# In[43]:


# Sorting
df.sort_values("Age", ascending=False).head()


# In[44]:


# Grouping
sex_group = df.groupby("Survived")
sex_group.mean()


# In[45]:


# Selecting row
df.iloc[0, :] # iloc gets rows (or columns) at particular positions in the index (so it only takes integers)


# In[46]:


# Selecting specific value
df.iloc[0, 1]


# In[47]:


# Selecting by index
df.loc[0] # loc gets rows (or columns) with particular labels from the index


# ### Data Preprocessing

# In[48]:


# Rows with at least one NaN value
df[pd.isnull(df).any(axis=1)].head()


# In[49]:


# Drop rows with Nan values
df = df.dropna() # removes rows with any NaN values
df = df.reset_index() # reset's row indexes in case any rows were dropped
df.head()


# In[50]:


# Dropping multiple rows
df = df.drop(["Name", "Cabin", "Ticket"], axis=1) # we won't use text features for our initial basic models
df.head()


# In[51]:


# Map feature values
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df["Embarked"] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
df.head()


# ### Feature Engineering

# In[52]:


# Lambda expressions to create new features
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size

df["Family_Size"] = df[["SibSp", "Parch"]].apply(lambda x: get_family_size(x["SibSp"], x["Parch"]), axis=1)
df.head()


# In[53]:


# Reorganize headers
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Family_Size', 'Fare', 'Embarked', 'Survived']]
df.head()


# ### Saving data

# In[54]:


# Saving dataframe to CSV
df.to_csv("processed_titanic.csv", index=False)


# In[55]:


# See your saved file
get_ipython().system('ls -l')


# ###### [Go to top](#top)

# <a id="1.4"></a> <br>
# ## 1.4 Linear Regression
# 
# In this lesson we will learn about linear regression. We will first understand the basic math behind it and then implement it in Python. We will also look at ways of interpreting the linear model.

# ### Overview
# 
# <img src="https://raw.githubusercontent.com/GokuMohandas/practicalAI/master/images/linear.png" width=250>
# 
# $\hat{y} = XW$
# 
# *where*:
# * $\hat{y}$ = prediction | $\in \mathbb{R}^{NX1}$ ($N$ is the number of samples)
# * $X$ = inputs | $\in \mathbb{R}^{NXD}$ ($D$ is the number of features)
# * $W$ = weights | $\in \mathbb{R}^{DX1}$ 

# * **Objective:**  Use inputs $X$ to predict the output $\hat{y}$ using a linear model. The model will be a line of best fit that minimizes the distance between the predicted and target outcomes. Training data $(X, y)$ is used to train the model and learn the weights $W$ using stochastic gradient descent (SGD).
# * **Advantages:**
#   * Computationally simple.
#   * Highly interpretable.
#   * Can account for continuous and categorical features.
# * **Disadvantages:**
#   * The model will perform well only when the data is linearly separable (for classification).
#   * Usually not used for classification and only for regression.
# * **Miscellaneous:** You can also use linear regression for binary classification tasks where if the predicted continuous value is above a threshold, it belongs to a certain class. But we will cover better techniques for classification in future lessons and will focus on linear regression for continuos regression tasks only.
# 

# ### Training
# 
# *Steps*: 
# 1. Randomly initialize the model's weights $W$.
# 2. Feed inputs $X$ into the model to receive the predictions $\hat{y}$.
# 3. Compare the predictions $\hat{y}$ with the actual target values $y$ with the objective (cost) function to determine loss $J$. A common objective function for linear regression is mean squarred error (MSE). This function calculates the difference between the predicted and target values and squares it. (the $\frac{1}{2}$ is just for convenicing the derivative operation).
#   * $MSE = J(\theta) = \frac{1}{2}\sum_{i}(\hat{y}_i - y_i)^2$
# 4. Calculate the gradient of loss $J(\theta)$ w.r.t to the model weights.
#   * $J(\theta) = \frac{1}{2}\sum_{i}(\hat{y}_i - y_i)^2 = \frac{1}{2}\sum_{i}(X_iW - y_i)^2 $
#   * $\frac{\partial{J}}{\partial{W}} = X(\hat{y} - y)$
# 4. Apply backpropagation to update the weights $W$ using a learning rate $\alpha$ and an optimization technique (ie. stochastic gradient descent). The simplified intuition is that the gradient tells you the direction for how to increase something so subtracting it will help you go the other way since we want to decrease loss $J(\theta)$.
#   * $W = W- \alpha\frac{\partial{J}}{\partial{W}}$
# 5. Repeat steps 2 - 4 until model performs well.

# ### Data
# 
# We're going to create some simple dummy data to apply linear regression on.

# In[56]:


from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[57]:


# Arguments
args = Namespace(
    seed=1234,
    data_file="sample_data.csv",
    num_samples=100,
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)

# Set seed for reproducability
np.random.seed(args.seed)


# In[58]:


# Generate synthetic data
def generate_data(num_samples):
    X = np.array(range(num_samples))
    y = 3.65*X + 10
    return X, y


# In[59]:


# Generate random (linear) data
X, y = generate_data(args.num_samples)
data = np.vstack([X, y]).T
df = pd.DataFrame(data, columns=['X', 'y'])
df.head()


# In[60]:


# Scatter plot
plt.title("Generated data")
plt.scatter(x=df["X"], y=df["y"])
plt.show()


# ### Scikit-learn Implementation
# 
# **Note**: The `LinearRegression` class in Scikit-learn uses the normal equation to solve the fit. However, we are going to use Scikit-learn's `SGDRegressor` class which uses stochastic gradient descent. We want to use this optimization approach because we will be using this for the models in subsequent lessons.

# In[61]:


# Import packages
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[62]:


# Create data splits
X_train, X_test, y_train, y_test = train_test_split(
    df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size, 
    random_state=args.seed)
print ("X_train:", X_train.shape)
print ("y_train:", y_train.shape)
print ("X_test:", X_test.shape)
print ("y_test:", y_test.shape)


# We need to standardize our data (zero mean and unit variance) in order to properly use SGD and optimize quickly.

# In[63]:


# Standardize the data (mean=0, std=1) using training data
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))

# Apply scaler on training and test data
standardized_X_train = X_scaler.transform(X_train)
standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()
standardized_X_test = X_scaler.transform(X_test)
standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()


# Check
print ("mean:", np.mean(standardized_X_train, axis=0), 
       np.mean(standardized_y_train, axis=0)) # mean should be ~0
print ("std:", np.std(standardized_X_train, axis=0), 
       np.std(standardized_y_train, axis=0))   # std should be 1


# In[64]:


# Initialize the model
lm = SGDRegressor(loss="squared_loss", penalty="none", max_iter=args.num_epochs)


# In[65]:


# Train
lm.fit(X=standardized_X_train, y=standardized_y_train)


# In[66]:


# Predictions (unstandardize them)
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_


# ### Evaluation
# 
# There are several evaluation techniques to see how well our model performed.

# In[67]:


import matplotlib.pyplot as plt


# In[68]:


# Train and test MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))


# Besides MSE, when we only have one feature, we can visually inspect the model.

# In[69]:


# Figure size
plt.figure(figsize=(15,5))

# Plot train data
plt.subplot(1, 2, 1)
plt.title("Train")
plt.scatter(X_train, y_train, label="y_train")
plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
plt.legend(loc='lower right')

# Plot test data
plt.subplot(1, 2, 2)
plt.title("Test")
plt.scatter(X_test, y_test, label="y_test")
plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
plt.legend(loc='lower right')

# Show plots
plt.show()


# ### Inference

# In[70]:


# Feed in your own inputs
X_infer = np.array((0, 1, 2), dtype=np.float32)
standardized_X_infer = X_scaler.transform(X_infer.reshape(-1, 1))
pred_infer = (lm.predict(standardized_X_infer) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
print (pred_infer)
df.head(3)


# ### Interpretability
# 
# Linear regression offers the great advantage of being highly interpretable. Each feature has a coefficient which signifies it's importance/impact on the output variable y. We can interpret our coefficient as follows: By increasing X by 1 unit, we increase y by $W$ (~3.65) units. 
# 
# **Note**: Since we standardized our inputs and outputs for gradient descent, we need to apply an operation to our coefficients and intercept to interpret them. See proof below.

# In[71]:


# Unstandardize coefficients 
coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)
print (coef) # ~3.65
print (intercept) # ~10


# ### Proof for unstandardizing coefficients:
# 
# 
# Note that both X and y were standardized.
# 
# $\frac{\mathbb{E}[y] - \hat{y}}{\sigma_y} = W_0 + \sum_{j=1}^{k}W_jz_j$
# 
# $z_j = \frac{x_j - \bar{x}_j}{\sigma_j}$
# 
# $ \hat{y}_{scaled} = \frac{\hat{y}_{unscaled} - \bar{y}}{\sigma_y} = \hat{W_0} + \sum_{j=1}^{k} \hat{W}_j (\frac{x_j - \bar{x}_j}{\sigma_j}) $
# 
# $\hat{y}_{unscaled} = \hat{W}_0\sigma_y + \bar{y} - \sum_{j=1}^{k} \hat{W}_j(\frac{\sigma_y}{\sigma_j})\bar{x}_j + \sum_{j=1}^{k}(\frac{\sigma_y}{\sigma_j})x_j $
# 

# ### Regularization
# 
# Regularization helps decrease over fitting. Below is L2 regularization (ridge regression). There are many forms of regularization but they all work to reduce overfitting in our models. With L2 regularization, we are penalizing the weights with large magnitudes by decaying them. Having certain weights with high magnitudes will lead to preferential bias with the inputs and we want the model to work with all the inputs and not just a select few. There are also other types of regularization like L1 (lasso regression) which is useful for creating sparse models where some feature cofficients are zeroed out, or elastic which combines L1 and L2 penalties. 
# 
# **Note**: Regularization is not just for linear regression. You can use it to regualr any model's weights including the ones we will look at in future lessons.

# * $ J(\theta) = = \frac{1}{2}\sum_{i}(X_iW - y_i)^2 + \frac{\lambda}{2}\sum\sum W^2$
# * $ \frac{\partial{J}}{\partial{W}}  = X (\hat{y} - y) + \lambda W $
# * $W = W- \alpha\frac{\partial{J}}{\partial{W}}$
# where:
#   * $\lambda$ is the regularzation coefficient

# In[72]:


# Initialize the model with L2 regularization
lm = SGDRegressor(loss="squared_loss", penalty='l2', alpha=1e-2, 
                  max_iter=args.num_epochs)


# In[73]:


# Train
lm.fit(X=standardized_X_train, y=standardized_y_train)


# In[74]:


# Predictions (unstandardize them)
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_


# In[75]:


# Train and test MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(
    train_mse, test_mse))


# Regularization didn't help much with this specific example because our data is generation from a perfect linear equation but for realistic data, regularization can help our model generalize well.

# In[76]:


# Unstandardize coefficients 
coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - (coef*X_scaler.mean_)
print (coef) # ~3.65
print (intercept) # ~10


# ### Categorical variables
# 
# In our example, the feature was a continuous variable but what if we also have features that are categorical? One option is to treat the categorical variables as one-hot encoded variables. This is very easy to do with Pandas and once you create the dummy variables, you can use the same steps as above to train your linear model.

# In[77]:


# Create data with categorical features
cat_data = pd.DataFrame(['a', 'b', 'c', 'a'], columns=['favorite_letter'])
cat_data.head()


# In[78]:


dummy_cat_data = pd.get_dummies(cat_data)
dummy_cat_data.head()


# Now you can concat this with your continuous features and train the linear model.
# 
# ### TODO
# 
# - polynomial regression
# - simple example with normal equation method (sklearn.linear_model.LinearRegression) with pros and cons vs. SGD linear regression
# 
# ###### [Go to top](#top)

# <a id="1.3"></a> <br>
# ## 1.5 Logistic Regression
# 
# In the previous lesson, we saw how linear regression works really well for predicting continuous outputs that can easily fit to a line/plane. But linear regression doesn't fare well for classification asks where we want to probabilititcally determine the outcome for a given set on inputs.

# <img src="https://raw.githubusercontent.com/ArunkumarRamanan/practicalAI/master/images/logistic.jpg" width=270>
# 
# $ \hat{y} = \frac{1}{1 + e^{-XW}} $ 
# 
# *where*:
# * $\hat{y}$ = prediction | $\in \mathbb{R}^{NX1}$ ($N$ is the number of samples)
# * $X$ = inputs | $\in \mathbb{R}^{NXD}$ ($D$ is the number of features)
# * $W$ = weights | $\in \mathbb{R}^{DX1}$ 
# 
# This is the binomial logistic regression. The main idea is to take the outputs from the linear equation ($z=XW$) and use the sigmoid (logistic) function ($\frac{1}{1+e^{-z}}$) to restrict the value between (0, 1). 

# When we have more than two classes, we need to use multinomial logistic regression (softmax classifier). The softmax classifier will use the linear equation ($z=XW$) and normalize it to product the probabiltiy for class y given the inputs.
# 
# $ \hat{y} = \frac{e^{XW_y}}{\sum e^{XW}} $ 
# 
# *where*:
# * $\hat{y}$ = prediction | $\in \mathbb{R}^{NX1}$ ($N$ is the number of samples)
# * $X$ = inputs | $\in \mathbb{R}^{NXD}$ ($D$ is the number of features)
# * $W$ = weights | $\in \mathbb{R}^{DXC}$ ($C$ is the number of classes)
# 

# * **Objective:**  Predict the probability of class $y$ given the inputs $X$. The softmax classifier normalizes the linear outputs to determine class probabilities. 
# * **Advantages:**
#   * Can predict class probabilities given a set on inputs.
# * **Disadvantages:**
#   * Sensitive to outliers since objective is minimize cross entropy loss. (Support vector machines ([SVMs](https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f)) are a good alternative to counter outliers).
# * **Miscellaneous:** Softmax classifier is going to used widely in neural network architectures as the last layer since it produces class probabilities.

# # Training
# 
# *Steps*:
# 
# 1. Randomly initialize the model's weights $W$.
# 2. Feed inputs $X$ into the model to receive the logits ($z=XW$). Apply the softmax operation on the logits to get the class probabilies $\hat{y}$ in one-hot encoded form. For example, if there are three classes, the predicted class probabilities could look like [0.3, 0.3, 0.4]. 
# 3. Compare the predictions $\hat{y}$ (ex.  [0.3, 0.3, 0.4]]) with the actual target values $y$ (ex. class 2 would look like [0, 0, 1]) with the objective (cost) function to determine loss $J$. A common objective function for logistics regression is cross-entropy loss. 
#   * $J(\theta) = - \sum_i y_i ln (\hat{y_i}) =  - \sum_i y_i ln (\frac{e^{X_iW_y}}{\sum e^{X_iW}}) $
#    * $y$ = [0, 0, 1]
#   * $\hat{y}$ = [0.3, 0.3, 0.4]]
#   * $J(\theta) = - \sum_i y_i ln (\hat{y_i}) =  - \sum_i y_i ln (\frac{e^{X_iW_y}}{\sum e^{X_iW}}) = - \sum_i [0 * ln(0.3) + 0 * ln(0.3) + 1 * ln(0.4)] = -ln(0.4) $
#   * This simplifies our cross entropy objective to the following: $J(\theta) = - ln(\hat{y_i})$ (negative log likelihood).
#   * $J(\theta) = - ln(\hat{y_i}) = - ln (\frac{e^{X_iW_y}}{\sum_i e^{X_iW}}) $
# 4. Calculate the gradient of loss $J(\theta)$ w.r.t to the model weights. Let's assume that our classes are mutually exclusive (a set of inputs could only belong to one class).
#  * $\frac{\partial{J}}{\partial{W_j}} = \frac{\partial{J}}{\partial{y}}\frac{\partial{y}}{\partial{W_j}} = - \frac{1}{y}\frac{\partial{y}}{\partial{W_j}} = - \frac{1}{\frac{e^{W_yX}}{\sum e^{XW}}}\frac{\sum e^{XW}e^{W_yX}0 - e^{W_yX}e^{W_jX}X}{(\sum e^{XW})^2} = \frac{Xe^{W_jX}}{\sum e^{XW}} = XP$
#   * $\frac{\partial{J}}{\partial{W_y}} = \frac{\partial{J}}{\partial{y}}\frac{\partial{y}}{\partial{W_y}} = - \frac{1}{y}\frac{\partial{y}}{\partial{W_y}} = - \frac{1}{\frac{e^{W_yX}}{\sum e^{XW}}}\frac{\sum e^{XW}e^{W_yX}X - e^{W_yX}e^{W_yX}X}{(\sum e^{XW})^2} = \frac{1}{P}(XP - XP^2) = X(P-1)$
# 5. Apply backpropagation to update the weights $W$ using gradient descent. The updates will penalize the probabiltiy for the incorrect classes (j) and encourage a higher probability for the correct class (y).
#   * $W_i = W_i - \alpha\frac{\partial{J}}{\partial{W_i}}$
# 6. Repeat steps 2 - 4 until model performs well.

# # Data
# 
# We're going to the load the titanic dataset we looked at in lesson 03_Pandas.

# In[79]:


from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib


# In[80]:


# Arguments
args = Namespace(
    seed=1234,
    data_file="titanic.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)

# Set seed for reproducability
np.random.seed(args.seed)


# In[81]:


# Upload data from GitHub to notebook's local drive
url = "https://raw.githubusercontent.com/ArunkumarRamanan/practicalAI/master/data/titanic.csv"
response = urllib.request.urlopen(url)
html = response.read()
with open(args.data_file, 'wb') as f:
    f.write(html)


# In[82]:


# Read from CSV to Pandas DataFrame
df = pd.read_csv(args.data_file, header=0)
df.head()


# # Scikit-learn implementation
# 
# **Note**: The `LogisticRegression` class in Scikit-learn uses coordinate descent to solve the fit. However, we are going to use Scikit-learn's `SGDClassifier` class which uses stochastic gradient descent. We want to use this optimization approach because we will be using this for the models in subsequent lessons.

# In[83]:


# Import packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[84]:


# Preprocessing
def preprocess(df):
  
    # Drop rows with NaN values
    df = df.dropna()

    # Drop text based features (we'll learn how to use them in later lessons)
    features_to_drop = ["name", "cabin", "ticket"]
    df = df.drop(features_to_drop, axis=1)

    # pclass, sex, and embarked are categorical features
    categorical_features = ["pclass","embarked","sex"]
    df = pd.get_dummies(df, columns=categorical_features)

    return df


# In[85]:


# Preprocess the dataset
df = preprocess(df)
df.head()


# In[86]:


# Split the data
mask = np.random.rand(len(df)) < args.train_size
train_df = df[mask]
test_df = df[~mask]
print ("Train size: {0}, test size: {1}".format(len(train_df), len(test_df)))


# **Note**: If you have preprocessing steps like standardization, etc. that are calculated, you need to separate the training and test set first before spplying those operations. This is because we cannot apply any knowledge gained from the test set accidentally during preprocessing/training.

# In[87]:


# Separate X and y
X_train = train_df.drop(["survived"], axis=1)
y_train = train_df["survived"]
X_test = test_df.drop(["survived"], axis=1)
y_test = test_df["survived"]


# In[88]:


# Standardize the data (mean=0, std=1) using training data
X_scaler = StandardScaler().fit(X_train)

# Apply scaler on training and test data (don't standardize outputs for classification)
standardized_X_train = X_scaler.transform(X_train)
standardized_X_test = X_scaler.transform(X_test)

# Check
print ("mean:", np.mean(standardized_X_train, axis=0)) # mean should be ~0
print ("std:", np.std(standardized_X_train, axis=0))   # std should be 1


# In[89]:


# Initialize the model
log_reg = SGDClassifier(loss="log", penalty="none", max_iter=args.num_epochs, 
                        random_state=args.seed)


# In[90]:


# Train
log_reg.fit(X=standardized_X_train, y=y_train)


# In[91]:


# Probabilities
pred_test = log_reg.predict_proba(standardized_X_test)
print (pred_test[:5])


# In[92]:


# Predictions (unstandardize them)
pred_train = log_reg.predict(standardized_X_train) 
pred_test = log_reg.predict(standardized_X_test)
print (pred_test)


# # To Be Updated Soon
# ###### [Go to top](#top)

# ## Credits (Reference)
# 
# > - [practicalAI - Goku Mohandas](https://github.com/GokuMohandas/practicalAI/)
# > - [GitHub Awesome Lists Topic](https://github.com/topics/awesome)
# > - [GitHub Machine Learning Topic](https://github.com/topics/machine-learning)
# > - [GitHub Deep Learning Topic](https://github.com/topics/deep-learning)
# > - [GitHub Awesome Lists Topic](https://github.com/topics/awesome)
# 
# ## License
# 
# [![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://raw.githubusercontent.com/ArunkumarRamanan/practicalAI/master/LICENSE)
# 
# ### Please ***UPVOTE*** my kernel if you like it or wanna fork it.
# 
# ##### Feedback: If you have any ideas or you want any other content to be added to this curated list, please feel free to make any comments to make it better.
# #### I am open to have your *feedback* for improving this ***kernel***
# ###### Hope you enjoyed this kernel!
# 
# ### Thanks for visiting my *Kernel* and please *UPVOTE* to stay connected and follow up the *further updates!*
