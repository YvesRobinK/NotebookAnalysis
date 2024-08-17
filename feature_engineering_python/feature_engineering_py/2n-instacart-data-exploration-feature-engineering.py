#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel has been created by the [Information Systems Lab](http://islab.uom.gr) at the University of Macedonia, Greece for the needs of the elective course Special Topics of Information Systems I at the [Business Administration](http://www.uom.gr/index.php?tmima=2&categorymenu=2) department of the University of Macedonia, Greece.
# 
# 
# # Business Insights
# * What is the size of the orders (basket size)?
# * Which products have the highest probability of being reordered?
# * How many reorders products do orders contain?
# 
# # Python Skills
# * Aggregate data to calculate new variables
# * Turn a Series to a DataFrame
# * Renaming DataFrame columns
# * Group data to filter them
# * Calculating ratios
# * Sorting values
# * Selecting rows
# * Sort results on a barplot
# * Visualize Frequency Distributions
# * Calculate allocation rate (percentage)
# 
# # Packages 
# * pandas: .group_by(), .count(), .max(), .filter(), .mean(),  .sort_values(), .iloc[ ]
# * seaborn: .barplot()
# * matplotlib: .hist()

# # Import data into Python
# We load the required packages 

# In[1]:


import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics


# and we load the CSV files that we will work with.

# In[2]:


orders = pd.read_csv('../input/orders.csv' )
products = pd.read_csv('../input/products.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')


# # 1. What is the size of the orders (basket size)?
# To answer this question we have to :
# 1. Find the basket size (number of products) of each order.
# 2. Find the number of orders for each basket size.
# 3. Visualize the results

# ## Step 1.1 Find the size (number of products) of each order.
# In this step we want explore the basket size of the orders. To get this piece of information, we will have to explore the order_products_prior 
# data frame, which contains all the products placed in each order.

# In[3]:


order_products_prior.head(12)


# As you can see, for the first order (order_id=2) , 9 products were placed in the cart. Our goal is to find how many products are included in each order.
# 
# To achieve this, we follow a procedure that consists of two steps:
# 1. Split our DataFrame into groups: The groups are created based on the different values that can be found on a specific column (in our case different order numbers in the column "order_id"). Note that the column has categorical data rather than actual values.
# 2. Apply an aggregation function on them: Aggregation functions are actually all these functions that can turn the values of a column of group into a single value. Some aggregation functions are the mean, count, sum, max & min.
# 
# In our case we use count() aggregation function which returns the number of values found on a column (in our case product_id). Subsequently, we will get the number of products placed on each order.

# In[4]:


size = order_products_prior.groupby('order_id')[['product_id']].count()
size.head(10)


# Actually with the .head(10) we have selected to see the order size of only 10 out of 3.214.874 orders. For example, the order with id=2 has 9 products, the id=3 has 8 products and so on. 
# 
# As you can see the results are saved in a column with label 'product_id' , the same as the label as of column that we applied the aggregation function on. As we do not want toget confused with the label of the initial column, we modify the column label of size DataFrame:

# In[5]:


size.columns= ['order_size']
size.head()


# ## Your Turn 📚📝
# How can you produce the above results with a different way?

# In[6]:


# First check the available data on order_products_prior
order_products_prior.head()


# In[7]:


# Write your answer
size = order_products_prior.groupby('order_id')[['add_to_cart_order']].max()
size.columns= ['order_size']
size.head()


# ## Step 1.2 Find the number of orders for each basket size.
# Now we groupby size DataFrame by order_size. We use the aggregating function count() on the same column to find the total orders for each order size.

# In[8]:


size_results = size.groupby('order_size')[['order_size']].count()
size_results.columns = ['total_orders']
size_results.head()


# ## Step 1.3 Visualize the results
# Now we visualize these results with the use of sns.barplot function. <br/>
# For the x-axis values, we will get the index ( order_size - [1,2,3 ...] ) from size DataFrame. <br/>
# And for the y-axis the column total_orders [156748, 186993, ...] of the same DataFrame.<br/>
# In addition we modify the range for the x-ticks starting from zero and ending to the highest value. 

# In[9]:


plt.figure(figsize=(15,10))
#size_of_order will be on our x-axis and total_orders the y-axis
graph = sns.barplot(size_results.index, size_results.total_orders)
# we modify the x-ticks
graph.set( xticks=list( range(0,size_results.index.max(),10) ), xticklabels=list( range(0,size_results.index.max(),10) ) )
plt.ylabel('Number of orders', fontsize=15)
plt.xlabel('Number of products', fontsize=15)
plt.show()


# # 2. Which products have the highest probability of being reordered?
# In this section we want to find the products which have the highest probability of being reordered. Towards this end it is necessary to define the probability as below:
# ![Probability](https://imgur.com/VLgKGeY.png)
# Example: The product with product_id=2 is included in 90 purchases but only 12 are reorders. So we have:  
# ![prob2](https://latex.codecogs.com/gif.latex?reordered%5C_pr%28product%5C_id%3D%3D2%29%3D%5Cfrac%7B12%7D%7B90%7D%3D0%2C133)
# 
# 
# ## 2.1. Remove products with less than 40 purchases
# ### 2.1.1 Filter with .shape[0]
# Before we proceed to this estimation, we remove all these products that have less than 40 purchases in order the calculation of the aforementioned ratio to be meaningful.
# 
# Have a look on order_products data frame:

# Using groupby() we create groups for each product and using filter( ) we keep only groups with more than 40 rows. Towards this end, we indicate a lambda function.

# In[10]:


# execution time: 25 sec
# the x on lambda function is a temporary variable which represents each group
# shape[0] on a DataFrame returns the number of rows
reorder = order_products_prior.groupby('product_id').filter(lambda x: x.shape[0] >40)
reorder.head()


# ### 2.1.2 Your Turn 📚📝
# How can you produce the above results with another filter?

# In[11]:


#execution time 30 sec
reorder = order_products_prior.groupby('product_id').filter(lambda x: x.product_id.count() >40)
reorder.head()


# ## 2.2 Group products, calculate the mean of reorders
# 
# Now to calculate the reorder probability we will use the aggregation function mean() to the reordered column. In the reorder data frame, the reordered column indicates that a product has been reordered when the value is 1.
# 
# So the mean() calculates how many times a product has been reordered, divided by how many times has been ordered in total. 
# 
# E.g., for a product that has been ordered 6 times in total, where 3 times has been reordered, the ratio will be:
# 
# ![example ratio](https://latex.codecogs.com/gif.latex?\bg_white&space;mean=&space;\frac{0&plus;1&plus;0&plus;0&plus;1&plus;1}{6}&space;=&space;0,5) 
# 
# Now we calculate the ratio for each product. The aggregation function is limited to column 'reordered' and it calculates the mean value of each group.

# In[12]:


reorder = reorder.groupby('product_id')[['reordered']].mean()
reorder.columns = ['reorder_ratio']
reorder.head()


# And now we sort the products by their mean and we select the 10 products which have the highest reorder probability

# In[13]:


reorder = reorder.sort_values(by='reorder_ratio', ascending=False)
reorder_10 = reorder.iloc[0:10]
reorder_10.head(10)


# ## 2.3 Visualize the results
# Here we show how we can visualize the results for the 10 products with the highest ratio. To make the bars ordered by the highest to the lowest value, we pass the argument <b> order=reorder_10.index </b> to the sns.barplot( ) function.

# 
# 

# In[14]:


plt.figure(figsize=(12,8))
sns.barplot(reorder_10.index, reorder_10.reorder_ratio, order=reorder_10.index)
plt.xlabel('10 top products \n Note that each ID corresponds to a product from products data frame', size=15)
plt.ylabel('Reorder probability', size=15)
#we set the range of y-axis to a bit lower from the lowest probability and a bit higher from the higest probability
plt.ylim(0.87,0.95)
plt.show()


# ## 2.4 Your Turn 📚📝
# 
# Can you get the name of product with the highest probability?
# You can search for it on products DataFrame.

# In[15]:


products[products.product_id == 6433]


# ## 2.5 Create a distribution plot of reorder probability
# Now we want to summarize the information for reorder probability of all products. To achieve this, we create a distribution plot for the reordered ratio with the hist( ) plot of matplotlib. The argument bins=100 indicates that we want 100 bins for our distribution

# In[16]:


plt.hist(reorder.reorder_ratio, bins=100)
plt.show()


# # 3. How frequent an order has reordered products?
# ## 3.1 Group orders, calculate the mean of reorders 📚📝
# In this business insight we create a ratio which shows for each order in what extent has products that have been reordered in the past.
# So we create the following ratio: <br>
# ![ratio](https://latex.codecogs.com/gif.latex?probability\&space;reordered=&space;\frac{count\&space;of\&space;reorder\&space;products}{total\&space;products\&space;purchased})
# 
# To create this ratio we groupby order_products_prior by each order and then calculate the mean of reordered.

# In[17]:


reorder_ratio_orders= order_products_prior.groupby('order_id')[['reordered']].mean()
reorder_ratio_orders.columns= ['reordered_ratio']
reorder_ratio_orders.head()


# A value equal to 1 means that all products have been reordered where 0 means none has been reordered.
# ## 3.2 Create a distribution plot of reorder probability
# Now we create a distribution for the ratio across the different orders.
# 
# 

# In[18]:


plt.hist(reorder_ratio_orders.reordered_ratio, bins=20)
plt.show()


# ## 3.3 Your Turn 📚📝
# Count how many orders have reorder ratio = 1. What is the allocation rate (percentage) compared to total orders?
# 

# In[19]:


# Write your code here
reorder_ratio_orders[reorder_ratio_orders.reordered_ratio== 1].count()


# In[20]:


ratio_one_count = reorder_ratio_orders[reorder_ratio_orders.reordered_ratio== 1].count()
all_orders = reorder_ratio_orders.reordered_ratio.count()
percentage = (ratio_one_count / all_orders)*100
print('Orders with reorder ratio = 1 are ' + str(round(percentage[0], 2)) + ' % of all orders.')

