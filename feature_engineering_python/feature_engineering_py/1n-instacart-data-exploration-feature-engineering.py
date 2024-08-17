#!/usr/bin/env python
# coding: utf-8

# # 1 Introduction
# This kernel has been created by the [Information Systems Lab](http://islab.uom.gr) at the University of Macedonia,  Greece for the needs of the elective course  **Special Topics of Information Systems I**  at the [Business Administration](http://www.uom.gr/index.php?newlang=eng&tmima=2&categorymenu=2) department of the University of Macedonia, Greece.
# 
# ## 1.1 Business Insights
# In this notebook you will explore Instacart data in order to answer the following business questions:
# * When do customers order?
# * How many orders do customers make? 
# * How often do customers place orders?
# 
# ## 1.2 Python Skills
# In this notebook you will practice the following Python skills:
# * Import packages 
# * Import .csv files into Pandas DataFrames
# * Explore DataFrames
# * Use Pandas to create boxplots & barplots
# * Use Seaborn package to create histograms
# * Use Pyplot module of the Matplotlib package to create histograms
# * Use Pyplot module of the Matplotlib package to create subplots
# 
# ## 1.3 Packages and methods
# * pandas:  .shape , .info() , .head(), .boxplot(), .value_counts(), .plot() <br>
# * seaborn:  countplot() <br>
# * matplotlib.pyplot: figure(), xlabel(), ylabel(), title(), show(), subplots()

# # 2 Import Packages
# The first step is the installation of the necessary packages. In this notebook we use three packages, namely pandas, seaborn, and matplotlib.
# 
# * Pandas: a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive
# 
# * Seaborn: a Python data visualization library. It provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn is based on Matplotlib (see below).
# 
# * Matplotlib: a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms
# 
# The aforementioned Python packages are pre-installed for us, so we just need to load them. We use the command **import ** which loads the packages. The **as** keyword is used to indicate an alias to every package; an alternative way to call each imported package. 

# In[1]:


import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics


# # 3 Import Dataset
# The dataset that was opened up by Instacart is a relational set of files describing customers' orders over time. It consists of information about 3.4 million grocery orders from more than 200,000 Instacart users, distributed across 6 csv files. For each user, it provides between 4 and 100 of their orders, with the sequence of products purchased in each order. It also provides the week and hour of day the order was placed, and a relative measure of time between orders.
# 
# ## 3.1 Dataset Description
# * orders.csv: All the grocery orders
# ![Orders](https://imgur.com/835yq4H.png)
# * products.csv: All the products
# ![Products](https://imgur.com/T14BswO.png)
# * order_products_train.csv - order_products_prior.csv: These files specify which products were purchased in each order. order_products__prior.csv contains previous order contents for all customers. 'reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items.
# ![Order_Products](https://imgur.com/A4JSWju.png)
# * aisles.csv: All the different aisles
# ![Aisles](https://imgur.com/ZQZQTwn.png)
# * departments.csv: All the different departments
# ![Departments](https://imgur.com/HRp7KSV.png)
# ____

# # 3.2 Create DataFrames
# After that, we have to import the data from the available .CSV files. Towards this end, we use the pd.read_csv() function, which is included in the pandas package. This function facilitates importing large datasets into Python. Reading in your data with the <b>pd.read_csv()</b> function returns you a DataFrame, an object that can host data tables. For the time being we will only import the orders.csv file:

# In[2]:


orders = pd.read_csv('../input/orders.csv' )


# # 4 DataFrames exploration
# Now we show how we can get some initial information for the orders DataFrame.
# 
# The <b>.shape</b> retrieves the dimensions of an object. 

# In[3]:


orders.shape


# The <b>.info()</b> presents the columns' names, types and more details regarding the DataFrame 

# In[4]:


orders.info()


# While <b>.head()</b> returns the first rows of the DataFrame.

# In[5]:


#the argument in .head() represents how many first rows we want to get.
orders.head(12)


# What we can understand by looking at the result of the <b>.head()</b> at the orders DataFrame? 
# 
# We see that there is a sequence of all orders made from customers. One order per row. For example, we can see that user 1 has 11 orders, 1 of which is in the train set, and 10 of which are prior orders. The orders.csv does not include information about the products that were ordered. This piece of information is contained in the order_products.csv .

# ## Your turn 📚📝
# 
# You will now explore the departments.csv file. First you will need to import it and save it as a DataFrame:

# In[6]:


#1. Import departments.csv from directory: ../input/departments.csv'
departments = pd.read_csv('../input/departments.csv')


# Now use the appropiate method to get the first 10 rows of the DataFrame.

# In[7]:


departments.head(10)


# Find how many rows and columns the departments DataFrame has.

# In[8]:


departments.shape


# Can you get the type of each column?

# In[9]:


departments.info()


# # 5 When do customers order?
# Let’s have a look when customers buy groceries online. Towards this end, we explore the orders DataFrame.

# In[10]:


orders.head()


# ## 5.1 Hour of Day
# To find out how many orders are placed in each hour of the day, we use the **order_hour_of_day** column of the DataFrame. In particular, we need to count how many times each hour appears on the **order_hour_of_day**.
# For example, for the first 5 orders that have been placed on hours: 8, 7, 12, 7, 15, one order has been place at 8hr, two at 7hr, one at 12hr and one at 15hr.  To extend this to all values on **order_hour_of_day**, we use the <b>.value_counts( )</b> method, which returns a Series with the unique values that can be found on a column and how many times they appear on it. 

# In[11]:


order_hours = orders.order_hour_of_day.value_counts()
order_hours


# So, from the above Series we conclude that most orders were placed at 10hr and the fewest at 3hr. 
# With **.plot.bar( )** method of Pandas for Series we can create a barplot to visualize the results:

# In[12]:


#alternative syntax : order_hours.plot(kind='bar')
order_hours.plot.bar()


# Note that the results are presented in the same order as of order_hours, starting from the most occuring value to the least occurring value.

# Now we show how we can calulate the above series and visualize it using the <b>countplot( )</b> function of seaborn package. 
# The countplot( ) function counts observations in each categorical bin and then visualize the results in a histogram chart.
# 
# So for our example, we pass to the first argument (x= ) the column name which contain the hour of the day where an order has been placed and in the second argument (data = ) we pass the data frame where the column can be found on.

# In[13]:


#Remember that the alias that we have defined for seaborn is the sns.
sns.countplot(x="order_hour_of_day", data=orders, color='red')


# Looking at the histogram we can understand that there is a clear effect of hour of day on order volume. Most orders are between 8.00-18.00. Here the results are presented following the arithmetic sequence of the order_hour_of_day (0,1,2, .. ,23).
# 
# Now we show some examples of how to adjust the size of the plot, the color of the bar chart, the names of the axis and the title of the plot. 
# 
# To achieve this, we use the <b>matplotlib.pyplot</b> package (remember that it has alias plt) to edit further the produced plot. <br/>
# In this case, we use the following structure:
# 1. Define the size of our plot using the matplotlib.pyplot
# 2. Define the plot that we want to produce with seaborn
# 3. Add the names of axes and the title of plot using matplotlib.pyplot
# 4. Use the plt.show( ) from matplotlib.pyplot to produce our plot

# In[14]:


# Step one - define the dimensions of the plot (15 for x axis, 5 for y axis)
plt.figure(figsize=(15,5))

# Step two - define the plot that we want to produce with seaborn
# Here we also define the color of the bar chart as 'red'
sns.countplot(x="order_hour_of_day", data=orders, color='red')

# Step three - we define the name of the axes and we add a title to our plot
# fontsize indicates the size of the titles
plt.ylabel('Total Orders', fontsize=10)
plt.xlabel('Hour of day', fontsize=10)
plt.title("Frequency of order by hour of day", fontsize=15)

# Step four - we produce our plot
plt.show()


# The produced plot, has exactly the same information as the previous, but now is more easy to be interpreted.

# ## 5.2 Your turn: Day of Week 📚📝
# 
# Use the above analysis to find out which day of the week has the most orders. To answer this question, you will need to use the **order_dow** column of orders DataFrame.

# In[15]:


sns.countplot(x="order_dow" , data=orders )


# Which day has the most orders and which the fewest?
# 
# As you can see the produced plot is small and it is difficult to be interpreted. Try now to:
# 1. Create the same plot with dimensions 10x10
# 2. Use a color of your desire for the bars - the name of all available colors can be found here: [Available colors on Seaborn](https://python-graph-gallery.com/100-calling-a-color-with-seaborn/)
# 3. Add a proper title to the axes and the plot

# In[16]:


plt.figure(figsize=(10,10))
sns.countplot(x="order_dow", data=orders, color='red')
plt.ylabel('Volume of orders', fontsize=10)
plt.xlabel('Day of week', fontsize=10)
plt.title("Orders placed in each day of the week", fontsize=15)
plt.show()


# Now create a DataFrame that keeps only the first order of each customer

# In[17]:


orders_first = orders[orders.order_number==1]
orders_first.head()


# And a DataFrame that keeps only the second order of each customer

# In[18]:


orders_second = orders[orders.order_number==2]
orders_second.head()


# So to create two subplots of **order_dow** for first and second orders.

# In[19]:


#create a subplot which contains two plots; one down the other
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15,8))

#assign each plot to the appropiate axes
sns.countplot(ax=axes[0], x="order_dow", data=orders_first, color='red')
sns.countplot(ax=axes[1], x="order_dow", data=orders_second, color='red')

# produce the final plot
plt.show()


# ## 5.3 Create a countplot that combines days and hours
# Now we will visualise the distributions of different days ('order_dow') and different hours (order_hour_of_day) on the same plot.
# 
# Towards this end, we use the argument **hue**, which splits a variable based on an other variable. I our case we spit **order_hour_of_day** using the **order_dow** variable. 

# In[20]:


plt.figure(figsize=(15,5))
sns.countplot(x="order_hour_of_day", data=orders, color='red',  hue='order_dow')
plt.ylabel('Total Orders', fontsize=10)
plt.xlabel('Hour of day', fontsize=10)
plt.title("Frequency of order by hour of day", fontsize=15)
plt.show()


# What we get here is a plot that describes for each day the orders placed for each hour.

# # 6 How many orders do customers make? 
# ## 6.1 Create the countplot
# Now we want to find out how many orders do customers make.  To answer this question we will use again the orders DataFrame.
# Let's recall what information orders DataFrame include:

# In[21]:


orders.head(15)


# Orders DataFrame include a column called **order_number**, which shows when an order has been placed (e.g. 1st order=1, 2nd order=2 etc.) . For example, the customer with used_id=1 has placed in total 11 orders. Thereafter, we could use on column **order_number** the Pandas method <b>.value_counts( )</b> to find how many times each value appears.

# In[22]:


order_count = orders.order_number.value_counts()
order_count


# From the above Series we see that all customers (206209) have made at least 4 orders. In other words, all users have 4 orders with order_number= 1 , 2 , 3 & 4. Finally only 1374 customers have made 100 orders. <br/>
# 
# Once again, we will use the ready function countplot of seaborn to count how many times each value appears and visualize the results.

# In[23]:


# Set size 15x5 and bar color red
plt.figure(figsize=(15,5))
sns.countplot(x='order_number', data=orders, color='red')
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)
plt.show()


# ## 6.2 Modify the ticks on x-axis of a plot
# ### 6.2.1 Manually edit the ticks
# To address the problem of the overlapping labels on the x-axis, we manually edit the ticks on x-axis.
# 
# To achieve this, show a way to modify manually the x-ticks (the overlapping numbers). Towards this end we:
# * Assign the produced plot in a variable (in our case we name it 'graph')
# * Use the method .set( ) to set aesthetic parameters in one step [aesthetics definition; [ref.1 ](https://www.interaction-design.org/literature/book/the-encyclopedia-of-human-computer-interaction-2nd-ed/visual-aesthetics), [ref.2](http://www.visual-arts-cork.com/definitions/aesthetics.htm)]
# 
# 
# Note that this procedure is used on seaborn graphs.

# In[24]:


plt.figure(figsize=(15,5))
graph = sns.countplot(x='order_number', data=orders, color='red')
graph.set(xticks=[25,50,75,100], xticklabels=['25 orders','50 orders', '75 orders', '100 orders'] )
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)
plt.title('How many orders do customers make?')
plt.show()


# Have a look on arguments xticks & xticklabels of .set( ) method:
# * xticks=[25,50,75,100] indicates which ticks to select
# * xticklabels=['25 orders','50 orders', '75 orders', '100 orders'] ) indicates what labels to use on each tick
# 
# While xticks must match the corresponding labels of dependented value (x='order_number') , the xticklabels can have any name

# ### 6.2.2 Create a sequence for x-ticks; the use of built-in function range( ) 

# In[25]:


rg = list(range(0,101,10))
rg


# So in the above results we request a sequence starting from 0, ending to 101, with step 10. To retrieve the results of a range function we need to pass it to the list( ) function.
# 
# Now we use the above command as argument for both xticks & xticklabels on the produced plot above.

# In[26]:


plt.figure(figsize=(15,5))
graph=sns.countplot(x='order_number', data=orders, color='red')
graph.set( xticks=list( range(0,101,10) ), xticklabels=list( range(0,101,10) ) )
plt.ylabel('Total Customers', fontsize=10)
plt.xlabel('Total Orders', fontsize=10)
plt.title('How many orders do customers make?')
plt.show()


# # 7 How often do customers place orders?
# ## 7.1 Analysis of days_since_prior_order
# To answer this business question we examine the **days_since_prior_order** column. This column contains the number of days that have passed since a prior order. <br>
# With .max( ) method we can get the longest period that has passed since a prior order:

# In[27]:


orders.days_since_prior_order.max()


# With .mean() the average days that pass since a prior order:

# In[28]:


orders.days_since_prior_order.mean()


# But also we can get the .median() of days_since_prior_order:

# In[29]:


orders.days_since_prior_order.median()


# To have a broader view of variable <b>days_since_prior_order</b> from the <b>orders</b> DataFrame we create a [boxplot](https://en.wikipedia.org/wiki/Box_plot).
# With the pandas'  <b>.boxplot()</b> method for DataFrames, we can calculate the median as well as the quartiles of a set of observations. Now we create a boxplot for the column "days_since_prior_order" .

# In[30]:


# alternative syntax: orders.days_since_prior_order.plot(kind='box')
orders.boxplot('days_since_prior_order')


# From the above plot we see that 25% of the orders are placed at most 4 days after their previous order. In addition, 50% of the orders are placed between 4 to 15 days after their previous order.  

# ## 7.2 More orders mean more often orders?
# In the following example we check whether more active users (i.e., users with many total orders) order more often than users with few total orders. In particular we compare the behaviour of users with more than 10 orders and users with more than 20 orders
# 
# Towards this end we show how we can filter our data based on specific criteria. 

# ### 7.2.1 Select the orders from users with more 10 orders
# To select the orders from users with at least 10 orders, first we need to create a Series with the user_ids that have at least 10 orders. In this case we keep rows that have order_number equal to 11 (more than 10 orders).

# In[31]:


eleven = orders.order_number==11
eleven.head()


# And now we select to keep these user_id where the condition is True

# In[32]:


user_10 = orders.user_id[eleven]
user_10.head()


# In[33]:


user_10.shape


# Which are 101.696 unique user_id (customers).
# 
# And now we select to keep from orders all these rows with a user_id that .isin( ) user_10 Series.
# The method .isin() return a DataFrame showing whether each element in the DataFrame is contained in a Series.

# In[34]:


orders_10 = orders[orders.user_id.isin(user_10)]
orders_10.head()


# In[35]:


orders_10.shape


# Which are 2.757.619 orders.

# ## 7.2.2 Create comparative boxplots
# Now we follow the same procedure for users with more than 20 orders

# In[36]:


twentyone = orders.user_id[orders.order_number==21]
orders_20 = orders[orders.user_id.isin(twentyone)]
orders_20.head()


# And now create three subplots for orders, orders_10, orders_20 that create a boxplot for days_since_prior_order

# In[37]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(15,7))

orders.boxplot(column='days_since_prior_order', ax=axes[0])
orders_10.boxplot(column='days_since_prior_order',  ax=axes[1])
orders_20.boxplot(column='days_since_prior_order',  ax=axes[2])

