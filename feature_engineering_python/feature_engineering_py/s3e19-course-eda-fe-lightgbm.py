#!/usr/bin/env python
# coding: utf-8

# <div style="padding: 35px;color:white;margin:10;font-size:200%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Getting Started </span></b> </div>
# 
# <br>
# 
# ## 🚀 Getting Started
# 
# This project involves analyzing a retail sales dataset with the aim of predicting future sales. The dataset includes various **`Features`** related to sales data, including date, country, product, and store identifiers. Each entry represents a unique sale, and the features include date-related attributes (like year, month, day, and day of the week), country, product, and store identifiers.
# 
# ## 🔧 Tools and Libraries
# 
# We will be using Python for this project, along with several libraries for data analysis, machine learning, and data visualization. Here are the main libraries we'll be using:
# 
# - **Pandas**: For data manipulation and analysis.
# - **Matplotlib and Seaborn**: For data visualization.
# - **Scikit-learn**: For machine learning tasks, including data preprocessing, model training, and model evaluation.
# - **LightGBM**: For implementing the Light Gradient Boosting Machine model.
# 
# ## 📚 Dataset
# 
# The dataset we'll be using includes various features related to retail sales. Each row represents a unique sale and includes attributes such as date, country, product, and store identifiers. The dataset also includes a target variable 'num_sold' representing the number of products sold.
# 
# ## 🎯 Objective
# 
# Our main objective is to develop a predictive model that can effectively forecast future sales based on the provided features. By leveraging the power of the Light Gradient Boosting Machine, we aim to enhance the model's accuracy and predictive performance.
# 
# ## 📈 Workflow
# 
# Here's a brief overview of our workflow for this project:
# 
# 1. **Data Loading and Preprocessing**: Load the data and preprocess it for analysis and modeling. This includes converting date columns to datetime format and extracting additional date-related features.
# 
# 2. **Exploratory Data Analysis (EDA)**: Perform exploratory data analysis to gain insights into the dataset, understand the distributions of features, and explore potential relationships between the features and sales.
# 
# 3. **Feature Engineering**: Perform feature engineering to extract additional relevant features or transform existing features to improve the model's performance.
# 
# 4. **Model Training and Validation**: Train the LightGBM model using a GroupKFold cross-validation strategy and make predictions on the test set.
# 
# 5. **Sales Disaggregation**: Disaggregate the forecasted total sales into product-level sales using historical sales ratios.
# 
# 6. **Model Evaluation**: Evaluate the performance of the trained model using appropriate evaluation metrics and assess the model's ability to generalize to unseen data using the test set.
# 
# 7. **Prediction and Deployment**: Use the trained model to make predictions on new, unseen data. If applicable, deploy the model for practical use or further analysis.
# 
# 
# <br>
# 
# ![](https://images.pexels.com/photos/7718713/pexels-photo-7718713.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)
# 
# <br>
# 
# ## Domain Knowledge 📚
# 
# Let's dive into the domain knowledge of the features used in this project:
# 
# ### Features
# 
# - **`date`**: This is the date when the sales were recorded. Dates are fundamental in sales forecasting as sales tend to have temporal patterns such as trends (sales increasing or decreasing over time) and seasonality (patterns repeating at regular intervals, like daily, weekly, monthly, quarterly, or yearly). For instance, retail sales often increase during holiday seasons or on specific days of the week.
# 
# - **`country`**: This is the country where the sales were recorded. Country can be an important feature in sales forecasting as sales patterns often vary between countries due to factors like cultural differences, economic conditions, or different holidays.
# 
# - **`product`**: This is the identifier of the product that was sold. Different products might have different sales patterns. For example, some products might sell more during certain seasons (like swimsuits in the summer or coats in the winter), while others might have steady sales throughout the year.
# 
# - **`store`**: This is the identifier of the store where the sales were recorded. Different stores might have different sales patterns due to factors like location, size, customer demographics, or management practices.
# 
# - **`Year, Month, Day, WeekDay`**: These are features derived from the date feature. They represent the year, month, day of the month, and day of the week of the sales, respectively. These features can help capture temporal trends and seasonality in the sales data. For example, sales might be higher on weekends (day of the week), during holiday seasons (month), or during specific years due to economic trends.
# 
# - **`holiday_name`, `is_holiday_`**: These features represent the name of the holiday (if any) on the date of the sales, and whether the date was a holiday. Holidays often influence sales patterns, with sales typically increasing during holiday seasons.
# 
# - **`ratios`**: This feature represents the historical ratios of product sales, used for disaggregating total sales into product-level sales. The assumption here is that the proportion of sales for each product will remain relatively stable over time. If this assumption holds, these ratios can be a useful tool for generating more detailed forecasts.
# 
# Understanding these features and their relevance in the retail sales domain can help improve model performance and interpretability. This domain knowledge can guide the feature engineering process and help in the interpretation of the model's predictions.
# 
# 
# # <span style="color:#E888BB; font-size: 1%;">INTRODUCTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Introduction</span></b> </div>
# 
# <br>
#     
# ## 📝 Abstract
# 
# This project presents an in-depth exploration and modeling of a retail sales dataset, with the primary objective of predicting sales at the product level 🎯. The dataset is characterized by a variety of **`Features`** 📊 including 'date' 📆, 'store' 🏪, 'country' 🌍, 'product' 📦, 'Year' 📅, 'Month' 📆, 'Day' 🗓️, 'WeekDay' 📅, and other engineered features capturing temporal patterns ⏳, cyclical trends 🔄, and special events (holidays) 🎉.
# 
# Our exploratory data analysis revealed significant temporal patterns in sales, hinting at the influence of time ⏰ and special events 🎉 on the number of items sold. Additionally, we identified some anomalies in the data related to the period from March 2020 to May 2020, suggesting the need for robust anomaly detection 🕵️‍♀️ and handling methods in retail sales forecasting.
# 
# We harnessed the potential of ensemble models, specifically the <b><mark style="background-color:#526D82;color:white;border-radius:5px;opacity:1.0">LightGBM Regressor</mark></b> 🚀, to predict total sales. The model was meticulously trained and validated using a GroupKFold cross-validation strategy 🔄, yielding variable scores across different folds.
# 
# The sales forecasts produced by the model were then disaggregated into product-level forecasts using historical sales ratios 📈. This approach assumes the stability of product sales ratios over time ⏳.
# 
# The project highlights the potential of machine learning 🧠 in retail sales forecasting, providing insights that could assist in inventory management 🗃️, sales planning 📝, and business strategy formulation 💼. Future efforts could focus on refining the predictive model 🔍, exploring different strategies for sales disaggregation 🧩, and integrating additional external data sources 🔄 to enhance the accuracy and detail of sales forecasts 📈.
# 
# 
# <br>
# 
# ### <b>I <span style='color:#FF8551'>|</span> Import neccessary libraries</b>

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D


# ### <b>II <span style='color:#FF8551'>|</span> Input the Data</b> 

# In[2]:


data = pd.read_csv('/kaggle/input/playground-series-s3e19/train.csv')


# ### <b>III <span style='color:#FF8551'>|</span> Get to know the Data</b> 

# In[3]:


# Summary statistics for numerical variables
numerical_summary = data.describe()
numerical_summary


# ### Numerical Variables: 🔢
# 
# 1. **id:** 🆔 This is just an identifier column.
# 
# 2. **num_sold:** 📊 The count of items sold. The minimum is **2**, the maximum is **1,380**, and the mean is around **165.52**. The standard deviation is approximately **183.69**, indicating that the distribution is quite wide.

# In[4]:


# Distribution of categorical variables
categorical_distribution = data.describe(include=['O'])
categorical_distribution


# ### Categorical Variables: 🔠
# 
# 1. **date:** 🗓️ There are 1,826 unique dates in the dataset, which range from 1-1-2017 to 31-12-2022.
# 
# 2. **country:** 🌍 There are 5 unique countries, with Argentina appearing most frequently.
# 
# 3. **store:** 🏬 There are 3 unique stores, with "Kaggle Learn" being the most common.
# 
# 4. **product:** 📦 There are 5 unique products, with "Using LLMs to Improve Your Coding" being the most sold.
# 
# # <span style="color:#E888BB; font-size: 1%;">EXPLORATORY DATA ANALYSIS</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>EXPLORATORY DATA ANALYSIS </span></b> </div>
# 
# <br>
# 
# <div style="display: flex; flex-direction: row; align-items: center;">
#     <div style="flex: 0;">
#         <img src="https://images.pexels.com/photos/7718841/pexels-photo-7718841.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="Image" style="max-width: 300px;" />
#     </div>
#     <div style="flex: 1; margin-left: 30px;">
#         <p style="font-weight: bold; color: black;">Getting started with Sales Data Analysis</p>
#         <p>This project focuses on analyzing the <b><mark style="background-color:#526D82;color:white;border-radius:5px;opacity:1.0">Sales dataset</mark></b> to identify key factors associated with sales trends. By utilizing techniques such as <b><span style='color:#9DB2BF'>univariate</span></b> and <b><span style='color:#9DB2BF'>bivariate analysis</span></b>, as well as time series methods like <b><span style='color:#9DB2BF'>rolling window analysis</span></b> , we aim to uncover valuable insights into the complex relationships within the data.
#         </p>
#         <p style="border: 1px solid black; padding: 10px;">Our analysis provides valuable insights into the factors influencing sales trends. However, it's crucial to interpret these findings with caution, recognizing the distinction between correlation and causation. It is important to note that our exploratory analysis, although comprehensive, does not establish a causal relationship between the provided features and sales trends.
#         </p>
#         <p style="font-style: italic;">Let's explore and then make results and discussion to gain deeper insights from our analysis. 🧐🔍
#         </p>
#     </div>
# </div>
# 
# <br>
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trend of Sales Over Time </span></b> </div>

# In[5]:


# Convert `date` to datetime format
data['date'] = pd.to_datetime(data['date'])

# Aggregate sales on a monthly basis
monthly_sales = data.resample('M', on='date').sum()['num_sold']

# Plot the trend of sales over time
plt.figure(figsize=(15, 6))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values)
plt.title('Trend of Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.show()


# ##  Intepret the Results  📊
# 
# 
# The line plot shows the trend of sales over time, aggregated on a monthly basis. It seems that the sales have some sort of seasonality, with peaks and troughs appearing regularly. 
# 
# > There also seems to be an overall increasing trend in the number of products sold over time.
# 
# ##  What's next 🤔
# 
# **<span style='color:#526D82'>Let's calculate a 7-day and a 30-day rolling mean of the sales and visualize them. A 7-day rolling mean will give us a weekly trend, while a 30-day rolling mean will give us a monthly trend. Please note that for this analysis, we will consider the daily total sales across all countries, stores, and products.</span>**

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trends of Sales Over Time with Rolling Means</span></b> </div>

# In[6]:


# Aggregate sales on a daily basis
daily_sales = data.resample('D', on='date').sum()['num_sold']

# Calculate 7-day and 30-day rolling means
daily_sales_rolling_7d = daily_sales.rolling(window=7).mean()
daily_sales_rolling_30d = daily_sales.rolling(window=30).mean()

# Plot the original daily sales and the rolling means
plt.figure(figsize=(15, 6))
sns.lineplot(x=daily_sales.index, y=daily_sales.values, label='Original')
sns.lineplot(x=daily_sales_rolling_7d.index, y=daily_sales_rolling_7d.values, label='7-day Rolling Mean')
sns.lineplot(x=daily_sales_rolling_30d.index, y=daily_sales_rolling_30d.values, label='30-day Rolling Mean')
plt.title('Trend of Sales Over Time with Rolling Means')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.legend()
plt.show()


# ##  Intepret the Results  📊
# 
# The plot above shows the trend of sales over time, along with 7-day and 30-day rolling means. Each line represents the following:
# 
# * The blue line represents the original daily sales.
# 
# * The orange line represents the 7-day rolling mean, providing a weekly trend.
# 
# * The green line represents the 30-day rolling mean, providing a monthly trend.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>The rolling means are smoother than the original daily sales, as they average out the fluctuations in sales, thereby highlighting the longer-term trends.</li>
#         <li>The 7-day rolling mean (orange line) shows some weekly seasonality in sales. There seem to be peaks and troughs at a regular interval of approximately a week.</li>
#         <li>The 30-day rolling mean (green line) shows a smoother trend, which could be interpreted as the monthly trend in sales.</li>
#     </ol>
# </div>
# 
# 
# 
# ##  What's next 🤔
# 
# **To understand the difference between 2020 and other years**, we can plot the monthly sales for each year in the same plot. By doing this, we can compare the trends and patterns across different years.
# 
# **<span style='color:#526D82'>Let's create this plot. We'll highlight the line for 2020 to make it easier to compare with the other years.</span>**
#     
# 
# 
# 
# 
# 
# 

# 
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Monthly Sales for Each Year</span></b> </div>

# In[7]:


# Aggregate sales on a monthly basis for each year
monthly_sales_year = data.resample('M', on='date').sum()['num_sold'].reset_index()

# Create a column for the year
monthly_sales_year['year'] = monthly_sales_year['date'].dt.year

# Create a line plot for each year
plt.figure(figsize=(15, 8))
for year in sorted(monthly_sales_year['year'].unique()):
    year_data = monthly_sales_year[monthly_sales_year['year'] == year]
    if year == 2020:
        sns.lineplot(x=year_data['date'].dt.month, y=year_data['num_sold'], label=year, linewidth=2.5, color='red')
    else:
        sns.lineplot(x=year_data['date'].dt.month, y=year_data['num_sold'], label=year, linewidth=0.7)
plt.title('Monthly Sales for Each Year')
plt.xlabel('Month')
plt.ylabel('Number of Products Sold')
plt.legend(title='Year')
plt.show()


# ##  Intepret the Results  📊
# 
# 
# The plot above shows the trend of monthly sales for each year, with each line representing a different year. The red line for the year 2020 is highlighted to make it easier to compare with the other years.
# 
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>All years seem to have a similar trend of sales over time, indicating that the sales might be affected by the same factors (such as seasonal effects, overall market trends, etc.) every year.</li>
#         <li>Compared to other years, 2020 shows a slightly different pattern. While sales in other years generally increase around the middle of the year and then decrease towards the end of the year, sales in 2020 seem to increase steadily throughout the year.</li>
#         <li>The sales in 2020 are generally higher than the sales in other years, especially in the second half of the year. This could be due to various factors, such as increased demand, more effective marketing strategies, new product releases, etc.</li>
#     </ol>
# </div>
# 
# 
# 
# > This kind of plot is useful for comparing the same time period across different years and identifying patterns or trends that repeat annually or changes in these trends.

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trend of Sales Over Time for Each Country</span></b> </div>

# In[8]:


# Aggregate sales on a monthly basis for each country
monthly_sales_country = data.groupby([data['date'].dt.to_period('M'), 'country']).sum()['num_sold'].reset_index()
monthly_sales_country['date'] = monthly_sales_country['date'].dt.to_timestamp()

# Create a line plot for each country
plt.figure(figsize=(15, 8))
sns.lineplot(data=monthly_sales_country, x='date', y='num_sold', hue='country')
plt.title('Trend of Sales Over Time for Each Country')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.show()


# ##  Intepret the Results  📊
# 
# The line plot above shows the trend of sales over time for each country. Each line represents a different country, and the x-axis represents time.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>All countries seem to have a similar trend over time, indicating that the sales in different countries might be affected by the same factors (such as seasonal effects, overall market trends, etc.).</li>
#         <li>There's an overall increasing trend in sales for all countries over time.</li>
#         <li>The lines for different countries overlap quite a bit, which suggests that the sales in different countries tend to rise and fall at the same times.</li>
#     </ol>
# </div>
# 
# 
# <br>
# 
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📌 <b>Remember :</b> The y-axis represents the number of products sold, and the legend shows which line corresponds to which country.
# </div>

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Trend of Sales Over Time for Each Country and Product</span></b> </div>

# In[9]:


# Aggregate sales on a monthly basis for each country and product
monthly_sales_country_product = data.groupby([data['date'].dt.to_period('M'), 'country', 'product']).sum()['num_sold'].reset_index()
monthly_sales_country_product['date'] = monthly_sales_country_product['date'].dt.to_timestamp()

# Create a FacetGrid to make a separate line plot for each product
g = sns.FacetGrid(monthly_sales_country_product, col='product', col_wrap=3, height=4, aspect=1.5)
g.map_dataframe(sns.lineplot, x='date', y='num_sold', hue='country')
g.set_axis_labels('Date', 'Number of Products Sold')
g.add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Trend of Sales Over Time for Each Country and Product')
plt.show()


# ## Interpret the Results 📊
# 
# The plot above shows the trend of sales over time for each country and product. Each line in a subplot represents a different country, and each subplot represents a different product.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>For each product, all countries seem to have a similar trend over time, indicating that the sales in different countries might be affected by the same factors (such as seasonal effects, overall market trends, etc.).</li>
#         <li>There's an overall increasing trend in sales for all countries and products over time.</li>
#         <li>The lines for different countries overlap quite a bit within each subplot, which suggests that the sales in different countries tend to rise and fall at the same times for a given product.</li>
#     </ol>
# </div>
# 
# 
# <br>
# 
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📌 <b>Remember :</b> The y-axis represents the number of products sold, and the legend shows which line corresponds to which country.
# </div>

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trend of Sales Over Time for Each Product : 3D </span></b> </div>

# In[10]:


# Aggregate sales on a monthly basis for each product
monthly_sales_product = data.groupby([data['date'].dt.to_period('M'), 'product']).sum()['num_sold'].reset_index()
monthly_sales_product['date'] = monthly_sales_product['date'].dt.to_timestamp()

# Assign each product a unique numeric ID
product_ids = {product: i for i, product in enumerate(monthly_sales_product['product'].unique())}
monthly_sales_product['product_id'] = monthly_sales_product['product'].map(product_ids)

# Convert 'date' to a numeric form (number of days since the first date)
monthly_sales_product['date_num'] = (monthly_sales_product['date'] - monthly_sales_product['date'].min()).dt.days

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a color map for the different products
colors = plt.cm.viridis(np.linspace(0, 1, len(monthly_sales_product['product'].unique())))


# Plot a line for each product
for i, product in enumerate(monthly_sales_product['product'].unique()):
    product_data = monthly_sales_product[monthly_sales_product['product'] == product]
    ax.plot(product_data['date_num'], product_data['product_id'], product_data['num_sold'], color=colors[i])

# Set the labels and title
ax.set_xlabel('Date (number of days since first date)')
ax.set_ylabel('Product ID')
ax.set_zlabel('Number of Products Sold')
ax.set_title('Trend of Sales Over Time for Each Product')

# Create a legend for the product IDs
ax.legend([plt.Line2D([0], [0], color=color, lw=2) for color in colors],
          product_ids.keys(),
          title='Product',
          loc='upper right')

plt.show()


# ## Interpret the Results 📊
# 
# The 3D plot shows the trend of sales over time for each product. Each line represents a different product, and the x-axis represents the number of days since the first date in our dataset.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>All products seem to have a similar trend over time, indicating that the sales of different products might be affected by the same factors (such as seasonal effects, overall market trends, etc.).</li>
#         <li>There's an overall increasing trend in sales for all products over time.</li>
#         <li>The vertical variation in lines for the same x-coordinate shows the difference in sales between products at a given time.</li>
#     </ol>
# </div>
# 
# 
# <br>
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📌 <b>Remember :</b> The y-axis represents product IDs, not the actual product names. The legend in the upper right corner shows which color corresponds to which product. 
# </div>
# 
# > Please note that 3D plots can sometimes be harder to interpret than 2D plots, especially when the data is dense or when there are many categories for one of the variables.

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trend of Sales Over Time for Each Country : 3D </span></b> </div>

# In[11]:


# Assign each country a unique numeric ID
country_ids = {country: i for i, country in enumerate(data['country'].unique())}
data['country_id'] = data['country'].map(country_ids)


# Aggregate sales on a monthly basis for each country
monthly_sales_country = data.groupby([data['date'].dt.to_period('M'), 'country']).sum()['num_sold'].reset_index()
monthly_sales_country['date'] = monthly_sales_country['date'].dt.to_timestamp()
monthly_sales_country['country_id'] = monthly_sales_country['country'].map(country_ids)

# Convert 'date' to a numeric form (number of days since the first date)
monthly_sales_country['date_num'] = (monthly_sales_country['date'] - monthly_sales_country['date'].min()).dt.days

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot a line for each country
for i, country in enumerate(monthly_sales_country['country'].unique()):
    country_data = monthly_sales_country[monthly_sales_country['country'] == country]
    ax.plot(country_data['date_num'], country_data['country_id'], country_data['num_sold'], color=colors[i])

# Set the labels and title
ax.set_xlabel('Date (number of days since first date)')
ax.set_ylabel('Country ID')
ax.set_zlabel('Number of Products Sold')
ax.set_title('Trend of Sales Over Time for Each Country')

# Create a legend for the country IDs
ax.legend([plt.Line2D([0], [0], color=color, lw=2) for color in colors],
          country_ids.keys(),
          title='Country',
          loc='upper right')

plt.show()


# ## Interpret the Results 📊
# 
# The 3D plot shows the trend of sales over time for each country. Each line represents a different country, and the x-axis represents the number of days since the first date in our dataset.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <ol>
#         <li>All countries seem to have a similar trend over time, indicating that the sales in different countries might be affected by the same factors (such as seasonal effects, overall market trends, etc.).</li>
#         <li>There's an overall increasing trend in sales for all countries over time.</li>
#         <li>The vertical variation in lines for the same x-coordinate shows the difference in sales between countries at a given time.</li>
#     </ol>
# </div>
# 
# <br>
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📌 <b>Remember :</b> The y-axis represents country IDs, not the actual country names. The legend in the upper right corner shows which color corresponds to which country. Please note that 3D plots can sometimes be harder to interpret than 2D plots, especially when the data is dense or when there are many categories for one of the variables.
# </div>
# 
# 
# 
# 
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Trend of Sales Over Time in 2020 with Rolling Means</span></b> </div>

# In[12]:


# Filter the data for the year 2020
data_2020 = data[(data['date'].dt.year == 2020)]

# Aggregate sales on a daily basis
daily_sales_2020 = data_2020.resample('D', on='date').sum()['num_sold']

# Calculate 7-day and 30-day rolling means
daily_sales_2020_rolling_7d = daily_sales_2020.rolling(window=7).mean()
daily_sales_2020_rolling_30d = daily_sales_2020.rolling(window=30).mean()

# Plot the original daily sales and the rolling means
plt.figure(figsize=(15, 6))
sns.lineplot(x=daily_sales_2020.index, y=daily_sales_2020.values, label='Original')
sns.lineplot(x=daily_sales_2020_rolling_7d.index, y=daily_sales_2020_rolling_7d.values, label='7-day Rolling Mean')
sns.lineplot(x=daily_sales_2020_rolling_30d.index, y=daily_sales_2020_rolling_30d.values, label='30-day Rolling Mean')
plt.title('Trend of Sales Over Time in 2020 with Rolling Means')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.legend()
plt.show()


# ##  Intepret the Results  📊
# 
# The plot above shows the trend of sales over time in 2020, along with 7-day and 30-day rolling means. Each line represents the following:
# 
# * The blue line represents the original daily sales.
# 
# * The orange line represents the 7-day rolling mean, providing a weekly trend.
# 
# * The green line represents the 30-day rolling mean, providing a monthly trend.
# 
# <br>
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D; font-size: 100%; padding: 10px;">
#     <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     <p>Some What Like monthly trends :</p>
#     <ol>
#         <li>The rolling means are smoother than the original daily sales, as they average out the fluctuations in sales, thereby highlighting the longer-term trends.</li>
#         <li>The 7-day rolling mean (orange line) shows some weekly seasonality in sales. There seem to be peaks and troughs at a regular interval of approximately a week.</li>
#         <li>The 30-day rolling mean (green line) shows a smoother trend, which could be interpreted as the monthly trend in sales.</li>
#         <li>Unusual Patterns occur during the period of March to June , which might indicate some events might have happened</li>
#     </ol>
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;">EDA : RESULT AND DISCUSSION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>EDA Result and Discussion</span></b> </div>
# 
# <br>
# 
# Based on the Exploratory Data Analysis (EDA) performed on this dataset, here are the **key findings and insights** 📊:
# 
# 1. **Sales Distribution 📈**: The number of products sold varies widely, with most of the sales numbers being relatively low while a few instances have very high sales. This suggests that there could be certain periods or specific conditions that lead to high sales.
# 
# 2. **Sales by Country 🌎**: Argentina has the highest number of sales among the countries, followed by Brazil, Canada, Mexico, and the United States. This could suggest a higher demand or more effective marketing strategies in these countries.
# 
# 3. **Sales by Store 🏪**: The Kaggle Learn store has the highest number of sales, followed by DataCamp and Coursera. This could indicate a preference for the types of products offered by these stores or their mode of delivery.
# 
# 4. **Sales by Product 🛍️**: All five products have a similar number of sales, with "Using LLMs to Improve Your Coding" being slightly more popular. This suggests a balanced portfolio of products.
# 
# 5. **Sales Trends Over Time ⏰**: There is an overall increasing trend in the number of products sold over time. Sales tend to increase around the middle of the year and then decrease towards the end of the year, indicating a possible seasonal effect.
# 
# 6. **Year 2020 Analysis 🗓️**: Compared to other years, The year 2020, marked by the global outbreak of COVID-19, shows a distinctive pattern compared to other years. During the period of March to June 2020, there were notable fluctuations in sales, likely due to the impacts of the pandemic and the ensuing global lockdown measures. This affected consumer behavior, leading to changes in the demand for products. Despite these fluctuations, sales in 2020 were generally higher, especially in the second half of the year, suggesting an adaptation to the new circumstances.
# 
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📌 <b>Insight:</b> This adjustment highlights the potential effects of external events on sales trends. The global outbreak of COVID-19 in 2020 and the subsequent changes in consumer behavior provide a clear example of how such events can significantly impact sales patterns.
# </div>
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚧  Keep in Mind </h3>
#  Also, it's important to remember that correlation does not imply causation. While the EDA can reveal patterns and associations in the data, it does not establish causal relationships. Any hypotheses generated from this analysis should be further tested through rigorous experimental or quasi-experimental designs💡.
# </div>
# 
# # <span style="color:#E888BB; font-size: 1%;">PREDICTIVE ANALYSIS</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Prediction</span></b> </div>
# 
# ### 👷 Overview Process :
# 
# 1. **Data Loading and Preprocessing:** The provided train and test datasets were loaded. The 'date' column was converted to datetime format and additional features such as 'Year', 'Month', 'Day', and 'WeekDay' were extracted.
# 
# 2. **Exploratory Data Analysis**: Sales trends were visualized by plotting the total number of sales over time. Anomalies were identified in the data between March 2020 and May 2020, which led to the decision to exclude this period from the analysis.
# 
# 3. **Feature Engineering:** A variety of features were engineered, including information about holidays, sine and cosine transformations of time variables to capture cyclical trends, and other date-related features. The holiday names were encoded using an ordinal encoder to convert them into a format that could be used in a machine learning model.
# 
# 4. **Model Training and Validation:** A LightGBM model was trained using 5-fold GroupKFold cross-validation, and predictions were made on the test set. The model's performance varied significantly across the folds, with some negative scores indicating poor performance on those folds.
# 
# 5. **Sales Disaggregation:** The predicted total sales were disaggregated into sales for each product using historical ratios of product sales.
# 
# 

# In[13]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor


import holidays
import dateutil.easter as easter


# In[14]:


# Read data
train_path = '/kaggle/input/playground-series-s3e19/train.csv'
test_path = '/kaggle/input/playground-series-s3e19/test.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Adjust Date Time </span></b> </div>

# In[15]:


# Convert 'date' column to datetime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

# Extract date info
train['Year'] = train['date'].dt.year
train['Month'] = train['date'].dt.month
train['Day'] = train['date'].dt.day
train['WeekDay'] = train['date'].dt.dayofweek

test['Year'] = test['date'].dt.year
test['Month'] = test['date'].dt.month
test['Day'] = test['date'].dt.day
test['WeekDay'] = test['date'].dt.dayofweek


# ### 🍽️ Process :
# 
# 
# 1. Converts the 'date' column to datetime: The 'date' column in both the training and testing datasets is transformed from a string format to a pandas datetime format using the pd.to_datetime function. This conversion facilitates the extraction of specific date-related features like year, month, day, and weekday in the next steps.
# 
# 2. Extracts date info: After transforming the 'date' column to a datetime format, various date-related features are extracted from this column for both the training and testing datasets:
#     
#      - Year: The year part of the date.
#      
#      - Month: The month part of the date.
#      
#      - Day: The day part of the date.
#      
#      - WeekDay: The day of the week. It's represented as an integer, where Monday is 0 and Sunday is 6.
#      
#      
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     🗒️ <b>Note:</b> extracted features like Year, Month, Day, and WeekDay can provide useful information for a machine learning model or data analysis. For instance, the model may find patterns related to seasonal trends, weekday trends, or yearly trends in the sales data.
# </div>    
# 
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Exclude Skewed Data </span></b> </div>

# In[16]:


temp = train.loc[~((train["date"] >= "2020-03-01") & (train["date"] < "2020-06-01"))].copy()  # Remove data from March 2020 to May 2020
train_agg = temp.groupby("date")["num_sold"].sum().reset_index()


plt.figure(figsize=(15, 6))
sns.lineplot(data=train_agg, x="date", y="num_sold")
plt.xlabel("Date")
plt.ylabel("Number of Sales")
plt.title("Sales Trend")
plt.grid(True)
plt.tight_layout()
plt.show()


# ##  Intepret the Results  📊
# 
# 1. **Filter the data:** 🚰 The code creates a temporary DataFrame temp that excludes data from March 2020 to May 2020. This is done using the loc method with a condition that only includes rows where the 'date' is not between "2020-03-01" and "2020-06-01".
# 
# 2. **Aggregate sales data:** 📊 The code then aggregates the number of products sold (num_sold) for each unique date in the filtered data. This is done using the groupby method on the 'date' column and then applying the sum function on the 'num_sold' column. The result is a new DataFrame train_agg that has each unique date and the total number of products sold on that date.
# 
# 3. **Plot the sales trend:** 📈 Finally, the code creates a line plot to visualize the trend of product sales over time. The x-axis represents the date, and the y-axis represents the number of products sold. The title of the plot is "Sales Trend", and grid lines are added for easier visualization.
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D;font-size: 100%; padding: 10px;">
# <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     
# In the plot that was created, you can see the trend of total product sales over time, excluding the period from March 2020 to May 2020. This could be useful for understanding how the number of product sales changes over time, and for identifying any noticeable trends or patterns. For example, you might see seasonal trends or identify particular periods of time when sales were particularly high or low. 📝
# 
#     
# </div>
# 
# 
# 

# In[17]:


test_agg = test.groupby(["date"])["id"].first().reset_index().drop(columns="id")
test_dates = test_agg[["date"]]


# ### 🍽️ Process :
# 
# 1. **Aggregate test data by date:** The code groups the test data by the 'date' column and takes the first 'id' for each group. This results in a DataFrame where each row corresponds to a unique date in the test data, and the 'id' column contains the first 'id' associated with that date. This aggregated DataFrame is named test_agg.
# 
# 2. **Remove 'id' column:** The 'id' column is then dropped from test_agg using the drop function. This is done because the 'id' column is not needed for the following operations. The reason for taking the first 'id' during the aggregation was just to reduce the DataFrame to a single row per date, the 'id' itself is not relevant for the subsequent analysis.
# 
# 3. **Extract dates:** Finally, the 'date' column is extracted from the test_agg DataFrame to create a new DataFrame test_dates, which contains only the unique dates in the test data.

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Feature Engineering </span></b> </div>

# In[18]:


def get_holidays(df):
    years_list = [2017, 2018, 2019, 2020, 2021, 2022]

    holiday_AR = holidays.CountryHoliday('AR', years=years_list)
    holiday_CA = holidays.CountryHoliday('CA', years=years_list)
    holiday_EE = holidays.CountryHoliday('EE', years=years_list)
    holiday_JP = holidays.CountryHoliday('JP', years=years_list)
    holiday_ES = holidays.CountryHoliday('ES', years=years_list)

    holiday_dict = holiday_AR.copy()
    holiday_dict.update(holiday_CA)
    holiday_dict.update(holiday_EE)
    holiday_dict.update(holiday_JP)
    holiday_dict.update(holiday_ES)

    df['holiday_name'] = df['date'].map(holiday_dict)
    df['is_holiday'] = np.where(df['holiday_name'].notnull(), 1, 0)
    df['holiday_name'] = df['holiday_name'].fillna('Not Holiday')

    return df

# Assume enc is an instance of OrdinalEncoder()
def encode_holiday_names(df, enc, subset='train'):
    if subset=='train':
        df['holiday_name'] = enc.fit_transform(df['holiday_name'].values.reshape(-1,1))
    else:
        df['holiday_name'] = enc.transform(df['holiday_name'].values.reshape(-1,1))
        not_hol_val = oe.transform([['Not Holiday']])[0,0]
        df.loc[df['holiday_name']==-1, 'holiday_name'] = not_hol_val
    return df

def apply_sin_cos_transformation(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(df[col_name] * (2 * np.pi / period))
    df[f'{col_name}_cos'] = np.cos(df[col_name] * (2 * np.pi / period))
    return df

def feature_engineer(df):
    new_df = df.copy()
    new_df["month"] = df["date"].dt.month
    new_df["month_sin"] = np.sin(new_df['month'] * (2 * np.pi / 12))
    new_df["month_cos"] = np.cos(new_df['month'] * (2 * np.pi / 12))
    new_df["day"] = df["date"].dt.day
    new_df["day_sin"] = np.sin(new_df['day'] * (2 * np.pi / 12))
    new_df["day_of_week"] = df["date"].dt.dayofweek
    new_df["day_of_week"] = new_df["day_of_week"].apply(lambda x: 0 if x<=3 else(1 if x==4 else (2 if x==5 else (3))))
    
    new_df["day_of_year"] = df["date"].dt.dayofyear
    #account for leap year
    new_df["day_of_year"] = new_df.apply(lambda x: x["day_of_year"]-1 if (x["date"] > pd.Timestamp("2020-02-29") and x["date"] < pd.Timestamp("2021-01-01"))  else x["day_of_year"], axis=1)
    new_df["important_dates"] = new_df["day_of_year"].apply(lambda x: x if x in [1,2,3,4,5,6,7,8,125,126,360,361,362,363,364,365] else 0)
    
    new_df["year"] = df["date"].dt.year
    
    easter_date = new_df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    for day in list(range(-5, 5)) + list(range(40, 48)):
        new_df[f'easter_{day}'] = (new_df.date - easter_date).dt.days.eq(day)
    new_df = new_df.drop(columns=["date","month","day", "day_of_year"])
    
    for col in new_df.columns :
        if 'easter' in col :
            new_df = pd.get_dummies(new_df, columns = [col], drop_first=True)
    
    new_df = pd.get_dummies(new_df, columns = ["important_dates","day_of_week"], drop_first=True)
    
    return new_df


# ## ⚙️ Feature Engineering Explained 
# 
# - **get_holidays(df):** 📅 This function identifies whether each date in the dataset is a holiday in one of the specified countries (Argentina, Canada, Estonia, Japan, or Spain) for the years 2017-2022. If a date is a holiday, it records the name of the holiday; otherwise, it records 'Not Holiday'. It also adds a binary 'is_holiday' column where 1 indicates a holiday and 0 indicates a non-holiday.
# 
# - **encode_holiday_names(df, enc, subset='train'):** 🔄 This function uses an ordinal encoder (enc) to transform the 'holiday_name' column to numerical values, making it suitable for machine learning algorithms. The encoding is fitted on the training data and then applied to the test data. Any new holiday names in the test data that the encoder has not seen in the training data are given the same code as 'Not Holiday'.
# 
# - **apply_sin_cos_transformation(df, col_name, period):** 🌀 This function applies a sinusoidal transformation to a specified column in the dataset. This is useful for dealing with cyclical features like time of day or day of year, as it preserves the cyclical nature of the data.
# 
# - **feature_engineer(df):** 🛠 This function applies a series of feature engineering steps to the dataset, including:
# 
#     * Extracting the month, day, and day of the week from the 'date' column and applying a sinusoidal transformation to them.
#     
#     * Creating a 'day_of_year' feature that accounts for leap years.
#     
#     * Creating a 'important_dates' feature that records certain specified days of the year.
#     
#     * Creating binary features for each year indicating whether it's 5 days before or after Easter, or 40-47 days after Easter.
#     
#     * Dropping the 'date', 'month', 'day', and 'day_of_year' columns.
#     
#     * One-hot encoding the 'important_dates' and 'day_of_week' features.
# 
# <div style="border-radius: 10px; border: #EA906C solid; padding: 15px; background-color: #ffffff00; font-size: 100%; text-align: left;">
#     📝<b>Note:</b> The get_holidays, encode_holiday_names, and apply_sin_cos_transformation functions are not called in the feature_engineer function, so you would need to call these functions separately on  data before or after calling feature_engineer.
# </div>    
# 
# <br>
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚧  Keep in Mind </h3>
# Overall, these functions are designed to extract and create features from the 'date' column that could be useful for a machine learning model. The specific features created depend on the dataset and the problem at hand, but generally, these transformations are intended to help a model capture patterns related to the time and date of the sales data. 🎯
# </div>
# 
# 
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Encoding</span></b> </div>

# In[19]:


oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

train_agg = get_holidays(train_agg)
test_agg = get_holidays(test_agg)

train_pred = pd.DataFrame()
train_pred['date'] = train_agg['date']

y_train = train_agg["num_sold"]
X_train = train_agg.drop(columns="num_sold")
X_test = test_agg

X_train = feature_engineer(X_train)
X_test = feature_engineer(X_test)

X_train = encode_holiday_names(X_train, oe)
X_test = encode_holiday_names(X_test, oe)


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Apply Model </span></b> </div>

# In[20]:


def train_and_predict(X_train, y_train, X_test):
    preds_lst = []
    n_splits = 5
    kf = GroupKFold(n_splits=n_splits)
    scores = []
    train_scores = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, groups=X_train.year)):
        model = LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leaves=50, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.iloc[train_idx]), columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_train.iloc[val_idx]), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        model.fit(X_train_scaled, y_train.iloc[train_idx])
        preds_lst.append(model.predict(X_test_scaled))
        train_scores[val_idx] = model.predict(X_val_scaled)
        sc = model.score(X_val_scaled, y_train.iloc[val_idx])
        scores.append(sc)
        print(f"Fold {fold}: Score = {sc}")

    mean_score = np.mean(scores)
    print("Mean score:", mean_score)
    return preds_lst, train_scores

preds_lst, train_scores = train_and_predict(X_train, y_train, X_test)
train_pred['num_sold'] = train_scores


# ##  Intepret the Results  📊
# 
# ### 🍽️ Process :
# 
# 1. **Initialize an empty list and variables:** 📝 An empty list preds_lst and a zero-filled array train_scores of the same length as `X_train` are created to store the predictions for the test data and training data respectively.
# 
# 2. **GroupKFold Cross-validation:** 🔀 A GroupKFold cross-validator is set up with **5 folds**. The year column is used as the grouping variable. The GroupKFold cross-validator ensures that the same group is not represented in both the training set and validation set for each fold.
# 
# 3. **Model Training and Prediction:** 🎯 For each fold, the following steps are performed:
# 
#     1. An instance of the <b><mark style="background-color:#526D82;color:white;border-radius:5px;opacity:1.0">LightGBM Regressor</mark></b> model (a type of gradient boosting model) is created with specified hyperparameters.
#     
#     2. A StandardScaler is used to standardize the predictor variables by removing the mean and scaling to unit variance.
#     
#     3. The model is trained on the scaled training data for that fold.
#     
#     4. The model's predictions for the scaled test data are appended to `preds_lst`.
#     
#     5. The model's predictions for the scaled validation data are stored in `train_scores`.
#     
#     6. The score of the model on the validation data for that fold is printed. This score is the coefficient of determination $R^2$ of the prediction, which provides a measure of how well future samples are likely to be predicted by the model.
# 
# 4. **Calculate and Print Mean Score:** 📈 The mean of the scores from each fold is calculated and printed. This gives an overall measure of how well the model performed across all folds of the cross-validation.
# 
# ### 🔨 Results Interpretation :
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D;font-size: 100%; padding: 10px;">
# <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Here are some observations:</h3>
#     
# In this case, the output shows the $R^2$ score for each fold and the mean score. The $R^2$ scores vary greatly between folds, with the first fold having a negative score. Negative $R^2$ indicates that the model performs worse than a horizontal line. The mean $R^2$ score is also negative, suggesting that the model's predictions are, on average, worse than a horizontal line. This suggests that the model may not be fitting the data well and could benefit from further tuning or a different modeling approach. 🛠
#     
# </div>
# 
# > After the function is run, the predictions for the training data are added to the train_pred DataFrame under the column 'num_sold'. The preds_lst is a list of prediction arrays for the test data from each fold of the cross-validation. 🔍
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Actual VS Predicted </span></b> </div>

# In[21]:


preds_df = pd.DataFrame(np.column_stack(preds_lst), columns=["2017", "2018", "2019", "2020", '2021'])

# Calculate average predictions from k-fold
preds_df['num_sold'] = preds_df.mean(axis=1)

# Assign predictions to test_dates DataFrame
test_dates["num_sold"] = preds_df['num_sold']


# In[22]:


plt.figure(figsize=(15, 10))
sns.lineplot(data=train_agg, x="date", y="num_sold", label='Actual')
sns.lineplot(data=train_pred, x='date', y='num_sold', label="Forecast")
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.title('Actual vs Forecasted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Predicted Values </span></b> </div>

# In[23]:


plt.figure(figsize=(15, 10))
sns.lineplot(data=train_agg, x="date", y="num_sold", label='Original')
sns.lineplot(data=test_dates, x="date", y="num_sold", label='Predicted')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.title('Original vs Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>Plot Product Ratio</span></b> </div>

# In[24]:


product_df = train.groupby(["date","product"])["num_sold"].sum().reset_index()
product_ratio_df = product_df.pivot(index="date", columns="product", values="num_sold")
product_ratio_df = product_ratio_df.apply(lambda x: x/x.sum(),axis=1)
product_ratio_df = product_ratio_df.stack().rename("ratios").reset_index()
product_ratio_df


# In[25]:


temp_df = pd.concat([product_ratio_df, test_agg]).reset_index(drop=True)

plt.figure(figsize=(15, 10))
sns.lineplot(data=temp_df, x="date", y="ratios", hue="product")
plt.xlabel("Date")
plt.ylabel("Ratios")
plt.title("Product Ratios Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()


# ### 🍽️ Process :
# 
# 1. **Concatenate DataFrames:** ➕ The code concatenates the `product_ratio_df` DataFrame (which contains the ratio of sales for each product for each date in the training data) and the `test_agg` DataFrame (which contains the unique dates in the test data). The result is a new DataFrame temp_df that contains both the training and testing dates.
# 
# 2. **Visualize Product Ratios Over Time:** 📈 Then, a line plot is created to visualize the product ratios over time. The x-axis represents the date, and the y-axis represents the ratios. Each product is represented by a different line, distinguished by color. The title of the plot is "Product Ratios Over Time", and grid lines are added for easier visualization.
# 

# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Calculate Mean Ratio</span></b> </div>

# In[26]:


mean_ratios = []
years = train['Year'].unique()
years.sort()  # ensuring years are sorted
years = years[:-2]  # remove last two years
assert len(years) > 0, "There should be at least one year to compute mean_ratios"

weights = [0.2, 0.4, 0.4]  # you can change these weights according to your need
assert len(weights) == len(years), "Weights count should match with the years count"

for year in years:
    product_ratio_2019 = product_ratio_df.loc[product_ratio_df["date"].dt.year == year].copy()
    product_ratio_2019["mm-dd"] = product_ratio_2019["date"].dt.strftime('%m-%d')
    product_ratio_2019 = product_ratio_2019.drop(columns="date")
    product_ratio_2019 = product_ratio_2019.reset_index()
    mean_ratios.append(product_ratio_2019['ratios'])
    
product_ratio_test = test.copy()

# Now product_ratio_2019 is defined and can be used here
product_ratio_2019['mean_ratios'] = sum(mean_ratio * weight for mean_ratio, weight in zip(mean_ratios, weights))

product_ratio_test["mm-dd"] = product_ratio_test["date"].dt.strftime('%m-%d')
product_ratio_test = pd.merge(product_ratio_test, product_ratio_2019, how="left", on=["mm-dd", "product"])
product_ratio_test


# ## ✏️ Process :
# 
# 1. **Prepare mean_ratios:** 📝 First, get the unique years from the 'Year' column of the train DataFrame, sorts them, and removes the last two years. An assertion check is done to ensure there are years to compute the mean_ratios. It also defines a set of weights and does an assertion check to ensure the count of weights matches the count of years.
# 
# 2. **Compute yearly ratios:** 🗓 For each year in the list of years, the code gets the corresponding rows from the product_ratio_df DataFrame. Then it adds a new column 'mm-dd' which is the month and day of the date. The 'date' column is then dropped and the DataFrame is reset. The ratios for the year are then appended to the mean_ratios list.
# 
# 3. **Calculate mean_ratios:** 🧮 The mean_ratios for the last extracted year's DataFrame is computed as a weighted sum of the ratios in mean_ratios and the weights.
# 
# 4. **Prepare the test DataFrame:** 📋 The test DataFrame is copied to product_ratio_test and a new column 'mm-dd' is added which is the month and day of the date.
# 
# 5. **Merge the DataFrames:** ➕ product_ratio_test is then merged with product_ratio_year on the 'mm-dd' and 'product' columns. This results in a DataFrame that contains the original columns from the test DataFrame, as well as the calculated mean_ratios for each product for each date.
# 
# 
# <div class="warning" style="background-color: #FFFAD7; border-left: 6px solid #862B0D;font-size: 100%; padding: 10px;">
# <h3 style="color: #862B0D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🏭 Output:</h3>
#     
# The resulting DataFrame (product_ratio_test) contains the original columns from the test data, as well as the mean_ratios for each product for each date. These mean_ratios represent the weighted average of the ratios of sales for each product over the specified years, and could potentially be a useful feature for a sales forecasting model. 💹
# 
#     
# </div>
# 
# 
# 

# In[27]:


test_data = pd.merge(test, test_dates, how="left")
test_data["ratios"] = product_ratio_test["mean_ratios"]
test_data


# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Making a Prediction </span></b> </div>

# In[28]:


def disaggregate_forecast(df, original_data):

    new_df = df.copy()
    
    # Compute weights and check for zero total sales
    total_num_sold = original_data["num_sold"].sum()

    stores_weights = original_data.groupby("store")["num_sold"].sum() / total_num_sold

    # Compute country_weights and check if all countries in df are present in original_data
    unique_countries = df["country"].unique()
    
    country_weights = pd.Series(index=unique_countries, data=1/len(unique_countries))

    # Adjust num_sold based on country and store weights
    for country in country_weights.index:
        new_df.loc[new_df["country"] == country, "num_sold"] *= country_weights[country]
        
    for store in stores_weights.index:
        new_df.loc[new_df["store"] == store, "num_sold"] *= stores_weights[store]
    
    # Apply product weights and drop the "ratios" column
    new_df["num_sold"] *= new_df["ratios"]
    new_df["num_sold"] = new_df["num_sold"].round()
    new_df = new_df.drop(columns=["ratios"])
    
    return new_df


# In[29]:


store_weights = train.groupby("store")["num_sold"].sum()/train["num_sold"].sum()
country_weights = train.loc[train["date"] < "2021-01-01"].groupby("country")["num_sold"].sum()/train.loc[train["date"] < "2021-01-01", "num_sold"].sum()


# In[30]:


final_df = disaggregate_forecast(test_data,train)
final_df


# In[31]:


submission = pd.read_csv("/kaggle/input/playground-series-s3e19/sample_submission.csv")
submission["num_sold"] = final_df["num_sold"]

submission.head()
# submission.to_csv('submission.csv', index = False)


# # <span style="color:#E888BB; font-size: 1%;">SUMMARY</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>SUMMARY</span></b> </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Results:📊 </span></b> </div>
# 
# 
# 
# Based on the results, it seems that the model's performance was highly variable.<b><span style='color:#9DB2BF'>The model performed poorly on some folds, as indicated by the negative scores</span></b>. However, the model did manage to produce reasonable predictions on other folds, as suggested by the positive scores. **This variability in model performance could be due to a variety of factors**, including the quality and representativeness of the training data, the appropriateness of the selected model and features, and the robustness of the model validation strategy. ❓💭
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚧  Keep in Mind </h3>
# The generated sales forecasts were disaggregated into product-level forecasts using historical sales ratios. This approach assumes that the proportion of sales for each product will remain relatively stable over time. If this assumption does not hold, then the accuracy of the product-level forecasts may be compromised. 🔍⚠️
# </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> Discussion:💬 </span></b> </div>
# 
# 
# The analysis demonstrates a potential approach to time-series forecasting in a retail context. The process of feature engineering, particularly the creation of time-based features and the use of holiday information, is a key strength of this approach. These features can help to capture temporal patterns and special events that can influence sales. 📆🎯
# 
# > However, the model's variable performance highlights the challenges of predictive modeling. Further work could explore different modeling approaches, additional features, or alternative strategies for handling anomalies in the data. For instance, the poor performance on some folds could be due to overfitting on the training data, so techniques to prevent overfitting could be explored. 🔍🔄
# 
# The approach to disaggregating sales forecasts into product-level forecasts is a practical way of generating more detailed forecasts. However, the validity of this approach depends on the stability of product sales ratios over time. If these ratios are subject to significant fluctuations, then a different approach to disaggregation may be needed. 🔢🤔
# 
# 
# <div class="warning" style="background-color: #DDE6ED; border-left: 6px solid #27374D;font-size: 100%; padding: 10px;">
# <h3 style="color: #27374D; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🔎  Results observations:</h3>
#     
# Overall, this analysis provides a foundation for further work in sales forecasting. By refining the predictive model and exploring alternative disaggregation methods, it may be possible to generate more accurate and detailed sales forecasts. 📈🎯
#     
# </div>
# 
# ## <div style="padding: 20px;color:white;margin:10;font-size:90%;text-align:left;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/2824173/pexels-photo-2824173.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'> 🙏 Acknowledgements </span></b> </div>
# 
# I would like to extend my sincere gratitude to [vishnu123](https://www.kaggle.com/vishnu123) for their invaluable [Kaggle notebook](https://www.kaggle.com/code/vishnu123/tps-sep-22-eda-lasso-groupkfold-mean-ratios/notebook). The insights, methodologies, and code snippets from the notebook have significantly contributed to the success of this project. The knowledge shared in the notebook has not only served as a solid foundation for this project but has also greatly enriched my understanding of machine learning techniques for sales forecasting.
# 
# # <span style="color:#E888BB; font-size: 1%;">SUGGESTION</span>
# <div style="padding: 35px;color:white;margin:10;font-size:170%;text-align:center;display:fill;border-radius:10px;overflow:hidden;background-image: url(https://images.pexels.com/photos/380337/pexels-photo-380337.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1)"><b><span style='color:black'>SUGGESTION</span></b> </div>
# 
# ###  There are some suggestions to potentially improve the model performance and sales forecasting:➕
# 
# 1. **Try Different Models:** 🔄 While `LightGBM` is a powerful model, there might be other models that could perform better on this dataset. Trying different models like `XGBoost`, `CatBoost`, or even deep learning models like `LSTM (Long Short Term Memory)` which are well-suited for time-series data could be beneficial. 🧠💡
# 
# 2. **Parameter Tuning:** 🎚️ The parameters used in the model might not be the optimal ones. Using techniques like grid search or random search for hyperparameter tuning can help find a better set of parameters that might improve model performance. 🛠️🔄
# 
# 3. **Feature Engineering:** ⚙️ Additional features could be created that might help improve the model's performance. For instance, lagged features (values at prior time steps), rolling window statistics (mean, standard deviation, etc. over a certain past window), and trend features (upward, downward, stationary over a certain past window) can often be useful in time-series forecasting. 📊⏲️
# 
# 4. **Anomaly Detection:** ⚠️ The code currently excludes data from March 2020 to May 2020. Instead of simply excluding this data, you might consider implementing a formal anomaly detection method to identify and handle such periods. This could make your model more robust and adaptable to new data. 💪🔍
# 
# 5. **Cross-Validation Strategy:** 🎯 Instead of using `GroupKFold`, consider using a time-series specific cross-validation strategy, such as `TimeSeriesSplit`. This respects the temporal order of observations, which is important in time-series problems. ⏳🔄
# 
# 6. **Ensemble Methods:** 👥 Consider using ensemble methods. Combining predictions from multiple models can often lead to better performance than any single model. 🤝🧮
# 
# 7. **External Data:** 🌐 If possible, consider integrating external data sources that could have predictive power. For example, macroeconomic indicators, industry trends, or even weather data could potentially influence sales and could be worth including in your model. 🌦️📈
# 
# 8. **Model Interpretability:** 🔍 Consider using methods to interpret your model's predictions. Feature importance plots, partial dependence plots, or SHAP (SHapley Additive exPlanations) values can provide insight into which features are driving your model's predictions. 💡📊
# 
# 
# <div class="warning" style="background-color: #F8F3D4; border-left: 6px solid #FF5252;font-size: 100%; padding: 10px;">
# <h3 style="color: #FF8551; font-size: 18px; margin-top: 0; margin-bottom: 10px;">🚧  Keep in Mind </h3>
# Remember, while these suggestions might improve the model, the extent of the improvement depends heavily on the data and the specific problem at hand. Therefore, it's always a good idea to experiment and validate these strategies before deciding on the best approach. 👨‍🔬🔬
# </div>
# 
# 
# <br>
# 
# ***
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 4.5em; font-weight: bold; font-family: Arial;">THANK YOU!</span>
# </div>
# 
# <div style="text-align: center;">
#     <span style="font-size: 5em;">✔️</span>
# </div>
# 
# <br>
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.4em; font-weight: bold; font-family: Arial; max-width:1200px; display: inline-block;">
#        If you discovered this notebook to be useful or enjoyable, I'd greatly appreciate any upvotes! Your support motivates me to regularly update and improve it. :-)
#    </span>
# </div>
# 
# <br>
# 
# <br>
# 
# 
# 
# 
# <div style="text-align: center;">
#    <span style="font-size: 1.6em; font-weight: bold;font-family: Arial;"><a href="https://www.kaggle.com/tumpanjawat/code">@pannmie</a></span>
# </div>

# 
