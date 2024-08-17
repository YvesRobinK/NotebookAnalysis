#!/usr/bin/env python
# coding: utf-8

# @author: André Daniël VOLSCHENK  
# 
# Kaggle project {Google Analytics Customer Revenue Prediction}  
# kaggle.com/andredanielvolschenk  
# 
# # Preface
# The intention of this notebook is to continue from Part 1 by visualizing and performing Exploratory Data Analysis on the data. Features will be engineered and explored.
# 
# # Contents
# * [Setup](#Setup)
# * [Visualization functions](#Visualization-functions)  
# Feature collections:
# * [1. channelGrouping](#1.-channelGrouping)
# * [2. customDimensions](#2.-customDimensions)
# * [3. date](#3.-date)
# * [4. device](#4.-device)
# * [5. fullVisitorId](#5.-fullVisitorId)
# * [6. geoNetwork](#6.-geoNetwork)
# * [7. hits](#7.-hits)
# * [8. totals](#8.-totals)
# * [9. trafficSource](#9.-trafficSource)
# * [10. visitId](#10.-visitId)
# * [11. visitNumber](#11.-visitNumber)
# * [12. visitStartTime](#12.-visitStartTime)
# 
# 
# # Setup
# 
# ## Problem Statement
# Please see Part 1 : https://www.kaggle.com/andredanielvolschenk/gstore-part-1-data-cleansing  
# 
# ## Import libraries
# Lets import libraries and see what datafiles we have in our environment.

# In[ ]:


import numpy as np
import pandas as pd
import statistics as ss

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot') # if error, use plt.style.use('ggplot') instead

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

import datetime as dt

import plotly as ply
import plotly.offline as plyo
plyo.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import seaborn as sns


import json
import pandas.io.json as pdjson
import ast

import gc
gc.enable()

import os
print(os.listdir("../input"))


# ## Load data as per Part 1
# In Part 1 (https://www.kaggle.com/andredanielvolschenk/gstore-part-1-data-cleansing-with-v2-data) we listed the columns we want to keep.  
# Here we shall only import the training set for now and save only selected columns to `data1`.  
# First we save a variable with the columns declared at the end of Part 1, and a variable with JSON columns too:

# In[ ]:


json_vars = ['device', 'geoNetwork', 'totals', 'trafficSource', 'hits', 'customDimensions']

final_vars = ['channelGrouping','customDimensions_index','customDimensions_value','date',
'device_browser','device_deviceCategory','device_isMobile','device_operatingSystem',
'fullVisitorId','geoNetwork_city','geoNetwork_continent','geoNetwork_country',
'geoNetwork_metro','geoNetwork_networkDomain','geoNetwork_region','geoNetwork_subContinent',
'hits_appInfo.exitScreenName','hits_appInfo.landingScreenName','hits_appInfo.screenDepth',
'hits_appInfo.screenName','hits_contentGroup.contentGroup1','hits_contentGroup.contentGroup2',
'hits_contentGroup.contentGroup3','hits_contentGroup.contentGroup4','hits_contentGroup.contentGroup5',
'hits_contentGroup.contentGroupUniqueViews1','hits_contentGroup.contentGroupUniqueViews2',
'hits_contentGroup.contentGroupUniqueViews3','hits_contentGroup.previousContentGroup1',
'hits_contentGroup.previousContentGroup2','hits_contentGroup.previousContentGroup3',
'hits_contentGroup.previousContentGroup4','hits_contentGroup.previousContentGroup5',
'hits_customDimensions','hits_customMetrics','hits_customVariables','hits_dataSource',
'hits_eCommerceAction.action_type','hits_eCommerceAction.option','hits_eCommerceAction.step',
'hits_eventInfo.eventAction','hits_eventInfo.eventCategory','hits_eventInfo.eventLabel',
'hits_exceptionInfo.isFatal','hits_experiment','hits_hitNumber','hits_hour','hits_isEntrance',
'hits_isExit','hits_isInteraction','hits_item.currencyCode','hits_item.transactionId',
'hits_latencyTracking.domContentLoadedTime','hits_latencyTracking.domInteractiveTime',
'hits_latencyTracking.domLatencyMetricsSample','hits_latencyTracking.domainLookupTime',
'hits_latencyTracking.pageDownloadTime','hits_latencyTracking.pageLoadSample',
'hits_latencyTracking.pageLoadTime','hits_latencyTracking.redirectionTime',
'hits_latencyTracking.serverConnectionTime','hits_latencyTracking.serverResponseTime',
'hits_latencyTracking.speedMetricsSample','hits_minute','hits_page.hostname','hits_page.pagePath',
'hits_page.pagePathLevel1','hits_page.pagePathLevel2','hits_page.pagePathLevel3',
'hits_page.pagePathLevel4','hits_page.pageTitle','hits_page.searchCategory','hits_page.searchKeyword',
'hits_promotionActionInfo.promoIsClick','hits_promotionActionInfo.promoIsView','hits_publisher_infos',
'hits_referer','hits_social.hasSocialSourceReferral','hits_social.socialInteractionNetworkAction',
'hits_social.socialNetwork','hits_time','hits_transaction.affiliation','hits_transaction.currencyCode',
'hits_transaction.localTransactionRevenue','hits_transaction.localTransactionShipping',
'hits_transaction.localTransactionTax','hits_transaction.transactionId',
'hits_transaction.transactionRevenue','hits_transaction.transactionShipping',
'hits_transaction.transactionTax','hits_type','totals_bounces','totals_hits','totals_newVisits',
'totals_pageviews','totals_sessionQualityDim','totals_timeOnSite','totals_totalTransactionRevenue',
'totals_transactionRevenue','totals_transactions','trafficSource_adContent',
'trafficSource_adwordsClickInfo.adNetworkType','trafficSource_adwordsClickInfo.gclId',
'trafficSource_adwordsClickInfo.isVideoAd','trafficSource_adwordsClickInfo.page',
'trafficSource_adwordsClickInfo.slot','trafficSource_campaign','trafficSource_isTrueDirect',
'trafficSource_keyword','trafficSource_medium','trafficSource_referralPath','trafficSource_source',
'visitId','visitNumber','visitStartTime']

print('created json_var and final_var')


# Because our memory is limited, we import 500'000 rows from `train_v2` to do this notebook. This should be representative of the entire data set which has 1'708'337 rows.

# In[ ]:


# lets append json_vars with final_vars, because we still need to import the json vars before expanding them
all_vars  = json_vars + final_vars # the master list of columns to import

def load_df(csv_path, usecols=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    
    dfs = pd.read_csv(csv_path, sep=',',
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                    chunksize = 100000,   # 100000
                     nrows=500000  # TODO: take out
                     )
    
    for df in dfs:
        df.reset_index(drop = True,inplace = True)
        
        device_list=df['device'].tolist()
        #deleting unwanted columns before normalizing
        for device in device_list:
            del device['browserVersion'],device['browserSize'],device['flashVersion'],device['mobileInputSelector'],device['operatingSystemVersion'],device['screenResolution'],device['screenColors']
        df['device']=pd.Series(device_list)
        
        geoNetwork_list=df['geoNetwork'].tolist()
        for network in geoNetwork_list:
            del network['latitude'],network['longitude'],network['networkLocation'],network['cityId']
        df['geoNetwork']=pd.Series(geoNetwork_list)
        
        df['hits']=df['hits'].apply(ast.literal_eval)
        df['hits']=df['hits'].str[0]
        df['hits']=df['hits'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
        df['customDimensions']=df['customDimensions'].apply(ast.literal_eval)
        df['customDimensions']=df['customDimensions'].str[0]
        df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
        JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

        for column in JSON_COLUMNS:
            column_as_df = pdjson.json_normalize(df[column])
            column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        
        
        print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        
        # we wont see each and every column in each chunk that we load, so we need to find where our master list intersects with the actual data
        usecols = set(usecols).intersection(df.columns)
        usecols = list(usecols)
        use_df = df[usecols]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis = 0).reset_index(drop = True)
        print('Stored shape:', ans.shape)
        
    return ans



data1 = load_df('../input/train_v2.csv', usecols=final_vars)
#data2 = load_df("../input/test_v2.csv", usecols=final_vars)

print('data1 shape: ', data1.shape)
#print('data2 shape: ', data2.shape)

print("data1 loaded")
#print("data2 loaded")


# ## Target variable preparation
# Two variables are given: `totals_transactionRevenue` and `totals_totalTransactionRevenue`.  
# 
# Note that the description for `totals_transactionRevenue` is given as:  
# "This field is deprecated. Use "totals.totalTransactionRevenue" instead (see above)."  
# (see https://support.google.com/analytics/answer/3437719?hl=en)  
# However ! The problem statement for the v2 competition requires us to predict based on  `totals_transactionRevenue`. So we will drop `totals_totalTransactionRevenue`.
# 
# Before we start, it is worth noting that our target variabe (`totals_transactionRevenue`) is the actual purchase amount in $ multiplied by 1 million.  
# Lets divide this by 1x10^6 in order to get the actual purchase amount again.  
# Before any of this, we must replace NaNs with 0, because a 'no purchase' is encoded as NaN.

# In[ ]:


data1['totals_transactionRevenue'].fillna(0, inplace=True)
data1['totals_transactionRevenue'] = data1['totals_transactionRevenue'].astype('float')/1000000

data1[['totals_transactionRevenue']].head()


# # Visualization functions
# Lets look at features to see whether they have information about the target  
# We will go through each one, and even create some new ones.
# 
# Specifically there are 4 plots per feature that may give us some insight:
# * 1. a count of how many times an instance is encounterred in the dataset
# * 2. a count of how many purchases were made for that instance
# * 3. the mean revenue of the purchases for that instance
# * 4. the total revenue for that instance
# 
# Since we are doing comparisons to `totals_transactionRevenue`, we want to look at the training data only.  
# 
# Declare a `bar_plots` function to handle qualitative variables:

# In[ ]:


def bar_plots(cstr, data, n_bars=None):
    
    
    if n_bars:
        counts=data[cstr].value_counts()
        counts = counts.iloc[:n_bars]
        data = data[data[cstr].isin(counts.index)]
    #
    
    
    df = data.copy()
    df[cstr] = df[cstr].astype('str')
    
    fig = plt.figure(figsize=(14,10))   # define (new) plot area
    ax = fig.gca()   # define axis
    plt.suptitle(str('Plots for '+cstr))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # FIGURE 1
    ax = fig.add_subplot(221)
    ax.set_xlabel('Count of instances')   # Set text for the x axis
    counts = df[cstr].value_counts() # This is important to get the full nr of instances
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 2
    ax = fig.add_subplot(222)
    ax.set_xlabel('Count of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2=df[cstr].where(df.totals_transactionRevenue>0).value_counts()
    counts.update(counts2)
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 3
    ax = fig.add_subplot(223)
    ax.set_xlabel('Mean of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2 = df[df.totals_transactionRevenue > 0].groupby(cstr)['totals_transactionRevenue'].agg(['mean'])
    idx=counts2.index
    counts2=pd.Series(counts2.values.reshape(-1,))
    counts2.index = idx
    counts.update(counts2)
    counts = counts.astype('int64')
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
    
    
    # FIGURE 4
    ax = fig.add_subplot(224)
    ax.set_xlabel('Sum of purchases')   # Set text for the x axis
    counts[:] = 0  # set all to 0
    counts2 = df.groupby(cstr)['totals_transactionRevenue'].agg(['sum'])
    idx=counts2.index
    counts2=pd.Series(counts2.values.reshape(-1,))
    counts2.index = idx
    counts.update(counts2)
    counts = counts.astype('int64')
    counts.plot.barh(ax = ax)   # Use the plot.bar method on the counts data frame
    
    for i, v in enumerate( counts ):  # give each 3 decimal points
        d=''
        if v>1000000000:
            v=v/1000000000
            d='B'
        elif v>1000000:
            v=v/1000000
            d='M'
        elif v>1000:
            v=v/1000
            d='k'
        ax.text(v + 0.5, i + .25, str(round(v,3))+d, color='black', fontweight='bold')
#

print("'bar_plots' function declared")


# Declare  a `scatter_plots` function to handle datetime variables:

# In[ ]:


def scatter_plots(train_df, test_df):
    
    
    def scatter_plot(cnt_srs, color):
        trace = go.Scatter(
            x=cnt_srs.index[::-1],
            y=cnt_srs.values[::-1],
            showlegend=False,
            marker=dict(
                color=color,
            ),
        )
        return trace
    
    train_df['date'] = train_df['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    cnt_srs = train_df.groupby('date')['totals_transactionRevenue'].agg(['size', 'count', 'mean', 'sum'])
    cnt_srs.columns = ["count", "count of non-zero revenue", "mean", "sum"]
    cnt_srs = cnt_srs.sort_index()
    #cnt_srs.index = cnt_srs.index.astype('str')
    trace1 = scatter_plot(cnt_srs["count"], 'blue')
    trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')
    trace3 = scatter_plot(cnt_srs["mean"], 'blue')
    trace4 = scatter_plot(cnt_srs["sum"], 'blue')
    
    fig = ply.tools.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                              subplot_titles=["Date - Count", "Date - Non-zero Revenue count",
                                              "Date - Mean of purchases", "Date - Sum of purchases"])
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    fig.append_trace(trace3, 3, 1)
    fig.append_trace(trace4, 4, 1)
    fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
    plyo.iplot(fig, filename='date-plots')
    
    
    # test set
    test_df['date'] = test_df['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
    cnt_srs = test_df.groupby('date')['fullVisitorId'].size()
    
    
    trace = scatter_plot(cnt_srs, 'red')
    
    layout = go.Layout(
        height=400,
        width=800,
        paper_bgcolor='rgb(233,233,233)',
        title='Dates in Test set'
    )
    
    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    plyo.iplot(fig, filename="ActivationDate")
    
    
#
print("'scatter_plots' function declared")


# Declare a `kde_scatter_plots` function to handle quantitative variables:

# In[ ]:


def kde_scatter_plots(cstr, train):
    
    # Figure 1
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax1 = sns.kdeplot(train[cstr].astype('float64'), color="#006633", shade=True)
    
    # Figure 2
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax2 = sns.kdeplot(train[np.isnan(train['totals_transactionRevenue'])][cstr].astype('float64'),
                      label='No revenue', color="#0000ff")
    ax2 = sns.kdeplot(train[train['totals_transactionRevenue'] >0][cstr].astype('float64'),
                      label='Has revenue', color="#ff6600")
    
    # Figure 3
    temp = train.groupby(cstr, as_index=False)['totals_transactionRevenue'].mean()
    
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax3 = sns.scatterplot(x=temp[cstr], y=temp.totals_transactionRevenue, label='Mean of purchases', color="#ff6600")
    
    # Figure 3
    temp = train.groupby(cstr, as_index=False)['totals_transactionRevenue'].sum()
    
    plt.figure(figsize=(15,5))
    plt.title(str(cstr)+" distribution")
    ax4 = sns.scatterplot(x=temp[cstr], y=temp.totals_transactionRevenue, label='Sum of purchases', color="#ff6600")
    
#
print("'kde_scatter_plots' function declared")


# Declare a `globe_plot` function for nationality-based geographical variables:

# In[ ]:


def globe_plot(train):
    
    def globe(tmp, title, var2=None):
        
        if var2 != None:
            locations = tmp.geoNetwork_country
            z = tmp.totals_transactionRevenue
            text = tmp.totals_transactionRevenue
        else:
            locations = tmp.index
            z = tmp.values
            text = tmp.values
        
        # plotly globe credits - https://www.kaggle.com/arthurtok/generation-unemployed-interactive-plotly-visuals
        colorscale = [[0, 'rgb(102,194,165)'], [0.005, 'rgb(102,194,165)'], 
                      [0.01, 'rgb(171,221,164)'], [0.02, 'rgb(230,245,152)'], 
                      [0.04, 'rgb(255,255,191)'], [0.05, 'rgb(254,224,139)'], 
                      [0.10, 'rgb(253,174,97)'], [0.25, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
        
        data = [ dict(
                type = 'choropleth',
                autocolorscale = False,
                colorscale = colorscale,
                showscale = True,
                locations = locations,
                z = z,
                locationmode = 'country names',
                text = text,
                marker = dict(
                    line = dict(color = '#fff', width = 2)) )           ]
        
        layout = dict(
            height=500,
            title = title,
            geo = dict(
                showframe = True,
                showocean = True,
                oceancolor = '#222',
                projection = dict(
                type = 'orthographic',
                    rotation = dict(
                            lon = 60,
                            lat = 10),
                ),
                lonaxis =  dict(
                        showgrid = False,
                        gridcolor = 'rgb(102, 102, 102)'
                    ),
                lataxis = dict(
                        showgrid = False,
                        gridcolor = 'rgb(102, 102, 102)'
                        )
                    ),
                )
        fig = dict(data=data, layout=layout)
        plyo.iplot(fig)
    #
    
    
    
    tmp = train["geoNetwork_country"].value_counts()
    title = 'Visits by Country'
    globe(tmp, title, var2=None)
    
    tmp = train[train.totals_transactionRevenue > 0]["geoNetwork_country"].value_counts()
    title = 'Number of purchases by Country'
    globe(tmp, title, var2=None)
    
    tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "mean"}).reset_index()
    var2='totals_transactionRevenue'
    title = 'Mean Revenue by Countries'
    globe(tmp, title, var2)
    
    tmp = train.groupby("geoNetwork_country").agg({"totals_transactionRevenue" : "sum"}).reset_index()
    var2='totals_transactionRevenue'
    title = 'Sum Revenue by Countries'
    globe(tmp, title, var2)
    
#
print("'globe_plot' function defined")


# ## Review column names
# Lets look at the list of column names one last time

# In[ ]:


data1 = data1.reindex_axis(sorted(data1.columns), axis=1)
print('Number of columns:', len(data1.columns))
for col in data1.columns:
    print("'"+col+"',")


# We will discuss the variables in alphabetical order as shown above.
# 
# # 1. channelGrouping
# STRING 	The Default Channel Group associated with an end user's session for this View.  
# Lets start with the first variable: `channelGrouping`.  
# `channelGrouping` is the channel through which the user came to the store.  
# We will print out the unique values first:

# In[ ]:


print(data1.channelGrouping.unique())


# First we describe the unique channels:  
# * 'Organic search' is a method for entering one or several search terms as a single string of text into a search engine.  
# * 'Referral' traffic is Google's method of reporting visits that came to your site from sources outside of its search engine. When someone clicks on a hyperlink to go to a new page on a different website, Analytics tracks the click as a referral visit to the second site.  
# * 'Paid search' marketing means you advertise within the sponsored listings of a search engine or a partner site.  
# * 'Affiliate' marketing is a marketing arrangement by which an online retailer pays commission to an external website for traffic or sales generated from its referrals.  
# * 'Direct' traffic is most often the result of a user entering a URL into their browser or using a bookmark to directly access the site. Essentially, Direct sessions occur any time Google Analytics cannot determine another referring source or channel.  
# * Display marketing channel can be made up of any number of traffic sources as long as the medium of the traffic sources is 'display', 'cpm' or 'banner' and 'Ad Distribution Network' matches 'content'.  
# * 'Social' marketing channel can be made up of any number of traffic sources as long as the medium of the traffic sources is 'social', 'social media', 'social-media', 'social network', or 'social-network'.  
# * Other: anything else.  
#  
# 
# Now lets Visualize:

# In[ ]:


bar_plots('channelGrouping', data1)


# After reviewing the Figures above, I would advize the marketing team in this way:
# 
# `Organic Search`
# * Has the highest proportion of user visits, but the 2nd highest proportion of buyers.  
# * The mean of the purchases made is not particularly high, but Organic search still makes 3rd most of our revenue due to sheer number.  
# * I hypothesize that when a user arrives by Organic Search, they may leave to view other options on the search engine.  
# * Users arriving through Organic Search are clearly important. Perhaps we can encourage more buying if we promote the product as *the* choice option. Clearly state why it is the 'recommeded' product. This might result in higher buying ratio like that for 'Referral'.  
# 
# `Social`
# * Has a high portion of the traffic, but a tiny fraction of the purchases.  
# * Mean purchase amount is low, so is the portion of our revenue.  
# * It would be easy to write 'Social' off as worthless. However, this channel presents a unique opportunity! Why is it that our views are not translated into purchases?  
# * I reccomend the marketing team takes advantage of this huge traffic stream and focus on converting these views to purchases! This will require some in-depth research.
# 
# `Direct`
# * 3rd highest traffic share, 3rd highest purchase share.  
# * The mean purchase amount is high, and this generates the 2nd most of our revenue.  
# * This is an important channel! I think some users save the URL and come back later to buy. How they see the URL before saving is most likely after an Organic Search, Social, or Referral, respectively. But we canot know for sure. The only thing we can do is as I recommended for 'Organic Search'. Promote the product as *the* choice.
# 
# `Referral`
# * Has a fairly sizeable portion of the traffic, but a disproportionally large portion of purchases!  
# * Mean purchase amount is not particularly high, but this is where most of our revenue comes from currently.  
# * When a user is on some starting website and that site refers them to some product, then the user probably feels like the authors of the starting website has done the research and narrowing down of options on their behalf, and that they can trust their reccomendation. This explains the high purchase ratio.  
# * Referrals are arguably the most important for the company right now. Focus on promoting the products to authors of websites!!!
# 
# `Display`
# * Has a small traffic share, and a small number of purchases.  
# * The mean purchase amount is, however, very high!!! Only due to the small number of visitors is the share of our revenue low.  
# * Here is another excellent opportunity for the marketing team! The Display channel certainly encourages users to spend highly. Perhaps the alure of the visual banner is enticing enough to convince users to spend highly?  
# * I reccomend increasing the number of these advertisements! Specifically target demohraphics that are likely to spend big! We will discover who these demograhics are later...
# 
# `Paid Search`, `Affiliates`, and `Other`
# * These have low traffic share, and low number of our purchases.  
# * Their mean purchase and share of our revenue are likewise not very high.  
# * Im not sure that these are worhtwhile to expand on. Perhaps the 'Paid Search' can be said to have decent mean purchase amount. Maybe this is the only channel in this bunch worth considering seriously.  
# 
# 
# 
# # 2. customDimensions
# RECORD 	This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
# 
# ## 2.1. customDimensions_index
#  INTEGER 	The index of the custom dimension.

# In[ ]:


print('Number of unique values (incl. nan):', data1.customDimensions_index.nunique(dropna=False))


# There are only 2 unique values, so this variable can be encoded as a boolean.  
# For now lets explore this variable graphically:

# In[ ]:


bar_plots('customDimensions_index', data1)


# This feature seems to  have powerful predictive value!  
# Consider the disparity between the 'nan' category's count and purchase count.
# 
# ## 2.2. customDimensions_value
# STRING 	The value of the custom dimension.  
# Lets see how many unique values there are, and visualize:

# In[ ]:


print('Number of unique values (incl. nan):', data1.customDimensions_value.nunique(dropna=False))

bar_plots('customDimensions_value', data1)


# This feature also seems to have predictive power!  
# It seems that when the custom dimensions are set to 'North America', then sales are high relative to other settings.  
# We could group 'North America' as a category against the category of all others.
# 
# # 3. date
# STRING 	The date of the session in YYYYMMDD format.
# We also neeed to look at the `date` feature.  
# In order to compare the dates from the train and test sets, we need to load the test set's `date` column:  

# In[ ]:


data2 = load_df("../input/test_v2.csv", usecols=['date', 'fullVisitorId'])

scatter_plots(data1, data2)


# What do we learn from this?
# * the testing data `data2` does not overlap in time with `data1`.  
# * This means we need to ensure that our testing and training sets throughout this notebook follows the rule:  
# * train set is followed by non-overlapping test set.
# * Lets keep this in mind for when we split our data later !!!!  
# 
# This feature is not useful in this format. Let us extract some useful features like:  
# * month: The month (1 to 12)
# * week: The week (1 to 52)
# * weekday: Return the day of the week as an integer, where Monday is 0 and Sunday is 6  
# 
# We can now drop `date` from further analysis.

# In[ ]:


data1['date'] = data1['date'].apply(lambda x: dt.date(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:])))

#% feature representation
data1.date = pd.to_datetime(data1.date, errors='coerce')

#% feature extraction - time and date features

# Get the month value from date
data1['month'] = data1['date'].dt.month
# Get the week value from date
data1['week'] = data1['date'].dt.week
# Get the weekday value from date
data1['weekday'] = data1['date'].dt.weekday

data1 = data1.drop(labels=['date'], axis=1)
data1.head()


# Now lets look at the extracted features graphically:

# In[ ]:


kde_scatter_plots('month', data1)
kde_scatter_plots('week', data1)
kde_scatter_plots('weekday', data1)


# # 4. device
# RECORD 	This section contains information about the user devices.
# ## 4.1. device_browser
#  STRING 	The browser used (e.g., "Chrome" or "Firefox").  
#  Lets look at the data. First lets see how many unique browsers there were:

# In[ ]:


print('Number of unique values (incl. nans) is:', data1.device_browser.nunique(dropna=False), 'out of', data1.shape[0])


# That is a lot of different browsers!  
# Far more than the common Chrome, Firefox, Opera, Safari, Internet Explorer, and Microsoft Edge that we are used to!  
# Lets visualize the top 10.  
# I hypothesize that almost all of the traffic will be through these:

# In[ ]:


bar_plots('device_browser', data1, n_bars=10)


# The hypothesis was correct. The first 3 or 4 have nearly all the traffic.  
# * by far most of our traffic is from Chrome, and correspondingly, so is our number of purchasers.
# * Strangely, there is a *significantly* higher mean purchase for Firefox users over any other users. Other browsers are quite even in this category.
# * Unsurprisingly, most of our revenue is from Chrome users. This is mainly due to their sheer number.
# * If Firefox users visited the GStore more often, I hypothesize that they would become a large part of our revenue.
# * I reccomend GStore seriously focus on supporting Firefox users to encourage purchases by those users.  
# We can categorize the lesser used browsers as one group.  
# 
# ## 4.2. device_deviceCategory
# STRING 	The type of device (Mobile, Tablet, Desktop).

# In[ ]:


bar_plots('device_deviceCategory', data1)


# This visual is straightforward:
# * most users access the site with their desktop, follwed by mobile, and then tablet.
# * the count of purchases for these categories correspond accordingly, however, desktop has an even higher share on purchases that of pure counts.
# * the mean purchase amount is highest for desktop, at almost double that of tablet and mobile.
# * these factors account for the fact that the revenue generated is simply dominated by the desktop users.
# * marketing can keep focussing on desktop users.
# 
# ## 4.3. device_isMobile
# This field is deprecated. Use device.deviceCategory instead. 	
# BOOLEAN 	If the user is on a mobile device, this value is true, otherwise false.  
# This is simply a boolean logic of the previous feature. We do not expect much new information:

# In[ ]:


bar_plots('device_isMobile', data1)


# Similar to the previous feature, we can say that Mobile users are the minority, and that they spend less on average than 'others'.  
# We can remove this feature from our data.  
# 
# ## 4.4. device_operatingSystem
#  STRING 	The operating system of the device (e.g., "Macintosh" or "Windows").

# In[ ]:


del(data1['device_isMobile']) # remove

print('Number of unique values (incl. nans) is:', data1.device_operatingSystem.nunique(dropna=False))
bar_plots('device_operatingSystem', data1)


# It is not a surprise that the visits are dominated by Windows, Macintosh, Android, iOS, and then Linux. These are the larger operating systems.  
# `Windows` users are the most frequent, and therefore it makes sense that they have the 2nd most purchases. They also have the 2nd highest mean purchase amount, and make 2nd most of our revenue. These users have potential - if marketing can convince a higher proportion of these users to purchase, then their contribution will almost certainly rival that of the Mac users.  
# `Macintosh` users are 2nd most frequent visitors, but dominate in purchases. They have the 3rd highest mean purchase amounts, and therefore are the primary source of our income. This is an important demographic! If marketing can manage to increase their mean purchase even slightly, this will result in massive earnings!  
# `Android` users are the 3rd highest count of visitors, but are unlikely to purchase. Their mean purchase is 4th highest, but due to low number of purchases they make very little of our revenue. Marketing needs to find a way to encourage a higher fraction of purchases to go through. Andoid users need to be supported, as they could otherwise have made a significant share of our income.  
# `iOS`: Similar to Android, they have a fairly large share of visitors, but the number of purchases are relatively small. Their mean purchase amount is only 6th highest, which all leads to a tiny share on our revenue. This is a difficult situation, because we need to encourage more purchases, but also higher mean spending amount. Perhaps special products could be designed for these users, but I doubt that the investment to do that will translate to significantly higher revenue generated.  
# `Linux` users are a small number, but they make 4th most of our sales. They also have 4th highest mean purchase amounts. Due to the small number of visitors, their revenue share is very small. Perhaps more Linux users need to be encouraged to visit the store.  
# `Chrome OS` is an interesting one! The number of users are small, but they are 3rd highest in number of sales! Their mean purchase amount is also the highest overall! They are, inceredibly, 3rd highest revenue generators. It is only due to their small number that they do not have a significantly higher revenue share. Chrome is part of Google, so it should not be too surprising that they are relevent in Google Store (GStore). Here, marketing should focus on promoting Chrome OS as a product. The technical team should maintiain excellent support for these users. If the number of users grow due to great technical support, then it would translate to a significant boost in income for GStore.  
# **All others**: Other OSs have an almost insignificant share of visitors, and therefore of sales. They do not seem to be particularly promising for investing in.  
# We can categorize these lesser used OSs into one category.  
# 
# # 5. fullVisitorId
# STRING 	The unique visitor ID (also known as client ID).  
# There is not much to say about `fullVisitorId` at this point.  
# We know that this is unique for each user. We can look at how many unique users we have:

# In[ ]:


print("Number of unique visitors in train set : ",data1.fullVisitorId.nunique(), " out of rows : ",data1.shape[0])
print("Number of unique visitors in test set : ",data2.fullVisitorId.nunique(), " out of rows : ",data2.shape[0])
print("Number of common visitors in train and test set : ",len(set(data1.fullVisitorId.unique()).intersection(set(data2.fullVisitorId.unique())) ))

del(data2)


# Evidently, most of the visits to the GStore are different users. There are some who return a few times.  
# 
# # 6. geoNetwork
# RECORD 	This section contains information about the geography of the user.
# 
# # 6.1. geoNetwork_city
# STRING 	Users' city, derived from their IP addresses or Geographical IDs.  
# There are a lot of cities on this planet. Lets find out how many there represented at GStore:

# In[ ]:


print('Number of unique values (incl. nans) is:', data1.geoNetwork_city.nunique(dropna=False), 'out of', data1.shape[0])


# That is too many cities to visualize on a bar chart. So let only look at the top... say... 25:

# In[ ]:


bar_plots('geoNetwork_city', data1, n_bars=25)


# These figures are not particularly useful, and it feels like a bit of an information overload.  
# What we can infer is that the vast majority visits do not have the city location available. Altough mean purchases vary significantly, I doubt that this is indicative of some underlying truth, since the number of purchases for most cities are tiny, and as such the mean purchase amount os probably not a good indicator.  
# It even difficult to comment with confidence on the fact that `not available in dataset` and `(not set)` seem to have different mean purchases. The `(not set)` category has vew sales to take the mean from. It is unclear whether these two categories should be combined, or whether they represent a significantly different type of user.   
# 
# Lets aggregate cities in terms of how many times they appear. We will also agrgate cities based on `totals_hits` and `totals_pageviews`.  
# We print these new columns.  
# 
# We shall create a `make_countsum` function to aggregate features with high cardinality in terms of its frequency (count) as well as the sums of `totals_hits` and `totals_pageviews`.  Features with high cardinality are those with a large number of unique categories. High cardinality is a serious problem for Data Science problems.

# In[ ]:


def make_countsum(df, dfstr):
    df['totals_hits']=df['totals_hits'].fillna(0).astype('int')
    df['totals_pageviews']=df['totals_pageviews'].fillna(0).astype('int')
    
    df[str(dfstr+'_count')] = df[dfstr]
    df[str(dfstr+'_count')]=df.groupby(dfstr).transform('count')
    
    df[str(dfstr+'_hitssum')] = df.groupby(dfstr)['totals_hits'].transform('sum')
    df[str(dfstr+'_viewssum')] = df.groupby(dfstr)['totals_pageviews'].transform('sum')
    #del(df[dfstr])
    return df

print('make_countsum function created')

data1 = make_countsum(data1, 'geoNetwork_city')

print( 'created geoNetwork_city_count,geoNetwork_city_hitssum,and geoNetwork_city_viewssum features')


# Now lets view these new features grahically:

# In[ ]:


bar_plots('geoNetwork_city_count', data1, n_bars=10)
bar_plots('geoNetwork_city_hitssum', data1, n_bars=10)
bar_plots('geoNetwork_city_viewssum', data1, n_bars=10)


# 
# 
# # 6.2. geoNetwork_metro
# STRING 	The Designated Market Area (DMA) from which sessions originate.  
# Perhaps if we zoom out slightly we can make better inferences? We will look at the top 20 metros:

# In[ ]:


bar_plots('geoNetwork_metro', data1, n_bars=20)


# Unfortunately most of the metros are unavailable. We do however see a significant portion of San Francisco-Oakland-San Jose users. They, together with New York, seem to have a disproportionately large number of sales, and therefore revenue share.  
# I wuold recommend that GStore support these users in particular and perhaps encourage these users to promote GStore to their peers.   
# We will be using the `make_countsum` function for this feature too.
# 
# # 6.3. geoNetwork_region
# STRING 	The region from which sessions originate, derived from IP addresses. In the U.S., a region is a state, such as New York.  
# Lets zoom out even more, to look at regions of the world.

# In[ ]:


bar_plots('geoNetwork_region', data1, 20)


# Similar to cities and metros, we have the issue that most regions are 'not available in dataset'.  
# Perhaps one inference we *can* make is that California, New York, Texas, Washington, and Illnois have a disproportionately high share of purchases. These regions clearly make most of our revenue (aside from the 'not available in dataset category').
# Take note that, again, there seems to be a difference between the 'not available in dataset', and the '(not set)' categories, however we cant be sure yet.  
# We will be using the `make_countsum` function for this feature too.
# 
# # 6.4. geoNetwork_continent
# STRING 	The continent from which sessions originated, based on IP address.  
# There are just too many cities, metros, and regions. The low number of sales in each of these make our comparisons weak with uncertainty.  
# Lets zoom out... way out... to look at the continents!

# In[ ]:


bar_plots('geoNetwork_continent', data1)


# Hmmm... not what i expected.  
# Lets break down what our inferences are:
# * The majority of our visitors are from the Americas, Europe, and Asia.
# * The Americas totally dominate in number of purchases. In fact the number of purchases from other continents are surprisingly small!
# * Aside from the Americas, the estimate for mean purchase amount is probably not very reliable.
# * Africa has an enormous mean purchase amount. Keep in mind however, that there were only 8 sales in Africa. A single high spender may have caused the mean purchase amout to inflate to this extent.
# * Since the Americas are the vast majority of our sales numbers, they also contribute to nearly the entirety of our revenue.  
# Marketting should focus on converting European and Asian visits into purchases. This is currently a lost opportunity for revenue generation. Finally, the americas should be supported technically, and targetted by consistent marketing to keep the numbers up.   
# Looks like 'Americas' can be one category and all others can be grouped into another.
# 
# # 6.5. geoNetwork_subContinent
#  STRING 	The sub-continent from which sessions originated, based on IP address of the visitor.  
#  Perhaps if we zoom in slightly, we can get a better idea of what is going on in some of the more relevant continents:

# In[ ]:


bar_plots('geoNetwork_subContinent', data1)


# Somehow it is not very surprising that it is Northern America that is driving the Americas figures so highly. In fact, the Carribbean, Southern- and Central America seem underrepresented here.  
# We have a number of subcontinents making a respectable share of visitors, but again, their sales numbers are completely overshadowed by Northern America.  
# Some exceptionally high mean purchases in Africa are still likely due to the aforementioned outliers. This can occur due to the tiny number of sales in these regions.  
# The reccomendations are similar to before: subcontinents other than northern america need to be encouraged to purchase. Their respectable visitor numbers are not being utilized!  
# It looks like we can again make a binary variable, where all categories except 'Northern America' can be grouped together.
# 
# # 6.6. geoNetwork_country
# STRING 	The country from which sessions originated, based on IP address.  
# Perhaps subContinent is still too vague. Countries within the same continent can be *vastly* different! Lets look at countries.  
# First see how many countries are represented:

# In[ ]:


print('Number of unique values (incl. nans) is:', data1.geoNetwork_country.nunique(dropna=False), 'out of', data1.shape[0])


# This is far too many to represent on a bar chart.  
# Lets look at only the 20 most frequent countries...

# In[ ]:


bar_plots('geoNetwork_country', data1, n_bars=20)


# Well, this really matches what we saw previously. The United States Seem to be dominating visiting representation, as well as number of purchases. However, it is still noticeable that other nations seem to visit the store, but not make purchases.  
# 
# Really what all this has boiled down to is that marketing need to target countries with significant visits, who are not currently, for some reason, making purchases. India, the UK, Canada, *etc* have potential to become a significant source of revenue.  
# 
# Lets take a look at all countries now to see which to target for converting visits into purchases. We will also look at countries with large populations that have low visits for some reason. The best way to quickly look at this type of information is by looking at some globes:

# In[ ]:


globe_plot(data1)


# Now we can quickly make inferences about any country in the world!  
# 
# Lets look in particular at the 10 nations in the world with the largest populations: China, India, USA, Indonesia, Brazil, Pakistan, Nigeria, Bangladesh, Russia, Mexico.  
# Of these nations, the USA clearly dominates in number of visitors. India features too, but at a number far lower than it should, given its population. The other large nations barely feature at all. This remains an untapped source of visitors that marketing needs to target!  
# In order to diminish cardinality, we will be using the `make_countsum` function for this feature.
# 
# # 6.7. geoNetwork_networkDomain
# STRING 	The domain name of user's ISP, derived from the domain name registered to the ISP's IP address.  
# The domain name of user's ISP, derived from the domain name registered to the ISP's IP address.  
# There may be very many of these.. lets see how many:

# In[ ]:


print('Number of unique values (incl. nans) is:', data1.geoNetwork_networkDomain.nunique(dropna=False), 'out of', data1.shape[0])


# That is a lot of network domains...  
# Lets only look at the top 20 for now:

# In[ ]:


bar_plots('geoNetwork_networkDomain', data1, n_bars=20)


# Our data capturing records the majority of visits as having unset '(not set)' network domains, or 'unknown.unknown'.  
# These may initially sound like they should be the same thing, however a look at the number of purchases plot shows that users with 'unknown.unknown' are far less likely to purchase than users with '(not set)'.  
# This is a good example of why Visualization and EDA is important. We may have otherwise grouped these as one category, however they seem to be different, and may thus lead to predictive insight in our model.  
# 
# One interesting observation is the mean purchase amount of the 'comcastbusiness.net' domain. Users enterring the GStore through this domain are purchasing expensive products. The number of purchases for this domain are 294, so it is unlikely that the mean are unreasonably boosted by a few outliers.  
# Lets take a quick look to be sure:

# In[ ]:


fig = plt.figure(figsize=(8,6))   # define (new) plot area
ax = fig.gca()   # define axis
plt.suptitle('Boxplot for comcastbusiness.net')
temp = data1[data1.geoNetwork_networkDomain=='comcastbusiness.net'][['totals_transactionRevenue']]
print(temp.describe())
temp.boxplot(ax=ax)
del(temp)


# Surprisingly, it seems that the mean may well be inflated due to some large outliers.  
# Comparing the mean and median indicates that this is the case.  
# Regardless, the 'comcastbusiness.net' domain is significant, as it is the second highest source of revenue, despite having a tiny visitor share!  
# We will be using the `make_countsum` function for this feature too.
# 
# # 7. hits
# RECORD 	This row and nested fields are populated for any and all types of hits.
# ## 7.1. hits_appInfo
# RECORD 	This section will be populated for each hit with type = "APPVIEW" or "EXCEPTION".
# 
# ### 7.1.1. hits_appInfo.exitScreenName
# STRING 	The exit screen of the session.

# In[ ]:


print('Number of unique values (incl. nans) is:', data1['hits_appInfo.exitScreenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.exitScreenName', data1, n_bars=10)


# We will be using the `make_countsum` function for this feature too.
# 
# ### 7.1.2. hits_appInfo.landingScreenName
# STRING 	The landing screen of the session.

# In[ ]:


print('Number of unique values (incl. nans) is:', data1['hits_appInfo.landingScreenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.landingScreenName', data1, n_bars=10)


# We will be using the `make_countsum` function for this feature too.
# 
# ### 7.1.3. hits_appInfo.screenDepth
# STRING 	The number of screenviews per session reported as a string. Can be useful for historgrams.

# In[ ]:


print('Number of unique values (incl. nans) is:', data1['hits_appInfo.screenDepth'].nunique(dropna=False), 'out of', data1.shape[0])
data1['hits_appInfo.screenDepth']=data1['hits_appInfo.screenDepth'].astype('str')
bar_plots('hits_appInfo.screenDepth', data1, n_bars=10)


# This feature does not seem useful at all. Lets drop it !  
# 
# ### 7.1.4. hits_appInfo.screenName
# STRING  The name of the string.

# In[ ]:


del(data1['hits_appInfo.screenDepth'])

print('Number of unique values (incl. nans) is:', data1['hits_appInfo.screenName'].nunique(dropna=False), 'out of', data1.shape[0])
bar_plots('hits_appInfo.screenName', data1, n_bars=10)


# We will be using the `make_countsum` function for this feature too.
# 
# ## 7.2. hits_contentGroup
# RECORD 	This section contains information about content grouping.
# 
# ### 7.2.1. hits_contentGroup.contentGroup1
# 
# hits.contentGroup.contentGroupX   :    STRING 	The content group on a property. A content group is a collection of content that provides a logical structure that can be determined by tracking-code or page-title/URL regex match, or predefined rules. (Index X can range from 1 to 5.)

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup1'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup1', data1)


# This feature seems weak, and will be dropped!  
# 
# ### 7.2.2. hits_contentGroup.contentGroup2

# In[ ]:


del(data1['hits_contentGroup.contentGroup1'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup2'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup2', data1)


# ### 7.2.3. hits_contentGroup.contentGroup3

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup3'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup3', data1)


# This faeture does not seem to have much redictive power, it will be dropped!
# 
# ### 7.2.4. hits_contentGroup.contentGroup4

# In[ ]:


del(data1['hits_contentGroup.contentGroup3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup4'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup4', data1)


# This feature seems weak in predictive power and will be dropped!
# 
# ### 7.2.5. hits_contentGroup.contentGroup5

# In[ ]:


del(data1['hits_contentGroup.contentGroup4'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroup5'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroup5', data1)


# This feature seems the same as `hits_contentGroup.contentGroup4` and will be dropped!
# 
# ### 7.2.6. hits_contentGroup.contentGroupUniqueViews1
# 
# hits.contentGroup.contentGroupUniqueViewsX 	:   STRING 	The number of unique content group views. Content group views in different sessions are counted as unique content group views. Both the pagePath and pageTitle are used to determine content group view uniqueness. (Index X can range from 1 to 5.)

# In[ ]:


del(data1['hits_contentGroup.contentGroup5'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews1'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews1', data1)


# This feature does not seem to have predictive power. It will be dropped.
# 
# ### 7.2.7. hits_contentGroup.contentGroupUniqueViews2

# In[ ]:


del(data1['hits_contentGroup.contentGroupUniqueViews1'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews2'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews2', data1)


# ### 7.2.8. hits_contentGroup.contentGroupUniqueViews3

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_contentGroup.contentGroupUniqueViews3'].nunique(dropna=False))
bar_plots('hits_contentGroup.contentGroupUniqueViews3', data1)


# This feature is not powerful. Lets drop it.
# 
# ### 7.2.9. hits_contentGroup.previousContentGroup1
# 
# hits.contentGroup.previousContentGroupX   :	STRING 	Content group that was visited before another content group. (Index X can range from 1 to 5.)

# In[ ]:


del(data1['hits_contentGroup.contentGroupUniqueViews3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup1'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup1', data1)


# This feature is not discriminative, so it will be dropped.
# 
# ### 7.2.10. hits_contentGroup.previousContentGroup2

# In[ ]:


del(data1['hits_contentGroup.previousContentGroup1'])


print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup2'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup2', data1)


# Similar to group1, group2 is not powerful, and so it will be dropped. 
# 
# ### 7.2.11. hits_contentGroup.previousContentGroup3

# In[ ]:


del(data1['hits_contentGroup.previousContentGroup2'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup3'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup3', data1)


# Group3 is also not helpful, and so it will be dropped.
# 
# ### 7.2.12. hits_contentGroup.previousContentGroup4

# In[ ]:


del(data1['hits_contentGroup.previousContentGroup3'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup4'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup4', data1)


# Group4 is also useless and will be dropped.
# 
# ### 7.2.13. hits_contentGroup.previousContentGroup5

# In[ ]:


del(data1['hits_contentGroup.previousContentGroup4'])

print('Number of unique values (incl. nan):', data1['hits_contentGroup.previousContentGroup5'].nunique(dropna=False))
bar_plots('hits_contentGroup.previousContentGroup5', data1)


# Group5 also does not offer anything new. These were all similar and pointless. This feature will be dropped.
# 
# ## 7.3. hits_customDimensions
# RECORD 	This section contains any hit-level custom dimensions. This is a repeated field and has an entry for each dimension that is set.

# In[ ]:


del(data1['hits_contentGroup.previousContentGroup5'])

print('Number of unique values (incl. nan):', data1['hits_customDimensions'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customDimensions', data1)


# Just like the features before, this is not usefull and will be dropped.
# 
# ## 7.4. hits_customMetrics
# RECORD 	This section contains any hit-level custom metrics. This is a repeated field and has an entry for each metric that is set.

# In[ ]:


del(data1['hits_customDimensions'])

print('Number of unique values (incl. nan):', data1['hits_customMetrics'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customMetrics', data1)


# Similar to all those before... this is not useful and will be dopped.
# 
# ## 7.5. hits_customVariables
# RECORD 	This section contains any hit-level custom variables. This is a repeated field and has an entry for each variable that is set.

# In[ ]:


del(data1['hits_customMetrics'])

print('Number of unique values (incl. nan):', data1['hits_customVariables'].astype('str').nunique(dropna=False)) # convert to str otherwise error
bar_plots('hits_customVariables', data1)


# It seems all the 'custom' fields were not usefull. This faeture shall be dropped.
# 
# ## 7.6. hits_dataSource
# STRING 	The data source of a hit. By default, hits sent from analytics.js are reported as "web" and hits sent from the mobile SDKs are reported as "app".

# In[ ]:


del(data1['hits_customVariables'])

print('Number of unique values (incl. nan):', data1['hits_dataSource'].nunique(dropna=False))
bar_plots('hits_dataSource', data1)


# ## 7.7. hits_eCommerceAction
# RECORD 	This section contains all of the ecommerce hits that occurred during the session. This is a repeated field and has an entry for each hit that was collected.
# 
# ### 7.7.1. hits_eCommerceAction.action_type
# STRING 	
# 
# The action type. Click through of product lists = 1, Product detail views = 2, Add product(s) to cart = 3, Remove product(s) from cart = 4, Check out = 5, Completed purchase = 6, Refund of purchase = 7, Checkout options = 8, Unknown = 0.  
# Usually this action type applies to all the products in a hit, with the following exception: when hits.product.isImpression = TRUE, the corresponding product is a product impression that is seen while the product action is taking place (i.e., a "product in list view").

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_eCommerceAction.action_type'].nunique(dropna=False))
bar_plots('hits_eCommerceAction.action_type', data1)


# This feature does not have sufficient observations in its categories to be usefull. Lets drop it.
# 
# ### 7.7.2. hits_eCommerceAction.step
# INTEGER 	This field is populated when a checkout step is specified with the hit.

# In[ ]:


del(data1['hits_eCommerceAction.action_type'])

print('Number of unique values (incl. nan):', data1['hits_eCommerceAction.step'].nunique(dropna=False))
bar_plots('hits_eCommerceAction.step', data1)


# Again, not enough observations in its categories to be useful. Lets drop it.
# 
# ## 7.8.  hits_eventInfo
# RECORD 	This section is populated for each hit with type = "EVENT".
# 
# ### 7.8.1. hits_eventInfo.eventAction
# STRING 	The event action.

# In[ ]:


del(data1['hits_eCommerceAction.step'])

print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventAction'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventAction', data1)


# This faeture also does not have enough observations in its categories. Lets drop it.
# 
# ### 7.8.2. hits_eventInfo.eventCategory
# STRING 	The event category.

# In[ ]:


del(data1['hits_eventInfo.eventAction'])

print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventCategory'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventCategory', data1)


# ### 7.8.3. hits_eventInfo.eventLabel
# STRING 	The event label.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_eventInfo.eventLabel'].nunique(dropna=False))
bar_plots('hits_eventInfo.eventLabel', data1, n_bars=20)


# This is another high cardinality feature for which we shall use `make_countsum`.
# 
# ## 7.9. hits_exceptionInfo.isFatal
# BOOLEAN     If the exception was fatal, this is set to true.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_exceptionInfo.isFatal'].nunique(dropna=False))
bar_plots('hits_exceptionInfo.isFatal', data1)


# Another feature with the same information as many before it, and it shall be removed.
# 
# ## 7.10. hits_experiment
# 
# RECORD   This row and the nested fields are populated for each hit that contains data for an experiment.

# In[ ]:


del(data1['hits_exceptionInfo.isFatal'])

print('Number of unique values (incl. nan):', data1['hits_experiment'].astype('str').nunique(dropna=False))   # convert to 'str' otherwise error
bar_plots('hits_experiment', data1)


# Similar to the previous feature, this feature shows the same useless info.
# 
# ## 7.11. hits_hitNumber
# INTEGER 	The sequenced hit number. For the first hit of each session, this is set to 1.

# In[ ]:


del(data1['hits_experiment'])

#print('Number of unique values (incl. nan):', data1['hits_hitNumber'].nunique(dropna=False))
kde_scatter_plots('hits_hitNumber', data1)


# ## 7.12. hits_hour
# INTEGER 	The hour in which the hit occurred (0 to 23).

# In[ ]:


print('Unique values (incl. nan):', data1['hits_hour'].unique())
print('Number of nans:', data1.hits_hour.isnull().sum() )
print('The mode is:', ss.mode(data1.hits_hour) )
print("Replace nans with mode")
data1.hits_hour.fillna(ss.mode(data1.hits_hour))
kde_scatter_plots('hits_hour', data1)


# ## 7.13. hits_isEntrance
# BOOLEAN 	If this hit was the first pageview or screenview hit of a session, this is set to true.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_isEntrance'].nunique(dropna=False))
bar_plots('hits_isEntrance', data1)


# ## 7.14. hits_isExit
# BOOLEAN 	If this hit was the last pageview or screenview hit of a session, this is set to true.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_isExit'].nunique(dropna=False))
bar_plots('hits_isExit', data1)


# ## 7.15. hits_isInteraction
# BOOLEAN 	If this hit was an interaction, this is set to true. If this was a non-interaction hit (i.e., an event with interaction set to false), this is false.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_isInteraction'].nunique(dropna=False))
bar_plots('hits_isInteraction', data1)


# This is yet another fearture like those in the past. It offers no power and will be dropped.
# 
# ## 7.16. hits_item.currencyCode
# hits_item: RECORD 	This section will be populated for each hit with type = "ITEM".
# currencyCode:   STRING 	The local currency code for the transaction.

# In[ ]:


del(data1['hits_isInteraction'])

print('Number of unique values (incl. nan):', data1['hits_item.currencyCode'].nunique(dropna=False))
bar_plots('hits_item.currencyCode', data1)


# ## 7.17. hits_minute
# INTEGER 	The minute in which the hit occurred (0 to 59).  
# Logically, the minute over all hours should not have any statistically significant influence on whether a purchase is made.  
# Lets investigate this nonetheless:

# In[ ]:


kde_scatter_plots('hits_minute', data1)


# As expected. there seems to be no discernable pattern or logical indicator. Lets drop this feature immediately.
# 
# ## 7.18. hits_page
# This section is populated for each hit with type = "PAGE".
# 
# ### 7.18.1. hits_page.hostname
# STRING 	The hostname of the URL.

# In[ ]:


del(data1['hits_minute'])  # drop immediately

print('Number of unique values (incl. nan):', data1['hits_page.hostname'].nunique(dropna=False))
bar_plots('hits_page.hostname', data1)


# ### 7.18.2. hits_page.pagePath
# STRING 	The URL path of the page.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pagePath'].nunique(dropna=False))
#bar_plots('hits_page.pagePath', data1)


# We will have to use the `make_countsum` function to change this high cardinality feature to numeric features.
# 
# ### 7.18.3. hits_page.pagePathLevel1
# STRING 	This dimension rolls up all the page paths in the 1st hierarchical level in pagePath.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel1'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel1', data1)


# We can use `make_countsum` to numerize this feature.
# 
# ### 7.18.4. hits_page.pagePathLevel2
# STRING 	This dimension rolls up all the page paths in the 2nd hierarchical level in pagePath.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel2'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel2', data1)


# We shall use `make_countsum` here too.
# 
# ### 7.18.5. hits_page.pagePathLevel3
# STRING 	This dimension rolls up all the page paths in the 3rd hierarchical level in pagePath.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel3'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel3', data1)


# We shall use `make_countsum` here too.
# 
# ### 7.18.5. hits_page.pagePathLevel4
# STRING 	This dimension rolls up all the page paths in the 4th hierarchical level in pagePath.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pagePathLevel4'].nunique(dropna=False))
#bar_plots('hits_page.pagePathLevel4', data1)


# We shall use `make_countsum` here too.
# 
# ### 7.18.6. hits_page.pageTitle
# STRING 	The page title.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.pageTitle'].nunique(dropna=False))
#bar_plots('hits_page.pageTitle', data1)


# Again, we will have to use `make_countsum`
# 
# ### 7.18.7. hits_page.searchCategory
# STRING 	If this was a search-results page, this is the category selected.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_page.searchCategory'].nunique(dropna=False))
bar_plots('hits_page.searchCategory', data1)


# This feature seems powerfless, and shall be deleted.
# 
# ### 7.18.8. hits_page.searchKeyword
# STRING 	If this was a search results page, this is the keyword entered.

# In[ ]:


del(data1['hits_page.searchCategory'])

print('Number of unique values (incl. nan):', data1['hits_page.searchKeyword'].nunique(dropna=False))
bar_plots('hits_page.searchKeyword', data1)


# Ths feature seems to have some low representation in all but 1 categories. Lets delete it.
# 
# ## 7.19. hits_promotionActionInfo
# RECORD 	This row and nested fields are populated for each hit that contains Enhanced Ecommerce PROMOTION action information.
# ### 7.19.1. hits_promotionActionInfo.promoIsClick
# BOOLEAN 	True if the Enhanced Ecommerce action is a promo click.

# In[ ]:


del(data1['hits_page.searchKeyword'])

#print('Number of unique values (incl. nan):', data1['hits_promotionActionInfo.promoIsClick'].nunique(dropna=False))
bar_plots('hits_promotionActionInfo.promoIsClick', data1)


# This feature seems low power. Lets delete it,
# 
# ### 7.19.3. hits_promotionActionInfo.promoIsView
# BOOLEAN 	True if the Enhanced Ecommerce action is a promo view.

# In[ ]:


del(data1['hits_promotionActionInfo.promoIsClick'])

#print('Number of unique values (incl. nan):', data1['hits_promotionActionInfo.promoIsView'].nunique(dropna=False))
bar_plots('hits_promotionActionInfo.promoIsView', data1)


# ## 7.20. hits_publisher_infos
# 

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_publisher_infos'].astype('str').nunique(dropna=False))
bar_plots('hits_publisher_infos', data1)


# This feature is similar to many before, it offers no predictive power. Lets delete it.
# 
# ## 7.21. hits_referer
# STRING 	The referring page, if the session has a goal completion or transaction. If this page is from the same domain, this is blank.

# In[ ]:


del(data1['hits_publisher_infos'])

print('Number of unique values (incl. nan):', data1['hits_referer'].nunique(dropna=False))
#bar_plots('hits_referer', data1, n_bars=15)


# We shall use `make_countsum` for this feature.
# 
# ## 7.22. hits_social
# RECORD 	This section is populated for each hit with type = "SOCIAL".  
# 
# ### 7.22.1. hits_social.hasSocialSourceReferral
# STRING 	A string, either Yes or No, that indicates whether sessions to the property are from a social source.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_social.hasSocialSourceReferral'].nunique(dropna=False))
data1['hits_social.hasSocialSourceReferral']=data1['hits_social.hasSocialSourceReferral'].astype('str')
bar_plots('hits_social.hasSocialSourceReferral', data1, n_bars=15)


# ### 7.22.2. hits_social.socialInteractionNetworkAction
# STRING 	For social interactions, this represents the social network being tracked.

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_social.socialInteractionNetworkAction'].nunique(dropna=False))
bar_plots('hits_social.socialInteractionNetworkAction', data1)


# Another repeat feature as before. We drop this feature immediately.
# 
# ### 7.22.3. hits_social.socialNetwork
# STRING 	The social network name. This is related to the referring social network for traffic sources; e.g., Google+, Blogger.

# In[ ]:


del(data1['hits_social.socialInteractionNetworkAction'])

print('Number of unique values (incl. nan):', data1['hits_social.socialNetwork'].nunique(dropna=False))
bar_plots('hits_social.socialNetwork', data1)


# For this feature I reccommend grouping all categories (except 'not set' and 'Youtube) into 1 category. We will then have 3 categories.
# 
# ## 7.23. hits_time
# INTEGER 	The number of milliseconds after the visitStartTime when this hit was registered. The first hit has a hits.time of 0 

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_time'].nunique(dropna=False))
bar_plots('hits_time', data1)


# This is again a repeat of what we had before so many times. We will delete this feature immediately.
# 
# ## 7.24. hits_transaction.currencyCode
# STRING 	The local currency code for the transaction.

# In[ ]:


del(data1['hits_time'])

print('Number of unique values (incl. nan):', data1['hits_transaction.currencyCode'].nunique(dropna=False))
bar_plots('hits_transaction.currencyCode', data1)


# ## 7.25. hits_type
# STRING 	The type of hit. One of: "PAGE", "TRANSACTION", "ITEM", "EVENT", "SOCIAL", "APPVIEW", "EXCEPTION".

# In[ ]:


print('Number of unique values (incl. nan):', data1['hits_type'].nunique(dropna=False))
bar_plots('hits_type', data1)


# # 8. totals
# RECORD 	This section contains aggregate values across the session.
# ## 8.1. totals_bounces
# INTEGER 	Total bounces (for convenience). For a bounced session, the value is 1, otherwise it is null.  
# Bounces are used to determine 'bounce rate':  
# Bounce rate is the percentage of single page visits (or web sessions). It is the percentage of visits in which a person leaves your website from the landing page without browsing any further. Google analytics calculates and report the bounce rate of a web page and bounce rate of a website.  So if a visit has a bounce of 1, then the user did not browse any further after they landed on the page. If the user *did* browse further, then `totals_bounces` is nan.  
# Lets find out how that affects sales:

# In[ ]:


print('Number of unique values (incl. nan):', data1['totals_bounces'].nunique(dropna=False))
bar_plots('totals_bounces', data1)


# This seems incredibly powerful!  
# From our data, we can infer that
# * bounced sessions never have sales!
# * approximately half of store visits are bounced.
# * i hypothesize that many users who land in the store and do not see imediately what they want will simply leave.
# * Users should ideally be directed in a helpful way from the start and convinced that they are in the right place with the right product where they are now!
# * This recommendation matches the recommendation given under `channelGrouping` 
# 
# ## 8.2. totals_hits
# INTEGER 	Total number of hits within the session.  
# `totals_hits` is the total number of hits within the session.  
# Google Analytics counts hits as “interactions”, which sends data to Analytics and is recorded as a user activity.
# 
# Some hit types can include:
# * Page (e.g. when a page is loaded on a website, or even inside a mobile application)
# * Event (e.g. when a user clicks play on a video)
# * Ecommerce (e.g. making a purchase online)
# * Social Interaction (e.g. clicking an embedded “Like” or “Retweet” button)

# In[ ]:


kde_scatter_plots('totals_hits', data1)


# The inferences and observations to be made here are similar to that of `visitNumber`.
# * Users who interact more with the website seem to also be more inclined to purchase higher.
# * Perhaps the site should have embedded (do not take user to another site/page/tab) videos or infographics to encourage purchases.
# 
# ## 8.3. totals_newVisits
# totals_newVisits: Total number of new users in session (for convenience). If this is the first visit, this value is 1, otherwise it is null. This is essentially a binned version of `visitNumber`

# In[ ]:


bar_plots('totals_newVisits', data1)


# Our inferences here are:
# * Most visitors are first time visitors
# * Most purchases are made by returning visitors
# * Returning visitors spend a higher amount on average
# * Returning visitors make us most of our money
# 
# ## 8.4. totals_pageviews
# Pageviews is the total number of pages viewed. Repeated views of a single page are counted. According to Google’s Google Analytics support site, a pageview is: 
# A pageview (or pageview hit, page tracking hit) is an instance of a page being loaded (or reloaded) in a browser. Pageviews is a metric defined as the total number of pages viewed. The metric itself says nothing about how many visitors saw that page or how many times the page was viewed per session. It’s just the total number of pageviews per page.

# In[ ]:


kde_scatter_plots('totals_pageviews', data1)


# The inferences are much of the same:
# * users generally see a given page only a few times per session
# * Purchases are therefore usually made for low pageviews
# * There seems to be some correlation between number of pageviews and average 
# 
# spending amount.
# * Most of our money comes, unsurprisingly, from low pageviews
# 
# ## 8.5. totals_sessionQualityDim
# INTEGER 	An estimate of how close a particular session was to transacting, ranging from 1 to 100, calculated for each session. A value closer to 1 indicates a low session quality, or far from transacting, while a value closer to 100 indicates a high session quality, or very close to transacting. A value of 0 indicates that Session Quality is not calculated for the selected time range.

# In[ ]:


kde_scatter_plots('totals_sessionQualityDim', data1)


# This feature seems to be powerful in predicting whether a user makes a purchase !!! There also seems to be a correlation between this variable and mean- and sum of purchase amount.
# 
# ## 8.6. totals_timeOnSite
# INTEGER 	Total time of the session expressed in seconds.

# In[ ]:


kde_scatter_plots('totals_timeOnSite', data1)


# ## 8.7. totals_transactionRevenue

# In[ ]:


data1['totals_transactionRevenue'].describe()


# Lets see the distribution of revenue graphically:

# In[ ]:


data1["totals_transactionRevenue"] = data1["totals_transactionRevenue"].astype('float')


gdf = data1.groupby("fullVisitorId")["totals_transactionRevenue"].sum().reset_index()    # sum each user's revenue
plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals_transactionRevenue"].values)))    # take log of revenue
plt.xlabel('index', fontsize=12)
plt.ylabel('totals_transactionRevenue', fontsize=12)
plt.show()

# NOTE
# Why are using natural log plus 1 (np.log1p) instead of natural log only (np.log) ?  
# Answer: np.log1p is used to deal with log(0). np.log1p(0) = np.log(0+1) = 0


# The Figure above shows that only a small number of customers make all GStore's revenue.  
# 
# ## 8.8. totals_totalTransactionRevenue

# In[ ]:


del(data1['totals_totalTransactionRevenue'])
print('totals_totalTransactionRevenue deleted')


# ## 8.9. totals_transactions
# INTEGER 	Total number of ecommerce transactions within the session.

# In[ ]:


#kde_scatter_plots('totals_transactions', data1)
bar_plots('totals_transactions', data1)


# Low representation in categories suggest reducing to 2 categories: nan vs all else.
# 
# # 9. trafficSource
# RECORD 	This section contains information about the Traffic Source from which the session originated.
# ## 9.1. trafficSource_adContent
# STRING 	The ad content of the traffic source. Can be set by the utm_content URL parameter. Lets look at the top 15 most frequent categories.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_adContent'].nunique(dropna=False))
bar_plots('trafficSource_adContent', data1, n_bars=15)


# For this feature we could use `make_countsum` to deal with high cardinality.
# 
# ## 9.2. trafficSource_adwordsClickInfo
# RECORD 	This section contains information about the Google Ads click info if there is any associated with this session. Analytics uses the last-click model.
# ### 9.2.1. trafficSource_adwordsClickInfo.adNetworkType
# STRING 	Network Type. Takes one of the following values: {“Google Search", "Content", "Search partners", "Ad Exchange", "Yahoo Japan Search", "Yahoo Japan AFS", “unknown”}

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.adNetworkType'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.adNetworkType', data1)


# ### 9.2.2. trafficSource_adwordsClickInfo.gclId
# STRING 	The Google Click ID.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.gclId'].nunique(dropna=False))
#bar_plots('trafficSource_adwordsClickInfo.adNetworkType', data1)


# We will use `make_countsum` on this feature.
# 
# ### 9.2.3. trafficSource_adwordsClickInfo.page
# INTEGER 	Page number in search results where the ad was shown.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.page'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.page', data1)


# This feature does not seem very useful. Low representation in most categories. Lets delete it!
# 
# ### 9.2.4. trafficSource_adwordsClickInfo.slot
# STRING 	Position of the Ad. Takes one of the following values:{“RHS", "Top"}

# In[ ]:


del(data1['trafficSource_adwordsClickInfo.page'])

print('Number of unique values (incl. nan):', data1['trafficSource_adwordsClickInfo.slot'].nunique(dropna=False))
bar_plots('trafficSource_adwordsClickInfo.slot', data1)


# ## 9.3. trafficSource_campaign
# STRING 	The campaign value. Usually set by the utm_campaign URL parameter.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_campaign'].nunique(dropna=False))
bar_plots('trafficSource_campaign', data1, n_bars=15)


# Low representation in all but 1 category makes this feature low power. Lets delete it.
# 
# ## 9.4. trafficSource_isTrueDirect
# BOOLEAN 	True if the source of the session was Direct (meaning the user typed the name of your website URL into the browser or came to your site via a bookmark), This field will also be true if 2 successive but distinct sessions have exactly the same campaign details. Otherwise NULL.

# In[ ]:


del(data1['trafficSource_campaign'])

print('Number of unique values (incl. nan):', data1['trafficSource_isTrueDirect'].nunique(dropna=False))
bar_plots('trafficSource_isTrueDirect', data1)


# ## 9.5. trafficSource_keyword
# STRING 	The keyword of the traffic source, usually set when the trafficSource.medium is "organic" or "cpc". Can be set by the utm_term URL parameter.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_keyword'].nunique(dropna=False))
bar_plots('trafficSource_keyword', data1, n_bars=20)


# Low frrequenct in all but 1 category. Lets delete this feature.
# 
# ## 9.6. trafficSource_medium
# STRING 	The medium of the traffic source. Could be "organic", "cpc", "referral", or the value of the utm_medium URL parameter.

# In[ ]:


del(data1['trafficSource_keyword'])

print('Number of unique values (incl. nan):', data1['trafficSource_medium'].nunique(dropna=False))
bar_plots('trafficSource_medium', data1)


# ## 9.7. trafficSource_referralPath
# STRING 	If trafficSource.medium is "referral", then this is set to the path of the referrer. (The host name of the referrer is in trafficSource.source.)

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_referralPath'].nunique(dropna=False))
bar_plots('trafficSource_referralPath', data1, n_bars=20)


# This feature could perhaps be reduced to 2 categories. With '/' being a category on its own.
# 
# ## 9.8. trafficSource_source
# STRING 	The source of the traffic source. Could be the name of the search engine, the referring hostname, or a value of the utm_source URL parameter.

# In[ ]:


print('Number of unique values (incl. nan):', data1['trafficSource_source'].nunique(dropna=False))
bar_plots('trafficSource_source', data1, n_bars=20)


# This feature seems to hold some predictive power. We can for example see that 'youtube.com' does not contribute to purchases relative to its frequency of occurence.  We could perhaps group categories that are less frequent than 'Youtube' as 1 category.
# 
# # 10. visitId
# INTEGER 	An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.

# In[ ]:


print('For data1:')
print("Number of unique 'fullVisitorId' entries", data1.fullVisitorId.nunique(dropna=False))
print("Number of unique 'visitId' entries", data1.visitId.nunique(dropna=False))
full_Vis = data1.fullVisitorId + data1.visitId.astype('str')
print('Number of unique combinations (out of ',data1.shape[0],' possible):',full_Vis.nunique(dropna=False))
del(full_Vis)


# Oddly enough, combining `fullVisitorId` and `visitId` we do NOT obtain a totally unique identifier.
# 
# # 11. visitNumber
# INTEGER 	The session number for this user. If this is the first session, then this is set to 1.

# In[ ]:


print('Number of unique values (incl. nan):', data1['visitNumber'].nunique(dropna=False))
kde_scatter_plots('visitNumber', data1)


# # 12. visitStartTime
# INTEGER 	The timestamp (expressed as POSIX time).  
# 
# This feature is useful when converted. We can extract the hour of the day from the timestamp. We shall call this feature `hour`. We can delete `visitStartTime` directly after extracting `hour`.

# In[ ]:


data1['visitStartTime'] = pd.to_datetime(data1['visitStartTime'], unit='s')
data1['hour'] = data1['visitStartTime'].dt.hour
kde_scatter_plots('hour', data1)

data1 = data1.drop(labels=['visitStartTime'], axis=1)


# It is interesting to see that this is not the same as the `hits_hour` feature.  
# 
# Lets see the final feature list to be used in Part 3:

# In[ ]:


print('Number of columns:', len(data1.columns))
for col in data1.columns:
    print("'"+col+"',")


# # Concluding remarks
# We have visualized and explored the features in our dataset.  
# 
# Acknowledgements  
# Special thank you to the following authors for their insightful kernels:  
# https://www.kaggle.com/codlife/pre-processing-for-huge-train-data-with-chunksize  
# https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue  
# https://www.kaggle.com/jsaguiar/complete-exploratory-analysis-all-columns  
# https://www.kaggle.com/shivamb/exploratory-analysis-ga-customer-revenue  
# https://www.kaggle.com/yoshoku/gacrp-v2-starter-kit/code  
# 
# I welcome comments and suggestions for improvement!  
