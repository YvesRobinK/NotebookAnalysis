#!/usr/bin/env python
# coding: utf-8

# <div style="padding:20px;color:#9E79BA;margin:0;font-size:240%;text-align:center;display:fill;border-radius:5px;background-color:white;overflow:hidden;font-weight:600">JPX Tokyo Network Analysis and Feature Generation for Stock Price Prediction</div>
# 
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">1 | Competition Overview</div>
# 

# Success in any financial market requires one to identify solid investments. When a stock or derivative is undervalued, it makes sense to buy. If it's overvalued, perhaps it's time to sell. While these finance decisions were historically made manually by professionals, technology has ushered in new opportunities for retail investors. Data scientists, specifically, may be interested to explore quantitative trading, where decisions are executed programmatically based on predictions from trained models.
# 
# There are plenty of existing quantitative trading efforts used to analyze financial markets and formulate investment strategies. To create and execute such a strategy requires both historical and real-time data, which is difficult to obtain especially for retail investors. This competition will provide financial data for the Japanese market, allowing retail investors to analyze the market to the fullest extent.
# 
# Japan Exchange Group, Inc. (JPX) is a holding company operating one of the largest stock exchanges in the world, Tokyo Stock Exchange (TSE), and derivatives exchanges Osaka Exchange (OSE) and Tokyo Commodity Exchange (TOCOM). JPX is hosting this competition and is supported by AI technology company AlpacaJapan Co.,Ltd.

# This competition will compare your models against real future returns after the training phase is complete. The competition will involve building portfolios from the stocks eligible for predictions (around 2,000 stocks). Specifically, each participant ranks the stocks from highest to lowest expected returns and is evaluated on the difference in returns between the top and bottom 200 stocks. You'll have access to financial data from the Japanese market, such as stock information and historical stock prices to train and test your model.

# All winning models will be made public so that other participants can learn from the outstanding models. Excellent models also may increase the interest in the market among retail investors, including those who want to practice quantitative trading. At the same time, you'll gain your own insights into programmatic investment methods and portfolio analysis―and you may even discover you have an affinity for the Japanese market.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sns
import altair as alt
import itertools
import time
import copy
import gc
import jpx_tokyo_market_prediction

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
alt.data_transformers.disable_max_rows()
#pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_train = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv", parse_dates=['Date'])
df_securities = df_train['SecuritiesCode'].drop_duplicates().to_frame()
df_securities


# # <div style="padding:10px;color:white;margin:0;font-size:60%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">Preparing the Data and Taking a First Look</div>
# 
# We load our input data and do some data manipulation in pandas to prepare it in formats that will suit our task at hand. Our transformed input data looks like this:

# In[2]:


df = df_train.pivot(index='Date', columns='SecuritiesCode', values='Target').dropna(axis=0)
df.head()


# We can visualise the target through time with a boxplot. We notice that there are moments of higher volatility for example around the beginning of May 2021 and mid-August 2021.

# In[3]:


df_chrt = df.unstack().to_frame().reset_index()
df_chrt.rename(columns={0:"Target"}, inplace=True)

np.random.seed(10)
D = []
unique_dates = list(df_chrt['Date'].astype(str).unique())

for ud in unique_dates:
    D.append(df_chrt.loc[df_chrt['Date']==ud, "Target"].values)


fig, ax = plt.subplots(figsize=(12,8))
#ax.violinplot(D, widths=0.001,
#                   showmeans=True, showmedians=True, showextrema=True)
ax.boxplot(D)
ax.set_xticklabels(unique_dates)
loc = plticker.MultipleLocator(base=30.0) # this locator puts ticks at regular intervals
ax.xaxis.set_major_locator(loc)
ax.set_ylabel('Target') 
ax.set_xlabel('Date') 

plt.title("Securities Target Boxplot")
plt.show()


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">2 | Why use a Network?</div>
# 
# Stock price movements are a time-series and patterns of previous price movements are useful to predict future price movements. In this competition contestants have to rank the stock movements per day identifying the 200 stocks predicted to go up the most and the 200 predicted to go down the most. The target is calculated as the difference between the adjusted closing price of the next valid trading day and the adjusted closing price of the valid trading day after that. Because the time frame is short this increases the difficulty of the predictive task. Since there are so many stocks involved we will investigate whether modelling past stock movements as a network can be useful to create features to predict future stock price movements.
# 
# Specifically, we transform the input data to have each stock as a separate column with the target per date and per stock as the contents of the dataframe. We restrict our dataset to the 232 days of training data where all the original 2000 stocks had a valid target. For earlier dates some stocks had not yet listed and will not have a valid target. Once we have done this we create a network with each stock as a node. Then we create edges between any stocks that both had a positive target value on the same date. The weight of the edge between the nodes is the sum total of all the positive target values for both stocks but only on the days that both had a positive target. This may uncover some interesting links between stocks that other traditional feature engineering may not discover. This is an exploratory piece of work so there is no guarantee the features will be effective, but at least we should get some cool visuals that may also help us to understand the data better.
# 
# In case this terminology seems confusing there is a tutorial [here](https://networkx.org/documentation/stable/tutorial.html) which provides an introduction to network models using networkx.
# 
# To illustrate the process I first start with creating a network just for the first 50 of the 2000 stocks. The network is shown in the viz below. The nodes are colored by how many connections they have to other nodes. The technical term for this in networks is the node degree, which is the number of edges they share with other nodes. The brigher the red the more edges are shared with other nodes and the fainter the red the fewer edges are shared with other nodes.

# In[4]:


def generate_options(node_color, weights, edges):
    colors = plt.cm.Reds(np.linspace(0, 1, 9))
    color_codes = [colors[x] for x in node_color]
    options = {
        "node_color": color_codes,
        "node_size": 20,
        "width": 0.25,
        "edge_color": weights,
        "edgelist": edges, 
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
    }
    
    return options

def get_node_color():
    node_color = []
    for node in G.nodes():
        if G.degree(node) == 0:
            node_color.append(0)        
        elif G.degree(node) == 1:
            node_color.append(1)        
        elif G.degree(node) < 10:
            node_color.append(2)        
        elif G.degree(node) < 40:
            node_color.append(3)        
        elif G.degree(node) < 100:
            node_color.append(4)        
        elif G.degree(node) < 400:
            node_color.append(5)        
        elif G.degree(node) < 1000:
            node_color.append(6)        
        elif G.degree(node) < 1900:
            node_color.append(7)
        else:
            node_color.append(8)
            
    return node_color

def create_graph(colz):
    G = nx.Graph()

    
    colz

    for c in colz:
        G.add_node(c)

    for o in colz:
        for i in colz:
            if o==i:
                continue
            else:
                iter_mask = (df[o] > 0) * (df[i] > 0)
                iter_weight = df.loc[iter_mask == True, [o, i]].sum().sum()
                G.add_edge(o, i, weight=iter_weight)


    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    df_weights = (pd.DataFrame(np.array(list(zip(*nx.get_edge_attributes(G,'weight').items())), dtype=object).T, 
                               columns=['edge', 'weight']).sort_values(by=['weight'])).reset_index(drop=True)
    df_weights
    
    return G, edges, weights, df_weights


colz = list(df.columns)[:50]

G, edges, weights, df_weights = create_graph(colz)

mquantiles = [0.003, 0.05, 0.1, 0.2, 0.32, 0.5, 0.68, 0.8, 0.9, 0.95, 0.997, 0.999, 0.9999]
#mquantiles = [0.1, 0.9, 0.95]#, 0.997, 0.999, 0.9999]

colors = range(8)
node_color = get_node_color()       
options = generate_options(node_color, weights, edges)

plt.figure(figsize=(12,8))
plt.title("Securities Price Movement Up Days Network for 50 stocks")
nx.draw(G, **options)

del edges, weights, node_color, options
gc.collect();


# <b>Please remember to upvote if you like this notebook</b>
# 
# We can get a sense of the weights by a cumulative count chart. Here we see that our median weight is around 1.9 and only a small tail of edges have weight greater than 3.

# In[5]:


def chart_cumulative_count(df_weights):
    chrt = alt.Chart(df_weights, title="Cumulative Count of Network Edge Weights").transform_window(
        cumulative_count="count()",
        sort=[{"field": "weight"}],
    ).mark_area().encode(
        x="weight:Q",
        y=alt.Y("cumulative_count:Q", title="Cumulative Count")
    ).properties(
        height=360,
        width=540
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=False, 
        domain=False,
        labelFontSize=12,
        titleFontSize=16
    ).configure_title(
        fontSize=20
    )
    
    return chrt
    
    
chart_cumulative_count(df_weights)


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">3 | Feature Generation Illustration with 50 Stocks</div>
# 
# With the network we now have we can compute some network metrics per node. We start with current flow closeness centrality. A centrality measure indicates how central a node is in a network. Current flow closeness centrality measures centrality that also takes into account the weight of edges. Using this we can create our first new features.

# In[6]:


#graph needs to be fully connected for this metric. Try running this before removing edges
def get_cfcc(G, df_securities):
    cfcc = nx.current_flow_closeness_centrality(G, weight="weight")
    df_securities = df_securities.merge(pd.DataFrame.from_dict(cfcc, orient='index'), how="left", left_on="SecuritiesCode", right_index=True)
    df_securities.rename(columns={0: "current_flow_closeness_centrality"}, inplace=True)
    
    ec = nx.eigenvector_centrality(G, weight="weight")
    df_securities = df_securities.merge(pd.DataFrame.from_dict(ec, orient='index'), how="left", left_on="SecuritiesCode", right_index=True)
    df_securities.rename(columns={0: "eigenvector_centrality"}, inplace=True)
    
    return df_securities


df_securities = get_cfcc(G, df_securities)
df_securities.head()

# include here as don't want an extra cell visible in the output and this cell won't need rerunning
env = jpx_tokyo_market_prediction.make_env()
iter_test = env.iter_test()


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">4 | Creating more features by removing weaker edges - Illustration with 50 stocks</div>
# 
# With networks some techniques involve starting to remove edges and calculating metrics after that is done. We will use community detection techniques to find communities within our networks. However, since our networks have a very high proportion of edges (most nodes are connected to most others) we will gradually remove the weaker edges and run community detection techniques to identify communities at this different stages of paring back the network. Community detection is a bit like clustering as it places nodes (in our case stocks) that are close to each other or closely connected to each other into the same communities but nodes that are not closely connected will be in other communities. At each stage of the paring back we create new categorical features by placing stocks into communities.

# In[7]:


def paring_back_network(G, mquantiles, df_securities, show_viz=True):
    graph_list = [] # capture all graphs with deepcopy to make sure the variable isn't updated in the list when it is updated outside the list
    for m in range(len(mquantiles)):
        df_securities['gmc_' + str(m)] = "cat_0"

        remove = list(df_weights.loc[df_weights['weight'] < df_weights['weight'].quantile(mquantiles[m]), "edge"])
        len(remove)
        G.remove_edges_from(remove)
        graph_list.append(copy.deepcopy(G))

        c = community.greedy_modularity_communities(G, weight="weight")
        #sorted(c[0])

        for mc in range(len(c)):
            df_securities.loc[df_securities['SecuritiesCode'].isin(c[mc]), "gmc_" + str(m)] = "cat_" + str(mc)

        if show_viz:
            plt.figure(figsize=(12,8))
            plt.title("Securities Price Movement Up Days Network with weakest " + str(round(100 * mquantiles[m],3)) + "% of edges removed")
            
            edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
            node_color = get_node_color()       
            options = generate_options(node_color, weights, edges)

            nx.draw(G, **options)
            
            del edges, weights, node_color, options
            gc.collect();
        
    return G, df_securities, graph_list


G, df_securities, graph_list_50 = paring_back_network(G, mquantiles, df_securities)
gc.collect();


# We can see our network at different stages of the paring back process above. Near the end we can clearly visually see at the centre of the chart are a few stocks that are much more central than the others as they are much more strongly connected to many other stocks. As we remove more and more edges there are fewer and fewer nodes (stocks) in the centre. These are the stocks that have risen most strongly together on up days.
# 
# Below we can see the new features that we have generated by network modeling. Remember currently this is just for the first 50 stocks. We will go on to run this for all 2000 stocks. However be aware that the run times will be substantially longer and the visualisations may not be as clear, or it may be difficult to even generate visualisations due to the run-time and compute requirements. However, the intuition we can see is that there is a core group of stocks that have gone up together by a combine substantial amount over the last 232 days of the training day. As this notebook develops we may also explore a similar technique for stocks that go down together. 
# 
# Here is what the new features look like. Remember this is currently only for the first 50 stocks.

# In[8]:


df_securities.head()


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">5 | Running our Network Modelling and Feature Generation Process on All Stocks</div>
# 
# Now we have seen the intuition for how the features are created on 50 stocks, we will create a graph for all stocks. The new features for all stocks will be displayed below and saved to csv. Since the run-time to create the graph is quite substantial for 2000 stocks a [dataset](https://www.kaggle.com/datasets/datahobbit/jpx-network-models-features) has been created to save the output. The dataset will continue to be updated as the notebook is still under active development. We can now load the saved graph from earlier versions rather than recalculating it every time. 
# 
# We will start off again with our cumulative count to get a sense of the distribution of weight values in our full-sized network. Unfortunately the large number of data points means that altair struggles with this viz so we will switch to matplotlib. The chart this shows us is that the median is now a little over 2 but there is a longer tail to the right of values above 3 with some going up to 8. This tail contains the strongest edges. Paring back the edges will increasingly focus in towards this tail and the community detection algorithms will find the communities of nodes that feature strongly in that tail.
# 
#  

# In[9]:


generate_full_graph = False

if generate_full_graph:
    # create graph
    colz = list(df.columns)
    G, edges, weights, df_weights = create_graph(colz)
    # save the full graph - in a later version we will load this rather than recalculating
    nx.write_gpickle(G, "fullgraph.gpickle")
else:
    G = nx.read_gpickle("../input/jpx-network-models-features/fullgraph.gpickle")
    # we need to save this back to the dataset because it gets overwritten by new runs of the notebook
    nx.write_gpickle(G, "fullgraph.gpickle")
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    df_weights = (pd.DataFrame(np.array(list(zip(*nx.get_edge_attributes(G,'weight').items())), dtype=object).T, 
                               columns=['edge', 'weight']).sort_values(by=['weight'])).reset_index(drop=True)


values, base = np.histogram(df_weights['weight'], bins=len(df_weights['weight']))
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
plt.figure(figsize=(12,8))
plt.plot(base[:-1], cumulative, c='blue')
plt.title("Cumulative Count of Network Edge Weights")
plt.ticklabel_format(style='plain')

plt.show()


# In[10]:


# prepare the first numeric features
df_securities = df_securities['SecuritiesCode'].to_frame()
df_securities = get_cfcc(G, df_securities)


# In[11]:


edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
node_color = get_node_color()       
options = generate_options(node_color, weights, edges)
plt.figure(figsize=(12,8))
plt.title("Securities Price Movement Up Days Network for All Stocks")
nx.draw(G, **options)

del edges, weights, node_color, options
gc.collect();


# In[12]:


# prepare the community detection categorical features
G, df_securities, graph_list = paring_back_network(G, mquantiles, df_securities, show_viz=True)


# Here are our new features created for all securities. <i>More to follow </i>

# In[13]:


df_securities.to_csv("network_features.csv", index=False)
df_securities


# <b>Please remember to upvote if you found this notebook informative.</b>
# 
# We save our graph and features for later use. This could save a lot of time later. It takes 2 hours to generate the graph (could probably be optimised a bit) but only a few seconds to load.

# In[14]:


plt.figure(figsize=(12,8))
plt.title("All Stocks with weakest " + str(round(100 * mquantiles[-1],3)) + "% of edges removed")
nx.write_gpickle(G, "graph9999.gpickle")
G2 = nx.read_gpickle("graph9999.gpickle")
#nx.draw(G2, **options)


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">6 | Alternative Network Visualisations </div>
# 
# Graphs can be visualised with various formats. Here we see the spring format.

# In[15]:


edges,weights = zip(*nx.get_edge_attributes(G2,'weight').items())
node_color = get_node_color()       
options = generate_options(node_color, weights, edges)

plt.figure(figsize=(11,7))
plt.title("Spring Layout with weakest " + str(round(100 * mquantiles[-1],3)) + "% of edges removed")
nx.draw(G2, pos=nx.spring_layout(G2), **options)

del edges, weights, node_color, options
gc.collect();


# Let me know in the comments below if you have any feedback or questions. Thanks for taking a look at my notebook!

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">7 | Using the features to predict future returns</div>
# 
# Now that we have generated all these new features we will use them to generate a forecast in the required format for the competition.

# In[16]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
df_train = pd.read_csv("../input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv", parse_dates=['Date'])
df_securities = df_train['SecuritiesCode'].drop_duplicates().to_frame()
df_network_features = pd.read_csv("../input/jpx-network-models-features/network_features.csv")
df_train = df_train.merge(df_network_features, how="left", on="SecuritiesCode")
df_train = df_train.groupby(['SecuritiesCode'], as_index=False).apply(lambda group: group.ffill())    
df_train = df_train.groupby(['SecuritiesCode'], as_index=False).apply(lambda group: group.bfill()) 
excl_cols = ['Date', 'RowId', 'ExpectedDividend', 'ExpectedDividend', 'SupervisionFlag', 'Target']
cols = [c for c in df_train.columns if c not in excl_cols]
X = df_train.loc[df_train.Date>='2021-08-01', cols]
y = df_train.loc[df_train.Date>='2021-08-01', "Target"]
df_train.drop(columns=['Target'], inplace=True)

cat_cols = [c for c in cols if c[:3]=="gmc"]
le = preprocessing.LabelEncoder()
for c in cat_cols:
    X[c] = le.fit_transform(X[c])
    
regr = RandomForestRegressor(
    max_depth=5, 
    max_samples=0.75, 
    n_estimators=500, 
    n_jobs=-1, 
    random_state=1234, 
    oob_score=True
)
regr.fit(X, y)


# In[17]:


counter = 0
# The API will deliver six dataframes in this specific order:

for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    prices = prices.iloc[:,:11]
    prices = prices.merge(df_network_features, how="left", on="SecuritiesCode")
    current_date = prices['Date'].max()
    
    df_train = df_train.append(prices).reset_index(drop=True)
    df_train = df_train.groupby(['SecuritiesCode'], as_index=False).apply(lambda group: group.ffill())    
    df_train = df_train.groupby(['SecuritiesCode'], as_index=False).apply(lambda group: group.bfill()) 
    excl_cols = ['Date', 'RowId', 'ExpectedDividend', 'ExpectedDividend', 'SupervisionFlag', 'Target']
    cols = [c for c in df_train.columns if c not in excl_cols]
    X_test = df_train.loc[df_train.Date==current_date, cols]
    
    for c in cat_cols:
        X_test[c] = le.transform(X_test[c])
    
    y_pred = regr.predict(X_test)
    
    sample_prediction['Rank'] = y_pred
    sample_prediction["Rank"] = (sample_prediction["Rank"].rank(method="first", ascending=False)-1).astype(int).values
    
    assert sample_prediction["Rank"].notna().all()
    assert sample_prediction["Rank"].min() == 0
    assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1
    
    env.predict(sample_prediction)
    counter += 1
    
#print(counter)


# In[18]:


# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">5 | Adding the New Features to an Existing Model</div>

# Watch this space... 
# Possible next steps
# A short equivalent to the above - stocks where target < 0
# instead of doing a sum do a count as the edge weights for long and / or short networks
# The notebook will shortly be expanded to add the features shown above (and others) to an existing notebook.


# # <div style="padding:10px;color:white;margin:0;font-size:60%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">Note on Time Series and Leakage</div>
# Remember that these features have been created by looking at the training data up to 03 Dec 2021. Using these features could lead to a leakage of information back in time if they are used in a time-series based cross-validation approach.

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#9E79BA;overflow:hidden">References </div>
# 
# * This well laid out notebook contains useful EDA and a solid cv approach: [jpx-stock-market-analysis-prediction-with-lgbm](https://www.kaggle.com/code/kellibelcher/jpx-stock-market-analysis-prediction-with-lgbm)
# 
# 
