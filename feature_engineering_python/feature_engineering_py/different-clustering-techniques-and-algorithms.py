#!/usr/bin/env python
# coding: utf-8

# # 1. What is Clustering?

# Many things around us can be categorized as “this and that” or to be less vague and more specific, we have groupings that could be binary or groups that can be more than two, like a type of pizza base or type of car that you might want to purchase. The choices are always clear – or, how the technical lingo wants to put it – predefined groups and the process predicting that is an important process in the Data Science stack called Classification.
# 
# But what if we bring into play a quest where we don’t have pre-defined choices initially, rather, we derive those choices! Choices that are based out of hidden patterns, underlying similarities between the constituent variables, salient features from the data etc. This process is known as Clustering in Machine Learning or Cluster Analysis, where we group the data together into an unknown number of groups and later use that information for further business processes.
# 
# 
# So, to put it in simple words, in machine learning clustering is the process by which we create groups in a data, like customers, products, employees, text documents, in such a way that objects falling into one group exhibit many similar properties with each other and are different from objects that fall in the other groups that got created during the process.

# ## 1.1. Data Exploration

# In[15]:


import pandas as pd
df=pd.read_csv("/kaggle/input/tabular-playground-series-jul-2022/data.csv")
ss=pd.read_csv("/kaggle/input/tabular-playground-series-jul-2022/sample_submission.csv")


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(24,20)})
sns.heatmap(df.corr(),annot=True,fmt='.2f')


# In[17]:


sns.set(rc={'figure.figsize':(15,15)})
for i, column in enumerate(list(df.columns), 1):
    plt.subplot(5,6,i)
    p=sns.histplot(x=column,data=df.sample(1000),stat='count',kde=True,color='green')


# # 2. What are clustering algorithms?

# Clustering is an unsupervised machine learning task. You might also hear this referred to as cluster analysis because of the way this method works. Using a clustering algorithm means you're going to give the algorithm a lot of input data with no labels and let it find any groupings in the data it can.
# 
# Those groupings are called clusters. A cluster is a group of data points that are similar to each other based on their relation to surrounding data points. Clustering is used for things like feature engineering or pattern discovery.
# 
# **When you're starting with data you know nothing about, clustering might be a good place to get some insight.**

# # 3. Types of clustering algorithms
# There are different types of clustering algorithms that handle all kinds of unique data.

# ## 3.1. Density-based
# In density-based clustering, data is grouped by areas of high concentrations of data points surrounded by areas of low concentrations of data points. Basically the algorithm finds the places that are dense with data points and calls those clusters.
# The great thing about this is that the clusters can be any shape. You aren't constrained to expected conditions.
# The clustering algorithms under this type don't try to assign outliers to clusters, so they get ignored.
# 
# Density-based clustering connects areas of high example density into clusters. This allows for arbitrary-shaped distributions as long as dense areas can be connected. These algorithms have difficulty with data of varying densities and high dimensions. Further, by design, these algorithms do not assign outliers to clusters.
# 
# ![](https://www.researchgate.net/publication/334279038/figure/fig5/AS:960330599514121@1605972065828/Density-based-clustering-techniques-DBSCAN_Q320.jpg)

# ## 3.2. Distribution-based
# With a distribution-based clustering approach, all of the data points are considered parts of a cluster based on the probability that they belong to a given cluster.
# It works like this: there is a center-point, and as the distance of a data point from the center increases, the probability of it being a part of that cluster decreases.
# If you aren't sure of how the distribution in your data might be, you should consider a different type of algorithm.
# 
# This clustering approach assumes data is composed of distributions, such as Gaussian distributions. In Figure 3, the distribution-based algorithm clusters data into three Gaussian distributions. As distance from the distribution's center increases, the probability that a point belongs to the distribution decreases. The bands show that decrease in probability. When you do not know the type of distribution in your data, you should use a different algorithm.
# 
# ![](https://www.researchgate.net/publication/332053160/figure/fig2/AS:741417534107649@1553779123709/Distribution-model-of-clustering.png)

# ## 3.3. Centroid-based
# Centroid-based clustering is the one you probably hear about the most. It's a little sensitive to the initial parameters you give it, but it's fast and efficient.
# 
# These types of algorithms separate data points based on multiple centroids in the data. Each data point is assigned to a cluster based on its squared distance from the centroid. This is the most commonly used type of clustering.
# 
# ![](https://www.researchgate.net/publication/334279038/figure/fig6/AS:960330603712512@1605972066015/Centroid-based-clustering-algorithm.png)

# ## 3.4. Hierarchical-based
# Hierarchical-based clustering is typically used on hierarchical data, like you would get from a company database or taxonomies. It builds a tree of clusters so everything is organized from the top-down.
# This is more restrictive than the other clustering types, but it's perfect for specific kinds of data sets.
# 
# **Hierarchical clustering creates a tree of clusters. Hierarchical clustering, not surprisingly, is well suited to hierarchical data, such as taxonomies.**
# 
# ![](https://www.analytixlabs.co.in/blog/wp-content/uploads/2020/07/image-3-28-1-600x400.jpg)

# ### 3.4.1. Divisive Approach
# This approach of hierarchical clustering follows a top-down approach where we consider that all the data points belong to one large cluster and try to divide the data into smaller groups based on a termination logic or, a point beyond which there will be no further division of data points. This termination logic can be based on the minimum sum of squares of error inside a cluster or for categorical data, the metric can be the GINI coefficient inside a cluster.
# 
# ![](https://www.analytixlabs.co.in/blog/wp-content/uploads/2020/07/image-4-17-1-600x323.jpg)

# ### 3.4.2. Agglomerative Approach
# Agglomerative is quite the contrary to Divisive, where all the “N” data points are considered to be a single member of “N” clusters that the data is comprised into. We iteratively combine these numerous “N” clusters to fewer number of clusters, let’s say “k” clusters and hence assign the data points to each of these clusters accordingly. This approach is a bottom-up one, and also uses a termination logic in combining the clusters. This logic can be a number based criterion (no more clusters beyond this point) or a distance criterion (clusters should not be too far apart to be merged) or variance criterion (increase in the variance of the cluster being merged should not exceed a threshold, Ward Method)

# # 4. Types of Clustering Algorithms

# # >> 4.1. k-Means Clustering

# k-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 
# 
# K-means clustering uses “centroids”, K different randomly-initiated points in the data, and assigns every data point to the nearest centroid. After every point has been assigned, the centroid is moved to the average of all of the points assigned to it.
# 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/20190812011831/Screenshot-2019-08-12-at-1.09.42-AM.png)
# 
# #### Where is K-means clustering used?
# kmeans algorithm is very popular and used in a variety of applications such as market segmentation, document clustering, image segmentation and image compression, etc. The goal usually when we undergo a cluster analysis is either: Get a meaningful intuition of the structure of the data we're dealing with.

# ## 4.1.1. Advantages of k-means
# - Relatively simple to implement
# - Scales to large data sets.
# - Guarantees convergence.
# - Can warm-start the positions of centroids.
# - Easily adapts to new examples.
# - Generalizes to clusters of different shapes and sizes, such as elliptical clusters.
# 
# ## 4.1.2. Disadvantages of k-means
# - Choosing K manually.
#    - Use the “Loss vs. Clusters” plot to find the optimal (k), as discussed in Interpret Results.
# - Being dependent on initial values.
#    - For a low , you can mitigate this dependence by running k-means several times with different initial values and picking the best result. As  increases, you need advanced versions of k-means to pick better values of the initial centroids (called k-means seeding). For a full discussion of k- means seeding see, A Comparative Study of Efficient Initialization Methods for the K-Means Clustering Algorithm by M. Emre Celebi, Hassan A. Kingravi, Patricio A. Vela.
# - Clustering data of varying sizes and density.
#    - k-means has trouble clustering data where clusters are of varying sizes and density. To cluster such data, you need to generalize k-means as described in the Advantages section.
# - Clustering outliers.
#    - Centroids can be dragged by outliers, or outliers might get their own cluster instead of being ignored. Consider removing or clipping outliers before clustering.
# - Scaling with number of dimensions.
#    - As the number of dimensions increases, a distance-based similarity measure converges to a constant value between any given examples. Reduce dimensionality either by using PCA on the feature data, or by using “spectral clustering” to modify the clustering algorithm as explained below.

# ## 4.1.3. Code Example:

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
kmeans_model = KMeans(n_clusters=2)

# assign each data point to a cluster
kmeans_result = kmeans_model.fit_predict(training_data)

# get all of the unique clusters
kmeans_clusters = unique(kmeans_result)

# plot the DBSCAN clusters
for dbscan_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = where(kmeans_result == kmeans_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the DBSCAN plot
pyplot.show()


# # >> 4.2. DBSCAN clustering algorithm

# DBSCAN is a density-based clustering algorithm that works on the assumption that clusters are dense regions in space separated by regions of lower density. It groups 'densely grouped' data points into a single cluster.
# 
# #### Is DBSCAN faster than KMeans?
# DBSCAN produces a varying number of clusters, based on the input data. Here's a list of advantages of KMeans and DBScan: KMeans is much faster than DBScan. DBScan doesn't need number of clusters.
# 
# ![](https://www.analytixlabs.co.in/blog/wp-content/uploads/2020/07/image-12-1-600x314.jpg)
# 
# 
# #### What is the basic principle of DBSCAN clustering?
# The principle of DBSCAN is to find the neighborhoods of data points exceeds certain density threshold. The density threshold is defined by two parameters: the radius of the neighborhood (eps) and the minimum number of neighbors/data points (minPts) within the radius of the neighborhood.
# 
# #### What is the difference between KMeans and DBSCAN?
# K-means needs a prototype-based concept of a cluster. DBSCAN needs a density-based concept. K-means has difficulty with non-globular clusters and clusters of multiple sizes. DBSCAN is used to handle clusters of multiple sizes and structures and is not powerfully influenced by noise or outliers.

# ## 4.2.1. Advantages of DBSCAN clustering
# - Can easily deal with noise, not affected by outliers.
# - Doesn’t require prior specification of clusters.
# - It has no strict shapes, it can correctly accommodate many data points.
# 
# ## 4.2.2. Disadvantages of DBSCAN clustering
# - Sensitive to the clustering hyper-parameters – the eps and the min_points.
# - Cannot work with datasets of varying densities.
# - Fails if the data is too sparse.
# - The density measures (Reachability and Connectivity) can be affected by sampling.

# ## 4.2.3. Code Example:

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
dbscan_model = DBSCAN(eps=0.25, min_samples=9)

# train the model
dbscan_model.fit(training_data)

# assign each data point to a cluster
dbscan_result = dbscan_model.predict(training_data)

# get all of the unique clusters
dbscan_cluster = unique(dbscan_result)

# plot the DBSCAN clusters
for dbscan_cluster in dbscan_clusters:
    # get data points that fall in this cluster
    index = where(dbscan_result == dbscan_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the DBSCAN plot
pyplot.show()


# # >> 4.3. Gaussian Mixture Model algorithm

# Gaussian mixture models (GMMs) are often used for data clustering. You can use GMMs to perform either hard clustering or soft clustering on query data. To perform hard clustering, the GMM assigns query data points to the multivariate normal components that maximize the component posterior probability, given the data.
# 
# Gaussian Mixture Models (GMMs) assume that there are a certain number of Gaussian distributions, and each of these distributions represent a cluster. Hence, a Gaussian Mixture Model tends to group the data points belonging to a single distribution together.
# 
# ![](https://www.analytixlabs.co.in/blog/wp-content/uploads/2020/07/image-14-1-600x450.jpg)
# 
# At its simplest, GMM is also a type of clustering algorithm. As its name implies, each cluster is modelled according to a different Gaussian distribution. This flexible and probabilistic approach to modelling the data means that rather than having hard assignments into clusters like k-means, we have soft assignments.

# ## 4.3.1. Advantages of Gaussian Mixture Model algorithm
# - The associativity of a data point to a cluster is quantified using probability metrics – which can be easily interpreted.
# - Proven to be accurate for real-time data sets.
# - Some versions of GMM allows for mixed membership of data points, hence it can be a good alternative to Fuzzy C Means to achieve fuzzy clustering.
# 
# 
# ## 4.3.2. Disadvantages of Gaussian Mixture Model algorithm
# - Complex algorithm and cannot be applicable to larger data
# - It is hard to find clusters if the data is not Gaussian, hence a lot of data preparation is required.

# ## 4.3.3. Code Example :

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
gaussian_model = GaussianMixture(n_components=2)

# train the model
gaussian_model.fit(training_data)

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(training_data)

# get all of the unique clusters
gaussian_clusters = unique(gaussian_result)

# plot Gaussian Mixture the clusters
for gaussian_cluster in gaussian_clusters:
    # get data points that fall in this cluster
    index = where(gaussian_result == gaussian_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Gaussian Mixture plot
pyplot.show()


# # > 4.4. BIRCH algorithm
# 
# **The Balance Iterative Reducing and Clustering using Hierarchies (BIRCH) algorithm works better on large data sets than the k-means algorithm.**
# It breaks the data into little summaries that are clustered instead of the original data points. The summaries hold as much distribution information about the data points as possible.
# 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/20200612004451/BIRCH.png)
# 
# This algorithm is commonly used with other clustering algorithm because the other clustering techniques can be used on the summaries generated by BIRCH.
# The main downside of the BIRCH algorithm is that it only works on numeric data values. You can't use this for categorical values unless you do some data transformations.

# ## 4.3.1. Advantages of BIRCH algorithm
# - Finds a good clustering with a single scan and improves the quality with a few additional scans
# 
# ## 4.3.2. Disadvantages of BIRCH algorithm
# - Handles only numeric data
# 
# ## 4.3.3. Applications of BIRCH algorithm
# - Pixel classification in images
# - Image compression
# - Works with very large data sets

# ## 4.3.4. Code Example:

# In[ ]:


# Import required libraries and modules
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import Birch
 
# Generating 600 samples using make_blobs
dataset, clusters = make_blobs(n_samples = 600, centers = 8, cluster_std = 0.75, random_state = 0)
 
# Creating the BIRCH clustering model
model = Birch(branching_factor = 50, n_clusters = None, threshold = 1.5)
 
# Fit the data (Training)
model.fit(dataset)
 
# Predict the same data
pred = model.predict(dataset)
 
# Creating a scatter plot
plt.scatter(dataset[:, 0], dataset[:, 1], c = pred, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')
plt.show()


# # > 4.5. Affinity Propagation clustering algorithm

# Affinity propagation (AP) is a graph based clustering algorithm similar to k Means or K medoids, which does not require the estimation of the number of clusters before running the algorithm. Affinity propagation finds “exemplars” i.e. members of the input set that are representative of clusters.
# 
# Each data point communicates with all of the other data points to let each other know how similar they are and that starts to reveal the clusters in the data. You don't have to tell this algorithm how many clusters to expect in the initialization parameters.
# 
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhsCOnmydquBzvTqjIoAj1p6vamOiO5J0Ix7LKrUjzxw&s)

# ## 4.5.1. Code Examples

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
model = AffinityPropagation(damping=0.7)

# train the model
model.fit(training_data)

# assign each data point to a cluster
result = model.predict(training_data)

# get all of the unique clusters
clusters = unique(result)

# plot the clusters
for cluster in clusters:
    # get data points that fall in this cluster
    index = where(result == cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the plot
pyplot.show()


# # > 4.6. Mean-Shift clustering algorithm

# Mean shift is a non-parametric feature-space mathematical analysis technique for locating the maxima of a density function, a so-called mode-seeking algorithm. Application domains include cluster analysis in computer vision and image processing.
# 
# It is a density-based clustering algorithm where it firstly, seeks for stationary points in the density function. Then next, the clusters are eventually shifted to a region with higher density by shifting the center of the cluster to the mean of the points present in the current window. The shift if the window is repeated until no more points can be accommodated inside of that window.
# 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/20190429213154/1354.png)

# ## 4.6.1. Advantages of Mean-Shift clustering 
# - The following are some advantages of Mean-Shift clustering algorithm −
# - It does not need to make any model assumption as like in K-means or Gaussian mixture.
# - It can also model the complex clusters which have nonconvex shape.
# - It only needs one parameter named bandwidth which automatically determines the number of clusters.
# - There is no issue of local minima as like in K-means.
# - No problem generated from outliers.
# 
# ## 4.6.2. Disadvantages of Mean-Shift clustering 
# - The following are some disadvantages of Mean-Shift clustering algorithm −
# - Mean-shift algorithm does not work well in case of high dimension, where number of clusters changes abruptly.
# - We do not have any direct control on the number of clusters but in some applications, we need a specific number of clusters.
# - It cannot differentiate between meaningful and meaningless modes.

# ## 4.6.3. Code Example

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
mean_model = MeanShift()

# assign each data point to a cluster
mean_result = mean_model.fit_predict(training_data)

# get all of the unique clusters
mean_clusters = unique(mean_result)

# plot Mean-Shift the clusters
for mean_cluster in mean_clusters:
    # get data points that fall in this cluster
    index = where(mean_result == mean_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Mean-Shift plot
pyplot.show()


# # > 4.7. OPTICS algorithm
# **OPTICS Clustering stands for Ordering Points To Identify Cluster Structure.** It draws inspiration from the DBSCAN clustering algorithm. It adds two more terms to the concepts of DBSCAN clustering.
# 
# They are:-
# - **Core Distance:** It is the minimum value of radius required to classify a given point as a core point. If the given point is not a Core point, then it’s Core Distance is undefined.
# - **Reachability Distance:** It is defined with respect to another data point q(Let). The Reachability distance between a point p and q is the maximum of the Core Distance of p and the Euclidean Distance(or some other distance metric) between p and q. Note that The Reachability Distance is not defined if q is not a Core point.
# 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/20190711114717/reachability_distance1.png)

# ## 4.7.1. OPTICS Clustering v/s DBSCAN Clustering:
# 
# - **Memory Cost :** The OPTICS clustering technique requires more memory as it maintains a priority queue (Min Heap) to determine the next data point which is closest to the point currently being processed in terms of Reachability Distance. It also requires more computational power because the nearest neighbour queries are more complicated than radius queries in DBSCAN.
# - **Fewer Parameters :** The OPTICS clustering technique does not need to maintain the epsilon parameter and is only given in the above pseudo-code to reduce the time taken. This leads to the reduction of the analytical process of parameter tuning.
# This technique does not segregate the given data into clusters. It merely produces a Reachability distance plot and it is upon the interpretation of the programmer to cluster the points accordingly.

# ## 4.7.2. Code Example: 

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
optics_model = OPTICS(eps=0.75, min_samples=10)

# assign each data point to a cluster
optics_result = optics_model.fit_predict(training_data)

# get all of the unique clusters
optics_clusters = unique(optics_clusters)

# plot OPTICS the clusters
for optics_cluster in optics_clusters:
    # get data points that fall in this cluster
    index = where(optics_result == optics_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the OPTICS plot
pyplot.show()


# # > 4.8. Agglomerative Hierarchy clustering algorithm
# The agglomerative clustering is the most common type of hierarchical clustering used to group objects in clusters based on their similarity. It's also known as AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a singleton cluster.

# ## 4.8.1. Advantages of Agglomerative Hierarchy clustering algorithm
# - No prior knowledge about the number of clusters is needed, although the user needs to define a threshold for divisions.
# - Easy to implement across various forms of data and known to provide robust results for data generated via various sources. Hence it has a wide application area.
# 
# ## 4.8.2. Disadvantages of Agglomerative Hierarchy clustering algorithm
# - The cluster division (DIANA) or combination (AGNES) is really strict and once performed, it cannot be undone and re-assigned in subsequesnt iterations or re-runs.
# - It has a high time complexity, in the order of O(n^2 log n) for all the n data-points, hence cannot be used for larger datasets.
# - Cannot handle outliers and noise

# 
# ![](https://www.researchgate.net/profile/Satinder-Bal/publication/348137354/figure/fig2/AS:975396417830915@1609564036826/Agglomerative-hierarchical-clustering-algorithm.png)

# ## 4.8.3. Code Example: 

# In[ ]:


from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
agglomerative_model = AgglomerativeClustering(n_clusters=2)

# assign each data point to a cluster
agglomerative_result = agglomerative_model.fit_predict(training_data)

# get all of the unique clusters
agglomerative_clusters = unique(agglomerative_result)

# plot the clusters
for agglomerative_cluster in agglomerative_clusters:
    # get data points that fall in this cluster
    index = where(agglomerative_result == agglomerative_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Agglomerative Hierarchy plot
pyplot.show()


# # 4.9. DIANA or Divisive Analysis

# DIANA is also known as DIvisie ANAlysis clustering algorithm. It is the top-down approach form of hierarchical clustering where all data points are initially assigned a single cluster. Further, the clusters are split into two least similar clusters. The divisive clustering algorithm is a top-down clustering approach, initially, all the points in the dataset belong to one cluster and split is performed recursively as one moves down the hierarchy. 
# ![](https://media.geeksforgeeks.org/wp-content/uploads/20190508025314/781ff66c-b380-4a78-af25-80507ed6ff261-300x300.png)

# ## 4.9.1. What are the advantages of divisive clustering techniques?
# Divisive clustering is more efficient if we do not generate a complete hierarchy all the way down to individual data leaves. The time complexity of a naive agglomerative clustering is O(n3) because we exhaustively scan the N x N matrix dist_mat for the lowest distance in each of N-1 iterations.

# ## 4.9.2. Code Example (R) :

# In[ ]:


"""
# Compute diana()
library(cluster)
res.diana <- diana(USArrests, stand = TRUE)

# Plot the dendrogram
library(factoextra)
fviz_dend(res.diana, cex = 0.5,
          k = 4, # Cut in four groups
          palette = "jco" # Color palette
          )

"""


# # 4.10. Fuzzy Analysis Clustering

# Fuzzy c-means (FCM) is a data clustering technique in which a data set is grouped into N clusters with every data point in the dataset belonging to every cluster to a certain degree. For example, a data point that lies close to the center of a cluster will have a high degree of membership in that cluster, and another data point that lies far away from the center of a cluster will have a low degree of membership to that cluster.
# 
# The fcm function performs FCM clustering. It starts with a random initial guess for the cluster centers; that is the mean location of each cluster. Next, fcm assigns every data point a random membership grade for each cluster. By iteratively updating the cluster centers and the membership grades for each data point, fcm moves the cluster centers to the correct location within a data set and, for each data point, finds the degree of membership in each cluster. This iteration minimizes an objective function that represents the distance from any given data point to a cluster center weighted by the membership of that data point in the cluster.
# 
# This algorithm follows the fuzzy cluster assignment methodology of clustering. The working of FCM Algorithm is almost similar to the k-means – distance-based cluster assignment – however, the major difference is, as mentioned earlier, that according to this algorithm, a data point can be put into more than one cluster. 
# 
# ![](https://es.mathworks.com/help/examples/fuzzy/win64/fcmdemo_codepad_01.png)

# ## 4.10.1. Advantages of Fuzzy Analysis Clustering
# - FCM works best for highly correlated and overlapped data, where k-means cannot give any conclusive results.
# - It is an unsupervised algorithm and it has a higher rate of convergence than other partitioning based algorithms.
# 
# ## 4.10.2. Disadvantages of Fuzzy Analysis Clustering
# - We need to specify the number of clusters “k” prior to the start of the algorithm
# - Although convergence is always guaranteed but the process is very slow and this cannot be used for larger data.
# - Prone to errors if the data has noise and outliers.

# ## 4.10.3. Code Example 

# In[ ]:


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# Define three cluster centers
centers = [[4, 2],
           [1, 7],
           [5, 6]]

# Define three cluster sigmas in x and y, respectively
sigmas = [[0.8, 0.3],
          [0.3, 0.5],
          [1.1, 0.7]]

# Generate test data
np.random.seed(42)  # Set seed for reproducibility
xpts = np.zeros(1)
ypts = np.zeros(1)
labels = np.zeros(1)
for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
    xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
    ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
    labels = np.hstack((labels, np.ones(200) * i))

# Visualize the test data
fig0, ax0 = plt.subplots()
for label in range(3):
    ax0.plot(xpts[labels == label], ypts[labels == label], '.',
             color=colors[label])
ax0.set_title('Test data: 200 points x3 clusters.')


# # 5. Comperative Analysis of Algorithms
# Credit: [*Sunit Prasad, Different Types of Clustering Methods and Applications*](https://www.analytixlabs.co.in/blog/types-of-clustering-algorithms/)

# | Clustering Method                              | Description                                                                                                           | Advantages                                                                                                                                                                | Disadvantages                                                                                                                                          | Algorithms                                       |
# | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
# | Hierarchical Clustering                        | Based on top-to-bottom hierarchy of the data points to create clusters.                                               | Easy to implement, the number of clusters need not be specified apriori, dendrograms are easy to interpret.                                                               | Cluster assignment is strict and cannot be undone, high time complexity, cannot work for a larger dataset                                              | DIANA, AGNES, hclust etc.                        |
# | Partitioning methods                           | Based on centroids and data points are assigned into a cluster based on its proximity to the cluster centroid         | Easy to implement, faster processing, can work on larger data, easy to interpret the outputs                                                                              | We need to specify the number of cenrtroids apriori, clusters that get created are of inconsistent sizes and densities, Effected by noise and outliers | k-means, k-medians, k-modes                      |
# | Distribution-based Clustering                  | Based on the probability distribution of the data, clusters are derived from various metrics like mean, variance etc. | Number of clusters need not be specified apriori, works on real-time data, metrics are easy to understand and tune                                                        | Complex algorithm and slow, cannot be scaled to larger data                                                                                            | Gaussian Mixed Models, DBCLASD                   |
# | Density-based Clustering (Model-based methods) | Based on density of the data points, also known as model based clustering                                             | Can handle noise and outliers, need not specify number of clusters in the start, clusters that are created are highly homogenous, no restrictions on cluster shapes.      | Complex algorithm and slow, cannot be scaled to larger data                                                                                            | DENCAST, DBSCAN                                  |
# | Fuzzy Clustering                               | Based on Partitioning Approach but data points can belong to more than one cluster                                    | Can work on highly overlapped data, a higher rate of convergence                                                                                                          | We need to specify the number of centroids apriori, Effected by noise and outliers, Slow algorithm and cannot be scaled                                | Fuzzy C Means, Rough k means                     |
# | Constraint Based (Supervised Clustering)       | Clustering is directed and controlled by user constraints                                                             | Creates a perfect decision boundary, can automatically determine the outcome classes based on constraints, future data can be classified based on the training boundaries | Overfitting, high level of misclassification errors, cannot be trained on larger datasets                                                              | Decision Trees, Random Forest, Gradient Boosting |

# ## Appreciate, Upvote, Comment, Share, Enjoy!!

# # References and Credits
# - [Different Types of Clustering Methods and Applications](https://www.analytixlabs.co.in/blog/types-of-clustering-algorithms/)
# - [The 5 Clustering Algorithms Data Scientists Need to Know](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)
# - [What is Clustering and Different Types of Clustering Methods](https://www.upgrad.com/blog/clustering-and-types-of-clustering-methods/)
# - [8 Clustering Algorithms in Machine Learning that All Data Scientists Should Know](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/)
# - [Google Developers : Clustering Algorithms ](https://developers.google.com/machine-learning/clustering/clustering-algorithms)
# - [Clustering Technique](https://www.sciencedirect.com/topics/computer-science/clustering-technique)
# - [TYPES OF CLUSTERING METHODS: OVERVIEW AND QUICK START R CODE](https://www.datanovia.com/en/blog/types-of-clustering-methods-overview-and-quick-start-r-code/)
# - [5 Clustering Methods and Applications](https://www.analyticssteps.com/blogs/5-clustering-methods-and-applications)
# - [ML - Clustering Mean Shift Algorithm](https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_clustering_algorithms_mean_shift.htm#:~:text=Advantages%20and%20Disadvantages&text=It%20can%20also%20model%20the,No%20problem%20generated%20from%20outliers.)
# - [Explain BIRCH algorithm with example](https://www.ques10.com/p/9298/explain-birch-algorithm-with-example/)
# - [Build Better and Accurate Clusters with Gaussian Mixture Models](https://www.analyticsvidhya.com/blog/2019/10/gaussian-mixture-models-clustering/)
# - [ML | BIRCH Clustering](https://www.geeksforgeeks.org/ml-birch-clustering/)
# - [Fig - Centroid Based](https://www.researchgate.net/figure/Centroid-based-clustering-algorithm_fig6_334279038)
# - [Agglomerative hierarchical clustering algorithm. | Download Scientific Diagram](https://www.google.com/search?q=Agglomerative+Hierarchy+clustering+algorithm&rlz=1C1UEAD_enBD996BD996&hl=en&sxsrf=ALiCzsbN-VXLpxg4c8kxBys8tx8VzMsvlw:1656698373967&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiS9NGwotj4AhXo-TgGHTM-DXAQ_AUoAXoECAIQAw&biw=1920&bih=937&dpr=1#imgrc=iROfxWllHmNg0M)
# - [HOW THE HIERARCHICAL CLUSTERING ALGORITHM WORKS](https://dataaspirant.com/hierarchical-clustering-algorithm/)
# - [Agglomerative Hierarchical Clustering - Datanovia](https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/#:~:text=The%20agglomerative%20clustering%20is%20the,object%20as%20a%20singleton%20cluster.)
# - [Divisive Hierarchical Clustering](https://www.datanovia.com/en/lessons/divisive-hierarchical-clustering/)
# - [Fuzzy C-Means Clustering](https://es.mathworks.com/help/fuzzy/fuzzy-c-means-clustering.html;jsessionid=af46b531dd62efa365ff83aa21ed)
# - [Fuzzy C-Means Clustering](https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html)
# - [www.analytixlabs.co.in](www.analytixlabs.co.in)
