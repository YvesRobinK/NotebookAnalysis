#!/usr/bin/env python
# coding: utf-8

# 
# ## End-to-end object-based solution for DSTL
# 
# There are typically two approaches for geo-image-segmentation: pixel-based and object-based and I'm just surprised that the latter was rarely mentioned in the forum and there seems no kernels available for this competition so I decided to share my object-based solution and hopefully it will be of help.
# 
# Again, this competition comes with tons of challenges mostly in programmming/engineering which have been a pain for me. I'm grateful for those who shared their scripts/solutions. Without them, I weren't be able to make a single valid submission.
# 
# 
# 
# This solution was inspried by the following two articles:
# 
# * Python for Object Based Image Analysis (OBIA)
# https://www.machinalis.com/blog/obia/
# 
# * A Python-Based Open Source System for Geographic Object-Based Image Analysis (GEOBIA) Utilizing Raster Attribute Tables
# http://www.mdpi.com/2072-4292/6/7/6111/htm
# 
# Many ideas/functions/tools were borrowed from Konstantin Lopuhin's great kernel:
# https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
#  

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Prepping

# In[2]:


import pandas as pd

GRID_SIZE = pd.read_csv('../input/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
GRID_SIZE.columns = ['ImageId','Xmax','Ymin']
TRAIN_WKT = pd.read_csv('../input/train_wkt_v4.csv')

## Exclude empty polygons
# TRAIN_WKT[TRAIN_WKT['MultipolygonWKT']!='MULTIPOLYGON EMPTY']

CLASSES = {
        1 : 'Buildings',
        2 : 'Misc',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Waterway',
        8 : 'Standing water',
        9 : 'Vehicle Large',
        10 : 'Vehicle Small',
        }

COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }
ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        }    

sample_submission = pd.read_csv('../input/sample_submission.csv')
test_image_ids = sample_submission.ImageId.unique()
train_image_ids = TRAIN_WKT.ImageId.unique()


# ### Run following command to create working folders if needed

# In[3]:


## !mkdir ../input/feature;mkdir ../input/feature_train;mkdir ../input/feature/segment


# ## Image segmentation
# 
# * We will use RSGISLib for image segmentation
# 
#     * Webiste: http://www.rsgislib.org/index.html
#     * Installation:
#     
#         * http://www.rsgislib.org/download.html#binary-downloads 
#     
#         * https://groups.google.com/forum/#!searchin/rsgislib-support/MAC$20INSTALLATION%7Csort:relevance/rsgislib-support/OqnN9y--ff0/R-Hevkw2BAAJ
# 
# 
# **IMPORTANT NOTE:** RSGISlib installation will create a new Python environment osgeoenv and you'll need to reinstall necessary packages in this environment if you want to use it to continue subsquent works. Other wise, once image segmentation is done, you can switch back to your normal environment to continue following works.

# In[4]:


from rsgislib.segmentation import segutils
import time
def run_image_segmentation(input_image, output_image, num_clusters=60, min_pixels=100, dist_thres=100):
    '''
    This funcation will segment input image and create a mask file where the pixel value is segment id
    '''
    print ('Running segmentation for image "%s" ...' % (input_image))    
    start = time.time()
    segutils.runShepherdSegmentation(input_image, output_image, 
                                     gdalformat='GTiff',
                                     numClusters=num_clusters, 
                                     minPxls=min_pixels, 
                                     noStats=True, ## have to setup this param otherwise there will be an error
                                     distThres=dist_thres,
                                     processInMem=True)
    print ('Segmentation for image "%s" finished in %d seconds.' % (input_image, time.time()-start))


# In[5]:


# imageIds in a DataFrame
all_image_ids = GRID_SIZE.ImageId.unique()

for image_id in all_image_ids:
    input_image = '../input/sixteen_band/'+image_id+'_M.tif'
    output_image = '../input/segment/'+image_id+'_M_SEG.tif'
    run_image_segmentation(input_image, output_image)


# ## Feature extraction

# In[6]:


import tifffile as tiff
import numpy as np
import warnings
from scipy import stats as sc_stats

def get_poly_features(poly_pixels):
    """For each band, compute: min, max, mean, variance, skewness, kurtosis"""
    features = []
    if len(poly_pixels.shape)<2:
        return None
    n_pixels, n_bands = poly_pixels.shape
    for b in range(n_bands):
        stats = sc_stats.describe(poly_pixels[:,b])
        band_stats = list(stats.minmax) + list(stats)[2:]
        if n_pixels == 1:
            # scipy.stats.describe raises a Warning and sets variance to nan
            band_stats[3] = 0.0  # Replace nan with something (zero)
        features += band_stats
    return features

def segment_to_features(segment_mask, image):
    '''
    Extract features from polygons generated by segmentation.
    '''
    image_shape = image.shape[:2]
    poly_ids = np.unique(segment_mask)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        features = []
        for poly_id in poly_ids:
            poly_pixels = image[segment_mask==poly_id]
            poly_features = get_poly_features(poly_pixels)
            features.append(poly_features)     
    return features, poly_ids 

def image_to_array(image_id,image_type='M'):
    '''
    image to array
    '''
    image_type ='M'
    if image_type =='3':
        image = tiff.imread('../input/three_band/{}.tif'.format(image_id)).transpose([1, 2, 0])
    elif image_type =='M':
        image = tiff.imread('../input/sixteen_band/{}_M.tif'.format(image_id)).transpose([1, 2, 0])
    elif image_type =='A':
        image = tiff.imread('../input/sixteen_band/{}_A.tif'.format(image_id)).transpose([1, 2, 0])
    elif image_type =='P':
        image = tiff.imread('../input/sixteen_band/{}_P.tif'.format(image_id))
#     image = exposure.rescale_intensity(image)
    return image



# ## Extract features for testing images

# In[7]:


## Feature extraction for testing images
for image_id in test_image_ids:
    image = image_to_array(image_id,image_type='M')
    segment_mask = tiff.imread('../input/segment/'+image_id+'_M_SEG.tif')
    start = time.time()

    features, segment_ids = segment_to_features(segment_mask, image)
    feature_df = pd.DataFrame(features
                ,columns=['b1_min','b1_max','b1_mean','b1_variance','b1_skewness','b1_kurtosis',
                          'b2_min','b2_max','b2_mean','b2_variance','b2_skewness','b2_kurtosis',
                          'b3_min','b3_max','b3_mean','b3_variance','b3_skewness','b3_kurtosis',
                          'b4_min','b4_max','b4_mean','b4_variance','b4_skewness','b4_kurtosis',
                          'b5_min','b5_max','b5_mean','b5_variance','b5_skewness','b5_kurtosis',
                          'b6_min','b6_max','b6_mean','b6_variance','b6_skewness','b6_kurtosis',
                          'b7_min','b7_max','b7_mean','b7_variance','b7_skewness','b7_kurtosis',
                          'b8_min','b8_max','b8_mean','b8_variance','b8_skewness','b8_kurtosis'
                         ])
    feature_df['segment_id'] = segment_ids
    feature_df['image_id'] = image_id
    feature_df.to_csv('../input/feature/'+image_id+'.csv',index = False)
    print ('Feature extraction for image %s finished in %d seconds' % (image_id,time.time() - start))


# In[8]:


## Extract features for training images
### Run following commands if required pacakges have not been installed in osgoenv environment

get_ipython().system('source activate osgeoenv;pip install shapely;pip install opencv-python;pip install rasterio')


# In[9]:


## !source activate osgeoenv;pip install shapely
import shapely.wkt
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon
import cv2
from collections import Counter
# make sure rasterio was imported after shapely otherwise it may cause kernel error!!!
import rasterio.features

def get_grid_size(image_id):
    '''
    '''
    x_max = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Xmax.values[0]
    y_min = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Ymin.values[0]
    return x_max, y_min

def get_scalers(image_shape, x_max, y_min):
    '''
    To provide scalers that will be used to scale predicted polygons
    '''
    h, w = image_shape  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1))
    h_ = h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

def image_to_features(image_id,image_type='M', class_type = 0):
    image = image_to_array(image_id)
    image_shape = image.shape[:2]
    x_max, y_min = get_grid_size(image_id)
    x_scaler, y_scaler = get_scalers(image_shape, x_max, y_min)

    start = time.time()
    features = np.zeros((0,48)) ## 48 is the number of features from scipy.stats (6) * number of bands (8)
    labels = np.array([])
    poly_ids = np.array([])
    mask = np.zeros(image_shape)
    if class_type==0: ##all classes
        for cls in CLASSES:
            train_polygons = TRAIN_WKT[(TRAIN_WKT['ImageId']==image_id) & 
                                       (TRAIN_WKT['ClassType']==cls)].MultipolygonWKT.values[0]
            if train_polygons != 'MULTIPOLYGON EMPTY':
                ## Check if polygons is empty
                train_polygons = shapely.wkt.loads(train_polygons)
                ## Scale polygons based on image grid size
                train_polygons = shapely.affinity.scale(train_polygons,
                                                        xfact=x_scaler,
                                                        yfact=y_scaler,
                                                        origin=(0, 0, 0))
                poly_features, pids, poly_mask = poly_to_features(train_polygons, image, cls)
                features = np.vstack((features, poly_features))
                labels = np.hstack((labels, np.full((len(poly_features)),cls, dtype=np.int)))
                poly_ids = np.hstack((poly_ids, pids))
                mask = np.max(np.stack((mask,poly_mask)),axis=0)
    print ("Feature extracted in %d seconds" % (time.time()-start) )        
    return features,labels,poly_ids,mask

def mask_for_polygons(polygons, image_size):
    image_mask = np.zeros(image_size, np.uint8)
    if not polygons:
        return image_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(image_mask, exteriors, 1)
    cv2.fillPoly(image_mask, interiors, 0)
    return image_mask

def poly_to_mask(polygons, image_shape, class_type):
    mask = np.zeros(image_shape, np.uint8)    
    if not polygons:
        return image
    int_coords = lambda x: np.array(x).round().astype(np.int32)

    exteriors = []
    interiors = []
    for pid, poly in enumerate(polygons):
        poly_mask = np.zeros(image_shape, np.uint8)
        exteriors=[int_coords(poly.exterior.coords)]
        interiors = []
        for pi in poly.interiors:
            interiors.append(int_coords(pi.coords) )

        cv2.fillPoly(poly_mask, exteriors, 1)

        cv2.fillPoly(poly_mask, interiors, 0)
        poly_id = (pid + 1) * 100 + class_type
        poly_mask = poly_mask * poly_id
        mask = np.max(np.stack((mask, poly_mask)),axis=0)
    return mask 

def poly_to_features(polygons, image, class_type):
    '''
    Extract features for training image. 
    Polygons are extracted from training WKT then converted to masks.
    Featuers are statistical metrics of each polygon
    '''
    image_shape = image.shape[:2]
    poly_mask = poly_to_mask(polygons, image_shape, class_type)
    poly_ids = np.unique(poly_mask)
    poly_ids = poly_ids[poly_ids != 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        features = []
        for poly_id in poly_ids:
            poly_pixels = image[poly_mask==poly_id]
            poly_features = get_poly_features(poly_pixels)
            features.append(poly_features)     
    return features, poly_ids, poly_mask 

def image_to_train(image_id, class_type, image_type='3'):
    # Get grid size: x_max and y_min
    x_max = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Xmax.values[0]
    y_min = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Ymin.values[0]

    # Load train poly with shapely
    train_polygons = shapely.wkt.loads(TRAIN_WKT[(TRAIN_WKT['ImageId']==image_id) & 
                                                (TRAIN_WKT['ClassType']==class_type)].MultipolygonWKT.values[0])

    # Read image with tiff
    if image_type =='3':
        image = tiff.imread('../input/three_band/{}.tif'.format(image_id)).transpose([1, 2, 0])
    if image_type =='M':
        image = tiff.imread('../input/sixteen_band/{}_M.tif'.format(image_id)).transpose([1, 2, 0])
    if image_type =='A':
        image = tiff.imread('../input/sixteen_band/{}_A.tif'.format(image_id)).transpose([1, 2, 0])
    if image_type =='P':
        image = tiff.imread('../input/sixteen_band/{}_P.tif'.format(image_id))
    image_size = image.shape[:2]
    x_scaler, y_scaler = get_scalers(image_size, x_max, y_min)

    # Scale polygons
    train_polygons_scaled = shapely.affinity.scale(train_polygons,
                                                   xfact=x_scaler,
                                                   yfact=y_scaler,
                                                   origin=(0, 0, 0))

    train_mask = mask_for_polygons(train_polygons_scaled, image_size)
    if image_type =='3':
        X = image.reshape(-1, 3).astype(np.float32)
    if image_type =='M':
        X = image.reshape(-1, 8).astype(np.float32)
    if image_type =='A':
        X = image.reshape(-1, 8).astype(np.float32)
    if image_type =='P':
        X = image
    y = train_mask.reshape(-1)
    return train_mask*class_type

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

def train_image_to_feature(image_id):
    # original image tif
    image = image_to_array(image_id)
    # Segmented training image mask
    image_segment_mask = tiff.imread('../input/segment/'+image_id+'_M_SEG.tif')
    # Training image mask by classes
    image_class_mask = np.max([ image_to_train(image_id,c,'M') for c in CLASSES],axis=0)

    start = time.time()
    # for each segment, set its class as the one which has most pixels
    segment_class_mask=np.zeros(image_segment_mask.shape)
    segment_ids = np.unique(image_segment_mask)
    labels = []
    features = []
    for segment_id in segment_ids:
    #         segment_class_mask[image_segment_mask==segment_id] = 
        # Labels
        labels.append(most_common(image_class_mask[image_segment_mask==segment_id]))
        # Features
        segment_pixels = image[image_segment_mask==segment_id]
        features.append(get_poly_features(segment_pixels))

    return features, labels, segment_ids  


# In[10]:


## Feature extraction for training images
train_image_ids = TRAIN_WKT.ImageId.unique()
for image_id in train_image_ids:
    start = time.time()
    features, class_types, segment_ids= train_image_to_feature(image_id)
    feature_df = pd.DataFrame(features
                ,columns=['b1_min','b1_max','b1_mean','b1_variance','b1_skewness','b1_kurtosis',
                          'b2_min','b2_max','b2_mean','b2_variance','b2_skewness','b2_kurtosis',
                          'b3_min','b3_max','b3_mean','b3_variance','b3_skewness','b3_kurtosis',
                          'b4_min','b4_max','b4_mean','b4_variance','b4_skewness','b4_kurtosis',
                          'b5_min','b5_max','b5_mean','b5_variance','b5_skewness','b5_kurtosis',
                          'b6_min','b6_max','b6_mean','b6_variance','b6_skewness','b6_kurtosis',
                          'b7_min','b7_max','b7_mean','b7_variance','b7_skewness','b7_kurtosis',
                          'b8_min','b8_max','b8_mean','b8_variance','b8_skewness','b8_kurtosis'
                         ])
    feature_df['segment_id'] = segment_ids
    feature_df['image_id'] = image_id
    feature_df['class_type'] = class_types
    feature_df.to_csv('../input/feature_train/'+image_id+'.csv',index = False)
    print ('Feature extraction for image %s finished in %d seconds' % (image_id,time.time() - start))






# ### Load training data

# In[11]:


train_df = pd.DataFrame()
for image_id in train_image_ids:
#     print (image_id)
    train_df = pd.concat([train_df, pd.read_csv('../input/feature_train/'+image_id+'.csv')],axis=0)


# In[12]:


full_cols = train_df.columns.tolist()
full_cols.remove('segment_id')
full_cols.remove('image_id')
full_cols.remove('class_type')

target = 'class_type'


# ## Training

# In[13]:


from sklearn.model_selection import train_test_split
import xgboost as xgb

start = time.time()
clf = xgb.XGBClassifier(n_estimators=720, learning_rate = 0.1, max_depth=5)
clf.fit(train_df[full_cols].values,train_df[target].values)
print (time.time()-start)


# ## Prediction

# In[14]:


from matplotlib import pyplot as plt
from matplotlib import colors


def pixels_to_poly(image,mask):
    poly=[]
    for vec in rasterio.features.shapes(image,mask):
        poly.append(shapely.geometry.geo.shape(vec[0]))
    poly = MultiPolygon(poly)
    return poly

def predict_image(image_id, clf, full_cols,plot_image = False):
    test_df = pd.read_csv('../input/feature/'+image_id+'.csv')


    test_x = test_df[full_cols].values
    test_segment_id = test_df['segment_id'].values

    pred_test_segment_y = clf.predict(test_x)
    start = time.time()
    test_segment_mask = tiff.imread('../input/segment/{}_M_SEG.tif'.format(image_id))
    test_pred_mask = np.zeros(test_segment_mask.shape,dtype=np.int32)

    for pid, cls in zip(test_segment_id,pred_test_segment_y):
        test_pred_mask[test_segment_mask==pid] = np.int(cls)
    
    if plot_image:
        cmap = colors.ListedColormap([COLORS.get(c,'1') for c in np.unique(test_pred_mask)])
        plt.imshow(test_pred_mask, interpolation='none',cmap=cmap)
    return test_pred_mask


# In[15]:


# Make predictions
test_df = pd.DataFrame()
preds = []
for image_id in test_image_ids:
    print ("Predicting image ",image_id)
    start = time.time()    
    # Make predictions - pixels
    pred_image = predict_image(image_id,clf,full_cols, plot_image = False)

    image_shape = pred_image.shape[:2]
    x_max = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Xmax.values[0]
    y_min = GRID_SIZE[GRID_SIZE['ImageId']==image_id].Ymin.values[0]    

    x_scaler, y_scaler = get_scalers(image_shape, x_max, y_min)   

    ##Generate polygons from pixels
    for cls in CLASSES:
        polygons = pixels_to_poly(pred_image,pred_image==cls)
        # Scale polygons
        polygons = shapely.affinity.scale(polygons, xfact=1 / x_scaler, yfact=1 / y_scaler, origin=(0, 0, 0))
        preds.append([image_id,cls,polygons])
    print ("Predictions for image finishend in %d seconds" % (time.time()-start))
        
    


# ## Make submission

# In[16]:


preds_df=pd.DataFrame(preds, columns = ['ImageId','ClassType','MultipolygonWKT'])

## Convert polygon precision - this is to reduce the volume of submission data as well as chance of errors for submission 

preds_df['MultipolygonWKT'] = preds_df['MultipolygonWKT'].\
apply(lambda x:x.simplify(0.0001, preserve_topology=False)).\
apply(lambda x:x if x.is_valid else x.buffer(0)).\
apply(lambda x:shapely.wkt.dumps(x,rounding_precision=5))   


## This step is to ensure output will have the same sequence as sample submission
output_df = pd.merge(sample_submission[['ImageId','ClassType']],preds_df, how = 'left', on = ['ImageId','ClassType'])

## Final output
output_df[['ImageId','ClassType','MultipolygonWKT']].to_csv("../output/submission.csv", index=False)

