#!/usr/bin/env python
# coding: utf-8

# # SIIM: Step-by-Step Image Detection for Beginners 
# ## Part 1 - EDA to Preprocessing
# 
# 👉 Part 2. [Basic Modeling (The Easiest Model using Keras)](https://www.kaggle.com/songseungwon/siim-covid-19-detection-10-step-tutorial-2)
# 
# 👉 Mini Part. [Preprocessing for Multi-Output Regression that Detect Opacities](https://www.kaggle.com/songseungwon/siim-covid-19-detection-mini-part-preprocess)

# ### Thanks for nice reference : 
# 
# `handling dcm file`
# - [SIIM-FISABIO-RSNA_COVID-19_Detection_Starter, DrCapa](https://www.kaggle.com/drcapa/siim-fisabio-rsna-covid-19-detection-starter)
# 
# `Image Visualization`
# - [2. ONE STOP:Understanding+InDepth EDA+Model](https://www.kaggle.com/harshsharma511/one-stop-understanding-indepth-eda-model-progress)
# 
# `using gdcm without internet access`
# - [pydicom_conda_helper](https://www.kaggle.com/awsaf49/pydicom-conda-helper)
# 
# `Get the box informations`
# - [catch up on positive samples / plot submission.csv](https://www.kaggle.com/yujiariyasu/catch-up-on-positive-samples-plot-submission-csv?scriptVersionId=63394385)

# > Index
# 
# ```
# Step 1. Import Libraries
# Step 2. Load Data
# Step 3. Read DCM File
#      3-a. explore path with python code
#      3-b. make image extractor(function)
# Step 4. Show Sample Image
#      4-a. explore image data with python code
#      4-b. check position to draw box
# Step 5. Show Multiple Images
# Step 6. Feature Engineering I
#      6-a. count opacity
#      6-b. simplify 'id'
#      6-c. rename colume 'id' to 'StudyInstanceUID for merge on 'StudyInstanceUID'
#      6-d. check the relation between 'OpacityCount' and other columes in train_study
#      6-e. visualize the relation between 'OpacityCount' and other columes in train_study
#      6-f. check duplicate values(One row and Two Appearances)
# Step 7. Feature Engineering II
#      7-a. explore data analysis
#      7-b. check duplicates in dataset
#      7-c. modify some of the code in function that extract image(.dcm)
# Step 8. Visualize X-ray with bbox
#      8-a. negative for pneumonia
#      8-b. typical appearance
#      8-c. indeterminate appearance
#      8-d. atypical Appearance
# Step 9. Featrue Engineering III
#      9-a. anomaly detection
#      9-b. show outliers in `Typical Appearance`
#      9-c. show outliers in `Intermiate Appearance`
#      9-d. show outliers in `Atypical Appearance`
# Step 10. Image Data Preprocessing
#      10-a. add image path to a separate column
#      10-b. Resize the image (uniform to 150x150) and Scale each pixel values (uniform range 1~255)
#      10-c. Calculate the resize ratio(x, y) and Apply the same to the bounding box
# ```  

# ## Step 1. Import Libraries

# We need to load the gdcm package before import `pydicom`. because some of the dcm files are `jpeg lossless` type.

# In[1]:


get_ipython().system("wget 'https://anaconda.org/conda-forge/gdcm/2.8.9/download/linux-64/gdcm-2.8.9-py37h500ead1_1.tar.bz2' -q")
get_ipython().system("conda install 'gdcm-2.8.9-py37h500ead1_1.tar.bz2' -c conda-forge -y")


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))


# In[3]:


import matplotlib.pyplot as plt
import matplotlib
import pydicom as dicom
import cv2
import ast
import warnings
warnings.filterwarnings('ignore')


# ## Step 2. Load Data

# In[4]:


path = '/kaggle/input/siim-covid19-detection/'


# In[5]:


os.listdir(path)


# In[6]:


train_image = pd.read_csv(path+'train_image_level.csv')
train_study = pd.read_csv(path+'train_study_level.csv')
sample_submission = pd.read_csv(path+'sample_submission.csv')


# In[7]:


len(sample_submission)


# In[8]:


train_image


# In[9]:


train_study


# ## Step 3. Read DCM File

# ### 3-a. explore path with python code

# In[10]:


temp = train_image.loc[0, 'StudyInstanceUID']
temp


# In[11]:


temp_depth2 = os.listdir(path+'train/'+temp)
temp_depth2[0]


# In[12]:


temp_train_path = path+'train/'+temp+'/'+temp_depth2[0]
temp_train_path


# In[13]:


os.listdir('/kaggle/input/siim-covid19-detection/train/5776db0cec75/81456c9c5423')  


# In[14]:


train_image.loc[0, 'id']


# ### 3-b. make image extractor(function)

# In[15]:


def extraction(i):
    path_train = path + 'train/' + train_image.loc[i, 'StudyInstanceUID']
    last_folder_in_path = os.listdir(path_train)[0]
    path_train = path_train + '/{}/'.format(last_folder_in_path)
    img_id = train_image.loc[i, 'id'].replace('_image','.dcm')
    print(img_id)
    data_file = dicom.dcmread(path_train+img_id)
    img = data_file.pixel_array
    return img


# ## Step 4. Show Sample Image

# ### 4-a. Explore Image Data with python code

# In[16]:


sample_img = extraction(0)


# In[17]:


sample_img


# In[18]:


sample_img.shape


# ### 4-b. check position to draw box

# In[19]:


train_image.loc[0, 'boxes']


# In[20]:


boxes = ast.literal_eval(train_image.loc[0, 'boxes'])
boxes


# In[21]:


fig, ax = plt.subplots(1,1, figsize=(8,4))
for box in boxes:
    p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                      box['width'], box['height'],
                                      ec='r', fc='none', lw=1.5)
    ax.add_patch(p)
ax.imshow(sample_img, cmap='gray')
plt.show()


# ## Step 5. Show Multiple Images

# In[22]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()

for row in range(9):
    img = extraction(row)
    # if (nan == nan)
    # False
    if (train_image.loc[row,'boxes'] == train_image.loc[row,'boxes']):
        boxes = ast.literal_eval(train_image.loc[row,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='r', fc='none', lw=2.
                                            )
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_image.loc[row, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])


# ## Step 6. Feature Engineering I

# In[23]:


train_image


# ### 6-a. Count Opacity in Image

# In[24]:


OpacityCount = train_image['label'].str.count('opacity')
OpacityCount


# In[25]:


train_image['OpacityCount'] = OpacityCount.values


# In[26]:


train_image


# In[27]:


train_image['id'].isnull().sum()


# ### 6-b. Simplify 'id' (study)

# In[28]:


id_extract = lambda x : x[0]


# In[29]:


train_study


# In[30]:


train_study['id'].isnull().sum()


# In[31]:


train_study['id'].str.split('_')


# In[32]:


train_study['id'].str.split('_').apply(id_extract)


# In[33]:


train_study['id'] = train_study['id'].str.split('_').apply(id_extract)


# In[34]:


sum(train_study['id'].str.contains(train_image['StudyInstanceUID'][0]))


# ### 6-c. rename colume 'id' to 'StudyInstanceUID for merge on 'StudyInstanceUID'

# In[35]:


train_study = train_study.rename({'id':'StudyInstanceUID'}, axis=1)


# In[36]:


train_study


# In[37]:


train_df = pd.merge(train_image, train_study, on='StudyInstanceUID')
train_df


# ### 6-d. Check the Relation between 'OpacityCount' and other Columes in train_study

# In[38]:


train_df['OpacityCount'].value_counts()


# In[39]:


train_df.iloc[:,5:].columns


# In[40]:


i = 5
for col in train_df.iloc[:,5:].columns:
    print('The Count of {} : '.format(col), sum(train_df.iloc[:,i]))
    i += 1


# In[41]:


train_df[train_df['OpacityCount'] == 0]


# In[42]:


OCount = sorted(list(train_df['OpacityCount'].value_counts().index))
print(OCount)


# In[43]:


for count in OCount:
    print('Opacity Count = {}\n------------------------------'.format(count))
    print(train_df[train_df['OpacityCount'] == count].iloc[:,5:].sum())
    print(' ')


# ### 6-e. Visualize the Relation between 'OpacityCount' and other Columes in train_study

# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


for count in OCount:
    Count_Series = train_df[train_df['OpacityCount'] == count].iloc[:,5:].sum()
    fig = plt.figure(figsize=(12,3))
    sns.barplot(x=Count_Series.index, y=Count_Series.values/sum(train_df['OpacityCount']==count))
    plt.title('OpacityCount : {} '.format(count))
    plt.plot();


# ### 6-f. Check Duplicate Values(One row and Two Appearances)

# In[46]:


sum(train_df['OpacityCount']==1)


# In[47]:


train_df[(train_df['OpacityCount']==1)&(train_df['Indeterminate Appearance'] == 1)]


# In[48]:


train_df[(train_df['OpacityCount']==1)&(train_df['Atypical Appearance'] == 1)]


# In[49]:


train_df[(train_df['OpacityCount']==1)&(train_df['Typical Appearance'] == 1)]


# In[50]:


len(train_df[(train_df['OpacityCount']==1)&(train_df['Indeterminate Appearance'] == 1)]) + len(train_df[(train_df['OpacityCount']==1)&(train_df['Atypical Appearance'] == 1)]) + len(train_df[(train_df['OpacityCount']==1)&(train_df['Typical Appearance'] == 1)])


# In[51]:


sum(train_df['OpacityCount']==1)


# In[52]:


sample_submission


# ## Step 7. Feature Engineering II

# ### 7-a. explore data analysis

# The number of `StudyInstanceUID` in train_study(original id) is different from the number of `StudyInstanceUID` in train_df(==train_image)
# 
# Let's check them

# In[53]:


train_study


# In[54]:


train_image


# In[55]:


train_df


# In[56]:


len(train_df['StudyInstanceUID'])


# In[57]:


len(train_df['StudyInstanceUID'].unique())


# ### 7-b. Check duplicates in dataset

# We can find that No duplicates in train_study(original ID) because the length of unique `StudyInstanceUID` in train_df and the length of train_study's rows are the same

# In[58]:


train_image['StudyInstanceUID'].unique().sort() == train_study['StudyInstanceUID'].unique().sort()


# In[59]:


len(train_image['StudyInstanceUID'].unique())


# Naturally, because `train_df` is a merged data frame based on `train_image`, `train_image` has also the same result.

# Now, Let's check duplicated images(id)

# In[60]:


train_image[train_image.duplicated(['StudyInstanceUID'])==True]['StudyInstanceUID']


# In[61]:


du_StudyId = train_image[train_image.duplicated(['StudyInstanceUID'])==True]['StudyInstanceUID'].values


# In[62]:


du_images = train_image[train_image['StudyInstanceUID'].isin(du_StudyId)].sort_values(by=['StudyInstanceUID'])
du_images


# ```
# 232 original ID
# 280 duplicate ID
# ```
# 
# - one or more duplicate Image at the same ID

# ### 7-c. modify some of the code in function that extract image(.dcm)

# So, Some of the code(function extraction) needs to be modified to accurately target the image to be extracted.

# Duplicate ID - ex. 1 ID(74ba8f2badcb) - 4 Path(in each path, All 4 Images are the same)

# In[63]:


train_df[train_df['StudyInstanceUID'].str.contains('74ba8f2')]


# In[64]:


os.listdir(path + 'train/' + '74ba8f2badcb')


# In[65]:


long_path = path + 'train/' + '74ba8f2badcb/'
for i in os.listdir(long_path):
    print(os.listdir(long_path+i))


# In[66]:


os.listdir('/kaggle/input/siim-covid19-detection/train/ff0879eb20ed/d8a644cc4f93')


# 
# Search all paths through a loop and check whether it matches the id value.

# In[67]:


def error_processed_extraction(i):
    long_path = path + 'train/' + train_df.loc[i, 'StudyInstanceUID'] + '/'
    img_id = train_df.loc[i, 'id'].replace('_image','.dcm')
    for dcm in os.listdir(long_path):
        dcm_path = long_path+dcm+'/'
        if img_id == os.listdir(dcm_path)[0]:
            data_file = dicom.dcmread(dcm_path+img_id)
            print('index : {} - DCM File Path :{}'.format(i, dcm_path+img_id))
        else:
            continue
            
    img = data_file.pixel_array
    return img


# ## 8. Visualize X-ray with bbox

# In[68]:


OpacityType = list(train_df.iloc[:,5:].columns)
OpacityType


# In[69]:


train_df[train_df[OpacityType[0]]==1]


# ### 8-a. Negative for Pneumonia

# In[70]:


Negative_Idx = list(train_df[train_df[OpacityType[0]]==1].index)
Negative_Idx[:9]


# In[71]:


train_df.iloc[Negative_Idx, :]


# In[72]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Negative_Idx[:9]:
    img = error_processed_extraction(idx)
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='r', fc='none', lw=2.
                                            )
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(str(train_df.loc[idx, 'label'].split(' ')[0])+ str(idx))
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ### 8-b. Typical Appearance

# In[73]:


Typical_Idx = list(train_df[train_df[OpacityType[1]]==1].index)
Typical_Idx[:9]


# In[74]:


train_df.iloc[Typical_Idx, :]


# In[75]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Typical_Idx[:9]:
    img = error_processed_extraction(idx)
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='r', fc='none', lw=2.
                                            )
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ### 8-c. Indeterminate Appearance

# In[76]:


Indeterminate_Idx = list(train_df[train_df[OpacityType[2]]==1].index)
Indeterminate_Idx[:9]


# In[77]:


train_df.iloc[Indeterminate_Idx, :]


# In[78]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Indeterminate_Idx[:9]:
    img = error_processed_extraction(idx)
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='b', fc='none', lw=2.
                                            )
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ### 8-d. Atypical Appearance

# In[79]:


Atypical_Idx = list(train_df[train_df[OpacityType[3]]==1].index)
Atypical_Idx[:9]


# In[80]:


train_df.iloc[Atypical_Idx, :]


# In[81]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Atypical_Idx[:9]:
    img = error_processed_extraction(idx)
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='g', fc='none', lw=2.
                                            )
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ## 9. Feature Engineering III

# 
# Outliers detected through above visualizations. let's check them

# In[82]:


train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1)]


# **Anomaly 304 rows : label is 'none' but Non Negative for Pneumonia**

# ### 9-a. anomaly detection
# 
# - Cases with no opacity detected but classified as symptomatic

# In[83]:


i=0
fig, axes = plt.subplots(nrows=1,ncols=3, figsize=(12,4))
for type in OpacityType[1:]:
    sr = train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1)].loc[:,type].value_counts()
    sns.barplot(x=sr.index, y=sr.values, ax=axes[i])
    axes[i].set_title(type)
    i += 1


# In[84]:


Anom_Count = train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1)][OpacityType[1:]].sum()
Anom_Count


# In[85]:


plt.figure(figsize=(8,4))
sns.barplot(x=Anom_Count.index, y=Anom_Count.values)
plt.title('Count of "label==none"')
plt.show()


# ### 9-b. Show Outliers in `Typical Appearance`

# In[86]:


train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Typical Appearance']==1)].head()


# In[87]:


Outlier_Typical_Idx = list(train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Typical Appearance']==1)].index)
Outlier_Typical_Idx[:6]


# In[88]:


fig, axes = plt.subplots(2,3, figsize=(20,10))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Outlier_Typical_Idx[:6]:
    img = error_processed_extraction(idx)

    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ### 9-c. Show Outliers in `Indeterminate Appearance`

# In[89]:


train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Indeterminate Appearance']==1)].head()


# In[90]:


Outlier_Indeterminate_Idx = list(train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Indeterminate Appearance']==1)].index)
Outlier_Indeterminate_Idx[:6]


# In[91]:


fig, axes = plt.subplots(2,3, figsize=(20,10))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Outlier_Indeterminate_Idx[:6]:
    img = error_processed_extraction(idx)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ### 9-d. Show Outliers in `Atypical Appearance`

# In[92]:


train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Atypical Appearance']==1)].head()


# In[93]:


Outlier_Atypical_Idx = list(train_df[(train_df['OpacityCount']==0) & (train_df['Negative for Pneumonia']!=1) & (train_df['Atypical Appearance']==1)].index)
Outlier_Atypical_Idx[:6]


# In[94]:


fig, axes = plt.subplots(2,3, figsize=(20,10))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in Outlier_Atypical_Idx[:6]:
    img = error_processed_extraction(idx)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# ## Step 10. Image Data Preprocessing

# ### 10-a. Add image path to a separate column

# In[95]:


for _, row in train_df.iloc[:5].iterrows():
    print(row)


# In[96]:


from glob import glob


# In[97]:


for _, row in train_df.iloc[:5].iterrows():
    image_id = row['id'].split('_')[0]
    study_id = row['StudyInstanceUID']
    img_path = glob(f'{path}/train/{study_id}/*/{image_id}.dcm')
    print(img_path)


# In[98]:


path_list = []
for _, row in train_df.iterrows():
    image_id = row['id'].split('_')[0]
    study_id = row['StudyInstanceUID']
    img_path = glob(f'{path}/train/{study_id}/*/{image_id}.dcm')
    if len(img_path)==1:
        path_list.append(img_path[0])
    else:
        print(img_path)


# In[99]:


len(path_list)


# In[100]:


path_list[:10]


# In[101]:


train_df['Path'] = path_list


# All Image path have been saved (6334)

# In[102]:


train_df


# We use this dataframe at Part 2. Let's save this to csv file.

# In[103]:


train_df.to_csv('train_df.csv')


# ### 10-b. Resize the image (uniform to 150x150) and Scale each pixel values (uniform range 1~255)

# In[104]:


data_file = dicom.read_file(path_list[0])


# In[105]:


data_file.pixel_array


# In[106]:


data_file


# In[107]:


data_file.Rows


# In[108]:


data_file.Columns


# In[109]:


# Test function, We don't use this
def extract_img_size(path_list):
    origin_img_heights = []
    origin_img_widths = []
    i = 0
    for path in path_list:
        data_file = dicom.read_file(path)
        origin_img_heights.append(data_file.Rows)
        origin_img_widths.append(data_file.Columns)
        i += 1
        if i % 100 == 0:
            print('{}/{}'.format(i,len(path_list)))
            
    return origin_img_heights, origin_img_widths


# In[110]:


origin_img_heights, origin_img_widths = extract_img_size(path_list[:10])


# In[111]:


origin_img_heights


# In[112]:


origin_img_widths


# In[113]:


# We use this function
import cv2
def extract_resized_and_origin_img_info(path_list):
    img_list = []
    origin_img_heights = []
    origin_img_widths = []
    i = 0
    for path in path_list:
        data_file = dicom.read_file(path)
        img = data_file.pixel_array

            
        origin_img_heights.append(img.shape[0])
        origin_img_widths.append(img.shape[1])

        
        # scailing to 0~255
        img = (img - np.min(img)) / np.max(img)
        img = (img * 255).astype(np.uint8)
        
        # resizing to 4000+ to 150 default
        img = cv2.resize(img, (150,150))
        img_list.append(img)
        img_array = np.array(img_list)
        i += 1
        if i % 100 == 0:
            print('{} / {}'.format(len(img_array),len(path_list)))
    return img_array, origin_img_heights, origin_img_widths


# In[114]:


test_imgs, origin_img_heights2, origin_img_widths2 = extract_resized_and_origin_img_info(path_list[:10])


# In[115]:


test_imgs.shape


# In[116]:


test_imgs[0].shape


# In[117]:


test_imgs[0]


# In[118]:


print('pixel range : ', '{} ~ {}'.format(min(test_imgs[0].reshape(-1)),max(test_imgs[0].reshape(-1))))


# In[119]:


origin_img_heights2


# In[120]:


origin_img_widths2


# In[121]:


print((origin_img_heights == origin_img_heights2), (origin_img_widths == origin_img_widths2))


# exactly same result on two function

# ### 10-c. Calculate the resize ratio(x, y) and Apply the same to the bounding box

# In[122]:


x_scale_list=[]
y_scale_list=[]
if len(origin_img_heights) == len(origin_img_widths):
    for i in range(len(origin_img_heights)):
        x_scale = 150 / origin_img_widths[i]
        x_scale_list.append(x_scale)
        print(i)
        y_scale = 150 / origin_img_heights[i]
        y_scale_list.append(y_scale)


# In[123]:


x_scale_list


# In[124]:


y_scale_list


# Before vs after resizing

# In[125]:


fig, axes = plt.subplots(3,3, figsize=(20,16))
fig.subplots_adjust(hspace=.1, wspace=.1)
axes = axes.ravel()
row = 0
for idx in range(9):
    img = error_processed_extraction(idx)
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x'], box['y']),
                                              box['width'], box['height'],
                                              ec='b', fc='none', lw=2.)
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1


# In[126]:


fig, axes = plt.subplots(3,3, figsize=(16,16))
fig.subplots_adjust(hspace=.1, wspace=.05)
axes = axes.ravel()
row = 0
for idx in range(9):
    img = test_imgs[idx]
    # if (nan == nan)
    # False
    if (train_df.loc[idx,'boxes'] == train_df.loc[idx,'boxes']):
        boxes = ast.literal_eval(train_df.loc[idx,'boxes'])
        for box in boxes:
            p = matplotlib.patches.Rectangle((box['x']*x_scale_list[idx], box['y']*y_scale_list[idx]),
                                              box['width']*x_scale_list[idx], box['height']*y_scale_list[idx],
                                              ec='b', fc='none', lw=2.)
            axes[row].add_patch(p)
    
    axes[row].imshow(img, cmap='gray')
    axes[row].set_title(train_df.loc[idx, 'label'].split(' ')[0])
    axes[row].set_xticklabels([])
    axes[row].set_yticklabels([])
    row += 1

