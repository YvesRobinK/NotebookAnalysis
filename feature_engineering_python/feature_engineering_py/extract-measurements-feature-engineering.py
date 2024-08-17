#!/usr/bin/env python
# coding: utf-8

# # Extract Measurements 
# 
# ---
# 
# Around 20% of the products in the dataset contain some kind of measurement (24mm Tape, 980gr powder, 1 kg coffee). 
# 
# A King size bed is 76” x 80” and is also known as an Eastern King bed. A California King bed, marketed toward taller people, is 72” x 84”. If you can extract measurements, you can differentiate between 100 gram coffee vs 250 gm coffee, 8 pack coke vs 6 pack coke, or 32 GB iphone vs 64 GB iphone. Over 6000 items in the listing have at least one measurement in them.
# 

# ### If you fork or use this notebook, do leave an upvote!

# In[1]:


import pandas as pd 

train = pd.read_csv('../input/shopee-product-matching/train.csv')
label_to_images = train.groupby('label_group').posting_id.unique().to_dict()
train['target'] = train.label_group.apply(lambda label: label_to_images[label])
train.sample(10)


# In[2]:


import re 
def is_measurement(word): 
    measurement_cats = ['kg', 'g', 'cm', 'pcs', 'gb', 'ml', 'mm', 'gr', 'gram']
    for m in measurement_cats: 
        pat = '(\d+)' + m + ''
        res = re.findall(pat, word)
        if res != []: 
            return f' {m} '.join(res) + ' ' + m

    for m in measurement_cats: 
        pat = '(\d+) ' + m + ''
        res = re.findall(pat, word)
        if res != []: 
            return f' {m} '.join(res) + ' ' + m
        
        
    return False 

train['measurement'] = train.title.apply(is_measurement)
num_products_with_measurement = len(train) - (train.measurement == False).sum()
print(f'{num_products_with_measurement} products containing measurements found!')


# #### If you have any suggestions or improvements, do comment below. This is still primitive code and misses some measurements. Also don't forget to upvote!

# In[ ]:




