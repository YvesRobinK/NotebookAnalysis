#!/usr/bin/env python
# coding: utf-8

# # BACTERIA AS IMAGE?
# Let's imagine that datset is a image classification problem. Each bacteria is represented as image. How bacteria will look like? How NN deal with this problem? Let's try and make experiment ...
# 
# In this notebook you will find:
# * How to convert tabular data into images and redefine problem
# * Keras neural network hybrid model - 2DConvolution (images) and Dense (additional tabular features)
# * NN model cross validation using StratifiedKFold - to evalueate model locally 
# * NN feature maps visualizations
# * Grad-CAM - show parts of images (bacteria) which influence the choice of the class
# * Simplex model prediction optimizations - to find best weights for class 
# * Another way of looking on tabular data (as images or NLP problem)
# * Tricks with Magic Functions (conditional cell run) - could be helpful and used instead of if function
# * Replacing target class in test dataset based on duplicates in train dataset
# 
# <div align="center"><img src="https://i.ibb.co/ydZdXCX/BACT-001.jpg" width=800/></div>
# <div align="center"><img src="https://i.ibb.co/CtK0YFg/oryg-grad.jpg" width=800/></div>

# In[1]:


get_ipython().run_cell_magic('capture', '', '\nimport pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport gc\nimport os\nimport time \n\nfrom scipy.stats import mode\nfrom sklearn.preprocessing import LabelEncoder\nfrom sklearn.model_selection import train_test_split, StratifiedKFold\nfrom sklearn.preprocessing import MinMaxScaler, OneHotEncoder\nfrom sklearn.metrics import accuracy_score\n\nimport keras\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Dropout, Flatten\nfrom keras.layers import Conv2D, MaxPooling2D, BatchNormalization\nfrom keras import backend as K\nfrom keras.utils import np_utils\nfrom keras.utils.vis_utils import plot_model\nimport matplotlib.pyplot as plt\nimport tensorflow as tf\n\nimport matplotlib.cm as cm\nfrom IPython.display import Image, display\n\nimport warnings\nwarnings.filterwarnings("ignore")\n')


# ### CUTOM MAGIC FUNCTION
# I implemented magic function to skip execution of notebook cell. It can be helpful for your other notebooks.

# In[2]:


from IPython.core.magic import (register_line_cell_magic)

@register_line_cell_magic
def skip(line, cell=None):
    if eval(line):
        print("Cell skipped - not executed")
        return
    get_ipython().ex(cell)


# In[3]:


# notebook configuration 

cfg = {
    'TARGET' : 'target',
    'N_FOLDS' : 10,
    'RANDOM': 42,
    'SCORING': 'accuracy',
    'PROD': True
}


# ## DATA PREPARATION

# In[4]:


test = pd.read_csv("../input/tabular-playground-series-feb-2022/test.csv", index_col = 'row_id')
train = pd.read_csv("../input/tabular-playground-series-feb-2022/train.csv", index_col = 'row_id')
sub_df = pd.read_csv("../input/tabular-playground-series-feb-2022/sample_submission.csv")

train.drop_duplicates(keep='first', inplace=True)
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

target = train.target
num_classes = target.nunique()

FEATURES = [col for col in train.columns if col not in ['target']]


# In[5]:


# This dataset I create only for NN filter and Grad-CAM visualization
scaler = MinMaxScaler()
train_df = pd.DataFrame(scaler.fit_transform(train[FEATURES]), columns = FEATURES)
test_df = pd.DataFrame(scaler.transform(test[FEATURES]), columns = FEATURES)

# Target feature encoder
lb = LabelEncoder()
enc_target = lb.fit_transform(target)
train_df['target_enc'] = enc_target
train['target_enc'] = enc_target
train_df['target'] = target


# ## CONVERT TABULAR DATA TO IMAGES

# In[6]:


# We creatively convert each obserwation to 2D (1 channel) image
img_rows = 13
img_cols = 22


# ## SHOW BACTERIA "SPECTROGRAM" BY CLASS
# - Do you see any patterns in images?
# - Are bacterias mixed or shifted?

# In[7]:


def plot_bacteria(images, labels, indexes):
    num_row = 10
    num_col = 5

    fig, axes = plt.subplots(num_row, num_col, constrained_layout=True,  sharex=True, sharey=True, figsize=(3*num_col,2*num_row))

    for i in range(len(images)):
        ax = axes[i//num_col, i%num_col]
        image = images[i].reshape(img_rows, img_cols, 1)
        ax.imshow(image, cmap='Spectral')
        ax.set_title(f'{labels[i]}\n{indexes[i]}')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.5)
    plt.show()


# In[8]:


# Plot bacteria representants (4x 10) 

images = train_df.groupby("target").sample(5).drop(["target_enc"], axis = 1)
labels = images.target.values
indexes = images.index.values
images = images[FEATURES].values.reshape(images.shape[0], img_rows, img_cols, 1).astype('float32')

plot_bacteria(images, labels, indexes)


# ## LETS'S LOOK INTO PATTERNS

# In[9]:


train_df.target.value_counts()


# In[10]:


# Plot first 50 Bacteroides_fragilis         

images = train_df.query("target == 'Campylobacter_jejuni'").sort_index()[:50].drop(["target_enc"], axis = 1)
labels = images.target.values
indexes = images.index.values
images = images[FEATURES].values.reshape(images.shape[0], img_rows, img_cols, 1).astype('float32')

plot_bacteria(images, labels, indexes)


# In[11]:


# Plot first 50 observations       

images = train_df.sort_index()[:50].drop(["target_enc"], axis = 1)
labels = images.target.values
indexes = images.index.values
images = images[FEATURES].values.reshape(images.shape[0], img_rows, img_cols, 1).astype('float32')

plot_bacteria(images, labels, indexes)


# ## CREATE NN ARCHITECTURE (USING KERAS SEQUENTIAL)

# In[12]:


batch_size = 512
epochs = 100

input_shape = (img_rows, img_cols, 1)


# In[13]:


get_ipython().run_cell_magic('capture', '', "\ndef get_model():\n    model = Sequential()\n    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='swish', input_shape=input_shape, kernel_initializer='he_uniform')\n    model.add(x)\n    model.add(Conv2D(64, (3, 3), padding='same', activation='swish', kernel_initializer='he_uniform'))\n    model.add(Dropout(0.2))\n    model.add(Conv2D(128, (3, 3), padding='same', activation = 'swish', kernel_initializer='he_uniform', name='conv_last'))\n    model.add(MaxPooling2D((2, 2)))\n    model.add(Dropout(0.2))\n    model.add(Flatten())\n    model.add(Dense(128, activation='swish', kernel_initializer='he_uniform'))\n    model.add(Dropout(0.2))\n    model.add(Dense(64, activation='swish', kernel_initializer='he_uniform'))\n    model.add(Dense(32, activation='swish', kernel_initializer='he_uniform'))\n    model.add(Dense(num_classes, activation='softmax'))\n\n    model.compile(loss=keras.losses.categorical_crossentropy,\n                  optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),\n                  metrics=['accuracy'])\n    \n    return model, x\n")


# ## PLOT MODEL

# In[14]:


model_global, _ = get_model()


# In[15]:


plot_model(model_global, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# ## MODEL VALIDATION (k-FOLD) AND PREDICTION
# 
# The most important think in experimentation with NN architecture and feature engineering is to make good validation procedure. I use StratifiedKFold validation loop. I foud this better then train_test split approach. This attitude give us many advantages: 
# * you can validate model on whole dataset (no test/train bias)
# * you can customize prediction for further preprocessing (in this notebook I use Simple optimization)

# In[16]:


# FOLD validation
oof_preds = np.zeros((len(train_df), 1))
oof_proba = np.zeros((len(train_df), num_classes))
test_proba = np.zeros((len(test), num_classes))
test_preds = []
fold_scores = []
x = None #This is only for NN filter and Grad-CAM visualizations

kf = StratifiedKFold(n_splits = cfg['N_FOLDS'], random_state = cfg['RANDOM'])
print(f"Start CV model - {cfg['N_FOLDS']} for folds")

for fold, (train_idx, val_idx) in enumerate(kf.split(train, train.target_enc)):
    X_train, y_train = train[FEATURES].iloc[train_idx], train[['target_enc']].iloc[train_idx]
    X_val, y_val = train[FEATURES].iloc[val_idx], train[['target_enc']].iloc[val_idx]
    
    ## in fold scaling
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test[FEATURES].values)
    
    # in fold data transformation -> convert to images
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')
    
    # in fold target one hot encoding 
    enc = OneHotEncoder(categories = 'auto', drop = None, sparse = False)
    y_train = enc.fit_transform(y_train)
    y_val = enc.transform(y_val) 
    
    model, conv = get_model()
    
    es = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.00001, 
                                          patience = 6, verbose = 0, mode = 'max', baseline=None, 
                                          restore_best_weights=True)

    plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', 
                                                   factor = 0.04, patience = 5, 
                                                   verbose = 0, mode = 'max')
    
    history = model.fit(X_train, y_train,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 0,
                    validation_data = (X_val, y_val),
                    callbacks = [es, plateau])
    
    val_score = model.evaluate(X_val, y_val, verbose=0)
    print(f"   FOLD: {fold + 1} - accuracy on val set: ", val_score[1])
    
    fold_scores.append(history)
    oof_preds[val_idx, 0] = np.argmax(model.predict(X_val), axis = -1)
    oof_proba[val_idx, :] = model.predict(X_val)
    
    test_preds.append(np.argmax(model.predict(X_test), axis = -1))
    test_proba[: , :] += model.predict(X_test)
    
    # Save last model for filter/grad-CAM visualization
    if fold == cfg['N_FOLDS'] - 1:
        model_global = model
        x = conv
    else:
        del model, val_score, es, plateau
        gc.collect()

test_proba = test_proba / cfg['N_FOLDS']
oof_score = accuracy_score(train_df.target_enc.values, oof_preds)
print(f"OOF accuracy: {oof_score}")


# In[17]:


# 2022.02.10 - BASELINE (5 first fold - 10f cv) 
# FOLD: 1 - accuracy on val set:  0.935967743396759
# FOLD: 2 - accuracy on val set:  0.9441128969192505
# FOLD: 3 - accuracy on val set:  0.946854829788208
# FOLD: 4 - accuracy on val set:  0.9544318318367004
# FOLD: 5 - accuracy on val set:  0.9666101932525635


# ## PLOT LEARNING HISTORY

# In[18]:


import matplotlib.pyplot as plt

for fold in range(len(fold_scores)):
    history_f = fold_scores[fold]

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(14,4))
    fig.suptitle('Fold : '+str(fold), fontsize=14)
        
    plt.subplot(1,2,1)
    plt.plot(history_f.history['loss'], label= ['loss'])
    plt.plot(history_f.history['val_loss'], label= ['val_loss'])
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('loss')
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(history_f.history['accuracy'], label= ['accuracy'])
    plt.plot(history_f.history['val_accuracy'], label= ['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid()


# ## SHOW MODEL SAMPLE CONV2D FILTERS

# In[19]:


filters, biases = x.get_weights()
conv_weight = filters[:,:,0,:]

# Check the shape of first Conv2D layer
print(f'First conv2D shape: {filters.shape}')
print(f'First conv2D output size: {x.output.shape} \n')

plt.figure(figsize = (10,10))
print("First 16 filters of conv2D layer")
for i in range(1,17):
    plt.subplot(4,4,i)
    plt.imshow(conv_weight[:,:,i], interpolation='nearest', cmap='summer', aspect='auto')

plt.show()


# ## PLOT IMAGES FROM FIRST CONV2D LAYER  

# In[20]:


bacteria_sample = train_df[FEATURES].iloc[5].values.reshape(1, img_rows, img_cols, 1).astype('float32')


# In[21]:


from numpy import expand_dims
from keras.models import Model

#img = expand_dims(bacteria_samples[1], axis=0)
# Then hijacked output from first layer
model_first2D = Model(inputs=model.inputs, outputs=x.output)

# Made prediction of first sample
feature_maps = model_first2D.predict(bacteria_sample)

# Plot all (32) images from our conv2D layer 
plt.figure(figsize = (40,20))
square = 8
ix = 1
for _ in range(4):
    for _ in range(square):
        ax = plt.subplot(square, square, ix)
        plt.imshow(feature_maps[0, :, :, ix-1], cmap='cool', interpolation='nearest')
        ix += 1
plt.show()


# ## SIMPLEX OPTIMIZATIONS
# 
# The Nelder-Mead simplex method uses a simplex to traverse the space in search of a minimum. A simplex is a generalization of a tetrahedron to n-dimensional space. A simplex in one dimension is a line, and in two dimensions it is a triangle. The simplex derives its name from the fact that it is the simplest possible polytope in any given space. Source: Algorithms for Optimization (Mykel J. Kochenderfer, Tim A. Wheeler)

# In[22]:


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(oof_proba[0:10])


# In[23]:


from scipy.optimize import minimize

def inter_class_optimizer(weights, oof_preds_opt):
    oof = oof_preds_opt * weights
    oof = np.argmax(oof, axis = -1)
    y_val = enc_target
    return (1 - accuracy_score(y_val, oof))


def pred_fold_optimizer(oof_preds_opt, test_preds_opt):   
    
    print(f"OOF ACCURACY score BEFORE optimization: {accuracy_score(enc_target, np.argmax(oof_preds_opt, axis = -1))}")
    
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    
    bon = [(0.0, 1.0)] * 10
    
    res = minimize(fun = inter_class_optimizer,
                   x0 = [1/10 for _ in range(10)],
                   args = oof_preds_opt,
                   method = 'Nelder-Mead',
                   options = {'maxiter': 500, 'maxfev': 500},
                   bounds = bon,
                   constraints = cons)
    
    print(f"   Class scaler: {res.x}")
    
    oof_preds_opt = np.array([res.x[i] * oof_preds_opt[ :, i] for i in range(10)]).transpose()
    test_preds_opt = np.array([res.x[i] * test_preds_opt[ :, i] for i in range(10)]).transpose()
    
    print(f"OOF ACCURACY score AFTER optimization: {accuracy_score(enc_target, np.argmax(oof_preds_opt, axis = -1))}")
    print('Status : %s' % res['message'])
    print('Total Evaluations: %d' % res['nfev'])

    solution = res['x']
    evaluation = inter_class_optimizer(solution, oof_preds_opt)
    print('Solution: f(%s) = %.5f' % (solution, evaluation))
    return res["fun"], oof_preds_opt, test_preds_opt 


# In[24]:


res, _, optim_preds = pred_fold_optimizer(oof_proba, test_proba)


# ## MAKE PREDICTION AND SUBMISSION

# In[25]:


preds = lb.inverse_transform(mode(test_preds).mode[0])
pd.Series(preds, index=test_df.index).value_counts().sort_index() / len(test) * 100


# In[26]:


sub_df.target = preds
sub_df.to_csv("TPS02-Conv2D-BASE-submission.csv", index=False)
sub_df.head(10)


# ## MAKE SIMPLEX OPTIMIZED SUBMISSION

# In[27]:


_, _, optim_preds = pred_fold_optimizer(oof_proba, test_proba)


# In[28]:


preds = lb.inverse_transform(np.argmax(optim_preds, axis = -1))
pd.Series(preds, index=test_df.index).value_counts().sort_index() / len(test) * 100


# In[29]:


sub_df.target = preds
sub_df.to_csv("TPS02-Conv2D-SIMPLEX-submission.csv", index=False)
sub_df.head(10)


# ## FIX SUBMISSION PREDICTION WITH TEST DATA
# Part of data in training is included in test dataset so we can fix model prediction using oryginal data. It should improve our score a little bit.

# In[30]:


train.index.name = 'row_id'
train = train.reset_index(drop = False)

test.index = sub_df.index
test.index.name = 'row_id'
test = test.reset_index(drop = False)

s1 = pd.merge(train, test, how='inner', on=FEATURES)


# In[31]:


counter = 0
dic = {}
for i in range(len(s1)):
    dic[s1.loc[i]['row_id_y']] = s1.loc[i]['row_id_x']

for e in dic:
    sub_df.loc[sub_df[sub_df.index==e].index.to_list(),'target'] = \
    train.loc[train[train.index==dic[e]].index.tolist()[0],'target']
    counter +=1
print(f"Changed {counter} rows!")
    
pd.Series(sub_df.target, index=test_df.index).value_counts().sort_index() / len(test) * 100


# In[32]:


sub_df.to_csv("TPS02-Conv2D-SIM-TRAIN-submission.csv", index=False)


# # Grad-CAM class activation visualization
# 
# Based on Keras "[Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/)" Let's see which part of image is responsible for class assignment.

# In[33]:


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):

    img = np.squeeze(img_path, axis=0)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img


# In[34]:


train_df.target[7:16]


# ### Let's print some bacteria from dataset

# In[35]:


bacteria_sample_set = train_df[FEATURES][7:16]
bacteria_sample_set = bacteria_sample_set.values.reshape(len(bacteria_sample_set), img_rows, img_cols, 1).astype('float32')


# In[36]:


plt.figure(figsize = (16,8))
for i, bacteria in enumerate(bacteria_sample_set):
    plt.subplot(3,3,i+1)
    plt.imshow(np.squeeze(expand_dims(bacteria, axis=0), axis=0), interpolation='nearest', cmap='Spectral', aspect='auto')
plt.show()


# ### Now we can see grad-cam heatmaps

# In[37]:


plt.figure(figsize = (16,8))
for i, bacteria in enumerate(bacteria_sample_set):
    plt.subplot(3,3,i+1)
    img = expand_dims(bacteria, axis=0)
    preds = model_global.predict(img)
    heatmap = make_gradcam_heatmap(img, model, "conv_last")
    plt.imshow(heatmap, interpolation='nearest', cmap='Spectral', aspect='auto')
plt.show()


# ### And final products .... grad-CAM heatmaps on bacteria images
# 
# Now we can see which part of bacteria image is important for network to recognize appropriate class

# In[38]:


model.layers[-1].activation = None
plt.figure(figsize = (16,8))
for i, bacteria in enumerate(bacteria_sample_set):
    plt.subplot(3,3,i+1)
    img = expand_dims(bacteria, axis=0)
    preds = model_global.predict(img)
    heatmap = make_gradcam_heatmap(img, model, "conv_last")
    grad_cam = save_and_display_gradcam(img, heatmap)
    plt.imshow(grad_cam, interpolation='nearest', cmap='Spectral', aspect='auto')
plt.show()

