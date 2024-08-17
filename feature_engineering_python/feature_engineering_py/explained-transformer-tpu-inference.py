#!/usr/bin/env python
# coding: utf-8

# # Explained Transformer (TPU) inference 💨
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Stanford Ribonanza RNA Folding to edicts the structures of any RNA molecule.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of greySnow
# , available at [this Kaggle project](https://www.kaggle.com/code/shlomoron/srrf-transformer-tpu-inference/notebook). I extend my gratitude to greySnow
#  for sharing their insights and code.
# 
# 🌟 Explore my profile and other public projects, and don't forget to share your feedback! 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍
# 
# ## Purpose 🎯
# The primary purpose of this notebook is to:
# - Load and preprocess the competition data 📁
# - Engineer relevant features for model training 🏋️‍♂️
# - Train predictive models to make target variable predictions 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook is structured as follows:
# 1. **Data Preparation**: In this section, we load and preprocess the competition data.
# 2. **Feature Engineering**: We generate and select relevant features for model training.
# 3. **Model Training**: We train machine learning models on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 
# 
# ## How to Use 🛠️
# To use this notebook effectively, please follow these steps:
# 1. Ensure you have the competition data and environment set up.
# 2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
# 
# **Note**: Make sure to replace any placeholder paths or configurations with your specific information.
# 
# ## Acknowledgments 🙏
# We acknowledge the Stanford University organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 
# 

# ## 📦 Import necessary libraries

# In[1]:


import tensorflow as tf  # Import TensorFlow library for machine learning tasks
import numpy as np      # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import pickle           # Import Pickle for saving and loading data
import shutil           # Import shutil for file operations
import math             # Import math for mathematical operations
import pandas as pd     # Import Pandas for data manipulation
import gc               # Import gc for garbage collection
import os               # Import os for interacting with the operating system


# # Model

# In[2]:


DEBUG = False


# **Cell 3**
# 
# **Explaination** :
# 
# This code defines two custom layers for use in a transformer-based neural network model. 
# 
# 1. `class transformer_block(tf.keras.layers.Layer):`
#    - This line defines a custom layer class called `transformer_block` that inherits from `tf.keras.layers.Layer`.
# 
# 2. `def __init__(self, dim, num_heads, feed_forward_dim, rate=0.1):`
#    - This is the constructor method for the `transformer_block` class. It takes four arguments: `dim` (dimension), `num_heads`, `feed_forward_dim`, and `rate`. `dim` represents the dimension of the model, `num_heads` is the number of attention heads, `feed_forward_dim` is the dimension of the feed-forward network, and `rate` is the dropout rate (with a default value of 0.1).
# 
# 3. `super().__init()`
#    - This line calls the constructor of the parent class (`tf.keras.layers.Layer`) to initialize the layer.
# 
# 4. `self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)`
#    - This line creates a Multi-Head Attention layer with the specified number of heads (`num_heads`) and key dimension (`key_dim`) based on the provided values.
# 
# 5. `self.ffn = tf.keras.Sequential([...])`
#    - This line defines a feed-forward neural network as a sequential model. It consists of two dense layers with ReLU activation functions.
# 
# 6. `self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)`
#    - This creates a layer normalization layer with a small epsilon value.
# 
# 7. `self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)`
#    - Another layer normalization layer is created, similar to the previous one.
# 
# 8. `self.dropout1 = tf.keras.layers.Dropout(rate)`
#    - This defines a dropout layer with the specified dropout rate.
# 
# 9. `self.dropout2 = tf.keras.layers.Dropout(rate)`
#    - A second dropout layer is created.
# 
# 10. `self.supports_masking = True`
#     - This line sets the `supports_masking` attribute to `True`, indicating that the layer can handle input masks.
# 
# 11. `def call(self, inputs, training, mask):`
#     - This method is the core of the layer. It defines how the layer processes input data. It takes three arguments: `inputs` (the input data), `training` (a boolean indicating whether the model is in training mode), and `mask` (an optional mask for input data).
# 
# 12. `att_mask = tf.expand_dims(mask, axis=-1)`
#     - This line expands the dimension of the mask tensor along the last axis.
# 
# 13. `att_mask = tf.repeat(att_mask, repeats=tf.shape(att_mask)[1], axis=-1)`
#     - The mask tensor is repeated along the last axis to match the shape of the input data.
# 
# 14. `attn_output = self.att(inputs, inputs, attention_mask = att_mask)`
#     - Multi-head attention is applied to the input data using the attention mask.
# 
# 15. `attn_output = self.dropout1(attn_output, training=training)`
#     - The output of the attention layer is passed through the first dropout layer.
# 
# 16. `out1 = self.layernorm1(inputs + attn_output)`
#     - Layer normalization is applied to the sum of the input and the attention output.
# 
# 17. `ffn_output = self.ffn(out1)`
#     - The feed-forward neural network is applied to the output of the previous layer.
# 
# 18. `ffn_output = self.dropout2(ffn_output, training=training)`
#     - The output of the feed-forward network is passed through the second dropout layer.
# 
# 19. `return self.layernorm2(out1 + ffn_output)`
#     - Layer normalization is applied to the sum of the output from the previous layer and the feed-forward network's output, and the final result is returned.
# 
# The second class, `positional_encoding_layer`, is similar in structure but implements positional encoding. It also has an `__init__` method for initialization and a `call` method for processing inputs. It calculates and adds positional encodings to the input data, which is a crucial component in transformer models for handling sequence data.
# 
# 

# In[3]:


class transformer_block(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim//num_heads)
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.supports_masking = True

    def call(self, inputs, training, mask):
        att_mask = tf.expand_dims(mask, axis=-1)
        att_mask = tf.repeat(att_mask, repeats=tf.shape(att_mask)[1], axis=-1)

        attn_output = self.att(inputs, inputs, attention_mask = att_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class positional_encoding_layer(tf.keras.layers.Layer):
    def __init__(self, num_vocab=5, maxlen=500, hidden_dim=384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_emb = self.positional_encoding(maxlen-1, hidden_dim)
        self.supports_masking = True

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        x = tf.math.multiply(x, tf.math.sqrt(tf.cast(self.hidden_dim, tf.float32)))
        return x + self.pos_emb[:maxlen, :]

    def positional_encoding(self, maxlen, hidden_dim):
        depth = hidden_dim/2
        positions = tf.range(maxlen, dtype = tf.float32)[..., tf.newaxis]
        depths = tf.range(depth, dtype = tf.float32)[np.newaxis, :]/depth
        angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
        angle_rads = tf.linalg.matmul(positions, angle_rates)
        pos_encoding = tf.concat(
          [tf.math.sin(angle_rads), tf.math.cos(angle_rads)],
          axis=-1)
        return pos_encoding


# **Cell 4**
# 
# **Explaination** :
# 
# This code defines a function `get_model` that creates a transformer-based neural network model for some natural language processing or sequence-related task. 
# 
# 1. `X_max_len = 457` and `num_vocab = 5`: These are defined as global variables to specify the maximum sequence length and the number of vocabulary items.
# 
# 2. The `get_model` function is defined with several arguments, including `hidden_dim` and `max_len` (with default values of 384 and 206, respectively).
# 
# 3. `inp = tf.keras.Input([None])`: This line creates a Keras input layer with an undefined sequence length. It is used to define the input shape for the model.
# 
# 4. `x = inp`: The variable `x` is set to the input layer, which will be used for building the model.
# 
# 5. `x = tf.keras.layers.Embedding(num_vocab, hidden_dim, mask_zero=True)(x)`: An embedding layer is added, which converts integer-encoded inputs to dense vectors of dimension `hidden_dim`. The `mask_zero=True` argument indicates that the padding values will be masked.
# 
# 6. `x = positional_encoding_layer(num_vocab=num_vocab, maxlen=500, hidden_dim=hidden_dim)(x)`: This line adds a positional encoding layer to the input data. The positional encoding helps the model understand the positions of elements in the input sequence.
# 
# 7. Several transformer blocks are added to the model. Each block is defined using the `transformer_block` class that you provided earlier.
# 
#    - `x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)`: This line adds a transformer block with the specified parameters to the model. You have multiple transformer blocks stacked in the model.
# 
# 8. After all transformer blocks, a dropout layer is added: `x = tf.keras.layers.Dropout(0.5)(x)`. This dropout layer helps prevent overfitting by randomly setting a fraction of the input units to 0 during training.
# 
# 9. `x = tf.keras.layers.Dense(2)(x)`: A dense layer with 2 units is added to the model. This is typically used for classification tasks, where you want to produce an output with two classes.
# 
# 10. Finally, a Keras model is created: `model = tf.keras.Model(inp, x)`, which specifies the input and output layers of the model.
# 
# The `get_model` function essentially assembles a transformer-based neural network model with embedding, positional encoding, multiple transformer blocks, dropout, and a final dense layer for classification or regression. The number of transformer blocks and other hyperparameters can be adjusted by calling this function with different arguments when creating a model.

# In[4]:


X_max_len = 457
num_vocab = 5

def get_model(hidden_dim = 384, max_len = 206):
    inp = tf.keras.Input([None])
    x = inp

    x = tf.keras.layers.Embedding(num_vocab, hidden_dim, mask_zero=True)(x)
    x = positional_encoding_layer(num_vocab=num_vocab, maxlen=500, hidden_dim=hidden_dim)(x)

    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)

    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)

    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)
    x = transformer_block(hidden_dim, 6, hidden_dim*4)(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2)(x)

    model = tf.keras.Model(inp, x)
    return model


# **Cell 5**
# 
# **Explaination** :
# 
# 
# The `.head()` method is often used to quickly inspect the contents of the DataFrame and get an overview of the data it contains. It displays the first 5 rows by default, but you can pass a different number as an argument to display a different number of rows. In your case, it's displaying the first 5 rows to provide a glimpse of the data in `test_sequences_df`.

# In[5]:


test_sequences_df = pd.read_csv('/kaggle/input/stanford-ribonanza-rna-folding/test_sequences.csv')
test_sequences_df.head()


# **Cell 6**
# 
# **Explaination** :
# 
# Converting RNA sequences from the `test_sequences_df` DataFrame into a numerical representation using the `encoding_dict`. 
# 1. `test_sequences = test_sequences_df.sequence.to_numpy()`: This line extracts the 'sequence' column from the `test_sequences_df` DataFrame and converts it to a NumPy array. This will give you an array of RNA sequences that you can work with.
# 
# 2. `encoding_dict = {'A': 1, 'C': 2, 'G': 3, 'U': 4}`: Here, you define a dictionary called `encoding_dict`. This dictionary maps the RNA bases 'A', 'C', 'G', and 'U' to the numerical values 1, 2, 3, and 4, respectively. This is a common encoding for converting RNA sequences into a numerical format where each base is represented by an integer.
# 
# 3. `encoding_dict`: This line is not doing anything other than displaying the content of the `encoding_dict` dictionary. It's a way to show the mapping between RNA bases and their numerical representations in the output.
# 
# After executing these lines, you have the `test_sequences` array with RNA sequences, and you can use `encoding_dict` to convert the bases into numerical values for further processing, such as feeding them into a neural network or performing other computations that require numerical input.
# 

# In[6]:


test_sequences = test_sequences_df.sequence.to_numpy()
encoding_dict = {'A':1, 'C': 2, 'G': 3, 'U': 4}
encoding_dict


# **Cell 7**
# 
# **Explaination** :
# 
# Encoding RNA sequences from the `test_sequences` array using the `encoding_dict` and padding them to a maximum length (`max_len`) with zeros. 
# 
# 1. `max_len = 457`: This line sets the maximum length to which you want to pad the RNA sequences.
# 
# 2. `test_sequences_encoded = []`: This initializes an empty list called `test_sequences_encoded`, where you will store the encoded and padded sequences.
# 
# 3. The code uses a for loop to process each sequence in the `test_sequences` array:
# 
#    ```
#    for seq in test_sequences:
#    ```
# 
# 4. Inside the loop, the following operations are performed to encode and pad each sequence:
#    - `np.asarray([encoding_dict[x] for x in seq])`: This list comprehension encodes each character in the RNA sequence by looking up the corresponding value in the `encoding_dict`. It creates a NumPy array of integers.
#    - `np.zeros((max_len - len(seq)))`: This generates a NumPy array of zeros with a length equal to the difference between the maximum length (`max_len`) and the length of the current RNA sequence.
#    - `np.concatenate(...)`: This concatenates the encoded sequence and the zero-padding to reach the desired maximum length.
#    - `.astype(np.float32)`: The concatenated array is cast to a NumPy array with a data type of `float32`. This might be necessary if you plan to use the sequences in numerical computations.
# 
# 5. The resulting encoded and padded sequence is appended to the `test_sequences_encoded` list:
# 
#    ```
#    test_sequences_encoded.append(...)
#    ```
# 
# After the loop is complete, `test_sequences_encoded` will contain all the RNA sequences, each encoded as a NumPy array with a length of `max_len`, padded with zeros. 

# In[7]:


max_len = 457 
test_sequences_encoded = []
for seq in test_sequences:
    test_sequences_encoded.append(
        np.concatenate([np.asarray([encoding_dict[x] for x in seq]), np.zeros((max_len - len(seq)))]).astype(np.float32))


# **Cell 8**
# 
# **Explaination** :
# 
# Creating a TensorFlow Dataset (`test_ds`) from the encoded and padded RNA sequences, setting batch size, and preparing the dataset for efficient processing.
# 
# 1. `test_ds = tf.data.Dataset.from_tensor_slices(test_sequences_encoded)`: This line creates a TensorFlow Dataset from the list of encoded and padded RNA sequences (`test_sequences_encoded`). Each element of the dataset is a NumPy array representing an RNA sequence.
# 
# 2. `batch_size = 256`: You define the batch size as 256. This is the number of sequences that will be processed together during training or evaluation.
# 
# 3. `if DEBUG:`: It seems that you're checking if a variable `DEBUG` is set, and if it is, you reduce the dataset size and batch size for debugging purposes.
# 
#    - `test_ds = test_ds.take(8)`: If `DEBUG` is `True`, the dataset is limited to the first 8 elements.
#    - `batch_size = 2`: The batch size is set to 2 in debugging mode.
# 
# 4. `test_ds = test_ds.padded_batch(batch_size, padding_values=(0.0), padded_shapes=([max_len]), drop_remainder=False)`: This line creates a padded batch of the dataset with the specified batch size. It pads sequences with zeros to match the specified maximum length (`max_len`) and sets `drop_remainder` to `False`, which means the last batch may have fewer elements if the total number of sequences is not a multiple of the batch size.
# 
# 5. `test_ds = test_ds.prefetch(tf.data.AUTOTUNE)`: This line prefetches data to improve training performance. It uses the `tf.data.AUTOTUNE` option to automatically determine the optimal number of elements to prefetch.
# 
# 6. `batch = next(iter(test_ds))`: This line retrieves the next batch from the dataset by creating an iterator and using `next()`. The `batch` variable now holds a batch of RNA sequences.
# 
# 7. `batch.shape`: This line checks the shape of the `batch`, which should be `(batch_size, max_len)` if everything is set up correctly. It confirms that the batch size and sequence lengths match your expectations.
# 
# 

# In[8]:


test_ds = tf.data.Dataset.from_tensor_slices(test_sequences_encoded)
batch_size = 256
if DEBUG:
    test_ds = test_ds.take(8)
    batch_size = 2
#test_ds = test_ds.take(10000)

test_ds = test_ds.padded_batch(batch_size, padding_values=(0.0), padded_shapes=([max_len]), drop_remainder=False)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
batch = next(iter(test_ds))
batch.shape


# **Cell 9**
# 
# **Explaination** :
# 
# 
# 1. `model = get_model(hidden_dim=192, max_len=max_len)`: You create an instance of a deep learning model using the `get_model` function. You pass the `hidden_dim` and `max_len` as arguments, setting the `hidden_dim` to 192 and `max_len` to the previously defined maximum sequence length.
# 
# 2. `model.load_weights('/kaggle/input/srrf-transformer-tpu-training/weights/model_epoch_199.h5')`: You load the pre-trained weights for the model from the specified file path. This is assuming that you've previously trained the model and saved its weights to this location.
# 
# 3. `model(batch)`: You pass the `batch` of data through the model for inference. This line effectively performs a forward pass of the model on the input data.
# 
# 4. `model.summary()`: You print a summary of the model's architecture, showing the layers, the number of parameters, and other relevant information.
# 
# Here's what each of these lines does in a bit more detail:
# 
# - The first line creates an instance of your model with specific hyperparameters.
# - The second line loads pre-trained weights, which can be useful for transfer learning or inference.
# - The third line runs a batch of data through the model, obtaining predictions or output.
# - The fourth line prints a summary of the model's structure and layer configurations.
# 
# 

# In[9]:


model = get_model(hidden_dim = 192,max_len = max_len)
model.load_weights('/kaggle/input/srrf-transformer-tpu-training/weights/model_epoch_199.h5')
model(batch)
model.summary()


# In[10]:


preds = model.predict(test_ds)


# **Cell 11**
# 
# **Explaination** :
# 
# 1. `preds_processed = []`: You initialize an empty list called `preds_processed` to store the processed predictions.
# 
# 2. You use a for loop to iterate over the predictions and sequences:
#    ```
#    for i, pred in enumerate(preds):
#    ```
#    - `i` represents the index, and `pred` is a prediction for a sequence.
# 
# 3. Inside the loop, you do the following:
#    - `preds_processed.append(pred[:len(test_sequences[i])])`: For each prediction `pred`, you slice it to match the length of the corresponding RNA sequence `test_sequences[i]`. This ensures that the predictions are aligned with their respective sequences. The sliced predictions are then appended to the `preds_processed` list.
# 
# 4. After the loop, you use `np.concatenate` to concatenate the processed predictions into a single NumPy array:
#    ```
#    concat_preds = np.concatenate(preds_processed)
#    ```
#    - `concat_preds` will contain all the processed predictions, aligned with the original RNA sequences.
# 
# 

# In[11]:


preds_processed = []
for i, pred in enumerate(preds):
    preds_processed.append(pred[:len(test_sequences[i])])
concat_preds = np.concatenate(preds_processed)


# **Cell 12**
# 
# **Explaination** :
# 
# In this code, you are preparing a submission DataFrame based on the model's predictions and saving it to a CSV file. Let's go through each line of code:
# 
# 
# The 'submission.csv' file will contain the 'id' column and two prediction columns, 'reactivity_DMS_MaP' and 'reactivity_2A3_MaP', which can be submitted as part of a competition .

# In[12]:


submission = pd.DataFrame({'id':np.arange(0, len(concat_preds), 1), 'reactivity_DMS_MaP':concat_preds[:,1], 'reactivity_2A3_MaP':concat_preds[:,0]})
submission.to_csv('submission.csv', index=False)
submission.head()


# ## Explore More! 👀
# Thank you for exploring this notebook! If you found this notebook insightful or if it helped you in any way, I invite you to explore more of my work on my profile.
# 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Feedback and Gratitude 🙏
# We value your feedback! Your insights and suggestions are essential for our continuous improvement. If you have any comments, questions, or ideas to share, please don't hesitate to reach out.
# 
# 📬 Contact me via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# I would like to express our heartfelt gratitude for your time and engagement. Your support motivates us to create more valuable content.
# 
# Happy coding and best of luck in your data science endeavors! 🚀
# 
