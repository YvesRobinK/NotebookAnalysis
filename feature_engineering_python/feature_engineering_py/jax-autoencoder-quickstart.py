#!/usr/bin/env python
# coding: utf-8

# # JAX Autoencoder Playground
# 
# This notebook creates an autoencoder in JAX, using best practices around random keys. We read in features created from other Vendekagon Labs notebooks that enrich our small molecule information, one-hot encode our cell type and small molecule mechanism of action, and then train the autoencoder on the full dimensionality of the resulting dataset.
# 
# Once we have the autoencoder trained, we use it to train basic linear regression models. We validate that we can use these to cross-predict cell types, when trained on other cell types that are present in our dataset.
# 
# We then train linear models on the few B cell and Myeloid lineage cells provided, one for each cell type in the `de_train` data. We then use these models to make out-of-fold predictions on the remaining B cells and Myeloid cells for gene perturbation outcomes. After that, we average these together into one DataFrame, and use that as our submission.
# 
# _NOTE_: this notebook deliberately avoids doing any rigorous cross-validation, hyperparameter search, or much other neural network tuning. Instead we make a best guess about how to provide a little regularization and not train too long. It's intended to be used as a starting point, showing a very simple implementation and training loop, so as to support more elaborate follow-on work.

# For some reason, TPU instances don't always start with the parquet reading libraries available, so we do a just-in-case pip install.

# In[1]:


get_ipython().run_cell_magic('capture', '', '%pip install pyarrow fastparquet\n')


# In[2]:


import jax
import flax
import pandas as pd
from pathlib import Path
from jax import numpy as jnp
from jax import random
from flax import linen as nn
import optax
from flax.training import train_state


# # Supplementary Datasets
# 
# We read in supplemental compound information from these notebooks:
# 
# - [Come on Chemicals!](https://www.kaggle.com/code/vendekagonlabs/come-on-chemicals-r-version/data)
# - [Supplemental Compound Info](https://www.kaggle.com/code/vendekagonlabs/supplemental-compound-info/data)
# 
# _NOTE_: this notebook makes no claim these are the best way to encode our chemical features (they almost certainly aren't), but simply uses these as a starting example for how to incorporate feature engineering before using our AE model.

# In[3]:


data_dir = Path('/kaggle/input/') 
comp_dir = Path(data_dir / 'open-problems-single-cell-perturbations')
de_train_path = comp_dir / 'de_train.parquet'
finger_features_path = '/kaggle/input/come-on-chemicals-r-version/finger_features.csv'
supp_compound_path = '/kaggle/input/supplemental-compound-info/compounds.tsv'
working_dir = Path('/kaggle/working/')


# In[4]:


de_train = pd.read_parquet(de_train_path)
supp_compound = pd.read_csv(supp_compound_path, sep='\t')
finger_features = pd.read_csv(finger_features_path)


# In[5]:


moa = pd.get_dummies(supp_compound['moa'])
sm_moa = supp_compound[['sm_name']].join(moa).drop_duplicates('sm_name')
sm_moa


# # Preprocessing
# 
# ## Adding Features
# 
# I've rolled up all the feature additions into one function, so that it's easy to use this as a starting point to incorporate your own feature engineering. This makes it easy to get started with an AE model by simply just changing this function, then running the rest of the notebook.

# In[6]:


def add_features(df):
    one_hot = df.join(pd.get_dummies(df['cell_type']), how='left')
    add_chem  = one_hot.merge(finger_features, on='sm_name', how='left')
    one_hot_moa = add_chem.join(pd.get_dummies(supp_compound['moa']), how='left')
    return one_hot_moa

de_train_feats = add_features(de_train)


# # Extracting and Scaling Training Data
# 
# In this case, we train the AE using only the four cell types which we have (mostly) complete gene perturbation information for. Later, we'll use cross-prediction across these types as a proxy task to evaluate how to set model parameters for fitting a simple model to our AE's embeddings.

# In[7]:


train_cells = ['NK cells', 'T cells CD4+', 'T regulatory cells', 'T cells CD8+']

train = de_train_feats[de_train_feats['cell_type'].isin(train_cells)]
X_pre = train[train.columns[5:]].values

print(f"Training data shape: {X_pre.shape}")


# In[8]:


train.head(10)


# In[9]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(X_pre)
X = ss.transform(X_pre)


# # Creating our Autoencoder
# 
# Here we define the autoencoder using `flax.linen` layers, a functional style JAX wrapping library that provides more high-level routines (some which overlapy with `pytorch`, some more similar to wrapping libs in that ecosystem like `pytorch-lightning`).
# 
# We define the Encoder and Decoder separately, and for the quickstart example, we use a very basic Autoencoder class to wrap these, which just calls the two in sequence, returning the reconstructed output.

# In[10]:


class Encoder(nn.Module):
    c_hid : int
    latent_dim : int
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dropout(rate=0.10, deterministic=not self.training)(x)
        x = nn.Dense(features=2*self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.latent_dim)(x)
        return x
    
    
class Decoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=2*self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=2*self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.c_out)(x)
        x = nn.tanh(x)
        return x

    
class AutoEncoder(nn.Module):
    c_hid: int
    latent_dim : int
    input_dim: int
    training: bool

    def setup(self):
        self.encoder = Encoder(c_hid=self.c_hid,
                               latent_dim=self.latent_dim,
                               training=self.training)
        self.decoder = Decoder(c_hid=self.c_hid,
                               latent_dim=self.latent_dim,
                               c_out=self.input_dim, training=self.training)

    def __call__(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# # Training Regime
# 
# One way our JAX approach differs from other NN training packages, is that it decouples our model objects from the state of training. Here we use the flax `TrainState` helper to group the state of our training run as properties on a single object.
# 
# It also makes random keys required arguments, and provides a lot of functions for creating additional random keys when need be, by using `split` on our existing keys. This way we can link up the random components of our model so that it will be fully reproducible in a deterministic way, while also specifying that some of our random state needs to be independent from the rest.
# 
# Apart from that, we use an all caps variable convention as is common in other notebooks to define the model & training regime's hyperparameters (which you're most likely to target with model searches and/or manual adjustments). As a note: a lot of JAX models you'll find elsewhere use a wrapping object or dict and a  `config.property` or `config["property"]` convention for settings these.
# 
# In this case, we use a very simple mean squared error metric as our training loss. There are other ways to define reconstruction loss, and we'll use some of these in the _Bonus: VAE_ section of the notebook.

# In[11]:


# ------ model & training hyperparams ------
LATENT_DIM = 512
HIDDEN_BASE_DIM = 1024
INPUT_DIM = X.shape[1]  # 19319
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS=200

# ------------------------------------------
rng = random.PRNGKey(0)
main_key, params_key, dropout_key = jax.random.split(key=rng, num=3)
# ------------------------------------------


# ----------- initialize model -------------
ae = AutoEncoder(
    input_dim=INPUT_DIM,
    c_hid=HIDDEN_BASE_DIM,
    latent_dim=LATENT_DIM,
    training=False
)
variables = ae.init(params_key, jnp.ones([BATCH_SIZE, INPUT_DIM]))
state = train_state.TrainState.create(
        apply_fn = ae.apply,
        tx=optax.adam(LEARNING_RATE),
        params=variables['params']
)


# ------------ fns to drive training -------
@jax.jit
def mse(params, x_batched, y_batched):
    def squared_error(x, y):
        pred = ae.apply({'params': params}, x, rngs={'dropout': dropout_key})
        return jnp.inner(y - pred, y - pred) / 2.0
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

@jax.jit
def train_step(
    state: train_state.TrainState, batch: jnp.ndarray
):

    def loss_fn(params):
        # logits = state.apply_fn({'params': params}, batch)
        loss = mse(params, batch, batch)
        return loss

    gradient_fn = jax.value_and_grad(loss_fn)
    loss, grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# The actual training loop is below. We might the `EPOCHS` value here while developing to see if the model looks like it's getting anywhere.
# 
# This is a cartoonishly simple training regime. We're not tracking additional metrics, taking snapshots and using a validation reconstruction loss to trigger early stopping, etc. There are many ways to improve this, deliberately left out to keep this:
# 
# - simple for beginners
# - easy to improve

# In[12]:


from itertools import cycle

EPOCHS = 200

# the jitter term is a crude way to see 
total = X.shape[0]
iters = total // BATCH_SIZE
jitter = cycle(range(total % BATCH_SIZE))

X_jax = jnp.array(X)
loss = 1e6

print(f"Training for {EPOCHS} epochs.")

for e in range(EPOCHS):
    epoch_offset = next(jitter)
    print(f"Epoch: {e}, most recent training loss: {loss}")

    # -- you can uncomment this to permute rows randomly across epochs --
    # random permutation of rows seems to hurt, probably because we lose
    # being distributed evenly across cell types, why we use jitter strategy
    # X_jax = random.permutation(main_key, X_jax, axis=0, independent=False)
    for i in range(iters):
        x0 = i*4 + epoch_offset
        x1 = (i+1)*4 + epoch_offset
        state, loss = train_step(state, X_jax[x0:x1, :])


# # Using our Trained Encoder
# 
# The encoder wouldn't do us much good if we couldn't use us to encode features! We create a function that will let us call the encoder model on data we pass in that's of the same shape as training data. We then use this to encode features for our cell type + chemical + gene perturbation training space.
# 
# _Note_: we have to use the `state.params` here, as again the model weights, as a stateful part of our training, are not a part of our model definition. Instead, this is more like partial function changing, i.e. `model(state.params, arr) => embeddings` as `model.bind(state.params).encoder(arr) => embeddings`.

# In[13]:


@jax.jit
def encode(arr):
    return ae.bind({'params': state.params}).encoder(arr)

embedded = encode(X[:,:])
print(f"Original dims: {X.shape} reduced to {embedded.shape} dimensionality.")


# # Cell Type Cross-Prediction
# 
# We have a big challenge in this dataset, in that we have very limited training samples in `de_train` for our target cell types, B cells and Myeloid cells. We can also see that these cell phenotypes are fairly different from our training data, three of which are subtypes of T cells!
# 
# This notebook uses an oversimplified model selection technique. We use cross-prediction to predict cell types against the other cell types we do have, using this as a proxy task to predict the cells we don't have. We then use that to set a regularization term for basic ridge regression, then use that regularization term in the model we fit for predicting B cells and Myeloid cells. We don't use any sophisticated cross-validation (you could, for instance do k-fold where the prediction task is similarly limited, from the same number of instances of B cells we have, predict all the rest). Left as an exercise for the reader...

# In[14]:


import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


# In[15]:


train_cells


# ## Cross-prediction training details
# 
# We use our autoencoder to train the source/training cell type's data. To ensure we're not overly coupled to the dimensionality reduction learned by the autoencoder, we make our training target a more straight forward dimensionality reduction of the other cell type's gene logfold change space, using `TruncatedSVD`. Note that the `n_components` param was set with naive manual inspection, and could be more rigorously tuned.
# 
# We also have to re-use the `StandardScaler` we used as an input for model training, to match the autoencoder's expected range and distributino of data, then invert our TruncatedSVD transform to get our predictions back into gene expression space, then evaluate our resulting model error there.

# In[16]:


genes = de_train.columns[5:]


def naive_cross_predict(df, cell1='T cells CD8+', cell2='T cells CD4+'):
    cell1_df = df[df['cell_type'] == cell1]
    cell2_df = df[df['cell_type'] == cell2]
    cell1_sms = list(cell1_df['sm_name'].unique())
    cell2_sms = list(cell2_df['sm_name'].unique())
    common = [sm for sm in cell1_sms if sm in cell2_sms]
    print(f"Found {len(common)} sm_names shared between cell types.")
    cell1_sub = (cell1_df[cell1_df['sm_name'].isin(common)]).sort_values(by='sm_name')
    cell2_sub = (cell2_df[cell2_df['sm_name'].isin(common)]).sort_values(by='sm_name')
    Xc1 = cell1_sub[cell1_sub.columns[5:]].values
    y = cell2_sub[genes].values
    tsvd = TruncatedSVD(n_components=40)
    tsvd.fit(y)
    scaled = ss.transform(Xc1)
    embed = encode(jnp.array(scaled))
    y_dr = tsvd.transform(y)
    lm = Ridge(alpha=55)
    lm.fit(embed, y_dr)
    y_dr_hat = lm.predict(embed)
    y_hat = tsvd.inverse_transform(y_dr_hat)
    mse = mean_squared_error(y, y_hat)
    return lm, tsvd, mse


# ## Cross-prediction Loop
# 
# With our training defined, we run it across all combinations of cell types we havei n the data.

# In[17]:


from itertools import combinations

models = {}

for ct1, ct2 in combinations(train_cells, 2):
    lm, tsvd, mse = naive_cross_predict(train, cell1=ct1, cell2=ct2)
    models[f'{ct1} => {ct2}'] = {
        'linear_regression_model': lm,
        'tsvd_trasnform_obj': tsvd,
        'mae': mse
    }
    print(f"Naive cross-prediction from {ct1} => {ct2}, MSE: {mse}")


# ## Predicting B and Myeloid Cell Gene Perturbation
# 
# Now that we've done some test cross-prediction, we read in the data for which we have no gene perturbation provided, to use the same strategy to make predictions. We re-use our same `naive_cross_predict` function from our prior cross-type prediction to fit a model to the data included in `de_train`.

# In[18]:


id_map = pd.read_csv(comp_dir / 'id_map.csv')
id_map


# In[19]:


for ct1 in train_cells:
    for ct2 in ['B cells', 'Myeloid cells']:
        lm, tsvd, mse = naive_cross_predict(de_train_feats, cell1=ct1, cell2=ct2)
        models[f'{ct1} => {ct2}'] = {
            'linear_regression_model': lm,
            'tsvd_trasnform_obj': tsvd,
            'mae': mse
        }
        print(f"Naive cross-prediction from {ct1} => {ct2}, MSE: {mse}")


# This next section is just to show how we transform the ` => ` key to and back from tuples. It's a little bit of extra string munging, but was helpful for me to see the ` => ` prediction ordering while developing this, so I lefti t in.

# In[20]:


import pprint
pp = pprint.PrettyPrinter(indent=1, depth=2)

def to_mapping(s1, s2):
    return f"{s1} => {s2}"

model_key_tuples = [k.split(' => ') for k in models.keys()]
pp.pprint(model_key_tuples)


# In[21]:


[to_mapping(*k) for k in model_key_tuples]


# In[22]:


b_cell_keys = [(src, target) for src, target in model_key_tuples if target == 'B cells']
myeloid_cell_keys = [(src, target) for src, target in model_key_tuples if target == 'Myeloid cells']
[to_mapping(*k) for k in b_cell_keys] + [to_mapping(*k) for k in myeloid_cell_keys]


# ## Using Our Simple Models to Make Submission Predictions
# 
# Using the models we fit to our data above, we make now fit our data. There is some shared logic with our naive cross prediction function here, none of it refactored out. (In this case, it's a little easier to read with each in its own place). A lot of this is just collection manipulation in base Python to ensure we're predicting B cells from the other cell types.
# 
# Note that the strategy being used here is to fit a model for each of our reference cell types --(CD8+, CD4+, regulatory) T cells, NK cells -- then average the predictions. You could also make one big predictive modeling using the information you have on all cells, and thoughtfully built model of that form is likely to do better than the simpler averaging strategy.

# In[23]:


bcell_predictions = {}
bcells = id_map[id_map['cell_type'] == 'B cells']


for mod_key, bcell in b_cell_keys:
    mk = to_mapping(mod_key, bcell)
    lrm = models[mk]['linear_regression_model']
    tsvd = models[mk]['tsvd_trasnform_obj']
    mod_cells_df = train[train['cell_type'] == mod_key]
    mod_sms = list(mod_cells_df['sm_name'].unique())
    bcell_sms = list(bcells['sm_name'].unique())
    common = [sm for sm in mod_sms if sm in bcell_sms]
    print(f"Found {len(common)} sm_names shared between cell types, for {len(bcell_sms)} needed for submission.")
    mod_sub = (mod_cells_df[mod_cells_df['sm_name'].isin(common)]).sort_values(by='sm_name')
    bcell_sub = (bcells[bcells['sm_name'].isin(common)]).sort_values(by='sm_name')
    Xmc = mod_sub[mod_sub.columns[5:]].values
    Xmc = ss.transform(Xmc)
    Xmc = encode(jnp.array(Xmc))
    bcell_gx_pred_dr = lrm.predict(Xmc)
    bcell_gx_pred = tsvd.inverse_transform(bcell_gx_pred_dr)
    bcell_predictions[mod_key] = {
        'pred': bcell_gx_pred,
        'sm_names': bcell_sub['sm_name']
    }


# In[24]:


bcells = id_map[id_map['cell_type'] == 'Myeloid cells']
myeloid_predictions = {}

# just use bcell name again for myeloid cells, fix later
for mod_key, bcell in myeloid_cell_keys:
    mk = to_mapping(mod_key, bcell)
    lrm = models[mk]['linear_regression_model']
    tsvd = models[mk]['tsvd_trasnform_obj']
    mod_cells_df = train[train['cell_type'] == mod_key]
    mod_sms = list(mod_cells_df['sm_name'].unique())
    bcell_sms = list(bcells['sm_name'].unique())
    common = [sm for sm in mod_sms if sm in bcell_sms]
    print(f"Found {len(common)} sm_names shared between cell types, for {len(bcell_sms)} needed for submission.")
    mod_sub = (mod_cells_df[mod_cells_df['sm_name'].isin(common)]).sort_values(by='sm_name')
    bcell_sub = (bcells[bcells['sm_name'].isin(common)]).sort_values(by='sm_name')
    Xmc = mod_sub[mod_sub.columns[5:]].values
    Xmc = ss.transform(Xmc)
    Xmc = encode(jnp.array(Xmc))
    bcell_gx_pred_dr = lrm.predict(Xmc)
    bcell_gx_pred = tsvd.inverse_transform(bcell_gx_pred_dr)
    myeloid_predictions[mod_key] = {
        'pred': bcell_gx_pred,
        'sm_names': bcell_sub['sm_name']
    }


# As a sanity check, we assert that our predictions are the right shape w/r/t the space of gene logfold change.

# In[25]:


assert myeloid_predictions['NK cells']['pred'].shape[1] == len(genes)
assert bcell_predictions['NK cells']['pred'].shape[1] == len(genes)


# Then we average our predictions together.

# In[26]:


from functools import reduce

each_pred = [k for k in bcell_predictions.keys()]
for pred in each_pred:
    bcell_predictions[pred]['df'] = pd.DataFrame(bcell_predictions[pred]['pred'], columns=genes,
                                                 index=bcell_predictions[pred]['sm_names'].index)
    myeloid_predictions[pred]['df'] = pd.DataFrame(myeloid_predictions[pred]['pred'], columns=genes,
                                                   index=myeloid_predictions[pred]['sm_names'].index)
bcell_all = reduce(lambda x, y: x.add(y, fill_value=0.0), [val['df'] for val in bcell_predictions.values()])
bcell_all = bcell_all / 4.0
myeloid_all = reduce(lambda x, y: x.add(y, fill_value=0.0), [val['df'] for val in myeloid_predictions.values()])
myeloid_all = myeloid_all / 4.0


# # Creating Submissions File
# 
# We then concatenate our results into the form expected for submissions.
# 
# _NOTE_: on its own, this very simple use of a very basic autoencoder model hits ~0.72 on the public leaderboard. Like with other notebooks, we ensemble to get a better score.

# In[27]:


result = pd.concat([bcell_all, myeloid_all])

# we use a different copy of the dataframe to munge into
# submissions column form, so we can use `result` later
# in ensembling
result_id = result.copy(deep=True)
result_id['id'] = result_id.index
result_id.to_csv('ae_submission.csv', index=False)


# # Ensembling
# 
# Before ensembling this with the result of simple models and average predictions (what most of the leaderboard and public notebooks do at the moment), we check how our predictions correlate. Remember that we want some evidence that it's only weak correlation! (If it's perfectly anti-correlated the predictions just cancel out. If it's too correlated, we'll increase our combined models' bias, rather than correct for it when averaging.)

# In[28]:


should_be_06_02 = pd.read_csv('/kaggle/input/sep28-0-602-submission-reference/submission.csv')
should_be_06_02


# In[29]:


gene_corr = should_be_06_02.corrwith(result_id)
print(gene_corr.mean(), gene_corr.std())
gene_corr.hist(bins=50)


# In[30]:


new_sub = 0.88*should_be_06_02 + 0.12*result
new_sub


# In[31]:


new_sub['id'] = new_sub.index
new_sub.to_csv('submission.csv', index=False)


# # Bonus: VAE
# 
# A variational auto-encoder, in simple terms, expands on the basic idea of an autoencoder by learning a probability distribution in the latent space/embeddings, rather than just where to place single training points. This is done using the 'reparameterization trick' -- there's a 
# [good video lecture here](https://www.youtube.com/watch?v=iL1c1KmYPM0) that goes into details on latent space models and distributions in general, and uses that as a starting poitn for explaining VAEs.
# 
# I've included a VAE starting point here, with none of the follow-on steps as above, for those who want to explore that model. Note that this model has not been tuned much, though I have provided a cosine learning rate scheduler as a tool for tuning the training regime more carefully.
# 
# Many of the below examples are adapted from the JAX and Flax official documentation.

# In[32]:


@jax.jit
def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


class VAEncoder(nn.Module):
    c_hid : int
    latent_dim : int
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dropout(rate=0.1, deterministic=not self.training)(x)
        x = nn.Dense(features=2*self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        mean_x = nn.Dense(self.latent_dim, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latent_dim, name='fc2_logvar',
                            kernel_init=jax.nn.initializers.variance_scaling(
                                scale=0.25, distribution='truncated_normal', mode='fan_in'
                            ))(x)
        return mean_x, logvar_x
    
    
class VADecoder(nn.Module):
    c_out : int
    c_hid : int
    latent_dim : int
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=2*self.c_hid)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=0.25, deterministic=not self.training)(x)
        x = nn.Dense(features=self.c_out)(x)
        x = nn.tanh(x)
        return x

    
class VAE(nn.Module):
    latent_dim: int
    c_hid: int
    input_dim: int
    training: bool

    def setup(self):
        self.encoder = VAEncoder(c_hid=self.c_hid,
                                 latent_dim=self.latent_dim,
                                 training=self.training)
        self.decoder = VADecoder(c_hid=self.c_hid,
                                 latent_dim=self.latent_dim,
                                 c_out=self.input_dim, training=self.training)

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar


# In[33]:


def create_learning_rate_fn(n_epochs, base_learning_rate, steps_per_epoch, warmup_epochs=5):
    warmup_fn = optax.linear_schedule(
          init_value=0., end_value=base_learning_rate,
          transition_steps=1 * steps_per_epoch)
    cosine_epochs = max(n_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
          init_value=base_learning_rate,
          decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, cosine_fn],
          boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


# In[34]:


import functools

# ------ model & training hyperparams ------
LATENT_DIM = 1024
HIDDEN_BASE_DIM = 1024
INPUT_DIM = X.shape[1]  # 19319
BATCH_SIZE = 16
INIT_LEARNING_RATE = 5e-5
# ------------------------------------------
rng = random.PRNGKey(0)
main_key, params_key, dropout_key, z_key = jax.random.split(key=rng, num=4)
# ------------------------------------------


# ----------- initialize model -------------
vae = VAE(
    input_dim=INPUT_DIM,
    c_hid=HIDDEN_BASE_DIM,
    latent_dim=LATENT_DIM,
    training=False
)

# ------------ fns to drive training -------
# note that we need a more complex loss function, including
# both KL divergence and binary cross-entropy.
# ------------------------------------------
@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))

@functools.partial(jax.jit, static_argnums=3)
def train_step(state, batch, z_rng, learning_rain_fn):
    def loss_fn(params):
        recon_x, mean, logvar = vae.apply({'params': params}, batch, z_rng)
        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        loss = bce_loss + kld_loss
        return loss, recon_x

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    lr = learning_rate_fn(state.step)
    metrics = {'learning_rate': lr,
               'loss': loss}
    return state, metrics


# In[35]:


from itertools import cycle

EPOCHS = 20

total = X.shape[0]
iters = total // BATCH_SIZE
jitter = cycle(range(total % BATCH_SIZE))

learning_rate_fn = create_learning_rate_fn(EPOCHS, INIT_LEARNING_RATE, iters)

init_data = jnp.ones([BATCH_SIZE, INPUT_DIM], jnp.float32)
variables = vae.init(params_key, init_data, z_key)
state = train_state.TrainState.create(
        apply_fn = vae.apply,
        tx=optax.adam(learning_rate_fn),
        params=variables['params']
)

X_jax = jnp.array(X)

print(f"Training for {EPOCHS} epochs.")

for e in range(EPOCHS + 1):
    epoch_offset = next(jitter)
    if e != 0:
        print(f"Epoch: {e}, most recent training loss (bce): {metrics['loss']}")
    # random permutation of rows seems to hurt, probably because we lose
    # being distributed evenly across cell types, why we use jitter strategy
    # instead
    # X_jax = random.permutation(main_key, X_jax, axis=0, independent=False)
    for i in range(iters):
        z_rng, _ = random.split(z_key)
        x0 = i*4 + epoch_offset
        x1 = (i+1)*4 + epoch_offset
        state, metrics = train_step(state, X_jax[x0:x1, :], z_rng, learning_rate_fn)


# # Where To Go From Here
# 
# Now that we've got some interesting models, maybe we can make use of the single cell data? If that's an angle you want to pursue, you might check out the [Vendekagon Labs Single Cell EDA notebook](https://www.kaggle.com/code/vendekagonlabs/op2-single-cell-eda-10x-multiome/data) as a starting point. How could you use some of those features to represent cells differently? Or to account for the expression pertubation space as a distribution of values, rather than the aggregated values we have in `de_train`?
# 
# Perhaps there's a better way to explore the information we have on our small molecules? Maybe we can get more interesting data from the supplemental compound information provided with this contest? We explore that [here](https://www.kaggle.com/code/vendekagonlabs/supplemental-compound-info/data). Maybe we can incorporate prior expectations on gene perturbation with the compound lincs-ids using [other data sources](https://lincs.hms.harvard.edu/db/sm/10018-101/).
# 
# Maybe we could also come up with other ways to encode these features? The features uses here came from [Come on Chemicals!](https://www.kaggle.com/code/vendekagonlabs/come-on-chemicals-r-version/data) which was influenced by other public work in this competition. But you'll see there it only looks weakly associated with our gene perturbation results. Perhaps some better deep model embeddings could be useful?
# 
# Maybe we can enrich our gene information with techniques from [Pathway/Gene Set Enrichment](https://www.kaggle.com/code/vendekagonlabs/op2-pathway-enrichment)? Will our knowledge of how the proteins encoded by genes interact help or restrict our models?
# 
# We've all got a lot more to explore. Good luck!
