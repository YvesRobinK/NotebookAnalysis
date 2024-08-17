#!/usr/bin/env python
# coding: utf-8

# First version of the code is copied from https://www.kaggle.com/code/shoheiazuma/lux-ai-with-imitation-learning.
# 
# See https://www.kaggle.com/code/huikang/kore-2022-feature-generator/notebook for feature and target engineering.
# 
# Feel free to clarify.

# In[1]:


get_ipython().run_line_magic('reset', '-sf')
get_ipython().system('echo $KAGGLE_KERNEL_RUN_TYPE')


# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install kaggle-environments -U > /dev/null\n!cp ../input/kore-2022-feature-generator/kore_analysis.py .\n!cp ../input/kore-2022-feature-generator/feature_generator.py .\n')


# In[3]:


from IPython.core.magic import register_cell_magic

@register_cell_magic
def writefile_and_run(line, cell):
    argz = line.split()
    file = argz[-1]
    mode = 'w'
    if len(argz) == 2 and argz[0] == '-a':
        mode = 'a'
    with open(file, mode) as f:
        f.write(cell)
    get_ipython().run_cell(cell)


# In[4]:


get_ipython().run_cell_magic('writefile_and_run', '-a imitation_training_helper.py', '\nimport os, collections, random\nfrom tqdm.notebook import tqdm\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport matplotlib\nfrom sklearn.model_selection import train_test_split\nfrom scipy.special import softmax\n\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom torch.utils.data import Dataset, DataLoader\nimport torch.optim as optim\n\ntorch_device = "cuda" if torch.cuda.is_available() else "cpu"\n')


# In[5]:


from feature_generator import plot_3d_matrix


# In[6]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)


# # Dataset and Dataloader

# In[7]:


actions_df = pd.read_csv("../input/kore-2022-feature-generator/actions_df.csv")
actions_df.shape


# In[8]:


actions_df = actions_df[(actions_df["diff_x"] != 0) | (actions_df["diff_y"] != 0)]
actions_df = actions_df[abs(actions_df["diff_x"]) <= 10]
actions_df = actions_df[abs(actions_df["diff_y"]) <= 10]
actions_df = actions_df[actions_df["action_class"] >= 0]
# actions_df = actions_df[abs(actions_df["turn_idx"]) <= 20]
# actions_df = actions_df[abs(actions_df["diff_x"]) + abs(actions_df["diff_y"]) <= 11]
# actions_df = actions_df[abs(actions_df["diff_x"]) + abs(actions_df["diff_y"]) >= 3]
actions_df["diff_x"] = (actions_df["diff_x"] + 10)
actions_df["diff_y"] = (actions_df["diff_y"] + 10)
actions_df.loc[actions_df["action_class"] == 3, "action_class"] = 0  # recast attack action as build action
actions_df.shape


# In[9]:


actions_df.sample(5)


# In[10]:


actions_df["action_class"].value_counts()


# In[11]:


actions_df.head()


# In[12]:


plt.figure(figsize=(12,12))
plt.scatter(actions_df["diff_x"] + (actions_df["turn_idx"]//20)/ 25 - 0.4, 
            actions_df["diff_y"] + (actions_df["turn_idx"] %20)/ 25 - 0.4,
            s=actions_df["ship_amount"], c=actions_df["action_class"], cmap="winter_r")
plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
plt.gca().set_aspect('equal')
plt.show()


# In[13]:


get_ipython().run_cell_magic('writefile_and_run', '-a imitation_training_helper.py', '\ndef append_source_specific_features(input_matrix):\n    kore_matrix = input_matrix[3,:,:]\n    kore_matrix_hori = np.zeros((21, 21))\n    kore_matrix_vert = np.zeros((21, 21))\n    dist_from_shipyard = np.add.outer(np.abs(np.arange(-10,11)), np.abs(np.arange(-10,11)))\n    assert dist_from_shipyard[10,10] == 0\n    dist_from_shipyard[10,10] = 1  # avoid divide by zero error later\n    shipyard_ship_count = np.full((21, 21), input_matrix[-11,10,10])\n    \n    # assert kore_matrix[10,10] == 0  # shipyard position should not have kore\n    # assert input_matrix[-12,10,10] != 0  # launch position indeed has a shipyard\n\n    for i in range(10):  # first direction\n        kore_matrix_hori[10,10+i+1] += kore_matrix_hori[10,10+i] + kore_matrix[10,10+i+1]\n        kore_matrix_hori[10,10-i-1] += kore_matrix_hori[10,10-i] + kore_matrix[10,10-i-1]\n        kore_matrix_vert[10+i+1,10] += kore_matrix_vert[10+i,10] + kore_matrix[10+i+1,10]\n        kore_matrix_vert[10-i-1,10] += kore_matrix_vert[10-i,10] + kore_matrix[10-i-1,10]\n    \n    for i in range(10):  # second direction\n        kore_matrix_vert[:,10+i+1] += kore_matrix_vert[:,10+i] + kore_matrix[:,10+i+1]\n        kore_matrix_vert[:,10-i-1] += kore_matrix_vert[:,10-i] + kore_matrix[:,10-i-1]\n        kore_matrix_hori[10+i+1,:] += kore_matrix_hori[10+i,:] + kore_matrix[10+i+1,:]\n        kore_matrix_hori[10-i-1,:] += kore_matrix_hori[10-i,:] + kore_matrix[10-i-1,:]\n    \n    # each cell is visited twice except the destination cell\n    kore_matrix_hori = (2*kore_matrix_hori - kore_matrix) / dist_from_shipyard\n    kore_matrix_vert = (2*kore_matrix_vert - kore_matrix) / dist_from_shipyard\n    input_matrix = np.concatenate(([kore_matrix_hori, kore_matrix_vert, \n                                    dist_from_shipyard, shipyard_ship_count], input_matrix), axis=0)\n    input_matrix = np.clip(input_matrix, 0, 10)\n    # input_matrix = input_matrix[[0,1,2,3,9,-12,-9]]\n    return input_matrix\n\ndef action_encoder(action_class, diff_x, diff_y):\n    assert 0 <= action_class < 3\n    assert 0 <= diff_x < 21\n    assert 0 <= diff_y < 21\n    return action_class*21*21 + diff_x*21 + diff_y\n\ndef action_decoder(clf_idx):\n    action_class, clf_idx = divmod(clf_idx, 21*21)\n    diff_x, diff_y = divmod(clf_idx, 21)\n    return action_class, diff_x, diff_y\n')


# In[14]:


class KoreDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (submission_id, episode_id, turn_idx), samples = self.samples[idx]
        # load numpy object
        npy_path_name = f"""../input/kore-2022-feature-generator/npy/{submission_id}_{episode_id}_{turn_idx-1:03d}_inputs.npy"""
        sample = random.choice(samples)
        state = np.load(npy_path_name)
        state = np.roll(state, (0, -sample["shipyard_x"] + 10, -sample["shipyard_y"] + 10),
                        axis = (0, 1, 2))  # center shipyard
        state = append_source_specific_features(state)
        # assert state[-12,10,10] != 0  # ensure there is shipyard at the center
        action_tuple = sample["action_class"], sample["diff_x"], sample["diff_y"]
        action = action_encoder(*action_tuple)
        assert action_decoder(action) == action_tuple
        return state, action


# In[15]:


train_actions_df = actions_df[actions_df["episode_id"]%10 != 0]
val_actions_df = actions_df[actions_df["episode_id"]%10 == 0]

def aggregate_into_episode_and_turn(df):
    samples_build = collections.defaultdict(list)
    samples = collections.defaultdict(list)
    for record in df.to_dict('records'):
        if record["turn_idx"] <= 3:
            continue
        submission_episode_turnidx = record["submission_id"], record["episode_id"], record["turn_idx"]
        submission_id, episode_id, turn_idx = submission_episode_turnidx
        npy_path_name = f"""../input/kore-2022-feature-generator/npy/{submission_id}_{episode_id}_{turn_idx-1:03d}_inputs.npy"""
        if not os.path.isfile(npy_path_name):
            continue
        if record["action_class"] == 0:  # is build action
            samples_build[submission_episode_turnidx].append(record)
        else:
            samples[submission_episode_turnidx].append(record)            
    return list(samples.items()) + list(samples_build.items())

train_samples = aggregate_into_episode_and_turn(train_actions_df)
val_samples = aggregate_into_episode_and_turn(val_actions_df)


# In[16]:


input_matrix, action = KoreDataset(train_samples)[200]
NUM_LAYERS = input_matrix.shape[0]
NUM_LAYERS, input_matrix.shape, action


# In[17]:


batch_size = 64
train_loader = DataLoader(
    KoreDataset(train_samples), 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2
)
val_loader = DataLoader(
    KoreDataset(val_samples), 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=2
)


# # Model

# In[18]:


class BasicConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.conv = nn.Conv2d(
            input_dim, output_dim, 
            kernel_size=kernel_size, 
            padding_mode='circular',
            padding=(1,1),
        )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h) if self.bn is not None else h
        return h


class KoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = BasicConv2d(NUM_LAYERS, filters, (3, 3), True)
        self.blocks = nn.ModuleList([BasicConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.conv1 = BasicConv2d(filters, 3, (3, 3), True)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h = self.conv1(h)
        return torch.flatten(h, start_dim=1)


# In[19]:


kore_net = KoreNet()
kore_net.to(torch_device)
pass


# # Training

# In[20]:


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.to(torch_device)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_acc = 0
            
            dataloader = dataloaders_dict[phase]
            for states, actions in tqdm(dataloader, leave=False):
                states = states.to(torch_device).float()
                actions = actions.to(torch_device).long()

                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    policy = model(states)
                    loss = criterion(policy, actions)
                    _, preds = torch.max(policy, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(policy)
                    epoch_acc += torch.sum(preds == actions.data)

            data_size = len(dataloader.dataset)
            epoch_loss = epoch_loss / data_size
            epoch_acc = epoch_acc.double() / data_size

            print(f'Epoch {epoch + 1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
        
        if epoch_acc > best_acc:
            traced = torch.jit.trace(model.cpu(), torch.rand(1, NUM_LAYERS, 21, 21))
            traced.save('model.pth')
            best_acc = epoch_acc

        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and epoch == 2:
            break  # for interactive runs, only check that it is working


# In[21]:


model = KoreNet()
dataloaders_dict = {"train": train_loader, "val": val_loader}
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


# In[22]:


train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=30)


# # Inference

# In[23]:


for states, actions in val_loader:
    break

with torch.no_grad():
    p = kore_net(states.to(torch_device).float())
p.shape, actions.shape


# In[24]:


assert states[0].numpy()[-2,10,10] != 0


# In[25]:


plot_3d_matrix(states[0].numpy())


# In[26]:


kore_slice = states[0].numpy()[:1]
plot_3d_matrix(kore_slice, scene_camera_eye=dict(x=3, y=3, z=3))


# In[27]:


kore_slice = states[0].numpy()[1:2]
plot_3d_matrix(kore_slice, scene_camera_eye=dict(x=3, y=3, z=3))


# In[28]:


probs = softmax(p[0].to('cpu').numpy()).reshape(3,21,21)
plot_3d_matrix(probs, scene_camera_eye=dict(x=1, y=1, z=1))


# In[29]:


action_decoder(actions[0].to('cpu').numpy())


# The use of the model to build an imitation agent will be done in another notebook.

# In[30]:


get_ipython().system('rm -rf __pycache__/')

