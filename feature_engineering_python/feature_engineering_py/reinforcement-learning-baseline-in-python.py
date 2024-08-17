#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning baseline in Python with stable-baselines3
# This code will make your life easier if you want to try Reinforcement Learning (RL) as a solution to kaggle's kore 2022 challenge.
# One of the (multiple) difficulties of RL is achieving a clean implementation. While you can of course try to build yourself
# one of the RL models described in literature, chances are that you will spend more time debugging your model than actually competing.
# 
# [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/#) is powerful RL library with a number of very nice features for this competition:
# - It implements the most popular modern deep RL algorithms
# - It is simple and ellegant to use
# - It is rather well documented
# - There are plenty of tutorials and examples
# 
# In other words, it's a fantastic starting point. Alas, it requires an environment compatible with OpenAI-gym and the kore environment is not. What you'll find in this notebook is **KoreGymEnv**, a wrapper around the kore env that makes it play nice with stable-baselines3. It includes very simple feature and action engineering, so the only thing you need to care about is building upon them, choosing a model and throwing yourself into the cold, unforgiving and yet very rewarding reinforcement learning waters ;)
# 
# As a bonus, this notebook also demonstrates the end-to-end process that you need to follow to submit any model with external dependencies. Click on submit and you're good to go.
# 
# #### Notes:
# 
# - In stable-baselines3, states and actions are numpy arrays. In the kore environment, states are lists of dicts and actions are dicts with shipyard ids as keys and shipyard actions as values. Thus, we need an interface to "translate" them. This interface is effectively where you implement your state & action engineering. You'll find more details in the KoreGymEnv class.
# - In the ideal case, you would use self-play and let your agent play a very large number of games against itself, improving at ever step. Unfortunately, [it is not clear how to implement self-play in the kore env](https://www.kaggle.com/competitions/kore-2022/discussion/323382). So we have to train against static opponents. In this baseline, we'll use the starter bot. Of course, nothing prevents you from implementing pseudo-self-play and train against ever improving versions of your agent.

# ## tl; dr
# 
# ```python
# # Train a PPO agent
# from environment import KoreGymEnv
# from stable_baselines3 import PPO
# 
# kore_env = KoreGymEnv()
# model = PPO('MlpPolicy', kore_env, verbose=1)
# model.learn(total_timesteps=100000)
# ```

# # Dependencies

# In[1]:


get_ipython().system('pip install --target=lib --no-deps stable-baselines3 gym')


# #### A note on dependencies
# The kaggle notebook environment and the actual competition environment are different. I couldn't find any documentation on the differences other than through comments from more experienced kagglers. So let's take a minute to understand the cell below. I hope that this information saves fellow competitors a lot of time and trial-and-error!
# 
# `stable-baselines` is not (yet) a part of the kaggle docker environment, so we have to install it manually. In the notebook environment, you start at `/kaggle/working/`, so the cell above installs the libraries into `/kaggle/working/lib/`. We have two options to load the library now, `import lib.stable-baselines3` or add `/kaggle/working/lib/` to [sys.path](https://docs.python.org/3/library/sys.html#sys.path), which tells Python where look for modules.
# 
# When you submit your agent as an archive, however, your code is unzipped to `/kaggle_simulations/agent/`, _but the working directory remains `/kaggle/working/`_. In the competition env, neither of the options above work, because `lib` isn't `/kaggle/working/lib` anymore, it has been unzipped with the rest of your code to `/kaggle_simulations/agent/lib`. Surprise!
# 
# The code below then checks whether we are in the simulation environment, and adds the right location of the external dependencies to `sys.path`.
# 
# Additionally, there is a limit on the submission size, that's why we are installing with `--no-deps` to keep the submission size small.

# In[2]:


import os
import sys
KAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"
if os.path.exists(KAGGLE_AGENT_PATH):
    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, 'lib'))
else:
    sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))


# # Utils

# ### Config

# In[3]:


get_ipython().run_cell_magic('writefile', 'config.py', "import numpy as np\nfrom kaggle_environments import make\n\n# Read env specification\nENV_SPECIFICATION = make('kore_fleets').specification\nSHIP_COST = ENV_SPECIFICATION.configuration.spawnCost.default\nSHIPYARD_COST = ENV_SPECIFICATION.configuration.convertCost.default\nGAME_CONFIG = {\n    'episodeSteps':  ENV_SPECIFICATION.configuration.episodeSteps.default,  # You might want to start with smaller values\n    'size': ENV_SPECIFICATION.configuration.size.default,\n    'maxLogLength': None\n}\n\n# Define your opponent. We'll use the starter bot in the notebook environment for this baseline.\nOPPONENT = 'opponent.py'\nGAME_AGENTS = [None, OPPONENT]\n\n# Define our parameters\nN_FEATURES = 4\nACTION_SIZE = (2,)\nDTYPE = np.float64\nMAX_OBSERVABLE_KORE = 500\nMAX_OBSERVABLE_SHIPS = 200\nMAX_ACTION_FLEET_SIZE = 150\nMAX_KORE_IN_RESERVE = 40000\nWIN_REWARD = 1000\n")


# In[4]:


get_ipython().run_cell_magic('writefile', 'opponent.py', 'from kaggle_environments.envs.kore_fleets.helpers import *\n\n\n# This is just the starter bot. Change this with the agent of your choice.\ndef agent(obs, config):\n    board = Board(obs, config)\n\n    me = board.current_player\n    turn = board.step\n    spawn_cost = board.configuration.spawn_cost\n    kore_left = me.kore\n\n    for shipyard in me.shipyards:\n        if shipyard.ship_count > 10:\n            direction = Direction.from_index(turn % 4)\n            action = ShipyardAction.launch_fleet_with_flight_plan(2, direction.to_char())\n            shipyard.next_action = action\n        elif kore_left > spawn_cost * shipyard.max_spawn:\n            action = ShipyardAction.spawn_ships(shipyard.max_spawn)\n            shipyard.next_action = action\n            kore_left -= spawn_cost * shipyard.max_spawn\n        elif kore_left > spawn_cost:\n            action = ShipyardAction.spawn_ships(1)\n            shipyard.next_action = action\n            kore_left -= spawn_cost\n\n    return me.next_actions\n')


# ### Reward utilities

# In[5]:


get_ipython().run_cell_magic('writefile', 'reward_utils.py', 'from config import GAME_CONFIG, SHIP_COST, SHIPYARD_COST\nfrom kaggle_environments.envs.kore_fleets.helpers import Board\nimport numpy as np\nfrom math import floor\n\n# Compute weight constants -- See get_board_value\'s docstring\n_max_steps = GAME_CONFIG[\'episodeSteps\']\n_end_of_asset_value = floor(.5 * _max_steps)\n_weights_assets = np.linspace(start=1, stop=0, num=_end_of_asset_value)\n_weights_kore = np.linspace(start=0, stop=1, num=_end_of_asset_value)\nWEIGHTS_ASSETS = np.append(_weights_assets, np.zeros(_max_steps - _end_of_asset_value))\nWEIGHTS_KORE = np.append(_weights_kore, np.ones(_max_steps - _end_of_asset_value))\nWEIGHTS_MAX_SPAWN = {x: (x+3)/4 for x in range(1, 11)}  # Value multiplier of a shipyard as a function of its max spawn\nWEIGHTS_KORE_IN_FLEETS = WEIGHTS_KORE * WEIGHTS_ASSETS/2  # Always equal or smaller than either, almost always smaller\n\n\ndef get_board_value(board: Board) -> float:\n    """Computes the board value for the current player.\n\n    The board value captures how are we currently performing, compared to the opponent. Each player\'s partial board\n    value assesses the player\'s situation, taking into account their current kore, ship count, shipyard count\n    (including their max spawn) and kore carried by fleets. We then define the board value as the difference between\n    player\'s partial board values.\n    Flight plans and the positioning of fleet and shipyards do not flow into the board value (yet).\n\n    To keep things simple, we\'ll take a weighted sum as the partial board value. We need weighting since\n    the importance of each item changes over time. We don\'t need to have the most kore at the beginning of the game,\n    but we do at the end. Ship count won\'t help us win games in the latter stages, but it is crucial in the beginning.\n    Fleets and shipyards will be accounted for proportionally to their kore cost.\n\n    For efficiency, the weight factors are pre-computed at module level. Here is the logic behind the weighting:\n    WEIGHTS_KORE: Applied to the player\'s kore count. Increases linearly from 0 to 1. It reaches one before\n        the maximum game length is reached.\n    WEIGHTS_ASSETS: Applied to fleets and shipyards. Decreases linearly from 1 to 0 and reaches zero before the maximum\n        length. It emphasizes the need of having ships over kore at the beginning of the game.\n    WEIGHTS_MAX_SPAWN: Shipyard value is multiplied by its max spawn. This captures the idea that long-held shipyards\n        are more valuable.\n    WEIGHTS_KORE_IN_FLEETS: Kore in fleets should be valued, too. But its value must be upper-bounded by WEIGHTS_KORE\n        (it can never be better to have kore in cargo than home) and it must decrease in time, since it doesn\'t\n        count towards the end kore count.\n\n    Args:\n        board: The board for which we want to compute the value.\n\n    Returns:\n        The value of the board.\n    """\n    board_value = 0\n    if not board:\n        return board_value\n\n    # Get the weights as a function of the current game step\n    step = board.step\n    weight_kore, weight_assets, weight_cargo = WEIGHTS_KORE[step], WEIGHTS_ASSETS[step], WEIGHTS_KORE_IN_FLEETS[step]\n\n    # Compute the partial board values\n    for player in board.players.values():\n        player_fleets, player_shipyards = list(player.fleets), list(player.shipyards)\n\n        value_kore = weight_kore * player.kore\n\n        value_fleets = weight_assets * SHIP_COST * (\n                sum(fleet.ship_count for fleet in player_fleets)\n                + sum(shipyard.ship_count for shipyard in player_shipyards)\n        )\n\n        value_shipyards = weight_assets * SHIPYARD_COST * (\n            sum(shipyard.max_spawn * WEIGHTS_MAX_SPAWN[shipyard.max_spawn] for shipyard in player_shipyards)\n        )\n\n        value_kore_in_cargo = weight_cargo * sum(fleet.kore for fleet in player_fleets)\n\n        # Add (or subtract) the partial values to the total board value. The current player is always us.\n        modifier = 1 if player.is_current_player else -1\n        board_value += modifier * (value_kore + value_fleets + value_shipyards + value_kore_in_cargo)\n\n    return board_value\n')


# # The KoreGymEnv wrapper

# In[6]:


get_ipython().run_cell_magic('writefile', 'environment.py', 'import gym\nimport numpy as np\nfrom gym import spaces\nfrom math import floor\nfrom kaggle_environments import make\nfrom kaggle_environments.envs.kore_fleets.helpers import ShipyardAction, Board, Direction\nfrom typing import Union, Tuple, Dict\nfrom reward_utils import get_board_value\nfrom config import (\n    N_FEATURES,\n    ACTION_SIZE,\n    GAME_AGENTS,\n    GAME_CONFIG,\n    DTYPE,\n    MAX_OBSERVABLE_KORE,\n    MAX_OBSERVABLE_SHIPS,\n    MAX_ACTION_FLEET_SIZE,\n    MAX_KORE_IN_RESERVE,\n    WIN_REWARD,\n)\n\n\nclass KoreGymEnv(gym.Env):\n    """An openAI-gym env wrapper for kaggle\'s kore environment. Can be used with stable-baselines3.\n\n    There are three fundamental components to this class which you would want to customize for your own agents:\n        The action space is defined by `action_space` and `gym_to_kore_action()`\n        The state space (observations) is defined by `state_space` and `obs_as_gym_state()`\n        The reward is computed with `compute_reward()`\n\n    Note that the action and state spaces define the inputs and outputs to your model *as numpy arrays*. Use the\n    functions mentioned above to translate these arrays into actual kore environment observations and actions.\n\n    The rest is basically boilerplate and makes sure that the kaggle environment plays nicely with stable-baselines3.\n\n    Usage:\n        >>> from stable_baselines3 import PPO\n        >>>\n        >>> kore_env = KoreGymEnv()\n        >>> model = PPO(\'MlpPolicy\', kore_env, verbose=1)\n        >>> model.learn(total_timesteps=100000)\n    """\n\n    def __init__(self, config=None, agents=None, debug=None):\n        super(KoreGymEnv, self).__init__()\n\n        if not config:\n            config = GAME_CONFIG\n        if not agents:\n            agents = GAME_AGENTS\n        if not debug:\n            debug = True\n\n        self.agents = agents\n        self.env = make("kore_fleets", configuration=config, debug=debug)\n        self.config = self.env.configuration\n        self.trainer = None\n        self.raw_obs = None\n        self.previous_obs = None\n\n        # Define the action and state space\n        # Change these to match your needs. Normalization to the [-1, 1] interval is recommended. See:\n        # https://araffin.github.io/slides/rlvs-tips-tricks/#/13/0/0\n        # See https://www.gymlibrary.ml/content/spaces/ for more info on OpenAI-gym spaces.\n        self.action_space = spaces.Box(\n            low=-1,\n            high=1,\n            shape=ACTION_SIZE,\n            dtype=DTYPE\n        )\n\n        self.observation_space = spaces.Box(\n            low=-1,\n            high=1,\n            shape=(self.config.size ** 2 * N_FEATURES + 3,),\n            dtype=DTYPE\n        )\n\n        self.strict_reward = config.get(\'strict\', False)\n\n        # Debugging info - Enable or disable as needed\n        self.reward = 0\n        self.n_steps = 0\n        self.n_resets = 0\n        self.n_dones = 0\n        self.last_action = None\n        self.last_done = False\n\n    def reset(self) -> np.ndarray:\n        """Resets the trainer and returns the initial observation in state space.\n\n        Returns:\n            self.obs_as_gym_state: the current observation encoded as a state in state space\n        """\n        # agents = self.agents if np.random.rand() > .5 else self.agents[::-1]  # Randomize starting position\n        self.trainer = self.env.train(self.agents)\n        self.raw_obs = self.trainer.reset()\n        self.n_resets += 1\n        return self.obs_as_gym_state\n\n    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:\n        """Execute action in the trainer and return the results.\n\n        Args:\n            action: The action in action space, i.e. the output of the stable-baselines3 agent\n\n        Returns:\n            self.obs_as_gym_state: the current observation encoded as a state in state space\n            reward: The agent\'s reward\n            done: If True, the episode is over\n            info: A dictionary with additional debugging information\n        """\n        kore_action = self.gym_to_kore_action(action)\n        self.previous_obs = self.raw_obs\n        self.raw_obs, _, done, info = self.trainer.step(kore_action)  # Ignore trainer reward, which is just delta kore\n        self.reward = self.compute_reward(done)\n\n        # Debugging info\n        # with open(\'logs/tmp.log\', \'a\') as log:\n        #    print(kore_action.action_type, kore_action.num_ships, kore_action.flight_plan, file=log)\n        #    if done:\n        #        print(\'done\', file=log)\n        #    if info:\n        #        print(\'info\', file=log)\n        self.n_steps += 1\n        self.last_done = done\n        self.last_action = kore_action\n        self.n_dones += 1 if done else 0\n\n        return self.obs_as_gym_state, self.reward, done, info\n\n    def render(self, **kwargs):\n        self.env.render(**kwargs)\n\n    def close(self):\n        pass\n\n    @property\n    def board(self):\n        return Board(self.raw_obs, self.config)\n\n    @property\n    def previous_board(self):\n        return Board(self.previous_obs, self.config)\n\n    def gym_to_kore_action(self, gym_action: np.ndarray) -> Dict[str, str]:\n        """Decode an action in action space as a kore action.\n\n        In other words, transform a stable-baselines3 action into an action compatible with the kore environment.\n\n        This method is central - It defines how the agent output is mapped to kore actions.\n        You can modify it to suit your needs.\n\n        Let\'s start with an übereasy mapping. Our gym_action is a 1-dimensional vector of size 2 (as defined in\n        self.action_space). We will interpret the values as follows:\n        if gym_action[0] > 0 launch a fleet, elif < 0 build ships, else wait.\n        abs(gym_action[0]) encodes the number of ships to build/launch.\n        gym_action[1] represents the direction in which to launch the fleet.\n\n        Notes: The same action is sent to all shipyards, though we make sure that the actions are valid.\n\n        Args:\n            gym_action: The action produces by our stable-baselines3 agent.\n\n        Returns:\n            The corresponding kore environment actions or None if the agent wants to wait.\n\n        """\n        action_launch = gym_action[0] > 0\n        action_build = gym_action[0] < 0\n        # Mapping the number of ships is an interesting exercise. Here we chose a linear mapping to the interval\n        # [1, MAX_ACTION_FLEET_SIZE], but you could use something else. With a linear mapping, all values are\n        # evenly spaced. An exponential mapping, however, would space out lower values, making them easier for the agent\n        # to distinguish and choose, at the cost of needing more precision to accurately select higher values.\n        number_of_ships = int(\n            clip_normalize(\n                x=abs(gym_action[0]),\n                low_in=0,\n                high_in=1,\n                low_out=1,\n                high_out=MAX_ACTION_FLEET_SIZE\n            )\n        )\n\n        # Broadcast the same action to all shipyards\n        board = self.board\n        me = board.current_player\n        for shipyard in me.shipyards:\n            action = None\n            if action_build:\n                # Limit the number of ships to the maximum that can be actually built\n                max_spawn = shipyard.max_spawn\n                max_purchasable = floor(me.kore / self.config["spawnCost"])\n                number_of_ships = min(number_of_ships, max_spawn, max_purchasable)\n                if number_of_ships:\n                    action = ShipyardAction.spawn_ships(number_ships=number_of_ships)\n\n            elif action_launch:\n                # Limit the number of ships to the amount that is actually present in the shipyard\n                shipyard_count = shipyard.ship_count\n                number_of_ships = min(number_of_ships, shipyard_count)\n                if number_of_ships:\n                    direction = round((gym_action[1] + 1) * 1.5)  # int between 0 (North) and 3 (West)\n                    action = ShipyardAction.launch_fleet_in_direction(number_ships=number_of_ships,\n                                                                      direction=Direction.from_index(direction))\n            shipyard.next_action = action\n\n        return me.next_actions\n\n    @property\n    def obs_as_gym_state(self) -> np.ndarray:\n        """Return the current observation encoded as a state in state space.\n\n        In other words, transform a kore observation into a stable-baselines3-compatible np.ndarray.\n\n        This property is central - It defines how the kore board is mapped to our state space.\n        You can modify it to include as many features as you see convenient.\n\n        Let\'s keep start with something easy: Define a 21x21x4+3 state (size x size x n_features and 3 extra features).\n        # Feature 0: How much kore there is in a cell\n        # Feature 1: How many ships there are in a cell (>0: friendly, <0: enemy)\n        # Feature 2: Fleet direction\n        # Feature 3: Is a shipyard present? (1: friendly, -1: enemy, 0: no)\n        # Feature 4: Progress - What turn is it?\n        # Feature 5: How much kore do I have?\n        # Feature 6: How much kore does the opponent have?\n\n        We\'ll make sure that all features are in the range [-1, 1] and as close to a normal distribution as possible.\n\n        Note: This mapping doesn\'t tackle a critical issue in kore: How to encode (full) flight plans?\n        """\n        # Init output state\n        gym_state = np.ndarray(shape=(self.config.size, self.config.size, N_FEATURES))\n\n        # Get our player ID\n        board = self.board\n        our_id = board.current_player_id\n\n        for point, cell in board.cells.items():\n            # Feature 0: How much kore\n            gym_state[point.y, point.x, 0] = cell.kore\n\n            # Feature 1: How many ships (>0: friendly, <0: enemy)\n            # Feature 2: Fleet direction\n            fleet = cell.fleet\n            if fleet:\n                modifier = 1 if fleet.player_id == our_id else -1\n                gym_state[point.y, point.x, 1] = modifier * fleet.ship_count\n                gym_state[point.y, point.x, 2] = fleet.direction.value\n            else:\n                # The current cell has no fleet\n                gym_state[point.y, point.x, 1] = gym_state[point.y, point.x, 2] = 0\n\n            # Feature 3: Shipyard present (1: friendly, -1: enemy)\n            shipyard = cell.shipyard\n            if shipyard:\n                gym_state[point.y, point.x, 3] = 1 if shipyard.player_id == our_id else -1\n            else:\n                # The current cell has no shipyard\n                gym_state[point.y, point.x, 3] = 0\n\n        # Normalize features to interval [-1, 1]\n        # Feature 0: Logarithmic scale, kore in range [0, MAX_OBSERVABLE_KORE]\n        gym_state[:, :, 0] = clip_normalize(\n            x=np.log2(gym_state[:, :, 0] + 1),\n            low_in=0,\n            high_in=np.log2(MAX_OBSERVABLE_KORE)\n        )\n\n        # Feature 1: Ships in range [-MAX_OBSERVABLE_SHIPS, MAX_OBSERVABLE_SHIPS]\n        gym_state[:, :, 1] = clip_normalize(\n            x=gym_state[:, :, 1],\n            low_in=-MAX_OBSERVABLE_SHIPS,\n            high_in=MAX_OBSERVABLE_SHIPS\n        )\n\n        # Feature 2: Fleet direction in range (1, 4)\n        gym_state[:, :, 2] = clip_normalize(\n            x=gym_state[:, :, 2],\n            low_in=1,\n            high_in=4\n        )\n\n        # Feature 3 is already as normal as it gets\n\n        # Flatten the input (recommended by stable_baselines3.common.env_checker.check_env)\n        output_state = gym_state.flatten()\n\n        # Extra Features: Progress, how much kore do I have, how much kore does opponent have\n        player = board.current_player\n        opponent = board.opponents[0]\n        progress = clip_normalize(board.step, low_in=0, high_in=GAME_CONFIG[\'episodeSteps\'])\n        my_kore = clip_normalize(np.log2(player.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))\n        opponent_kore = clip_normalize(np.log2(opponent.kore+1), low_in=0, high_in=np.log2(MAX_KORE_IN_RESERVE))\n\n        return np.append(output_state, [progress, my_kore, opponent_kore])\n\n    def compute_reward(self, done: bool, strict=False) -> float:\n        """Compute the agent reward. Welcome to the fine art of RL.\n\n         We\'ll compute the reward as the current board value and a final bonus if the episode is over. If the player\n          wins the episode, we\'ll add a final bonus that increases with shorter time-to-victory.\n        If the player loses, we\'ll subtract that bonus.\n\n        Args:\n            done: True if the episode is over\n            strict: If True, count only wins/loses (Useful for evaluating a trained agent)\n\n        Returns:\n            The agent\'s reward\n        """\n        board = self.board\n        previous_board = self.previous_board\n\n        if strict:\n            if done:\n                # Who won?\n                # Ugly but 99% sure correct, see https://www.kaggle.com/competitions/kore-2022/discussion/324150#1789804\n                agent_reward = self.raw_obs.players[0][0]\n                opponent_reward = self.raw_obs.players[1][0]\n                return int(agent_reward > opponent_reward)\n            else:\n                return 0\n        else:\n            if done:\n                # Who won?\n                agent_reward = self.raw_obs.players[0][0]\n                opponent_reward = self.raw_obs.players[1][0]\n                if agent_reward is None or opponent_reward is None:\n                    we_won = -1\n                else:\n                    we_won = 1 if agent_reward > opponent_reward else -1\n                win_reward = we_won * (WIN_REWARD + 5 * (GAME_CONFIG[\'episodeSteps\'] - board.step))\n            else:\n                win_reward = 0\n\n            return get_board_value(board) - get_board_value(previous_board) + win_reward\n\n\ndef clip_normalize(x: Union[np.ndarray, float],\n                   low_in: float,\n                   high_in: float,\n                   low_out=-1.,\n                   high_out=1.) -> Union[np.ndarray, float]:\n    """Clip values in x to the interval [low_in, high_in] and then MinMax-normalize to [low_out, high_out].\n\n    Args:\n        x: The array of float to clip and normalize\n        low_in: The lowest possible value in x\n        high_in: The highest possible value in x\n        low_out: The lowest possible value in the output\n        high_out: The highest possible value in the output\n\n    Returns:\n        The clipped and normalized version of x\n\n    Raises:\n        AssertionError if the limits are not consistent\n\n    Examples:\n        >>> clip_normalize(50, low_in=0, high_in=100)\n        0.0\n\n        >>> clip_normalize(np.array([-1, .5, 99]), low_in=-1, high_in=1, low_out=0, high_out=2)\n        array([0., 1.5, 2.])\n    """\n    assert high_in > low_in and high_out > low_out, "Wrong limits"\n\n    # Clip outliers\n    try:\n        x[x > high_in] = high_in\n        x[x < low_in] = low_in\n    except TypeError:\n        x = high_in if x > high_in else x\n        x = low_in if x < low_in else x\n\n    # y = ax + b\n    a = (high_out - low_out) / (high_in - low_in)\n    b = high_out - high_in * a\n\n    return a * x + b\n')


# ### Check that we have a valid environment

# In[7]:


# The bad news: this check will fail in the kaggle docker environment. The most likely reason is a version mismatch between packages.
# The good news: That's alright since everything else works! We're doing some unconventional dependency management here, so we'll have to live with
# a failed check.

#from stable_baselines3.common.env_checker import check_env
#from environment import KoreGymEnv

#env = KoreGymEnv()
#check_env(env)


# # Train the agent!

# In[8]:


from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from environment import KoreGymEnv


# In[9]:


kore_env = KoreGymEnv(config=dict(randomSeed=997269658))  # TODO: This seed is not enough. Seed everything!
monitored_env = Monitor(env=kore_env)
model = PPO('MlpPolicy', monitored_env, verbose=1)


# In[10]:


get_ipython().run_cell_magic('time', '', '# For serious training, likely many more iterations will be needed, as well as hyperparameter tuning!\n# Even so, sometimes training will still fail. RL is like that. Try a couple times with the same config before giving up!\nmodel.learn(total_timesteps=50000)  \n')


# In[11]:


# Watch it mercilessly beat the baseline bot - Note: The current episode might not be over yet
kore_env.render(mode="ipython", width=1000, height=800)


# In[12]:


model.save("baseline_agent")


# # Evaluate agent performance

# In[13]:


import numpy as np

eval_env = KoreGymEnv(config=dict(strict=True))  # The 'strict' flags sets rewards to 1 if the agent won the episode and 0 else. Useful for evaluation.
monitored_env = Monitor(env=eval_env)
model_loaded = PPO.load('baseline_agent')

def evaluate(model, num_episodes=1):
    """
    Evaluate a RL agent - Adapted from 
    https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = monitored_env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = monitored_env.step(action)
            if done:
                agent_reward = monitored_env.env.raw_obs.players[0][0]
                opponent_reward = monitored_env.env.raw_obs.players[1][0]
                reward = agent_reward > opponent_reward
            else:
                reward = 0
            # print(reward)
            # monitored_env.render(mode='ipython', height=400, width=300)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward

evaluate(model_loaded, 20)


# ### Prepare the submission

# In[14]:


get_ipython().run_cell_magic('writefile', 'main.py', '# All this syspath wranglig is needed to make sure that the agent runs on the target environment and can load both the external dependencies\n# and the saved model. Dear kaggle, if possible, please make this easier!\nimport os\nimport sys\nKAGGLE_AGENT_PATH = "/kaggle_simulations/agent/"\nif os.path.exists(KAGGLE_AGENT_PATH):\n    # We\'re in the kaggle target system\n    sys.path.insert(0, os.path.join(KAGGLE_AGENT_PATH, \'lib\'))\n    agent_path = os.path.join(KAGGLE_AGENT_PATH, \'baseline_agent\')\nelse:\n    # We\'re somewhere else\n    sys.path.insert(0, os.path.join(os.getcwd(), \'lib\'))\n    agent_path = \'baseline_agent\'\n\n# Now for the actual agent\nfrom stable_baselines3 import PPO\nfrom environment import KoreGymEnv\n\nmodel = PPO.load(agent_path)\nkore_env = KoreGymEnv()\n\ndef agent(obs, config):\n    kore_env.raw_obs = obs\n    state = kore_env.obs_as_gym_state\n    action, _ = model.predict(state)\n    return kore_env.gym_to_kore_action(action)\n')


# In[15]:


get_ipython().run_cell_magic('capture', '', '# This is for debugging purposes only before submitting - Are there any errors?\nfrom kaggle_environments import make\nfrom config import OPPONENT\nenv = make("kore_fleets", debug=True)\nenv.run([\'main.py\', OPPONENT])\n')


# In[16]:


get_ipython().system('tar -czf submission.tar.gz main.py config.py environment.py reward_utils.py baseline_agent.zip lib')

