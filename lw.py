import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from ray.rllib.algorithms.dqn import DQNConfig
import ray
import torch

class LineWorld(gym.Env):
    def __init__(self,config=True): #remove config if using sb3
        super().__init__()
        self.observation_space = gym.spaces.Box(0, 1, (5,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state = 2

    def reset(self, seed=None, options=None): # keep options none for rlib remove for sb3; but seed must be there for the sb3
        self.state = 2
        return self._get_obs(), {}

    def _get_obs(self):
        v = np.zeros(5, dtype=np.float32)
        v[self.state] = 1.0
        return v

    def step(self, action):
        # Move based on action
        if action == 0:  # left
            self.state = max(0, self.state - 1)
        else:            # right
            self.state = min(4, self.state + 1)

        # Check terminal
        if self.state == 4:
            return self._get_obs(), 1, True, False, {}
        elif self.state == 0:
            return self._get_obs(), 0, True, False, {}
        else:
            return self._get_obs(), 0, False, False, {}
        

        ## DNQ or Double DQN 
# env = LineWorld()
# model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": [16]}, verbose=1,exploration_fraction=0.8,exploration_final_eps=0.1)
# model.learn(total_timesteps=20000)
# print("Result for State 3:", model.predict(np.array([[0,0,0,1,0]], dtype=np.float32)))

# for i  in range(5):
#     v = np.zeros(5, dtype=np.float32);v[i] = 1.0
#     action, _ = model.predict(v, deterministic=True)
#     print(f"State {i} => Action {action}")

# model.save("dqn_lineworld")


        ## Deuling DQN 
ray.init()
algo = (
    DQNConfig()
    .environment(env=LineWorld)
    .training(dueling=True,train_batch_size=32)   # optional
    .env_runners(num_env_runners=0)
    .build()
)

for i in range(1):
    result = algo.train()
    reward = result["env_runners"]["episode_return_mean"]
    length = result["env_runners"]["episode_len_mean"]

    print(f"Iter {i} | Reward: {reward:.2f} | Length: {length:.2f}")

for i in range(5):
    v = np.zeros(5, dtype=np.float32)
    v[i] = 1.0

    module = algo.get_module()
    obs_batch = torch.tensor([v], dtype=torch.float32)
    out = module.forward_inference({"obs":obs_batch })
    action = out["actions"][0].item()
    print(f"State {i} => Action {action}")




# pip install "ray[rllib]"
# pip install stable-baselines3
# pip install gymnasium

    
