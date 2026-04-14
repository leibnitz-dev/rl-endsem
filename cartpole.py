import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C

# 1. Setup Environment
env = gym.make("CartPole-v1", render_mode="human")

# 2. CHOOSE YOUR ALGORITHM (Uncomment the one you want to use)

# # --- DQN Variants ---
# # Standard DQN (Double DQN is enabled by default in SB3)
model = DQN("MlpPolicy", env, verbose=1)

# Dueling DQN thrugh rlib

# --- Policy Gradient / Actor-Critic ---
# PPO (The modern "Vanilla Policy Gradient")
# model = PPO("MlpPolicy", env, verbose=1)

# A2C (Synchronous Actor-Critic)
# model = A2C("MlpPolicy", env, verbose=1)

# 3. Train
print(f"Training {model.__class__.__name__}...")
model.learn(total_timesteps=100)

# 4. Watch the trained agent
print("Training finished. Watch the agent!")
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
