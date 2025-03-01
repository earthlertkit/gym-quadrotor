from quadrotor_env import QuadrotorEnv
from stable_baselines3 import PPO
import time

# Create the environment
env = QuadrotorEnv(render_mode="human")  # Enable live rendering
    
# Reset environment
obs, _ = env.reset()
dt = env.model.opt.timestep

# Define PID gains for position and orientation
model = PPO.load("quadrotor_ppo.zip")

# Simulate while GUI is open
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)  # Apply PID gains as action
    env.render()
    time.sleep(dt)