from quadrotor_env import QuadrotorEnv
import numpy as np
import mujoco
import mujoco.viewer
import time

# Create the environment
env = QuadrotorEnv(render_mode="human")  # Enable live rendering
    
# Reset environment
obs, _ = env.reset()
dt = env.model.opt.timestep

# Define PID gains for position and orientation
gains = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 

# Simulate while GUI is open
while True:
    obs, reward, done, _, _ = env.step(gains)  # Apply PID gains as action
    env.render()
    time.sleep(dt)