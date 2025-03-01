from quadrotor_env import QuadrotorEnv
import numpy as np

# Create the environment
env = QuadrotorEnv(render_mode="human")  # Enable live rendering

# Reset environment
obs, _ = env.reset()

# Define PID gains for position and orientation
pid_gains = np.array([0.1, 0., 0.1, 0.1, 0., 0.1, 0.1, 0., 0.1, 0.01, 0.01]) 

# Simulate for 200 steps
for _ in range(200):
    obs, reward, done, _, _ = env.step(pid_gains)  # Apply PID gains as action
    
    if done:  # Stop if simulation ends
        break

env.close()
