from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from quadrotor_env import QuadrotorEnv

# Create the MuJoCo environment
env = QuadrotorEnv()

# Instantiate the PPO model
model = PPO(
    policy=MlpPolicy,
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
)

# Train for a certain number of timesteps
timesteps = 1000000
model.learn(total_timesteps=timesteps)

# Save the model
model.save("quadrotor_ppo")

env.close()