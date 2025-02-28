import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

class QuadrotorEnv(gym.Env):
    '''
    Custom quadrotor environment
    Observation space = [x[3], v[3], q[4], w[3]]
    Action space = [Kp[3], Ki[3], Kd[3], Kq, Kw]
    '''
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}


    def __init__(self, xml_path="quadrotor.xml", render_mode=None):
        super().__init__()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
        )

        # Action space
        self.action_space = gym.spaces.Box(
            low=0, high=10, shape=(6,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None  # Viewer is not created until needed
        self.target_position = np.array([0, 0, 1])  # Hover position
        self.max_steps = 1000
        self.current_step = 0


    def step(self, action):
        '''Peforms a step in the environment'''
        self.current_step += 1

        # Extract PID gains
        Kp_pos, Ki_pos, Kd_pos = action[:3]
        Kp_ang, Ki_ang, Kd_ang = action[3:]

        # Get current state
        pos = self.data.qpos[:3]  # Position
        vel = self.data.qvel[:3]  # Linear velocity
        quat = self.data.qpos[3:7]  # Orientation quaternion
        ang_vel = self.data.qvel[3:6]  # Angular velocity

        # Compute position error and control signal
        pos_error = self.target_position - pos
        control_signal = Kp_pos * pos_error - Kd_pos * vel  # Simple PD controller

        # Convert control signal to rotor forces (thrust distribution)
        thrust = np.clip(np.linalg.norm(control_signal), 0, 10)
        self.data.ctrl[:] = thrust

        # Step MuJoCo simulation
        mujoco.mj_step(self.model, self.data)

        # Render if real-time visualization is enabled
        if self.render_mode == "human":
            self.render()

        # Compute reward (hover stability, low control effort)
        reward = -np.linalg.norm(pos_error) - 0.1 * np.linalg.norm(vel)

        # Check termination
        done = self.current_step >= self.max_steps or np.abs(pos[2]) > 10

        return self._get_obs(), reward, done, False, {}
    

    def _get_obs(self):
        ''' Return the current observation (state) '''
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]
        return np.concatenate([pos, vel, quat, ang_vel])


    def reset(self, seed=None, options=None):
        ''' Reset the environment '''
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}


    def render(self):
        ''' Render the environment in real-time '''
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.model, self.data)  # Create a real-time viewer
            #mujoco.viewer.sync(self.viewer, self.data)
            self.viewer.render()


    def close(self):
        ''' Close the environment and free memory '''
        if self.viewer is not None:
            self.viewer = None  # Free up resources