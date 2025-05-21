import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as rot

class QuadrotorEnv(gym.Env):
    '''
    Custom quadrotor environment
    Observation space = [x_error[3], v_error[3], q_error[4], w_error[3]]
    Action space = [T1, T2, T3, T4]
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
            low=0, high=1, shape=(4,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None  # Viewer is not created until needed
        self.target_position = np.array([0, 0, 1])  # Hover position
        self.target_orientation = np.array([1, 0, 0, 0])  
        self.max_steps = int(20/self.model.opt.timestep)
        self.current_step = 0


    def step(self, action):
        '''Peforms a step in the environment'''

        # Send motor force data as control input ()
        self.data.ctrl[:] = action

        # Step MuJoCo simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Render if real-time visualization is enabled
        if self.render_mode == "human":
            self.render()
            
        # Compute reward (hover stability, low control effort)
        reward = - np.linalg.norm(self.target_position - self.data.qpos[0:3])**2

        # Check termination
        done = self.current_step >= self.max_steps or self.data.qpos[2] <= 0.025

        return self._get_obs(), reward, done, False, {}
    

    def _get_obs(self):
        ''' Return the current observation (state) '''
        pos_error = self.target_position - self.data.qpos[:3]
        vel_error = - self.data.qvel[:3]
        ang_error = self.quat_mul(self.target_orientation, self.quat_conj(self.data.qpos[:4]))
        ang_vel_error = - self.data.qvel[3:6]
        return np.array([*pos_error, *vel_error, *ang_error, *ang_vel_error])


    def reset(self, seed=None, options=None):
        ''' Reset the environment '''
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        # Randomize spawn point
        #x_random = np.random.uniform(-2, 2)
        #y_random = np.random.uniform(-2, 2)
        #z_random = np.random.uniform(0.5, 2)
        #self.data.qpos[0:3] = np.array([x_random, y_random, z_random])
        
        return self._get_obs(), {}


    def render(self):
        ''' Render the environment in real-time '''

        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Camera Settings
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.azimuth = 45
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 4.0
            self.viewer.cam.lookat[:] = self.data.qpos[:3]

            self.viewer.sync()


    def close(self):
        ''' Close the environment and free memory '''
        if self.viewer is not None:
            self.viewer = None  # Free up resources


    ''' Helper functions '''
    def quat_mul(self, q1, q2):

        q3_0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q3_1 = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q3_2 = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q3_3 = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        return np.array([q3_0, q3_1, q3_2, q3_3])
    

    def quat_conj(self, q):

        return np.array([q[0], -q[1], -q[2], -q[3]])