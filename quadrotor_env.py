import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as rot

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
            low=0, high=10, shape=(11,), dtype=np.float32
        )

        self.render_mode = render_mode
        self.viewer = None  # Viewer is not created until needed
        self.target_position = np.array([0, 0, 1])  # Hover position
        self.target_yaw = 0  # Desired yaw  
        self.max_steps = int(20/self.model.opt.timestep)
        self.current_step = 0

        # Additional variables
        self.e_i = np.zeros(3)
        self.acc_des = np.zeros(3)
        self.thrust = 0
        self.torque = np.zeros(3)
        self.motor_force = np.zeros(4)


    def step(self, action):
        '''Peforms a step in the environment'''

        # Flight Controller
        if self.current_step % 10 == 0:
            self.position_controller(action)
        self.attitude_controller(action)
        self.mixer()

        # Send motor force data as control input ()
        self.data.ctrl[:] = self.motor_force

        # Step MuJoCo simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Render if real-time visualization is enabled
        if self.render_mode == "human":
            self.render()
            
        # Compute reward (hover stability, low control effort)
        reward = -np.linalg.norm(self.target_position - self.data.qpos[0:3]) - 0.1 * np.linalg.norm(self.data.qvel[0:3])

        # Check termination
        done = self.current_step >= self.max_steps or self.data.qpos[2] <= 0.025

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

        # Randomize spawn point
        x_random = np.random.uniform(-1, 1)
        y_random = np.random.uniform(-1, 1)
        z_random = np.random.uniform(0.5, 2)
        self.data.qpos[0:3] = np.array([x_random, y_random, z_random])
        
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

    
    ''' Flight Controller Functions '''
    def position_controller(self, action):

        # Extract gains
        Kp_x, Kp_y, Kp_z = action[0:3]
        Ki_x, Ki_y, Ki_z = action[3:6]
        Kd_x, Kd_y, Kd_z = action[6:9]

        # Errors
        e_p = self.target_position - self.data.qpos[0:3]
        self.e_i += e_p * self.model.opt.timestep
        e_d = -self.data.qvel[0:3]

        # Desired acceleration
        x_acc = Kp_x * e_p[0] + Ki_x * self.e_i[0] + Kd_x * e_d[0]
        y_acc = Kp_y * e_p[1] + Ki_y * self.e_i[1] + Kd_y * e_d[1]
        z_acc = Kp_z * e_p[2] + Ki_z * self.e_i[2] + Kd_z * e_d[2]
        self.acc_des = np.array([x_acc, y_acc, z_acc]) - self.model.opt.gravity

        # Desired thrust
        self.thrust = sum(self.model.body_mass) * self.acc_des[2]

    
    def attitude_controller(self, action):

        # Extract gains
        Kq = action[9]
        Kw = action[10]

        # Desired orientation
        z_b = self.acc_des
        if np.linalg.norm(z_b) > 1e-6:
            z_b /= np.linalg.norm(z_b)
        else:
            z_b = np.array([0, 0, 1])
        z_b /= np.linalg.norm(z_b)
        x_c = np.array([np.cos(self.target_yaw), np.sin(self.target_yaw), 0])
        y_b = np.cross(z_b, x_c) / np.linalg.norm(np.cross(z_b, x_c))
        x_b = np.cross(y_b, z_b)
        R = rot.from_matrix(np.array([x_b, y_b, z_b]).T)
        q_desired = R.as_quat()
        q_desired /= np.linalg.norm(q_desired)
        q_desired = rot.from_quat(q_desired)

        # Error
        q_current = rot.from_quat(self.data.qpos[3:7])
        q_error = q_desired * q_current.inv()
        q_error = rot.as_quat(q_error)
        q_error /= np.linalg.norm(q_error)

        # Required torque
        self.torque = - Kq * q_error[1:4] - Kw * self.data.qvel[3:6]

    
    def mixer(self):
        
        l = 0.141
        k = 0.1
        mixer_mat = np.array([[1, 1, 1, 1],
                               [l, l, -l, -l],
                               [-l, l, l, -l],
                               [-k, k, -k, k]])
        
        input_vec = np.array([self.thrust, *self.torque])

        self.motor_force = np.clip(np.linalg.solve(mixer_mat, input_vec), 0.0, 1.0)

