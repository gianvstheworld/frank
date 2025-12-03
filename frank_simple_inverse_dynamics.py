#!/usr/bin/env python3

import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as smath
import time
import threading
import pandas as pd
import datetime

'''
Avaliação de Performance Dinâmica com Variação de Carga

    - Ideia:
        Adicionar cargas simples na ponta (0g, 200g, 500g, 1kg) e executar a mesma trajetória.

    - Medir:
        - atraso do efetuador
        - torques necessários
        - erro de rastreamento
        - overshoot

    - Resultado desejado:
        - dataset comparativo de performance
        - análise de robustez do manipulador
        - Gráficos de torque vs. carga
'''

# ---

class RobotController:
    def __init__(self, force):
        """Initialize simulation environment handler"""

        try:
            model = mujoco_py.load_model_from_path("./assets/frank-robot.xml")

            self.handler = mujoco_py.MjSim(model)
            self.viewer  = mujoco_py.MjViewer(self.handler)
        except Exception as e:
            raise RuntimeError(f"MuJoCo initialization failed: {e}")

        self.duration: float = 10
        self.timestep: float = self.handler.model.opt.timestep # 1 kHz
        self.n_timesteps = int(self.duration / self.timestep)

        # Trajectory properties
        self.traj_timestep = 0.01 # 100 Hz
        self.n_traj_points = int(self.duration / self.traj_timestep)

        self._init_log_data()

        # Trajectory parameters
        self.init_trajectory()

        self.run_controller = True

        # Initial joint state
        self.q_d      = np.zeros(4)
        self.dot_q_d  = np.zeros(4)
        self.ddot_q_d = np.zeros(4)

        self.force_magnitude = force # [N]

    def _init_log_data(self) -> None:
        """Initialize log data strucutures."""

        self.data_sim = {
            # Time related
            'step': np.zeros(self.n_timesteps),
            'time': np.zeros(self.n_timesteps),
            # External force
            'apply_ext_force': np.zeros(self.n_timesteps),
            'ext_force': np.zeros((self.n_timesteps, 3))
        }

        self.data_robot = {
            # Current robot joints state
            "q": np.zeros((self.n_timesteps, 4)),
            "dot_q": np.zeros((self.n_timesteps, 4)),
            # Desired robot state
            "q_d":      np.zeros((self.n_timesteps, 4)),
            "dot_q_d":  np.zeros((self.n_timesteps, 4)),
            "ddot_q_d": np.zeros((self.n_timesteps, 4)),
            # Computed errors
            "e": np.zeros((self.n_timesteps, 4)),
            "dot_e": np.zeros((self.n_timesteps, 4)),
            # Robot torques
            "tau": np.zeros((self.n_timesteps, 4)),
        }

        self.current_step = 0

    def apply_ctrl_action(self, u):
        self.handler.data.ctrl[:] = u

    def apply_external_force(self, apply: bool):
        # Get the body index of the body where the force will be applied
        body_id = self.handler.model.body_name2id("frank_robot_ee")

        if apply:
            force = np.array([0.0, 0.0, -self.force_magnitude])
        else:
            force = np.zeros(3)
        torque = np.zeros(3)

        self.ext_force = force

        # Apply the force (and torque) to the body at the current simulation step
        self.handler.data.xfrc_applied[body_id] = np.concatenate([force, torque])

    def init_trajectory(self) -> None:
        t_vals = np.linspace(0, self.duration, self.n_traj_points)
        frequency = self.duration * (2 * np.pi / self.duration) # 1 cycle per second
        angles = frequency * t_vals

        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        delta_position = np.radians(5)

        self.trajectory_pos = np.column_stack([
            delta_position * sin_angles,
            delta_position * sin_angles,
            delta_position * sin_angles - np.radians(90),
            delta_position * sin_angles
        ])

        self.trajectory_vel = np.column_stack([
            delta_position * frequency * cos_angles,
            delta_position * frequency * cos_angles,
            delta_position * frequency * cos_angles,
            delta_position * frequency * cos_angles
        ])

        self.trajectory_accel = np.column_stack([
            - delta_position * (frequency**2) * sin_angles,
            - delta_position * (frequency**2) * sin_angles,
            - delta_position * (frequency**2) * sin_angles,
            - delta_position * (frequency**2) * sin_angles
        ])

        self.traj_started = False

    def is_robot_still(self, threshold: float = 0.001) -> bool:
        """
        Check if robot has stopped moving.

        Parameters
        ----------
        threshold : float
            Velocity norm threshold

        Returns
        -------
        bool
            True if robot is stationary
        """

        return np.linalg.norm(self.handler.data.qvel) < threshold

    def log(self, step: int) -> None:
        """
        Log current simulation state.

        Parameters
        ----------
        step : int
            Current simulation step
        step_time : float
            Current simulation time
        """

        if step >= self.n_timesteps:
            return

        # -----

        # Time related
        current_time = time.perf_counter_ns()
        self.data_sim["step"][step] = step
        self.data_sim["time"][step] = current_time
        # External force
        self.data_sim["apply_ext_force"][step] = self.actuate
        self.data_sim["ext_force"][step] = self.ext_force

        # -----

        # Current robot joints state
        self.data_robot["q"][step]     = self.q
        self.data_robot["dot_q"][step] = self.dot_q
        # Desired robot joints state
        self.data_robot["q_d"][step]      = self.q_d
        self.data_robot["dot_q_d"][step]  = self.dot_q_d
        self.data_robot["ddot_q_d"][step] = self.ddot_q_d
        # Computed errors
        self.data_robot["e"][step]     = self.e
        self.data_robot["dot_e"][step] = self.dot_e
        # Robot torques
        self.data_robot["tau"][step] = self.tau

        # -----

        self.current_step = step + 1

    def save_log(self):
        """Save simulation data to a timestamped CSV file."""

        # Define valid data range with only populated data points
        valid_steps = slice(0, self.current_step)

        # ---

        def flatten_data_dict(data_dict) -> dict:
            """
            Convert multi-dimensional numpy arrays to flat dictionary structure.

            Parameters
            ----------
            data_dict : dict
                Dictionary containing numpy arrays of various dimensions

            Returns
            -------
            dict
                Flattened dictionary with 1D arrays suitable for pandas DataFrame
            """

            flat_dict = {}

            for key, value in data_dict.items():
                # Handle 1-dimensional arrays (e.g., time and manipulability)
                if value.ndim == 1:
                    flat_dict[f"{key}"] = value[valid_steps]

                # Handle 2-dimensional arrays (vectors and matrices)
                elif value.ndim == 2:
                    vector_shape = value.shape[1]
                    if vector_shape not in [3, 4]:
                        raise RuntimeError(f"Log data \"{key}\" has unknown shape [{vector_shape}]: {value}")

                    # 3D vectors (x, y, z)
                    list_3d_vectors = ["ext_force"]
                    # Joint vectors (q1, q2, q3)
                    list_joints_vectors = ["q", "dot_q", "q_d", "dot_q_d", "ddot_q_d", "e", "dot_e", "tau"]

                    if key in list_3d_vectors:
                        flat_dict[f"{key}_x"] = value[valid_steps, 0]
                        flat_dict[f"{key}_y"] = value[valid_steps, 1]
                        flat_dict[f"{key}_z"] = value[valid_steps, 2]
                    elif key in list_joints_vectors:
                        for i in range(vector_shape):
                            flat_dict[f"{key}_joint{i+1}"] = value[valid_steps, i]

            return flat_dict

        # ---

        # Flatten all data structures into pandas-compatible format
        flat_sim   = flatten_data_dict(self.data_sim)
        flat_robot   = flatten_data_dict(self.data_robot)

        # Combine all flattened dictionaries into a single data structure
        combined_data = {**flat_sim, **flat_robot}

        # Create pandas DataFrame from combined data
        combined_df = pd.DataFrame(combined_data)

        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"simulations_data/simulation_output_{timestamp}_f{self.force_magnitude}.csv"

        # Save DataFrame to CSV file without row indices
        combined_df.to_csv(filename, index=False)
        print(f"Simulation log saved to {filename}")

    def jointspace_inverse_dynamics(self, q_d, dot_q_d, ddot_q_d):
        """
        Compute control torques with jointspace inverse dynamics
        """

        # Inertia matrix
        H = np.zeros(self.handler.model.nv * self.handler.model.nv)
        mujoco_py.functions.mj_fullM(self.handler.model, H, self.handler.data.qM)
        H = H.reshape(self.handler.model.nv, self.handler.model.nv)

        # Coriolis + gravitational
        b = self.handler.data.qfrc_bias

        # ---

        # Current joint state
        self.q     = self.handler.data.qpos[:4].copy()
        self.dot_q = self.handler.data.qvel[:4].copy()

        self.e     = q_d - self.q
        self.dot_e = dot_q_d - self.dot_q

        k = 30.0
        Kp = np.diag([k, k, k, 3*k])
        Kv = 2 * np.sqrt(Kp)

        # Siciliano - Springer Handbook of Robotics (section 6.5)
        v = ddot_q_d + Kp @ self.e + Kv @ self.dot_e
        tau = H @ v + b

        return tau

    def control_loop(self) -> None:
        """
        Main control loop running in separate thread.
        """

        sim_step = 0

        while self.run_controller:
            loop_start = time.perf_counter()

            # ---

            self.tau = self.jointspace_inverse_dynamics(self.q_d, self.dot_q_d, self.ddot_q_d)
            self.apply_ctrl_action(self.tau)

            # ---

            # Log data
            if self.traj_started:
                self.log(sim_step)
                sim_step += 1

            self.handler.step()

            # ---

            # Maintain control rate
            elapsed = time.perf_counter() - loop_start
            if (sleep_time := self.timestep - elapsed) > 0:
                time.sleep(sleep_time)

    def trajectory_loop(self) -> None:
        try:
            # Start control thread
            control_thread = threading.Thread(target = self.control_loop)
            control_thread.start()

            # ---

            # Desired joint state
            self.q_d      = np.radians([0.0, 0.0, -90.0, 0.0]) # fully extended = [0.0, 0.0, 0.0, 0.0]
            self.dot_q_d  = np.array([0.0, 0.0, 0.0, 0.0])
            self.ddot_q_d = np.array([0.0, 0.0, 0.0, 0.0])

            time.sleep(0.5)

            # Waits for the robot to stop moving
            while self.is_robot_still() == False:
                self.viewer.render()

            # ---

            self.traj_started = True
            ext_force_start = int(1*self.n_traj_points/3)
            ext_force_end   = int(2*self.n_traj_points/3)

            # ---

            # Main simulation loop
            for trajectory_index in range(self.n_traj_points):
                loop_start = time.perf_counter()

                # ---

                # Apply external forces at specific times
                if trajectory_index >= ext_force_start and trajectory_index < ext_force_end:
                    self.actuate = True
                else:
                    self.actuate = False
                self.apply_external_force(self.actuate)

                # ---

                # Desired joint state
                self.q_d      = self.trajectory_pos[trajectory_index].copy()
                self.dot_q_d  = self.trajectory_vel[trajectory_index].copy()
                self.ddot_q_d = self.trajectory_accel[trajectory_index].copy()
                # self.q_d      = np.radians([0.0, 0.0, -90.0, 0.0])
                # self.dot_q_d  = np.array([0.0, 0.0, 0.0, 0.0])
                # self.ddot_q_d = np.array([0.0, 0.0, 0.0, 0.0])

                # ---

                self.viewer.render()

                # Maintain control rate
                elapsed = time.perf_counter() - loop_start
                if (sleep_time := self.traj_timestep - elapsed) > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass

        finally:
            self.save_log()

            self.run_controller = False
            control_thread.join()
            print("Simulation completed.")

# ---

if __name__ == "__main__":
    robot_controller = RobotController(force=10)
    robot_controller.trajectory_loop()