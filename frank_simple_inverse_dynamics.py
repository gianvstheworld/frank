#!/usr/bin/env python3

import mujoco_py
import glfw
import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import spatialmath as smath

# ---

class SimulationHandler:
    """Class for handling MuJoCo simulation and visualization"""

    def __init__(self):
        """Initialize simulation environment handler"""

        try:
            model = mujoco_py.load_model_from_path("./assets/frank-robot.xml")

            self.handler = mujoco_py.MjSim(model)
            self.viewer  = mujoco_py.MjViewer(self.handler)
        except Exception as e:
            raise RuntimeError(f"MuJoCo initialization failed: {e}")

        self.duration: float = 60
        self.timestep: float = self.handler.model.opt.timestep
        self.num_steps = int(self.duration / self.timestep)

    def apply_ctrl_action(self, u):
        self.handler.data.ctrl[:] = u

    def apply_external_force(self, apply, ext_force_direction):
        # Get the body index of the body where the force will be applied
        body_name = "frank_robot_ee"
        body_id = self.handler.model.body_name2id(body_name)

        f = 140.0

        if apply:
            # Define the force and torque you want to apply (in the body frame)
            if ext_force_direction == 0:  # Move robot along the y-axis
                force = np.array([f, 0.0, 0.0])  # Apply a force of 100 N in the x direction
            elif ext_force_direction == 1:  # Move robot along the y-axis
                force = np.array([-f, 0.0, 0.0])  # Apply a force of -100 N in the x direction
            elif ext_force_direction == 2:  # Move robot along the y-axis
                force = np.array([0.0, f, 0.0])  # Apply a force of 100 N in the y direction
            elif ext_force_direction == 3:  # Move robot along the y-axis
                force = np.array([0.0, -f, 0.0])  # Apply a force of -100 N in the y direction
            elif ext_force_direction == 4:  # Move robot along the y-axis
                force = np.array([0.0, 0.0, f])  # Apply a force of 100 N in the z direction
            elif ext_force_direction == 5:  # Move robot along the y-axis
                force = np.array([0.0, 0.0, -f])  # Apply a force of -100 N in the z direction

            torque = np.array([0.0, 0.0, 0.0])  # No torque applied
        else:
            # Define the force and torque you want to apply (in the body frame)
            force = np.array([0.0, 0, 0.0])  # Apply a force of 10 N in the x direction
            torque = np.array([0.0, 0.0, 0.0])  # No torque applied

        # Apply the force (and torque) to the body at the current simulation step
        self.handler.data.xfrc_applied[body_id] = np.concatenate([force, torque])

    def step(self):
        """
        Advance simulation by one timestep and Takes all forces and disturbances
        in the system and calculates the dynamics of the model
        """

        self.handler.step()
        self.viewer.render()

class GLFWHandler:
    def __init__(self):
        # Set up GLFW for keyboard and mouse input
        if not glfw.init():
            print("GLFW initialization failed!")
            exit(1)
        self.window = glfw.create_window(200, 200, "MuJoCo Robot Control", None, None)
        if not self.window:
            print("Failed to create GLFW window.")
            glfw.terminate()
            exit(1)
        glfw.make_context_current(self.window)

        # Set callbacks for keyboard and mouse events
        glfw.set_key_callback(self.window, lambda window, key, scancode, action, mods: self.handle_keyboard_input(key))
        glfw.set_mouse_button_callback(self.window, self.handle_mouse_button_input)

        self.actuate: bool = False
        self.ext_force_direction: int = 0

    def should_close(self):
        return glfw.window_should_close(self.window)

    def process_events(self):
        """Process GLFW events"""

        glfw.poll_events()

        glfw.make_context_current(self.window)
        glfw.swap_buffers(self.window)

    def handle_mouse_button_input(self, sim, button, action, mods):
        """Handle mouse inputs from the GLFW window"""

        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.actuate = True
            elif action == glfw.RELEASE:
                self.actuate = False

    def handle_keyboard_input(self, key):
        """
        Handle keyboard inputs from the GLFW window

        Parameters
        ----------
        key
            GLFW key pressed
        """

        if key == glfw.KEY_RIGHT:
            self.ext_force_direction = 0 # Move robot along the x-axis
        elif key == glfw.KEY_LEFT:
            self.ext_force_direction = 1 # Move robot along the -x-axis
        elif key == glfw.KEY_UP:
            self.ext_force_direction = 2 # Move robot along the y-axis
        elif key == glfw.KEY_DOWN:
            self.ext_force_direction = 3 # Move robot along the -y-axis
        elif key == glfw.KEY_KP_1:
            self.ext_force_direction = 4 # Move robot along the z-axis
        elif key == glfw.KEY_KP_0:
            self.ext_force_direction = 5 # Move robot along the -z-axis

    def close(self):
        glfw.terminate()

class KinematicsFrank:
    """Kinematics model for the FRANK robot"""

    def __init__(self, simulation: SimulationHandler):
        """
        Initialize kinematics with given parameters.
        """

        # Define the Denavit-Hartenberg parameters for each link of the robot
        self.link1 = rtb.robot.DHLink(d=0.080+0.0751502092, a=0.287331693, alpha=np.pi/2, offset=0,              qlim=np.radians([-180, 180]))
        self.link2 = rtb.robot.DHLink(d=0,                  a=0.8,         alpha=0,       offset=np.radians(90), qlim=np.radians([-180, 180]))
        self.link3 = rtb.robot.DHLink(d=0.0628881,          a=0.8,         alpha=0,       offset=0,              qlim=np.radians([-180, 180]))
        # √((0.0225)² + (0.3165)²) = 0.317298755, atan(0.025/0.3165) = 4.516355661°
        self.link4 = rtb.robot.DHLink(d=0,                  a=0.317298755, alpha=0,       offset=np.radians(-4.516355661), qlim=np.radians([-180, 180]))

        # Create the robot by combining the individual DH links into a robot model
        self.robot = rtb.DHRobot([self.link1, self.link2, self.link3, self.link4])

        self.last_joint_angles = np.zeros(4)

        self.simulation = simulation.handler

    def inverse_kinematics(self, position: np.ndarray) -> np.ndarray:
        """
        Compute inverse kinematics for given end-effector position.

        Parameters:
        -----------
        position : np.ndarray, shape (3,)
            Desired end-effector position in Cartesian space (x,y,z)

        Returns:
        --------
        np.ndarray, shape (4,)
            Joint angles (θ1, θ2, θ3, θ4) that achieve desired position
        """

        theta1, theta2, theta3, theta4 = np.zeros(4)

        self.last_joint_angles = self.simulation.data.qpos[:4]
        try:
            target_pose = smath.SE3(position[0], position[1], position[2])

            q_solution = self.robot.ikine_LM(target_pose, self.last_joint_angles, mask=[1, 1, 1, 0, 0, 0], joint_limits=True)
            self.last_joint_angles = q_solution
        except Exception as e:
            print(f"KinematicsFrank::inverse_kinematics() - {e}")

        return self.last_joint_angles

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint angles.

        Parameters:
        -----------
        joint_angles : np.ndarray, shape (4,)
            Joint angles (θ1, θ2, θ3, θ4)

        Returns:
        --------
        np.ndarray, shape (3,)
            End-effector position in Cartesian space (x,y,z)
        """

        px, py, pz = np.zeros(3)

        try:
            # Calculate the forward kinematics
            end_effector_position = self.robot.fkine(joint_angles)
            # Compute end-effector position
            px, py, pz = end_effector_position.t

            self.last_joint_angles = joint_angles
        except Exception as e:
            print(f"KinematicsFrank::forward_kinematics() - {e}")

        return np.array([px, py, pz])

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the end-effector Jacobian matrix for current joint configuration.

        Parameters:
        -----------
        joint_angles : np.ndarray
            Joint angles (3D)

        Returns:
        --------
        np.ndarray
            6x4 Jacobian matrix
        """
        return self.robot.jacobe(joint_angles)

    def get_joint_velocity(self, joint_angles: np.ndarray, desired_velocity: np.ndarray) -> np.ndarray:
        """
        Compute desired joint velocities for given end-effector velocity.

        Parameters:
        -----------
        joint_angles : np.ndarray, shape (4,)
            Current joint angles
        desired_velocity : np.ndarray, shape (3,)
            Desired end-effector velocity

        Returns:
        --------
        np.ndarray, shape (4,)
            Required joint velocities
        """

        J = self.jacobian(joint_angles)

        J_pinv = np.linalg.pinv(J.T @ J) @ J.T

        return J_pinv @ desired_velocity

class Controller:
    """Implements methods to control the robot"""

    def __init__(self, simulation: SimulationHandler):
        """
        Initialize controller with simulation handler

        Parameters
        ----------
        simulation : SimulationManager
            Simulation interface
        """

        self.simulation = simulation.handler
        self.J_ant = None

    def jointspace_inverse_dynamics(self, qpos_ref, qvel_ref, qacc_ref):
        """
        Compute control torques with jointspace inverse dynamics

        Parameters
        ----------
        qacc_ref : np.ndarray, shape (4,)
            Desired accelaration
        """

        # Inertia matrix
        M = np.zeros(self.simulation.model.nv * self.simulation.model.nv)
        mujoco_py.functions.mj_fullM(self.simulation.model, M, self.simulation.data.qM)
        M = M.reshape(self.simulation.model.nv, self.simulation.model.nv)

        # Coriolis + gravitational
        b = self.simulation.data.qfrc_bias

        # ---

        # Current joint state
        qpos = self.simulation.data.qpos[:4]
        qvel = self.simulation.data.qvel[:4]

        K = np.array([
            [30.0, 0.0, 0.0, 0.0],
            [0.0, 30.0, 0.0, 0.0],
            [0.0, 0.0, 30.0, 0.0],
            [0.0, 0.0, 0.0, 80.0],
        ])
        B = np.array([
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 30.0],
        ])
        qpos_error = qpos_ref - qpos
        qvel_error = qvel_ref - qvel

        # ---

        v = qacc_ref + K @ qpos_error + B @ qvel_error

        # Siciliano - Springer Handbook of Robotics (section 6.5)
        tau = np.dot(M, v) + b

        return tau

    def calculate_torque(self, qpos_ref, qvel_ref, qacc_ref):
        return self.jointspace_inverse_dynamics(qpos_ref, qvel_ref, qacc_ref)

# ---

def main():
    """Main simulation execution loop."""

    try:
        # Initialize components
        simulation   = SimulationHandler() # Initialize simulation and viewer
        frank_robot  = KinematicsFrank(simulation) # Initialize robot kinematics
        controller   = Controller(simulation) # Initialize controller
        glfw_handler = GLFWHandler() # Initialize GLFW user input handler

        # ---

        # Main simulation loop
        for step in range(simulation.num_steps):
            if glfw_handler.should_close():
                break

            # Process GLFW events
            #glfw_handler.process_events()

            # ---

            # Apply external force
            #simulation.apply_external_force(glfw_handler.actuate, glfw_handler.ext_force_direction)

            # ---

            # Current simulation time
            t = step * simulation.timestep

            # Define a Y-Z circle in end-effector space (x, y, z)
            radius = 0.5
            target_x = radius * np.cos(t)
            target_y = radius * np.sin(t)
            target_z = 1.0

            # use inverse_kinematics to get q angles 
            qpos_d = frank_robot.inverse_kinematics(np.array([target_x, target_y, target_z]))

            qvel_d   = np.array([0.0, 0.0, 0.0, 0.0])
            qaccel_d = np.array([0.0, 0.0, 0.0, 0.0])

            # Compute torque control action
            u = controller.calculate_torque(qpos_ref=qpos_d.q, qvel_ref=qvel_d, qacc_ref=qaccel_d)

            # Apply control action
            simulation.apply_ctrl_action(u)

            # Print error between simulation data and calculated values
            print(f"IK Error [deg]: {np.rad2deg(frank_robot.inverse_kinematics(simulation.handler.data.get_site_xpos(simulation.handler.model.site_id2name(0))).q - simulation.handler.data.qpos[:4])}")
            print(f"FK Error [m]:   {frank_robot.forward_kinematics(simulation.handler.data.qpos[:4]) - simulation.handler.data.get_site_xpos(simulation.handler.model.site_id2name(0))} \n")

            # ---

            simulation.step()

    except KeyboardInterrupt:
        pass
    finally:
        # Close the GLFW window and terminate the simulation
        glfw_handler.close()

if __name__ == "__main__":
    main()