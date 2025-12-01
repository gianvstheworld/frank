#!/usr/bin/env python3
import mujoco
import glfw
import numpy as np
import matplotlib.pyplot as plt

# ---- quick monkeypatch for old roboticstoolbox expecting scipy.randn ----
import scipy
scipy.randn = np.random.randn  # fixes: "cannot import name randn from scipy"

import roboticstoolbox as rtb
import spatialmath as smath


# ============================================================
# Simple Matplotlib Stick Viewer + Tip Trail
# ============================================================
class SimpleMatplotlibViewer:
    """
    Stick-figure viewer using matplotlib.
    Plots base -> joint1 -> joint2 -> joint3 -> ee as a polyline,
    plus a persistent tip trail and a current tip marker.
    """

    def __init__(self, robot: rtb.DHRobot,
                 xlim=(-2, 2), ylim=(-2, 2), zlim=(-0.5, 2),
                 trail_max=3000):
        self.robot = robot
        self.trail_max = trail_max

        plt.ion()  # interactive mode ON
        self.fig = plt.figure("FRANK simple viewer")
        self.ax = self.fig.add_subplot(111, projection='3d')

        # robot polyline
        self.line, = self.ax.plot([], [], [], "-o", lw=2, markersize=4)

        # trail line
        self.trail_line, = self.ax.plot([], [], [], "-", lw=1)

        # current tip marker
        self.tip_marker, = self.ax.plot([], [], [], "ro", markersize=6)

        self.trail_pts = []

        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.ax.set_zlim(*zlim)
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_title("Stick Figure FRANK (DH model)")

        self.fig.tight_layout()

    def _joint_positions(self, q):
        """
        Returns array of shape (n_joints+1, 3):
        [base, joint1, ..., ee]
        """
        if hasattr(self.robot, "fkine_all"):
            Ts = self.robot.fkine_all(q)
            pts = np.array([T.t for T in Ts])
            return pts

        pts = [np.zeros(3)]
        for i in range(len(q)):
            Ti = self.robot.fkine(q, end=i+1)
            pts.append(np.array(Ti.t).reshape(3,))
        return np.vstack(pts)

    def update(self, q):
        pts = self._joint_positions(q)

        # update robot line
        self.line.set_data(pts[:, 0], pts[:, 1])
        self.line.set_3d_properties(pts[:, 2])

        # tip position (end-effector)
        tip = pts[-1].copy()

        # append to trail
        self.trail_pts.append(tip)
        if len(self.trail_pts) > self.trail_max:
            self.trail_pts = self.trail_pts[-self.trail_max:]

        trail_arr = np.array(self.trail_pts)

        # update trail
        self.trail_line.set_data(trail_arr[:, 0], trail_arr[:, 1])
        self.trail_line.set_3d_properties(trail_arr[:, 2])

        # update current tip marker
        self.tip_marker.set_data([tip[0]], [tip[1]])
        self.tip_marker.set_3d_properties([tip[2]])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# ============================================================
# MuJoCo Simulation Handler
# ============================================================
class SimulationHandler:
    """Class for handling MuJoCo simulation"""

    def __init__(self):
        try:
            self.model = mujoco.MjModel.from_xml_path(
                r"D:\Antigravity Google\frank-mujoco-files\assets\frank-robot.xml"
            )
            self.data = mujoco.MjData(self.model)

            # make sure all derived quantities (site positions etc.) are valid
            mujoco.mj_forward(self.model, self.data)

        except Exception as e:
            raise RuntimeError(f"MuJoCo initialization failed: {e}")

        self.duration: float = 60.0
        self.timestep: float = self.model.opt.timestep
        self.num_steps = int(self.duration / self.timestep)

    def apply_ctrl_action(self, u):
        self.data.ctrl[:len(u)] = u

    def apply_external_force(self, apply, ext_force_direction):
        body_name = "frank_robot_ee"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        f = 140.0
        if apply:
            if ext_force_direction == 0:
                force = np.array([ f, 0.0, 0.0])
            elif ext_force_direction == 1:
                force = np.array([-f, 0.0, 0.0])
            elif ext_force_direction == 2:
                force = np.array([0.0,  f, 0.0])
            elif ext_force_direction == 3:
                force = np.array([0.0, -f, 0.0])
            elif ext_force_direction == 4:
                force = np.array([0.0, 0.0,  f])
            elif ext_force_direction == 5:
                force = np.array([0.0, 0.0, -f])
            torque = np.array([0.0, 0.0, 0.0])
        else:
            force  = np.zeros(3)
            torque = np.zeros(3)

        self.data.xfrc_applied[body_id] = np.concatenate([force, torque])

    def step(self):
        mujoco.mj_step(self.model, self.data)


# ============================================================
# GLFW Handler (optional small window for keyboard/mouse input)
# ============================================================
class GLFWHandler:
    def __init__(self):
        if not glfw.init():
            print("GLFW initialization failed!")
            exit(1)

        self.window = glfw.create_window(200, 200, "MuJoCo Robot Control", None, None)
        if not self.window:
            print("Failed to create GLFW window.")
            glfw.terminate()
            exit(1)

        glfw.make_context_current(self.window)

        glfw.set_key_callback(
            self.window,
            lambda window, key, scancode, action, mods: self.handle_keyboard_input(key)
        )
        glfw.set_mouse_button_callback(self.window, self.handle_mouse_button_input)

        self.actuate: bool = False
        self.ext_force_direction: int = 0

    def should_close(self):
        return glfw.window_should_close(self.window)

    def process_events(self):
        glfw.poll_events()
        glfw.make_context_current(self.window)
        glfw.swap_buffers(self.window)

    def handle_mouse_button_input(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.actuate = (action == glfw.PRESS)

    def handle_keyboard_input(self, key):
        if key == glfw.KEY_RIGHT:
            self.ext_force_direction = 0
        elif key == glfw.KEY_LEFT:
            self.ext_force_direction = 1
        elif key == glfw.KEY_UP:
            self.ext_force_direction = 2
        elif key == glfw.KEY_DOWN:
            self.ext_force_direction = 3
        elif key == glfw.KEY_KP_1:
            self.ext_force_direction = 4
        elif key == glfw.KEY_KP_0:
            self.ext_force_direction = 5

    def close(self):
        glfw.terminate()


# ============================================================
# FRANK Kinematics (DH)
# ============================================================
class KinematicsFrank:
    """Kinematics model for the FRANK robot"""

    def __init__(self, simulation: SimulationHandler):
        self.link1 = rtb.robot.DHLink(
            d=0.080 + 0.0751502092, a=0.287331693, alpha=np.pi/2,
            offset=0, qlim=np.radians([-180, 180])
        )
        self.link2 = rtb.robot.DHLink(
            d=0, a=0.8, alpha=0,
            offset=np.radians(90), qlim=np.radians([-180, 180])
        )
        self.link3 = rtb.robot.DHLink(
            d=0.0628881, a=0.8, alpha=0,
            offset=0, qlim=np.radians([-180, 180])
        )
        self.link4 = rtb.robot.DHLink(
            d=0, a=0.317298755, alpha=0,
            offset=np.radians(-4.516355661), qlim=np.radians([-180, 180])
        )

        self.robot = rtb.DHRobot([self.link1, self.link2, self.link3, self.link4])
        self.last_joint_angles = np.zeros(4)
        self.sim = simulation

    def inverse_kinematics(self, position: np.ndarray) -> np.ndarray:
        self.last_joint_angles = self.sim.data.qpos[:4].copy()
        try:
            target_pose = smath.SE3(position[0], position[1], position[2])
            q_solution = self.robot.ikine_LM(
                target_pose,
                self.last_joint_angles,
                mask=[1, 1, 1, 0, 0, 0],   # position-only IK
                joint_limits=True
            )
            self.last_joint_angles = np.array(q_solution.q).flatten()
        except Exception as e:
            print(f"KinematicsFrank::inverse_kinematics() - {e}")

        return self.last_joint_angles

    def forward_kinematics(self, joint_angles: np.ndarray) -> np.ndarray:
        try:
            end_effector_position = self.robot.fkine(joint_angles)
            px, py, pz = end_effector_position.t
            self.last_joint_angles = joint_angles
            return np.array([px, py, pz])
        except Exception as e:
            print(f"KinematicsFrank::forward_kinematics() - {e}")
            return np.zeros(3)

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        return self.robot.jacobe(joint_angles)


# ============================================================
# Inverse Dynamics Controller
# ============================================================
class Controller:
    """Implements inverse dynamics control"""

    def __init__(self, simulation: SimulationHandler,
                 K=None, B=None):
        self.sim = simulation

        # default gains (you can tweak)
        if K is None:
            K = np.diag([50.0, 50.0, 50.0, 80.0])
        if B is None:
            B = np.diag([20.0, 20.0, 20.0, 50.0])

        self.K = K
        self.B = B

    def jointspace_inverse_dynamics(self, qpos_ref, qvel_ref, qacc_ref):
        nv = self.sim.model.nv

        # dense inertia matrix (nv x nv)
        M = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(self.sim.model, M, self.sim.data.qM)

        # bias forces (coriolis + gravity)
        b = self.sim.data.qfrc_bias.copy()

        qpos = self.sim.data.qpos[:4].copy()
        qvel = self.sim.data.qvel[:4].copy()

        qpos_error = qpos_ref - qpos
        qvel_error = qvel_ref - qvel

        v = qacc_ref + self.K @ qpos_error + self.B @ qvel_error

        tau = M[:4, :4] @ v + b[:4]
        return tau

    def calculate_torque(self, qpos_ref, qvel_ref, qacc_ref):
        return self.jointspace_inverse_dynamics(qpos_ref, qvel_ref, qacc_ref)


    # ------------------ NUEVO: control en espacio de tarea ------------------ #
    def taskspace_inverse_dynamics(self, x_ref, xdot_ref=None, xddot_ref=None, site_id=0):
        """
        x_ref      : posición deseada del EE (3,)
        xdot_ref   : velocidad deseada (3,) (si None -> 0)
        xddot_ref  : aceleración deseada (3,) (si None -> 0)
        site_id    : id del site del EE en MuJoCo
        """
        model = self.sim.model
        data  = self.sim.data
        nv = model.nv

        if xdot_ref is None:
            xdot_ref = np.zeros(3)
        if xddot_ref is None:
            xddot_ref = np.zeros(3)

        # Estado actual en task space
        x = data.site_xpos[site_id].copy()
        xdot = data.site_xvelp[site_id].copy()

        # Jacobiano lineal del site
        Jp = np.zeros((3, nv))
        mujoco.mj_jacSite(model, data, Jp, None, site_id)
        J = Jp[:, :4]   # solo 4 primeras articulaciones

        # Errores
        x_err = x_ref - x
        xdot_err = xdot_ref - xdot

        # Aceleración deseada en task space (PD + feedforward)
        xddot_des = xddot_ref + self.Kx @ x_err + self.Bx @ xdot_err

        # Aproximación: J qdd ≈ xddot_des  ->  qdd ≈ J^+ xddot_des
        # (ignoramos Jdot * qdot)
        qacc_des = np.linalg.pinv(J) @ xddot_des

        # Dinámica inversa en joint space
        M = np.zeros((nv, nv), dtype=np.float64)
        mujoco.mj_fullM(model, M, data.qM)
        b = data.qfrc_bias.copy()

        tau = M[:4, :4] @ qacc_des + b[:4]

        return tau



#########TRAJECTORY PLANNER

class LineWithZDropTrajectory:
    """
    Trajectory generator that can do:
    - a back-and-forth line with Z drop (mode='line')
    - a small circle in XY with constant Z (mode='circle')

    It converts task-space references to joint-space via IK.
    """

    def __init__(self, kin: KinematicsFrank, sim: SimulationHandler,
                 site_id=0,
                 line_length=0.1,
                 T_line=1.0,
                 z_final_scale=0.8,
                 t_drop_start=5.0,
                 t_drop_duration=3.0,
                 mode: str = "line"):   # <-- NEW: 'line' or 'circle'
        self.kin = kin
        self.sim = sim
        self.dt = sim.timestep
        self.mode = mode

        # Initial site position
        p0 = sim.data.site_xpos[site_id].copy()
        self.p0 = p0.copy()

        
        # --------- parameters shared / line-specific ---------
        # Line direction and endpoints (used if mode == 'line')
        direction = np.array([1.0, 0.0, 0.0])
        direction /= np.linalg.norm(direction)

        self.p_start = p0 - 0.5 * line_length * direction
        self.p_end   = p0 + 0.5 * line_length * direction

        # Z-drop parameters (used if mode == 'line')
        self.z0 = p0[2]
        self.z_final_scale = z_final_scale
        self.t_drop_start = t_drop_start
        self.t_drop_duration = t_drop_duration

        # Temporal parameters for both modes
        self.T_line = T_line
        self.omega = 2 * np.pi / T_line  # rad/s

        # --------- circle-specific parameters ---------
        # We'll interpret line_length as circle radius when mode == 'circle'
        self.circle_center = p0.copy()
        self.circle_radius = line_length  # small radius, e.g., 0.05–0.1 m

        # For numerical q̇ and q̈
        self.q_prev = self.sim.data.qpos[:4].copy()
        self.qvel_prev = np.zeros(4)

        # Last task-space reference
        self.last_pd = p0.copy()

    # ========================= line helpers =========================
    def _alpha(self, t: float) -> float:
        """Smooth 0 -> 1 -> 0 using a cosine profile (for line mode)."""
        return 0.5 * (1.0 - np.cos(self.omega * t))

    def _z_scale(self, t: float) -> float:
        """Piecewise-linear scale from 1.0 to z_final_scale (for line mode)."""
        if t < self.t_drop_start:
            s = 0.0
        elif t > self.t_drop_start + self.t_drop_duration:
            s = 1.0
        else:
            s = (t - self.t_drop_start) / self.t_drop_duration
        return 1.0 - (1.0 - self.z_final_scale) * s

    # ========================= circle helper =========================
    def _circle_point(self, t: float) -> np.ndarray:
        """Computes XY circle point at time t, with constant Z."""
        theta = self.omega * t  # full turn in T_line seconds

        pd = self.circle_center.copy()
        pd[0] += self.circle_radius * np.cos(theta)  # X
        pd[1] += self.circle_radius * np.sin(theta)  # Y
        pd[2] = self.z0                               # Z constant

        return pd

    # ========================= main API =========================
    def get_refs(self, t: float):
        """
        Returns:
            q_ref, qd_ref, qdd_ref (all np.array of shape (4,))
        """

        # --------- 1) Task-space reference pd(t) ---------
        if self.mode == "circle":
            # small circle in XY, constant Z
            pd = self._circle_point(t)
        else:
            # default: line with Z-drop
            alpha = self._alpha(t)

            # Base line in XY
            pd = self.p_start + alpha * (self.p_end - self.p_start)

            # Z drop
            z_scale = self._z_scale(t)
            pd[2] = self.z0 * z_scale

        # store ideal task-space reference
        self.last_pd = pd.copy()

        # --------- 2) IK -> joint pos (raw) ---------
        qpos_d_raw = self.kin.inverse_kinematics(pd)

        # --------- 3) limit joint step size ---------
        max_delta_q = np.radians(10.0)  # max 10 deg per step
        dq = qpos_d_raw - self.q_prev
        dq_norm = np.linalg.norm(dq, ord=np.inf)  # max(|dq_i|)

        if dq_norm > max_delta_q:
            dq = dq * (max_delta_q / dq_norm)
            qpos_d = self.q_prev + dq
        else:
            qpos_d = qpos_d_raw

        # --------- 4) numerical joint velocities / accelerations ---------
        qvel_d = (qpos_d - self.q_prev) / self.dt
        qaccel_d = (qvel_d - self.qvel_prev) / self.dt

        # update stored previous values
        self.q_prev = qpos_d.copy()
        self.qvel_prev = qvel_d.copy()

        return qpos_d, qvel_d, qaccel_d


# ============================================================
# Main
# ============================================================
def main():
    simulation   = SimulationHandler()
    frank_robot  = KinematicsFrank(simulation)
    controller   = Controller(simulation)
    glfw_handler = GLFWHandler()

    simple_view = SimpleMatplotlibViewer(
        frank_robot.robot,
        xlim=(-2, 2), ylim=(-2, 2), zlim=(-0.5, 2),
        trail_max=4000
    )

    decim = 5

    # ---- NEW: trajectory generator ----
    # Ensure mode="line" is used here if you want the line trajectory
    traj_gen = LineWithZDropTrajectory(
        kin=frank_robot,
        sim=simulation,
        site_id=0,
        line_length=0.1,
        T_line=1.0,
        z_final_scale=0.8,
        t_drop_start=5.0,
        t_drop_duration=3.0,
        mode="line"  # <-- Make sure this is "line" or omitted (it defaults to line)
    )

    # ---------------------------------------------------------------
    # 1. INITIALIZE LISTS FOR DATA COLLECTION 📊
    # ---------------------------------------------------------------
    ideal_trajectory = []
    actual_trajectory = []
    site_id = 0 # End-effector site ID
    # ---------------------------------------------------------------

    # ------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------
    for k in range(simulation.num_steps):
        if glfw_handler.should_close():
            break

        glfw_handler.process_events()
        simulation.apply_external_force(glfw_handler.actuate,
                                        glfw_handler.ext_force_direction)

        t = k * simulation.timestep

        # ----- Get desired references -----
        qpos_d, qvel_d, qaccel_d = traj_gen.get_refs(t)

        # control
        u = controller.calculate_torque(qpos_ref=qpos_d,
                                        qvel_ref=qvel_d,
                                        qacc_ref=qaccel_d)
        simulation.apply_ctrl_action(u)

        simulation.step()

        # 2. COLLECT DATA INSIDE THE LOOP 💾
        # ---------------------------------------------------------------
        # Ideal path (from trajectory generator)
        ideal_pos_d = traj_gen.last_pd.copy()
        ideal_trajectory.append(ideal_pos_d)

        # Actual path (from simulation)
        actual_pos = simulation.data.site_xpos[site_id].copy()
        actual_trajectory.append(actual_pos)
        # ---------------------------------------------------------------


        if k % decim == 0:
            q_now = simulation.data.qpos[:4].copy()
            simple_view.update(q_now)

    glfw_handler.close()
    
    # ---------------------------------------------------------------
    # 3. PLOT THE COMPARISON 📈
    # ---------------------------------------------------------------
    ideal_trajectory = np.array(ideal_trajectory)
    actual_trajectory = np.array(actual_trajectory)

    plt.ioff() 
    # Use a regular figure, not the 3D figure from the viewer
    plt.figure("Trajectory Comparison (XY Plane)", figsize=(10, 6))
    
    # Plot Ideal (Desired) Path (X vs Y)
    plt.plot(
        ideal_trajectory[:, 0], ideal_trajectory[:, 1], 
        '--', label='Ideal (Desired) Trajectory', color='blue', linewidth=2
    )
    
    # Plot Actual (Simulated) Path (X vs Y)
    plt.plot(
        actual_trajectory[:, 0], actual_trajectory[:, 1], 
        '-', label='Actual (Simulated) Trajectory', color='red', alpha=0.6
    )

    # Highlight the initial starting position
    # The line mode uses self.p0 as the center point of the motion
    center_x = traj_gen.p0[0]
    center_y = traj_gen.p0[1]
    plt.plot(center_x, center_y, 'ok', label='Initial Position / Line Center', markersize=6)
    
    # Highlight the start/end points of the line
    plt.plot(traj_gen.p_start[0], traj_gen.p_start[1], 'go', label='Line Start', markersize=5)
    plt.plot(traj_gen.p_end[0], traj_gen.p_end[1], 'rx', label='Line End', markersize=8)

    plt.title('Comparison of Ideal vs. Simulated End-Effector Trajectories (XY Plane)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show() 
    # ---------------------------------------------------------------

if __name__ == "__main__":
    main()