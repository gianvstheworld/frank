#!/usr/bin/env python3
"""
Run Frank robot experiments and record to LeRobot dataset.

Usage:
    python run_lerobot_experiment.py
    
Visualize:
    python -m lerobot.scripts.visualize_dataset \
        --repo-id local/frank_load_experiments \
        --root ./lerobot_data
"""

from frank_simple_inverse_dynamics import RobotController
from frank_lerobot_recorder import FrankLeRobotRecorder
import numpy as np


def run_experiment(force: float, recorder: FrankLeRobotRecorder):
    """Run a single experiment and record frames."""
    
    task = f"Sinusoidal trajectory tracking, {force}N external load"
    controller = RobotController(force=force)
    
    # Initial position
    controller.q_d = np.radians([0.0, 0.0, -90.0, 0.0])
    controller.dot_q_d = np.zeros(4)
    controller.ddot_q_d = np.zeros(4)
    controller.ext_force = np.zeros(3)
    
    # Wait for robot to settle (500 steps at 1kHz = 0.5s)
    for _ in range(500):
        controller.tau = controller.jointspace_inverse_dynamics(
            controller.q_d, controller.dot_q_d, controller.ddot_q_d
        )
        controller.apply_ctrl_action(controller.tau)
        controller.handler.step()
    
    # Run trajectory and record at 100Hz
    ext_force_start = int(1 * controller.n_traj_points / 3)
    ext_force_end = int(2 * controller.n_traj_points / 3)
    
    for traj_idx in range(controller.n_traj_points):
        # External force timing
        actuate = ext_force_start <= traj_idx < ext_force_end
        controller.apply_external_force(actuate)
        
        # Update desired state
        controller.q_d = controller.trajectory_pos[traj_idx]
        controller.dot_q_d = controller.trajectory_vel[traj_idx]
        controller.ddot_q_d = controller.trajectory_accel[traj_idx]
        
        # Compute control
        controller.tau = controller.jointspace_inverse_dynamics(
            controller.q_d, controller.dot_q_d, controller.ddot_q_d
        )
        controller.apply_ctrl_action(controller.tau)
        
        # Step simulation (10 steps per trajectory point = 1kHz / 100Hz)
        for _ in range(10):
            controller.handler.step()
        
        # Record frame
        recorder.add_frame(
            q=controller.handler.data.qpos[:4].copy(),
            dot_q=controller.handler.data.qvel[:4].copy(),
            q_d=controller.q_d,
            dot_q_d=controller.dot_q_d,
            ddot_q_d=controller.ddot_q_d,
            ext_force=controller.ext_force,
            tau=controller.tau,
            task=task,
        )
    
    recorder.save_episode()
    print(f"✓ Episode saved: {task}")


def main():
    force_conditions = [0, 10, 50, 500]
    
    # One recorder for all episodes
    recorder = FrankLeRobotRecorder()
    
    for i, force in enumerate(force_conditions):
        print(f"\n[{i+1}/{len(force_conditions)}] Running: {force}N")
        run_experiment(force, recorder)
    
    print("✓ All episodes recorded")


if __name__ == "__main__":
    main()
