"""
LeRobot Dataset Recorder for Frank Robot Simulations.

Usage:
    from frank_lerobot_recorder import FrankLeRobotRecorder
    
    recorder = FrankLeRobotRecorder()
    recorder.add_frame(q, dot_q, q_d, dot_q_d, ddot_q_d, ext_force, tau, task="...")
    recorder.save_episode()
    recorder.finalize()
"""

import numpy as np
import shutil
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "observation.target": {"dtype": "float32", "shape": (12,), "names": None},
    "observation.force": {"dtype": "float32", "shape": (3,), "names": None},
    "action": {"dtype": "float32", "shape": (4,), "names": None},
}


class FrankLeRobotRecorder:
    """Records Frank robot simulation data in LeRobot v3 format."""

    def __init__(
        self,
        dataset_name: str = "frank_load_experiments",
        root_dir: Path = Path("./lerobot_data"),
        fps: int = 100,
    ):
        # Clean up existing dataset
        dataset_path = root_dir / dataset_name
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        
        self.dataset = LeRobotDataset.create(
            repo_id=f"local/{dataset_name}",
            fps=fps,
            root=str(root_dir),
            robot_type="frank_4dof",
            features=FEATURES,
            use_videos=False,
        )

    def add_frame(
        self,
        q: np.ndarray,
        dot_q: np.ndarray,
        q_d: np.ndarray,
        dot_q_d: np.ndarray,
        ddot_q_d: np.ndarray,
        ext_force: np.ndarray,
        tau: np.ndarray,
        task: str,
    ) -> None:
        """Add a single frame to the current episode."""
        frame = {
            "observation.state": np.concatenate([q, dot_q]).astype(np.float32),
            "observation.target": np.concatenate([q_d, dot_q_d, ddot_q_d]).astype(np.float32),
            "observation.force": ext_force.astype(np.float32),
            "action": tau.astype(np.float32),
            "task": task,
        }
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        """Save buffered frames as an episode."""
        self.dataset.save_episode()

