#!/usr/bin/env python3
"""
LeRobot Dataset Analysis Module

Generates metrics.json from a LeRobot dataset on HuggingFace.
This script fetches episode parquet files and calculates performance metrics
(RMS error, overshoot, settling time, etc.) for the comparison page.

Usage:
    python analysis_from_lerobot.py
    python analysis_from_lerobot.py --repo tommaselli/frank_load_experiments
    python analysis_from_lerobot.py --output metrics.json

Author: FRANK Robot Performance Study
"""

import numpy as np
import pandas as pd
import json
import re
import requests
import io
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Try to import pyarrow for parquet reading
try:
    import pyarrow.parquet as pq
except ImportError:
    print("Installing pyarrow...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyarrow"])
    import pyarrow.parquet as pq


@dataclass
class PerformanceMetrics:
    """Container for performance metrics of a single episode."""
    episode_id: int
    task_name: str
    force_magnitude: float
    rms_error: Dict[str, float]  # Per joint RMS error
    peak_torque: Dict[str, float]  # Per joint peak torque (from action)
    mean_torque: Dict[str, float]  # Per joint mean absolute torque
    overshoot_percent: Dict[str, float]  # Per joint overshoot percentage
    settling_time: Dict[str, float]  # Per joint settling time (seconds)
    steady_state_error: Dict[str, float]  # Per joint final error
    iae: Dict[str, float]  # Integral Absolute Error per joint
    peak_force: Dict[str, float]  # Peak force readings
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class LeRobotDatasetAnalyzer:
    """Analyzer for LeRobot datasets on HuggingFace."""
    
    def __init__(self, repo_id: str = "tommaselli/frank_load_experiments", branch: str = "main"):
        self.repo_id = repo_id
        self.branch = branch
        self.base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}"
        self.info = None
        self.episodes_meta = None
        self.tasks_meta = None
        
    def fetch_json(self, path: str) -> dict:
        """Fetch JSON file from HuggingFace."""
        url = f"{self.base_url}/{path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def fetch_parquet(self, path: str) -> pd.DataFrame:
        """Fetch and parse parquet file from HuggingFace."""
        url = f"{self.base_url}/{path}"
        response = requests.get(url)
        response.raise_for_status()
        
        # Read parquet from bytes
        buffer = io.BytesIO(response.content)
        table = pq.read_table(buffer)
        return table.to_pandas()
    
    def load_metadata(self):
        """Load dataset metadata."""
        print("Loading dataset metadata...")
        self.info = self.fetch_json("meta/info.json")
        
        # Load tasks metadata
        try:
            self.tasks_meta = self.fetch_parquet("meta/tasks.parquet")
            print(f"  Found {len(self.tasks_meta)} tasks")
        except Exception as e:
            print(f"  Warning: Could not load tasks metadata: {e}")
            self.tasks_meta = pd.DataFrame()
        
        # Load episodes metadata
        try:
            self.episodes_meta = self.fetch_parquet("meta/episodes/chunk-000/file-000.parquet")
            print(f"  Found {len(self.episodes_meta)} episodes")
        except Exception as e:
            print(f"  Warning: Could not load episodes metadata: {e}")
            self.episodes_meta = pd.DataFrame()
        
        print(f"  Dataset: {self.repo_id}")
        print(f"  Total episodes: {self.info.get('total_episodes', 'unknown')}")
        print(f"  Total frames: {self.info.get('total_frames', 'unknown')}")
        print(f"  FPS: {self.info.get('fps', 100)}")
    
    def get_task_name(self, episode_idx: int) -> str:
        """Get task name for an episode."""
        if self.episodes_meta is None or self.episodes_meta.empty:
            return f"Episode {episode_idx}"
        
        # Find the row for this episode
        ep_row = self.episodes_meta[self.episodes_meta['episode_index'] == episode_idx]
        if ep_row.empty:
            return f"Episode {episode_idx}"
        
        # Try to get task name from 'tasks' column (may be string, array, or index)
        if 'tasks' in self.episodes_meta.columns:
            task_value = ep_row.iloc[0]['tasks']
            
            # Handle numpy arrays (LeRobot stores as array with task name)
            if hasattr(task_value, '__iter__') and not isinstance(task_value, str):
                if len(task_value) > 0:
                    task_value = task_value[0]  # Get first element
            
            # If it's a string, use it directly
            if isinstance(task_value, str):
                return task_value
            # Otherwise, it's an index into tasks_meta
            try:
                task_idx = int(task_value)
                if self.tasks_meta is not None and not self.tasks_meta.empty:
                    return str(self.tasks_meta.index[task_idx])
            except (ValueError, IndexError, KeyError):
                pass
        
        # Fallback: try task_index column
        if 'task_index' in self.episodes_meta.columns:
            task_idx = int(ep_row.iloc[0]['task_index'])
            if self.tasks_meta is not None and not self.tasks_meta.empty:
                try:
                    return str(self.tasks_meta.index[task_idx])
                except (IndexError, KeyError):
                    pass
        
        return f"Episode {episode_idx}"
    
    def extract_force_from_task(self, task_name: str) -> float:
        """Extract force magnitude from task name (e.g., 'Sinusoidal trajectory with 10N' -> 10.0)"""
        match = re.search(r'(\d+)N', task_name)
        if match:
            return float(match.group(1))
        return 0.0
    
    def load_episode_data(self, episode_idx: int) -> pd.DataFrame:
        """Load episode data from parquet files."""
        # LeRobot v3 format: all episodes in chunked files
        chunks_size = self.info.get('chunks_size', 1000)
        chunk_idx = episode_idx // chunks_size
        
        # The data path uses chunk_index and file_index
        # In v3, typically file_index=0 for all data
        path = f"data/chunk-{chunk_idx:03d}/file-000.parquet"
        
        print(f"  Loading episode {episode_idx} from {path}...")
        df = self.fetch_parquet(path)
        
        # Filter for this specific episode
        if 'episode_index' in df.columns:
            df = df[df['episode_index'] == episode_idx].reset_index(drop=True)
        
        return df
    
    def analyze_episode(self, episode_idx: int) -> PerformanceMetrics:
        """Analyze a single episode and return performance metrics."""
        task_name = self.get_task_name(episode_idx)
        force_magnitude = self.extract_force_from_task(task_name)
        
        df = self.load_episode_data(episode_idx)
        fps = self.info.get('fps', 100)
        timestep = 1.0 / fps
        
        # Get feature shapes from info
        features = self.info.get('features', {})
        state_shape = features.get('observation.state', {}).get('shape', [8])
        target_shape = features.get('observation.target', {}).get('shape', [12])
        action_shape = features.get('action', {}).get('shape', [4])
        
        n_joints = action_shape[0]  # Number of joints = number of actions
        
        # Convert list columns to numpy arrays
        state = np.array(df['observation.state'].tolist())  # Shape: (N, 8) = 4 pos + 4 vel
        target = np.array(df['observation.target'].tolist())  # Shape: (N, 12) = 4 pos + 4 vel + 4 accel
        action = np.array(df['action'].tolist())  # Shape: (N, 4) = 4 torques
        
        # Handle force data (3D force vector)
        if 'observation.force' in df.columns:
            force = np.array(df['observation.force'].tolist())
        else:
            force = np.zeros((len(df), 3))
        
        # Calculate position error: state positions (first 4) - target positions (first 4)
        state_pos = state[:, :n_joints]  # First 4 elements are positions
        target_pos = target[:, :n_joints]  # First 4 elements are positions
        error = state_pos - target_pos
        
        # Calculate metrics per joint
        rms_error = {}
        peak_torque = {}
        mean_torque = {}
        overshoot_percent = {}
        settling_time = {}
        steady_state_error = {}
        iae = {}
        
        for j in range(n_joints):
            joint_name = f"joint{j+1}"
            joint_error = error[:, j]
            joint_target = target_pos[:, j]
            joint_action = action[:, j]
            
            # RMS Error
            rms_error[joint_name] = float(np.sqrt(np.mean(joint_error ** 2)))
            
            # Peak and Mean Torque (from action)
            peak_torque[joint_name] = float(np.max(np.abs(joint_action)))
            mean_torque[joint_name] = float(np.mean(np.abs(joint_action)))
            
            # Overshoot
            target_range = np.max(joint_target) - np.min(joint_target)
            if target_range > 0.01:
                max_overshoot = np.max(np.abs(joint_error))
                overshoot_percent[joint_name] = float((max_overshoot / target_range) * 100)
            else:
                overshoot_percent[joint_name] = 0.0
            
            # Settling time (time to reach within 2% of final value)
            threshold = 0.02 * (np.max(np.abs(joint_error)) + 0.001)
            steady_region = np.abs(joint_error) <= threshold
            if np.any(~steady_region):
                last_unsettled = np.where(~steady_region)[0][-1]
                settling_time[joint_name] = float(last_unsettled * timestep)
            else:
                settling_time[joint_name] = 0.0
            
            # Steady state error (mean error in last 10% of data)
            n_steady = max(1, len(joint_error) // 10)
            steady_state_error[joint_name] = float(np.mean(np.abs(joint_error[-n_steady:])))
            
            # IAE (Integral Absolute Error)
            iae[joint_name] = float(np.trapz(np.abs(joint_error), dx=timestep))
        
        # Peak force readings
        peak_force = {
            "x": float(np.max(np.abs(force[:, 0]))) if force.shape[1] > 0 else 0.0,
            "y": float(np.max(np.abs(force[:, 1]))) if force.shape[1] > 1 else 0.0,
            "z": float(np.max(np.abs(force[:, 2]))) if force.shape[1] > 2 else 0.0,
        }
        
        return PerformanceMetrics(
            episode_id=episode_idx,
            task_name=task_name,
            force_magnitude=force_magnitude,
            rms_error=rms_error,
            peak_torque=peak_torque,
            mean_torque=mean_torque,
            overshoot_percent=overshoot_percent,
            settling_time=settling_time,
            steady_state_error=steady_state_error,
            iae=iae,
            peak_force=peak_force,
        )
    
    def analyze_all_episodes(self) -> List[PerformanceMetrics]:
        """Analyze all episodes in the dataset."""
        if self.info is None:
            self.load_metadata()
        
        total_episodes = self.info.get('total_episodes', 0)
        print(f"\nAnalyzing {total_episodes} episodes...")
        
        metrics_list = []
        for ep_idx in range(total_episodes):
            try:
                metrics = self.analyze_episode(ep_idx)
                metrics_list.append(metrics)
                print(f"    Episode {ep_idx}: {metrics.task_name} ({metrics.force_magnitude}N)")
            except Exception as e:
                print(f"    Episode {ep_idx}: Error - {e}")
        
        # Sort by force magnitude
        metrics_list.sort(key=lambda m: m.force_magnitude)
        
        return metrics_list
    
    def generate_comparison_report(self, metrics_list: List[PerformanceMetrics]) -> List[dict]:
        """Generate comparison data for the web app."""
        comparison = []
        
        for m in metrics_list:
            row = {
                'Episode': m.episode_id,
                'Task': m.task_name,
                'Force (N)': m.force_magnitude,
                'RMS Error (rad)': np.mean(list(m.rms_error.values())),
                'Peak Torque (Nm)': np.mean(list(m.peak_torque.values())),
                'Mean Torque (Nm)': np.mean(list(m.mean_torque.values())),
                'Overshoot (%)': np.mean(list(m.overshoot_percent.values())),
                'Settling Time (s)': np.mean(list(m.settling_time.values())),
                'SS Error (rad)': np.mean(list(m.steady_state_error.values())),
                'IAE': np.mean(list(m.iae.values())),
            }
            comparison.append(row)
        
        return comparison
    
    def export_to_json(self, metrics_list: List[PerformanceMetrics], output_path: str):
        """Export metrics to JSON for the comparison page."""
        comparison = self.generate_comparison_report(metrics_list)
        
        data = {
            'dataset': self.repo_id,
            'metrics': [m.to_dict() for m in metrics_list],
            'comparison': comparison
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nExported metrics to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LeRobot dataset from HuggingFace')
    parser.add_argument('--repo', type=str, 
                        default='tommaselli/frank_load_experiments',
                        help='HuggingFace dataset repository ID')
    parser.add_argument('--output', type=str,
                        default='metrics.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LeRobot Dataset Analysis")
    print("=" * 60)
    
    # Create analyzer
    analyzer = LeRobotDatasetAnalyzer(repo_id=args.repo)
    
    # Load metadata
    analyzer.load_metadata()
    
    # Analyze all episodes
    metrics_list = analyzer.analyze_all_episodes()
    
    if not metrics_list:
        print("\nNo episodes found or all failed to analyze!")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Comparison Report:")
    print("=" * 60)
    comparison = analyzer.generate_comparison_report(metrics_list)
    df_comp = pd.DataFrame(comparison)
    print(df_comp.to_string(index=False))
    
    # Export to JSON
    analyzer.export_to_json(metrics_list, args.output)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output: {args.output}")
    print("\nTo update the comparison page, upload metrics.json to HuggingFace:")
    print(f"  huggingface-cli upload {args.repo} {args.output} metrics.json")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
