#!/usr/bin/env python3
"""
Robot Performance Analysis Module

Provides functions for analyzing MuJoCo simulation data, calculating
performance metrics like RMS error, overshoot, settling time, and torque analysis.

Author: Analysis module for FRANK Robot Performance Study
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Container for performance metrics of a single simulation."""
    force_magnitude: float
    rms_error: Dict[str, float]  # Per joint RMS error
    peak_torque: Dict[str, float]  # Per joint peak torque
    mean_torque: Dict[str, float]  # Per joint mean absolute torque
    overshoot_percent: Dict[str, float]  # Per joint overshoot percentage
    settling_time: Dict[str, float]  # Per joint settling time (seconds)
    rise_time: Dict[str, float]  # Per joint rise time (seconds)
    steady_state_error: Dict[str, float]  # Per joint final error
    iae: Dict[str, float]  # Integral Absolute Error per joint
    max_velocity_error: Dict[str, float]  # Max velocity tracking error per joint
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def load_simulation(filepath: str) -> pd.DataFrame:
    """
    Load simulation data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        Loaded simulation data
    """
    df = pd.read_csv(filepath)
    return df


def extract_force_from_filename(filename: str) -> float:
    """
    Extract force magnitude from filename (e.g., 'simulation_output_..._f10.csv' -> 10.0)
    
    Parameters
    ----------
    filename : str
        Filename to parse
    
    Returns
    -------
    float
        Force magnitude in Newtons
    """
    import re
    match = re.search(r'_f(\d+)\.csv$', filename)
    if match:
        return float(match.group(1))
    return 0.0


def get_joint_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """Get column names for a specific variable across all joints."""
    return [col for col in df.columns if col.startswith(prefix + '_joint')]


def calculate_rms_error(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Root Mean Square error for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data with error columns (e_joint1, e_joint2, etc.)
    
    Returns
    -------
    Dict[str, float]
        RMS error for each joint
    """
    error_cols = get_joint_columns(df, 'e')
    rms_errors = {}
    
    for col in error_cols:
        joint_name = col.replace('e_', '')
        rms_errors[joint_name] = np.sqrt(np.mean(df[col].values ** 2))
    
    return rms_errors


def calculate_peak_torque(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate peak (maximum absolute) torque for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data with torque columns (tau_joint1, tau_joint2, etc.)
    
    Returns
    -------
    Dict[str, float]
        Peak torque for each joint
    """
    torque_cols = get_joint_columns(df, 'tau')
    peak_torques = {}
    
    for col in torque_cols:
        joint_name = col.replace('tau_', '')
        peak_torques[joint_name] = np.max(np.abs(df[col].values))
    
    return peak_torques


def calculate_mean_torque(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate mean absolute torque for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data with torque columns
    
    Returns
    -------
    Dict[str, float]
        Mean absolute torque for each joint
    """
    torque_cols = get_joint_columns(df, 'tau')
    mean_torques = {}
    
    for col in torque_cols:
        joint_name = col.replace('tau_', '')
        mean_torques[joint_name] = np.mean(np.abs(df[col].values))
    
    return mean_torques


def calculate_overshoot(df: pd.DataFrame, threshold: float = 0.02) -> Dict[str, float]:
    """
    Calculate overshoot percentage for each joint.
    
    Overshoot is defined as the maximum deviation beyond the reference
    relative to the total trajectory range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data with position and desired position columns
    threshold : float
        Minimum movement threshold to consider (radians)
    
    Returns
    -------
    Dict[str, float]
        Overshoot percentage for each joint
    """
    overshoot = {}
    
    for i in range(1, 5):  # 4 joints
        q_col = f'q_joint{i}'
        q_d_col = f'q_d_joint{i}'
        
        if q_col not in df.columns or q_d_col not in df.columns:
            continue
        
        q = df[q_col].values
        q_d = df[q_d_col].values
        error = q - q_d
        
        # Find the reference range
        q_d_range = np.max(q_d) - np.min(q_d)
        
        if q_d_range < threshold:
            overshoot[f'joint{i}'] = 0.0
            continue
        
        # Calculate overshoot as percentage of reference range
        max_overshoot = np.max(np.abs(error))
        overshoot[f'joint{i}'] = (max_overshoot / q_d_range) * 100
    
    return overshoot


def calculate_settling_time(df: pd.DataFrame, threshold_percent: float = 0.02) -> Dict[str, float]:
    """
    Calculate settling time for each joint (time to reach ±threshold of final value).
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    threshold_percent : float
        Settling threshold as percentage of trajectory range
    
    Returns
    -------
    Dict[str, float]
        Settling time in seconds for each joint
    """
    settling_times = {}
    
    # Estimate timestep from data
    if 'step' in df.columns and len(df) > 1:
        # Assume 1kHz control rate based on simulation
        timestep = 0.001
    else:
        timestep = 0.001
    
    for i in range(1, 5):
        e_col = f'e_joint{i}'
        
        if e_col not in df.columns:
            continue
        
        error = np.abs(df[e_col].values)
        
        # Find steady-state error (last 10% of data)
        n_steady = max(1, len(error) // 10)
        steady_error = np.mean(error[-n_steady:])
        
        # Threshold based on trajectory range
        threshold = threshold_percent * (np.max(error) - np.min(error) + 0.001)
        threshold = max(threshold, 0.001)  # Minimum threshold
        
        # Find last time error exceeds threshold + steady state error
        settled_mask = error <= (steady_error + threshold)
        
        if np.all(settled_mask):
            settling_times[f'joint{i}'] = 0.0
        else:
            # Find the last index where it's not settled
            not_settled_indices = np.where(~settled_mask)[0]
            if len(not_settled_indices) > 0:
                settling_times[f'joint{i}'] = not_settled_indices[-1] * timestep
            else:
                settling_times[f'joint{i}'] = 0.0
    
    return settling_times


def calculate_rise_time(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate rise time (10% to 90% of trajectory) for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    
    Returns
    -------
    Dict[str, float]
        Rise time in seconds for each joint
    """
    rise_times = {}
    timestep = 0.001  # 1kHz
    
    for i in range(1, 5):
        q_col = f'q_joint{i}'
        q_d_col = f'q_d_joint{i}'
        
        if q_col not in df.columns or q_d_col not in df.columns:
            continue
        
        q = df[q_col].values
        q_d = df[q_d_col].values
        
        # Calculate response relative to reference
        q_d_min, q_d_max = np.min(q_d), np.max(q_d)
        q_d_range = q_d_max - q_d_min
        
        if q_d_range < 0.01:  # No significant movement
            rise_times[f'joint{i}'] = 0.0
            continue
        
        # Find 10% and 90% thresholds
        thresh_10 = q_d_min + 0.1 * q_d_range
        thresh_90 = q_d_min + 0.9 * q_d_range
        
        # Find first time reaching 10% and 90%
        above_10 = np.where(q >= thresh_10)[0]
        above_90 = np.where(q >= thresh_90)[0]
        
        if len(above_10) > 0 and len(above_90) > 0:
            t_10 = above_10[0] * timestep
            t_90 = above_90[0] * timestep
            rise_times[f'joint{i}'] = max(0, t_90 - t_10)
        else:
            rise_times[f'joint{i}'] = 0.0
    
    return rise_times


def calculate_steady_state_error(df: pd.DataFrame, n_samples: int = 100) -> Dict[str, float]:
    """
    Calculate steady-state error (average error in final portion of trajectory).
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    n_samples : int
        Number of samples at end to average
    
    Returns
    -------
    Dict[str, float]
        Steady state error for each joint
    """
    ss_errors = {}
    
    for i in range(1, 5):
        e_col = f'e_joint{i}'
        
        if e_col not in df.columns:
            continue
        
        error = df[e_col].values
        n = min(n_samples, len(error))
        ss_errors[f'joint{i}'] = np.mean(np.abs(error[-n:]))
    
    return ss_errors


def calculate_iae(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Integral Absolute Error for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    
    Returns
    -------
    Dict[str, float]
        IAE for each joint
    """
    timestep = 0.001  # 1kHz
    iae = {}
    
    for i in range(1, 5):
        e_col = f'e_joint{i}'
        
        if e_col not in df.columns:
            continue
        
        error = np.abs(df[e_col].values)
        iae[f'joint{i}'] = np.trapz(error, dx=timestep)
    
    return iae


def calculate_max_velocity_error(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate maximum velocity tracking error for each joint.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulation data
    
    Returns
    -------
    Dict[str, float]
        Max velocity error for each joint
    """
    max_vel_errors = {}
    
    for i in range(1, 5):
        dot_e_col = f'dot_e_joint{i}'
        
        if dot_e_col not in df.columns:
            continue
        
        vel_error = np.abs(df[dot_e_col].values)
        max_vel_errors[f'joint{i}'] = np.max(vel_error)
    
    return max_vel_errors


def analyze_simulation(filepath: str) -> PerformanceMetrics:
    """
    Perform complete analysis on a simulation file.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    
    Returns
    -------
    PerformanceMetrics
        Complete performance metrics
    """
    df = load_simulation(filepath)
    filename = os.path.basename(filepath)
    force = extract_force_from_filename(filename)
    
    return PerformanceMetrics(
        force_magnitude=force,
        rms_error=calculate_rms_error(df),
        peak_torque=calculate_peak_torque(df),
        mean_torque=calculate_mean_torque(df),
        overshoot_percent=calculate_overshoot(df),
        settling_time=calculate_settling_time(df),
        rise_time=calculate_rise_time(df),
        steady_state_error=calculate_steady_state_error(df),
        iae=calculate_iae(df),
        max_velocity_error=calculate_max_velocity_error(df)
    )


def analyze_all_simulations(data_dir: str) -> List[PerformanceMetrics]:
    """
    Analyze all simulation files in a directory.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing CSV files
    
    Returns
    -------
    List[PerformanceMetrics]
        List of performance metrics for each simulation
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))
    
    metrics_list = []
    for csv_file in csv_files:
        print(f"Analyzing: {csv_file.name}")
        metrics = analyze_simulation(str(csv_file))
        metrics_list.append(metrics)
    
    # Sort by force magnitude
    metrics_list.sort(key=lambda m: m.force_magnitude)
    
    return metrics_list


def generate_comparison_report(metrics_list: List[PerformanceMetrics]) -> pd.DataFrame:
    """
    Generate a comparison report across all simulations.
    
    Parameters
    ----------
    metrics_list : List[PerformanceMetrics]
        List of metrics from different simulations
    
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    rows = []
    
    for m in metrics_list:
        row = {'Force (N)': m.force_magnitude}
        
        # Average metrics across joints
        for metric_name, metric_dict in [
            ('RMS Error (rad)', m.rms_error),
            ('Peak Torque (Nm)', m.peak_torque),
            ('Mean Torque (Nm)', m.mean_torque),
            ('Overshoot (%)', m.overshoot_percent),
            ('Settling Time (s)', m.settling_time),
            ('Rise Time (s)', m.rise_time),
            ('SS Error (rad)', m.steady_state_error),
            ('IAE', m.iae),
            ('Max Vel Error (rad/s)', m.max_velocity_error)
        ]:
            if metric_dict:
                row[metric_name] = np.mean(list(metric_dict.values()))
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def export_to_json(metrics_list: List[PerformanceMetrics], output_path: str) -> None:
    """
    Export metrics to JSON for web application.
    
    Parameters
    ----------
    metrics_list : List[PerformanceMetrics]
        List of metrics to export
    output_path : str
        Path to output JSON file
    """
    data = {
        'metrics': [m.to_dict() for m in metrics_list],
        'comparison': generate_comparison_report(metrics_list).to_dict(orient='records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported metrics to {output_path}")


def export_timeseries_data(data_dir: str, output_path: str, downsample: int = 10) -> None:
    """
    Export time series data for plotting.
    
    Parameters
    ----------
    data_dir : str
        Directory with CSV files
    output_path : str
        Output JSON path
    downsample : int
        Downsample factor to reduce data size
    """
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob('*.csv'))
    
    timeseries = {}
    
    for csv_file in csv_files:
        force = extract_force_from_filename(csv_file.name)
        df = load_simulation(str(csv_file))
        
        # Downsample for web performance
        df_ds = df.iloc[::downsample].reset_index(drop=True)
        
        # Calculate time in seconds
        timestep = 0.001 * downsample
        time = np.arange(len(df_ds)) * timestep
        
        series_data = {
            'time': time.tolist(),
            'torque': {},
            'error': {},
            'position': {},
            'desired': {},
            'velocity_error': {},
            'external_force': df_ds['apply_ext_force'].tolist() if 'apply_ext_force' in df_ds.columns else []
        }
        
        for i in range(1, 5):
            joint = f'joint{i}'
            series_data['torque'][joint] = df_ds[f'tau_joint{i}'].tolist()
            series_data['error'][joint] = df_ds[f'e_joint{i}'].tolist()
            series_data['position'][joint] = df_ds[f'q_joint{i}'].tolist()
            series_data['desired'][joint] = df_ds[f'q_d_joint{i}'].tolist()
            series_data['velocity_error'][joint] = df_ds[f'dot_e_joint{i}'].tolist()
        
        timeseries[f'f{int(force)}'] = series_data
    
    with open(output_path, 'w') as f:
        json.dump(timeseries, f)
    
    print(f"Exported time series to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze robot simulation data')
    parser.add_argument('--data-dir', type=str, 
                        default='simulations_data',
                        help='Directory containing simulation CSV files')
    parser.add_argument('--output', type=str,
                        default='metrics.json',
                        help='Output JSON file path (upload this to HuggingFace)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Analyze all simulations
    print("=" * 50)
    print("Robot Performance Analysis")
    print("=" * 50)
    
    metrics_list = analyze_all_simulations(args.data_dir)
    
    if not metrics_list:
        print("\nNo simulation files found in", args.data_dir)
        print("Make sure you have CSV files from simulations with format:")
        print("  simulation_output_YYYYMMDD_HHMMSS_fXX.csv")
        exit(1)
    
    # Print comparison table
    print("\nComparison Report:")
    comparison = generate_comparison_report(metrics_list)
    print(comparison.to_string(index=False))
    
    # Export to JSON
    export_to_json(metrics_list, args.output)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"Output: {args.output}")
    print("\nTo update the comparison page, upload metrics.json to HuggingFace:")
    print("  huggingface-cli upload tommaselli/frank_load_experiments metrics.json metrics.json")
    print("=" * 50)

