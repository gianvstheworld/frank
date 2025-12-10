# Frank Robot MuJoCo Simulation

This project simulates the Frank robot using MuJoCo and Python.

> https://gianvstheworld.github.io/frank/

## Installation

1. Ensure you have the MuJoCo 2.10 binaries installed in `~/.mujoco/mujoco210`. Follow: https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da
2. Run the installation script:
   ```bash
   ./install_mujoco.sh
   ```

## Usage

To run the simulation, follow these steps in your terminal.

### 1. Activate Conda Environment

```bash
conda activate mujoco_py
```

### 2. Set Environment Variables

You must export these variables before running the script. You can run these commands in your terminal or add them to your `.bashrc` or `.zshrc`.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so.2.2
```

### 3. Run Simulation

Execute the Python script:

```bash
python frank_simple_inverse_dynamics.py
```

## Controls

- **Arrow Keys**: Move robot along X/Y axes.
- **Keypad 0/1**: Move robot along Z axis.
- **Mouse Left Click**: Actuate/Apply force.
