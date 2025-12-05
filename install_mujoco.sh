#!/bin/bash
set -e

echo "======================================"
echo "MuJoCo and mujoco-py Installation Script"
echo "======================================"

# Set up environment variables
export LD_LIBRARY_PATH=/home/tommaselli/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export PATH=/home/tommaselli/miniconda/bin:$PATH

echo ""
echo "Step 1: Checking conda environments..."
conda env list

echo ""
echo "Step 2: Creating mujoco_py conda environment with Python 3.8..."
conda create --name mujoco_py python=3.8 -y || echo "Environment may already exist"

echo ""
echo "Step 3: Activating mujoco_py environment..."
source /home/tommaselli/miniconda/etc/profile.d/conda.sh
conda activate mujoco_py

echo ""
echo "Step 4: Installing system dependencies..."
sudo apt update
sudo apt-get install -y patchelf
sudo apt-get install -y python3-dev build-essential libssl-dev libffi-dev libxml2-dev
sudo apt-get install -y libxslt1-dev zlib1g-dev libglew-dev python3-pip

echo ""
echo "Step 5: Cloning mujoco-py repository..."
cd ~/.mujoco
if [ -d "mujoco-py" ]; then
    echo "mujoco-py directory already exists, pulling latest..."
    cd mujoco-py
    git pull || true
else
    git clone https://github.com/openai/mujoco-py
    cd mujoco-py
fi

echo ""
echo "Step 6: Installing mujoco-py requirements..."
pip install -r requirements.txt
pip install -r requirements.dev.txt

echo ""
echo "Step 7: Installing mujoco-py in editable mode..."
pip3 install -e . --no-cache

echo ""
echo "Step 8: Installing additional packages..."
sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
sudo ln -sf /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so || true

echo ""
echo "Step 9: Installing mujoco-py from pip..."
pip3 install -U 'mujoco-py<2.2,>=2.1'

echo ""
echo "Step 10: Installing simulation dependencies..."
pip install matplotlib roboticstoolbox-python spatialmath-python

echo ""
echo "======================================"
echo "Installation completed!"
echo "======================================"
echo ""
echo "To test the installation:"
echo "1. Open a new terminal"
echo "2. Run: conda activate mujoco_py"
echo "3. Run: cd ~/.mujoco/mujoco-py/examples && python3 setting_state.py"
echo ""
echo "Note: You may need to reboot your machine for all changes to take effect."
