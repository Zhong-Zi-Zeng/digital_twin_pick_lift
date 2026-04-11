# Digital Twin Pick & Lift

A robotic arm grasping task environment based on Isaac Lab. The goal is to control the JetCobot robotic arm to pick up a block and lift it 5 cm above its initial position.

---

## Observation & Action Space

### 1. Observation Space
The agent receives a **25-dimensional** observation vector consisting of the robot's own state, the target object's position, and the previous action.

| Feature | Dimension | Description |
| :--- | :---: | :--- |
| **Joint Position (relative)** | 6 | Difference between the 6-axis joint angles and the default pose |
| **Joint Velocity (relative)** | 6 | Difference between the 6-axis joint velocities and the default velocities |
| **Object Position** | 3 | Block position in the Robot Root Frame (x, y, z) |
| **EE Position** | 3 | End-Effector position in the Robot Root Frame (x, y, z) |
| **Prev Actions** | 7 | Actions executed in the previous step (Δx, Δy, Δz, Δrx, Δry, Δrz, gripper) |
| **Total** | **25** | |

### 2. Action Space
The control scheme uses **continuous control** with a dimension of **7**.

| Type | Dimension | Description |
| :--- | :---: | :--- |
| **Continuous** | 7 | **First 6 dims**: Relative pose delta of the End-Effector (Δx, Δy, Δz, Δrx, Δry, Δrz), scaled by 0.005 m/step<br>**7th dim**: Gripper open/close command (>0 open, ≤0 close) |

**Control Logic:**
- **Arm**: The first 6 action dimensions represent the target EE pose delta. The system converts this to joint position targets via **Differential Inverse Kinematics (DLS method)**.
- **Gripper**: The 7th dimension is a binary command — values >0 set the gripper fully open (0.14), values ≤0 set it fully closed (-0.58).

---

## Overview

This project is based on the Isaac Lab framework. It implements a pick-and-lift task where the JetCobot robotic arm learns to grasp a block and raise it 5 cm using reinforcement learning (skrl + PPO).

**Keywords:** isaaclab, manipulation, pick-and-lift, digital-twin, skrl

---

## Installation

Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then install this package in editable mode:

```bash
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python -m pip install -e source/digital_twin
```

---

## Pre-trained Weights

Pre-trained model checkpoints are available for download here:

**[Download Pre-trained Weights (Google Drive)](https://drive.google.com/drive/folders/1CrSR2x3QY-e_YaO-HCt6y4YZBNY0ZZnM?usp=drive_link)**

Download the checkpoint and place it in your desired directory. See the [Testing](#testing-with-pre-trained-weights) section below for how to load it.

---

## Training

To train the agent from scratch:

```bash
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/skrl/train.py --task=Template-Digital-Twin-Direct-v0 --num_envs 256
```

Training logs and checkpoints are saved under `logs/skrl/` by default.

---

## Testing with Pre-trained Weights

To run inference using a downloaded or locally trained checkpoint:

```bash
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python scripts/skrl/play.py --task=Template-Digital-Twin-Direct-v0 --num_envs 16 --checkpoint <PATH_TO_CHECKPOINT>
```

Replace `<PATH_TO_CHECKPOINT>` with the path to the `.pt` checkpoint file you downloaded.