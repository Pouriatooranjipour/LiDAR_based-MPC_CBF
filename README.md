
# LiDAR-based MPC with Control Barrier Functions

## Overview

This repository contains the implementation of a Model Predictive Control (MPC) approach that leverages Control Barrier Functions (CBFs) to ensure recursive feasibility for robots equipped with LiDAR sensors. By incorporating CBFs within the MPC framework, the prediction horizon is shortened compared to standard MPC, effectively reducing computational complexity while ensuring the avoidance of unsafe sets.

A CBF is synthesized from 2D LiDAR data points by clustering obstacles using the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm and fitting an ellipse to each cluster using the OpenCV library's fitting tool. The resulting CBFs from each obstacle are unified and integrated into the MPC framework. Recursive feasibility is analyzed and guaranteed by choosing an appropriate terminal set.

The effectiveness of the approach is demonstrated through simulations in the Gazebo robotic simulator using ROS 2, followed by experimental validation with a unicycle-type robot equipped with a LiDAR sensor.

## Features

- **LiDAR Data Processing**: Clustering of obstacles using DBSCAN and ellipse fitting using OpenCV.
- **Control Barrier Functions**: Synthesis and integration of CBFs into the MPC framework.
- **Model Predictive Control**: Implementation of MPC with a shortened prediction horizon for computational efficiency.
- **Simulation and Experimental Validation**: Tested in Gazebo simulation and on a physical unicycle-type robot with a LiDAR sensor.

## Prerequisites

- **Operating System**: Ubuntu 22.04
- **ROS 2 Distribution**: ROS 2 Humble
- **Dependencies**:
  - **Python Packages**:
    - `numpy`
    - `rclpy`
    - `geometry_msgs`
    - `nav_msgs`
    - `tf_transformations`
    - `casadi`
    - `std_msgs`
    - `sensor_msgs`
    - `visualization_msgs`
    - `sklearn`
    - `opencv-python` (OpenCV)
    - `matplotlib`
    - `tf2_ros`
    - `tf2_geometry_msgs`
  - **ROSbot XL Environment Setup**:
    - Follow the [ROSbot XL ROS 2 Introduction Tutorial](https://husarion.com/tutorials/ros2-tutorials/1-ros2-introduction/) up to the "Launch Simulation" section.
  - **Husarion ROSbot XL Packages**
  - **Standard ROS 2 Packages**

## Installation

1. **Set Up the Environment for ROSbot XL**

   Follow the instructions provided in the [ROSbot XL ROS 2 Introduction Tutorial](https://husarion.com/tutorials/ros2-tutorials/1-ros2-introduction/) up to the "Launch Simulation" section.

2. **Clone This Repository**

   ```bash
   git clone https://github.com/yourusername/LiDAR_based-MPC_CBF.git
   ```

3. **Replace World File**

   Before running `ROSBOT_SIM`, replace the `empty_with_plugins.sdf` file:

   ```bash
   cp ~/LiDAR_based-MPC_CBF/src/empty_with_plugins.sdf ~/rosbot_ws/src/husarion_gz_worlds/worlds/
   ```

4. **Install Python Dependencies**

   Install the required Python packages:

   ```bash
   pip install numpy casadi sklearn opencv-python matplotlib
   ```

   > **Note**: Some packages like `rclpy` and other ROS-related packages are installed with ROS 2 and should not be installed via `pip`. Ensure that all ROS 2 Python packages are properly installed in your ROS 2 environment.

5. **Build the Package**

   ```bash
   cd LiDAR_based-MPC_CBF
   colcon build
   source install/setup.bash
   ```

## Usage

### Running the Simulation

Launch the simulation environment:

```bash
ROSBOT_SIM
```

### Running the LiDAR Clustering Node

Run the LiDAR clustering node, which subscribes to `/scan_filtered` and publishes `/lidar_ellipses` containing the ellipse parameters to be used for MPC:

```bash
ros2 run lidar_clustering lidar_clustering_node
```

> **Note**: If you receive a warning due to `tf_buffer.lookup_transform`, stop the node and run it again until `/lidar_ellipses` is published.

### Running the MPC Controller Node

Run the MPC controller node:

```bash
ros2 run controller MPC_controller
```

## Examples

Example codes from the paper can be found in the [`src/Examples_code`](src/Examples_code) directory.

## Videos

Demonstration videos are available in the [`src/videos`](src/videos) directory.

## Acknowledgments

- Based on research presented in the paper: *LiDAR-based MPC with Control Barrier Functions*.
- Special thanks to the developers of ROS 2, Gazebo, OpenCV, the DBSCAN algorithm, and **Husarion ROSbot XL**.

---

**Note**: If you used code or resources from other projects, such as Husarion ROSbot XL, please ensure compliance with their licensing terms when publishing your code.
