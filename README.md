# Real-time Grasping using Event-driven Camera (ROS)

This repository contains the ROS package of the thesis **“Real-time Grasping using Event-driven Camera.”**  
The pipeline implements event-based 2D clustering, 3D back-projection, and temporal filtering to provide a stable 3D centroid pose and orientation for grasping via **two modes**:
- **Search-then-Grasp** (static objects).
- **Track-then-Predictive-Grasp** (moving objects with PBVS + velocity estimation for moving objects).

---

## 1. Dependencies

- OS / ROS: Ubuntu + ROS (Kinetic/Melodic/Noetic – match your local setup).
- **Event Camera Driver**: [rpg\_dvs\_ros](https://github.com/uzh-rpg/rpg_dvs_ros) for DAVIS346 (APS/DVS).
- **Robot Driver**: [ros\_kortex](https://github.com/Kinovarobotics/ros_kortex) for Kinova Gen3 robotic arm.
- **TF**: A valid TF chain `base_link -> dvx_camera_link`.
- **Calibration**: Camera intrinsics (K) and TF are required;
  
---

## 2. Build

```bash
# create workspace if not exists
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src
# clone this repo
git clone https://github.com/hocinedl/Event-Grasping-Pipeline.git
cd ..
catkin_make        # or catkin build
source devel/setup.bash
```
## 3. Start Required Drivers and MoveIt
In separate terminals:
```bash
# 1. Event camera (DAVIS) driver
roslaunch rpg_dvs_ros davis_mono.launch

# 2. Kinova Gen3 Robot Driver
# NOTE: Replace the ip_address with your robot's actual IP.
roslaunch kortex_driver kortex_driver.launch ip_address:=192.168.1.10 dof:=6 gripper:=robotiq_2f_85

# 3. Set robot description parameter (Necessary for MoveIt! integration)
rosparam set /robot_description "$(rosparam get /my_gen3/robot_description)"

# 4. Launch MoveIt! environment and move_group node
roslaunch gen3_robotiq_2f_85_move_it_config move_group.launch

```

## 4. Static Grasp (“Search-then-Grasp”)

This mode performs a short exploratory motion to elicit events, then computes a stable 3D centroid pose and orientation and executes the grasp.
```bash
#Launch the full pipeline infrastructure (clustering, exploration logic, etc.):
roslaunch grasping_pipeline pipeline.launch

# Start the mission by running the mission manager node: 
rosrun grasping_pipeline mission_manager1.py

Parameters such as object dimensions (object_dims = [L, W, H]), cluster eps/minPts, and size tolerance, and exploration trajectory can be configured via the launch file.
```


## 5. Dynamic Grasp (“Track-then-Predictive-Grasp”)

This mode continuously tracks a moving object using PBVS and a short history of centroids to estimate planar velocity (Kalman Filter). Once the velocity estimate stabilizes, the controller predicts an intercept point and executes the grasp.

```bash
# To initialize the clustering node:
- rosrun event_clustering clustering_servoing.py

# To initialize the tracking and grasping node:
- rosrun grasping_pipeline track_kf.py

 # To give the start signal to the predictive grasping pipeline:
- rosservice call /predictive_grasp_manager/start_grasp "{}"

```
