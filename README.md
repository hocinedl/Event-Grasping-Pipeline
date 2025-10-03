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
## 3. Running the Drivers
In separate terminals:
```bash

- roslaunch dvs_renderer davis_mono.launch    # Event camera (DAVIS) driver
- roslaunch kortex_driver kortex_driver.launch ip_address:=192.168.1.10 dof:=6 gripper:=robotiq_2f_85    # ROBOT DRIVER.
- rosparam set /robot_description "$(rosparam get /my_gen3/robot_description)"    
- roslaunch gen3_robotiq_2f_85_move_it_config move_group.launch   #MOVE IT 

```

## 4. Static Grasp (“Search-then-Grasp”)
This mode performs a short exploratory motion to elicit events, then computes a stable 3D centroid pose and orientation and executes the grasp.
```bash
#The launch file containing all necessary nodes (clustering, exploration, grasping)
roslaunch grasping_pipeline pipeline.launch

# start the mission by running the mission manager node:
rosrun grasping_pipeline mission_manager1.py

#Parameters such as object dimensions (object_dims = [L, W, H]), cluster eps/minPts, and size tolerance can be configured via the launch file.
```


## 5. Dynamic Grasp (“Track-then-Predictive-Grasp”)
This mode continuously tracks a moving object using PBVS and a short history of centroids to estimate planar velocity (Kalman Filter). Once the velocity estimate stabilizes, the controller predicts an intercept point and executes the grasp.

```bash
# To initialize the clustering node:
- rosrun event_clustering clustering_servoing.py

# to initialize the tracking and graping node:

- rosrun grasping_pipeline track_kf.py

 # To give the start to the pipeline:
- rosservice call /predictive_grasp_manager/start_grasp "{}"

```
