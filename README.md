### Real-time Grasping using Event-driven Camera: ROS Package

This repository contains the full source code for the Master's Thesis, "Real-time Grasping using Event-driven Camera."

The code provides a complete ROS pipeline for robotic pick-and-place tasks using an event-based vision sensor (DAVIS346 camera).


### Real-time Grasping using Event-driven Camera: ROS Package

This repository contains the source code for the Master's thesis **“Real-time Grasping using Event-driven Camera.”**

The package implements a complete **ROS** pipeline for robotic pick-and-place using an **eye-in-hand DAVIS346** event camera. The perception module runs directly on the 2D event stream, performs **DBSCAN** clustering and a rotated-rectangle fit, validates clusters using a size-consistency test, and **back-projects** the centroid/orientation to 3D using camera intrinsics and the TF chain. A temporal filter (DBSCAN in 3D pose space) stabilizes the estimate. The filtered pose is used by:
- **Search-then-grasp** (static objects): short exploratory motion → grasp.
- **Track-then-predictive-grasp** (moving objects): PBVS tracking + velocity estimation to predict an intercept.

**Main components**
- `src/event_2d_clustering_action_server.py` – action server that publishes the filtered 3D centroid/yaw.
- `src/utils.py` – back-projection, DBSCAN filters, drawing utilities.
- `launch/*` – example bringup and demo launch files.
- `config/*` – camera intrinsics and pipeline parameters.

**Basic usage**
1. Start camera and robot bringup (eye-in-hand TF must be available).
2. Run the clustering action server.
3. Launch either the static or dynamic grasp demo.

> Notes:  
> • Object height and top-face dimensions are assumed known for size validation and depth computation.  
> • Tested in Docker (ROS Kinetic) and on a Kinova Gen3 with Robotiq 2F-85.

**License**  
Specify your license here (e.g., MIT/BSD-3-Clause).
