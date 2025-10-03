#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
import tf2_ros
import tf2_geometry_msgs 
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped
from std_srvs.srv import Empty, EmptyResponse
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from kortex_driver.msg import TwistCommand, CartesianReferenceFrame
import math
import actionlib
from action_client_interface.msg import ClusteringAction, ClusteringGoal, ClusteringActionFeedback
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from copy import deepcopy

# --- Kalman Filter Class (from your original node) ---
class KalmanFilter:
    def __init__(self):
        self.state = np.zeros(4) 
        self.covariance = np.eye(4) * 500.
        self.F = np.eye(4)
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        self.R = np.eye(2) * 0.1
        self.Q = np.eye(4) * 0.1
        self.last_update_time = None

    def predict(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement):
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        residual = measurement - np.dot(self.H, self.state)
        self.state += np.dot(K, residual)
        self.covariance = np.dot((np.eye(4) - np.dot(K, self.H)), self.covariance)

    def process_measurement(self, point):
        now = rospy.Time.now().to_sec()
        if self.last_update_time is None:
            self.state[0] = point.x
            self.state[1] = point.y
            self.last_update_time = now
            return

        dt = now - self.last_update_time
        self.last_update_time = now

        self.predict(dt)
        self.update(np.array([point.x, point.y]))

# --- Main Application Class (your original structure) ---
class PredictiveGraspManager:
    def __init__(self):
        rospy.init_node('predictive_grasp_manager')
        rospy.on_shutdown(self.shutdown_hook)

        # --- Parameters ---
        self.min_points_for_est = rospy.get_param('~min_points_for_est', 30)
        self.base_frame = "base_link"
        self.eef_frame = rospy.get_param('~eef_frame', "tool_frame")
        self.grasp_z_target = 0.122
        
        # --- MODIFIED: Predictive offset parameter ---
        self.predictive_offset = rospy.get_param('~predictive_offset', 0.025) # Look-ahead distance in meters

        # --- Servoing/Chase Parameters ---
        self.kp_track = rospy.get_param('~kp_track', 0.8)
        self.kd_track = rospy.get_param('~kd_track', 0.08)
        self.kp_chase_xy = rospy.get_param('~kp_chase_xy', 1.5)
        self.kp_chase_z = rospy.get_param('~kp_chase_z', 1.0)
        self.dead_zone_xy = rospy.get_param('~dead_zone_xy', 0.01)
        self.dead_zone_z = rospy.get_param('~dead_zone_z', 0.005)
        self.chase_loop_rate = 50 # Hz

        # --- State Machine & Data ---
        self.current_state = "IDLE"
        self.home_pose = None
        self.chase_timer = None
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_timestamp = None
        self.kf = KalmanFilter()
        self.measurement_count = 0
        self.estimated_state = np.zeros(4)

        # --- ROS Interfaces ---
        roscpp_initialize('')
        self.move_group = MoveGroupCommander("arm")
        self.move_group.set_pose_reference_frame(self.base_frame)
        
        self.vel_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.clustering_client = actionlib.SimpleActionClient('cluster_events', ClusteringAction)
        self.gripper_client = actionlib.SimpleActionClient('/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd', GripperCommandAction)
        rospy.loginfo("Waiting for servers...")
        self.clustering_client.wait_for_server()
        self.gripper_client.wait_for_server()
        rospy.loginfo("Servers found.")
        
        rospy.Service('~start_grasp', Empty, self.start_grasp_cb)
        rospy.Subscriber('/cluster_events/feedback', ClusteringActionFeedback, self.centroid_callback)

        rospy.loginfo("Predictive Grasp Manager is ready.")

    def shutdown_hook(self):
        rospy.loginfo("Shutdown signal received. Cancelling goals and stopping robot.")
        if self.current_state != "IDLE":
            self.clustering_client.cancel_goal()
        self.stop_robot()

    def start_grasp_cb(self, req):
        if self.current_state == "IDLE":
            rospy.loginfo("Start command received. Transitioning to TRACKING state.")
            self.kf = KalmanFilter()
            self.measurement_count = 0
            self.last_timestamp = rospy.Time.now()
            self.last_error_x = 0.0
            self.last_error_y = 0.0
            self.home_pose = self.move_group.get_current_pose().pose
            self.clustering_client.send_goal(ClusteringGoal())
            self.current_state = "TRACKING"
            return EmptyResponse()
        return None

    def centroid_callback(self, msg):
        if self.current_state != "TRACKING":
            return

        centroid_point = msg.feedback.centroid
        self.kf.process_measurement(centroid_point.point)
        self.measurement_count += 1
        self.perform_xy_servoing(centroid_point)

        rospy.loginfo_throttle(1.0, "Tracking... {}/{} points collected. Current vel: vx={:.3f}, vy={:.3f}".format(
            self.measurement_count, self.min_points_for_est, self.kf.state[2], self.kf.state[3]))

        if self.measurement_count >= self.min_points_for_est:
            self.start_blind_chase()

    def perform_xy_servoing(self, target_centroid):
        try:
            target_in_eef_frame = self.tf_buffer.transform(target_centroid, self.eef_frame, rospy.Duration(0.2))
            error_x, error_y = target_in_eef_frame.point.x, target_in_eef_frame.point.y
            current_time = rospy.Time.now()
            dt = (current_time - self.last_timestamp).to_sec() if self.last_timestamp else 0.01
            if dt > 0.001:
                derivative_x = (error_x - self.last_error_x) / dt
                derivative_y = (error_y - self.last_error_y) / dt
                control_vx = (self.kp_track * error_x) + (self.kd_track * derivative_x)
                control_vy = (self.kp_track * error_y) + (self.kd_track * derivative_y)
            else:
                control_vx, control_vy = self.kp_track * error_x, self.kp_track * error_y
            self.last_error_x, self.last_error_y, self.last_timestamp = error_x, error_y, current_time
            command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL)
            command.twist.linear_x, command.twist.linear_y = control_vx, control_vy
            self.vel_pub.publish(command)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "XY Servoing TF Exception: {}".format(e))

    def start_blind_chase(self):
        rospy.loginfo("Tracking complete. Using Kalman Filter estimate.")
        self.current_state = "ESTIMATING"
        self.stop_robot()
        
        self.estimated_state = self.kf.state
        self.chase_start_position = self.estimated_state[0:2]
        self.chase_start_time = self.kf.last_update_time

        
        # --- NEW CODE: Calculate and Log the Actual Measured Speed ---
        actual_speed_magnitude = math.sqrt(self.estimated_state[2]**2 + self.estimated_state[3]**2)
        rospy.loginfo("**************************************************")
        rospy.loginfo("*** Actual Measured Speed: {:.3f} m/s ***".format(actual_speed_magnitude))
        rospy.loginfo("**************************************************")

        rospy.loginfo("Velocity Estimated: vx={:.3f} m/s, vy={:.3f} m/s".format(self.estimated_state[2], self.estimated_state[3]))
        rospy.loginfo("Transitioning to BLIND_DESCENT state.")
        self.current_state = "BLIND_DESCENT"
        
        self.chase_timer = rospy.Timer(rospy.Duration(1.0/self.chase_loop_rate), self.chase_loop_callback)

    def chase_loop_callback(self, event):
        if self.current_state != "BLIND_DESCENT":
            return

        try:
            time_elapsed = rospy.Time.now().to_sec() - self.chase_start_time
            
            # --- MODIFIED: Predictive chase logic ---
            predicted_x_base = self.chase_start_position[0] + self.estimated_state[2] * time_elapsed
            predicted_y_base = self.chase_start_position[1] + self.estimated_state[3] * time_elapsed
            
            velocity_magnitude = math.sqrt(self.estimated_state[2]**2 + self.estimated_state[3]**2)
            if velocity_magnitude > 0.005:
                # Apply offset along the velocity vector
                predicted_x = predicted_x_base + (self.predictive_offset * self.estimated_state[2] / velocity_magnitude)
                predicted_y = predicted_y_base + (self.predictive_offset * self.estimated_state[3] / velocity_magnitude)
            else:
                # No offset if object is stationary
                predicted_x = predicted_x_base
                predicted_y = predicted_y_base

            eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0))
            current_position = np.array([
                eef_transform.transform.translation.x,
                eef_transform.transform.translation.y,
                eef_transform.transform.translation.z
            ])

            error_x = predicted_x - current_position[0]
            error_y = predicted_y - current_position[1]
            error_z = self.grasp_z_target - current_position[2]

            if abs(error_z) < self.dead_zone_z and math.sqrt(error_x**2 + error_y**2) < self.dead_zone_xy:
                rospy.loginfo("Target intercept point reached. Grasping now.")
                self.execute_final_grasp()
                return

            control_vx = (self.kp_chase_xy * error_x) + self.estimated_state[2]
            control_vy = (self.kp_chase_xy * error_y) + self.estimated_state[3]
            control_vz = self.kp_chase_z * error_z

            command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE)
            command.twist.linear_x, command.twist.linear_y, command.twist.linear_z = control_vx, control_vy, control_vz
            self.vel_pub.publish(command)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "Chase loop TF Exception: {}".format(e))
            self.stop_robot()

    def execute_final_grasp(self):
        if self.chase_timer:
            self.chase_timer.shutdown()
        self.stop_robot()
        
        # --- Grasp and Drop-off Sequence (your original logic) ---
        self.current_state = "GRASPING"
        rospy.loginfo("Closing gripper...")
        self.control_gripper(0.7) # Close to grasp
        rospy.sleep(1.5)
        rospy.loginfo("Grasp successful. Starting drop-off sequence.")

        rospy.loginfo("Lifting object by 5cm...")
        current_grasp_pose = self.move_group.get_current_pose().pose
        lift_pose = deepcopy(current_grasp_pose)
        lift_pose.position.z += 0.05
        self.move_to_pose(lift_pose)
        rospy.loginfo("Lift complete.")

        rospy.loginfo("Retreating in -X by 10cm...")
        retreat_pose = deepcopy(lift_pose)
        retreat_pose.position.x -= 0.10
        self.move_to_pose(retreat_pose)
        rospy.loginfo("Retreat complete.")

        rospy.loginfo("Moving to drop-off location...")
        drop_pose = deepcopy(retreat_pose)
        drop_pose.position.y -= 0.15
        self.move_to_pose(drop_pose)
        rospy.loginfo("At drop-off location.")

        rospy.loginfo("Opening gripper to release object...")
        self.control_gripper(0.0) # Open to release
        rospy.sleep(1.5)
        rospy.loginfo("Object released.")

        rospy.loginfo("Returning to pre-drop pose...")
        self.move_to_pose(retreat_pose)
        rospy.loginfo("Returned to pre-drop pose.")

        rospy.loginfo("Returning to home pose.")
        self.move_to_pose(self.home_pose)
        self.reset_to_idle()
        rospy.loginfo("Ready for next command.")

    def move_to_pose(self, pose):
        if not pose: return False
        self.move_group.set_pose_target(pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not success:
            rospy.logerr("MoveIt! failed to plan or execute path.")
        return success

    def control_gripper(self, position):
        goal = GripperCommandGoal()
        goal.command.position = position
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(rospy.Duration(3.0))

    def stop_robot(self):
        self.vel_pub.publish(TwistCommand())

    def reset_to_idle(self):
        rospy.loginfo("Resetting to IDLE state.")
        if self.chase_timer:
            self.chase_timer.shutdown()
        self.stop_robot()
        self.current_state = "IDLE"

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        manager = PredictiveGraspManager()
        manager.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        roscpp_shutdown()
