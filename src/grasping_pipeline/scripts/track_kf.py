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

# --- New Kalman Filter Class ---
class KalmanFilter:
    def __init__(self):
        # State is [x, y, vx, vy]. We track in 2D and handle Z separately.
        self.state = np.zeros(4) 
        # State covariance matrix (our uncertainty)
        self.covariance = np.eye(4) * 500.
        # State transition matrix
        self.F = np.eye(4)
        # Measurement matrix
        self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        # Measurement noise covariance
        self.R = np.eye(2) * 0.1
        # Process noise covariance
        self.Q = np.eye(4) * 0.1
        self.last_update_time = None

    def predict(self, dt):
        # Update state transition matrix with current time delta
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        # Predict state and covariance (using np.dot for Python 2/3 compatibility)
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement):
        # Kalman Gain calculation (using np.dot for Python 2/3 compatibility)
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        # Update state and covariance
        residual = measurement - np.dot(self.H, self.state)
        self.state += np.dot(K, residual)
        self.covariance = np.dot((np.eye(4) - np.dot(K, self.H)), self.covariance)

    def process_measurement(self, point):
        now = rospy.Time.now().to_sec()
        if self.last_update_time is None:
            # First measurement, initialize state
            self.state[0] = point.x
            self.state[1] = point.y
            self.last_update_time = now
            return

        dt = now - self.last_update_time
        self.last_update_time = now

        self.predict(dt)
        self.update(np.array([point.x, point.y]))

class PredictiveGraspManager:
    def __init__(self):
        rospy.init_node('predictive_grasp_manager')
        rospy.on_shutdown(self.shutdown_hook)

        # --- Parameters ---
        self.min_points_for_est = rospy.get_param('~min_points_for_est', 30) # Can be lower for KF
        self.base_frame = "base_link"
        self.eef_frame = rospy.get_param('~eef_frame', "tool_frame")
        self.grasp_z_target = 0.122

        # --- NEW: Added X offset parameter ---
        self.x_offset = rospy.get_param('~x_offset', 0.05) # 5cm offset in x

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
        
        # --- NEW: Kalman Filter and related data ---
        self.kf = KalmanFilter()
        self.measurement_count = 0
        self.estimated_state = np.zeros(4) # [x, y, vx, vy]

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
            # Reset Kalman Filter and counters
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

        rospy.loginfo("Velocity Estimated: vx={:.3f} m/s, vy={:.3f} m/s".format(self.estimated_state[2], self.estimated_state[3]))
        rospy.loginfo("Transitioning to BLIND_DESCENT state.")
        self.current_state = "BLIND_DESCENT"
        
        self.chase_timer = rospy.Timer(rospy.Duration(1.0/self.chase_loop_rate), self.chase_loop_callback)

    def chase_loop_callback(self, event):
        if self.current_state != "BLIND_DESCENT":
            return

        try:
            time_elapsed = rospy.Time.now().to_sec() - self.chase_start_time
            
            # Predict object's XY position and add the offset
            predicted_x = self.chase_start_position[0] + self.estimated_state[2] * time_elapsed + self.x_offset
            predicted_y = self.chase_start_position[1] + self.estimated_state[3] * time_elapsed
            
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

            rospy.loginfo_throttle(0.2, "Chase Vels (vx,vy,vz): ({:.3f}, {:.3f}, {:.3f})".format(
                control_vx, control_vy, control_vz))

            command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE)
            command.twist.linear_x = control_vx
            command.twist.linear_y = control_vy
            command.twist.linear_z = control_vz
            self.vel_pub.publish(command)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "Chase loop TF Exception: {}".format(e))
            self.stop_robot()

    def execute_final_grasp(self):
        if self.chase_timer:
            self.chase_timer.shutdown()
        self.stop_robot()

        try:
            # Correctly calculate time elapsed in seconds
            time_elapsed_sec = rospy.Time.now().to_sec() - self.chase_start_time
            # Apply the offset to the final logged position as well
            final_target_x = self.chase_start_position[0] + self.estimated_state[2] * time_elapsed_sec + self.x_offset
            final_target_y = self.chase_start_position[1] + self.estimated_state[3] * time_elapsed_sec

            eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0))
            actual_pos = eef_transform.transform.translation
            
            final_error_x = final_target_x - actual_pos.x
            final_error_y = final_target_y - actual_pos.y
            final_error_z = self.grasp_z_target - actual_pos.z

            rospy.loginfo(
                "\n*** GRASP TRIGGERED ***"
                "\n  Final Target Pos: x={:.4f}, y={:.4f}, z={:.4f}"
                "\n  Actual End Pos:   x={:.4f}, y={:.4f}, z={:.4f}"
                "\n  Final Error (m):  x={:.4f}, y={:.4f}, z={:.4f}".format(
                    final_target_x, final_target_y, self.grasp_z_target,
                    actual_pos.x, actual_pos.y, actual_pos.z,
                    final_error_x, final_error_y, final_error_z
                )
            )
        except Exception as e:
            rospy.logwarn("Could not log final grasp poses: {}".format(e))
        
        self.current_state = "GRASPING"
        rospy.loginfo("Closing gripper...")
        self.control_gripper(0.7)
        rospy.sleep(1.0)
        rospy.loginfo("Grasp successful. Starting drop-off sequence.")
        #self.control_gripper(0.0)
        ##self.reset_to_idle()

        # 1) Lift object 5 cm

        rospy.loginfo("Lifting object by 5cm...")
        # Get the current pose to perform a relative lift
        current_grasp_pose = self.move_group.get_current_pose().pose
        lift_pose = deepcopy(current_grasp_pose)
        lift_pose.position.z += 0.05
        self.move_to_pose(lift_pose)
        rospy.loginfo("Lift complete.")

        # 2) Retreat in -X for 10 cm
        rospy.loginfo("Retreating in -X by 10cm...")
        retreat_pose = deepcopy(lift_pose)
        retreat_pose.position.x -= 0.10
        self.move_to_pose(retreat_pose)
        rospy.loginfo("Retreat complete.")

        # 3) Move to drop-off pose (-15cm in Y)
        rospy.loginfo("Moving to drop-off location...")
        drop_pose = deepcopy(lift_pose)
        drop_pose.position.y -= 0.15
        self.move_to_pose(drop_pose)
        rospy.loginfo("At drop-off location.")

        # 3) Open gripper to release
        rospy.loginfo("Opening gripper to release object...")
        self.control_gripper(0.0)
        rospy.sleep(1.0)
        rospy.loginfo("Object released.")

        # 5) Return to the retreat pose
        rospy.loginfo("Returning to pre-drop pose...")
        self.move_to_pose(retreat_pose)
        rospy.loginfo("Returned to pre-drop pose.")

        # 6) Return home
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



# #!/usr/bin/env python
# import rospy
# import numpy as np
# from collections import deque
# import tf2_ros
# import tf2_geometry_msgs 
# from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped
# from std_srvs.srv import Empty, EmptyResponse
# from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
# from kortex_driver.msg import TwistCommand, CartesianReferenceFrame
# import math
# import actionlib
# from action_client_interface.msg import ClusteringAction, ClusteringGoal, ClusteringActionFeedback
# from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# # --- New Kalman Filter Class ---
# class KalmanFilter:
#     def __init__(self):
#         # State is [x, y, vx, vy]. We track in 2D and handle Z separately.
#         self.state = np.zeros(4) 
#         # State covariance matrix (our uncertainty)
#         self.covariance = np.eye(4) * 500.
#         # State transition matrix
#         self.F = np.eye(4)
#         # Measurement matrix
#         self.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
#         # Measurement noise covariance
#         self.R = np.eye(2) * 0.1
#         # Process noise covariance
#         self.Q = np.eye(4) * 0.1
#         self.last_update_time = None

#     def predict(self, dt):
#         # Update state transition matrix with current time delta
#         self.F[0, 2] = dt
#         self.F[1, 3] = dt
#         # Predict state and covariance (using np.dot for Python 2/3 compatibility)
#         self.state = np.dot(self.F, self.state)
#         self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

#     def update(self, measurement):
#         # Kalman Gain calculation (using np.dot for Python 2/3 compatibility)
#         S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
#         K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
#         # Update state and covariance
#         residual = measurement - np.dot(self.H, self.state)
#         self.state += np.dot(K, residual)
#         self.covariance = np.dot((np.eye(4) - np.dot(K, self.H)), self.covariance)

#     def process_measurement(self, point):
#         now = rospy.Time.now().to_sec()
#         if self.last_update_time is None:
#             # First measurement, initialize state
#             self.state[0] = point.x
#             self.state[1] = point.y
#             self.last_update_time = now
#             return

#         dt = now - self.last_update_time
#         self.last_update_time = now

#         self.predict(dt)
#         self.update(np.array([point.x, point.y]))

# class PredictiveGraspManager:
#     def __init__(self):
#         rospy.init_node('predictive_grasp_manager')

#         # --- Parameters ---
#         self.min_points_for_est = rospy.get_param('~min_points_for_est', 30) # Can be lower for KF
#         self.base_frame = "base_link"
#         self.eef_frame = rospy.get_param('~eef_frame', "tool_frame")
#         self.grasp_z_target = 0.13

#         # --- Servoing/Chase Parameters ---
#         self.kp_track = rospy.get_param('~kp_track', 0.8)
#         self.kd_track = rospy.get_param('~kd_track', 0.08)
#         self.kp_chase_xy = rospy.get_param('~kp_chase_xy', 1.5)
#         self.kp_chase_z = rospy.get_param('~kp_chase_z', 1.0)
#         self.dead_zone_xy = rospy.get_param('~dead_zone_xy', 0.01)
#         self.dead_zone_z = rospy.get_param('~dead_zone_z', 0.005)
#         self.chase_loop_rate = 50 # Hz

#         # --- State Machine & Data ---
#         self.current_state = "IDLE"
#         self.home_pose = None
#         self.chase_timer = None
#         self.last_error_x = 0.0
#         self.last_error_y = 0.0
#         self.last_timestamp = None
        
#         # --- NEW: Kalman Filter and related data ---
#         self.kf = KalmanFilter()
#         self.measurement_count = 0
#         self.estimated_state = np.zeros(4) # [x, y, vx, vy]

#         # --- ROS Interfaces ---
#         roscpp_initialize('')
#         self.move_group = MoveGroupCommander("arm")
#         self.move_group.set_pose_reference_frame(self.base_frame)
        
#         self.vel_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
#         self.clustering_client = actionlib.SimpleActionClient('cluster_events', ClusteringAction)
#         self.gripper_client = actionlib.SimpleActionClient('/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd', GripperCommandAction)
#         rospy.loginfo("Waiting for servers...")
#         self.clustering_client.wait_for_server()
#         self.gripper_client.wait_for_server()
#         rospy.loginfo("Servers found.")
        
#         rospy.Service('~start_grasp', Empty, self.start_grasp_cb)
#         rospy.Subscriber('/cluster_events/feedback', ClusteringActionFeedback, self.centroid_callback)

#         rospy.loginfo("Predictive Grasp Manager is ready.")

#     def start_grasp_cb(self, req):
#         if self.current_state == "IDLE":
#             rospy.loginfo("Start command received. Transitioning to TRACKING state.")
#             # Reset Kalman Filter and counters
#             self.kf = KalmanFilter()
#             self.measurement_count = 0
            
#             self.last_timestamp = rospy.Time.now()
#             self.last_error_x = 0.0
#             self.last_error_y = 0.0
#             self.home_pose = self.move_group.get_current_pose().pose
#             self.clustering_client.send_goal(ClusteringGoal())
#             self.current_state = "TRACKING"
#             return EmptyResponse()
#         return None

#     def centroid_callback(self, msg):
#         if self.current_state != "TRACKING":
#             return

#         centroid_point = msg.feedback.centroid
        
#         # Process every measurement with the Kalman Filter
#         self.kf.process_measurement(centroid_point.point)
#         self.measurement_count += 1
        
#         # Perform visual servoing to keep object in view
#         self.perform_xy_servoing(centroid_point)

#         rospy.loginfo_throttle(1.0, "Tracking... {}/{} points collected. Current vel: vx={:.3f}, vy={:.3f}".format(
#             self.measurement_count, self.min_points_for_est, self.kf.state[2], self.kf.state[3]))

#         # Start chase once the filter has stabilized
#         if self.measurement_count >= self.min_points_for_est:
#             self.start_blind_chase()

#     def perform_xy_servoing(self, target_centroid):
#         # This function remains the same as before
#         try:
#             target_in_eef_frame = self.tf_buffer.transform(target_centroid, self.eef_frame, rospy.Duration(0.2))
#             error_x, error_y = target_in_eef_frame.point.x, target_in_eef_frame.point.y
#             current_time = rospy.Time.now()
#             dt = (current_time - self.last_timestamp).to_sec() if self.last_timestamp else 0.01
#             if dt > 0.001:
#                 derivative_x = (error_x - self.last_error_x) / dt
#                 derivative_y = (error_y - self.last_error_y) / dt
#                 control_vx = (self.kp_track * error_x) + (self.kd_track * derivative_x)
#                 control_vy = (self.kp_track * error_y) + (self.kd_track * derivative_y)
#             else:
#                 control_vx, control_vy = self.kp_track * error_x, self.kp_track * error_y
#             self.last_error_x, self.last_error_y, self.last_timestamp = error_x, error_y, current_time
#             command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL)
#             command.twist.linear_x, command.twist.linear_y = control_vx, control_vy
#             self.vel_pub.publish(command)
#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#             rospy.logwarn_throttle(1.0, "XY Servoing TF Exception: {}".format(e))

#     def start_blind_chase(self):
#         rospy.loginfo("Tracking complete. Using Kalman Filter estimate.")
#         self.current_state = "ESTIMATING"
#         self.stop_robot()
        
#         # Get the final, most up-to-date state from the filter
#         self.estimated_state = self.kf.state
        
#         # The chase starts from the filter's current position estimate
#         self.chase_start_position = self.estimated_state[0:2]
#         self.chase_start_time = self.kf.last_update_time # Use the time of the last measurement

#         rospy.loginfo("Velocity Estimated: vx={:.3f} m/s, vy={:.3f} m/s".format(self.estimated_state[2], self.estimated_state[3]))
#         rospy.loginfo("Transitioning to BLIND_DESCENT state.")
#         self.current_state = "BLIND_DESCENT"
        
#         self.chase_timer = rospy.Timer(rospy.Duration(1.0/self.chase_loop_rate), self.chase_loop_callback)

#     def chase_loop_callback(self, event):
#         if self.current_state != "BLIND_DESCENT":
#             return

#         try:
#             time_elapsed = rospy.Time.now().to_sec() - self.chase_start_time
            
#             # Predict object's XY position
#             predicted_x = self.chase_start_position[0] + self.estimated_state[2] * time_elapsed
#             predicted_y = self.chase_start_position[1] + self.estimated_state[3] * time_elapsed
            
#             eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0))
#             current_position = np.array([
#                 eef_transform.transform.translation.x,
#                 eef_transform.transform.translation.y,
#                 eef_transform.transform.translation.z
#             ])

#             error_x = predicted_x - current_position[0]
#             error_y = predicted_y - current_position[1]
#             error_z = self.grasp_z_target - current_position[2]

#             if abs(error_z) < self.dead_zone_z and math.sqrt(error_x**2 + error_y**2) < self.dead_zone_xy:
#                 rospy.loginfo("Target intercept point reached. Grasping now.")
#                 self.execute_final_grasp()
#                 return

#             # Controller with Feed-Forward Term using KF velocity
#             control_vx = (self.kp_chase_xy * error_x) + self.estimated_state[2]
#             control_vy = (self.kp_chase_xy * error_y) + self.estimated_state[3]
#             control_vz = self.kp_chase_z * error_z

#             command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_BASE)
#             command.twist.linear_x = control_vx
#             command.twist.linear_y = control_vy
#             command.twist.linear_z = control_vz
#             self.vel_pub.publish(command)

#         except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#             rospy.logwarn_throttle(1.0, "Chase loop TF Exception: {}".format(e))
#             self.stop_robot()

#     def execute_final_grasp(self):
#         if self.chase_timer:
#             self.chase_timer.shutdown()
#         self.stop_robot()
#         self.current_state = "GRASPING"
#         rospy.loginfo("Closing gripper...")
#         self.control_gripper(0.7)
#         rospy.sleep(1.0)
#         rospy.loginfo("Grasp successful. Returning home.")
#         self.control_gripper(0.0)
#         self.move_to_pose(self.home_pose)
#         self.reset_to_idle()

#     def move_to_pose(self, pose):
#         if not pose: return False
#         self.move_group.set_pose_target(pose)
#         success = self.move_group.go(wait=True)
#         self.move_group.stop()
#         self.move_group.clear_pose_targets()
#         if not success:
#             rospy.logerr("MoveIt! failed to plan or execute path.")
#         return success

#     def control_gripper(self, position):
#         goal = GripperCommandGoal()
#         goal.command.position = position
#         self.gripper_client.send_goal(goal)
#         self.gripper_client.wait_for_result(rospy.Duration(3.0))

#     def stop_robot(self):
#         self.vel_pub.publish(TwistCommand())

#     def reset_to_idle(self):
#         rospy.loginfo("Resetting to IDLE state.")
#         if self.chase_timer:
#             self.chase_timer.shutdown()
#         self.stop_robot()
#         self.current_state = "IDLE"

#     def run(self):
#         rospy.spin()

# if __name__ == '__main__':
#     try:
#         manager = PredictiveGraspManager()
#         manager.run()
#     except rospy.ROSInterruptException:
#         pass
#     finally:
#         roscpp_shutdown()


