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

class PredictiveGraspManager:
    def __init__(self):
        rospy.init_node('predictive_grasp_manager')

        # --- Parameters ---
        self.history_size = rospy.get_param('~history_size', 60)
        self.min_points_for_est = rospy.get_param('~min_points_for_est', 50)
        self.pre_grasp_height = rospy.get_param('~pre_grasp_height', 0.22)
        self.base_frame = "base_link"
        self.eef_frame = rospy.get_param('~eef_frame', "tool_frame")
        self.grasp_z_target = 0.13

        # --- Servoing/Chase Parameters ---
        self.kp_track = rospy.get_param('~kp_track', 0.8) 
        self.kd_track = rospy.get_param('~kd_track', 0.08)
        # --- GAINS FOR PD CONTROLLER ---
        self.kp_chase_xy = rospy.get_param('~kp_chase_xy', 2.5) # Proportional gain
        self.kd_chase_xy = rospy.get_param('~kd_chase_xy', 0.15) # Derivative gain
        self.kp_chase_z = rospy.get_param('~kp_chase_z', 1.5)
        self.dead_zone_xy = rospy.get_param('~dead_zone_xy', 0.01)
        self.dead_zone_z = rospy.get_param('~dead_zone_z', 0.005)
        self.chase_loop_rate = 50 # Hz

        # --- State Machine & Data ---
        self.current_state = "IDLE"
        self.object_position_history = deque(maxlen=self.history_size)
        self.estimated_velocity = np.array([0.0, 0.0, 0.0])
        self.chase_start_time = None
        self.chase_start_position = None
        self.home_pose = None
        self.chase_timer = None
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_timestamp = None
        # --- Variables for PD control in chase loop ---
        self.last_chase_error_x = 0.0
        self.last_chase_error_y = 0.0
        self.last_chase_timestamp = None

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

    def start_grasp_cb(self, req):
        if self.current_state == "IDLE":
            rospy.loginfo("Start command received. Transitioning to TRACKING state.")
            self.object_position_history.clear()
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
        
        self.perform_xy_servoing(centroid_point)

        timestamp = centroid_point.header.stamp.to_sec()
        position = np.array([centroid_point.point.x, centroid_point.point.y, centroid_point.point.z])
        self.object_position_history.append((timestamp, position))
        
        rospy.loginfo_throttle(1.0, "Tracking... {}/{} points collected.".format(len(self.object_position_history), self.min_points_for_est))

        if len(self.object_position_history) >= self.min_points_for_est:
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
        rospy.loginfo("Tracking complete. Estimating velocity.")
        self.current_state = "ESTIMATING"
        self.stop_robot() # Stop the tracking motion before starting the chase
        
        times = np.array([p[0] for p in self.object_position_history])
        positions_x = np.array([p[1][0] for p in self.object_position_history])
        positions_y = np.array([p[1][1] for p in self.object_position_history])
        
        try:
            vx, _ = np.polyfit(times, positions_x, 1)
            vy, _ = np.polyfit(times, positions_y, 1)
            self.estimated_velocity = np.array([vx, vy, 0.0])
            
            # --- FIX 1: Compensate for "dead time" latency ---
            last_timestamp, last_position = self.object_position_history[-1]
            self.chase_start_time = rospy.Time.now()
            
            dead_time = self.chase_start_time.to_sec() - last_timestamp
            rospy.loginfo("Prediction latency (dead time): {:.4f} seconds".format(dead_time))

            # Adjust the starting position to account for object motion during latency
            self.chase_start_position = last_position + self.estimated_velocity * dead_time

            rospy.loginfo("Velocity Estimated: vx={:.3f} m/s, vy={:.3f} m/s".format(vx, vy))
            rospy.loginfo("Transitioning to BLIND_DESCENT state.")
            
            # Reset PD controller state for the chase
            self.last_chase_error_x = 0.0
            self.last_chase_error_y = 0.0
            self.last_chase_timestamp = rospy.Time.now()
            
            self.current_state = "BLIND_DESCENT"
            self.chase_timer = rospy.Timer(rospy.Duration(1.0/self.chase_loop_rate), self.chase_loop_callback)

        except (np.linalg.LinAlgError, TypeError):
            rospy.logwarn("Failed to estimate velocity. Aborting.")
            self.reset_to_idle()

    def chase_loop_callback(self, event):
        if self.current_state != "BLIND_DESCENT":
            return

        try:
            current_time = rospy.Time.now()
            time_elapsed = (current_time - self.chase_start_time).to_sec()
            predicted_position = self.chase_start_position + self.estimated_velocity * time_elapsed
            
            eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0))
            current_position = np.array([
                eef_transform.transform.translation.x,
                eef_transform.transform.translation.y,
                eef_transform.transform.translation.z
            ])

            error_x = predicted_position[0] - current_position[0]
            error_y = predicted_position[1] - current_position[1]
            error_z = self.grasp_z_target - current_position[2]

            if abs(error_z) < self.dead_zone_z and math.sqrt(error_x**2 + error_y**2) < self.dead_zone_xy:
                rospy.loginfo("Target intercept point reached. Grasping now.")
                self.execute_final_grasp()
                return

            # --- FIX 2: PD Controller with Feed-Forward Term ---
            dt = (current_time - self.last_chase_timestamp).to_sec()
            
            derivative_x = 0
            derivative_y = 0
            if dt > 0.001:
                derivative_x = (error_x - self.last_chase_error_x) / dt
                derivative_y = (error_y - self.last_chase_error_y) / dt

            # P-term + D-term + Feed-forward term
            control_vx = (self.kp_chase_xy * error_x) + (self.kd_chase_xy * derivative_x) + self.estimated_velocity[0]
            control_vy = (self.kp_chase_xy * error_y) + (self.kd_chase_xy * derivative_y) + self.estimated_velocity[1]
            control_vz = self.kp_chase_z * error_z

            # Update state for next loop
            self.last_chase_error_x = error_x
            self.last_chase_error_y = error_y
            self.last_chase_timestamp = current_time

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
        
        # --- FIX 3: Log final target and actual positions for debugging ---
        try:
            time_elapsed = (rospy.Time.now() - self.chase_start_time).to_sec()
            final_target_position = self.chase_start_position + self.estimated_velocity * time_elapsed

            eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0))
            current_actual_position = np.array([
                eef_transform.transform.translation.x,
                eef_transform.transform.translation.y,
                eef_transform.transform.translation.z
            ])
            final_error = final_target_position - current_actual_position

            rospy.loginfo(
                "\n*** GRASP TRIGGERED ***"
                "\n  Final Target Pos: x={:.4f}, y={:.4f}, z={:.4f}"
                "\n  Actual End Pos:   x={:.4f}, y={:.4f}, z={:.4f}"
                "\n  Final Error (m):  x={:.4f}, y={:.4f}, z={:.4f}".format(
                    final_target_position[0], final_target_position[1], final_target_position[2],
                    current_actual_position[0], current_actual_position[1], current_actual_position[2],
                    final_error[0], final_error[1], final_error[2]
                )
            )
        except Exception as e:
            rospy.logwarn("Could not log final grasp poses: {}".format(e))
        
        self.current_state = "GRASPING"
        
        rospy.loginfo("Closing gripper...")
        self.control_gripper(0.7)
        rospy.sleep(1.0)

        rospy.loginfo("Grasp successful. Returning home.")
        self.control_gripper(0.0)
        self.move_to_pose(self.home_pose)

        self.reset_to_idle()

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
