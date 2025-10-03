#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
import tf2_ros
import tf2_geometry_msgs 
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3Stamped, PointStamped
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
        self.grasp_lookahead_time = rospy.get_param('~grasp_lookahead_time', 0.5) # Shorter lookahead for continuous correction
        self.grasp_height_offset = rospy.get_param('~grasp_height_offset', 0.01)
        self.pre_grasp_height = rospy.get_param('~pre_grasp_height', 0.22)
        self.base_frame = "base_link"
        self.eef_frame = rospy.get_param('~eef_frame', "tool_frame")
        self.grasp_z_target = 0.13 # Absolute Z height for the final grasp

        # --- Servoing Parameters ---
        self.kp_xy = rospy.get_param('~kp_xy', 0.8)
        self.kd_xy = rospy.get_param('~kd_xy', 0.08)
        self.kp_z = rospy.get_param('~kp_z', 0.6) # P-gain for Z-axis movement
        self.dead_zone_xy = rospy.get_param('~dead_zone_xy', 0.005)
        self.dead_zone_z = rospy.get_param('~dead_zone_z', 0.005) # Dead zone for reaching grasp height

        # --- State Machine ---
        self.current_state = "IDLE"
        self.object_position_history = deque(maxlen=self.history_size)
        self.estimated_velocity = np.array([0.0, 0.0])
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_timestamp = None
        self.home_pose = None

        # --- ROS Interfaces ---
        roscpp_initialize('')
        self.move_group = MoveGroupCommander("arm")
        self.move_group.set_pose_reference_frame(self.base_frame)
        self.move_group.set_max_velocity_scaling_factor(1.0)
        self.move_group.set_max_acceleration_scaling_factor(1.0)

        self.vel_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.clustering_client = actionlib.SimpleActionClient('cluster_events', ClusteringAction)
        rospy.loginfo("Waiting for clustering action server...")
        self.clustering_client.wait_for_server()
        rospy.loginfo("Clustering action server found.")

        self.gripper_client = actionlib.SimpleActionClient('/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd', GripperCommandAction)
        rospy.loginfo("Waiting for gripper action server...")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server found.")
        
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
        else:
            rospy.logwarn("Cannot start grasp, not in IDLE state.")
            return None

    def centroid_callback(self, msg):
        centroid_point = msg.feedback.centroid # This is a PointStamped
        
        if self.current_state == "TRACKING":
            self.perform_xy_servoing(centroid_point)
            timestamp = centroid_point.header.stamp.to_sec()
            position = np.array([centroid_point.point.x, centroid_point.point.y, centroid_point.point.z])
            self.object_position_history.append((timestamp, position))
            
            rospy.loginfo_throttle(1.0, "Tracking... {}/{} points collected.".format(len(self.object_position_history), self.min_points_for_est))

            if len(self.object_position_history) >= self.min_points_for_est:
                self.estimate_velocity_and_descend()

        elif self.current_state == "DESCENDING":
            self.perform_3d_servoing(centroid_point)

    def perform_xy_servoing(self, target_centroid):
        try:
            target_in_eef_frame = self.tf_buffer.transform(target_centroid, self.eef_frame, rospy.Duration(0.2))
            error_x, error_y = target_in_eef_frame.point.x, target_in_eef_frame.point.y

            current_time = rospy.Time.now()
            dt = (current_time - self.last_timestamp).to_sec() if self.last_timestamp else 0.01

            if dt > 0.001:
                derivative_x = (error_x - self.last_error_x) / dt
                derivative_y = (error_y - self.last_error_y) / dt
                control_vx = (self.kp_xy * error_x) + (self.kd_xy * derivative_x)
                control_vy = (self.kp_xy * error_y) + (self.kd_xy * derivative_y)
            else:
                control_vx, control_vy = self.kp_xy * error_x, self.kp_xy * error_y

            self.last_error_x, self.last_error_y, self.last_timestamp = error_x, error_y, current_time
            command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL)
            command.twist.linear_x, command.twist.linear_y = control_vx, control_vy
            self.vel_pub.publish(command)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "XY Servoing TF Exception: {}".format(e))

    def estimate_velocity_and_descend(self):
        rospy.loginfo("State: ESTIMATING. Tracking for velocity complete.")
        times = np.array([p[0] for p in self.object_position_history])
        positions_x = np.array([p[1][0] for p in self.object_position_history])
        positions_y = np.array([p[1][1] for p in self.object_position_history])
        
        try:
            vx, _ = np.polyfit(times, positions_x, 1)
            vy, _ = np.polyfit(times, positions_y, 1)
            self.estimated_velocity = np.array([vx, vy])
            rospy.loginfo("Velocity Estimated: vx={:.3f} m/s, vy={:.3f} m/s".format(vx, vy))
            rospy.loginfo("Transitioning to DESCENDING state.")
            self.current_state = "DESCENDING"
        except (np.linalg.LinAlgError, TypeError):
            rospy.logwarn("Failed to estimate velocity. Aborting.")
            self.stop_robot()
            self.current_state = "IDLE"

    def perform_3d_servoing(self, current_centroid):
        # 1. Predict future position
        predicted_x = current_centroid.point.x + self.estimated_velocity[0] * self.grasp_lookahead_time
        predicted_y = current_centroid.point.y + self.estimated_velocity[1] * self.grasp_lookahead_time

        # Create a PointStamped for the predicted target
        predicted_target = PointStamped()
        predicted_target.header = current_centroid.header
        predicted_target.point.x = predicted_x
        predicted_target.point.y = predicted_y
        predicted_target.point.z = current_centroid.point.z # Z prediction is not needed for XY servoing

        try:
            # 2. Calculate XY error in tool frame for vx, vy
            target_in_eef_frame = self.tf_buffer.transform(predicted_target, self.eef_frame, rospy.Duration(0.2))
            error_x, error_y = target_in_eef_frame.point.x, target_in_eef_frame.point.y
            control_vx = self.kp_xy * error_x
            control_vy = self.kp_xy * error_y

            # 3. Calculate Z error in base frame for vz
            eef_transform = self.tf_buffer.lookup_transform(self.base_frame, self.eef_frame, rospy.Time(0), rospy.Duration(0.1))
            current_z = eef_transform.transform.translation.z
            error_z = self.grasp_z_target - current_z
            control_vz = self.kp_z * error_z

            # 4. Check if grasp is complete
            if abs(error_z) < self.dead_zone_z and math.sqrt(error_x**2 + error_y**2) < self.dead_zone_xy:
                rospy.loginfo("Target reached. Grasping now.")
                self.stop_robot()
                self.execute_final_grasp()
                return

            # 5. Publish Twist command in MIXED frame
            # XY velocities are in the tool frame, Z velocity is in the base frame.
            command = TwistCommand(reference_frame=CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED)
            command.twist.linear_x = control_vx
            command.twist.linear_y = control_vy
            command.twist.linear_z = control_vz 
            self.vel_pub.publish(command)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, "3D Servoing TF Exception: {}".format(e))

    def execute_final_grasp(self):
        self.current_state = "GRASPING"
        
        # 1. Close Gripper
        rospy.loginfo("Closing gripper...")
        self.control_gripper(0.7)
        rospy.sleep(1.0)

        # 2. Retreat
        rospy.loginfo("Retreating with object.")
        current_pose = self.move_group.get_current_pose().pose
        retreat_pose = Pose(position=Point(current_pose.position.x, current_pose.position.y, self.pre_grasp_height), orientation=current_pose.orientation)
        if not self.move_to_pose(retreat_pose):
            rospy.logerr("Failed to retreat.")
        
        # 3. Go home
        rospy.loginfo("Grasp successful. Returning home.")
        self.control_gripper(0.0) # Drop object
        if self.home_pose and not self.move_to_pose(self.home_pose):
             rospy.logwarn("Failed to return to home pose.")

        rospy.loginfo("Grasping sequence finished. Returning to IDLE state.")
        self.current_state = "IDLE"

    def move_to_pose(self, pose):
        self.move_group.set_pose_target(pose)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        return success

    def control_gripper(self, position):
        goal = GripperCommandGoal()
        goal.command.position = position
        self.gripper_client.send_goal(goal)
        self.gripper_client.wait_for_result(rospy.Duration(3.0))

    def stop_robot(self):
        self.vel_pub.publish(TwistCommand())

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




