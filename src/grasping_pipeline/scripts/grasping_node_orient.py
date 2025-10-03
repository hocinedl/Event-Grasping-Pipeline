#!/usr/bin/env python
import rospy
import actionlib
from grasping_pipeline.srv import GraspObject, GraspObjectResponse
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import math
from copy import deepcopy


def handle_grasp(req):
    # Extract target point from request
    target = req.target

    rospy.loginfo("[grasp] Received grasp request for target at [%f, %f, %f]", target.x, target.y, target.z)        
    rospy.loginfo("[grasp] Target angle: %f", req.angle)


    # Initialize MoveIt
    roscpp_initialize([])
    arm = MoveGroupCommander("arm")
    arm.set_pose_reference_frame("base_link")
    arm.set_max_velocity_scaling_factor(0.2)
    arm.set_max_acceleration_scaling_factor(0.2)

    # Record current pose as home
    home_pose = arm.get_current_pose().pose
    rospy.loginfo("[grasp] Home pose recorded: %s", home_pose)

    # 1) Move to pre-grasp position (10 cm above the centroid, keeping current orientation)
    rospy.loginfo("[grasp] Moving to pre-grasp position")
    pregrasp_pose = deepcopy(home_pose) # Start with a copy of home pose
    pregrasp_pose.position.x = target.x
    pregrasp_pose.position.y = target.y
    pregrasp_pose.position.z = 0.12 + 0.10  # 10 cm above grasp
    # Keep the orientation from home_pose for the initial approach
    arm.set_start_state_to_current_state()
    arm.set_pose_target(pregrasp_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] At pre-grasp position.")

    # 2) Set final grasp orientation at the pre-grasp position
    rospy.loginfo("[grasp] Adjusting orientation for grasp")
    orient_pose = deepcopy(pregrasp_pose) # Create a new, independent pose
    # Add 90 degrees (pi/2) to the object's angle to grasp from the side
    grasp_angle = req.angle + (math.pi / 2.0)
    q = quaternion_from_euler(math.pi, 0, grasp_angle)  # Roll=pi, Pitch=0, Yaw=object angle + 90 deg
    orient_pose.orientation.x = q[0]
    orient_pose.orientation.y = q[1]
    orient_pose.orientation.z = q[2]
    orient_pose.orientation.w = q[3]
    arm.set_start_state_to_current_state()
    arm.set_pose_target(orient_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] Orientation set.")


    # 3) Move down to grasp pose
    rospy.loginfo("[grasp] Moving down to grasp pose")
    grasp_pose = deepcopy(orient_pose) # Create a new, independent pose
    grasp_pose.position.z = 0.12
    arm.set_start_state_to_current_state()
    arm.set_pose_target(grasp_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] At grasp pose, closing gripper")

    # 4) Close gripper
    gripper_client = actionlib.SimpleActionClient(
        '/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd',
        GripperCommandAction
    )
    gripper_client.wait_for_server()
    close_goal = GripperCommandGoal()
    close_goal.command.position = 0.7  # closed
    close_goal.command.max_effort = 5.0
    gripper_client.send_goal(close_goal)
    gripper_client.wait_for_result()
    rospy.loginfo("[grasp] Gripper closed")

    # 5) Lift object 10 cm
    rospy.loginfo("[grasp] Lifting object by 10cm")
    lift_pose = deepcopy(arm.get_current_pose().pose) # Create a new, independent pose
    lift_pose.position.z += 0.10
    arm.set_start_state_to_current_state()
    arm.set_pose_target(lift_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] Lifted object")

    # 6) Move to drop-off pose
    rospy.loginfo("[grasp] Moving to drop-off pose")
    drop_pose = deepcopy(lift_pose) # Create a new, independent pose
    drop_pose.position.x = target.x
    drop_pose.position.y = target.y - 0.25
    drop_pose.position.z = 0.15
    arm.set_start_state_to_current_state()
    arm.set_pose_target(drop_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] At drop-off pose, opening gripper")

    # 7) Open gripper to release
    open_goal = GripperCommandGoal()
    open_goal.command.position = 0.0  # open
    open_goal.command.max_effort = 1.0
    gripper_client.send_goal(open_goal)
    gripper_client.wait_for_result()
    rospy.loginfo("[grasp] Gripper opened, object released")

    # 8) Raise slightly after drop
    rospy.loginfo("[grasp] Raising after drop")
    post_drop_pose = deepcopy(drop_pose) # Create a new, independent pose
    post_drop_pose.position.z += 0.10
    arm.set_start_state_to_current_state()
    arm.set_pose_target(post_drop_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] Raised after drop")

    # 9) Return to home pose
    rospy.loginfo("[grasp] Returning to home pose")
    arm.set_start_state_to_current_state()
    arm.set_pose_target(home_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] Returned to home pose")

    # Shutdown MoveIt
    roscpp_shutdown()
    rospy.loginfo("[grasp] Grasp sequence complete")
    return GraspObjectResponse(success=True, message="Grasp and deliver complete")


if __name__ == '__main__':
    rospy.init_node('grasping_node')
    rospy.Service('/perform_grasp', GraspObject, handle_grasp)
    rospy.loginfo("Grasp service ready.")
    rospy.spin()
