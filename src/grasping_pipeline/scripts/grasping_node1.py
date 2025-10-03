#!/usr/bin/env python
import rospy
import actionlib
from grasping_pipeline.srv import GraspObject, GraspObjectResponse
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
import math


def handle_grasp(req):

    target = req.target

    # Initialize MoveIt
    roscpp_initialize([])
    arm = MoveGroupCommander("arm")
    arm.set_pose_reference_frame("base_link")
    arm.set_max_velocity_scaling_factor(0.2)
    arm.set_max_acceleration_scaling_factor(0.2)

    # Record current pose as home
    home_pose = arm.get_current_pose().pose
    rospy.loginfo("[grasp] Home pose recorded: %s", home_pose)

    # 1) Move to pre grasp (10 cm above the centroid)
    pregrasp = Pose()
    pregrasp.position.x = target.x
    pregrasp.position.y = target.y
    #for the z we go to a fixed position of 12 cm for grasping so for pregrasping should be bit upper

    pregrasp.position.z = 0.12 + 0.10  #target.z + 0.10  # 10 cm above

    # added orientation considered
    pregrasp.orientation = arm.get_current_pose().pose.orientation
    rospy.loginfo("[grasp] Moving to pre grasp pose (10cm above target)")
    arm.set_start_state_to_current_state()
    arm.set_pose_target(pregrasp)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] At pre grasp pose ...")

    # 2) Move down to grasp pose
    rospy.loginfo("[grasp] Moving to grasp pose")
    grasp_pose = Pose()                 # it was grasp_pose = Pose()
    grasp_pose.position.x = target.x
    grasp_pose.position.y = target.y
    grasp_pose.position.z = 0.12 
    grasp_pose.orientation = pregrasp.orientation
    arm.set_start_state_to_current_state()
    arm.set_pose_target(grasp_pose)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.loginfo("[grasp] At grasp pose, closing gripper")

    # 3) Close gripper

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
    # for experiments i just need the gripper to close and not lift the object the open gripper 


    # 4) Lift object 10 cm

    # rospy.loginfo("[grasp] Lifting object by 10cm")
    # current = arm.get_current_pose().pose
    # lift_pose = current
    # lift_pose.position.z += 0.10
    # arm.set_start_state_to_current_state()
    # arm.set_pose_target(lift_pose)
    # arm.go(wait=True)
    # arm.stop()
    # arm.clear_pose_targets()
    # rospy.loginfo("[grasp] Lifted object")

    # 5) Move to drop-off pose
    # rospy.loginfo("[grasp] Moving to drop-off pose")
    # drop_pose = arm.get_current_pose().pose
    # drop_pose.position.x = target.x - 0.1
    # drop_pose.position.y = target.y - 0.15
    # drop_pose.position.z = 0.13
    # arm.set_start_state_to_current_state()
    # arm.set_pose_target(drop_pose)
    # arm.go(wait=True)
    # arm.stop()
    # arm.clear_pose_targets()
    # rospy.loginfo("[grasp] At drop-off pose, opening gripper")

    # 6) Open gripper to release
    open_goal = GripperCommandGoal()
    open_goal.command.position = 0.0  # open
    open_goal.command.max_effort = 1.0
    gripper_client.send_goal(open_goal)
    gripper_client.wait_for_result()
    rospy.loginfo("[grasp] Gripper opened, object released")

    # 7) Raise slightly after drop
    # rospy.loginfo("[grasp] Raising after drop")
    # post_drop_pose = drop_pose
    # post_drop_pose.position.z += 0.10
    # arm.set_start_state_to_current_state()
    # arm.set_pose_target(post_drop_pose)
    # arm.go(wait=True)
    # arm.stop()
    # arm.clear_pose_targets()
    # rospy.loginfo("[grasp] Raised after drop")

    # 8) Return to home pose
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



