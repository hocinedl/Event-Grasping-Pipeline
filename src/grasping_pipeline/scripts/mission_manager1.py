#!/usr/bin/env python
import rospy
import actionlib
from std_srvs.srv import Trigger
from grasping_pipeline.srv import GraspObject, GraspObjectRequest
from geometry_msgs.msg import Point
from action_client_interface.msg import ClusteringAction, ClusteringGoal

class MissionManager(object):
    def __init__(self):
        rospy.init_node("mission_manager")

        # 1) start exploration action server
        rospy.loginfo("Waiting for exploration action server...")
        self.exploration_client = actionlib.SimpleActionClient(
            'exploration_server',
            ClusteringAction
        )
        self.exploration_client.wait_for_server()
        rospy.loginfo("Exploration action server connected.")

        # 2) wait for services
        rospy.loginfo("Waiting for grasping service...")
        rospy.wait_for_service('/perform_grasp')

        # create proxies
        self.perform_grasp = rospy.ServiceProxy('/perform_grasp', GraspObject)

        rospy.loginfo("All services available, launching mission...")
        self.run_mission()

    def run_mission(self):
        # send an empty goal to start exploration feedback
        goal = ClusteringGoal()
        self.exploration_client.send_goal(goal)
        rospy.loginfo("Exploration goal sent to action server.")
        self.exploration_client.wait_for_result()
        exploration_result = self.exploration_client.get_result()              
        x, y, z, angle = exploration_result.centroid.point.x, exploration_result.centroid.point.y, exploration_result.centroid.point.z, exploration_result.angle
        rospy.loginfo("Filtered centroid: (%.3f, %.3f, %.3f)", x, y, z)

        # 3) Grasping at filtered target
        req = GraspObjectRequest()
        req.target = Point(x=x, y=y, z=z)
        req.angle = angle
        rospy.loginfo("Calling grasp service...")
        resp3 = self.perform_grasp(req)
        if not resp3.success:
            rospy.logerr("Grasp failed: %s", resp3.message)
            return
        rospy.loginfo("Grasp succeeded: %s", resp3.message)

        rospy.loginfo("Mission pipeline complete!")

if __name__ == '__main__':
    try:
        MissionManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


### mission manager
