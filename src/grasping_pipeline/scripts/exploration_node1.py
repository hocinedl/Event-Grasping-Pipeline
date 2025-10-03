#!/usr/bin/env python
import rospy
import actionlib
import math

from action_client_interface.msg import (
    ClusteringAction,
    ClusteringGoal,
    ClusteringResult,
    ClusteringActionFeedback,
)
from kortex_driver.msg import TwistCommand

class ExplorationServer(object):
    def __init__(self):
        # initialize node and params
        rospy.init_node('exploration_server')

        # publisher for movement commands
        self.cmd_pub = rospy.Publisher(
            '/my_gen3/in/cartesian_velocity',
            TwistCommand,
            queue_size=1
        )

        # action client to 'cluster_events'
        self.cluster_client = actionlib.SimpleActionClient(
            'cluster_events',
            ClusteringAction
        )
        rospy.loginfo('[exploration] Waiting for cluster_events action server...')
        self.cluster_client.wait_for_server()
        rospy.loginfo('[exploration] Connected to cluster_events.')

        # action server for exploration
        self.server = actionlib.SimpleActionServer(
            'exploration_server',
            ClusteringAction,
            execute_cb=self._execute_cb,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo('[exploration] Exploration action server ready.')

    

    def _execute_cb(self, goal):
        """Handle a new exploration goal."""
        # read movement parameters
        move       = rospy.get_param('~movement_type',  'zigzag')
        reps       = rospy.get_param('~repeats',        2)
        speed      = rospy.get_param('~speed',          0.05)
        dist       = rospy.get_param('~distance',       0.08)
        shift      = rospy.get_param('~stripe_shift',   0.08)
        rot_speed  = rospy.get_param('~rotation_speed', 0.5)
        spiral_step= rospy.get_param('~spiral_step',    0.1)
        yaw_sweep_angle_deg = rospy.get_param('~yaw_sweep_angle_deg', 90.0) # Total sweep angle in degrees


        rospy.loginfo(
            "[exploration] Starting '%s' (reps=%d)",
            move, reps
        )

        # send an empty goal to kick off clustering
        empty_goal = ClusteringGoal()
        self.cluster_client.send_goal(empty_goal)
        rospy.loginfo("[exploration] Sent empty goal to cluster_events.")

        def should_stop():
            if self.server.is_preempt_requested():
                rospy.loginfo("[exploration] Preempt requested!")
                return True
            return False

        # execute the chosen movement pattern
        if move == 'square':
            directions = [(1,0), (0,1), (-1,0), (0,-1)]
            for i in range(reps):
                for ux, uy in directions:
                    if should_stop(): break
                    duration = dist / speed
                    cmd = TwistCommand()
                    cmd.reference_frame = 0
                    cmd.twist.linear_x = ux * speed
                    cmd.twist.linear_y = uy * speed
                    cmd.duration = duration
                    self.cmd_pub.publish(cmd)
                    rospy.sleep(duration + 0.05)
                if should_stop(): break

        elif move == 'zigzag':
            # This implements a vertical "lawnmower" or "zigzag" pattern that
            # primarily moves along the Y-axis and shifts left along the X-axis.
            for i in range(reps):
                if should_stop(): break
                
                # Determine vertical direction: down on even reps, up on odd reps.
                y_direction = 1 if (i % 2 == 0) else -1

                # 1. Move vertically (down or up)
                rospy.loginfo("[exploration] Zigzag: Moving vertically (direction: %d).", y_direction)
                duration = dist / speed
                cmd = TwistCommand(reference_frame=0)
                cmd.twist.linear_y = y_direction * speed
                cmd.duration = duration
                self.cmd_pub.publish(cmd)
                rospy.sleep(duration + 0.05)
                if should_stop(): break

                # 2. Shift left (unless it's the very last stripe)
                if i < reps - 1:
                    rospy.loginfo("[exploration] Zigzag: Shifting left.")
                    duration2 = shift / speed
                    cmd = TwistCommand(reference_frame=0)
                    # Always move left (-x direction)
                    cmd.twist.linear_x = -speed
                    cmd.duration = duration2
                    self.cmd_pub.publish(cmd)
                    rospy.sleep(duration2 + 0.05)

        elif move == 'rotation':
            # This implements a circular movement in the XY plane.
            rospy.loginfo("[exploration] Starting circular movement.")
            
            diameter = 0.05  # 5cm diameter as requested
            radius = diameter / 2.0

            if speed == 0:
                rospy.logerr("[exploration] Speed for rotation cannot be zero.")
                return

            # Calculate the required angular velocity: radius = linear_speed / angular_speed
            rot_speed = speed / radius

            # Calculate the time needed to complete one full circle (2*pi radians)
            total_duration = (2 * math.pi) / rot_speed

            # Create the command for circular motion in the tool frame
            cmd = TwistCommand()
            cmd.reference_frame = 2  # TOOL_FRAME
            cmd.twist.linear_y = speed
            cmd.twist.angular_z = rot_speed
            cmd.duration = 0  # Command will be sent continuously

            # Loop to publish the command continuously for the required duration
            rate = rospy.Rate(100) # Publish at 100Hz
            start_time = rospy.get_time()
            
            rospy.loginfo("[exploration] Executing circular scan for %.2f seconds.", total_duration)
            while not rospy.is_shutdown() and (rospy.get_time() - start_time) < total_duration:
                if should_stop(): break
                self.cmd_pub.publish(cmd)
                rate.sleep()
            
            # IMPORTANT: Send a stop command after the loop
            stop_cmd = TwistCommand(reference_frame=0)
            self.cmd_pub.publish(stop_cmd)
            rospy.loginfo("[exploration] Circular scan complete.")

        elif move == 'yaw_sweep':
            rospy.loginfo("[exploration] Starting yaw sweep movement (Total Angle: %.1f deg, Reps: %d).", yaw_sweep_angle_deg, reps)
            
            # Convert sweep angle from degrees to radians
            sweep_angle_rad = math.radians(yaw_sweep_angle_deg)
            half_sweep_rad = sweep_angle_rad / 2.0

            # Define the three movements for the sweep
            # [angular_velocity_direction, duration]
            moves = [
                (1,  half_sweep_rad / rot_speed),  # 1. Center to Right
                (-1, sweep_angle_rad / rot_speed), # 2. Right to Left
                (1,  half_sweep_rad / rot_speed)   # 3. Left to Center
            ]

            # Repeat the entire sweep motion for the number of 'reps'
            for rep_num in range(reps):
                if should_stop(): break
                rospy.loginfo("[exploration] Yaw Sweep: Repetition %d/%d", rep_num + 1, reps)

                for i, (direction, duration) in enumerate(moves):
                    if should_stop(): break
                    rospy.loginfo("[exploration] Yaw Sweep: Step %d/3", i + 1)
                    cmd = TwistCommand(reference_frame=0) # Use BASE frame for wrist yaw
                    cmd.twist.angular_z = direction * rot_speed
                    cmd.duration = duration
                    self.cmd_pub.publish(cmd)
                    rospy.sleep(duration + 0.1) # Add a small buffer

            rospy.loginfo("[exploration] Yaw sweep complete.")

        else:
            msg = "Unknown movement_type: {}".format(move)
            rospy.logerr("[exploration] %s", msg)
            self.server.set_aborted(
                ClusteringResult(), text=msg
            )# this needs to be at the beginning so that the action server is not initialized in the first place
            return

        # stop any motion
        self.cmd_pub.publish(TwistCommand(reference_frame=0))
        

        # cancel clustering and wait for its result
        self.cluster_client.cancel_goal()
        rospy.loginfo("[exploration] Canceled cluster_events; waiting for result...")
        self.cluster_client.wait_for_result()
        cluster_result = self.cluster_client.get_result()
        rospy.loginfo("[exploration] Received cluster_events result: %s", cluster_result)

        # return the clustering result as our own
        self.server.set_succeeded(cluster_result)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    srv = ExplorationServer()
    srv.run()


# import rospy
# import actionlib
# import math

# from action_client_interface.msg import (
#     ClusteringAction,
#     ClusteringGoal,
#     ClusteringResult,
#     ClusteringActionFeedback,
# )
# from kortex_driver.msg import TwistCommand

# class ExplorationServer(object):
#     def __init__(self):
#         # initialize node and params
#         rospy.init_node('exploration_server')

#         # publisher for movement commands
#         self.cmd_pub = rospy.Publisher(
#             '/my_gen3/in/cartesian_velocity',
#             TwistCommand,
#             queue_size=1
#         )

#         # action client to 'cluster_events'
#         self.cluster_client = actionlib.SimpleActionClient(
#             'cluster_events',
#             ClusteringAction
#         )
#         rospy.loginfo('[exploration] Waiting for cluster_events action server...')
#         self.cluster_client.wait_for_server()
#         rospy.loginfo('[exploration] Connected to cluster_events.')

#         # action server for exploration
#         self.server = actionlib.SimpleActionServer(
#             'exploration_server',
#             ClusteringAction,
#             execute_cb=self._execute_cb,
#             auto_start=False
#         )
#         self.server.start()
#         rospy.loginfo('[exploration] Exploration action server ready.')

    

#     def _execute_cb(self, goal):
#         """Handle a new exploration goal."""
#         # read movement parameters
#         move       = rospy.get_param('~movement_type',  'zigzag')
#         reps       = rospy.get_param('~repeats',        2)
#         speed      = rospy.get_param('~speed',          0.05)
#         dist       = rospy.get_param('~distance',       0.08)
#         shift      = rospy.get_param('~stripe_shift',   0.08)
#         rot_speed  = rospy.get_param('~rotation_speed', 0.5)
#         spiral_step= rospy.get_param('~spiral_step',    0.1)
        


#         rospy.loginfo(
#             "[exploration] Starting '%s' (reps=%d)",
#             move, reps
#         )

#         # send an empty goal to kick off clustering
#         empty_goal = ClusteringGoal()
#         self.cluster_client.send_goal(empty_goal)
#         rospy.loginfo("[exploration] Sent empty goal to cluster_events.")

#         def should_stop():
#             if self.server.is_preempt_requested():
#                 rospy.loginfo("[exploration] Preempt requested!")
#                 return True
#             return False

#         # execute the chosen movement pattern
#         if move == 'square':
#             directions = [(1,0), (0,1), (-1,0), (0,-1)]
#             for i in range(reps):
#                 for ux, uy in directions:
#                     if should_stop(): break
#                     duration = dist / speed
#                     cmd = TwistCommand()
#                     cmd.reference_frame = 0
#                     cmd.twist.linear_x = ux * speed
#                     cmd.twist.linear_y = uy * speed
#                     cmd.duration = duration
#                     self.cmd_pub.publish(cmd)
#                     rospy.sleep(duration + 0.05)
#                 if should_stop(): break

#         elif move == 'zigzag':
#             for i in range(reps):
#                 if should_stop(): break
#                 direction = 1 if (i % 2 == 0) else -1
#                 # stripe
#                 duration = dist / speed
#                 cmd = TwistCommand(reference_frame=0)
#                 cmd.twist.linear_x = direction * speed
#                 cmd.twist.linear_y = 0.0
#                 cmd.duration = duration
#                 self.cmd_pub.publish(cmd)
#                 rospy.sleep(duration + 0.05)
#                 if should_stop(): break
#                 # shift
#                 duration2 = shift / speed
#                 cmd = TwistCommand(reference_frame=0)
#                 cmd.twist.linear_x = 0.0
#                 cmd.twist.linear_y = speed
#                 cmd.duration = duration2
#                 self.cmd_pub.publish(cmd)
#                 rospy.sleep(duration2 + 0.05)

#         elif move == 'rotation':
#             for i in range(reps):
#                 if should_stop(): break
#                 full_turn = 2 * math.pi
#                 duration = full_turn / rot_speed
#                 cmd = TwistCommand(reference_frame=0)
#                 cmd.twist.angular_z = rot_speed
#                 cmd.duration = duration
#                 self.cmd_pub.publish(cmd)
#                 rospy.sleep(duration + 0.05)

#         elif move == 'spiral':
#             length = spiral_step
#             multiplier = 1
#             dirs = [(1,0), (0,1), (-1,0), (0,-1)]
#             total = reps * 4
#             for i in range(total):
#                 if should_stop(): break
#                 ux, uy = dirs[i % 4]
#                 edge_len = length * multiplier
#                 duration = edge_len / speed
#                 cmd = TwistCommand(reference_frame=0)
#                 cmd.twist.linear_x = ux * speed
#                 cmd.twist.linear_y = uy * speed
#                 cmd.duration = duration
#                 self.cmd_pub.publish(cmd)
#                 rospy.sleep(duration + 0.05)
#                 if (i + 1) % 2 == 0:
#                     multiplier += 1

#         else:
#             msg = "Unknown movement_type: {}".format(move)
#             rospy.logerr("[exploration] %s", msg)
#             self.server.set_aborted(
#                 ClusteringResult(), text=msg
#             )# this needs to be at the beginning so that the action server is not initialized in the first place
#             return

#         # stop any motion
#         self.cmd_pub.publish(TwistCommand(reference_frame=0))
        

#         # cancel clustering and wait for its result
#         self.cluster_client.cancel_goal()
#         rospy.loginfo("[exploration] Canceled cluster_events; waiting for result...")
#         self.cluster_client.wait_for_result()
#         cluster_result = self.cluster_client.get_result()
#         rospy.loginfo("[exploration] Received cluster_events result: %s", cluster_result)

#         # return the clustering result as our own
#         self.server.set_succeeded(cluster_result)

#     def run(self):
#         rospy.spin()

# if __name__ == '__main__':
#     srv = ExplorationServer()
#     srv.run()
