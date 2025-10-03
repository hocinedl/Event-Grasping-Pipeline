#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from kortex_driver.msg import TwistCommand, Twist, CartesianReferenceFrame
import math # Import the math library for sqrt

class PositionBasedServoing:
    def __init__(self):
        rospy.init_node('position_based_servoing_node')

        # --- PD Controller Gains ---
        self.kp = rospy.get_param('~kp', 0.5) 
        self.kd = rospy.get_param('~kd', 0.05)

        # --- MODIFICATION 1: Add a dead zone parameter ---
        # This is the error tolerance in meters. If the error is less than this, the robot will stop.
        self.dead_zone = rospy.get_param('~dead_zone', 0.01) # Default to 1 cm

        self.robot_base_frame = rospy.get_param('~robot_base_frame', 'base_link')
        self.robot_eef_frame = rospy.get_param('~robot_eef_frame', 'tool_frame')
        self.control_rate = rospy.Rate(rospy.get_param('~rate', 100))

        # State
        self.target_centroid = None
        self.servoing_active = False
        
        # --- State for PD controller ---
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_timestamp = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Publisher
        self.vel_pub = rospy.Publisher('/my_gen3/in/cartesian_velocity', TwistCommand, queue_size=1)

        rospy.Subscriber('/object_centroid_smooth', PointStamped, self.centroid_callback)
        
        rospy.loginfo("PD Position-Based Servoing Node is ready.")

    def centroid_callback(self, msg):
        self.target_centroid = msg
        if not self.servoing_active:
            self.servoing_active = True
            rospy.loginfo("PBVS: Smooth centroid received. SERVOING IS NOW ACTIVE.")
            self.last_timestamp = rospy.Time.now()

    def run(self):
        rospy.loginfo("PBVS: Run loop started.")
        while not rospy.is_shutdown():
            if not self.servoing_active or self.target_centroid is None:
                rospy.loginfo_throttle(5.0, "PBVS: Waiting for smooth centroid...")
                self.control_rate.sleep()
                continue

            try:
                target_in_eef_frame = self.tf_buffer.transform(self.target_centroid, self.robot_eef_frame, rospy.Duration(1.0))

                error_x = target_in_eef_frame.point.x
                error_y = target_in_eef_frame.point.y
                
                rospy.loginfo_throttle(1.0, "PBVS: Calculated error in tool frame (ex, ey): ({:.3f}, {:.3f})".format(error_x, error_y))

                # --- MODIFICATION 2: Implement the dead zone logic ---
                error_magnitude = math.sqrt(error_x**2 + error_y**2)

                if error_magnitude < self.dead_zone:
                    rospy.loginfo_throttle(2.0, "PBVS: Target within dead zone. Stopping movement.")
                    control_vx = 0.0
                    control_vy = 0.0
                else:
                    # If outside the dead zone, use the PD controller
                    current_time = rospy.Time.now()
                    dt = (current_time - self.last_timestamp).to_sec()

                    if dt > 0.001:
                        derivative_x = (error_x - self.last_error_x) / dt
                        derivative_y = (error_y - self.last_error_y) / dt

                        control_vx = (self.kp * error_x) + (self.kd * derivative_x)
                        control_vy = (self.kp * error_y) + (self.kd * derivative_y)
                    else:
                        control_vx = self.kp * error_x
                        control_vy = self.kp * error_y

                self.last_error_x = error_x
                self.last_error_y = error_y
                self.last_timestamp = rospy.Time.now()
                
                command = TwistCommand()
                command.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_TOOL
                
                kortex_twist = Twist()
                kortex_twist.linear_x = control_vx
                kortex_twist.linear_y = control_vy
                kortex_twist.linear_z = 0.0
                kortex_twist.angular_x = 0.0
                kortex_twist.angular_y = 0.0
                kortex_twist.angular_z = 0.0

                command.twist = kortex_twist
                command.duration = 0
                
                self.vel_pub.publish(command)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(1.0, "PBVS: TF Exception: {}".format(e))

            self.control_rate.sleep()

if __name__ == '__main__':
    try:
        node = PositionBasedServoing()
        node.run()
    except rospy.ROSInterruptException:
        pass
