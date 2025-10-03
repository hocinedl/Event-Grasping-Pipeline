#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
from geometry_msgs.msg import PointStamped
from action_client_interface.msg import ClusteringActionFeedback

class CentroidSmoother:
    def __init__(self):
        rospy.init_node('centroid_smoother_node')

        # Parameters
        buffer_size = rospy.get_param('~buffer_size', 10) # Number of recent centroids to average
        publish_rate = rospy.get_param('~publish_rate', 50.0) # Hz

        # State
        self.buffer = deque(maxlen=buffer_size)
        self.last_header = None

        # Publisher for the smoothed centroid
        self.smooth_centroid_pub = rospy.Publisher('/object_centroid_smooth', PointStamped, queue_size=1)

        # Subscriber to the raw, noisy feedback from the clustering node
        rospy.Subscriber('/cluster_events/feedback', ClusteringActionFeedback, self.raw_centroid_callback)

        # Timer to publish the smoothed centroid at a fixed rate
        rospy.Timer(rospy.Duration(1.0/publish_rate), self.publish_smooth_centroid)

        rospy.loginfo("Centroid Smoother is running.")

    def raw_centroid_callback(self, msg):
        """Collects the latest raw centroid into the buffer."""
        p = msg.feedback.centroid.point
        self.buffer.append([p.x, p.y, p.z])
        self.last_header = msg.feedback.centroid.header

    def publish_smooth_centroid(self, event):
        """Calculates and publishes the average of the centroids in the buffer."""
        if not self.buffer or self.last_header is None:
            return

        # Calculate the mean of all points in the buffer
        data = np.array(self.buffer)
        mean_xyz = data.mean(axis=0)

        # Create and publish the PointStamped message
        smooth_point = PointStamped()
        smooth_point.header = self.last_header
        smooth_point.header.stamp = rospy.Time.now() # Use current time
        smooth_point.point.x = mean_xyz[0]
        smooth_point.point.y = mean_xyz[1]
        smooth_point.point.z = mean_xyz[2]

        self.smooth_centroid_pub.publish(smooth_point)

if __name__ == '__main__':
    try:
        CentroidSmoother()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
