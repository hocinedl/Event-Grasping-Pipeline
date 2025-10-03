#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
import actionlib

from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import cv2

from action_client_interface.msg import ClusteringAction, ClusteringFeedback, ClusteringResult
from utils import detect_object, build_transform, draw_detected_rect

# This node is a modified version of clustering_action_server1.py,
# specifically for the PBVS grasping pipeline. It provides continuous
# feedback instead of a single final result.

camera_frame = "dvx_camera_link"

class GraspingClusteringServer:
    def __init__(self):
        rospy.init_node("clustering_grasping_server")

        # Parameters
        self.time_window        = rospy.get_param("~time_window", 0.5)
        self.max_events         = rospy.get_param("~max_events", 3000)
        self.min_recent_events  = rospy.get_param("~min_recent_events", 50)
        self.process_rate       = rospy.get_param("~process_rate", 10.0)
        self.dims               = rospy.get_param("~object_dims", [0.085, 0.060, 0.145])
        # --- MODIFICATION: Added size_tol as a tunable parameter ---
        self.size_tol           = rospy.get_param("~size_tol", 2.0)

        self.K = np.array([[296.06, 0.0, 171.345],
                           [0.0, 297.68, 133.221],
                           [0.0, 0.0, 1.0]])

        rospy.loginfo("[GRASPING CLUSTERING] Time window: %f, Size tolerance: %f", self.time_window, self.size_tol)

        # State
        self.events      = deque(maxlen=self.max_events)
        self.bridge      = CvBridge()
        self.image       = None
        self.tf_listener = tf.TransformListener()

        # Subscribers
        rospy.Subscriber("/dvs/events", EventArray, self.event_callback, queue_size=10)
        rospy.Subscriber("/dvs_rendering", Image, self.image_callback, queue_size=1)

        # Publisher
        self.overlay_pub = rospy.Publisher("/grasping/overlay", Image, queue_size=1)

        # Action Server
        self._as = actionlib.SimpleActionServer(
            "clustering_for_grasping", # Using a new action name
            ClusteringAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo("GraspingClusteringServer ready and waiting for goals...")

    def event_callback(self, msg):
        for e in msg.events:
            self.events.append((e.x, e.y, e.ts.to_sec()))

    def image_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "mono8")
        except Exception:
            self.image = None

    def get_recent_events(self):
        now = rospy.get_time()
        return [e for e in list(self.events) if e[2] >= now - self.time_window]

    def publish_overlay(self, rect, cluster_pts):
        if self.image is None: return
        ov = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if cluster_pts is not None:
            for x, y in cluster_pts.astype(int):
                if 0 <= x < ov.shape[1] and 0 <= y < ov.shape[0]:
                    cv2.circle(ov, (x, y), 1, (0, 255, 0), -1)
        if rect is not None:
            ov = draw_detected_rect(ov, rect, color=(255, 0, 0), thickness=2)
        msg = self.bridge.cv2_to_imgmsg(ov, "bgr8")
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = camera_frame
        self.overlay_pub.publish(msg)

    def execute_cb(self, goal):
        rate = rospy.Rate(self.process_rate)
        # --- MODIFICATION: Create feedback message object ---
        feedback = ClusteringFeedback()

        rospy.loginfo("Goal received. Starting continuous clustering for grasping.")

        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                rospy.loginfo("Preempted: stopping clustering.")
                self._as.set_preempted()
                break

            evts = self.get_recent_events()
            if len(evts) < self.min_recent_events:
                rate.sleep()
                continue

            pts = np.array([(x, y) for x, y, _ in evts])
            try:
                t_cam, R_cam = self.tf_listener.lookupTransform("base_link", camera_frame, rospy.Time(0))
                T_b_c = build_transform(t_cam, R_cam)
            except (tf.LookupException, tf.ExtrapolationException):
                rospy.logwarn_throttle(2.0, "[GRASPING CLUSTERING] TF transform failed.")
                rate.sleep()
                continue

            # --- MODIFICATION: Pass the size_tol parameter ---
            out = detect_object(pts, T_b_c, self.K, self.dims, size_tol=self.size_tol)

            if not out:
                rospy.loginfo_throttle(1.0, "[GRASPING CLUSTERING] No object detected.")
                self.publish_overlay(None, pts)
                rate.sleep()
                continue

            x3, y3, z3, angle, rect, nbr_exct_pts = out

            # --- MODIFICATION: Populate and publish feedback on every detection ---
            feedback.centroid.header.stamp = rospy.Time.now()
            feedback.centroid.header.frame_id = "base_link"
            feedback.centroid.point.x = x3
            feedback.centroid.point.y = y3
            feedback.centroid.point.z = z3
            self._as.publish_feedback(feedback)
            rospy.loginfo_throttle(1.0, "[GRASPING CLUSTERING] Detected object, publishing feedback.")
            # --- END MODIFICATION ---

            self.publish_overlay(rect, pts)
            rate.sleep()

if __name__ == "__main__":
    try:
        GraspingClusteringServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
