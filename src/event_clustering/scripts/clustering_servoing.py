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
from utils import detect_object, build_transform, draw_detected_rect, compute_cluster_centroid

camera_frame = "dvx_camera_link"

class Event2DClusteringActionServer:
    def __init__(self):
        rospy.init_node("event_2d_clustering_action_server")  #working but not the best

        # Parameters
        self.time_window        = rospy.get_param("~time_window", 0.1) #was 2.5  worked 4.3 4.5 3,5 2,5 the first we start with was 0.3 0.3 best so far was 0,5   0.15 time_window  0.05
        self.max_events         = rospy.get_param("~max_events", 3000) #was  5000 3000 
        self.min_recent_events  = rospy.get_param("~min_recent_events", 30)  #was  30 50 100
        self.process_rate       = rospy.get_param("~process_rate", 20.0)  #was 10
        self.dims               = rospy.get_param("~object_dims", [0.085, 0.060, 0.145])
        self.size_tol           = rospy.get_param("~size_tol", 3.5) #was 2 Make size tolerance a parameter 3.5
        
        # Parameters for filtering the stream of poses
        self.pose_history_size  = rospy.get_param('~pose_history_size', 25)  #was 20 30 # Number of recent poses to filter
        self.dbscan_eps         = rospy.get_param('~dbscan_eps', 0.05)  #was 0.08 0.05 meters
        self.dbscan_min_samples = rospy.get_param('~dbscan_min_samples', 4) #was  3 5

        self.K = np.array([[296.06, 0.0, 171.345],
                           [0.0, 297.68, 133.221],
                           [0.0, 0.0, 1.0]])

        # State
        self.events = deque(maxlen=self.max_events)
        self.bridge = CvBridge()
        self.image = None
        self.tf_listener = tf.TransformListener()

        # Subscribers
        rospy.Subscriber("/dvs/events", EventArray, self.event_callback, queue_size=10)
        rospy.Subscriber("/dvs_rendering", Image, self.image_callback, queue_size=1)

        # Publisher
        self.overlay_pub = rospy.Publisher("/object/overlay", Image, queue_size=1)

        # Action Server
        self._as = actionlib.SimpleActionServer(
            "cluster_events", # This is the action name for this server
            ClusteringAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo("Event2DClusteringActionServer ready waiting for goals...")
        rospy.spin()

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

    def publish_overlay(self, rect, cluster):
        if self.image is None: return
        ov = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        if cluster is not None:
            for x, y in cluster.astype(int):
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
        feedback = ClusteringFeedback()
        
        # Use a deque for an efficient sliding window of recent poses
        pose_history = deque(maxlen=self.pose_history_size)

        rospy.loginfo("Goal received. Starting continuous, filtered clustering.")
        
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
                rospy.logwarn_throttle(2.0, "[CLUSTERING] TF transform failed.")
                rate.sleep()
                continue

            # Use the size_tol parameter
            out = detect_object(pts, T_b_c, self.K, self.dims, size_tol=self.size_tol)
            self.publish_overlay(out[4] if out else None, pts) # Publish overlay regardless of detection

            if not out:
                rospy.loginfo_throttle(1.0, "[CLUSTERING] No object detected in events.")
                rate.sleep()
                continue

            # Add the new raw detection to our history
            x_raw, y_raw, z_raw, angle_raw, rect, nbr_exct_pts = out
            pose_history.append((x_raw, y_raw, z_raw, angle_raw, nbr_exct_pts))
            
            # We need enough history to run the filter
            if len(pose_history) < self.dbscan_min_samples:
                rate.sleep()
                continue

            # Run the filtering on the recent history of poses
            filtered_pose = compute_cluster_centroid(np.array(pose_history), self.dbscan_min_samples, self.dbscan_eps, self.dbscan_min_samples)

            if filtered_pose:
                x_stable, y_stable, z_stable, angle_stable = filtered_pose
                
                # Publish the STABLE centroid as feedback
                feedback.centroid.header.stamp = rospy.Time.now()
                feedback.centroid.header.frame_id = "base_link"
                feedback.centroid.point.x = x_stable
                feedback.centroid.point.y = y_stable
                feedback.centroid.point.z = z_stable
                self._as.publish_feedback(feedback)
                rospy.loginfo_throttle(1.0, "[CLUSTERING] Publishing stable centroid.")
            else:
                rospy.loginfo_throttle(1.0, "[CLUSTERING] Filter removed all poses, no stable centroid.")

            rate.sleep()
        
        # When loop is finished, set a success state so the client doesn't hang
        rospy.loginfo("[CLUSTERING] Execution finished.")
        self._as.set_succeeded(ClusteringResult())

# --- FIX: Corrected the indentation of this block ---
if __name__ == "__main__":
    try:
        Event2DClusteringActionServer()
    except rospy.ROSInterruptException:
        pass
