#!/usr/bin/env python
import rospy
import numpy as np
from collections import deque
import actionlib

from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import tf
import cv2

from action_client_interface.msg import ClusteringAction, ClusteringFeedback, ClusteringResult
from utils import detect_object, build_transform, draw_detected_rect

camera_frame = "dvx_camera_link"

class Event2DClusteringActionServer:
    def __init__(self):
        rospy.init_node("event_2d_clustering_action_server")

        #  your existing parameters 
        self.time_window        = rospy.get_param("~time_window",        0.1)
        self.max_events         = rospy.get_param("~max_events",         3000)
        self.cluster_eps        = rospy.get_param("~cluster_eps",        6)
        self.min_samples        = rospy.get_param("~min_samples",        10)
        self.min_recent_events  = rospy.get_param("~min_recent_events",  50)
        self.stability_threshold= rospy.get_param("~stability_threshold",2.0)
        self.process_rate       = rospy.get_param("~process_rate",      10.0)
        self.dims               = rospy.get_param("~object_dims",[0.085,0.060,0.145])
        
        self.K                  = np.array([[296.0598696737529, 0.0, 171.34936457371717],
                                            [0.0, 297.676018255805, 133.2210310205901],
                                            [0.0, 0.0, 1.0]])

        #  your existing state 
        self.events      = deque(maxlen=self.max_events)
        self.bridge      = CvBridge()
        self.image       = None
        self.tf_listener = tf.TransformListener()
        self.last_centroid = None

        #  your existing subs 
        rospy.Subscriber("/dvs/events",    EventArray, self.event_callback, queue_size=10)
        rospy.Subscriber("/dvs_rendering", Image,      self.image_callback, queue_size=1)

        #  remove the old 2D centroid publisher 
        # self.centroid_pub = rospy.Publisher("/object/centroid_2d", PointStamped, queue_size=1)

        #  keep your overlay / bbox pubs 
        self.overlay_pub = rospy.Publisher("/object/overlay", Image, queue_size=1)
        self.bbox_pub    = rospy.Publisher("/object/bbox_2d", Float32MultiArray, queue_size=1)

        #  set up the Action Server 
        self._as = actionlib.SimpleActionServer(
            "cluster_events",
            ClusteringAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo("Event2DClusteringActionServer ready waiting for goals...")

        rospy.spin()


    #  unchanged callbacks 
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
        if self.image is None:
            return
        ov = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for x, y in cluster.astype(int):
            if 0 <= x < ov.shape[1] and 0 <= y < ov.shape[0]:
                cv2.circle(ov, (x, y), 1, (0,255,0), -1)

        if rect is not None:
            ov = draw_detected_rect(ov, rect, color=(255,0,0), thickness=2)
            bbox = Float32MultiArray(data=[
                rect[0][0], rect[0][1],
                rect[1][0], rect[1][1]
            ])
            self.bbox_pub.publish(bbox)

        msg = self.bridge.cv2_to_imgmsg(ov, "bgr8")
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = camera_frame
        self.overlay_pub.publish(msg)

    def clear_overlay(self):
        if self.image is None:
            return
        ov = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        msg = self.bridge.cv2_to_imgmsg(ov, "bgr8")
        msg.header.stamp    = rospy.Time.now()
        msg.header.frame_id = camera_frame
        self.overlay_pub.publish(msg)


    #  new execute cb driving your old process logic at fixed rate 
    def execute_cb(self, goal):
        rate     = rospy.Rate(self.process_rate)
        feedback = ClusteringFeedback()
        result   = ClusteringResult()

        rospy.loginfo("Goal received starting periodic clustering at %.1f Hz", self.process_rate)

        while not rospy.is_shutdown():
            # handle cancel
            if self._as.is_preempt_requested():
                rospy.loginfo("Preempted stopping clustering")
                self._as.set_preempted(result)
                return

            evts = self.get_recent_events()
            if len(evts) < self.min_recent_events:
		rospy.loginfo("[CLUSTERING] Not enough recent events: %d", len(evts))
                self.clear_overlay()
                self.last_centroid = None
                rate.sleep()
                continue

            pts = np.array([(x,y) for x,y,_ in evts])
            if len(pts) < self.min_samples:
		rospy.loginfo("[CLUSTERING] Not enough points after filtering: %d", len(pts))
                self.clear_overlay()
                self.last_centroid = None
                rate.sleep()
                continue

            # TF and clustering exactly as before
            try:
                t_cam, R_cam = self.tf_listener.lookupTransform("base_link", camera_frame, rospy.Time(0))
            except (tf.LookupException, tf.ExtrapolationException):
		rospy.loginfo("[CLUSTERING] TF transform failed.")
                rate.sleep()
                continue
            T_b_c = build_transform(t_cam, R_cam)

            out = detect_object(pts, T_b_c, self.K, self.dims)
            if not out:
		rospy.loginfo("[CLUSTERING] No object detected in events.")
                self.clear_overlay()
                self.last_centroid = None
                self.publish_overlay(None, pts)
                rate.sleep()
                continue

            #  assume out = (x3, y3, z3, rect)
	   
            #x3, y3, z3, rect = out
	    x3, y3, z3, angle, rect = out
	    #angle = angle
            self.last_centroid = (x3, y3, z3)

            #  publish 3D centroid as feedback 
            feedback.centroid.header.stamp    = rospy.Time.now()
            feedback.centroid.header.frame_id = "base_link"
            feedback.centroid.point.x         = x3
            feedback.centroid.point.y         = y3
            feedback.centroid.point.z         = z3
	    feedback.angle                    = angle
	    rospy.loginfo("[CLUSTERING] Success: Centroid detected: (%.3f, %.3f, %.3f)", x3, y3, z3)
            self._as.publish_feedback(feedback)

            #  keep your overlay as before 
            self.publish_overlay(rect, pts)

            rate.sleep()

        # if ROS is shutting down
        self._as.set_aborted(result)



if __name__ == "__main__":
    try:
        Event2DClusteringActionServer()
    except rospy.ROSInterruptException:
        pass

