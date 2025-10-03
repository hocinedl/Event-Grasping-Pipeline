#!/usr/bin/env python
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from action_client_interface.msg import ClusteringActionFeedback
import numpy as np
from collections import deque
from sklearn.cluster import DBSCAN
import math

# Parameters
BUFFER_SIZE         = rospy.get_param('~buffer_size', 50)
MIN_SAMPLES         = rospy.get_param('~min_samples_total', 20)
DBSCAN_EPS          = rospy.get_param('~dbscan_eps', 0.05)  # meters
DBSCAN_MIN_SAMPLES  = rospy.get_param('~dbscan_min_samples', 5)

# Rolling buffer of recent [x,y,z,angle]
buffer = deque(maxlen=BUFFER_SIZE)

def cb(msg):
    """
    Callback for clustering feedback: collect x,y,z plus angle.
    """
    p     = msg.feedback.centroid.point
    ang   = msg.feedback.angle                      # <--- added angle field
    buffer.append([p.x, p.y, p.z, ang])
    rospy.logdebug("[filter] Collected (x,y,z,theta)= (%.3f,%.3f,%.3f,%.3f). Buf: %d/%d",
                   p.x, p.y, p.z, ang, len(buffer), BUFFER_SIZE)

def handle_filter(req):
    """
    Service handler: DBSCAN on the 3D points, pick largest cluster,
    compute mean x,y,z and circular mean of angle.
    """
    n = len(buffer)
    if n < MIN_SAMPLES:
        msg = "Not enough samples ({}/{})".format(n, MIN_SAMPLES)
        rospy.logwarn("[filter] %s", msg)
        return TriggerResponse(False, msg)

    data = np.array(buffer)
    xyz  = data[:, :3]
    angs = data[:, 3]

    rospy.loginfo("[filter] Running DBSCAN on %d samples (eps=%.3f, min_samples=%d)",
                  n, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz)
    labels = clustering.labels_

    # only keep inlier clusters (label >= 0)
    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if valid_labels.size == 0:
        msg = "No clusters found"
        rospy.logwarn("[filter] %s", msg)
        return TriggerResponse(False, msg)

    # pick the largest cluster
    best_label = valid_labels[np.argmax(counts)]
    mask       = (labels == best_label)
    cluster_xyz = xyz[mask]
    cluster_ang = angs[mask]

    # mean x,y,z
    mean_xyz = cluster_xyz.mean(axis=0)
    x, y, z  = mean_xyz.tolist()

    # circular mean of angles
    sin_sum = np.sin(cluster_ang).sum()
    cos_sum = np.cos(cluster_ang).sum()
    mean_ang = math.atan2(sin_sum, cos_sum)

    rospy.loginfo("[filter] Cluster %d with %d/%d pts  mean (x,y,z,theta)= (%.3f,%.3f,%.3f,%.3f)",
                  best_label, mask.sum(), n, x, y, z, mean_ang)

    # clear buffer for next round
    buffer.clear()

    # return x,y,z,angle
    msg = "{:.3f},{:.3f},{:.3f},{:.3f}".format(x, y, z, mean_ang)
    return TriggerResponse(True, msg)

if __name__ == '__main__':
    rospy.init_node('filtering_node')
    rospy.Subscriber('/cluster_events/feedback', ClusteringActionFeedback, cb)
    rospy.Service('/filter_centroid', Trigger, handle_filter)
    rospy.loginfo("[filter] Service ready. Buf=%d, min_samples=%d, eps=%.3f, min_pts=%d",
                  BUFFER_SIZE, MIN_SAMPLES, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    rospy.spin()

