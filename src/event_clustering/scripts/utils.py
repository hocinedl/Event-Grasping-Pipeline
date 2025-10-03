import rospy
import math
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import tf.transformations as tft


def back_project_point(u, v, K, T, z_obj, Z):
    """
    Back-projects a 2D pixel point to a 3D point in the base frame.
    """
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # Back-project to camera frame
    x_cam = (u - cx) * Z / fx
    y_cam = (v - cy) * Z / fy
    p_cam = np.array([x_cam, y_cam, z_obj, 1.0])
    
    # Transform to base frame
    p_base = np.dot(T, p_cam)
    return p_base[:3] # Return x,y,z


def compute_cluster_centroid(data, min_samples, eps, dbscan_min_samples) :
    n = data.shape[0]
    if n < min_samples:
        return None

    xyz  = data[:, :3]
    angs = data[:, 3]
    pts = data[:, 4]
    
    clustering = DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit(xyz)
    labels = clustering.labels_

    valid_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if valid_labels.size == 0:
        return None

    best_label = valid_labels[np.argmax(counts)]
    mask = (labels == best_label)

    mean_xyz = xyz[mask].mean(axis=0)
    mean_ang = angs[mask].mean()
    valid_pts = pts[mask].mean()

    rospy.loginfo("[CLUSTERING] %d mean number of excited points found", valid_pts)
    return float(mean_xyz[0]), float(mean_xyz[1]), float(mean_xyz[2]), float(mean_ang)



def build_transform(trans, rot):
    """
    Build a 4 4 homogeneous transform from a translation and quaternion.

    Args:
        trans (tuple or list of float (3,)):  x, y, z translation
        rot   (tuple or list of float (4,)):  x, y, z, w quaternion

    Returns:
        np.ndarray (4 4):  T_base_cam so that
                            [X_base]   [ R  t ] [X_cam]
                            [  1   ] = [ 0  1 ] [  1  ]
    """
    # 1) make sure we have a numpy array
    t = np.array(trans, dtype=np.float64)
    q = np.array(rot,   dtype=np.float64)

    # 2) build a 4 4 matrix from the quaternion
    T = tft.quaternion_matrix(q)   # yields [[R, 0],
                                    #         [0, 1]]

    # 3) overwrite the translation part
    T[0:3, 3] = t

    return T

def draw_detected_rect(img, rect, color=(0,255,0),thickness=2) :
    """
    Draw a colored rotated rectangle on an image, handling
    grayscale or float images by converting to uint8 BGR.

    Args:
        img      : input image, grayscale or BGR, uint8 or float32[0..1].
        rect     : ((cx,cy),(w,h),angle) from cv2.minAreaRect.
        color    : BGR color (0 255 each).
        thickness: line thickness.

    Returns:
        BGR uint8 image with the rectangle drawn.
    """
    out = img.copy()

    # 1) If float in [0..1], scale to [0..255]
    if out.dtype == np.float32 or out.dtype == np.float64:
        out = np.clip(out * 255, 0, 255).astype(np.uint8)

    # 2) If single channel, convert to BGR
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # 3) Now draw the rectangle
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(out, [box], 0, color, thickness)

    # 4) Draw center point
    (cx, cy), _, _ = rect
    center = (int(round(cx)), int(round(cy)))
    cv2.circle(out, center, 4, color, -1)

    return out


def detect_object(points, T, K, dims, db_eps=5, db_min_samples=10, size_tol=0.6): #was 0.3
    """
    Detect a known-size rectangular object in an event cloud.
    """
    l, w, z_obj = dims
    # 1) cluster events in image space
    clustering = DBSCAN(eps=db_eps, min_samples=db_min_samples).fit(points)
    labels = clustering.labels_

    # compute object's distance along camera z:
    z_cam_base = T[2, 3]
    Z = z_cam_base - z_obj
    if Z <= 0:
        print("Object is too close to the camera: Z={:.2f} <= 0".format(Z))
        return None

    # predict pixel size
    fx, fy = K[0,0], K[1,1]
    f_mean = (fx + fy) / 2.0
    exp_long = f_mean * max(l,w) / Z
    exp_short= f_mean * min(l,w) / Z

    best = None
    best_err = np.inf
    for lbl in set(labels):
        if lbl < 0:
            continue
        pts = points[labels == lbl].astype(np.float32)
        if pts.shape[0] < 200:  #was4
            continue

        # 2) fit a rotated rectangle
        rect = cv2.minAreaRect(pts)  
        (cx, cy), (w_px, h_px), ang = rect

        # sort so actual_long be actual_short
        actual = sorted([w_px, h_px], reverse=True)

        # 3) measure size mismatch
        if actual[1] == 0:
            print("Detected rectangle has zero height/width, skipping.")
            continue
        err =  ((actual[0] - exp_long) / exp_long)**2 + ((actual[1] - exp_short) / exp_short)**2 
        err = np.sqrt(err)  # Euclidean distance in size space
        print("object angle={:.2f} err={:.2f} ".format(ang, err))
        if err < size_tol and err < best_err:
            best_err = err
            best = (cx, cy, Z, rect, len(pts))
        
    if best is None:
        print("No object found with size tolerance {:.2f}".format(size_tol))
        return None

    cx, cy, Z, rect, pts = best

    # --- NEW ORIENTATION LOGIC ---
    # Get the 4 corner points of the rectangle in the image
    box = cv2.boxPoints(rect)

    # The two longest sides of the rectangle define its orientation.
    # We find one of these sides by comparing the lengths of adjacent edges.
    edge1_len = np.linalg.norm(box[0] - box[1])
    edge2_len = np.linalg.norm(box[1] - box[2])

    if edge1_len > edge2_len:
        p1_img, p2_img = box[0], box[1]
    else:
        p1_img, p2_img = box[1], box[2]

    # Back-project these two 2D points into the 3D base frame
    p1_base = back_project_point(p1_img[0], p1_img[1], K, T, z_obj, Z)
    p2_base = back_project_point(p2_img[0], p2_img[1], K, T, z_obj, Z)

    # Calculate the yaw angle in the base frame (in radians)
    dx = p2_base[0] - p1_base[0]
    dy = p2_base[1] - p1_base[1]
    yaw_3d_rad = math.atan2(dy, dx)
    # --- END OF NEW LOGIC ---

    # 4) back project centroid into camera frame
    p_base_centroid = back_project_point(cx, cy, K, T, z_obj, Z)
    
    # Return the centroid, the *new* 3D yaw, the rect for drawing, and point count
    return p_base_centroid[0], p_base_centroid[1], p_base_centroid[2], yaw_3d_rad, rect, pts




