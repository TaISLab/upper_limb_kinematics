import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from upper_limb_kinematics.grasp_geometry_utils import sample_ellipse # Necesario para draw_overlay

def draw_overlay(marker_pub, hull, c, G, x_cota=0.0, ns_prefix=""):
    """
    Publica en RViz el polígono (hull), la elipse (calculada por c, G) y la diagonal mayor.
    Si marker_pub es None, no publica (solo dibuja si show=True).
    """
    # Polígono
    poly_marker = Marker()
    poly_marker.header.frame_id = "base_gripper"
    poly_marker.header.stamp = rospy.Time.now()
    poly_marker.ns = f"{ns_prefix}poly_overlay"
    poly_marker.id = 1
    poly_marker.type = Marker.LINE_STRIP
    poly_marker.action = Marker.ADD
    poly_marker.scale.x = 0.005
    poly_marker.color.r = 0.2
    poly_marker.color.g = 1.0
    poly_marker.color.b = 0.2
    poly_marker.color.a = 1.0
    poly_marker.points = []
    for v in hull + [hull[0]]:
        pt = Point()
        pt.x = x_cota
        pt.y = v[0]
        pt.z = v[1]
        poly_marker.points.append(pt)
    if marker_pub is not None:
        marker_pub.publish(poly_marker)

    # Elipse
    ellipse_pts = sample_ellipse(c, G, n=100)
    ellipse_marker = Marker()
    ellipse_marker.header.frame_id = "base_gripper"
    ellipse_marker.header.stamp = rospy.Time.now()
    ellipse_marker.ns = f"{ns_prefix}ellipse_overlay"
    ellipse_marker.id = 2
    ellipse_marker.type = Marker.LINE_STRIP
    ellipse_marker.action = Marker.ADD
    ellipse_marker.scale.x = 0.005
    ellipse_marker.color.r = 1.0
    ellipse_marker.color.g = 0.2
    ellipse_marker.color.b = 0.2
    ellipse_marker.color.a = 1.0
    ellipse_marker.points = []
    for v in ellipse_pts:
        pt = Point()
        pt.x = x_cota
        pt.y = v[0]
        pt.z = v[1]
        ellipse_marker.points.append(pt)
    if marker_pub is not None:
        marker_pub.publish(ellipse_marker)

    # Diagonal mayor de la elipse
    ellipse_pts = sample_ellipse(c, G, n=400)
    # Buscar los dos puntos más alejados en la elipse
    max_dist = -1
    idx1, idx2 = 0, 0
    for i in range(len(ellipse_pts)):
        for j in range(i+1, len(ellipse_pts)):
            dist = np.linalg.norm(ellipse_pts[i] - ellipse_pts[j])
            if dist > max_dist:
                max_dist = dist
                idx1, idx2 = i, j
    pt1 = ellipse_pts[idx1]
    pt2 = ellipse_pts[idx2]
    diag_marker = Marker()
    diag_marker.ns = f"{ns_prefix}ellipse_diagonal_overlay"
    diag_marker.header.stamp = rospy.Time.now()
    diag_marker.ns = "ellipse_diagonal_overlay"
    diag_marker.id = 3
    diag_marker.type = Marker.LINE_STRIP
    diag_marker.action = Marker.ADD
    diag_marker.scale.x = 0.01
    diag_marker.color.r = 0.2
    diag_marker.color.g = 0.2
    diag_marker.color.b = 1.0
    diag_marker.color.a = 1.0
    diag_marker.points = []
    for v in [pt1, pt2]:
        pt = Point()
        pt.x = x_cota
        pt.y = v[0]
        pt.z = v[1]
        diag_marker.points.append(pt)
    if marker_pub is not None:
        marker_pub.publish(diag_marker)

    return {"hull": hull, "ellipse": ellipse_pts}

