import rospy
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from upper_limb_kinematics.grasp_geometry_utils import sample_ellipse, ellipse_axes_from_G

def draw_overlay(marker_pub, hull, c, G, x_cota=0.0, ns_prefix="", frame_id="base_gripper"):
    """
    Publica en RViz el polígono (hull), la elipse y su DIAGONAL MAYOR (Eje Mayor).
    Usa cálculo analítico para precisión exacta.
    """
    if marker_pub is None:
        return

    timestamp = rospy.Time.now()

    # --- 1. Dibujar Polígono (Hull) ---
    poly_marker = Marker()
    poly_marker.header.frame_id = frame_id
    poly_marker.header.stamp = timestamp
    poly_marker.ns = f"{ns_prefix}poly_overlay"
    poly_marker.id = 1
    poly_marker.type = Marker.LINE_STRIP
    poly_marker.action = Marker.ADD
    poly_marker.scale.x = 0.005 # Grosor de línea
    poly_marker.color.r = 0.2; poly_marker.color.g = 1.0; poly_marker.color.b = 0.2; poly_marker.color.a = 1.0
    poly_marker.points = []
    
    # Cerrar el polígono repitiendo el primer punto
    for v in hull + [hull[0]]:
        pt = Point()
        pt.x = x_cota
        pt.y = float(v[0])
        pt.z = float(v[1])
        poly_marker.points.append(pt)
    
    marker_pub.publish(poly_marker)

    # --- 2. Dibujar Elipse (Anillo) ---
    ellipse_pts = sample_ellipse(c, G, n=100)
    
    ellipse_marker = Marker()
    ellipse_marker.header.frame_id = frame_id
    ellipse_marker.header.stamp = timestamp
    ellipse_marker.ns = f"{ns_prefix}ellipse_overlay"
    ellipse_marker.id = 2
    ellipse_marker.type = Marker.LINE_STRIP
    ellipse_marker.action = Marker.ADD
    ellipse_marker.scale.x = 0.005
    # Color rojizo para la elipse
    ellipse_marker.color.r = 1.0; ellipse_marker.color.g = 0.2; ellipse_marker.color.b = 0.2; ellipse_marker.color.a = 1.0
    ellipse_marker.points = []
    
    for v in ellipse_pts:
        pt = Point()
        pt.x = x_cota
        pt.y = float(v[0])
        pt.z = float(v[1])
        ellipse_marker.points.append(pt)
        
    marker_pub.publish(ellipse_marker)

    # --- 3. Dibujar Diagonal Mayor (Eje Mayor Analítico) ---
    # Obtenemos los ejes analíticamente desde la matriz G
    # a: semieje mayor, v_major: vector director del eje mayor
    a, b_val, v_major, v_minor, _ = ellipse_axes_from_G(G)

    # Calculamos los puntos extremos: Centro +/- (vector * longitud_semieje)
    p_start = c - v_major * a
    p_end   = c + v_major * a

    diag_marker = Marker()
    diag_marker.header.frame_id = frame_id
    diag_marker.header.stamp = timestamp
    # IMPORTANTE: Usar ns_prefix para diferenciar entre elipse 12 y 34
    diag_marker.ns = f"{ns_prefix}ellipse_diagonal_overlay" 
    diag_marker.id = 3
    diag_marker.type = Marker.LINE_LIST # Usamos LINE_LIST para una línea recta limpia
    diag_marker.action = Marker.ADD
    diag_marker.scale.x = 0.008  # Un poco más grueso que la elipse para resaltar
    # Color Azulado para la diagonal
    diag_marker.color.r = 0.2; diag_marker.color.g = 0.2; diag_marker.color.b = 1.0; diag_marker.color.a = 1.0
    
    pt1 = Point(); pt1.x = x_cota; pt1.y = float(p_start[0]); pt1.z = float(p_start[1])
    pt2 = Point(); pt2.x = x_cota; pt2.y = float(p_end[0]);   pt2.z = float(p_end[1])
    
    diag_marker.points = [pt1, pt2]

    marker_pub.publish(diag_marker)

    return {"hull": hull, "ellipse": ellipse_pts}