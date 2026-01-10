#!/usr/bin/env python3
import rospy
import numpy as np

from franka_msgs.msg import FrankaState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def mat44_from_list(data16, layout="col"):
    """
    Franka suele dar matrices 4x4 en array[16]. En muchas setups es column-major (Eigen).
    - layout="col": reshape con order='F' (column-major)
    - layout="row": reshape normal (row-major)
    """
    a = np.array(data16, dtype=np.float64)
    if a.size != 16:
        raise ValueError("Expected 16 elements for 4x4 matrix.")
    if layout == "col":
        return a.reshape((4, 4), order="F")
    elif layout == "row":
        return a.reshape((4, 4))
    else:
        raise ValueError("layout must be 'col' or 'row'")

def transform_point(T, p):
    """Aplica T(4x4) a punto p(3,)"""
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T @ ph
    return out[:3]

class WrenchLineVisualizer:
    def __init__(self):
        self.pub = rospy.Publisher("wrench_line_markers", MarkerArray, queue_size=1)

        # Params
        self.topic = rospy.get_param("~state_topic", "/franka_state_controller/franka_states")
        self.frame_id = rospy.get_param("~frame_id", "fr3_link0")  # cámbialo si tu base frame es otro
        self.layout = rospy.get_param("~matrix_layout", "col")       # "col" o "row"
        self.force_min = float(rospy.get_param("~force_min", 5.0))   # N: umbral para evitar división por ~0
        self.line_length = float(rospy.get_param("~line_length", 0.30))  # metros en cada dirección
        self.arrow_scale = float(rospy.get_param("~arrow_scale", 0.02))  # escala de flecha
        self.sphere_scale = float(rospy.get_param("~sphere_scale", 0.03))# radio esfera

        rospy.Subscriber(self.topic, FrankaState, self.cb, queue_size=1)
        rospy.loginfo("WrenchLineVisualizer subscribed to %s, publishing MarkerArray on /wrench_line_markers", self.topic)

        # IDs fijos para que RViz no acumule markers
        self.ns = "wrench_line"
        self.id_sphere = 0
        self.id_line = 1
        self.id_arrow = 2

    def make_sphere(self, stamp, pos):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.ns = self.ns
        m.id = self.id_sphere
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = pos.tolist()
        m.pose.orientation.w = 1.0
        m.scale.x = self.sphere_scale
        m.scale.y = self.sphere_scale
        m.scale.z = self.sphere_scale
        # color: amarillo
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.9
        return m

    def make_line(self, stamp, p0, p1):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.ns = self.ns
        m.id = self.id_line
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.006  # grosor línea
        # color: cian
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.9
        m.points = [Point(*p0.tolist()), Point(*p1.tolist())]
        return m

    def make_arrow(self, stamp, base, tip):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self.frame_id
        m.ns = self.ns
        m.id = self.id_arrow
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        # Para ARROW con points: scale.x=shaft diameter, scale.y=head diameter, scale.z=head length
        m.scale.x = self.arrow_scale
        m.scale.y = self.arrow_scale * 2.0
        m.scale.z = self.arrow_scale * 3.0
        # color: rojo
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.0, 0.0, 0.9
        m.points = [Point(*base.tolist()), Point(*tip.tolist())]
        return m

    def cb(self, msg: FrankaState):
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        # 1) Pose de EE en base O
        try:
            O_T_EE = mat44_from_list(msg.O_T_EE, layout=self.layout)
            EE_T_K = mat44_from_list(msg.EE_T_K, layout=self.layout)
        except Exception as e:
            rospy.logwarn_throttle(1.0, "Matrix parse error: %s", str(e))
            return

        # Pose de K en O
        O_T_K = O_T_EE @ EE_T_K

        # Posición de K en base
        p_K = O_T_K[:3, 3]

        # 2) Wrench externo en K expresado en O: [Fx, Fy, Fz, Mx, My, Mz]
        # Wrench en K
        wK = np.array(msg.K_F_ext_hat_K, dtype=np.float64)
        F_K = -wK[0:3]
        M_K = -wK[3:6]

        # Rotación de K en O
        R_OK = O_T_K[:3, :3]

        # Pasar wrench a O (solo rotación, porque fuerza y momento son vectores)
        F = R_OK @ F_K
        M = R_OK @ M_K

        Fnorm = np.linalg.norm(F)
        if Fnorm < self.force_min:
            # Publica borrado o simplemente no publiques
            return

        # 3) Punto equivalente más cercano a O sobre la línea de acción
        # r_perp = (F x M) / ||F||^2
        r_perp = np.cross(F, M) / (Fnorm ** 2)

        # Punto base de la línea (en el espacio) tomado en el frame O:
        p0 = p_K + r_perp

        # Dirección de la línea: unitario de F
        fhat = F / Fnorm

        # Segmento de línea para dibujar (p0 +/- line_length)
        pA = p0 - self.line_length * fhat
        pB = p0 + self.line_length * fhat

        # Flecha proporcional a F (recortada)
        arrow_len = min(0.25, 0.02 * Fnorm)  # escala visual simple
        pTip = p0 + arrow_len * fhat

        # 4) Publica markers
        arr = MarkerArray()
        arr.markers.append(self.make_sphere(stamp, p0))
        arr.markers.append(self.make_line(stamp, pA, pB))
        arr.markers.append(self.make_arrow(stamp, p0, pTip))
        self.pub.publish(arr)

def main():
    rospy.init_node("wrench_line_rviz")
    _ = WrenchLineVisualizer()
    rospy.spin()

if __name__ == "__main__":
    main()
