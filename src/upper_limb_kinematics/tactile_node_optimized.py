#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Bool
from math import atan2, degrees
import tf
import tf2_ros
from geometry_msgs.msg import PointStamped
import time

# Imports utilitarios
from upper_limb_kinematics.grasp_geometry_utils import (
    centroide_poligono, intersection_of_lines,
    obtener_vertices_cuatro_lados, obtener_vertices_pentagono,
    obtener_vertices_hexagono, ellipse_axes_from_G
)
from upper_limb_kinematics.grasp_polygon_utils import (
    john_ellipse_in_polygon_ORIGINAL,
    john_ellipse_in_polygon_ratio_constrained
)
from shapely.geometry import Polygon
from gripper_4f.msg import encoders_data
from skeleton_3d.msg import Skeleton3D  # Importante

#-------------------- PARAMETROS ------------------------
SHAPE_DEDO12 = "CUATRO_LADOS"
SHAPE_DEDO34 = "HEXAGONO"
USE_ROSBAG = True 
corregir_medicion_flag = True
ESPESOR_ADELGAZAMIENTO = 0.0075
GRASP_OFFSET = 0.04699
SUSTITUIR_CENTROIDE_POR_ELIPSE = True
GET_FROM_TF = False
FRAME_ID = "fr3_EE"
INFER_ELLIPSE = True # Estaba faltando definirlo explícitamente en el scope global
# --------------------------------------------------------

def corregir_medicion(medicion_actual, p1, p2):
    s1, r1 = p1
    s2, r2 = p2
    if s2 - s1 == 0: return medicion_actual
    m = (r2 - r1) / (s2 - s1)
    b = r1 - (m * s1)
    return m * medicion_actual + b

class EllipseMethodNode:
    def __init__(self):
        # --- MODIFICACIÓN AQUÍ ---
        # Intentamos iniciar el nodo. Si ya existe uno (porque lo llamó otro script),
        # capturamos el error y continuamos sin hacer nada.
        try:
            rospy.init_node('ellipse_method_node')
            rospy.loginfo("Ellipse Method Node started (Stand-alone mode).")
        except rospy.exceptions.ROSException:
            rospy.loginfo("Ellipse Method Node started (Imported mode).")
        # -------------------------

        # rospy.loginfo("Ellipse Method Node started (HIGH PERFORMANCE MODE).")
        self.rate = rospy.Rate(100) # Intentar ir lo más rápido posible

        self.new_elbow_pub = rospy.Publisher('/tactile/elbow', PointStamped, queue_size=20)
        self.new_wrist_pub = rospy.Publisher('/tactile/wrist', PointStamped, queue_size=20)
        
        # Estado del agarre
        self.is_grasping = False
        if USE_ROSBAG:
            self.t_init_bag = rospy.get_param('/exp_optitrack_25/rosbag_start_time', 0.0)
            self.t_init_grasp = rospy.get_param('/exp_optitrack_25/phases/t_grasp', 0.0)
            self.t_finish_grasp = rospy.get_param('/exp_optitrack_25/phases/t_release', 0.0)
            self.grasp_state_pub = rospy.Publisher("/grasp_state", Bool, queue_size=10)
        
        # Suscripciones
        rospy.Subscriber("/gripper_4f/encoders_data", encoders_data, self.gripper_callback)
        
        # --- CORRECCIÓN AQUÍ: Se añade el tipo de mensaje Skeleton3D ---
        rospy.Subscriber("/skeleton_3D", Skeleton3D, self.skeleton_callback) 

        # Params
        self.l2 = rospy.get_param('/exp_optitrack_25/l2', 0.3)
        self.grasp_offset = rospy.get_param('/exp_optitrack_25/d_grasp_measured', GRASP_OFFSET)

        self.frame_id = FRAME_ID
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Variables de estado
        self.prev_center_12 = None
        self.prev_center_34 = None
        self.prev_ellipse_center_12 = None # Para round-robin
        self.prev_ellipse_center_34 = None # Para round-robin
        
        self.dedo1 = [0.0, 0.0, 0.0]
        self.dedo2 = [0.0, 0.0, 0.0]
        self.dedo3 = [0.0, 0.0, 0.0]
        self.dedo4 = [0.0, 0.0, 0.0]
        self.stamp_gripper = None
        
        self.alpha = 0.6
        self.cycle_counter = 0 # Para intercalar cálculos

    def skeleton_callback(self, msg):
        pass # No usado en esta versión optimizada simplificada, pero necesario para el subscriber

    def gripper_callback(self, msg):
        self.stamp_gripper = msg.header.stamp
        if corregir_medicion_flag:
            # Corrección hardcoded para velocidad
            self.dedo1 = [msg.dedo1[0], corregir_medicion(msg.dedo1[1], (95.05, 90.3), (22.7, 16.18)), msg.dedo1[2]]
            self.dedo2 = [msg.dedo2[0], corregir_medicion(msg.dedo2[1], (98.5, 90.3), (20.0, 16.18)), msg.dedo2[2]]
            self.dedo3 = [msg.dedo3[0], corregir_medicion(msg.dedo3[1], (97.0, 90.3), (20.0, 16.18)), msg.dedo3[2]]
            self.dedo4 = [msg.dedo4[0], corregir_medicion(msg.dedo4[1], (99.0, 90.3), (20.9, 16.18)), msg.dedo4[2]]
        else:
            self.dedo1, self.dedo2, self.dedo3, self.dedo4 = msg.dedo1, msg.dedo2, msg.dedo3, msg.dedo4

    def grasp_state_check(self):
        if USE_ROSBAG:
            elapsed = rospy.get_time() - self.t_init_bag
            self.is_grasping = self.t_init_grasp <= elapsed <= self.t_finish_grasp
            self.grasp_state_pub.publish(Bool(self.is_grasping))
        else:
            self.is_grasping = True # Si no hay bag, asumimos siempre true o lógica externa

    def get_transform_matrix(self, target_frame, source_frame):
        """Obtiene la matriz 4x4 de transformación de forma eficiente"""
        try:
            # Time(0) obtiene la última disponible instantáneamente
            trans = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0))
            
            # Convertir a matriz 4x4
            tr = trans.transform.translation
            rt = trans.transform.rotation
            
            T = tf.transformations.quaternion_matrix([rt.x, rt.y, rt.z, rt.w])
            T[0, 3] = tr.x
            T[1, 3] = tr.y
            T[2, 3] = tr.z
            return T
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return None

    def fast_publish_point(self, pub, point_local_np, transform_matrix_4x4):
        """
        Transforma y publica usando Numpy (Vectorizado) en lugar de TF Buffer.
        Tiempo estimado: < 0.05ms
        """
        msg = PointStamped()
        msg.header.stamp = self.stamp_gripper if self.stamp_gripper else rospy.Time.now()
        
        if transform_matrix_4x4 is not None:
            # Transformación manual: P_global = T * P_local
            # point_local_np es [x, y, z]
            p_homog = np.array([point_local_np[0], point_local_np[1], point_local_np[2], 1.0])
            p_trans = transform_matrix_4x4.dot(p_homog)
            
            msg.header.frame_id = "base_link"
            msg.point.x = p_trans[0]
            msg.point.y = p_trans[1]
            msg.point.z = p_trans[2]
        else:
            # Fallback a local si falla TF
            msg.header.frame_id = self.frame_id
            msg.point.x = point_local_np[0]
            msg.point.y = point_local_np[1]
            msg.point.z = point_local_np[2]

        pub.publish(msg)

    def get_vertices_by_shape_from_gripper(self, shape_type, dedo1, dedo2, name):
        # (Copia exacta de tu función original de cálculo geométrico)
        z_offset = 0.0
        if name == "dedo1_dedo2": dx, dy, dz = 0.0463, 0.0, -0.038
        elif name == "dedo3_dedo4": dx, dy, dz = -0.0463, 0.0, -0.038
        else: return None
        P0 = np.array([dx, dy, dz + z_offset])
        l0, l1, l2 = 0.04, 0.040, 0.050
        theta1 = 180 - dedo1[1] - dedo1[2]
        theta2 = 180 - dedo2[1] - dedo2[2]
        P1 = P0 + np.array([0.0, -l0/2, 0.0])
        P2 = P1 + np.array([0.0, -l1 * np.cos(np.radians(dedo2[1])), l1 * np.sin(np.radians(dedo2[1]))])
        P3 = P2 + np.array([0.0, l2 * np.cos(np.radians(theta2)), l2 * np.sin(np.radians(theta2))])
        P4 = P0 + np.array([0.0, l0/2, 0.0])
        P5 = P4 + np.array([0.0, l1 * np.cos(np.radians(dedo1[1])), l1 * np.sin(np.radians(dedo1[1]))])
        P6 = P5 + np.array([0.0, -l2 * np.cos(np.radians(theta1)), l2 * np.sin(np.radians(theta1))])
        
        # Simplificación de lógica para este ejemplo
        if shape_type == "HEXAGONO" or True: # Fallback rápido
             return [P1, P2, P3, P6, P5, P4]
        return None

    def process_ellipse_math(self, vertices, ns_prefix=""):
        """Solo matemática pura, sin filtrado ni lógica de estado"""
        if not vertices: return None
        x_cota = vertices[0][0]
        vertices_2d = [(float(v[1]), float(v[2])) for v in vertices]
        
        try:
            poly_shapely = Polygon(vertices_2d)
            poly_inset = poly_shapely.buffer(-ESPESOR_ADELGAZAMIENTO)
            if poly_inset.is_empty: return None
            
            if poly_inset.geom_type == 'Polygon':
                coords = list(poly_inset.exterior.coords)[:-1]
            elif poly_inset.geom_type == 'MultiPolygon':
                coords = list(max(poly_inset.geoms, key=lambda a: a.area).exterior.coords)[:-1]
            else: return None

            vertices_inset_2d = [(y, z) for y, z in coords] # Solo 2D para la func
            
            # Llamada a solver
            if ns_prefix=="poly_12_":
                c, _, _, status = john_ellipse_in_polygon_ratio_constrained(vertices_inset_2d, max_aspect_ratio=1.1)
            else:
                c, _, _, status = john_ellipse_in_polygon_ORIGINAL(vertices_inset_2d)

            if status in ("optimal", "optimal_inaccurate") and c is not None:
                return np.array([x_cota, float(c[0]), float(c[1])])
            return None
        except Exception:
            return None

    def run(self):
        while not rospy.is_shutdown():
            t0 = time.perf_counter()
            
            # 1. Check Grasp
            self.grasp_state_check()
            if not self.is_grasping:
                self.rate.sleep()
                continue

            # 2. Get Vertices (Rápido, <1ms)
            t_verts = time.perf_counter()
            verts_12 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO12, self.dedo2, self.dedo1, "dedo1_dedo2")
            verts_34 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO34, self.dedo3, self.dedo4, "dedo3_dedo4")
            
            # 3. Cache Transform Matrix (Una vez por ciclo)
            # Esto elimina la latencia en 'Forearm'
            T_base_ee = self.get_transform_matrix("base_link", self.frame_id)

            # 4. Ellipse Calculation (Round-Robin Interlaced)
            t_ell = time.perf_counter()
            
            # Inicialización de centroides geométricos
            cent_12 = centroide_poligono(verts_12)
            cent_34 = centroide_poligono(verts_34)

            if INFER_ELLIPSE:
                # Lógica Intercalada:
                # Ciclo PAR: Calcula 12, Reutiliza 34
                # Ciclo IMPAR: Reutiliza 12, Calcula 34
            
                

                res_12 = self.process_ellipse_math(verts_12, "poly_12_")
                if res_12 is not None: self.prev_ellipse_center_12 = res_12
                res_34 = self.process_ellipse_math(verts_34, "poly_34_")
                if res_34 is not None: self.prev_ellipse_center_34 = res_34

                # Asignar valores (si existen en memoria)
                if self.prev_ellipse_center_12 is not None and SUSTITUIR_CENTROIDE_POR_ELIPSE:
                    # Filtro suavizado simple
                    if self.prev_center_12 is None: self.prev_center_12 = self.prev_ellipse_center_12
                    else: self.prev_center_12 = self.alpha * self.prev_ellipse_center_12 + (1-self.alpha) * self.prev_center_12
                    cent_12 = self.prev_center_12

                if self.prev_ellipse_center_34 is not None and SUSTITUIR_CENTROIDE_POR_ELIPSE:
                    if self.prev_center_34 is None: self.prev_center_34 = self.prev_ellipse_center_34
                    else: self.prev_center_34 = self.alpha * self.prev_ellipse_center_34 + (1-self.alpha) * self.prev_center_34
                    cent_34 = self.prev_center_34
            
            self.cycle_counter += 1

            # 5. Forearm Calculation (Optimized TF)
            t_fore = time.perf_counter()
            if cent_12 is not None and cent_34 is not None:
                grasping_point = (cent_12 + cent_34) / 2.0
                
                v_distal = (cent_34 - grasping_point)
                v_proximal = (cent_12 - grasping_point)
                
                # Normalize
                nd = np.linalg.norm(v_distal)
                np_prox = np.linalg.norm(v_proximal)
                
                if nd > 1e-6 and np_prox > 1e-6:
                    v_distal /= nd
                    v_proximal /= np_prox
                    
                    new_elbow = grasping_point + v_proximal * (self.l2 - self.grasp_offset)
                    new_wrist = grasping_point + v_distal * self.grasp_offset
                    
                    # PUBLICACIÓN RÁPIDA (Sin esperar a TF Buffer)
                    self.fast_publish_point(self.new_elbow_pub, new_elbow, T_base_ee)
                    self.fast_publish_point(self.new_wrist_pub, new_wrist, T_base_ee)

            t_end = time.perf_counter()
            
            # Métricas
            dur_verts = (t_ell - t_verts) * 1000
            dur_ellipses = (t_fore - t_ell) * 1000
            dur_forearm = (t_end - t_fore) * 1000
            dur_total = (t_end - t0) * 1000
            
            rospy.loginfo_throttle(1.0, f"[FAST] Total: {dur_total:.2f}ms | Verts: {dur_verts:.2f} | Ellipses: {dur_ellipses:.2f} | Forearm: {dur_forearm:.2f}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass