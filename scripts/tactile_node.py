#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String, Bool  # Cambia el tipo de mensaje según tus necesidades
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import cvxpy as cp
from math import atan2, degrees
import tf  # <--- Añadir esta línea para importar tf
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped, PoseStamped, TransformStamped, Vector3Stamped

import math
from typing import Sequence, Tuple, List
from shapely.geometry import Polygon
from upper_limb_kinematics.msg import AngleStamped
from gripper_4f.msg import encoders_data
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP

# Funciones auxiliares importadas
from upper_limb_kinematics.grasp_geometry_utils import (
    _signed_area_2d,
    offset_polygon_2d,
    convex_hull,
    polygon_halfspaces_ccw,
    ellipse_axes_from_G,
    sample_ellipse,
    longest_diagonal,
    centroide_poligono,
    incentro_poligono,
    intersection_of_lines,
    obtener_vertices_cuatro_lados,
    obtener_vertices_pentagono,
    obtener_vertices_hexagono,
    intersection_sphere_line,
)
from upper_limb_kinematics.grasp_polygon_utils import (
    john_ellipse_in_polygon,
    john_ellipse_in_polygon_ORIGINAL,
    john_ellipse_in_polygon_constrained,
    john_ellipse_in_polygon_ratio_constrained
)
from upper_limb_kinematics.grasp_visualization_utils import (
    draw_overlay,
)

#-------------------- PARAMETROS ------------------------

# --- GEOMETRIA DE CADA DEDO ---
# Opciones disponibles: "CUATRO_LADOS", "PENTAGONO", "HEXAGONO"
SHAPE_DEDO12 = "CUATRO_LADOS"   # Configuración para el par Dedo 1 y 2 (más cerca del codo)
SHAPE_DEDO34 = "HEXAGONO"       # Configuración para el par Dedo 3 y 4 (más cerca de la muneca)

# gripper_4f
corregir_medicion_flag = True # Los encoders magnéticos de la garra tienen una leve no linealidad que se corrige con dos puntos de calibración
ESPESOR_ADELGAZAMIENTO = 0.0075 # Metros

# GRASP
GET_D_GRASP_FROM_ROS_PARAM = True
GRASP_OFFSET = 0.04699 # HARDCODED [metros]
USE_ROSBAG = True
CALCULATE_Q5 = False

# --- OPCIONES DE CÁLCULO ---
SUSTITUIR_CENTROIDE_POR_ELIPSE = True  # True -> usar centro de elipse; False -> usar centroide geométrico
GET_FROM_TF = False  # True -> vértices desde TF; False -> gripper_4f
FOREARM_CALCULATION = True
INFER_ELLIPSE = True
ELBOW_AS_INTERSECTION_SPHERE_LINE = True

# --- FRAMES DE REFERENCIA ---
FRAME_ID = "fr3_EE"
PUBLISH_PS_IN_BASE_LINK = True # Publicar PointStamped en base_link (transformados) o en FRAME_ID (base_gripper, original)

# --------------------------------------------------------
def get_rotation_matrix(tfBuffer, source_frame, target_frame, time=rospy.Time(0)):
    """
    Obtiene la matriz de rotación (3x3) entre dos frames usando tf2_ros.
    Args:
        tfBuffer: instancia de tf2_ros.Buffer
        source_frame: str, frame origen (por ejemplo, 'base_link')
        target_frame: str, frame destino (por ejemplo, 'fr3_EE')
        time: rospy.Time, tiempo de la transformación (por defecto, 0 = más reciente)
    Returns:
        R: np.array shape (3,3), matriz de rotación
    """
    trans = tfBuffer.lookup_transform(source_frame, target_frame, time)
    q = trans.transform.rotation
    quat = [q.x, q.y, q.z, q.w]
    R = tf.transformations.quaternion_matrix(quat)[:3, :3]
    return R

def plane_from_points(p1, p2, p3):
    """
    Calcula el plano que contiene a tres puntos.
    Args:
        p1, p2, p3: array-like, puntos 3D (x, y, z)
    Returns:
        n: vector normal unitario al plano (np.array shape (3,))
        d: término independiente (float)
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    # Vectores en el plano
    v1 = p2 - p1
    v2 = p3 - p1
    # Vector normal (sentido según el orden de los puntos)
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        raise ValueError("Los puntos son colineales o coinciden")
    n = n / n_norm
    # Término independiente
    d = -np.dot(n, p1)
    return n, d

def corregir_medicion(medicion_actual, p1, p2):
    """
    Corrige una medición basándose en dos puntos de calibración (interpolación lineal).
    
    Args:
        medicion_actual (float): El valor que acabas de leer del sensor.
        p1 (tuple): (sensor_1, real_1) -> Primer punto de calibración.
        p2 (tuple): (sensor_2, real_2) -> Segundo punto de calibración.
        
    Returns:
        float: El valor corregido (real).
    """
    s1, r1 = p1  # Desempaquetar punto 1 (sensor, real)
    s2, r2 = p2  # Desempaquetar punto 2 (sensor, real)
    
    # Evitar división por cero si los puntos del sensor son iguales
    if s2 - s1 == 0:
        return medicion_actual # O lanzar un error
    
    # 1. Calcular la pendiente (m)
    m = (r2 - r1) / (s2 - s1)
    
    # 2. Calcular el offset (b) -> y = mx + b  =>  b = y - mx
    b = r1 - (m * s1)
    
    # 3. Aplicar corrección
    return m * medicion_actual + b

class EllipseMethodNode:
    def __init__(self):
        rospy.init_node('ellipse_method_node')
        rospy.loginfo("Ellipse Method Node started.")
        self.rate = rospy.Rate(100)  # 10 Hz

        # Publisher para los puntos en RViz
        self.marker_pub = rospy.Publisher('/ellipse_vertices_marker', Marker, queue_size=20)
        self.overlay_pub = rospy.Publisher('/ellipse_overlay_marker', Marker, queue_size=20)
        self.angle_pub_12 = rospy.Publisher('/tactile/ellipse_12_angle_deg', AngleStamped, queue_size=1)
        self.angle_pub_34 = rospy.Publisher('/tactile/ellipse_34_angle_deg', AngleStamped, queue_size=1)

        self.debug_sphere_pub = rospy.Publisher('/debug/shoulder_sphere', Marker, queue_size=1)
        self.debug_line_pub = rospy.Publisher('/debug/intersection_line', Marker, queue_size=1)
        self.forearm_line_pub = rospy.Publisher('/tactile/forearm_line_constraint', PoseStamped, queue_size=1)

        # Centroides hexagono garra
        self.centroide12_pub = rospy.Publisher('/tactile/centroide_12', PointStamped, queue_size=20)
        self.centroide_trapecio_12_pub = rospy.Publisher('/tactile/centroide_trapecio_12', PointStamped, queue_size=20)
        self.centroide34_pub = rospy.Publisher('/tactile/centroide_34', PointStamped, queue_size=20)
        self.grasping_point_pub = rospy.Publisher('/tactile/grasping_point', PointStamped, queue_size=20)
        self.new_elbow_pub = rospy.Publisher('/tactile/elbow', PointStamped, queue_size=20)
        self.new_wrist_pub = rospy.Publisher('/tactile/wrist', PointStamped, queue_size=20)
        self.centro_normal_to_q2_pub = rospy.Publisher('/tactile/centro_normal_q2', PointStamped, queue_size=20)
        self.elbow_point_pub = rospy.Publisher('/tactile/elbow_point_intersection', PointStamped, queue_size=20)

        # --- NUEVO: Publishers Dummy (Cinemática Pura) ---
        self.dummy_elbow_pub = rospy.Publisher('/dummy/elbow', PointStamped, queue_size=20)
        self.dummy_wrist_pub = rospy.Publisher('/dummy/wrist', PointStamped, queue_size=20)
        # -------------------------------------------------

        # subscripción a gripper_4f
        rospy.Subscriber("/gripper_4f/encoders_data", encoders_data, self.gripper_callback)
        rospy.Subscriber("/skeleton_3D", Skeleton3D, self.skeleton_callback)
        
        # rospy params
        self.l1 = rospy.get_param('/exp_optitrack_25/l1', 0.3)  # Longitud del brazo
        self.l2 = rospy.get_param('/exp_optitrack_25/l2', 0.3)  # Longitud del antebrazo

        # ROSBAGS
        if USE_ROSBAG:
            self.t_init_bag = rospy.get_param('/exp_optitrack_25/rosbag_start_time', 0.0)  # Tiempo inicial del bag
            self.t_init_grasp = rospy.get_param('/exp_optitrack_25/phases/t_grasp', 0.0)  # Tiempo inicial de agarre
            self.t_finish_grasp = rospy.get_param('/exp_optitrack_25/phases/t_release', 0.0)  # Tiempo final de agarre
            self.grasp_state = rospy.Publisher("/grasp_state", Bool, queue_size=10)
            
        
        if GET_D_GRASP_FROM_ROS_PARAM:
            self.grasp_offset = rospy.get_param('/exp_optitrack_25/d_grasp_measured', GRASP_OFFSET)  # Offset del punto de agarre
        else:
            self.grasp_offset = GRASP_OFFSET

        rospy.loginfo(f"L2 (get from ros params): {self.l2} m")
        rospy.loginfo(f"Grasp offset (get from ros params): {self.grasp_offset} m")
        
        self.frame_id = FRAME_ID

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Variables para filtro de suavizado (Exponential Moving Average)
        self.alpha = 0.6  # Factor de suavizado (0.0 = infinito, 1.0 = sin filtro)
        self.prev_center_12 = None  # Memoria exclusiva para dedos 1-2
        self.prev_center_34 = None  # Memoria exclusiva para dedos 3-4
        self.prev_G = None

        self.dedo1 = [0.0, 0.0, 0.0]
        self.dedo2 = [0.0, 0.0, 0.0]
        self.dedo3 = [0.0, 0.0, 0.0]
        self.dedo4 = [0.0, 0.0, 0.0]

        self.stamp_gripper = None
        self.stamp_skeleton = None
        self.rshoulder = None
        self.roll_arm_ee = 0.0
        self.v_major_34 = None  # Vector del eje mayor de la elipse dedos 3-4

        # --- Variables para el filtro del codo ---
        self.prev_elbow = None  # Memoria del filtro
        self.elbow_alpha = 0.3  # Factor de suavizado (0.0 a 1.0)
                                # 1.0 = Sin filtro (puro ruido)
                                # 0.1 = Muy suave (mucho lag)
                                # 0.2 - 0.3 = Buen balance para tracking humano

    def skeleton_callback(self, msg):
        # rospy.loginfo("Recibido Skeleton3D.")
        self.stamp_skeleton = msg.header.stamp

        # Extraer puntos clave relevantes: hombro derecho
        rshoulder = msg.keypoints[6]
        self.rshoulder = np.array([rshoulder.x, rshoulder.y, rshoulder.z])


    def grasp_state_publisher(self):
        """
        Publica el estado del agarre en el tópico /grasp_state.
        Args:
            bool is_grasping: True si está en fase de agarre, False si no.
        """

        current_time = rospy.get_time()
        elapsed_time = current_time - self.t_init_bag

        is_grasping = self.t_init_grasp <= elapsed_time <= self.t_finish_grasp

        grasp_msg = Bool()
        grasp_msg.data = True if is_grasping else False
        self.grasp_state.publish(grasp_msg)
        
        

    def gripper_callback(self, msg):
        
        self.stamp_gripper = msg.header.stamp
        # El sensor tiene una leve no linealidad que se corrige con dos puntos de calibración
        if corregir_medicion_flag == True:
            # rospy.loginfo("Corregir mediciones de los dedos usando calibración.")
            dedo11 = corregir_medicion(msg.dedo1[1], (95.05, 90.3), (22.7, 16.18) ) # sensor, real
            dedo21 = corregir_medicion(msg.dedo2[1], (98.5, 90.3), (20.0, 16.18))
            dedo31 = corregir_medicion(msg.dedo3[1], (97.0, 90.3), (20.0, 16.18))
            dedo41 = corregir_medicion(msg.dedo4[1], (99.0, 90.3), (20.9, 16.18))

            self.dedo1 = [msg.dedo1[0], dedo11, msg.dedo1[2]]
            self.dedo2 = [msg.dedo2[0], dedo21, msg.dedo2[2]]
            self.dedo3 = [msg.dedo3[0], dedo31, msg.dedo3[2]]
            self.dedo4 = [msg.dedo4[0], dedo41, msg.dedo4[2]]
        else:
            self.dedo1 = msg.dedo1
            self.dedo2 = msg.dedo2
            self.dedo3 = msg.dedo3
            self.dedo4 = msg.dedo4
    
    def calculate_and_publish_dummy_kinematics(self):
        """
        Calcula Codo y Muñeca basándose puramente en el frame fr3_EE y los parámetros L2 y Grasp Offset.
        - Muñeca: Dirección -X a distancia D_grasp
        - Codo: Dirección +X a distancia (L2 - D_grasp)
        """
        try:
            # Usamos el tiempo actual o el último disponible para asegurar que encontramos la tf
            # Si quieres sincronía estricta con el gripper, usa self.stamp_gripper
            tf_time = rospy.Time(0) 

            # 1. Definir MUÑECA en frame local (fr3_EE)
            # Dirección negativa de X a D_grasp distancia
            p_wrist_local = PointStamped()
            p_wrist_local.header.frame_id = "fr3_EE"
            p_wrist_local.header.stamp = tf_time
            p_wrist_local.point.x = -self.grasp_offset
            p_wrist_local.point.y = 0.0
            p_wrist_local.point.z = 0.0

            # 2. Definir CODO en frame local (fr3_EE)
            # Dirección positiva como L2 - D_grasp
            p_elbow_local = PointStamped()
            p_elbow_local.header.frame_id = "fr3_EE"
            p_elbow_local.header.stamp = tf_time
            p_elbow_local.point.x = self.l2 - self.grasp_offset
            p_elbow_local.point.y = 0.0
            p_elbow_local.point.z = 0.0

            # 3. Transformar a base_link y publicar
            # (Es necesario transformar porque RViz suele tener fixed frame en base_link/world)
            if self.tfBuffer.can_transform("base_link", "fr3_EE", tf_time, rospy.Duration(0.1)):
                p_wrist_global = self.tfBuffer.transform(p_wrist_local, "base_link")
                p_elbow_global = self.tfBuffer.transform(p_elbow_local, "base_link")
                
                # Actualizar timestamp al momento de publicación si usamos Time(0) para TF
                p_wrist_global.header.stamp = rospy.Time.now()
                p_elbow_global.header.stamp = rospy.Time.now()

                self.dummy_wrist_pub.publish(p_wrist_global)
                self.dummy_elbow_pub.publish(p_elbow_global)
            else:
                rospy.logwarn_throttle(2.0, "No se encuentra transformación de fr3_EE a base_link para dummy kinematics")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"Error en dummy kinematics TF: {e}")

    def publish_vectors_marker(self, points, vectors, frame_id="fr3_EE", ns="vectors", color=(0.2, 0.2, 1.0)):
        """
        Publica los vectores como Marker tipo ARROW en RViz.
        Args:
            points: lista de puntos de origen (np.array)
            vectors: lista de vectores (np.array)
            frame_id: frame de referencia
            ns: namespace del marker
            color: tupla RGB
        """
        for i, (p, v) in enumerate(zip(points, vectors)):
            # Primero, elimina el marker anterior con este id/ns
            delete_marker = Marker()
            delete_marker.header.frame_id = frame_id
            delete_marker.header.stamp = rospy.Time.now()
            delete_marker.ns = ns
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            self.marker_pub.publish(delete_marker)

            # Publicar nuevo marker
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = ns
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.005  # grosor
            marker.scale.y = 0.01   # ancho de la flecha
            marker.scale.z = 0.01
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.points = []
            p_start = Point()
            p_start.x, p_start.y, p_start.z = p[0], p[1], p[2] if len(p) > 2 else 0.0
            p_end = Point()
            p_end.x = p[0] + v[0]
            p_end.y = p[1] + v[1]
            p_end.z = p[2] + v[2] if len(p) > 2 else 0.0
            marker.points.append(p_start)
            marker.points.append(p_end)
            self.marker_pub.publish(marker)

    def publish_vertices_marker(self, vertices, frame_id="fr3_EE", ns="ellipse_vertices", id=0):
        """
        Publica los vértices como un Marker tipo SPHERE_LIST en RViz.
        Versión flexible: Acepta cualquier lista de vértices (3, 4, 5, 6...).
        """
        # CAMBIO AQUÍ: Validar que haya al menos 3 vértices, no exactamente 6
        if vertices is None or len(vertices) < 3:
            rospy.logwarn(f"publish_vertices_marker: Se requieren al menos 3 vértices. Recibidos: {len(vertices) if vertices else 0}")
            return

        marker = Marker()
        marker.header.frame_id = frame_id # Usa frame_id pasado como argumento para consistencia
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # diámetro de las esferas
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        
        # Color diferente según el ID para distinguirlos visualmente (opcional)
        if id == 0: # Dedo 3-4 (suele ser Rojo)
            marker.color.r = 1.0; marker.color.g = 0.2; marker.color.b = 0.2
        else:       # Dedo 1-2 (suele ser Azul o Verde)
            marker.color.r = 0.2; marker.color.g = 0.2; marker.color.b = 1.0
            
        marker.color.a = 1.0
        marker.points = []
        
        for i, v in enumerate(vertices):
            if v is None or len(v) < 3:
                continue
            p = Point()
            p.x = float(v[0])
            p.y = float(v[1])
            p.z = float(v[2])
            marker.points.append(p)

        self.marker_pub.publish(marker)

    def publish_pointstamped(self, pub, point, frame_id="fr3_EE"):
        """
        Publica un punto como geometry_msgs/PointStamped.
        Args:
            pub: rospy.Publisher configurado para PointStamped
            point: array-like (x, y, z)
            frame_id: frame de referencia
        """
        
        target_frame = 'base_link'  # Frame al que se quiere transformar

        msg = PointStamped()
        
        if self.stamp_gripper is not None:
            msg.header.stamp = self.stamp_gripper  ### INTENTO DE SINCRONIZACIÓN ###

            msg.header.frame_id = frame_id
            msg.point.x = float(point[0])
            msg.point.y = float(point[1])
            msg.point.z = float(point[2])

            # Transformar si el frame de destino es diferente
            if frame_id != target_frame and PUBLISH_PS_IN_BASE_LINK:
                try:
                    msg = self.tfBuffer.transform(msg, target_frame, rospy.Duration(1.0))
                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                    rospy.logwarn(f"TF transform failed: {e}")
                    return

            pub.publish(msg)

    def calculate_ellipse_from_vertices(self, vertices, ns_prefix=""):
        """
        Función principal que calcula la elipse de mayor área inscrita en un pentágono
        dado por sus 5 vértices (en formato de lista de tuplas).
        
        Args:
        - vertices (list of tuples): Los 5 vértices del pentágono en formato [(x1, y1), (x2, y2), ...]
        
        Returns:
        - diccionario con parámetros de la elipse (centro, semi-ejes, ángulo, etc.)
        """

        # guardar cota x
        x = vertices[0][0]
        # Quitar la coordenada X:
        vertices = [(float(v[1]), float(v[2])) for v in vertices]

        #################### BUSCAR ELIPSE #####################################
        # MÉTODOS:
        # john_ellipse_in_polygon -> va bien
        # john_ellipse_in_polygon_ORIGINAL -> va mejor para poly34
        # john_ellipse_in_polygon_ratio_constrained -> va bien para poly12

        if ns_prefix=="poly_12_":
            c, G, hull, status = john_ellipse_in_polygon_ratio_constrained(
                vertices,
                max_aspect_ratio=1.1,  # Relación de aspecto máxima (a/b)
            )

        elif ns_prefix=="poly_34_":
            c, G, hull, status = john_ellipse_in_polygon_ORIGINAL(vertices)

        #########################################################################

        if status not in ("optimal", "optimal_inaccurate") or c is None or G is None:
            rospy.logwarn(f"No se pudo calcular una elipse {ns_prefix} válida.")
            return {
                "centro_pixeles": (x, None, None),
                "semi_eje_mayor_a_px": None,
                "semi_eje_menor_b_px": None,
                "angulo_eje_mayor_grados": None,
                "poligono_vertices": vertices,
                "elipse_vertices": None,
                "v_major_3d": None,
            }

        # rospy.loginfo(f"Matriz G de la elipse:\n{G}")
        
        # Extraemos los ejes de la elipse
        a, b, v_major, v_minor, _ = ellipse_axes_from_G(G)
        # rospy.loginfo(f"{ns_prefix} Elipse calculada: a={a*100:.2f} cm, b={b*100:.2f} cm")

        # Calculamos el ángulo del eje mayor
        angle_deg = degrees(atan2(v_major[1], v_major[0]))
        
        # Dibujar la elipse junto con el polígono
        info = draw_overlay(self.overlay_pub, hull, c, G, x_cota=x, ns_prefix=ns_prefix, frame_id=self.frame_id)

        elipse_pts = sample_ellipse(c, G, n=400)
        elipse_pts_with_x = np.column_stack([np.full(elipse_pts.shape[0], x), elipse_pts])

        # Devolver los parámetros importantes de la elipse
        return {
            "centro_pixeles": (x, float(c[0]), float(c[1])),
            "semi_eje_mayor_a_px": float(a),
            "semi_eje_menor_b_px": float(b),
            "angulo_eje_mayor_grados": float(angle_deg),
            "poligono_vertices": vertices,
            "elipse_vertices": elipse_pts_with_x,  # Puntos de la elipse
            "v_major_3d": np.array([0.0, v_major[0], v_major[1]]),  # Vector 3D del eje mayor
        }

    def get_vertices_by_shape_from_TF(self, shape_type, dedoX, dedoY):
        """ Dispatcher actualizado """
        if shape_type == "CUATRO_LADOS":
            return obtener_vertices_cuatro_lados(self.tfBuffer, dedoX, dedoY, self.frame_id)
        elif shape_type == "PENTAGONO":
            return obtener_vertices_pentagono(self.tfBuffer, dedoX, dedoY, self.frame_id)
        elif shape_type == "HEXAGONO":
            v1, v2, v3, v4, v5, v6 = obtener_vertices_hexagono(self.tfBuffer, dedoX, dedoY, self.frame_id)
            if v1 is not None: return [v1, v2, v3, v4, v5, v6]
        return None

    def get_vertices_by_shape_from_gripper(self, shape_type, dedo1, dedo2, name):
        """ 
        Obtener puntos de los dedos basado en los ángulos de las falanges para distintas figuras
        dedo1 - semiplano y positivo
        dedo2 - semiplano y negativo
        
        """
        # z_offset = +0.012 # CORRECCION ERROR EN EL MODELADO DEL AGARRE ENTRE ROBOT Y GRIPPER
        # z_offset = 0.02
        z_offset = 0.0

        TF_fr3_EE_to_base_dedo12_x = 0.0463
        TF_fr3_EE_to_base_dedo12_y = 0.0
        TF_fr3_EE_to_base_dedo12_z = -0.038

        TF_fr3_EE_to_base_dedo34_x = -0.0463
        TF_fr3_EE_to_base_dedo34_y = 0.0
        TF_fr3_EE_to_base_dedo34_z = -0.038

        l0 = 0.04 # Distancia entre origen dedo11 y dedo21 (o dedo31 y dedo41)
        l1 = 0.040  # Longitud de la primera falange
        l2 = 0.050  # Longitud de la segunda falange


        # TF fr3_EE a base_falange (pto medio entre dedo10 y dedo20; o dedo30 y dedo40)
        if name == "dedo1_dedo2":
            P0 = np.array([TF_fr3_EE_to_base_dedo12_x, 
                           TF_fr3_EE_to_base_dedo12_y, 
                           TF_fr3_EE_to_base_dedo12_z + z_offset])  # Origen en fr3_EE dedo10 y dedo20

        elif name == "dedo3_dedo4":
            P0 = np.array([TF_fr3_EE_to_base_dedo34_x, 
                           TF_fr3_EE_to_base_dedo34_y, 
                           TF_fr3_EE_to_base_dedo34_z + z_offset])  # Origen en fr3_EE dedo30 y dedo40
        else:
            rospy.logerr(f"get_vertices_by_shape_from_gripper: Nombre desconocido {name}")
            P0 = np.array([0.0, 0.0, 0.0])


        theta1 = 180 - dedo1[1] - dedo1[2]
        theta2 = 180 - dedo2[1] - dedo2[2]



        # imprimir ángulos para depuración
        # if name == "dedo1_dedo2":
        #     rospy.loginfo("--------------------------------------------------------------")
        #     rospy.loginfo(f"{name} - Ángulos Dedo1: {dedo1}, Ángulos Dedo2: {dedo2}")
        #     rospy.loginfo(f"{name} - Theta1: {theta1}, Theta2: {theta2}")
        #     rospy.loginfo(f"{name} - dedo11: {dedo1[1]}, dedo12: {dedo1[2]}, dedo21: {dedo2[1]}, dedo22: {dedo2[2]}")

        # if name == "dedo3_dedo4":
        #     rospy.loginfo("--------------------------------------------------------------")
        #     rospy.loginfo(f"{name} - Ángulos Dedo3: {dedo1}, Ángulos Dedo4: {dedo2}")
        #     rospy.loginfo(f"{name} - Theta1: {theta1}, Theta2: {theta2}")
        #     rospy.loginfo(f"{name} - dedo31: {dedo1[1]}, dedo32: {dedo1[2]}, dedo41: {dedo2[1]}, dedo42: {dedo2[2]}")

        # Semiplano Y negativo
        P1 = P0 + np.array([0.0, -l0/2, 0.0])
        P2 = P1 + np.array([0.0, -l1 * np.cos(np.radians(dedo2[1])), l1 * np.sin(np.radians(dedo2[1]))])
        P3 = P2 + np.array([0.0, l2 * np.cos(np.radians(theta2)), l2 * np.sin(np.radians(theta2))])

        # Semiplano Y positivo
        P4 = P0 + np.array([0.0, l0/2, 0.0])
        P5 = P4 + np.array([0.0, l1 * np.cos(np.radians(dedo1[1])), l1 * np.sin(np.radians(dedo1[1]))])
        P6 = P5 + np.array([0.0, -l2 * np.cos(np.radians(theta1)), l2 * np.sin(np.radians(theta1))])


        # Comprobar figuras:
        if name == "dedo1_dedo2" and shape_type == "CUATRO_LADOS" and self.check_rombo_feasibility(dedo1, dedo2) == False:
            # rospy.logwarn_throttle(2.0, f"get_vertices_by_shape_from_gripper: No es posible formar un ROMBO con los ángulos dados: {dedo1}, {dedo2}. Usando HEXÁGONO como fallback.")
            shape_type = "HEXAGONO"

        if name == "dedo1_dedo2" and shape_type == "PENTAGONO" and self.check_pentagon_feasibility(dedo1, dedo2) == False:
            # rospy.logwarn_throttle(2.0, f"get_vertices_by_shape_from_gripper: No es posible formar un PENTAGONO con los ángulos dados: {dedo1}, {dedo2}. Usando HEXÁGONO como fallback.")
            shape_type = "HEXAGONO"


        if shape_type == "CUATRO_LADOS":
            # --- 1. Intersección Delantera (Hacia las puntas) ---
            # Definida por las falanges distales: Vector Codo -> Punta
            vec_distal_neg = P3 - P2
            origen_distal_neg = P2
            
            vec_distal_pos = P6 - P5
            origen_distal_pos = P5

            # Calculamos dónde se cruzarían las puntas
            rombo_tips = intersection_of_lines(origen_distal_neg, vec_distal_neg, origen_distal_pos, vec_distal_pos)

            # --- 2. Intersección Trasera (Hacia la muñeca) ---
            # Definida por las falanges proximales proyectadas hacia atrás: Vector Codo -> Base
            # Nota: P1 y P4 son las bases fijas
            vec_proximal_neg_back = P1 - P2
            origen_proximal_neg = P2 

            vec_proximal_pos_back = P4 - P5
            origen_proximal_pos = P5

            # Calculamos dónde se cruzarían hacia atrás (cierre del rombo)
            rombo_base = intersection_of_lines(origen_proximal_neg, vec_proximal_neg_back, origen_proximal_pos, vec_proximal_pos_back)

            # --- 3. Protecciones (Fallback) ---
            # Si los dedos están paralelos, intersection devuelve None. Usamos puntos medios.
            if rombo_tips is None:
                rombo_tips = (P3 + P6) / 2.0 # Punto medio entre puntas reales
            
            if rombo_base is None:
                rombo_base = (P1 + P4) / 2.0 # Punto medio entre bases reales

            # --- 4. Definición de Vértices (Orden Cíclico) ---
            # Orden: Codo Izq -> Puntas(virtual) -> Codo Der -> Base(virtual)
            vertices = [P2, rombo_tips, P5, rombo_base]
            
            return vertices

        elif shape_type == "PENTAGONO":
            # Vectores
            falange12 = P3 - P2
            origen_falange12 = P2

            falange22 = P6 - P5
            origen_falange22 = P5

            # Calcular intersección de las líneas que pasan por las falanges
            penta5 = intersection_of_lines(origen_falange12, falange12, origen_falange22, falange22)

            # Protección contra None ---
            if penta5 is None:
                # Si las líneas son paralelas, usamos el punto medio de las puntas reales
                penta5 = (P3 + P6) / 2.0

            vertices = [P1, P2, penta5, P5, P4]
            return vertices
        
        elif shape_type == "HEXAGONO":
            vertices = [P1, P2, P3, P6, P5, P4]
            return vertices

        return None

    def calculate_forearm_general(self, centroide_proximal, centroide_distal):
        """
        Cálculo general del antebrazo.
        Args:
            centroide_proximal: Centroide del par 1-2 (lado muñeca/pulgar)
            centroide_distal: Centroide del par 3-4 (lado dedos/exterior)
        """
        if centroide_proximal is None or centroide_distal is None:
            return

        # 1. Punto de agarre (Promedio de los dos centroides)
        grasping_point = (centroide_proximal + centroide_distal) / 2.0
        self.publish_pointstamped(self.grasping_point_pub, grasping_point, frame_id=self.frame_id)

        # 2. Vectores directores
        # Vector hacia los dedos (3-4)
        v_to_distal = (centroide_distal - grasping_point)
        norm_distal = np.linalg.norm(v_to_distal)
        if norm_distal < 1e-6: return # Evitar div/0
        v_to_distal /= norm_distal

        # Vector hacia la muñeca (1-2)
        v_to_proximal = (centroide_proximal - grasping_point)
        norm_proximal = np.linalg.norm(v_to_proximal)
        if norm_proximal < 1e-6: return
        v_to_proximal /= norm_proximal

        # 3. Cálculo de posiciones (Elbow y Wrist) en local
        v_elbow_dir = v_to_proximal 
        v_wrist_dir = v_to_distal

        new_elbow = grasping_point + v_elbow_dir * (self.l2 - self.grasp_offset)
        new_wrist = grasping_point + v_wrist_dir * self.grasp_offset

        self.publish_pointstamped(self.new_elbow_pub, new_elbow, frame_id=self.frame_id)
        self.publish_pointstamped(self.new_wrist_pub, new_wrist, frame_id=self.frame_id)


        if CALCULATE_Q5:
            # --- PASO 1: OBTENER TF COMPLETA (Rotación + Traslación) ---
            try:
                # Buscamos la transformación T de base_link a fr3_EE
                trans = self.tfBuffer.lookup_transform('base_link', self.frame_id, rospy.Time(0))
                
                # 1.1 Extraer Traslación (T)
                t_vec = np.array([trans.transform.translation.x, 
                                trans.transform.translation.y, 
                                trans.transform.translation.z])
                
                # 1.2 Extraer Rotación (R)
                q = trans.transform.rotation
                R_bl_to_ee = tf.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

                # --- PASO 2: TRANSFORMAR PUNTOS A BASE_LINK ---
                # Formula: P_global = R * P_local + T
                elbow_BL = np.dot(R_bl_to_ee, new_elbow) + t_vec
                wrist_BL = np.dot(R_bl_to_ee, new_wrist) + t_vec
                
                # Nota: rshoulder ya debe estar en base_link (verifícalo en tu skeleton_callback)
                shoulder_BL = self.rshoulder 

                # --- CORRECCIÓN AQUÍ: Protección contra NoneType ---
                if shoulder_BL is None:
                    rospy.logwarn_throttle(2.0, "Esperando datos del esqueleto (shoulder is None). Saltando cálculo de orientación del brazo.")
                    return 
                # ----------------------------------------------------

                # --- PASO 3: DEFINIR LOS EJES DEL BRAZO (TU SISTEMA PERSONALIZADO) ---
                
                # EJE X_arm: Vector unitario que va del Codo a la Muñeca (Longitudinal)
                X_arm = wrist_BL - elbow_BL
                X_arm /= np.linalg.norm(X_arm)

                # EJE Z_arm: Normal al plano Hombro-Codo-Muñeca
                # Vector A: Hombro -> Codo
                v_shoulder_elbow = elbow_BL - shoulder_BL # <--- Aquí fallaba antes
                
                # Producto Cruz para hallar la normal (Z)
                Z_arm = np.cross(v_shoulder_elbow, X_arm)
                norm_z = np.linalg.norm(Z_arm)
                
                # Protección por si el brazo está totalmente estirado (Singularidad: puntos colineales)
                if norm_z < 1e-6:
                    # Fallback: Usar Z del base_link o una referencia fija si están alineados
                    Z_arm = np.array([0, 0, 1]) 
                else:
                    Z_arm /= norm_z

                # EJE Y_arm: Para completar la base ortonormal (Z cross X)
                Y_arm = np.cross(Z_arm, X_arm)
                Y_arm /= np.linalg.norm(Y_arm)

                # --- PASO 4: CONSTRUIR LA MATRIZ DE ROTACIÓN (R_base_arm) ---
                # Las columnas de la matriz de rotación son los vectores unitarios X, Y, Z
                R_base_to_arm = np.column_stack((X_arm, Y_arm, Z_arm))

                # =================================================================
                # PUBLICAR EJES EN RVIZ (TF BROADCASTER)
                # =================================================================
                try:
                    # 1. Convertir Matriz 3x3 a Cuaternio (requiere matriz 4x4)
                    T_4x4 = np.eye(4)
                    T_4x4[:3, :3] = R_base_to_arm
                    quat_arm = tf.transformations.quaternion_from_matrix(T_4x4)

                    # 2. Crear mensaje TransformStamped
                    t_msg = TransformStamped()
                    t_msg.header.stamp = rospy.Time.now()
                    t_msg.header.frame_id = "base_link"          # Frame padre
                    t_msg.child_frame_id = "human_arm"    # Nombre del nuevo frame en RViz
                    
                    # 3. Asignar Posición (La muñeca en coordenadas base_link)
                    t_msg.transform.translation.x = wrist_BL[0]
                    t_msg.transform.translation.y = wrist_BL[1]
                    t_msg.transform.translation.z = wrist_BL[2]

                    # 4. Asignar Orientación (Los ejes calculados)
                    t_msg.transform.rotation.x = quat_arm[0]
                    t_msg.transform.rotation.y = quat_arm[1]
                    t_msg.transform.rotation.z = quat_arm[2]
                    t_msg.transform.rotation.w = quat_arm[3]

                    # 5. Publicar
                    self.tf_broadcaster.sendTransform(t_msg)
                
                except Exception as e:
                    rospy.logwarn(f"Error publicando TF de ejes humanos: {e}")
                # =================================================================



                # --- PASO 5: Calcular R_Arm_EE ---
                R_arm_to_ee = np.dot(R_base_to_arm.T, R_bl_to_ee)

                diagonal_in_EE = self.v_major_34

                # Transformar si el frame de destino es diferente

                try:
                    target_frame = 'human_arm'  # Frame al que se quiere transformar
                    vector3stamped_msg = Vector3Stamped()
                    vector3stamped_msg.header.frame_id = "fr3_EE"
                    vector3stamped_msg.header.stamp = rospy.Time.now()
                    vector3stamped_msg.vector.x = diagonal_in_EE[0]
                    vector3stamped_msg.vector.y = diagonal_in_EE[1]
                    vector3stamped_msg.vector.z = diagonal_in_EE[2]

                    diagonal_arm_vector3stamped = self.tfBuffer.transform(vector3stamped_msg, target_frame, rospy.Duration(1.0))

                except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                    rospy.logwarn(f"TF transform failed: {e}")
                    return



                # el sentido no tiene sentido fisico, luego, limitamos a YEE- y ZEE-
                
                # if diagonal_in_arm[1] > 0 and diagonal_in_arm[2] > diagonal_in_arm[1]:
                #     diagonal_in_arm = diagonal_in_arm
                # elif diagonal_in_arm[1] < 0 and diagonal_in_arm[2] < diagonal_in_arm[1]:
                #     diagonal_in_arm = diagonal_in_arm
                # else:
                #     diagonal_in_arm = -diagonal_in_arm

                diagonal_in_arm = np.array([diagonal_arm_vector3stamped.vector.x,
                                            diagonal_arm_vector3stamped.vector.y,
                                            diagonal_arm_vector3stamped.vector.z])

                theta = atan2(diagonal_in_arm[1], diagonal_in_arm[2])  # Ángulo en el plano YZ_arm




                if theta is not None:
                    
                    theta = theta % (2 * np.pi) # Asegurar que esté en [0, 2π]
                    
                    # if theta > np.pi:
                    #     theta -= np.pi


                    q5 = 90 - degrees(theta)

                    rospy.loginfo(f"theta (deg): {degrees(theta):.2f} grados.")
                    rospy.loginfo(f"q5 estimado: {q5:.2f} grados.")

                    # Construir mensaje y publicar
                    msg = AngleStamped()

                    if self.stamp_gripper is not None:
                        msg.header.stamp = self.stamp_gripper  ### INTENTO DE SINCRONIZACIÓN ###
                    else:
                        msg.header.stamp = rospy.Time.now()

                    msg.angle = q5
                    self.angle_pub_34.publish(msg)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Error transformando a base_link: {e}")
                return
        



    
    def draw_forearm_volume(self, info_12, info_34, d1, d2, step=20):
        """
        Dibuja el volumen (wireframe) entre la elipse 12 y la elipse 34,
        extendiéndolo:
        - d1 más allá de la elipse 34 siguiendo la dirección local de la superficie (p12[i] -> p34[i])
        - d2 más allá de la elipse 12 siguiendo la dirección local opuesta (p34[i] -> p12[i])
        """
        if info_12 is None or info_34 is None:
            return

        pts_12 = info_12.get("elipse_vertices")  # Nx3
        pts_34 = info_34.get("elipse_vertices")  # Nx3
        if pts_12 is None or pts_34 is None:
            return

        n = min(len(pts_12), len(pts_34))
        if n < 2:
            return

        # --- 1) Direcciones locales de la superficie ---
        # Para cada i: dir_i = normalize(pts_34[i] - pts_12[i])
        dirs = pts_34[:n] - pts_12[:n]
        norms = np.linalg.norm(dirs, axis=1)

        # Fallback: si alguna norma es ~0, usamos la dirección centro->centro
        c12 = np.array(info_12.get("centro_pixeles"), dtype=float)
        c34 = np.array(info_34.get("centro_pixeles"), dtype=float)
        vec_global = c34 - c12
        ng = np.linalg.norm(vec_global)
        if ng < 1e-9:
            vec_global = np.array([1.0, 0.0, 0.0])
        else:
            vec_global = vec_global / ng

        dirs_unit = np.zeros_like(dirs)
        ok = norms > 1e-9
        dirs_unit[ok] = dirs[ok] / norms[ok][:, None]
        dirs_unit[~ok] = vec_global  # fallback local

        # --- 2) Elipses extendidas usando la dirección local (no un vector global fijo) ---
        d1 = float(d1)
        d2 = float(d2)
        pts_34_ext = pts_34[:n] + dirs_unit * d1
        pts_12_ext = pts_12[:n] - dirs_unit * d2

        # --- 3) Wireframe longitudinal (LINE_LIST) ---
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "forearm_volume"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.002

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.4

        marker.points = []

        step = max(1, int(step))
        for i in range(0, n, step):
            p12e = Point(x=float(pts_12_ext[i][0]), y=float(pts_12_ext[i][1]), z=float(pts_12_ext[i][2]))
            p12  = Point(x=float(pts_12[i][0]),     y=float(pts_12[i][1]),     z=float(pts_12[i][2]))
            p34  = Point(x=float(pts_34[i][0]),     y=float(pts_34[i][1]),     z=float(pts_34[i][2]))
            p34e = Point(x=float(pts_34_ext[i][0]), y=float(pts_34_ext[i][1]), z=float(pts_34_ext[i][2]))

            # Prolongación proximal: 12_ext -> 12
            marker.points.append(p12e); marker.points.append(p12)

            # Tramo central: 12 -> 34
            marker.points.append(p12); marker.points.append(p34)

            # Prolongación distal: 34 -> 34_ext
            marker.points.append(p34); marker.points.append(p34e)

        self.marker_pub.publish(marker)

        # --- 4) Anillos finales (caps) ---
        def publish_ring(ns, mid, pts_ring, thickness=0.004, alpha=0.85):
            ring = Marker()
            ring.header.frame_id = self.frame_id
            ring.header.stamp = rospy.Time.now()
            ring.ns = ns
            ring.id = mid
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = thickness
            ring.color.r = 0.0
            ring.color.g = 1.0
            ring.color.b = 1.0
            ring.color.a = alpha
            ring.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts_ring]
            ring.points.append(Point(x=float(pts_ring[0][0]), y=float(pts_ring[0][1]), z=float(pts_ring[0][2])))
            self.marker_pub.publish(ring)

        publish_ring("forearm_cap_12_ext", 1, pts_12_ext)
        publish_ring("forearm_cap_34_ext", 2, pts_34_ext)

    def normal_point_to_q2(self, dedoX, dedoY, frame_base="fr3_EE"):
        """
        Calcula el punto de corte entre las normales desplazadas de las falanges 2 de dedo1 y dedo2.
        """
        try:
            # Obtener transformaciones
            t_D1 = self.tfBuffer.lookup_transform(frame_base, "dedo1_link2", rospy.Time(0))
            t_D2 = self.tfBuffer.lookup_transform(frame_base, "dedo2_link2", rospy.Time(0))

            # Posiciones
            p_D1 = np.array([t_D1.transform.translation.x, t_D1.transform.translation.y, t_D1.transform.translation.z])
            p_D2 = np.array([t_D2.transform.translation.x, t_D2.transform.translation.y, t_D2.transform.translation.z])

            # Orientaciones
            q_D1 = t_D1.transform.rotation
            q_D2 = t_D2.transform.rotation

            q_D1_tuple = [q_D1.x, q_D1.y, q_D1.z, q_D1.w]
            q_D2_tuple = [q_D2.x, q_D2.y, q_D2.z, q_D2.w]

            R_D1 = tf.transformations.quaternion_matrix(q_D1_tuple)[:3, :3]
            R_D2 = tf.transformations.quaternion_matrix(q_D2_tuple)[:3, :3]

            # Eje X local
            x_D1 = R_D1[:, 0]
            x_D2 = R_D2[:, 0]

            # Desplazamiento 0.02 en eje X local
            p_D1_x = p_D1 + 0.02 * x_D1
            p_D2_x = p_D2 + 0.02 * x_D2

            # Aplanar en el eje Z
            z_flat_D1 = p_D1_x[2]
            z_flat_D2 = p_D2_x[2]
            p_D1_x = np.array([p_D1_x[0], p_D1_x[1], z_flat_D1])
            p_D2_x = np.array([p_D1_x[0], p_D2_x[1], z_flat_D1])

            # Usar eje Y local como dirección de la normal
            y_D1 = R_D1[:, 1]
            y_D2 = R_D2[:, 1]

            # Calcular punto de intersección en el plano YZ
            punto_corte = intersection_of_lines(p_D1_x, y_D1, p_D2_x, y_D2)

            if punto_corte is not None:
                # rospy.loginfo(f"Punto de corte entre normales: {punto_corte}")
                # Opcional: publicar el punto en RViz
                self.publish_pointstamped(self.grasping_point_pub, punto_corte, frame_id=frame_base)

                # Dibuja los vectores normales en RViz
                # self.publish_vectors_marker(
                #     points=[p_D1_x, p_D2_x],
                #     vectors=[y_D1, y_D2],
                #     frame_id=frame_base,
                #     ns="normales_dedos",
                #     color=(0.0, 1.0, 1.0)  # Cian para distinguir
                # )
            else:
                rospy.logwarn("No se encontró punto de corte entre las normales.")

            return punto_corte

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Error al obtener las transformaciones para el cálculo de corte de normales.")
            return None
    
    def check_rombo_feasibility(self, dedo1, dedo2):
        """
        Comprueba si la configuración actual de los dedos permite formar un rombo válido.
        Args:
            dedo1: semiplano y positivo
            dedo2: semiplano y negativo
        Returns:
            True si es factible, False si no lo es.
        """
        
        # Medir ángulos falange2 desde eje horizontal positivo (y+)
        alpha1 = dedo1[1] + dedo1[2]
        alpha2 = 180 - dedo2[1] - dedo2[2]
        beta1 = dedo1[1]
        beta2 = 180 - dedo2[1]

        # rospy.loginfo(f"alpha1: {alpha1}")
        # rospy.loginfo(f"alpha2: {alpha2}")
        # rospy.loginfo(f"beta1: {beta1}")
        # rospy.loginfo(f"beta2: {beta2}")

        if alpha1 > alpha2 and beta2 > beta1:
            return True
        else:
            rospy.logwarn_throttle(1, f"check_rombo_feasibility: NOT FACTIBLE")
            return False
          
    def check_pentagon_feasibility(self, dedo1, dedo2):
        """
        Comprueba si la configuración actual de los dedos permite formar un pentagono válido.
        Args:
            dedo1: ángulos del dedo1
            dedo2: ángulos del dedo2
        Returns:
            True si es factible, False si no lo es.
        """
        
        alpha1 = dedo1[1] + dedo1[2]
        alpha2 = 180 - dedo2[1] - dedo2[2]


        if alpha1 < alpha2:
            return True
        else:
            rospy.logwarn_throttle(1, f"check_pentagon_feasibility: NOT FACTIBLE")
            return False

    def run(self):
        while not rospy.is_shutdown():
            
            # 0. Publicar estado del agarre. Solo para rosbag
            if USE_ROSBAG:
                self.grasp_state_publisher()
            
            # 1. Obtener vértices
            self.calculate_and_publish_dummy_kinematics()
            
            if GET_FROM_TF:
                verts_12 = self.get_vertices_by_shape_from_TF(SHAPE_DEDO12, "dedo1", "dedo2")
                verts_34 = self.get_vertices_by_shape_from_TF(SHAPE_DEDO34, "dedo3", "dedo4")

            else:
                verts_12 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO12, self.dedo2, self.dedo1, name="dedo1_dedo2")
                verts_34 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO34, self.dedo3, self.dedo4, name="dedo3_dedo4")

            if not verts_12 or not verts_34:
                rospy.logwarn_throttle(2.0, "Esperando transforms...")
                self.rate.sleep()
                continue
            
            # # Verificar que se obtienen los puntos necesarios según la forma seleccionada
            # if SHAPE_DEDO12 == "PENTAGONO" and len(verts_12) != 5:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 5 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
            #     self.rate.sleep()
            #     continue
            # if SHAPE_DEDO34 == "PENTAGONO" and len(verts_34) != 5:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 5 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
            #     self.rate.sleep()
            #     continue

            # if SHAPE_DEDO12 == "CUATRO_LADOS" and len(verts_12) != 4:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 4 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
            #     self.rate.sleep()
            #     continue
            # if SHAPE_DEDO34 == "CUATRO_LADOS" and len(verts_34) != 4:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 4 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
            #     self.rate.sleep()
            #     continue

            # if SHAPE_DEDO12 == "HEXAGONO" and len(verts_12) != 6:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 6 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
            #     self.rate.sleep()
            #     continue
            # if SHAPE_DEDO34 == "HEXAGONO" and len(verts_34) != 6:
            #     rospy.logwarn_throttle(2.0, f"Se esperaban 6 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
            #     self.rate.sleep()
            #     continue

            # 2. Visualizar polígonos
            self.publish_vertices_marker(verts_12, frame_id=self.frame_id, ns="poligono_12", id=0)
            self.publish_vertices_marker(verts_34, frame_id=self.frame_id, ns="poligono_34", id=1)


            # 3. Calcular Centroides (Inicialmente Geométricos por seguridad)
            cent_12 = centroide_poligono(verts_12)
            cent_34 = centroide_poligono(verts_34)

            # rospy.loginfo(f"Centroides geométricos 1-2: {cent_12}")
            # rospy.loginfo(f"Centroides geométricos 3-4: {cent_34}")

            info_12_dict = None
            info_34_dict = None

            # 4. Calcular Elipses y SUSTITUIR centroides
            if INFER_ELLIPSE:
                # Intentar obtener centro de elipse para 3-4
                ellipse_center_34, info_34_dict = self.process_ellipse_for_polygon(verts_34, ns_prefix="poly_34_")
                if ellipse_center_34 is not None and SUSTITUIR_CENTROIDE_POR_ELIPSE:
                    cent_34[1:] = ellipse_center_34[1:] # <--- Aquí se sustituye el centroide por el de la elipse

                # Intentar obtener centro de elipse para 1-2
                ellipse_center_12, info_12_dict = self.process_ellipse_for_polygon(verts_12, ns_prefix="poly_12_")
                
                if ellipse_center_12 is not None and SUSTITUIR_CENTROIDE_POR_ELIPSE:
                    cent_12[1:] = ellipse_center_12[1:] # <--- Aquí se sustituye el centroide por el de la elipse

            # 5. Publicar los puntos definitivos (sean geométricos o de elipse)
            if cent_12 is not None: 
                self.publish_pointstamped(self.centroide12_pub, cent_12, frame_id=self.frame_id)
                # rospy.loginfo(f"Centroide 1-2: {cent_12}")
            if cent_34 is not None: 
                self.publish_pointstamped(self.centroide34_pub, cent_34, frame_id=self.frame_id)
                # rospy.loginfo(f"Centroide 3-4: {cent_34}")

            # 6. Calcular Antebrazo (Usará los nuevos centros de elipse si existen)
            if FOREARM_CALCULATION:
                self.calculate_forearm_general(cent_12, cent_34)
                
                # --- NUEVO: DIBUJAR VOLUMEN ---
                if info_12_dict is not None and info_34_dict is not None:
                    d1 = 0.10  # extensión más allá de elipse34 (metros)
                    d2 = 0.20  # extensión más allá de elipse12 (metros)
                    self.draw_forearm_volume(info_12_dict, info_34_dict, d1, d2)

            if ELBOW_AS_INTERSECTION_SPHERE_LINE and cent_12 is not None and cent_34 is not None:
                if self.rshoulder is not None:
                    # Convertir cent_12 y cent_34 a PointStamped
                    ps_12 = PointStamped()
                    ps_12.header.frame_id = self.frame_id
                    ps_12.header.stamp = rospy.Time.now()
                    ps_12.point.x = float(cent_12[0])
                    ps_12.point.y = float(cent_12[1])
                    ps_12.point.z = float(cent_12[2])

                    ps_34 = PointStamped()
                    ps_34.header.frame_id = self.frame_id
                    ps_34.header.stamp = rospy.Time.now()
                    ps_34.point.x = float(cent_34[0])
                    ps_34.point.y = float(cent_34[1])
                    ps_34.point.z = float(cent_34[2])

                    try:
                        cent_12_bl = self.tfBuffer.transform(ps_12, 'base_link', rospy.Duration(1.0))
                        cent_34_bl = self.tfBuffer.transform(ps_34, 'base_link', rospy.Duration(1.0))
                    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
                        rospy.logwarn(f"TF transform failed: {e}")
                        continue # Usar continue en lugar de return para no matar el nodo

                    # 1. Definir Vectores
                    p_wrist = np.array([cent_12_bl.point.x, cent_12_bl.point.y, cent_12_bl.point.z])
                    p_fingers = np.array([cent_34_bl.point.x, cent_34_bl.point.y, cent_34_bl.point.z])
                    v_fore_gripper = p_fingers - p_wrist # Dirección de la línea

                    # --- PUBLICAR RESTRICCIÓN PARA EL SKELETON TRACKER ---
                    if v_fore_gripper is not None:
                        # 1. Normalizar vector
                        v_norm = v_fore_gripper / (np.linalg.norm(v_fore_gripper) + 1e-9)
                        
                        # 2. Calcular Cuaternio: Alinear eje X ([1,0,0]) con v_norm
                        # Eje de rotación = X_axis cross v_norm
                        # Ángulo = acos(X_axis dot v_norm)
                        x_axis = np.array([1.0, 0.0, 0.0])
                        rotation_axis = np.cross(x_axis, v_norm)
                        
                        # Manejo de casos colineales (si v_norm es paralelo a X)
                        if np.linalg.norm(rotation_axis) < 1e-6:
                            # Si son paralelos, cuaternio identidad (o rotación 180 si opuestos)
                            q = [0.0, 0.0, 0.0, 1.0] 
                        else:
                            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                            angle = math.acos(np.dot(x_axis, v_norm))
                            q = tf.transformations.quaternion_about_axis(angle, rotation_axis)

                        # 3. Crear Mensaje
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = "base_link" # O el frame donde calculaste p_wrist
                        pose_msg.header.stamp = rospy.Time.now()
                        
                        # Posición = Muñeca
                        pose_msg.pose.position.x = p_wrist[0]
                        pose_msg.pose.position.y = p_wrist[1]
                        pose_msg.pose.position.z = p_wrist[2]
                        
                        # Orientación = Vector Director codificado
                        pose_msg.pose.orientation.x = q[0]
                        pose_msg.pose.orientation.y = q[1]
                        pose_msg.pose.orientation.z = q[2]
                        pose_msg.pose.orientation.w = q[3]
                        
                        self.forearm_line_pub.publish(pose_msg)

                    # 2. VISUALIZAR (Llamada a la nueva función)
                    self.publish_debug_geometry(self.rshoulder, p_wrist, v_fore_gripper)

                    candidates = intersection_sphere_line(self.rshoulder, self.l1, p_wrist, v_fore_gripper)

                    best_elbow = None
                    
                    if len(candidates) == 1:
                        best_elbow = candidates[0]
                    elif len(candidates) == 2:
                        # ... tu lógica de distancias ...
                        p1 = candidates[0]
                        p2 = candidates[1]
                        dist1 = np.linalg.norm(p1 - p_wrist)
                        dist2 = np.linalg.norm(p2 - p_wrist)
                        if dist1 < dist2: best_elbow = p1
                        else: best_elbow = p2
                    
                    # CHIVATO 3: Ver si tenemos ganador
                    # 4. FILTRADO Y PUBLICACIÓN
                    if best_elbow is not None:
                        # Asegurar que sea numpy array
                        current_elbow_np = np.array(best_elbow)

                        # --- PASO DE SEGURIDAD 1: ¿El nuevo dato es válido? ---
                        if np.isnan(current_elbow_np).any() or np.isinf(current_elbow_np).any():
                            rospy.logwarn("Cálculo matemático devolvió NaN o Inf. Saltando frame.")
                            continue # Saltamos esta iteración para no romper el filtro

                        # --- PASO DE SEGURIDAD 2: ¿La memoria está corrupta? ---
                        if self.prev_elbow is not None:
                            if np.isnan(self.prev_elbow).any():
                                rospy.logwarn("Filtro corrupto (NaN detectado en memoria). Reseteando.")
                                self.prev_elbow = None # Forzamos reinicio

                        # --- APLICAR FILTRO ---
                        if self.prev_elbow is None:
                            # Primera iteración o tras reset
                            smoothed_elbow = current_elbow_np
                        else:
                            # Fórmula EMA
                            smoothed_elbow = (self.elbow_alpha * current_elbow_np) + \
                                            ((1.0 - self.elbow_alpha) * self.prev_elbow)
                        
                        # Actualizar memoria
                        self.prev_elbow = smoothed_elbow

                        # CHIVATO FINAL (Para confirmar que ahora salen números)
                        # rospy.loginfo(f"Elbow suavizado: {smoothed_elbow}")

                        # Publicar el punto SUAVIZADO
                        self.publish_pointstamped(self.elbow_point_pub, smoothed_elbow, frame_id='base_link')
            else:
                rospy.logwarn_throttle(5, "Esperando posición del hombro...")

            self.rate.sleep()

    def process_ellipse_for_polygon(self, vertices, ns_prefix=""):
            """ 
            Calcula, dibuja y devuelve el centro de la elipse.
            Returns: np.array([x, y, z]) del centro o None si falla.
            """
            if not vertices: return None, None

            # Adelgazamiento
            vertices_2d = [v[1:] for v in vertices] # YZ
            x_cota = vertices[0][0]
            
            try:
                poly_shapely = Polygon(vertices_2d)
                # Si usas adelgazamiento, el centro será del polígono interior
                poly_inset = poly_shapely.buffer(-ESPESOR_ADELGAZAMIENTO)
                
                if poly_inset.is_empty: 
                    return None, None

                # Manejo de MultiPolygon si el adelgazamiento divide la figura
                if poly_inset.geom_type == 'Polygon':
                    coords = list(poly_inset.exterior.coords)[:-1]
                elif poly_inset.geom_type == 'MultiPolygon':
                    coords = list(max(poly_inset.geoms, key=lambda a: a.area).exterior.coords)[:-1]
                else:
                    return None, None

                vertices_inset_3d = [(x_cota, y, z) for y, z in coords]
                
                # Calcular Elipse
                info = self.calculate_ellipse_from_vertices(vertices_inset_3d, ns_prefix=ns_prefix)
                
                # --- VALIDACIÓN DE ÁNGULOS (Añadido 'is not None') ---
                if ns_prefix == "poly_34_" and info.get("angulo_eje_mayor_grados") is not None:
                    self.v_major_34 = info["v_major_3d"]

                    # angle = info["angulo_eje_mayor_grados"] % 180.0
                    # msg = AngleStamped()

                    # if self.stamp_gripper is not None:
                    #     msg.header.stamp = self.stamp_gripper  ### INTENTO DE SINCRONIZACIÓN ###
                    # else:
                    #     msg.header.stamp = rospy.Time.now()

                    # msg.angle = (90 - (angle + degrees(self.roll_arm_ee))) % 360
                    # self.angle_pub_34.publish(msg)

                elif ns_prefix == "poly_12_" and info.get("angulo_eje_mayor_grados") is not None:
                    angle = info["angulo_eje_mayor_grados"] % 180.0
                    # msg = AngleStamped()

                    # if self.stamp_gripper is not None:
                    #     msg.header.stamp = self.stamp_gripper  ### INTENTO DE SINCRONIZACIÓN ###
                    # else:
                    #     msg.header.stamp = rospy.Time.now()
                        
                    # msg.angle = (90 - (angle + degrees(self.roll_arm_ee))) % 360
                    # self.angle_pub_12.publish(msg)

                # --- CORRECCIÓN DEL ERROR ---
                center_data = info.get("centro_pixeles")
                
                # Comprobamos si la tupla es None O si el segundo elemento (la coordenada Y) es None
                if center_data is None or center_data[1] is None:
                    return None, None

                curr_center = np.array(center_data, dtype=float) # Aseguramos que sea float

                # --- FILTRADO SUAVIZADO CORREGIDO ---
                
                # 1. Seleccionar la memoria adecuada según el prefijo
                prev_center = None
                if ns_prefix == "poly_12_":
                    prev_center = self.prev_center_12
                elif ns_prefix == "poly_34_":
                    prev_center = self.prev_center_34
                
                # 2. Aplicar filtro
                if prev_center is None:
                    smooth_center = curr_center
                else:
                    # Filtro: Nuevo = Alpha * Actual + (1-Alpha) * Anterior
                    smooth_center = self.alpha * curr_center + (1 - self.alpha) * prev_center
            
                # 3. Guardar en la memoria correspondiente
                if ns_prefix == "poly_12_":
                    self.prev_center_12 = smooth_center
                elif ns_prefix == "poly_34_":
                    self.prev_center_34 = smooth_center
                
                return smooth_center, info

            except Exception as e:
                rospy.logerr(f"Error procesando elipse {ns_prefix}: {e}")
                return None, None

    def publish_debug_geometry(self, shoulder, wrist, line_dir):
        """
        Publica en RViz la esfera del hombro y la línea infinita del antebrazo para depuración.
        Args:
            shoulder (np.array): [x, y, z] posición del hombro (centro esfera).
            wrist (np.array): [x, y, z] posición de la muñeca (punto en la línea).
            line_dir (np.array): Vector director de la línea.
        """
        if shoulder is None or wrist is None:
            return

        # 1. VISUALIZAR ESFERA (Radio L1)
        marker_sphere = Marker()
        marker_sphere.header.frame_id = "base_link"
        marker_sphere.header.stamp = rospy.Time.now()
        marker_sphere.ns = "debug_geometry_sphere"
        marker_sphere.id = 0
        marker_sphere.type = Marker.SPHERE
        marker_sphere.action = Marker.ADD
        
        marker_sphere.pose.position.x = shoulder[0]
        marker_sphere.pose.position.y = shoulder[1]
        marker_sphere.pose.position.z = shoulder[2]
        marker_sphere.pose.orientation.w = 1.0
        
        # Scale en esfera es diámetro
        diameter = self.l1 * 2.0
        marker_sphere.scale.x = diameter
        marker_sphere.scale.y = diameter
        marker_sphere.scale.z = diameter
        
        marker_sphere.color.r = 1.0; marker_sphere.color.g = 1.0; marker_sphere.color.b = 0.0; marker_sphere.color.a = 0.3
        
        self.debug_sphere_pub.publish(marker_sphere)

        # 2. VISUALIZAR LÍNEA INFINITA
        marker_line = Marker()
        marker_line.header.frame_id = "base_link"
        marker_line.header.stamp = rospy.Time.now()
        marker_line.ns = "debug_geometry_line"
        marker_line.id = 1
        marker_line.type = Marker.LINE_STRIP
        marker_line.action = Marker.ADD
        marker_line.scale.x = 0.005 # Grosor
        marker_line.color.r = 1.0; marker_line.color.g = 0.0; marker_line.color.b = 1.0; marker_line.color.a = 1.0

        # Crear segmento largo para simular infinito visualmente
        v_norm = line_dir / (np.linalg.norm(line_dir) + 1e-9)
        dist_visual = 1.0 
        p_start = wrist - (v_norm * dist_visual)
        p_end   = wrist + (v_norm * dist_visual)

        p1 = Point(x=p_start[0], y=p_start[1], z=p_start[2])
        p2 = Point(x=p_end[0],   y=p_end[1],   z=p_end[2])
        
        marker_line.points = [p1, p2]
        
        self.debug_line_pub.publish(marker_line)

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass