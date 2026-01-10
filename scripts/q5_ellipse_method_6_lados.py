#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import String  # Cambia el tipo de mensaje según tus necesidades
import numpy as np
# import cv2
import matplotlib.pyplot as plt
import cvxpy as cp
from math import atan2, degrees
import tf  # <--- Añadir esta línea para importar tf
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped

import math
from typing import Sequence, Tuple, List
from shapely.geometry import Polygon
from upper_limb_kinematics.msg import AngleStamped
from gripper_4f.msg import encoders_data

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
)
from upper_limb_kinematics.grasp_polygon_utils import (
    john_ellipse_in_polygon,
    john_ellipse_in_polygon_ORIGINAL,
    john_ellipse_in_polygon_biomimetic
)
from upper_limb_kinematics.grasp_visualization_utils import (
    draw_overlay,
)


# --- CONFIGURACIÓN DE FORMAS ---
# Opciones disponibles: "CUATRO_LADOS", "PENTAGONO", "HEXAGONO"
SHAPE_DEDO12 = "HEXAGONO"   # Configuración para el par Dedo 1 y 2
SHAPE_DEDO34 = "HEXAGONO"    # Configuración para el par Dedo 3 y 4

GET_FROM_TF = False  # True -> vértices desde TF; False -> gripper_4f
FOREARM_CALCULATION = True
INFER_ELLIPSE = True
ESPESOR_ADELGAZAMIENTO = 0.0075 # Metros
PUBLISH_PS_IN_BASE_LINK = True
corregir_medicion_flag = True

# FRAME_ID = "base_link" # para exp
FRAME_ID = "base_gripper" # para pruebas locales

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
        self.rate = rospy.Rate(10)  # 10 Hz

        # Publisher para los puntos en RViz
        self.marker_pub = rospy.Publisher('/ellipse_vertices_marker', Marker, queue_size=20)
        self.overlay_pub = rospy.Publisher('/ellipse_overlay_marker', Marker, queue_size=20)
        self.angle_pub_12 = rospy.Publisher('/tactile/ellipse_12_angle_deg', AngleStamped, queue_size=1)
        self.angle_pub_34 = rospy.Publisher('/tactile/ellipse_34_angle_deg', AngleStamped, queue_size=1)

        # Centroides hexagono garra
        self.centroide12_pub = rospy.Publisher('/tactile/centroide_12', PointStamped, queue_size=20)
        self.centroide_trapecio_12_pub = rospy.Publisher('/tactile/centroide_trapecio_12', PointStamped, queue_size=20)
        self.centroide34_pub = rospy.Publisher('/tactile/centroide_34', PointStamped, queue_size=20)
        self.grasping_point_pub = rospy.Publisher('/tactile/grasping_point', PointStamped, queue_size=20)
        self.new_elbow_pub = rospy.Publisher('/tactile/elbow', PointStamped, queue_size=20)
        self.new_wrist_pub = rospy.Publisher('/tactile/wrist', PointStamped, queue_size=20)
        self.centro_normal_to_q2_pub = rospy.Publisher('/tactile/centro_normal_q2', PointStamped, queue_size=20)

        # subscripción a gripper_4f
        rospy.Subscriber("/gripper_4f/encoders_data", encoders_data, self.gripper_callback)
        

        self.l2 = rospy.get_param('/exp_optitrack_25/l2', 0.3)  # Longitud del antebrazo
        self.grasp_offset = rospy.get_param('/exp_optitrack_25/grasp_offset', 0.1)  # Offset del punto de agarre

        rospy.loginfo(f"L2 (get from ros params): {self.l2} m")
        rospy.loginfo(f"Grasp offset (get from ros params): {self.grasp_offset} m")
        
        self.frame_id = FRAME_ID

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Variables para filtro de suavizado (Exponential Moving Average)
        self.alpha = 0.2  # Factor de suavizado (0.0 = infinito, 1.0 = sin filtro)
        self.prev_center = None
        self.prev_G = None

        self.dedo1 = [0.0, 0.0, 0.0]
        self.dedo2 = [0.0, 0.0, 0.0]
        self.dedo3 = [0.0, 0.0, 0.0]
        self.dedo4 = [0.0, 0.0, 0.0]

    def gripper_callback(self, msg):

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
    

    def publish_vectors_marker(self, points, vectors, frame_id="base_gripper", ns="vectors", color=(0.2, 0.2, 1.0)):
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

    def publish_vertices_marker(self, vertices, frame_id="base_gripper", ns="ellipse_vertices", id=0):
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

    def publish_pointstamped(self, pub, point, frame_id="base_gripper"):
        """
        Publica un punto como geometry_msgs/PointStamped.
        Args:
            pub: rospy.Publisher configurado para PointStamped
            point: array-like (x, y, z)
            frame_id: frame de referencia
        """
        from geometry_msgs.msg import PointStamped
        target_frame = 'base_link'  # Frame al que se quiere transformar

        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        pub.publish(msg)

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

        # Llamamos a la función que calcula la elipse de mayor área en el polígono
        ########################################################################################################################
        # REVISAR EL MÉTODO IMPLEMENTADO
        # john_ellipse_in_polygon -> va bien
        # john_ellipse_in_polygon_ORIGINAL -> va mejor
        # john_ellipse_in_polygon_biomimetic -> va mal











        c, G, hull, status = john_ellipse_in_polygon_ORIGINAL(vertices)














        ##################################################################################################################

        if status not in ("optimal", "optimal_inaccurate") or c is None or G is None:
            rospy.logwarn("No se pudo calcular una elipse válida.")
            return {
                "centro_pixeles": (x, None, None),
                "semi_eje_mayor_a_px": None,
                "semi_eje_menor_b_px": None,
                "angulo_eje_mayor_grados": None,
                "poligono_vertices": vertices,
                "elipse_vertices": None,
            }

        # rospy.loginfo(f"Matriz G de la elipse:\n{G}")
        
        # Extraemos los ejes de la elipse
        a, b, v_major, v_minor, _ = ellipse_axes_from_G(G)

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

        TF_base_gripper_to_base_dedo12_x = -0.04
        TF_base_gripper_to_base_dedo12_y = 0.0
        TF_base_gripper_to_base_dedo12_z = 0.058

        TF_base_gripper_to_base_dedo34_x = 0.04
        TF_base_gripper_to_base_dedo34_y = 0.0
        TF_base_gripper_to_base_dedo34_z = 0.058

        l0 = 0.04 # Distancia entre origen dedo11 y dedo21 (o dedo31 y dedo41)
        l1 = 0.040  # Longitud de la primera falange
        l2 = 0.040  # Longitud de la segunda falange


        # TF base_gripper a base_falange (pto medio entre dedo10 y dedo20; o dedo30 y dedo40)
        if name == "dedo1_dedo2":
            P0 = np.array([TF_base_gripper_to_base_dedo12_x, 
                           TF_base_gripper_to_base_dedo12_y, 
                           TF_base_gripper_to_base_dedo12_z])  # Origen en base_gripper dedo10 y dedo20

        elif name == "dedo3_dedo4":
            P0 = np.array([TF_base_gripper_to_base_dedo34_x, 
                           TF_base_gripper_to_base_dedo34_y, 
                           TF_base_gripper_to_base_dedo34_z])  # Origen en base_gripper dedo30 y dedo40
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

        # 3. Cálculo de posiciones (Elbow y Wrist)
        # El codo está hacia atrás (lado dedos) o hacia adelante? 
        # Basado en tu código original: 
        # Elbow se calcula proyectando hacia el lado 3-4 (distal) extendido L2 metros? 
        # Ojo: En tu codigo original: v_grasp_to_elbow = v_grasp_to_34 * (l2 - offset).
        
        v_elbow_dir = v_to_proximal # Asumiendo que la dirección 34 es hacia wrist
        v_wrist_dir = v_to_distal

        new_elbow = grasping_point + v_elbow_dir * (self.l2 - self.grasp_offset)
        new_wrist = grasping_point + v_wrist_dir * self.grasp_offset

        self.publish_pointstamped(self.new_elbow_pub, new_elbow, frame_id=self.frame_id)
        self.publish_pointstamped(self.new_wrist_pub, new_wrist, frame_id=self.frame_id)
    
    def normal_point_to_q2(self, dedoX, dedoY, frame_base="base_gripper"):
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
                rospy.loginfo(f"Punto de corte entre normales: {punto_corte}")
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

    def run(self):
        while not rospy.is_shutdown():
            # 1. Obtener vértices

            
            if GET_FROM_TF:
                verts_12 = self.get_vertices_by_shape_from_TF(SHAPE_DEDO12, "dedo1", "dedo2")
                verts_34 = self.get_vertices_by_shape_from_TF(SHAPE_DEDO34, "dedo3", "dedo4")

            else:
                verts_12 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO12, self.dedo1, self.dedo2, name="dedo1_dedo2")
                verts_34 = self.get_vertices_by_shape_from_gripper(SHAPE_DEDO34, self.dedo4, self.dedo3, name="dedo3_dedo4")

            if not verts_12 or not verts_34:
                rospy.logwarn_throttle(2.0, "Esperando transforms...")
                self.rate.sleep()
                continue
            
            # Verificar que se obtienen los puntos necesarios según la forma seleccionada
            if SHAPE_DEDO12 == "PENTAGONO" and len(verts_12) != 5:
                rospy.logwarn_throttle(2.0, f"Se esperaban 5 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
                self.rate.sleep()
                continue
            if SHAPE_DEDO34 == "PENTAGONO" and len(verts_34) != 5:
                rospy.logwarn_throttle(2.0, f"Se esperaban 5 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
                self.rate.sleep()
                continue

            if SHAPE_DEDO12 == "CUATRO_LADOS" and len(verts_12) != 4:
                rospy.logwarn_throttle(2.0, f"Se esperaban 4 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
                self.rate.sleep()
                continue
            if SHAPE_DEDO34 == "CUATRO_LADOS" and len(verts_34) != 4:
                rospy.logwarn_throttle(2.0, f"Se esperaban 4 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
                self.rate.sleep()
                continue

            if SHAPE_DEDO12 == "HEXAGONO" and len(verts_12) != 6:
                rospy.logwarn_throttle(2.0, f"Se esperaban 6 vértices para Dedo 1-2, pero se obtuvieron {len(verts_12)}.")
                self.rate.sleep()
                continue
            if SHAPE_DEDO34 == "HEXAGONO" and len(verts_34) != 6:
                rospy.logwarn_throttle(2.0, f"Se esperaban 6 vértices para Dedo 3-4, pero se obtuvieron {len(verts_34)}.")
                self.rate.sleep()
                continue

            # 2. Visualizar polígonos
            self.publish_vertices_marker(verts_12, frame_id=self.frame_id, ns="poligono_12", id=0)
            self.publish_vertices_marker(verts_34, frame_id=self.frame_id, ns="poligono_34", id=1)


            # 3. Calcular Centroides (Inicialmente Geométricos por seguridad)
            cent_12 = centroide_poligono(verts_12)
            cent_34 = centroide_poligono(verts_34)

            # 4. Calcular Elipses y SUSTITUIR centroides
            if INFER_ELLIPSE:
                # Intentar obtener centro de elipse para 3-4
                ellipse_center_34 = self.process_ellipse_for_polygon(verts_34, ns_prefix="poly_34_")
                if ellipse_center_34 is not None:
                    cent_34 = ellipse_center_34 # <--- Aquí se sustituye el centroide por el de la elipse
                    dummy = 0  # Solo para evitar warning de variable no usada

                # Intentar obtener centro de elipse para 1-2
                ellipse_center_12 = self.process_ellipse_for_polygon(verts_12, ns_prefix="poly_12_")
                if ellipse_center_12 is not None:
                    cent_12 = ellipse_center_12 # <--- Aquí se sustituye el centroide por el de la elipse
                    dummy = 0  # Solo para evitar warning de variable no usada

            # 5. Publicar los puntos definitivos (sean geométricos o de elipse)
            if cent_12 is not None: 
                self.publish_pointstamped(self.centroide12_pub, cent_12, frame_id=self.frame_id)
            if cent_34 is not None: 
                self.publish_pointstamped(self.centroide34_pub, cent_34, frame_id=self.frame_id)

            # 6. Calcular Antebrazo (Usará los nuevos centros de elipse si existen)
            if FOREARM_CALCULATION:
                self.calculate_forearm_general(cent_12, cent_34)

            self.rate.sleep()

    def process_ellipse_for_polygon(self, vertices, ns_prefix=""):
        """ 
        Calcula, dibuja y devuelve el centro de la elipse.
        Returns: np.array([x, y, z]) del centro o None si falla.
        """
        if not vertices: return None

        # Adelgazamiento
        vertices_2d = [v[1:] for v in vertices] # YZ
        x_cota = vertices[0][0]
        
        try:
            poly_shapely = Polygon(vertices_2d)
            # Si usas adelgazamiento, el centro será del polígono interior
            poly_inset = poly_shapely.buffer(-ESPESOR_ADELGAZAMIENTO)
            
            if poly_inset.is_empty: 
                return None

            # Manejo de MultiPolygon si el adelgazamiento divide la figura
            if poly_inset.geom_type == 'Polygon':
                coords = list(poly_inset.exterior.coords)[:-1]
            elif poly_inset.geom_type == 'MultiPolygon':
                coords = list(max(poly_inset.geoms, key=lambda a: a.area).exterior.coords)[:-1]
            else:
                return None

            vertices_inset_3d = [(x_cota, y, z) for y, z in coords]
            
            # Calcular Elipse
            info = self.calculate_ellipse_from_vertices(vertices_inset_3d, ns_prefix=ns_prefix)
            
            
            if ns_prefix == "poly_34_" and info.get("angulo_eje_mayor_grados"):

                # Normalizar ángulo a [0, 180)
                angle = info["angulo_eje_mayor_grados"] % 180.0
                msg = AngleStamped()
                msg.header.stamp = rospy.Time.now()
                msg.angle = angle - 90 # Coincidente con q5

                self.angle_pub_34.publish(msg)

            elif ns_prefix == "poly_12_" and info.get("angulo_eje_mayor_grados"):
                # Normalizar ángulo a [0, 180)
                angle = info["angulo_eje_mayor_grados"] % 180.0
                msg = AngleStamped()
                msg.header.stamp = rospy.Time.now()
                msg.angle = angle - 90 # Coincidente con q5

                self.angle_pub_12.publish(msg)

            if info.get("centro_pixeles") is None:
                return None

            curr_center = np.array(info["centro_pixeles"]) # (x, y, z)

            # --- FILTRADO SUAVIZADO ---
            if self.prev_center is None:
                smooth_center = curr_center
            else:
                # Filtro: Nuevo = Alpha * Actual + (1-Alpha) * Anterior
                smooth_center = self.alpha * curr_center + (1 - self.alpha) * self.prev_center
        
            # Actualizar memoria
            self.prev_center = smooth_center
            
            return smooth_center

        except Exception as e:
            rospy.logerr(f"Error procesando elipse {ns_prefix}: {e}")
            return None

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass