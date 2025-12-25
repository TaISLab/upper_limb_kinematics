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
)
from upper_limb_kinematics.grasp_visualization_utils import (
    draw_overlay,
)


# --- CONFIGURACIÓN DE FORMAS ---
# Opciones disponibles: "CUATRO_LADOS", "PENTAGONO", "HEXAGONO"
SHAPE_DEDO12 = "CUATRO_LADOS"   # Configuración para el par Dedo 1 y 2
SHAPE_DEDO34 = "HEXAGONO"    # Configuración para el par Dedo 3 y 4

FOREARM_CALCULATION = True
INFER_ELLIPSE = True
ESPESOR_ADELGAZAMIENTO = 0.005 # Metros


# FRAME_ID = "base_link" # para exp
FRAME_ID = "base_gripper" # para pruebas locales


class EllipseMethodNode:
    def __init__(self):
        rospy.init_node('ellipse_method_node')
        rospy.loginfo("Ellipse Method Node started.")
        self.rate = rospy.Rate(30)  # 30 Hz

        # Publisher para los puntos en RViz
        self.marker_pub = rospy.Publisher('/ellipse_vertices_marker', Marker, queue_size=1)
        self.overlay_pub = rospy.Publisher('/ellipse_overlay_marker', Marker, queue_size=1)
        self.angle_pub = rospy.Publisher('/ellipse_angle_deg', AngleStamped, queue_size=1)

        # Centroides hexagono garra
        self.centroide12_pub = rospy.Publisher('/tactile/centroide_12', PointStamped, queue_size=1)
        self.centroide_trapecio_12_pub = rospy.Publisher('/tactile/centroide_trapecio_12', PointStamped, queue_size=1)
        self.centroide34_pub = rospy.Publisher('/tactile/centroide_34', PointStamped, queue_size=1)
        self.grasping_point_pub = rospy.Publisher('/tactile/grasping_point', PointStamped, queue_size=1)
        self.new_elbow_pub = rospy.Publisher('/tactile/elbow', PointStamped, queue_size=1)
        self.new_wrist_pub = rospy.Publisher('/tactile/wrist', PointStamped, queue_size=1)
        self.centro_normal_to_q2_pub = rospy.Publisher('/tactile/centro_normal_q2', PointStamped, queue_size=1)


        self.l2 = rospy.get_param('/exp_optitrack_25/l2', 0.3)  # Longitud del antebrazo
        self.grasp_offset = rospy.get_param('/exp_optitrack_25/grasp_offset', 0.1)  # Offset del punto de agarre

        rospy.loginfo(f"L2 (get from ros params): {self.l2} m")
        rospy.loginfo(f"Grasp offset (get from ros params): {self.grasp_offset} m")
        
        self.frame_id = FRAME_ID

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)

        # Subscriber
        #self.sub = rospy.Subscriber('input_topic', String, self.callback)
    
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

    def publish_vertices_marker(self, vertices, ns="ellipse_vertices", id=0):
        """
        Publica los vértices como un Marker tipo SPHERE_LIST en RViz.
        Versión flexible: Acepta cualquier lista de vértices (3, 4, 5, 6...).
        """
        # CAMBIO AQUÍ: Validar que haya al menos 3 vértices, no exactamente 6
        if vertices is None or len(vertices) < 3:
            rospy.logwarn(f"publish_vertices_marker: Se requieren al menos 3 vértices. Recibidos: {len(vertices) if vertices else 0}")
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id # Usa self.frame_id para consistencia
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
        msg = PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
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
        c, G, hull, status = john_ellipse_in_polygon(vertices)

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
        info = draw_overlay(self.overlay_pub, hull, c, G, x_cota=x, ns_prefix=ns_prefix)

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

    def get_vertices_by_shape(self, shape_type, dedoX, dedoY):
        """ Dispatcher actualizado """
        if shape_type == "CUATRO_LADOS":
            return obtener_vertices_cuatro_lados(self.tfBuffer, dedoX, dedoY, self.frame_id)
        elif shape_type == "PENTAGONO":
            return obtener_vertices_pentagono(self.tfBuffer, dedoX, dedoY, self.frame_id)
        elif shape_type == "HEXAGONO":
            v1, v2, v3, v4, v5, v6 = obtener_vertices_hexagono(self.tfBuffer, dedoX, dedoY, self.frame_id)
            if v1 is not None: return [v1, v2, v3, v4, v5, v6]
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
            # A. Obtener Polígonos según Configuración
            # ----------------------------------------
            verts_12 = self.get_vertices_by_shape(SHAPE_DEDO12, "dedo1", "dedo2")
            verts_34 = self.get_vertices_by_shape(SHAPE_DEDO34, "dedo3", "dedo4")

            if verts_12 is None or verts_34 is None:
                rospy.logwarn_throttle(2.0, "Esperando transforms válidas para dedos...")
                self.rate.sleep()
                continue

            # B. Visualización de Vértices
            # ----------------------------------------
            self.publish_vertices_marker(verts_34, ns="poly_34", id=0)
            self.publish_vertices_marker(verts_12, ns="poly_12", id=1)

            # C. Cálculo de Centroides
            # ----------------------------------------
            cent_12 = centroide_poligono(verts_12)
            cent_34 = centroide_poligono(verts_34)
            
            if cent_12 is not None:
                self.publish_pointstamped(self.centroide12_pub, cent_12, frame_id=self.frame_id)
            if cent_34 is not None:
                self.publish_pointstamped(self.centroide34_pub, cent_34, frame_id=self.frame_id)

            # D. Forearm Calculation (Generalizado)
            # ----------------------------------------
            if FOREARM_CALCULATION:
                # Nota: Si usas SHAPE_DEDO12="HEXAGONO", cent_12 es el centroide de 6 puntos.
                # Si en el código antiguo usabas "centroide_trapecio_12" (solo puntos base),
                # puedes crear lógica específica aquí, pero lo estándar es usar el centroide del polígono elegido.
                self.calculate_forearm_general(centroide_proximal=cent_12, centroide_distal=cent_34)

            # E. Cálculo de Elipses (Opcional por polígono)
            # ----------------------------------------
            if INFER_ELLIPSE:
                # Procesar Dedo 3-4
                self.process_ellipse_for_polygon(verts_34, ns_prefix="poly_34_")
                # Procesar Dedo 1-2
                self.process_ellipse_for_polygon(verts_12, ns_prefix="poly_12_")

            self.rate.sleep()

    def process_ellipse_for_polygon(self, vertices, ns_prefix=""):
        """ Helper para calcular, dibujar y publicar elipse de un conjunto de vértices """
        if not vertices: return

        # Adelgazamiento
        vertices_2d = [v[1:] for v in vertices] # YZ
        x_cota = vertices[0][0]
        
        try:
            poly_shapely = Polygon(vertices_2d)
            poly_inset = poly_shapely.buffer(-ESPESOR_ADELGAZAMIENTO)
            
            if poly_inset.is_empty: return

            # Extraer coords del inset
            if poly_inset.geom_type == 'Polygon':
                coords = list(poly_inset.exterior.coords)[:-1]
            elif poly_inset.geom_type == 'MultiPolygon':
                # Tomar el más grande si se divide
                coords = list(max(poly_inset.geoms, key=lambda a: a.area).exterior.coords)[:-1]
            else:
                return

            vertices_inset_3d = [(x_cota, y, z) for y, z in coords]
            
            # Calcular Elipse
            info = self.calculate_ellipse_from_vertices(vertices_inset_3d, ns_prefix=ns_prefix)
            
            # Publicar ángulo (solo ejemplo para uno de los grupos, o publicar ambos en topics distintos)
            if ns_prefix == "poly_34_" and info["angulo_eje_mayor_grados"] is not None:
                msg = AngleStamped()
                msg.header.stamp = rospy.Time.now()
                msg.angle = info["angulo_eje_mayor_grados"]
                self.angle_pub.publish(msg)

        except Exception as e:
            rospy.logerr(f"Error procesando elipse {ns_prefix}: {e}")

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass