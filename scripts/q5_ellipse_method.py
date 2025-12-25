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
from geometry_msgs.msg import Point

import math
from typing import Sequence, Tuple, List
from shapely.geometry import Polygon
from upper_limb_kinematics.msg import AngleStamped

def _signed_area_2d(V: Sequence[Sequence[float]]) -> float:
    A = 0.0
    n = len(V)
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        A += x1 * y2 - x2 * y1
    return 0.5 * A

def offset_polygon_2d(vertices: Sequence[Sequence[float]],
                      d: float,
                      parallel_tol: float = 1e-12) -> List[Tuple[float, float]]:
    """
    Offset (inset/outset) de un polígono 2D por intersección de aristas desplazadas.
    - d > 0: hacia el interior (adelgaza)
    - d < 0: hacia el exterior (expande)
    Devuelve una lista de (x, y) con la misma orientación que la entrada.
    """
    if len(vertices) < 3:
        raise ValueError("Se requieren al menos 3 vértices.")

    V = [(float(x), float(y)) for x, y in vertices]
    n = len(V)

    # Asegura CCW para que la normal izquierda sea interior
    area = _signed_area_2d(V)
    flipped = area < 0.0
    if flipped:
        V = list(reversed(V))

    # Construye líneas desplazadas: a x + b y + c = 0, con (a,b) normal unitaria
    lines = []
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-12:
            
            # rospy.logdebug("Arista degenerada (longitud ~ 0). Ignorando offset.")
            return None
            
            
        # Normal izquierda (interior en CCW)
        a, b = -dy / L, dx / L
        c0 = -(a * x1 + b * y1)
        c = c0 + d  # desplaza +d hacia el interior
        lines.append((a, b, c))

    # Intersección de líneas adyacentes (miter). Fallback si casi paralelas.
    out: List[Tuple[float, float]] = []
    for i in range(n):
        a1, b1, c1 = lines[(i - 1) % n]
        a2, b2, c2 = lines[i]
        det = a1 * b2 - a2 * b1

        if abs(det) < parallel_tol:
            # Casi paralelas: desplaza el vértice por la media de las normales
            xi, yi = V[i]
            nx, ny = a1 + a2, b1 + b2
            norm = math.hypot(nx, ny)
            if norm < 1e-12:
                # Caso límite: usa una de las normales
                nx, ny, norm = a2, b2, 1.0
            out.append((xi + d * nx / norm, yi + d * ny / norm))
        else:
            # Intersección exacta
            x = (b1 * (-c2) - b2 * (-c1)) / det
            y = (a2 * (-c1) - a1 * (-c2)) / det
            out.append((x, y))

    # Devuelve con la orientación original de entrada
    if flipped:
        out.reverse()
    return out


def convex_hull(points):
        """Envolvente convexa (monotone chain). Devuelve puntos en orden CCW, sin repetir el primero."""
        pts = sorted(set(points))
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        return hull  # CCW

def polygon_halfspaces_ccw(poly_ccw):
    """Devuelve (A, b) para semiespacios A_i^T x <= b_i que definen el interior del polígono CCW."""
    A_list, b_list = [], []
    n = len(poly_ccw)
    for i in range(n):
        p = np.array(poly_ccw[i], dtype=float)
        q = np.array(poly_ccw[(i+1) % n], dtype=float)
        d = q - p
        # normal izquierda (interior para CCW)
        n_left = np.array([-d[1], d[0]], dtype=float)
        # Queremos a^T x <= b con interior a la derecha de a (consistencia con la elipse)
        a = -n_left
        b = -float(n_left @ p)
        A_list.append(a)
        b_list.append(b)
    return np.array(A_list), np.array(b_list)

def john_ellipse_in_polygon_ORIGINAL(points_xy):
    """
    Para un conjunto de puntos, toma su envolvente convexa (CCW) y calcula
    c, G de la elipse E = { x = G y + c, ||y||_2 <= 1 } de mayor área.
    """
    hull = convex_hull(points_xy)
    if len(hull) < 3:
        raise ValueError("Se necesitan al menos 3 puntos no colineales.")

    A, b = polygon_halfspaces_ccw(hull)

    G = cp.Variable((2, 2), symmetric=True)
    c = cp.Variable(2)

    constraints = [G >> 0]
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        constraints += [
            cp.norm(G @ a_i) <= b_i - a_i @ c,
            b_i - a_i @ c >= 0
        ]

    prob = cp.Problem(cp.Maximize(cp.log_det(G)), constraints)

    # SOLUCIONADOR
    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimización no óptima: {prob.status}")

    Gv = G.value
    cv = c.value
    Gv = 0.5 * (Gv + Gv.T)  # simetriza
    return cv, Gv, hull

def john_ellipse_in_polygon(points_xy, a_max=0.025, b_min=0.015):
    """
    Calcula la elipse de mayor área inscrita en el polígono, con thresholds mínimos y máximos para ambos semiejes.
    Para dedo3, dedo4: a_max=0.07, b_min=0.02 (en metros) 
    ES UN RADIO DESDE EL CENTRO!!!.
    """
    hull = convex_hull(points_xy)
    if len(hull) < 3:
        raise ValueError("Se necesitan al menos 3 puntos no colineales.")

    A, b = polygon_halfspaces_ccw(hull)

    G = cp.Variable((2, 2), symmetric=True)
    c = cp.Variable(2)

    constraints = [G >> 0]
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        constraints += [
            cp.norm(G @ a_i) <= b_i - a_i @ c,
            b_i - a_i @ c >= 0
        ]

    # Semieje mayor (a)
    constraints += [cp.lambda_max(G) <= a_max]
    # Semieje menor (b)
    constraints += [cp.lambda_min(G) >= b_min]

    prob = cp.Problem(cp.Maximize(cp.log_det(G)), constraints)

    prob.solve(solver=cp.SCS, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None, None, hull, prob.status

    Gv = G.value
    cv = c.value
    Gv = 0.5 * (Gv + Gv.T)
    return cv, Gv, hull, prob.status

    

def ellipse_axes_from_G(G):
    """Autovalores/autovectores de G -> semiejes y dirección del eje mayor."""
    vals, vecs = np.linalg.eigh(G)
    order = np.argsort(vals)[::-1]
    ########## IMPORTANTE ##########
    # ORDEN CAMBIADO PARA QUE EL EJE MAYOR CORRESPONDA AL MAYOR AUTOVALOR
    s = vals[order]
    U = vecs[:, order]
    a, b = s[0], s[1]
    v_major = U[:, 0]
    v_minor = U[:, 1]
    return a, b, v_major, v_minor, U

def sample_ellipse(c, G, n=600):
    """Puntos sobre la elipse x = G [cos t; sin t] + c."""
    t = np.linspace(0, 2*np.pi, n, endpoint=True)
    C = np.vstack([np.cos(t), np.sin(t)])
    X = (G @ C).T + c
    return X

def longest_diagonal(poly_ccw):
    """
    Devuelve la diagonal más larga (par de índices no adyacentes) del polígono.
    Si el polígono tiene < 4 vértices, no hay diagonales (devuelve None).
    """
    m = len(poly_ccw)
    if m < 4:
        return None
    best = None
    bestd = -1.0
    for i in range(m):
        for j in range(i+1, m):
            # evitar aristas adyacentes y la arista (0, m-1)
            if j == i+1 or (i == 0 and j == m-1):
                continue
            d = np.linalg.norm(np.array(poly_ccw[i]) - np.array(poly_ccw[j]))
            if d > bestd:
                bestd = d
                best = (tuple(map(int, poly_ccw[i])), tuple(map(int, poly_ccw[j])))
    return best  # (p, q)

# -----------------------------
# 2) Función principal para obtener la elipse de mayor área
# -----------------------------


def draw_overlay(marker_pub, hull, c, G, x_cota=0.0):
    """
    Publica en RViz el polígono (hull), la elipse (calculada por c, G) y la diagonal mayor.
    Si marker_pub es None, no publica (solo dibuja si show=True).
    """
    # Polígono
    poly_marker = Marker()
    poly_marker.header.frame_id = "base_gripper"
    poly_marker.header.stamp = rospy.Time.now()
    poly_marker.ns = "poly_overlay"
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
    ellipse_marker.ns = "ellipse_overlay"
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
    diag_marker.header.frame_id = "base_gripper"
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

class EllipseMethodNode:
    def __init__(self):
        rospy.init_node('ellipse_method_node')
        rospy.loginfo("Ellipse Method Node started.")
        self.rate = rospy.Rate(5)  # 10 Hz

        # Publisher para los puntos en RViz
        self.marker_pub = rospy.Publisher('/ellipse_vertices_marker', Marker, queue_size=1)
        self.overlay_pub = rospy.Publisher('/ellipse_overlay_marker', Marker, queue_size=1)

        self.angle_pub = rospy.Publisher('/ellipse_angle_deg', AngleStamped, queue_size=1)
        

        self.tfBuffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.tf_stamp = None  # Tiempo de la última transformación obtenida

        # Subscriber
        #self.sub = rospy.Subscriber('input_topic', String, self.callback)

    def calculate_ellipse_from_vertices(self, vertices):
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
        info = draw_overlay(self.overlay_pub, hull, c, G, x_cota=x)
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

    def intersection_of_lines(self, P1, v1, P2, v2, max_length=10.0):
        """
        Traza dos líneas desde P1 y P2 según los vectores v1 y v2 (máximo 10 unidades),
        y calcula el punto de intersección entre ellas (en 2D).

        Args:
            P1 (tuple or np.array): Origen de la primera recta (x1, y1, z1).
            v1 (tuple or np.array): Vector director de la primera recta (vx1, vy1, vz1).
            P2 (tuple or np.array): Origen de la segunda recta (x2, y2, z2).
            v2 (tuple or np.array): Vector director de la segunda recta (vx2, vy2, vz2).
            max_length (float): Longitud máxima de cada línea.

        Returns:
            np.array: El punto de intersección (x, y, z), o None si no hay intersección.
        """
        # Calculo en el plano YZ
        v1 = np.array(v1)
        v2 = np.array(v2)

        P1_2d = np.array([P1[1], P1[2]])  # Solo las componentes Y y Z de P1
        P2_2d = np.array([P2[1], P2[2]])  # Solo las componentes Y y Z de P2
        v1_2d = np.array([v1[1], v1[2]])  # Solo las componentes Y y Z de v1
        v2_2d = np.array([v2[1], v2[2]])  # Solo las componentes Y y Z de v2


        # Normaliza los vectores y escala a max_length
        
        if np.linalg.norm(v1_2d) > 0:
            v1_2d = v1_2d / np.linalg.norm(v1_2d) * max_length
        if np.linalg.norm(v2_2d) > 0:
            v2_2d = v2_2d / np.linalg.norm(v2_2d) * max_length

        # Matriz A que representa el sistema de ecuaciones (solo x, y)
        A = np.array([v1_2d, -v2_2d]).T
        b = P2_2d - P1_2d

        try:
            t, s = np.linalg.solve(A, b)
            # Limita t y s al rango [0, 1] para que estén dentro del segmento de longitud máxima
            t = np.clip(t, 0, 1)
            s = np.clip(s, 0, 1)
            intersection_2d = P1_2d + t * v1_2d
            # Para z, se puede interpolar linealmente entre los puntos iniciales
            # z = P1[2] + t * (v1[2] if len(v1) > 2 else 0)
            x  = P1[0]  # Mantener la coordenada X original de P1
            intersection = np.array([x, intersection_2d[0], intersection_2d[1]])
            return intersection
        except np.linalg.LinAlgError:
            rospy.loginfo("No hay solución única (las rectas pueden ser paralelas o coincidentes).")
            return None

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


    def obtener_vertices_elipse(self):
        # Lógica para obtener los vértices de la elipse
        # Obten el valor de TF entre los frames: base_gripper y dedo4_link1 y base_gripper y dedo4_link2
        
        try:
            t_d41 = self.tfBuffer.lookup_transform('base_gripper', 'dedo4_link1', rospy.Time(0), rospy.Duration(1.0))
            t_d42 = self.tfBuffer.lookup_transform('base_gripper', 'dedo4_link2', rospy.Time(0), rospy.Duration(1.0))
            t_d31 = self.tfBuffer.lookup_transform('base_gripper', 'dedo3_link1', rospy.Time(0), rospy.Duration(1.0))
            t_d32 = self.tfBuffer.lookup_transform('base_gripper', 'dedo3_link2', rospy.Time(0), rospy.Duration(1.0))
            self.tf_stamp = t_d41.header.stamp  # Guardar el tiempo de la última transformación obtenida

            q_d41 = t_d41.transform.rotation
            q_d42 = t_d42.transform.rotation
            q_d31 = t_d31.transform.rotation
            q_d32 = t_d32.transform.rotation

            q_d41_tuple = [q_d41.x, q_d41.y, q_d41.z, q_d41.w]
            q_d42_tuple = [q_d42.x, q_d42.y, q_d42.z, q_d42.w]
            q_d31_tuple = [q_d31.x, q_d31.y, q_d31.z, q_d31.w]
            q_d32_tuple = [q_d32.x, q_d32.y, q_d32.z, q_d32.w]

            R_d41 = tf.transformations.quaternion_matrix(q_d41_tuple)[:3, :3]
            R_d42 = tf.transformations.quaternion_matrix(q_d42_tuple)[:3, :3]
            R_d31 = tf.transformations.quaternion_matrix(q_d31_tuple)[:3, :3]
            R_d32 = tf.transformations.quaternion_matrix(q_d32_tuple)[:3, :3]


            # Dedo 4
            # Convertir de vector3 a np.array

            p_d42 = np.array([t_d41.transform.translation.x, t_d41.transform.translation.y, t_d41.transform.translation.z])
            p_d43 = np.array([t_d42.transform.translation.x, t_d42.transform.translation.y, t_d42.transform.translation.z])
            vector_x_d43 = R_d42[:, 0]  # Eje x del dedo4_link2
            self.vector_x_d43 = vector_x_d43

            # Dedo 3
            p_d32 = np.array([t_d31.transform.translation.x, t_d31.transform.translation.y, t_d31.transform.translation.z])
            p_d33 = np.array([t_d32.transform.translation.x, t_d32.transform.translation.y, t_d32.transform.translation.z])
            vector_x_d33 = R_d32[:, 0]  # Eje x del dedo3_link2
            self.vector_x_d33 = vector_x_d33

            # Aplanar en X local de base_gripper. ¿Por qué? En el modelo los puntos no están bien alineados
            p_d33[0] = p_d43[0]
            p_d32[0] = p_d43[0]
            p_d42[0] = p_d43[0]
            # Calcular P5:
            P5 = self.intersection_of_lines(p_d43, vector_x_d43, p_d33, vector_x_d33)

            
            # rospy.loginfo(f"Vector x dedo4_link2: {vector_x_d43}, Vector x dedo3_link2: {vector_x_d33}")

            # Publicar vectores en RViz
            self.publish_vectors_marker([p_d43, p_d33], [vector_x_d43, vector_x_d33], ns="finger_vectors", color=(0.2, 0.2, 1.0), frame_id="base_gripper")

            if P5 is not None:
                return p_d33, p_d32, p_d42, p_d43, P5
            else:
                return p_d42, p_d43, p_d32, p_d33, None
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Error al obtener las transformaciones.")
            # Retornar cinco None si ocurre excepción
            return None, None, None, None, None


    def publish_vertices_marker(self, vertices, ns="ellipse_vertices", id=0):
        """
        Publica los vértices como un Marker tipo SPHERE_LIST en RViz y los imprime por pantalla.
        Espera una lista de 5 vértices (P1, P2, P3, P4, P5), cada uno como tupla (x, y, z).
        """
        if vertices is None or len(vertices) != 5:
            rospy.logwarn("publish_vertices_marker: Se requieren exactamente 5 vértices (P1, P2, P3, P4, P5).")
            return

        marker = Marker()
        marker.header.frame_id = "base_gripper"
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = id
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # diámetro de las esferas
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.points = []
        for i, v in enumerate(vertices):
            if v is None or len(v) < 3:
                rospy.logwarn(f"publish_vertices_marker: El vértice {i+1} es inválido: {v}")
                continue
            p = Point()
            p.x = v[0]
            p.y = v[1]
            p.z = v[2]
            marker.points.append(p)
            # rospy.loginfo(f"Vértice P{i+1}: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
        self.marker_pub.publish(marker)


    def run(self):
        while not rospy.is_shutdown():
            t0 = rospy.Time.now()

            # rospy.loginfo("Calculando la elipse de mayor área inscrita...")
            P1, P2, P3, P4, P5 = self.obtener_vertices_elipse()

            vertices = [(P1[0], P1[1], P1[2]), 
                        (P2[0], P2[1], P2[2]), 
                        (P3[0], P3[1], P3[2]), 
                        (P4[0], P4[1], P4[2]), 
                        (P5[0], P5[1], P5[2])]

            if any(p is None for p in [P1, P2, P3, P4, P5]):
                rospy.logwarn("No se pudieron obtener todos los puntos necesarios para calcular la elipse.")
                self.rate.sleep()
                continue  # Nueva iteración

            self.publish_vertices_marker(vertices, ns="original_polygon", id=0)

            vertices_2d = [P1[1:], P2[1:], P3[1:], P4[1:], P5[1:]] # Plano YZ
            x_cota = P1[0]

            
            # Utilizamos Shapely's Polygon y buffer para realizar el offset (adelgazamiento) del polígono de forma robusta.
            poligono = Polygon(vertices_2d)
            espesor = 0.0075  # Espesor a adelgazar (en metros)
            poligono_adelgazado = poligono.buffer(-espesor)  # Polígono adelgazado (offset hacia el interior)
            vertices_inset_2d = tuple(poligono_adelgazado.exterior.coords)[:-1]  # Excluir el último punto que es igual al primero
            vertices_inset_3d = [(x_cota, y, z) for y, z in vertices_inset_2d]  # Reconstruir 3D

            t1 = rospy.Time.now()
            tiempo_offset = (t1 - t0).to_sec()

            if not vertices_inset_3d:
                rospy.logwarn("No se pudo obtener el polígono adelgazado.")
                self.rate.sleep()
                continue  # Nueva iteración

            # Optimización para encontrar la elipse de mayor área inscrita en el polígono adelgazado
            info_elipse = self.calculate_ellipse_from_vertices(vertices_inset_3d)
            
            t2 = rospy.Time.now()
            tiempo_elipse = (t2 - t1).to_sec()

            if info_elipse["semi_eje_mayor_a_px"] is None:
                # rospy.logwarn("No se pudo calcular una elipse válida en esta iteración.")
                self.rate.sleep()
                continue  # Nueva iteración

            centro = info_elipse["centro_pixeles"][1:]  # (y, z)
            x_cota = info_elipse["centro_pixeles"][0]

            
            angulo_deg = info_elipse["angulo_eje_mayor_grados"]
            angle_msg = AngleStamped()
            angle_msg.header.stamp = self.tf_stamp if self.tf_stamp is not None else rospy.Time.now()
            angle_msg.angle = angulo_deg
            self.angle_pub.publish(angle_msg)

            # Imprimir ángulo del eje mayor
            rospy.loginfo(f"Ángulo eje mayor: {info_elipse['angulo_eje_mayor_grados']:.2f} grados")

            # Haz un rospy.loginfo en una linea con los tiempos de cálculo
            rospy.loginfo(f"Tiempos: Offset = {tiempo_offset*1000:.2f} ms, Opt. elipse = {tiempo_elipse*1000:.2f} ms, Total ite = {(tiempo_offset + tiempo_elipse)*1000:.2f} ms")  
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass