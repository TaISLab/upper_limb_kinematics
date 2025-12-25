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


# OLD
INFER_ELLIPSE = False
FOREARM_CALCULATION = True
CUATRO_LADOS = False
PENTAGONO = False
HEXAGONO = True
CENTRO_NORMAL_TO_Q2 = True
ESPESOR_ADELGAZAMIENTO_HEXAGONO = 0.0  # en metros

# --- CONFIGURACIÓN DE FORMAS ---
# Opciones disponibles: "CUATRO_LADOS", "PENTAGONO", "HEXAGONO"
SHAPE_DEDO12 = "CUATRO_LADOS"   # Configuración para el par Dedo 1 y 2
SHAPE_DEDO34 = "HEXAGONO"    # Configuración para el par Dedo 3 y 4

FOREARM_CALCULATION = True
INFER_ELLIPSE = True
ESPESOR_ADELGAZAMIENTO = 0.005 # Metros


# FRAME_ID = "base_link" # para exp
FRAME_ID = "base_gripper" # para pruebas locales


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
        self.centroide12_pub = rospy.Publisher('/tactile/centroide_hexagono_12', PointStamped, queue_size=1)
        self.centroide_trapecio_12_pub = rospy.Publisher('/tactile/centroide_trapecio_12', PointStamped, queue_size=1)
        self.centroide34_pub = rospy.Publisher('/tactile/centroide_hexagono_34', PointStamped, queue_size=1)
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
        self.tf_stamp = None  # Tiempo de la última transformación obtenida

        # Subscriber
        #self.sub = rospy.Subscriber('input_topic', String, self.callback)

    def obtener_vertices_cuatro_lados(self, dedoX, dedoY, frame_base="base_gripper"):
        """
        Implementación basada en tu código original para CUATRO_LADOS.
        Calcula un romboide/cometa basado en intersecciones de vectores.
        P1, P3: Posiciones de los links distales (link2).
        P2: Intersección de los vectores X de los links distales (hacia las puntas).
        P4: Intersección de los vectores -X de los links proximales (hacia atrás).
        """
        try:
            # 1. Obtener transformaciones (X=dedo1/4, Y=dedo2/3 en tu lógica)
            t_X1 = self.tfBuffer.lookup_transform(frame_base, f'{dedoX}_link1', rospy.Time(0), rospy.Duration(1.0))
            t_X2 = self.tfBuffer.lookup_transform(frame_base, f'{dedoX}_link2', rospy.Time(0), rospy.Duration(1.0))
            t_Y1 = self.tfBuffer.lookup_transform(frame_base, f'{dedoY}_link1', rospy.Time(0), rospy.Duration(1.0))
            t_Y2 = self.tfBuffer.lookup_transform(frame_base, f'{dedoY}_link2', rospy.Time(0), rospy.Duration(1.0))
            
            self.tf_stamp = t_X1.header.stamp

            # 2. Obtener matrices de rotación
            def get_R(t):
                q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                return tf.transformations.quaternion_matrix(q)[:3, :3]

            R_X1 = get_R(t_X1)
            R_X2 = get_R(t_X2)
            R_Y1 = get_R(t_Y1)
            R_Y2 = get_R(t_Y2)

            # 3. Obtener vectores X locales y posiciones
            # Dedo X (equivalente a tu d4/d1)
            p_X1 = np.array([t_X1.transform.translation.x, t_X1.transform.translation.y, t_X1.transform.translation.z])
            vector_x_X1 = R_X1[:, 0]
            
            p_X2 = np.array([t_X2.transform.translation.x, t_X2.transform.translation.y, t_X2.transform.translation.z])
            vector_x_X2 = R_X2[:, 0]

            # Dedo Y (equivalente a tu d3/d2)
            p_Y1 = np.array([t_Y1.transform.translation.x, t_Y1.transform.translation.y, t_Y1.transform.translation.z])
            vector_x_Y1 = R_Y1[:, 0] # Nota: En tu código original usabas R_d31 (link1) para el vector negativo

            p_Y2 = np.array([t_Y2.transform.translation.x, t_Y2.transform.translation.y, t_Y2.transform.translation.z])
            vector_x_Y2 = R_Y2[:, 0]

            # 4. Aplanar en X local de base_gripper (Tu lógica original)
            x_ref = p_X2[0]
            p_Y2[0] = x_ref
            p_Y1[0] = x_ref # Asumimos que también quieres aplanar Y1 si se usara
            p_X1[0] = x_ref
            # p_X2 ya tiene x_ref

            # 5. Calcular Puntos e Intersecciones
            P1 = p_X2
            P3 = p_Y2

            # P2: Intersección proyecciones hacia adelante (Puntas)
            # Usa posiciones de link2 y vectores de link2
            P2 = self.intersection_of_lines(p_X2, vector_x_X2, p_Y2, vector_x_Y2)

            # P4: Intersección proyecciones hacia atrás (Bases)
            # Nota: Tu código original usaba p_d42 (link2 pos) con vector_x_d41 (link1 vec) negativo
            # y p_d32 (link2 pos) con vector_x_d31 (link1 vec) negativo.
            # Reproduzco EXACTAMENTE esa combinación: Origen en Link2, dirección -VectorLink1.
            P4 = self.intersection_of_lines(p_X2, -vector_x_X1, p_Y2, -vector_x_Y1)

            if P2 is not None and P4 is not None:
                # Orden para polígono cerrado: P1 -> P2 -> P3 -> P4
                return [P1, P2, P3, P4]
            else:
                return None

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"Error TF en cuatro_lados para {dedoX}-{dedoY}")
            return None

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
            return self.obtener_vertices_cuatro_lados(dedoX, dedoY, self.frame_id)
        elif shape_type == "PENTAGONO":
            return self.obtener_vertices_pentagono(dedoX, dedoY, self.frame_id)
        elif shape_type == "HEXAGONO":
            v1, v2, v3, v4, v5, v6 = self.obtener_vertices_hexagono(dedoX, dedoY, self.frame_id)
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

    def obtener_vertices_pentagono(self, dedoX, dedoY, frame_base="base_gripper"):
        """
        Implementación basada en tu código original para PENTAGONO.
        Cierra por arriba con una intersección, pero usa las bases abajo.
        """
        try:
            # 1. Transforms
            t_X1 = self.tfBuffer.lookup_transform(frame_base, f'{dedoX}_link1', rospy.Time(0), rospy.Duration(1.0))
            t_X2 = self.tfBuffer.lookup_transform(frame_base, f'{dedoX}_link2', rospy.Time(0), rospy.Duration(1.0))
            t_Y1 = self.tfBuffer.lookup_transform(frame_base, f'{dedoY}_link1', rospy.Time(0), rospy.Duration(1.0))
            t_Y2 = self.tfBuffer.lookup_transform(frame_base, f'{dedoY}_link2', rospy.Time(0), rospy.Duration(1.0))
            
            self.tf_stamp = t_X1.header.stamp

            def get_R(t):
                q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
                return tf.transformations.quaternion_matrix(q)[:3, :3]
            
            R_X2 = get_R(t_X2)
            R_Y2 = get_R(t_Y2)

            # 2. Posiciones y Vectores
            # En tu código original: 
            # p_d42 venía de t_d41 (Link1) -> Base X
            # p_d43 venía de t_d42 (Link2) -> Tip X (o base de tip)
            
            p_base_X = np.array([t_X1.transform.translation.x, t_X1.transform.translation.y, t_X1.transform.translation.z])
            p_tip_X  = np.array([t_X2.transform.translation.x, t_X2.transform.translation.y, t_X2.transform.translation.z])
            vector_x_tip_X = R_X2[:, 0]

            p_base_Y = np.array([t_Y1.transform.translation.x, t_Y1.transform.translation.y, t_Y1.transform.translation.z])
            p_tip_Y  = np.array([t_Y2.transform.translation.x, t_Y2.transform.translation.y, t_Y2.transform.translation.z])
            vector_x_tip_Y = R_Y2[:, 0]

            # 3. Aplanar en X (usando Tip X como referencia según tu código: p_d43)
            x_ref = p_tip_X[0]
            p_tip_Y[0]  = x_ref
            p_base_Y[0] = x_ref
            p_base_X[0] = x_ref
            # p_tip_X ya tiene x_ref

            # 4. Calcular Intersección Superior (P5)
            # Intersección desde los tips (Link2) proyectando sus vectores X
            P5 = self.intersection_of_lines(p_tip_X, vector_x_tip_X, p_tip_Y, vector_x_tip_Y)

            if P5 is not None:
                # Orden del return en tu código original: p_d33, p_d32, p_d42, p_d43, P5
                # Mapping: TipY, BaseY, BaseX, TipX, Interseccion
                return [p_tip_Y, p_base_Y, p_base_X, p_tip_X, P5]
            else:
                return None

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr(f"Error TF en pentagono para {dedoX}-{dedoY}")
            return None

    def obtener_vertices_hexagono(self, dedoX, dedoY, frame_base="base_gripper"):
        """
        Obtiene los 6 vértices del hexágono para los dedos especificados.
        Args:
            dedoX: nombre del primer dedo (ej: "dedo3")
            dedoY: nombre del segundo dedo (ej: "dedo4")
            frame_base: frame de referencia (ej: "base_gripper")
        Returns:
            p1, p2, p3, p4, p5, p6 (np.array de shape (3,))
        """
        try:
            t_X1 = self.tfBuffer.lookup_transform(frame_base, f"{dedoX}_link1", rospy.Time(0))
            t_X2 = self.tfBuffer.lookup_transform(frame_base, f"{dedoX}_link2", rospy.Time(0))
            t_Y2 = self.tfBuffer.lookup_transform(frame_base, f"{dedoY}_link2", rospy.Time(0))
            t_Y1 = self.tfBuffer.lookup_transform(frame_base, f"{dedoY}_link1", rospy.Time(0))
            self.tf_stamp = t_X1.header.stamp

            # Posiciones
            p_X1 = np.array([t_X1.transform.translation.x, t_X1.transform.translation.y, t_X1.transform.translation.z])
            p_X2 = np.array([t_X2.transform.translation.x, t_X2.transform.translation.y, t_X2.transform.translation.z])
            p_Y2 = np.array([t_Y2.transform.translation.x, t_Y2.transform.translation.y, t_Y2.transform.translation.z])
            p_Y1 = np.array([t_Y1.transform.translation.x, t_Y1.transform.translation.y, t_Y1.transform.translation.z])

            # Orientaciones
            q_X2 = t_X2.transform.rotation
            q_Y2 = t_Y2.transform.rotation

            q_X2_tuple = [q_X2.x, q_X2.y, q_X2.z, q_X2.w]
            q_Y2_tuple = [q_Y2.x, q_Y2.y, q_Y2.z, q_Y2.w]

            R_X2 = tf.transformations.quaternion_matrix(q_X2_tuple)[:3, :3]
            R_Y2 = tf.transformations.quaternion_matrix(q_Y2_tuple)[:3, :3]

            # Eje X local
            x_X2 = R_X2[:, 0]
            x_Y2 = R_Y2[:, 0]

            # Vértices desplazados
            p_X2_x = p_X2 + 0.045 * x_X2
            p_Y2_x = p_Y2 + 0.045 * x_Y2

            # Aplanar en X local del frame base (usar X de p_X1 para todos)
            x_flat = p_X1[0]
            p1 = np.array([x_flat, p_X1[1], p_X1[2]])
            p2 = np.array([x_flat, p_X2[1], p_X2[2]])
            p3 = np.array([x_flat, p_X2_x[1], p_X2_x[2]])
            p4 = np.array([x_flat, p_Y2_x[1], p_Y2_x[2]])
            p5 = np.array([x_flat, p_Y2[1], p_Y2[2]])
            p6 = np.array([x_flat, p_Y1[1], p_Y1[2]])

            return p1, p2, p3, p4, p5, p6
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Error al obtener las transformaciones.")
            return None, None, None, None, None, None

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
            punto_corte = self.intersection_of_lines(p_D1_x, y_D1, p_D2_x, y_D2)

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

    def centroide_poligono(self, vertices):
        """
        Calcula el centroide (punto medio geométrico) de un polígono 3D.
        Args:
            vertices: lista de tuplas o arrays (x, y, z)
        Returns:
            np.array([x, y, z]) con el centroide
        """
        v = np.array(vertices)
        return np.mean(v, axis=0)

    def incentro_poligono(self, vertices):
        """
        Calcula el incentro (punto equidistante a los lados) de un polígono 3D.
        Args:
            vertices: lista de tuplas o arrays (x, y, z)
        Returns:
            np.array([x, y, z]) con el incentro
        """
        v = np.array(vertices)
        n = len(v)
        if n < 3:
            return None  # No es un polígono válido

        # Calcular longitudes de los lados
        lados = []
        for i in range(n):
            p1 = v[i]
            p2 = v[(i + 1) % n]
            lados.append(np.linalg.norm(p2 - p1))
        lados = np.array(lados)

        # Calcular incentro como media ponderada por las longitudes de los lados
        incentro = np.zeros(3)
        perimeter = np.sum(lados)
        for i in range(n):
            incentro += v[i] * lados[i - 1]  # lado anterior
        incentro /= perimeter
        return incentro

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
            cent_12 = self.centroide_poligono(verts_12)
            cent_34 = self.centroide_poligono(verts_34)
            
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
            rospy.logdebug(f"Error procesando elipse {ns_prefix}: {e}")

if __name__ == '__main__':
    try:
        node = EllipseMethodNode()
        node.run()
    except rospy.ROSInterruptException:
        pass