import math
import numpy as np
from typing import Sequence, Tuple, List
import tf
import rospy
# Nota: Si usas cvxpy aquí o en polygon_utils, asegúrate de importarlo

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

def centroide_poligono(vertices):
    """
    Calcula el centroide (punto medio geométrico) de un polígono 3D.
    Args:
        vertices: lista de tuplas o arrays (x, y, z)
    Returns:
        np.array([x, y, z]) con el centroide
    """
    v = np.array(vertices)
    return np.mean(v, axis=0)

def incentro_poligono(vertices):
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

def intersection_of_lines(P1, v1, P2, v2, max_length=10.0):
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


def obtener_vertices_cuatro_lados(tfBuffer, dedoX, dedoY, frame_base="base_gripper"):
    """
    Implementación basada en tu código original para CUATRO_LADOS.
    Calcula un romboide/cometa basado en intersecciones de vectores.
    P1, P3: Posiciones de los links distales (link2).
    P2: Intersección de los vectores X de los links distales (hacia las puntas).
    P4: Intersección de los vectores -X de los links proximales (hacia atrás).
    """
    try:
        # 1. Obtener transformaciones (X=dedo1/4, Y=dedo2/3 en tu lógica)
        t_X1 = tfBuffer.lookup_transform(frame_base, f'{dedoX}_link1', rospy.Time(0), rospy.Duration(1.0))
        t_X2 = tfBuffer.lookup_transform(frame_base, f'{dedoX}_link2', rospy.Time(0), rospy.Duration(1.0))
        t_Y1 = tfBuffer.lookup_transform(frame_base, f'{dedoY}_link1', rospy.Time(0), rospy.Duration(1.0))
        t_Y2 = tfBuffer.lookup_transform(frame_base, f'{dedoY}_link2', rospy.Time(0), rospy.Duration(1.0))

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
        P2 = intersection_of_lines(p_X2, vector_x_X2, p_Y2, vector_x_Y2)

        # P4: Intersección proyecciones hacia atrás (Bases)
        # Nota: Tu código original usaba p_d42 (link2 pos) con vector_x_d41 (link1 vec) negativo
        # y p_d32 (link2 pos) con vector_x_d31 (link1 vec) negativo.
        # Reproduzco EXACTAMENTE esa combinación: Origen en Link2, dirección -VectorLink1.
        P4 = intersection_of_lines(p_X2, -vector_x_X1, p_Y2, -vector_x_Y1)

        if P2 is not None and P4 is not None:
            # Orden para polígono cerrado: P1 -> P2 -> P3 -> P4
            return [P1, P2, P3, P4]
        else:
            return None

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr(f"Error TF en cuatro_lados para {dedoX}-{dedoY}")
        return None

def obtener_vertices_pentagono(tfBuffer, dedoX, dedoY, frame_base="base_gripper"):
    """
    Implementación basada en tu código original para PENTAGONO.
    Cierra por arriba con una intersección, pero usa las bases abajo.
    """
    try:
        # 1. Transforms
        t_X1 = tfBuffer.lookup_transform(frame_base, f'{dedoX}_link1', rospy.Time(0), rospy.Duration(1.0))
        t_X2 = tfBuffer.lookup_transform(frame_base, f'{dedoX}_link2', rospy.Time(0), rospy.Duration(1.0))
        t_Y1 = tfBuffer.lookup_transform(frame_base, f'{dedoY}_link1', rospy.Time(0), rospy.Duration(1.0))
        t_Y2 = tfBuffer.lookup_transform(frame_base, f'{dedoY}_link2', rospy.Time(0), rospy.Duration(1.0))

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
        P5 = intersection_of_lines(p_tip_X, vector_x_tip_X, p_tip_Y, vector_x_tip_Y)

        if P5 is not None:
            # Orden del return en tu código original: p_d33, p_d32, p_d42, p_d43, P5
            # Mapping: TipY, BaseY, BaseX, TipX, Interseccion
            return [p_tip_Y, p_base_Y, p_base_X, p_tip_X, P5]
        else:
            return None

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr(f"Error TF en pentagono para {dedoX}-{dedoY}")
        return None

def obtener_vertices_hexagono(tfBuffer, dedoX, dedoY, frame_base="base_gripper"):
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
        t_X1 = tfBuffer.lookup_transform(frame_base, f"{dedoX}_link1", rospy.Time(0))
        t_X2 = tfBuffer.lookup_transform(frame_base, f"{dedoX}_link2", rospy.Time(0))
        t_Y2 = tfBuffer.lookup_transform(frame_base, f"{dedoY}_link2", rospy.Time(0))
        t_Y1 = tfBuffer.lookup_transform(frame_base, f"{dedoY}_link1", rospy.Time(0))

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
        offset_x = -0.00 # agarra por el lado de dentro

        x_flat = p_X1[0]
        p1 = np.array([x_flat + offset_x, p_X1[1], p_X1[2]])
        p2 = np.array([x_flat + offset_x, p_X2[1], p_X2[2]])
        p3 = np.array([x_flat + offset_x, p_X2_x[1], p_X2_x[2]])
        p4 = np.array([x_flat + offset_x, p_Y2_x[1], p_Y2_x[2]])
        p5 = np.array([x_flat + offset_x, p_Y2[1], p_Y2[2]])
        p6 = np.array([x_flat + offset_x, p_Y1[1], p_Y1[2]])

        return p1, p2, p3, p4, p5, p6
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr("Error al obtener las transformaciones.")
        return None, None, None, None, None, None
