import cvxpy as cp
import numpy as np

# Importamos las funciones geométricas necesarias del otro archivo de utilidades
from upper_limb_kinematics.grasp_geometry_utils import (
    convex_hull,
    polygon_halfspaces_ccw
)

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
    return cv, Gv, hull, prob.status

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


    """
    Calcula la elipse inscrita penalizando la desviación de una forma humana ideal.
    
    Args:
        points_xy: Lista de puntos 2D.
        lambda_reg: Peso de la regularización (0 = comportamiento original).
                    Aumentar si quieres forzar más la forma humana.
    """
    hull = convex_hull(points_xy)
    if len(hull) < 3:
        # Retorno seguro en lugar de lanzar excepción que pare el nodo
        return None, None, hull, "insufficient_points"

    A, b = polygon_halfspaces_ccw(hull)

    G = cp.Variable((2, 2), symmetric=True)
    c = cp.Variable(2)

    # --- DEFINICIÓN DEL MODELO BIOMIMÉTICO ---
    # Un antebrazo humano promedio: radio mayor ~4cm, radio menor ~2.5cm
    # G representa la inversa de los semiejes (1/r).
    # r_mayor_ideal = 0.04 m  --> G_val ~ 25.0
    # r_menor_ideal = 0.025 m --> G_val ~ 40.0
    # Creamos una matriz diagonal con estos valores objetivo.
    G_ideal = np.array([[20.0, 0.0], 
                        [0.0, 28.5]]) 

    constraints = [G >> 0]
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        constraints += [
            cp.norm(G @ a_i) <= b_i - a_i @ c,
            b_i - a_i @ c >= 0
        ]

    # --- FUNCIÓN OBJETIVO MODIFICADA ---
    # Maximizamos volumen (log_det) PERO restamos un coste por alejarse de G_ideal.
    # Usamos la norma de Frobenius para la diferencia.
    cost_function = cp.log_det(G) - lambda_reg * cp.norm(G - G_ideal, "fro")
    
    prob = cp.Problem(cp.Maximize(cost_function), constraints)

    # Usamos try/except para evitar crasheos del solver
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except cp.SolverError:
        return None, None, hull, "solver_error"

    if prob.status not in ("optimal", "optimal_inaccurate") or G.value is None:
        return None, None, hull, prob.status

    Gv = G.value
    cv = c.value
    # Simetrizar resultado numérico
    Gv = 0.5 * (Gv + Gv.T)
    
    return cv, Gv, hull, prob.status

def john_ellipse_in_polygon_constrained(points_xy, max_axis_len=0.032):
    """
    Calcula la elipse inscrita limitando su tamaño MÁXIMO.
    CORREGIDO: G representa los radios directamente.
    """
    # 1. ESCALADO (Metros -> Milímetros) para estabilidad numérica
    SCALE = 1000.0 
    
    # Puntos escalados
    points_scaled = [(p[0]*SCALE, p[1]*SCALE) for p in points_xy]
    max_axis_len_scaled = max_axis_len * SCALE # Ej: 0.032 -> 32.0
    
    hull = convex_hull(points_scaled)
    if len(hull) < 3:
        return None, None, hull, "insufficient_points"

    A, b = polygon_halfspaces_ccw(hull)

    G = cp.Variable((2, 2), symmetric=True)
    c = cp.Variable(2)

    # --- RESTRICCIONES ---
    constraints = [G >> 0] # G debe ser definida positiva
    
    # 1. Geométrica (Dentro del polígono)
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        constraints += [
            cp.norm(G @ a_i) <= b_i - a_i @ c,
            b_i - a_i @ c >= 0
        ]

    # 2. Dimensional (Tope de tamaño máximo) - CORREGIDO
    # Queremos que el radio (autovalores de G) sea MENOR que el límite.
    # En notación LMI: G <= max_len * I
    # Esto significa que (max_len * I) - G debe ser definida positiva.
    constraints += [G << max_axis_len_scaled * np.eye(2)]

    # --- SOLUCIÓN ---
    prob = cp.Problem(cp.Maximize(cp.log_det(G)), constraints)

    # Variable para saber qué método funcionó
    method_used = "constrained"

    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        # Si falla el solver, forzamos estado inválido para ir al fallback
        pass

    # --- PLAN B: FALLBACK (Si es imposible cumplir la restricción) ---
    # Nota: Si entra aquí, devolverá una elipse GRANDE (sin límite), 
    # pero al menos no crashea.
    if prob.status not in ("optimal", "optimal_inaccurate") or G.value is None:
        rospy.logwarn(f"Restricción de tamaño {max_axis_len}m imposible geométricamente. Usando fallback sin límites.")
        
        # Quitamos la restricción de tamaño (última en la lista)
        prob_fallback = cp.Problem(cp.Maximize(cp.log_det(G)), constraints[:-1]) 
        try:
            prob_fallback.solve(solver=cp.SCS, verbose=False)
            method_used = "fallback_unconstrained"
        except Exception:
            return None, None, [(p[0]/SCALE, p[1]/SCALE) for p in hull], "solver_fatal_error"
        
        # Si el fallback también falla
        if prob_fallback.status not in ("optimal", "optimal_inaccurate") or G.value is None:
             return None, None, [(p[0]/SCALE, p[1]/SCALE) for p in hull], "infeasible_all"

    # 3. DES-ESCALADO: Volvemos a Metros
    Gv_scaled = G.value
    cv_scaled = c.value
    
    # G representa longitudes directamente, así que dividimos por la escala
    Gv = Gv_scaled / SCALE
    cv = cv_scaled / SCALE
    
    Gv = 0.5 * (Gv + Gv.T)
    hull_meters = [(p[0]/SCALE, p[1]/SCALE) for p in hull]

    return cv, Gv, hull_meters, method_used


def john_ellipse_in_polygon_ratio_constrained(points_xy, max_aspect_ratio=2.0, a_max=None, b_min=None):
    """
    Calcula la elipse de mayor área con una restricción de RATIO (forma) y límites de tamaño.
    
    Args:
        max_aspect_ratio (float): Relación máxima a/b. 
        a_max (float, optional): Radio máximo (semieje mayor) en metros.
        b_min (float, optional): Radio mínimo (semieje menor) en metros. 
                                 Evita que la elipse colapse a una línea o punto.
    """
    # 1. ESCALADO (Metros -> Milímetros) para estabilidad numérica
    SCALE = 1000.0 
    points_scaled = [(p[0]*SCALE, p[1]*SCALE) for p in points_xy]
    
    hull = convex_hull(points_scaled)
    if len(hull) < 3:
        return None, None, hull, "insufficient_points"

    A, b = polygon_halfspaces_ccw(hull)

    G = cp.Variable((2, 2), symmetric=True)
    c = cp.Variable(2)

    # --- RESTRICCIONES ---
    constraints = [G >> 0] # Definida positiva
    
    # 1. Geométrica (Dentro del polígono)
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        constraints += [
            cp.norm(G @ a_i) <= b_i - a_i @ c,
            b_i - a_i @ c >= 0
        ]

    # 2. RESTRICCIÓN DE RATIO
    # lambda_max(G) <= ratio * lambda_min(G)
    constraints += [
        cp.lambda_max(G) <= max_aspect_ratio * cp.lambda_min(G)
    ]

    # 3. Restricción de Tamaño Máximo (a_max) - Para el rombo gigante
    if a_max is not None:
        max_len_mm = a_max * SCALE
        # G <= max_len * I  --> El radio más grande no puede superar a_max
        constraints += [G << max_len_mm * np.eye(2)]

    # 4. Restricción de Tamaño Mínimo (b_min) - [NUEVO]
    if b_min is not None:
        min_len_mm = b_min * SCALE
        # G >= min_len * I --> El radio más pequeño debe ser al menos b_min
        # Esto asegura que la elipse tenga "cuerpo"
        constraints += [G >> min_len_mm * np.eye(2)]

    # --- FUNCIÓN OBJETIVO ---
    prob = cp.Problem(cp.Maximize(cp.log_det(G)), constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        return None, None, [(p[0]/SCALE, p[1]/SCALE) for p in hull], "solver_error"

    # --- FALLBACK ---
    if prob.status not in ("optimal", "optimal_inaccurate") or G.value is None:
        # Nota: Si el polígono es físicamente más pequeño que b_min, esto fallará.
        # El status será 'infeasible'.
        return None, None, [(p[0]/SCALE, p[1]/SCALE) for p in hull], prob.status

    # 4. DES-ESCALADO
    Gv = G.value / SCALE
    cv = c.value / SCALE
    Gv = 0.5 * (Gv + Gv.T)
    hull_meters = [(p[0]/SCALE, p[1]/SCALE) for p in hull]

    return cv, Gv, hull_meters, "optimal"