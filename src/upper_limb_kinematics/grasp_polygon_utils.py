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

def john_ellipse_in_polygon_biomimetic(points_xy, lambda_reg=0.05):
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