import numpy as np
from scipy.spatial.transform import Rotation as R

# Funciones geometricas útiles para el cálculo de IK Model


def rotate_vector_local(vector, axis_local, angle):
    """
    Rota un vector alrededor de un eje local mediante cuaterniones.
    
    Args:
    - vector: np.array de forma (3,) con el vector a rotar.
    - axis_local: np.array de forma (3,) con el eje local alrededor del cual rotar.
    - angle: Ángulo en grados.

    Returns:
    - rotated_vector: np.array con el vector rotado.
    """
    if np.isnan(angle):
        raise ValueError("El ángulo no puede ser NaN.")
    
    axis_local = axis_local / np.linalg.norm(axis_local) # Normalizar vector. Es imprescindible

    # Se crea el cuaternion de rotación cómo:
    #   vect_r = theta * vect_unitario_director_eje_rot
    #   theta: ángulo (magnitud) a rotar
    quat = R.from_rotvec(np.radians(angle) * axis_local) 

    rotated_vector = quat.apply(vector) # Se aplica la rotación al vector

    return rotated_vector

def normalize_vector(vector):
    """
    Normaliza un vector.

    :param vector: np.array, vector a normalizar.
    :return: np.array, vector normalizado.
    """
    norm = np.linalg.norm(vector)  # Calcula la norma del vector
    if norm == 0:
        raise ValueError("No se puede normalizar un vector de magnitud cero.")
    return vector / norm

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''

    return not any(np.all(kp == 0.0) for kp in keypoints)

def calculate_angle_2_vect(v1, v2):
    """
    Calcula el ángulo entre dos vectores en grados.
    
    Parámetros:
    - v1: numpy array o lista, primer vector.
    - v2: numpy array o lista, segundo vector.
    
    Retorna:
    - El ángulo (sin signo) entre los vectores en grados.
    """
    # Convertir a numpy arrays si no lo son
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    # Calcular el producto punto y las normas
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Manejo de caso especial: vectores nulos
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Uno o ambos vectores son nulos y no tienen un ángulo definido.")
    
    # Calcular el coseno del ángulo
    cos_theta = dot_product / (norm_v1 * norm_v2)
    
    # Asegurarse de que el valor esté en el rango [-1, 1] para evitar errores numéricos
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calcular el ángulo en radianes
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_signed_angle_3d(v1, v2, normal):
    """
    Calcula el ángulo con signo entre dos vectores en 3D.
    
    Parámetros:
    - v1: numpy array o lista, primer vector.
    - v2: numpy array o lista, segundo vector.
    - normal: vector normal al plano definido por v1 y v2.
    
    Retorna:
    - El ángulo entre los vectores en grados, con signo, en el rango [-180, 180].
    """
    # Convertir a numpy arrays si no lo son
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    normal = np.array(normal, dtype=float)
    
    # Calcular el producto punto y las normas
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # Manejo de caso especial: vectores nulos
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Uno o ambos vectores son nulos y no tienen un ángulo definido.")
    
    # Calcular el coseno del ángulo
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Evitar errores numéricos
    
    # Calcular el ángulo en radianes
    angle_rad = np.arccos(cos_theta)
    
    # Calcular el producto cruzado para determinar el signo
    cross_product = np.cross(v1, v2)
    sign = np.sign(np.dot(cross_product, normal))  # Signo basado en el normal
    
    # Aplicar el signo al ángulo
    signed_angle_rad = sign * angle_rad
    
    # Convertir a grados
    angle_deg = np.degrees(signed_angle_rad)
    
    return angle_deg

def calculate_angle_3_points(A, B, C):
    '''
    This method recives 3 points A, B, C and calculate the angle between vect(BA) and vect(BC)
    B - Common point

    if any of the 3 kp is not avaliable, returns Nan.
    
    '''
    if are_kp_valid(A, B, C): 
        # Calculate vectors
        BA = A - B
        BC = C - B

        dot_product = np.dot(BA, BC)

        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)

        cos_theta = dot_product / (magnitude_BA*magnitude_BC)

        # Restringir el valor de theta para evitar errores
        cos_theta = np.clip(cos_theta, -1.0, 1.0) 

        theta_rad = np.arccos(cos_theta)
        theta_deg = np.degrees(theta_rad)

    else: theta_deg = np.nan

    return theta_deg

def project_Kp_to_plane(point, plane_point, plane_normal):
    """
    Projects a point in a plane

    Args:
        point (np.array): Coordenadas del punto a proyectar (x, y, z)
        plane_point (np.array): Coordenadas del punto del plano (x, y, z)
        plane_normal (np.array): Vector normal al plano (a, b, c)

    Returns:
        np.array: Coordenadas del punto proyectado en el plano (x', y', z')

    """
    # Convertir a arrays numpy
    point = np.array(point)
    plane_point = np.array(plane_point)

    if np.linalg.norm(plane_normal) == 0:
        raise ValueError("Plane normal vector is zero.")

    plane_normal = np.array(plane_normal)

    # Vector desde el punto del plano hasta el punto
    vector = point - plane_point

    # Proyección escalar
    projection_scalar = np.dot(vector, plane_normal) / np.dot(plane_normal, plane_normal)

    # Proyección del punto sobre el plano
    projected_point = point - projection_scalar * plane_normal

    return projected_point
