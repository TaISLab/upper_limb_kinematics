#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from human_articular_space.msg import HumanBody # Mensaje a publicar

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

class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """

        # Inicializar nodo ROS con nombre único
        rospy.init_node('IK_arm_calculator', anonymous=False)

        # Obtener el namespace del argumento pasado desde el archivo launch
        namespace = rospy.get_param('~namespace1', 'default_namespace')  # Valor por defecto

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # ROS publisher
        topic_name = f"/{namespace}/description"
        self.pub_human_body = rospy.Publisher(topic_name, HumanBody, queue_size=10) # Definir sin la / barra proporciona flexibilidad para cambiar el namespace del topic en un futuro

        # Set up subscriber for /skeleton_3D
        self.sub_skeleton_3D_keypoints = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.IK_calculator_callback) # Dont forget SELF.

        # FLAGS de estado
        self.last_valid_kp = True  # Estado anterior de los KP
        self.is_paused = False  # Estado actual de pausa
        self.was_paused = False  # Estado previo de pausa
        self.data_received = False  # Indica si se han recibido datos

        # Configuración del temporizador para detectar inactividad
        self.timeout_duration = 1.0  # Segundos antes de entrar en pausa
        self.timer = rospy.Timer(rospy.Duration(self.timeout_duration), self.timeout_callback)

    
    def timeout_callback(self, event):
        """
        Se ejecuta periódicamente para verificar si se están recibiendo datos.
        """
        if not self.data_received:
            if not self.is_paused:
                rospy.logwarn("No se están recibiendo datos en /skeleton_3D. Nodo en pausa.")
                self.is_paused = True
        else:
            if self.is_paused:
                rospy.loginfo("Datos de /skeleton_3D recibidos nuevamente. Nodo reanudado.")
                self.is_paused = False
            self.data_received = False  # Reset para la siguiente iteración
    
    def IK_calculator_callback(self, msg):
        """
        Callback function to calculate the DH angles of the right arm.
        - q1: Elevación frontal
        - q2: Elevación lateral
        - q3: Rotación interna/externa del hombro. 
                Interna-> Negativa 
                Externa -> Positiva
        - q4: Flexión del codo

        Parameters:
        - msg: Skeleton3D message

        Returns:
        - [q1, q2, q3, q4] del brazo derecho
        - Longitud del brazo superior (upperarm_length)
        - Longitud del antebrazo (forearm_length)
        """

        try:
            # Indicar que se han recibido datos
            self.data_received = True

            # Convert the received messages into a list of 3D keypoint arrays. Transform to ros msg to numpy vector
            keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])
            human_body = HumanBody() # msg que publicar

            # Guide of Kp
            #  5 left_shoulder
            #  6 right_shoulder
            #  7 left_elbow
            #  8 right_elbow
            #  9 left_wrist
            # 10 right_wrist
            # 11 left_hip
            # 12 right_hip
            # 17 neck. no funciona.

            # Asignacion de Kp
            kp_R_Shoulder = keypointsX[6]
            kp_R_Elbow = keypointsX[8]
            kp_R_Wrist = keypointsX[10]
            kp_L_Shoulder = keypointsX[5]

            human_body.header.stamp = rospy.Time.now()

            if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder, kp_R_Elbow, kp_R_Wrist):

                # Si antes eran inválidos, loguea el cambio
                if not self.last_valid_kp:
                    rospy.logdebug("Cálculo IK iniciado: keypoints detectados")
                self.last_valid_kp = True
                
                ################## Longitudes ##################
                upperarm_length = np.linalg.norm(kp_R_Elbow - kp_R_Shoulder)
                forearm_length = np.linalg.norm(kp_R_Wrist - kp_R_Elbow)

                human_body.right_arm.upperarm_length = upperarm_length
                human_body.right_arm.forearm_length = forearm_length

                ################## Vectores esenciales ##################
                # Vector = Final - Inicial

                vect_upperarm = normalize_vector(kp_R_Elbow - kp_R_Shoulder)
                vect_laterolateral = normalize_vector(kp_R_Shoulder - kp_L_Shoulder)

                ################## Calculo q1 ##################
                # Plano sagital
                    # vector normal: vect_laterolateral. Sentido de q1 positivo
                    # Punto del plano: right_shoulder

                vect_ref_q1 = ([0, 0, -1]) # SIMPLIFICACIÓN: El vector de ref es siempre perpendicular al plano del suelo

                # Calculo con signo
                human_body.right_arm.q1 = calculate_signed_angle_3d(
                    vect_ref_q1,
                    normalize_vector(project_Kp_to_plane(kp_R_Elbow, kp_R_Shoulder, vect_laterolateral) - kp_R_Shoulder), # Solo es necesario proyectar el codo, el RShoulder pertenece
                    vect_laterolateral
                )

                ################## Calculo q2 ##################
                # Coronal plane
                    # Metodo: Se establece el vector q2 de referencia y se rota en el eje local de aplicación de q1. Posteriormente se calcula el ángulo entre el vector de referencia rotado y el vector longitudinal
                    # Punto del plano: right_shoulder
                
                if (human_body.right_arm.q1 != np.nan):
                    vect_ref_q2 = ([0, 0, -1]) # SIMPLIFICACION: Vector Z negativo
                    vect_ref_q2_rot = rotate_vector_local(vect_ref_q2, vect_laterolateral, human_body.right_arm.q1) # CORREGIDO: Se3 rota sobre el eje local
                    vect_ref_q2_rot_norm = normalize_vector(vect_ref_q2_rot)
                    
                    normal_vector_q2_sign = np.cross(vect_upperarm, vect_laterolateral) # Para definir el signo de q2

                    human_body.right_arm.q2 = calculate_signed_angle_3d(
                        vect_ref_q2_rot_norm, # Vector de referencia q1
                        vect_upperarm, # No necesitan ser proyectados
                        normal_vector_q2_sign
                    )

                ################## Calculo q3 ##################
                if (human_body.right_arm.q1 != np.nan) and (human_body.right_arm.q2 != np.nan):
                
                    # Plano de proyección
                    #   vector normal: vector longitudinal del brazo
                    #   Pto de aplicación: codo
                    
                    vect_ref_q3 = normalize_vector(kp_L_Shoulder - kp_R_Shoulder) # Vector de referencia.
                    
                    vector_ref_q3_rot1 = rotate_vector_local(vect_ref_q3, vect_laterolateral, human_body.right_arm.q1) # No tiene interes porq se rota sobre si mismo
                    vector_ref_q3_rot2 = rotate_vector_local(vector_ref_q3_rot1, normal_vector_q2_sign, human_body.right_arm.q2)

                    human_body.right_arm.q3 = calculate_signed_angle_3d(
                        vector_ref_q3_rot2, # no hace falta proyectar, pertenece siempre
                        normalize_vector(project_Kp_to_plane(kp_R_Wrist, kp_R_Elbow, vect_upperarm) - kp_R_Elbow),
                        vect_upperarm
                    )

                ################## Calculo q4 ##################
                human_body.right_arm.q4 = calculate_angle_3_points(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist)

                ################## Publicar q ##################
                # Casos no estudiados
                human_body.left_arm.q1 = np.nan
                human_body.left_arm.q2 = np.nan
                human_body.left_arm.q3 = np.nan
                human_body.left_arm.q4 = np.nan
                human_body.left_arm.upperarm_length = np.nan
                human_body.left_arm.forearm_length = np.nan

                # Publish the human angles calculated
                self.pub_human_body.publish(human_body)
            else:
                # Solo imprime si el estado ha cambiado
                if self.last_valid_kp:
                    rospy.logdebug("Cálculo IK detenido: un KP esencial no se ha detectado")
                self.last_valid_kp = False

        except Exception as e:
            # Log any errors that occur during calculate
            rospy.logerr(f"Error calculating angles: {e}")




if __name__ == '__main__':
    try:
        # Create an instance of the ROSInterface and start listening for messages
        interface = ROSInterface()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down IK_arm_calculator node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
