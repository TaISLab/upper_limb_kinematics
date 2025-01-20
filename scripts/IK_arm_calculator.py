#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import roslib.packages
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

# Import the custom skeleton tracking class
# from skeleton_tracker_3d import SkeletonTracker3D  

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Custom message type for 3D skeleton data
from human_articular_space.msg import HumanBody, DHAnglesArm

def rotate_vector_quaternion(vector, q1, q2):

    if np.isnan(q1) or np.isnan(q2):
        raise ValueError("Angles q1 or q2 are NaN, cannot create quaternions.")

    # Crear quaternions para las rotaciones
    quat_y = R.from_euler('y', q1, degrees=True).as_quat()  # Rotación alrededor de Y
    quat_x = R.from_euler('x', q2, degrees=True).as_quat()  # Rotación alrededor de X

    # Combinar las rotaciones
    combined_quat = R.from_quat(quat_y) * R.from_quat(quat_x)

    # Rotar el vector
    rotated_vector = combined_quat.apply(vector)
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

# BORRAR: Función para calcular la matriz de rotación alrededor del eje Y (q1)
# def rotation_matrix_y(q1):
#     return np.array([
#         [np.cos(q1), 0, np.sin(q1)],
#         [0, 1, 0],
#         [-np.sin(q1), 0, np.cos(q1)]
#     ])

# BORRAR: Función para calcular la matriz de rotación alrededor del eje X (q2)
# def rotation_matrix_x(q2):
#     return np.array([
#         [1, 0, 0],
#         [0, np.cos(q2), -np.sin(q2)],
#         [0, np.sin(q2), np.cos(q2)]
#     ])

# BORRAR: Función para calcular la matriz de rotación alrededor del eje Z (q3)
# def rotation_matrix_z(q3):
#     return np.array([
#         [np.cos(q3), -np.sin(q3), 0],
#         [np.sin(q3),  np.cos(q3), 0],
#         [0, 0, 1]
#     ])

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''

    return not any(np.all(kp == 0.0) for kp in keypoints)

import numpy as np

def calculate_angle_2_vect(v1, v2):
    """
    Calcula el ángulo entre dos vectores en radianes.
    
    Parámetros:
    - v1: numpy array o lista, primer vector.
    - v2: numpy array o lista, segundo vector.
    
    Retorna:
    - El ángulo (sin signo) entre los vectores en radianes.
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

def normal_vector_from_points(A, B, C):
    """
    Calculates the normal vector to the plane defined by 3 points

    Args: 
        A (np.array): Coordenadas del punto A (x1, y1, z1)
        B (np.array): Coordenadas del punto B (x2, y2, z2)
        C (np.array): Coordenadas del punto C (x3, y3, z3)

    Returns:
        np.array: Vector normal al plano. Sentido según gira BA -> BC, es decir, BA gira hacia BC 
    """ 

    A, B, C = np.array(A), np.array(B), np.array(C)

    # Calcular los vectores BA, BC
    BA = A - B
    BC = C - B

    normal = np.cross(BA, BC)

    # Normalizar vector
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("No se puede normalizar un vector de magnitud cero.")
    
    normal_norm = normal / norm
    
    return normal_norm


class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """
        # Initialize the ROS node with a unique name
        rospy.init_node('IK_arm_calculator', anonymous=False)

        # Obtener el namespace del argumento pasado desde el archivo launch
        namespace = rospy.get_param('~namespace1', 'default_namespace')  # Valor por defecto

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # Set up a ROS publisher for Human Angles
        # self.pub_skeleton = rospy.Publisher('/topic', mensaje_importado, queue_size=10)
        topic_name = f"/{namespace}/description"
        self.pub_human_body = rospy.Publisher(topic_name, HumanBody, queue_size=10) # Definir sin la / barra proporciona flexibilidad para cambiar el namespace del topic en un futuro

        # Set up subscriber for /skeleton_3D
        self.sub_skeleton_3D_keypoints = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.calculate_angles_callback) # Dont forget SELF.

    def calculate_angles_callback(self, msg):
        """
        Callback function to calculate human articular angles.

        Parameters:
        - msg: Skeleton3D message
        """

        try:

            # Convert the received messages into a list of 3D keypoint arrays.
            # Transform to ros msg to numpy vector
            keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

            human_body = HumanBody() # msg que publicar

            # Guide of Kp
            #  0 nose
            #  1 left_eye
            #  2 right_eye
            #  3 left_ear
            #  4 right_ear
            #  5 left_shoulder
            #  6 right_shoulder
            #  7 left_elbow
            #  8 right_elbow
            #  9 left_wrist
            # 10 right_wrist
            # 11 left_hip
            # 12 right_hip
            # 13 left_knee
            # 14 right_knee
            # 15 left_ankle
            # 16 right_ankle
            # 17 neck

            # Asignacion de Kp
            kp_R_Shoulder = keypointsX[6]
            kp_R_Elbow = keypointsX[8]
            kp_R_Wrist = keypointsX[10]

            kp_L_Shoulder = keypointsX[5]
            kp_L_Elbow = keypointsX[7]
            kp_L_Wrist = keypointsX[9]
            
            kp_L_Hip = keypointsX[11]
            kp_R_Hip = keypointsX[12]

            human_body.header.stamp = rospy.Time.now()

            if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder, kp_R_Hip, kp_L_Hip, kp_R_Elbow, kp_L_Elbow, kp_R_Wrist):
                
                # Medida de longitudes
                rospy.loginfo(f"L_Hombros: {np.linalg.norm(kp_R_Shoulder - kp_L_Shoulder)}")
                rospy.loginfo(f"L_R_upperarm: {np.linalg.norm(kp_R_Elbow - kp_R_Shoulder)}")
                rospy.loginfo(f"L_R_forearm: {np.linalg.norm(kp_R_Wrist - kp_R_Elbow)}")
                rospy.loginfo(f"L_hips: {np.linalg.norm(kp_R_Hip - kp_L_Hip)}")

                # Vectores comunes
                half_hip = kp_L_Hip + (kp_R_Hip - kp_L_Hip) / 2 # Punto de aplicacion
                shoulder_half = kp_L_Shoulder + (kp_R_Shoulder - kp_L_Shoulder) / 2

                ################## Calculo q1 ############################
                # sagittal plane
                    # vector normal: vect_laterolateral
                    # Punto del plano: half_hip

                vect_laterolateral = normalize_vector(kp_R_Hip - kp_L_Hip) # vector normal del plano, normalizado
                

                ## Vector de referencia
                # vect_ref_q1 = half_hip - shoulder_half # Vector de referencia q1. Se mide la diferencia entre este vector y el vector longitudinal del brazo
                # vect_ref_q1_norm = normalize_vector(vect_ref_q1)

                # SIMPLIFICACIÓN: El vector de ref es siempre perpendicular al plano del suelo
                vect_ref_q1_norm = ([0, 0, -1])
                
                ## Calculo sin signo
                # human_body.right_arm.q1 = calculate_angle_2_vect(
                #     vect_ref_q1_norm, # Vector de referencia q1
                #     project_Kp_to_plane(kp_R_Elbow, half_hip, vect_laterolateral) - project_Kp_to_plane(kp_R_Shoulder, half_hip, vect_laterolateral) # Rev2. Se necesitan proyectar ambos puntos
                # )

                ## Calculo con signo
                human_body.right_arm.q1 = calculate_signed_angle_3d(
                    vect_ref_q1_norm, # Vector de referencia q1
                    project_Kp_to_plane(kp_R_Elbow, half_hip, vect_laterolateral) - project_Kp_to_plane(kp_R_Shoulder, half_hip, vect_laterolateral), # Rev2. Se necesitan proyectar ambos puntos
                    vect_laterolateral
                )

                ################### Calculo q2 ###################################

                # Coronal plane
                    # vector normal: vect_dorsoventral
                    # Punto del plano: half_hip
                vect_dorsoventral = normal_vector_from_points(kp_L_Shoulder, half_hip, kp_R_Shoulder) # MUY IMPORTANTE EL ORDEN
                # rospy.loginfo(f"vect_dorsoventral: {vect_dorsoventral}")

                ## Vector de referencia
                # vect_ref_q2 = half_hip - shoulder_half
                # vect_ref_q2_norm = normalize_vector(vect_ref_q2)

                ## SIMPLIFICACION
                vect_ref_q2_norm = ([0, 0, -1])

                # Calculo sin signo
                human_body.right_arm.q2 = calculate_angle_2_vect(
                    vect_ref_q2_norm,
                    project_Kp_to_plane(kp_R_Elbow, half_hip, vect_dorsoventral) - project_Kp_to_plane(kp_R_Shoulder, half_hip, vect_dorsoventral) # Vector longitudinal del brazo derecho
                )

                # Calculo con signo
                # human_body.right_arm.q2 = calculate_signed_angle_3d(
                #     vect_ref_q2_norm, # Vector de referencia q2
                #     project_Kp_to_plane(kp_R_Elbow, half_hip, vect_dorsoventral) - project_Kp_to_plane(kp_R_Shoulder, half_hip, vect_dorsoventral) # Vector longitudinal del brazo derecho
                #     vect_dorsoventral
                # )

                ################### Calculo q3 ######################################
                if (human_body.right_arm.q1 != np.nan) and (human_body.right_arm.q2 != np.nan):
                
                    # Plano de proyección
                    #   vector normal: vector longitudinal del brazo
                    #   Pto de aplicación: codo
                    
                    # vect_upper_arm = kp_R_Shoulder - kp_R_Elbow
                    vect_upper_arm = kp_R_Elbow - kp_R_Shoulder # Final - inicial
                    vect_ref_q3_norm = normalize_vector(kp_L_Shoulder - kp_R_Shoulder) # Vector de referencia. Otra opción. Considerar cadera
                    
                    ## DEBUG
                    # vect_ref_q3 = ([0, -1, 0])
                    # human_body.right_arm.q1 = 45
                    # human_body.right_arm.q2 = 45
                    ## Rotacion con angulos EULER. Dependiente del orden
                    # vector_ref_q3_rotated_try1 = rotation_matrix_x(np.radians(human_body.right_arm.q2)) @ (rotation_matrix_y(np.radians(human_body.right_arm.q1)) @ vect_ref_q3_norm)
                    # vector_ref_q3_rotated_try2 = rotation_matrix_y(np.radians(human_body.right_arm.q1)) @ (rotation_matrix_x(np.radians(human_body.right_arm.q2)) @ vect_ref_q3_norm)
                    
                    # Rotacion con cuaterniones. es independiente del orden de la rotacion
                    vector_ref_q3_rotated_try3 = rotate_vector_quaternion(vect_ref_q3_norm, human_body.right_arm.q1, human_body.right_arm.q2)

                    ## DEBUG
                    # rospy.loginfo(f"v_ref_q3: {vect_ref_q3}")
                    # rospy.loginfo(f"v_ref_q3_try1: {vector_ref_q3_rotated_try1}")
                    # rospy.loginfo(f"v_ref_q3_try2: {vector_ref_q3_rotated_try2}")
                    # rospy.loginfo(f"v_ref_q3_try3: {vector_ref_q3_rotated_try3}")

                    human_body.right_arm.q3 = calculate_signed_angle_3d(
                        vector_ref_q3_rotated_try3, # no hace falta proyectar, pertenece siempre
                        project_Kp_to_plane(kp_R_Wrist, kp_R_Elbow, vect_upper_arm) - kp_R_Elbow,
                        vect_upper_arm
                    )

                    # human_body.right_arm.q3 = calculate_angle_2_vect(
                    #     vector_ref_q3_rotated_try3, # no hace falta proyectar, pertenece siempre
                    #     project_Kp_to_plane(kp_R_Wrist, kp_R_Elbow, vect_upper_arm) - kp_R_Elbow
                    # )

                ########## Calculo q4 ################
                human_body.right_arm.q4 = calculate_angle_3_points(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist)

                ########### Publicar q ##############

                # Casos no estudiados
                human_body.left_arm.q1 = np.nan
                human_body.left_arm.q2 = np.nan
                human_body.left_arm.q3 = np.nan
                human_body.left_arm.q4 = np.nan

                # Publish the human angles calculated
                self.pub_human_body.publish(human_body)

            else:
                rospy.loginfo("Algun Kp no se ha detectado")
                rospy.loginfo("*                         *")
                rospy.loginfo("*                         *")
                rospy.loginfo("***************************")

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
            rospy.loginfo("Shutting down human_articular_space_calculator node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
