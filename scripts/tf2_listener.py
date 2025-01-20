#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
import tf2_ros

from std_msgs.msg import Bool  # Mensaje estándar para notificaciones
from geometry_msgs.msg import Point

from skeleton_3d.msg import Skeleton3D  # Import custom message type
from human_articular_space.msg import KP_URDF         # Import custom message type
from human_articular_space.msg import KP_vision     # Import custom message type

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''
    return not any(np.all(kp == 0.0) for kp in keypoints)

def vector_mean_square_error(keypoint, translation):
    """
    Calcula el error cuadrático medio entre un keypoint y una traslación.
    
    Args:
    - keypoint: Lista o numpy array con los valores del keypoint.
    - translation: Lista o numpy array con los valores de traslación (target).
    
    Returns:
    - float: Error cuadrático medio.
    """
    # Convertir la traslación en un array numpy
    translation_array = np.array([translation.x, translation.y, translation.z])

    # Verificar que las entradas sean del mismo tamaño
    if np.size(keypoint) != np.size(translation_array):
        raise ValueError("keypoint y translation deben tener el mismo tamaño.")   

    resultado = np.mean((keypoint - translation_array) ** 2)

    return resultado

def error(value1, value2):
    """
    Calcula el error entre dos valores.
    
    Args:
        -Value1
        -Value2
    
    Returns:
    - float: Error 
    """
    resultado = abs(value1 - value2)

    return resultado

def print_error(keypoint, translation, label, decimals):
    """
    Muestra el error cuadrático medio entre un keypoint y una traslación, junto con sus coordenadas.

    Args:
    - keypoint: numpy array con los valores del keypoint (x, y, z).
    - translation: geometry_msgs/Transform.translation con los valores (x, y, z).
    - label: str, etiqueta para identificar qué punto se está mostrando.
    - decimals: int, número de decimales a mostrar.
    """
    # Convertir la traslación a un array numpy
    translation_array = np.array([translation.x, translation.y, translation.z])

    # Calcular el error cuadrático medio
    try:
        error = vector_mean_square_error(keypoint, translation)
        format_str = f"{{:.{decimals}f}}"  # Formato dinámico para decimales
        rospy.loginfo(f"[{label}]")
        rospy.loginfo(f"KP: x={format_str.format(keypoint[0])}, y={format_str.format(keypoint[1])}, z={format_str.format(keypoint[2])}")
        rospy.loginfo(f"TF: x={format_str.format(translation.x)}, y={format_str.format(translation.y)}, z={format_str.format(translation.z)}")
        rospy.loginfo(f"MSE: {format_str.format(error)}")

    except ValueError as e:
        rospy.logerr(f"[{label}] Error calculating MSE: {e}")

class ROSInterface:
    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the TF listener.
        """
        # Initialize the ROS node with a unique name
        rospy.init_node('tf2_listener', anonymous=False)

        # Obtener el namespace del argumento pasado desde el archivo launch
        self.namespace = rospy.get_param('~namespace1', 'default_namespace')  # Valor por defecto

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            self.path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # Set up a TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up subscriber for /skeleton_3D
        self.sub_skeleton_3D_keypoints = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.kp_callback)

        # Publicador para notificar cambios
        self.parameter_update_pub = rospy.Publisher("/parameter_update", Bool, queue_size=10)

        # Publicar TF URDF
        self.kp_URDF_pub = rospy.Publisher("kp_URDF", KP_URDF, queue_size=10)

        # Publicar TF vision
        self.kp_vision_pub = rospy.Publisher("kp_vision", KP_vision, queue_size=10)

    def lookup_transform(self, target_frame, source_frame):
        """
        Lookup a transform between target_frame and source_frame.

        Parameters:
        - target_frame: The frame to transform to.
        - source_frame: The frame to transform from.

        Returns:
        - transform: A geometry_msgs/TransformStamped object containing the transform.
        """
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0)) # Providing rospy.Time(0) will just get us the latest available transform
            
            if transform:
                rospy.logdebug(f"Transform: {transform}")
            else:
                rospy.logwarn("Transform not available.")

            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not find transform from {source_frame} to {target_frame}: {e}")
            return None

    def kp_callback(self, msg):
        """
        Callback function for Skeleton3D messages.
        """
        # decimals = rospy.get_param('/tf2_listener/decimals', 6)  # Obtener número de decimales
        decimals = 6
        error_threshold = 0.03  # Umbral para actualizar los parámetros en el servidor

        try:
            # Convert the received messages into a list of 3D keypoint arrays.
            keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

            kp_URDF_X = KP_URDF()
            kp_vision_X = KP_vision()

            # Header de los mensajes a publicar
            kp_URDF_X.header.stamp = rospy.Time.now()
            kp_vision_X.header.stamp = rospy.Time.now()

            # Extract specific keypoints
            kp_R_Shoulder = keypointsX[6]
            kp_R_Elbow = keypointsX[8]
            kp_R_Wrist = keypointsX[10]

            kp_L_Shoulder = keypointsX[5]
            kp_L_Elbow = keypointsX[7]

            kp_L_Hip = keypointsX[11]
            kp_R_Hip = keypointsX[12]

            # Validate keypoints
            if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder, kp_R_Hip, kp_L_Hip, kp_R_Elbow, kp_L_Elbow, kp_R_Wrist):
                
                # # URDF lenghts
                # URDF_upperarm_length = rospy.get_param('/human_body/right_arm/dynamic_params/upperarm_length')
                # URDF_forearm_length = rospy.get_param('/human_body/right_arm/dynamic_params/forearm_length')
                # URDF_neck_shoulder_length = rospy.get_param('/human_body/right_arm/dynamic_params/neck_shoulder_length')

                # # KP lenghts
                # KP_upperarm_length = np.linalg.norm(kp_R_Elbow - kp_R_Shoulder)
                # KP_forearm_length = np.linalg.norm(kp_R_Wrist - kp_R_Elbow)
                # KP_neck_shoulder_length = np.linalg.norm((kp_R_Shoulder - kp_L_Shoulder)/2)
                # KP_hips_lenght = np.linalg.norm(kp_R_Hip - kp_L_Hip)

                # # Calcular errores
                # error_upperarm = error(KP_upperarm_length, URDF_upperarm_length)
                # error_forearm =  error(KP_forearm_length, URDF_forearm_length)
                # error_neck_shoulder = error(KP_neck_shoulder_length, URDF_neck_shoulder_length)

                # # Medida de longitudes
                # rospy.loginfo("*******************************************")
                # rospy.loginfo(f"KP   Neck_RShoulder_lenght: {round(KP_neck_shoulder_length, decimals)}")
                # rospy.loginfo(f"URDF Neck_RShoulder_lenght: {round(URDF_neck_shoulder_length, decimals)}")
                # rospy.logerr(f"Error: {round(error_neck_shoulder, decimals)}")
                # rospy.loginfo("-------------------------------------------")
                # rospy.loginfo(f"KP   RUpperarm_lenght     : {round(KP_upperarm_length, decimals)}")
                # rospy.loginfo(f"URDF RUpperarm_lenght     : {round(URDF_upperarm_length, decimals)}")
                # rospy.logerr(f"Error: {round(error_upperarm, decimals)}")
                # rospy.loginfo("-------------------------------------------")
                # rospy.loginfo(f"KP   RForearm_lenght      : {round(KP_forearm_length, decimals)}")
                # rospy.loginfo(f"URDF RForearm_lenght      : {round(URDF_forearm_length, decimals)}")
                # rospy.logerr(f"Error: {round(error_forearm, decimals)}")
                # rospy.loginfo("-------------------------------------------")
                # rospy.loginfo(f"KP Hips_lenght            : {round(KP_hips_lenght, decimals)}")
                # rospy.loginfo("-------------------------------------------")

                # Actualizacion de parametros

                # Actualizar parámetros si el error excede el umbral
                # if error_upperarm > error_threshold:
                #     rospy.set_param('/human_body/right_arm/dynamic_params/upperarm_length', float(round(KP_upperarm_length, decimals)))
                #     self.parameter_update_pub.publish(True) # Publicar notificación de actualización
                #     rospy.logwarn(f"Updated /human_body/right_arm/dynamic_params/upperarm_length to {round(KP_upperarm_length, decimals)}")
                    

                # if error_forearm > error_threshold:
                #     rospy.set_param('/human_body/right_arm/dynamic_params/forearm_length', float(round(KP_forearm_length, decimals)))
                #     self.parameter_update_pub.publish(True) # Publicar notificación de actualización
                #     rospy.logwarn(f"Updated /human_body/right_arm/dynamic_params/forearm_length to {round(KP_forearm_length, decimals)}")

                # if error_neck_shoulder > error_threshold:
                #     rospy.set_param('/human_body/right_arm/dynamic_params/neck_shoulder_length', float(round(KP_neck_shoulder_length, decimals)))
                #     self.parameter_update_pub.publish(True) # Publicar notificación de actualización
                #     rospy.logwarn(f"Updated /human_body/right_arm/dynamic_params/neck_shoulder_length to {round(KP_neck_shoulder_length, decimals)}")
                
                # Example: Look up a transform between 'base_link' and 'right_shoulder'
                TF_Neck = self.lookup_transform("base_link", "RightShoulder")
                TF_R_Shoulder = self.lookup_transform("base_link", "RightUpperArm_f1")
                TF_R_Elbow = self.lookup_transform("base_link", "RightUpperArm")
                TF_R_Wrist = self.lookup_transform("base_link", "RightForeArm")

                # # Mostrar errores
                # print_error(kp_R_Shoulder, TF_R_Shoulder.transform.translation, "Right Shoulder", decimals)
                # rospy.loginfo("-------------------------------------------")
                # print_error(kp_R_Elbow, TF_R_Elbow.transform.translation, "Right Elbow", decimals)
                # rospy.loginfo("-------------------------------------------")
                # print_error(kp_R_Wrist, TF_R_Wrist.transform.translation, "Right Wrist", decimals)
                # rospy.loginfo("*******************************************")

                # Publicar KP
                # Vision. Conversión a Point
                kp_vision_X.neck = Point(*(kp_R_Shoulder + kp_L_Shoulder) / 2)
                kp_vision_X.right_shoulder = Point(*kp_R_Shoulder)
                kp_vision_X.right_elbow = Point(*kp_R_Elbow)
                kp_vision_X.right_wrist = Point(*kp_R_Wrist)


                # URDF. Conversion de TransformStamped a Point
                kp_URDF_X.neck = Point(
                    TF_Neck.transform.translation.x,
                    TF_Neck.transform.translation.y,
                    TF_Neck.transform.translation.z
                )

                kp_URDF_X.right_shoulder = Point(
                    TF_R_Shoulder.transform.translation.x,
                    TF_R_Shoulder.transform.translation.y,
                    TF_R_Shoulder.transform.translation.z
                )

                kp_URDF_X.right_elbow = Point(
                    TF_R_Elbow.transform.translation.x,
                    TF_R_Elbow.transform.translation.y,
                    TF_R_Elbow.transform.translation.z
                )

                kp_URDF_X.right_wrist = Point(
                    TF_R_Wrist.transform.translation.x,
                    TF_R_Wrist.transform.translation.y,
                    TF_R_Wrist.transform.translation.z
                )

                self.kp_URDF_pub.publish(kp_URDF_X)
                self.kp_vision_pub.publish(kp_vision_X)

        except Exception as e:
            rospy.logerr(f"Error processing keypoints: {e}")


if __name__ == '__main__':
    try:
        # Create an instance of the ROSInterface and start listening for messages
        interface = ROSInterface()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down tf2_listener node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
