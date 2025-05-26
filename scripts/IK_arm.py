#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
from geometry_msgs.msg import Point, PoseStamped
from scipy.spatial.transform import Rotation as R
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from human_articular_space.msg import RightArm # Mensaje a publicar

# Importar módulo ubicado en human_articular_space/src/human_articular_space
from human_articular_space.geometric_utils import are_kp_valid, normalize_vector, calculate_signed_angle_3d, project_Kp_to_plane, rotate_vector_local, calculate_angle_3_points

FLAG_GRIPPED=True

class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """

        # Inicializar nodo ROS con nombre único
        rospy.init_node('IK_arm', anonymous=False)

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # ROS publisher/subs
        self.right_arm_pub = rospy.Publisher('description', RightArm, queue_size=1) # Definir sin la / barra proporciona flexibilidad para cambiar el namespace del topic en un futuro
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size=1) 
        
        rospy.Subscriber('/skeleton_3D', Skeleton3D, self.IK_calculator_callback) # Dont forget SELF.
        rospy.Subscriber('/franka_state_controller/franka_states', FrankaState, self.franka_pose_callback)

        # FLAGS de estado
        self.last_valid_kp = True  # Estado anterior de los KP
        self.is_paused = False  # Estado actual de pausa
        self.was_paused = False  # Estado previo de pausa
        self.data_received = False  # Indica si se han recibido datos

        # Configuración del temporizador para detectar inactividad
        self.timeout_duration = 1.0  # Segundos antes de entrar en pausa
        self.timer = rospy.Timer(rospy.Duration(self.timeout_duration), self.timeout_callback)

        self.position_franka_EE = np.zeros(3)

    
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
    
    def franka_pose_callback(self, msg):
        """
        Extrae la posición del efector final del franka. 
        """
        # Extraer las matrices de transformación homogénea
        O_T_EE = np.array(msg.O_T_EE).reshape(4, 4).T
        EE_T_K = np.array(msg.EE_T_K).reshape(4, 4).T

        # Calcular O_T_K = O_T_EE @ EE_T_K
        O_T_K = np.dot(O_T_EE, EE_T_K)

        # Extraer la posición de la herramienta (última columna, primeras 3 filas)
        tool_position = O_T_K[:3, 3]

        # Convertir directamente a array numpy
        self.position_franka_EE = tool_position

        # rospy.loginfo(tool_position)

    
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
            right_arm = RightArm() # msg a construir

            # Guide of Kp
            #  5 left_shoulder
            #  6 right_shoulder
            #  8 right_elbow
            # 10 right_wrist
            # 17 neck. no funciona.

            # Asignacion de Kp
            kp_R_Shoulder = keypointsX[6]
            kp_R_Elbow = keypointsX[8]
            kp_R_Wrist = keypointsX[10]
            kp_L_Shoulder = keypointsX[5]

            if FLAG_GRIPPED:
                kp_R_Wrist = self.position_franka_EE

            right_arm.header.stamp = rospy.Time.now()

            if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder, kp_R_Elbow, kp_R_Wrist):

                # Si antes eran inválidos, loguea el cambio
                if not self.last_valid_kp:
                    rospy.logdebug("Cálculo IK iniciado: keypoints detectados")
                self.last_valid_kp = True

                # Verificación de la Z de los hombros
                shoulder_z_error = np.abs(kp_R_Shoulder[2] - kp_L_Shoulder[2])
                rospy.loginfo(f"kp_R_Shoulder_z ={kp_R_Shoulder[2]}")
                rospy.loginfo(f"kp_L_Shoulder_z ={kp_L_Shoulder[2]}")
                rospy.loginfo(f"Error_z_shoulder ={shoulder_z_error}")

                if shoulder_z_error >= 0.05:
                    rospy.logerr_throttle("ARM IK calculator: Torso no alineado con eje Z. MCI del brazo incorrecto. Corrige tu postura!")
                    return
                else:
                    rospy.logdebug("ARM IK calculator: Postura correcta")

                
                ################## Longitudes ##################
                upperarm_length = np.linalg.norm(kp_R_Elbow - kp_R_Shoulder)
                forearm_length = np.linalg.norm(kp_R_Wrist - kp_R_Elbow)

                right_arm.upperarm_length = upperarm_length
                right_arm.forearm_length = forearm_length

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
                right_arm.q1 = calculate_signed_angle_3d(
                    vect_ref_q1,
                    normalize_vector(project_Kp_to_plane(kp_R_Elbow, kp_R_Shoulder, vect_laterolateral) - kp_R_Shoulder), # Solo es necesario proyectar el codo, el RShoulder pertenece
                    vect_laterolateral
                )

                ################## Calculo q2 ##################
                # Coronal plane
                    # Metodo: Se establece el vector q2 de referencia y se rota en el eje local de aplicación de q1. Posteriormente se calcula el ángulo entre el vector de referencia rotado y el vector longitudinal
                    # Punto del plano: right_shoulder
                
                if (right_arm.q1 != np.nan):
                    vect_ref_q2 = ([0, 0, -1]) # SIMPLIFICACION: Vector Z negativo
                    vect_ref_q2_rot = rotate_vector_local(vect_ref_q2, vect_laterolateral, right_arm.q1) # CORREGIDO: Se3 rota sobre el eje local
                    vect_ref_q2_rot_norm = normalize_vector(vect_ref_q2_rot)
                    
                    normal_vector_q2_sign = np.cross(vect_upperarm, vect_laterolateral) # Para definir el signo de q2

                    right_arm.q2 = calculate_signed_angle_3d(
                        vect_ref_q2_rot_norm, # Vector de referencia q1
                        vect_upperarm, # No necesitan ser proyectados
                        normal_vector_q2_sign
                    )

                ################## Calculo q3 ##################
                if (right_arm.q1 != np.nan) and (right_arm.q2 != np.nan):
                
                    # Plano de proyección
                    #   vector normal: vector longitudinal del brazo
                    #   Pto de aplicación: codo
                    
                    vect_ref_q3 = normalize_vector(kp_L_Shoulder - kp_R_Shoulder) # Vector de referencia.
                    
                    vector_ref_q3_rot1 = rotate_vector_local(vect_ref_q3, vect_laterolateral, right_arm.q1) # No tiene interes porq se rota sobre si mismo
                    vector_ref_q3_rot2 = rotate_vector_local(vector_ref_q3_rot1, normal_vector_q2_sign, right_arm.q2)

                    right_arm.q3 = calculate_signed_angle_3d(
                        vector_ref_q3_rot2, # no hace falta proyectar, pertenece siempre
                        normalize_vector(project_Kp_to_plane(kp_R_Wrist, kp_R_Elbow, vect_upperarm) - kp_R_Elbow),
                        vect_upperarm
                    )

                ################## Calculo q4 ##################
                right_arm.q4 = calculate_angle_3_points(kp_R_Shoulder, kp_R_Elbow, kp_R_Wrist)


                # Publish the right arm angles calculated
                right_arm_msg = RightArm()
                right_arm_msg = right_arm

                self.right_arm_pub.publish(right_arm_msg)

                # Publicar joint states
                joint_state = JointState()
                joint_state.header = msg.header 

                # Right arm assignment
                joint_state.name = np.array(['right_arm_q1', 'right_arm_q2', 'upperarm_length', 'right_arm_q3', 'right_arm_q4', 'forearm_length', 'right_arm_q5'])
                
                joint_state.position = np.array([np.radians(msg.right_arm.q1),
                                                np.radians(msg.right_arm.q2),
                                                msg.right_arm.upperarm_length,
                                                np.radians(msg.right_arm.q3),
                                                np.radians(msg.right_arm.q4),
                                                msg.right_arm.forearm_length,
                                                np.radians(0.0)
                                                ])
                
                self.joint_state_pub.publish(joint_state)

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
