#!/usr/bin/env python3

import message_filters
import rospy
import numpy as np
import roslib.packages
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from scipy.spatial.transform import Rotation as R
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState

from std_msgs.msg import Bool, Int32

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from upper_limb_kinematics.msg import RightArm, RightArmState # Mensaje a publicar

# Importar módulo ubicado en upper_limb_kinematics/src/upper_limb_kinematics
from upper_limb_kinematics.geometric_utils import are_kp_valid, normalize_vector, calculate_signed_angle_3d, project_Kp_to_plane, rotate_vector_local, calculate_angle_3_points

FLAG_GRIPPED=False

"""
Para probar el switch entre gripper y skeleton:
rostopic pub /grasp_state std_msgs/Bool "data: True"

"""

class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """

        # Inicializar nodo ROS con nombre único
        rospy.init_node('IK_arm_opti', anonymous=False)

        # Get the directory path for the ROS package 'upper_limb_kinematics'
        try:
            path = roslib.packages.get_pkg_dir('upper_limb_kinematics')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'upper_limb_kinematics' was not found.")
            raise e

        # ROS publisher/subs
        # self.right_arm_pub = rospy.Publisher('current_state', RightArmState, queue_size=1) # Definir sin la / barra proporciona flexibilidad para cambiar el namespace del topic en un futuro
        self.joint_state_pub = rospy.Publisher('optitrack/joint_states', JointState, queue_size=1) 
        
        # sub synchonizer
        lshoulder_sub = message_filters.Subscriber('/optitrack/lshoulder', PointStamped)
        rshoulder_sub = message_filters.Subscriber('/optitrack/rshoulder', PointStamped)
        elbow_sub = message_filters.Subscriber('/optitrack/elbow', PointStamped)
        wrist_sub = message_filters.Subscriber('/optitrack/wrist', PointStamped)

        self.sub_optitrack = [lshoulder_sub, rshoulder_sub, elbow_sub, wrist_sub]
        self.sync = message_filters.TimeSynchronizer(self.sub_optitrack, 10)
        self.sync.registerCallback(self.IK_calculator_callback)

        # FLAGS de estado
        self.last_valid_kp = True  # Estado anterior de los KP
        self.is_paused = False  # Estado actual de pausa
        self.was_paused = False  # Estado previo de pausa
        self.data_received = False  # Indica si se han recibido datos

        # Configuración del temporizador para detectar inactividad
        self.timeout_duration = 1.0  # Segundos antes de entrar en pausa
        self.timer = rospy.Timer(rospy.Duration(self.timeout_duration), self.timeout_callback)

        # Agarre
        self.position_franka_EE = np.zeros(3)
        self.flag_gripped = False # A la espera de que se actualice si es necesario
        self.q5_input = 90 # Valor por defecto de q5. Se actualiza con el joystick

        if FLAG_GRIPPED:
            rospy.logwarn("Gripper is enabled. Using gripper position for wrist keypoint.")
        else:
            rospy.logwarn("Gripper is disabled. Using skeleton keypoint for wrist.")

        rospy.loginfo("IK_arm node initialized successfully.")



    
    def timeout_callback(self, event):
        """
        Se ejecuta periódicamente para verificar si se están recibiendo datos.
        """
        if not self.data_received:
            if not self.is_paused:
                rospy.logwarn("No se están recibiendo datos en /optitrack. Nodo en pausa.")
                self.is_paused = True
        else:
            if self.is_paused:
                rospy.loginfo("Datos de /optitrack recibidos nuevamente. Nodo reanudado.")
                self.is_paused = False
            self.data_received = False  # Reset para la siguiente iteración
    
    def IK_calculator_callback(self, lshoulder_msg, rshoulder_msg, elbow_msg, wrist_msg):
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
            # keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])
            right_arm = RightArmState() # msg a construir

            # Guide of Kp
            #  5 left_shoulder
            #  6 right_shoulder
            #  8 right_elbow
            # 10 right_wrist
            # 17 neck. no funciona.

            # Asignacion de Kp
            lshoulder = np.array([lshoulder_msg.point.x, lshoulder_msg.point.y, lshoulder_msg.point.z])
            rshoulder = np.array([rshoulder_msg.point.x, rshoulder_msg.point.y, rshoulder_msg.point.z])
            elbow = np.array([elbow_msg.point.x, elbow_msg.point.y, elbow_msg.point.z])
            wrist = np.array([wrist_msg.point.x, wrist_msg.point.y, wrist_msg.point.z])
            

            if are_kp_valid(rshoulder, lshoulder, elbow, wrist):

                # Si antes eran inválidos, loguea el cambio
                if not self.last_valid_kp:
                    rospy.logdebug("Cálculo IK iniciado: keypoints detectados")
                self.last_valid_kp = True

                # Verificación de la Z de los hombros
                shoulder_z_error = np.abs(rshoulder[2] - lshoulder[2])
                # rospy.loginfo(f"rshoulder_z ={rshoulder[2]}")
                # rospy.loginfo(f"lshoulder_z ={lshoulder[2]}")
                # rospy.loginfo(f"Error_z_shoulder ={shoulder_z_error}")

                if shoulder_z_error >= 0.05:
                    rospy.logerr_throttle(5.0, "ARM IK: Torso no alineado con eje Z. Corrige tu postura!")
                    return
                else:
                    rospy.logdebug("ARM IK calculator: Postura correcta")

                
                ################## Longitudes ##################
                upperarm_length = np.linalg.norm(elbow - rshoulder)
                forearm_length = np.linalg.norm(wrist - elbow)

                right_arm.upperarm_length = upperarm_length
                right_arm.forearm_length = forearm_length

                ################## Vectores esenciales ##################
                # Vector = Final - Inicial

                vect_upperarm = normalize_vector(elbow - rshoulder)
                vect_laterolateral = normalize_vector(rshoulder - lshoulder)

                ################## Calculo q1 ##################
                # Plano sagital
                    # vector normal: vect_laterolateral. Sentido de q1 positivo
                    # Punto del plano: right_shoulder

                vect_ref_q1 = ([0, 0, -1]) # SIMPLIFICACIÓN: El vector de ref es siempre perpendicular al plano del suelo

                # Calculo con signo
                right_arm.q1 = calculate_signed_angle_3d(
                    vect_ref_q1,
                    normalize_vector(project_Kp_to_plane(elbow, rshoulder, vect_laterolateral) - rshoulder), # Solo es necesario proyectar el codo, el RShoulder pertenece
                    vect_laterolateral
                )

                ################## Calculo q2 ##################
                # Coronal plane
                    # Metodo: Se establece el vector q2 de referencia y se rota en el eje local de aplicación de q1. Posteriormente se calcula el ángulo entre el vector de referencia rotado y el vector longitudinal
                    # Punto del plano: right_shoulder
                
                if not np.isnan(right_arm.q1):
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
                    
                    vect_ref_q3 = normalize_vector(lshoulder - rshoulder) # Vector de referencia.
                    
                    vector_ref_q3_rot1 = rotate_vector_local(vect_ref_q3, vect_laterolateral, right_arm.q1) # No tiene interes porq se rota sobre si mismo
                    vector_ref_q3_rot2 = rotate_vector_local(vector_ref_q3_rot1, normal_vector_q2_sign, right_arm.q2)

                    right_arm.q3 = calculate_signed_angle_3d(
                        vector_ref_q3_rot2, # no hace falta proyectar, pertenece siempre
                        normalize_vector(project_Kp_to_plane(wrist, elbow, vect_upperarm) - elbow),
                        vect_upperarm
                    )

                ################## Calculo q4 ##################
                right_arm.q4 = calculate_angle_3_points(rshoulder, elbow, wrist)

                # Almacenar los KP usados para el cálculo del IK.
                right_arm.right_shoulder = Point(*rshoulder)
                right_arm.right_elbow    = Point(*elbow)
                right_arm.right_wrist    = Point(*wrist)
                right_arm.left_shoulder  = Point(*lshoulder)

                # Publish the right arm angles calculated. Se publican tb los kp
                right_arm.header.stamp = rshoulder_msg.header.stamp # Usar el mismo timestamp que el msg de entrada
                right_arm.header.frame_id = 'base_link' # para saber sobre que frame están los kp
                # self.right_arm_pub.publish(right_arm)

                ## Publicar joint states
                joint_state = JointState()
                joint_state.header.stamp = rshoulder_msg.header.stamp

                # Right arm assignment
                joint_state.name = np.array(['right_arm_q1', 'right_arm_q2', 'upperarm_length', 'right_arm_q3', 'right_arm_q4', 'forearm_length', 'right_arm_q5'])
                
                joint_state.position = np.array([np.radians(right_arm.q1),
                                                np.radians(right_arm.q2),
                                                right_arm.upperarm_length,
                                                np.radians(right_arm.q3),
                                                np.radians(right_arm.q4),
                                                right_arm.forearm_length,
                                                np.radians(0.0) # q5 supinación input from joystick
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
            import traceback
            traceback.print_exc()


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
