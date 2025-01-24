#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
import tf2_ros

from std_msgs.msg import Bool  # Mensaje estándar para notificaciones
from geometry_msgs.msg import Point

from skeleton_3d.msg import Skeleton3D          # Import custom message type
from human_articular_space.msg import KP_URDF   # Import custom message type
from human_articular_space.msg import KP_vision # Import custom message type

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''
    return not any(np.all(kp == 0.0) for kp in keypoints)


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
            transform = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))  
            rospy.logdebug(f"Transform found: {source_frame} -> {target_frame}")
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Could not find transform from {source_frame} to {target_frame}: {e}")
            return None

    def kp_callback(self, msg):
        """
        Callback function for Skeleton3D messages.
        Esta función obtiene la posición de las articulaciones tras la cinemática inversa.
        ROS hace el cálculo automáticamente dentro de TF gracias al URDF, supliendo la cinemática directa.
        """

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

            # Validate keypoints
            if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder, kp_R_Elbow, kp_R_Wrist):
                
                # Cinemática directa.
                TF_R_Shoulder = self.lookup_transform("base_link", "base_shoulder")
                TF_R_Elbow = self.lookup_transform("base_link", "RightUpperArm")
                TF_R_Wrist = self.lookup_transform("base_link", "RightForeArm")

                # Publicar KP
                # Vision. Conversión a Point
                kp_vision_X.right_shoulder = Point(*kp_R_Shoulder)
                kp_vision_X.right_elbow = Point(*kp_R_Elbow)
                kp_vision_X.right_wrist = Point(*kp_R_Wrist)

                # URDF. Conversion de TransformStamped a Point
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
