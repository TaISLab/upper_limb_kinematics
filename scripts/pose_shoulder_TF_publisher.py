#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import tf_conversions
import geometry_msgs.msg
import roslib.packages

from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Custom message type for 3D skeleton data

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''
    return not any(np.all(kp == 0.0) for kp in keypoints)

def calculate_yaw(kp_R_Shoulder, kp_L_Shoulder):
    """
    Calcula el ángulo yaw para un TF. 
    Este ángulo representa la orientación entre el Eje Y del Mundo (base_link) y el vector de hombros
    vector de hombros: hombro_izquierdo-hombro_derecho

    Parameters:
    - kp_R_Shoulder: Coordenadas (x, y, z) del hombro derecho.
    - kp_L_Shoulder: Coordenadas (x, y, z) del hombro izquierdo.

    Returns:
    - yaw: Ángulo en radianes.
    """
    # Calcular el vector del hombro
    shoulder_vector = np.array(kp_L_Shoulder) - np.array(kp_R_Shoulder)

    # Calcular yaw en el plano XY
    y_ref = np.array([0, 1, 0])
    num = np.dot(y_ref, shoulder_vector)
    den = np.linalg.norm(y_ref) * np.linalg.norm(shoulder_vector)

    cos_theta = np.clip(num / den, -1.0, 1.0) # -1.0 y 1.0 son los límites del cos
    yaw = np.arccos(cos_theta) 

    return yaw


class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """
        # Initialize the ROS node with a unique name
        rospy.init_node('locate_shoulder_TF_publisher', anonymous=False)
        self.child_id = rospy.get_param('~child_id')
        rospy.loginfo(self.child_id)

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # Set up subscriber for /skeleton_3D
        self.sub_skeleton_3D_keypoints = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.publish_base_to_rshoulder_tf) # Dont forget SELF.


    def publish_base_to_rshoulder_tf(self, msg):
        """
        Callback function to publish rshoulder tf

        Parameters:
        - msg: Skeleton3D message
        """
        # Guide of Kp
        #  5 left_shoulder
        #  6 right_shoulder

        keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])
        kp_R_Shoulder = keypointsX[6]
        kp_L_Shoulder = keypointsX[5]

        if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder):
            br = tf2_ros.TransformBroadcaster()
            t = geometry_msgs.msg.TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = self.child_id

            t.transform.translation.x = kp_R_Shoulder[0]
            t.transform.translation.y = kp_R_Shoulder[1]
            t.transform.translation.z = kp_R_Shoulder[2]

            # yaw_calculated = calculate_yaw(kp_R_Shoulder, kp_L_Shoulder)
            
            # roll_calculated = calculate_roll(kp_R_Shoulder, kp_L_Shoulder)
            # q = tf_conversions.transformations.quaternion_from_euler(-roll_calculated, 0, 1.5708+yaw_calculated)
            
            # Simplificacion: Los hombros no rotan en x
            yaw_calculated = calculate_yaw(kp_R_Shoulder, kp_L_Shoulder)
            q = tf_conversions.transformations.quaternion_from_euler(0, 0, -yaw_calculated)

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            br.sendTransform(t)
            
        else:
            rospy.loginfo("TF entre base_link y base_shoulder no publicada. Faltan kp_R_Shoulder, kp_L_Shoulder")

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
