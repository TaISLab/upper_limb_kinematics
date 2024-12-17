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
    Calcula el ángulo yaw y el quaternion correspondiente para un TF.

    Parameters:
    - kp_R_Shoulder: Coordenadas (x, y, z) del hombro derecho.
    - kp_L_Shoulder: Coordenadas (x, y, z) del hombro izquierdo.

    Returns:
    - yaw: Ángulo en radianes.
    - quaternion: Quaternion correspondiente [x, y, z, w].
    """
    # Calcular el vector del hombro
    shoulder_vector = np.array(kp_R_Shoulder) - np.array(kp_L_Shoulder)

    # Calcular yaw en el plano XY
    yaw = np.arctan2(shoulder_vector[1], shoulder_vector[0])

    # Convertir el ángulo yaw a un quaternion
    quaternion = R.from_euler('z', yaw).as_quat()  # Rotación solo en Z

    return yaw, quaternion


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
        # 17 neck
        keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])
        kp_R_Shoulder = keypointsX[6]
        kp_L_Shoulder = keypointsX[5]
        kp_neck = (kp_R_Shoulder + kp_L_Shoulder)/2 # no se publica el cuello keypointsX[17]

        if are_kp_valid(kp_R_Shoulder, kp_L_Shoulder):
            br = tf2_ros.TransformBroadcaster()
            t = geometry_msgs.msg.TransformStamped()

            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "base_link"
            t.child_frame_id = self.child_id

            t.transform.translation.x = kp_neck[0]
            t.transform.translation.y = kp_neck[1]
            t.transform.translation.z = kp_neck[2]

            yaw_calculated, quaternion = calculate_yaw(kp_R_Shoulder, kp_L_Shoulder)
            q = tf_conversions.transformations.quaternion_from_euler(0, 0, 1.5708+yaw_calculated)
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
