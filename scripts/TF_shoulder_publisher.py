#!/usr/bin/env python3

import rospy
import numpy as np
import tf2_ros
import tf_conversions
import geometry_msgs.msg
import roslib.packages
import tf
import math

from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Custom message type for 3D skeleton data

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''
    return not any(np.all(kp == 0.0) for kp in keypoints)


def angle_between_vectors_360(v1, v2):
    """
    Calculate the angle between two 2D vectors in the XY plane, returning a value in radians between 0 and 2pi.

    Args:
        v1 (array): The first 2D vector [x, y].
        v2 (array): The second 2D vector [x, y].

    Returns:
        float: The angle between the two vectors in radians, ranging from 0 to 2pi.
    """

    # NOTA: ATAN2 MIDE EL ÁNGULO ENTRE LAS DOS COMPONENTES DEL VECTOR EN EL PLANO XY MEDIDO SOBRE X
    #       Nosotros necesitamos el ángulo entre Y y Vector. Luego -90, que nos lo da el theta2.
    #       O de manera general, como se propone en esta función:
    
    theta1 = np.arctan2(v1[1], v1[0])
    
    theta2 = np.arctan2(v2[1], v2[0])
    
    delta = theta2 - theta1
    if delta < 0:
        delta += 2 * np.pi

    # rospy.loginfo(f"DEBUG theta1 deg = {math.degrees(theta1)}")
    # rospy.loginfo(f"DEBUG theta2 deg = {math.degrees(theta2)}")
    # rospy.loginfo(f"DEBUG delta deg = {math.degrees(delta)}")

    return delta  # en radianes entre 0 y 2pi


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
        
            shoulder_vector = np.array(kp_L_Shoulder) - np.array(kp_R_Shoulder)

            y_ref = np.array([0, 1, 0])
            yaw_calculated = angle_between_vectors_360(y_ref, shoulder_vector)
            q = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw_calculated)

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
