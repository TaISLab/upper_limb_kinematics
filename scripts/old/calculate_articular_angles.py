#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import roslib.packages
from geometry_msgs.msg import Point

# Import the custom skeleton tracking class
# from skeleton_tracker_3d import SkeletonTracker3D  

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Custom message type for 3D skeleton data
from human_articular_space.msg import HumanJointAngles, Elbow_angles, Shoulder_angles

def are_kp_valid(*keypoints):
    '''
    Verify if all the keypoints are valid. Kp valid: not(0.0)
    '''

    return not any(np.all(kp == 0.0) for kp in keypoints)

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

class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """
        # Initialize the ROS node with a unique name
        rospy.init_node('human_articular_space_calculator', anonymous=False)

        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # Set up a ROS publisher for Human Angles
        # self.pub_skeleton = rospy.Publisher('/topic', mensaje_importado, queue_size=10)
        self.pub_human_articular_angles = rospy.Publisher('/human_articular_angles', HumanJointAngles, queue_size=10)

        # Set up subscriber for /skeleton_3D
        self.sub_skeleton_3D_keypoints = rospy.Subscriber('/skeleton_3D', Skeleton3D, self.calculate_angles_callback) # Dont forget SELF.

    def calculate_angles_callback(self, msg):
        """
        Callback function to calculate human articular angles.

        Parameters:
        - msg: Skeleton3D message
        """

        try:

            #rospy.loginfo("Received Skeleton3D message") #only debug

            # Convert the received messages into a list of 3D keypoint arrays.
            # Transform to ros msg to numpy vector
            keypointsX = np.array([(kp.x, kp.y, kp.z) for kp in msg.keypoints])

            human_articular_angles = HumanJointAngles()

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

            kp_right_shoulder = keypointsX[6]
            kp_right_elbow = keypointsX[8]
            kp_right_wrist = keypointsX[10]

            kp_left_shoulder = keypointsX[5]
            kp_left_elbow = keypointsX[7]
            kp_left_wrist = keypointsX[9]
            
            kp_left_hip = keypointsX[11]
            kp_right_hip = keypointsX[12]

            human_articular_angles.header.stamp = rospy.Time.now()

            # Calculate alpha elbow angles
            human_articular_angles.right_elbow.alpha = calculate_angle_3_points(kp_right_shoulder, kp_right_elbow, kp_right_wrist)
            human_articular_angles.left_elbow.alpha = calculate_angle_3_points(kp_left_shoulder, kp_left_elbow, kp_left_wrist)

            # Calculate alpha shoulder angles
            human_articular_angles.right_shoulder.alpha = calculate_angle_3_points(kp_right_hip, kp_right_shoulder, kp_right_elbow)
            human_articular_angles.left_shoulder.alpha = calculate_angle_3_points(kp_left_hip, kp_left_shoulder, kp_left_elbow)

            # Calculate beta shoulder angles
            human_articular_angles.right_shoulder.beta = calculate_angle_3_points(kp_left_shoulder, kp_right_shoulder, kp_right_elbow)
            human_articular_angles.left_shoulder.beta = calculate_angle_3_points(kp_right_shoulder, kp_left_shoulder, kp_left_elbow)

            # Publish the human angles calculated
            self.pub_human_articular_angles.publish(human_articular_angles)

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
