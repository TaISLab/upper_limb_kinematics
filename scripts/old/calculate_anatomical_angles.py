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
from human_articular_space.msg import HumanAnatomicalAngles, Elbow_anatomical, Shoulder_anatomical

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
        np.array: Vector normal al plano
    """ 

    A, B, C = np.array(A), np.array(B), np.array(C)

    # Calcular los vectores BA, BC
    BA = A - B
    BC = C - B

    normal = np.cross(BA, BC)
    
    return normal

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
        self.pub_human_anatomical_angles = rospy.Publisher('/human_anatomical_angles', HumanAnatomicalAngles, queue_size=10)

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

            human_anatomical_angles = HumanAnatomicalAngles()

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

            human_anatomical_angles.header.stamp = rospy.Time.now()

            # Calculate sagittal elbow angles. its not necessary project kp
            human_anatomical_angles.right_elbow.sagittal = calculate_angle_3_points(kp_right_shoulder, kp_right_elbow, kp_right_wrist)
            human_anatomical_angles.left_elbow.sagittal = calculate_angle_3_points(kp_left_shoulder, kp_left_elbow, kp_left_wrist)

            # Calculate transverse shoulder angles
            if are_kp_valid(kp_right_shoulder, kp_left_shoulder, kp_right_hip, kp_left_hip, kp_right_elbow, kp_left_elbow):

                # Calculate planes and secondary points
                half_shoulder = (kp_right_shoulder + kp_left_shoulder) / 2
                half_hip = (kp_right_hip + kp_left_hip) / 2

                # Transverse (axial) plane
                    # vect
                    # point: half_hip
                longitudinal_vect = half_shoulder - half_hip # half_vect

                # Transverse plane. We want to measure horizontal movements. Project: shoulder1 shoulder2 elbow1

                # Projects Kp
                projected_kp_right_shoulder = project_Kp_to_plane(kp_right_shoulder, half_hip, longitudinal_vect)
                projected_kp_left_shoulder = project_Kp_to_plane(kp_left_shoulder, half_hip, longitudinal_vect)

                # Calculate angles
                human_anatomical_angles.right_shoulder.transverse = calculate_angle_3_points(
                    projected_kp_left_shoulder,
                    projected_kp_right_shoulder,
                    project_Kp_to_plane(kp_right_elbow, half_hip, longitudinal_vect)
                )

                human_anatomical_angles.left_shoulder.transverse = calculate_angle_3_points(
                    projected_kp_right_shoulder,
                    projected_kp_left_shoulder,
                    project_Kp_to_plane(kp_left_elbow, half_hip, longitudinal_vect)
                )

            else: 
                human_anatomical_angles.right_shoulder.transverse = np.nan
                human_anatomical_angles.left_shoulder.transverse = np.nan

            # Calculate coronal angles
            if are_kp_valid(kp_right_shoulder, kp_left_shoulder, kp_right_hip, kp_left_hip, kp_right_elbow, kp_left_elbow):

                # Coronal plane
                    # vect: dorsoventral_vect. normal vect to a 3 points
                    # point: half_hip

                half_hip = (kp_right_hip + kp_left_hip) / 2
                dorsoventral_vect = normal_vector_from_points(kp_right_shoulder, half_hip, kp_left_shoulder)

                # Coronal plane. We want to measure frontal movements. Project: Hip, shoulder, elbow
                # Projects Kp and calculate angles
                human_anatomical_angles.right_shoulder.coronal = calculate_angle_3_points(
                    half_hip, # no hace falta proyectar, ya pertenece
                    project_Kp_to_plane(kp_right_shoulder, half_hip, dorsoventral_vect),
                    project_Kp_to_plane(kp_right_elbow, half_hip, dorsoventral_vect)
                )

                human_anatomical_angles.left_shoulder.coronal = calculate_angle_3_points(
                    half_hip, # no hace falta proyectar, ya pertenece
                    project_Kp_to_plane(kp_left_shoulder, half_hip, dorsoventral_vect),
                    project_Kp_to_plane(kp_left_elbow, half_hip, dorsoventral_vect)
                )

            else: 
                human_anatomical_angles.right_shoulder.coronal = np.nan
                human_anatomical_angles.left_shoulder.coronal = np.nan

            # Calculate Sagittal shoulder angles
            if are_kp_valid(kp_right_shoulder, kp_left_shoulder, kp_right_hip, kp_left_hip, kp_right_elbow, kp_left_elbow):

                # sagittal plane
                    # vect: left-right axis, horizontal axis or frontal axis
                    # point: half_hip
                half_hip = (kp_right_hip + kp_left_hip) / 2
                laterolateral_vect = kp_right_hip - half_hip
                
                # Sagittal plane. We want to measure lateral movements. Project: half_hp(not necessary), shoulder, elbow
                # Projects Kp and calculate angles
                human_anatomical_angles.right_shoulder.sagittal = calculate_angle_3_points(
                    half_hip,
                    project_Kp_to_plane(kp_right_shoulder, half_hip, laterolateral_vect),
                    project_Kp_to_plane(kp_right_elbow, half_hip, laterolateral_vect)
                )

                human_anatomical_angles.left_shoulder.sagittal = calculate_angle_3_points(
                    half_hip,
                    project_Kp_to_plane(kp_left_shoulder, half_hip, laterolateral_vect),
                    project_Kp_to_plane(kp_left_elbow, half_hip, laterolateral_vect)
                )

            else: 
                human_anatomical_angles.right_shoulder.sagittal = np.nan
                human_anatomical_angles.left_shoulder.sagittal = np.nan

            # Publish the human angles calculated
            self.pub_human_anatomical_angles.publish(human_anatomical_angles)

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
