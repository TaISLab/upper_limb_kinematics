#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
import roslib.packages
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

# Import custom message types
from human_articular_space.msg import HumanBody

def has_nan_in_fields(msg):
    """
    Verifica campos específicos de un mensaje para detectar NaN.
    """
    try:
        fields_to_check = [msg.right_arm.q1, msg.right_arm.q2, msg.right_arm.q3, msg.right_arm.q4]
        for i, value in enumerate(fields_to_check):
            if np.isnan(value):
                # rospy.logwarn(f"Se detectó NaN en msg.right_arm.q{i+1}")
                return True
        return False
    except AttributeError as e:
        rospy.logerr(f"El mensaje no tiene los campos esperados: {e}")
        return True  # Considera inválido si el mensaje no tiene los campos


class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """
        # Initialize the ROS node with a unique name
        rospy.init_node('joint_states_publisher', anonymous=False)

        # Obtener el namespace del argumento pasado desde el archivo launch
        namespace = rospy.get_param('~namespace1', 'default_namespace')  # Valor por defecto


        # Get the directory path for the ROS package 'human_articular_space'
        try:
            path = roslib.packages.get_pkg_dir('human_articular_space')
        except roslib.packages.InvalidROSPkgException as e:
            rospy.logerr("The package 'human_articular_space' was not found.")
            raise e

        # Set up a ROS publisher
        topic_name = f"/{namespace}/description"
        self.pub_joint_state = rospy.Publisher('joint_states', JointState, queue_size=10) 

        # Set up subscriber
        self.sub_human_body = rospy.Subscriber(topic_name, HumanBody, self.convert_to_jointstates)


    def convert_to_jointstates(self, msg):
        """
        Callback function to convert human_articular_space/HumanBody msgs to sensor_msg/JointStates msgs

        Parameters:
        - msg: HumanBody message
        """

        try:
            
            if not has_nan_in_fields(msg): # Cumple cuando es falso
            
                joint_state = JointState()
                joint_state.header = msg.header 

                # Right arm assignment
                joint_state.name = np.array(['right_arm_q1', 'right_arm_q2', 'right_arm_q3', 'right_arm_q4', 'right_arm_q5'])
                joint_state.position = np.array([np.radians(msg.right_arm.q1),
                                                np.radians(msg.right_arm.q2),
                                                np.radians(msg.right_arm.q3),
                                                np.radians(msg.right_arm.q4), 
                                                np.radians(0.0)
                                                ])
            
                ## DEBUG
                #joint_state.position = [1.0, 0.0, 0.0, 0.0, 0.0]
                
                self.pub_joint_state.publish(joint_state)
            else:
                rospy.logwarn(f"Se detectó NaN en algún campo de joint_states")

        except Exception as e:
            # Log any errors that occur during calculate
            rospy.logerr(f"Error with the assignment: {e}")


if __name__ == '__main__':
    try:
        # Create an instance of the ROSInterface and start listening for messages
        interface = ROSInterface()


        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
