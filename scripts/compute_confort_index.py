#!/usr/bin/env python3

import rospy
import numpy as np

# Import custom message types
from human_articular_space.msg import HumanBody # Mensaje a subscribirnos

from franka_concerto.funciones_utiles import calculate_quaternion_0_F, np_array_to_point, np_array_to_vector3, point_to_np_array, vector3_to_np_array




class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """

        # Inicializar nodo ROS con nombre único
        rospy.init_node('Compute_Confort_Indexes', anonymous=False)

        # ROS publisher
        topic_name = "/human_body/description"
        rospy.Subscriber(topic_name, HumanBody, self.compute_indexes, queue_size=1)

    
    def compute_indexes(self, msg):
        """
        Callback function
        """
        human_body = msg
        q = np.array([msg.right_arm.q1, msg.right_arm.q2, msg.right_arm.q3, msg.right_arm.q4])
        q = np.radians(q)
            
        qmin = np.radians([-25, -10, -90, 0])
        qmax = np.radians([120, 75, 120, 180])
        qmid = (qmin + qmax) / 2  # Mid joint angle

        # workload = np.sum(q[:3, :]**2, axis=0) + (q[3, :] - np.pi/2)**2
        
        liegois = np.sum(((q - qmid) / (qmid - qmax))**2, axis=0) / 4
        liegois = np.clip(liegois, 0, 1)  # The closest to 0 the better
        
        rom_comfort = 2 * np.minimum(np.abs(qmax - q), np.abs(qmin - q)) / (qmax - qmin) # Por que??
        rom_comfort[(q > qmax) | (q < qmin)] = 0  # El /4 normaliza el indice de liegois entre 0 y 1
        rom_comfort = np.sum(rom_comfort, axis=0) / 4  # The closest to 1 the better

        # rospy.loginfo(f"ROM q1: {rom_comfort[0]}")
        # rospy.loginfo(f"ROM q2: {rom_comfort[1]}")
        # rospy.loginfo(f"ROM q3: {rom_comfort[2]}")
        # rospy.loginfo(f"ROM q4: {rom_comfort[3]}")


        # rospy.loginfo(f"ROM total: {rom_comfort}")
        # rospy.loginfo(f"liegois: {liegois}")
        

    
        
        n = q.shape[0]
        joint_usage_average = np.zeros(n)
        joint_usage_average_filtered = np.zeros(n)
        joint_usage_previus = np.zeros(n)
        
        qavg2 = np.zeros(4)  
        findex = 0.01 
        
        # for i in np.arange(0, n, 1):
        #     qinit = np.zeros(4)
        #     qavg = np.sum(q[:, :i], axis=1) / i
        #     qavg2 = findex * q[:, i] + (1 - findex) * qavg2  

        #     qavg[np.isnan(qavg)] = 0
        #     qavg2[np.isnan(qavg2)] = 0
            
        #     joint_usage_average[i] = np.sum(np.abs(q[:, i] - qavg))
        #     joint_usage_average_filtered[i] = np.sum(np.abs(q[:, i] - qavg2))
            
        #     if i == 0:
        #         joint_usage_previus[i] = np.sum(np.abs(q[:, i] - qinit))
        #     else:
        #         joint_usage_previus[i] = np.sum(np.abs(q[:, i] - q[:, i-1]))

        for i in np.arange(0, n, 1):
            qinit = np.zeros(4)
            qavg = np.sum(q[i], axis=1) / i
            qavg2 = findex * q[i] + (1 - findex) * qavg2  

            qavg[np.isnan(qavg)] = 0
            qavg2[np.isnan(qavg2)] = 0
            
            joint_usage_average[i] = np.sum(np.abs(q[i] - qavg))
            joint_usage_average_filtered[i] = np.sum(np.abs(q[i] - qavg2))
            
            if i == 0:
                joint_usage_previus[i] = np.sum(np.abs(q[i] - qinit))
            else:
                joint_usage_previus[i] = np.sum(np.abs(q[i] - q[i-1]))

        joint_usage_average
        rospy.loginfo(f"joint_usage_average: {joint_usage_average[0]}")

        
        # return workload, liegois, rom_comfort, joint_usage_average, joint_usage_previus, joint_usage_average_filtered

        
if __name__ == '__main__':
    try:
        # Create an instance of the ROSInterface and start listening for messages
        interface = ROSInterface()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down compute indexes node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
