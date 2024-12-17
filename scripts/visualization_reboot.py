#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
import os

class RobotStatePublisherWrapper:
    def __init__(self):
        rospy.init_node('robot_state_publisher_wrapper', anonymous=True)
        self.last_robot_description = None
        rospy.Subscriber('/parameter_update', Bool, self.check_robot_description)

    def check_robot_description(self, msg):
        """Callback para manejar actualizaciones del URDF."""
        # Obtener el parámetro actual
        current_description = rospy.get_param('/robot_description', '')

        # Verificar si el URDF ha cambiado
        if current_description != self.last_robot_description:
            rospy.loginfo("URDF actualizado. Reiniciando robot_state_publisher y RViz...")

            # Matar robot_state_publisher si existe
            ros_nodes = os.popen("rosnode list").read()
            if "/human_body/right_arm/robot_state_publisher" in ros_nodes:
                os.system("rosnode kill /human_body/right_arm/robot_state_publisher")
                rospy.sleep(2)  # Esperar a que el nodo se cierre

            # Relanzar robot_state_publisher
            os.system("rosrun robot_state_publisher robot_state_publisher __ns:=/human_body/right_arm &")
            rospy.loginfo("robot_state_publisher relanzado.")

            # Reiniciar RViz
            if "/rviz_arm" in ros_nodes:
                os.system("rosnode kill /rviz_arm")
                # rospy.sleep(2)  # Esperar para que RViz reinicie
                # os.system("rosrun rviz rviz -d $(rospack find human_articular_space)/config/config_right_arm.rviz __name:=rviz_arm &")
                # rospy.loginfo("RViz reiniciado con la configuración actualizada.")
            else:
                rospy.logwarn("El nodo RViz no está corriendo.")

            # Actualizar el estado del último URDF
            self.last_robot_description = current_description

if __name__ == "__main__":
    try:
        wrapper = RobotStatePublisherWrapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


