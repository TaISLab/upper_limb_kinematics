#!/usr/bin/env python

import rospy
from dynamic_reconfigure.server import Server
from human_articular_space.cfg import DynamicParametersConfig
from std_msgs.msg import Bool  # Mensaje estándar para notificaciones

class DynamicParameterNode:
    def __init__(self):
        rospy.init_node("dynamic_parameter_node")

        # Publicador para notificar cambios
        self.parameter_update_pub = rospy.Publisher("/parameter_update", Bool, queue_size=10)

        # Leer valores iniciales del servidor de parámetros o establecer predeterminados
        self.upperarm_length = rospy.get_param('/human_body/right_arm/dynamic_params/upperarm_length', 0.28)
        self.forearm_length = rospy.get_param('/human_body/right_arm/dynamic_params/forearm_length', 0.25)
        self.neck_shoulder_length = rospy.get_param('/human_body/right_arm/dynamic_params/neck_shoulder_length', 0.175)

        # Publicar los valores en el servidor de parámetros
        rospy.set_param('/human_body/right_arm/dynamic_params/upperarm_length', self.upperarm_length)
        rospy.set_param('/human_body/right_arm/dynamic_params/forearm_length', self.forearm_length)
        rospy.set_param('/human_body/right_arm/dynamic_params/neck_shoulder_length', self.neck_shoulder_length)

        # Configurar el servidor de dynamic_reconfigure con los valores iniciales
        self.server = Server(DynamicParametersConfig, self.dynamic_reconfigure_callback)
        rospy.loginfo("DynamicParameterNode inicializado con valores iniciales.")

        self.parameter_update_pub.publish(True)

    def dynamic_reconfigure_callback(self, config, level):
        """Callback para manejar cambios en los parámetros."""
        rospy.loginfo(f"Parámetros actualizados desde dynamic_reconfigure: {config}")

        # Actualizar los valores locales
        self.upperarm_length = config.upperarm_length
        self.forearm_length = config.forearm_length
        self.neck_shoulder_length = config.neck_shoulder_length

        # Actualizar los valores en el servidor de parámetros
        rospy.set_param('/human_body/right_arm/dynamic_params/upperarm_length', self.upperarm_length)
        rospy.set_param('/human_body/right_arm/dynamic_params/forearm_length', self.forearm_length)
        rospy.set_param('/human_body/right_arm/dynamic_params/neck_shoulder_length', self.neck_shoulder_length)

        # Publicar notificación de actualización
        self.parameter_update_pub.publish(True)

        return config


if __name__ == "__main__":
    try:
        node = DynamicParameterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
