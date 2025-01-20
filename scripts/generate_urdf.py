#!/usr/bin/env python

import rospy
import roslib
import subprocess
import os
from std_msgs.msg import Bool  # Mensaje para notificaciones

class DynamicURDFGenerator:
    def __init__(self):
        rospy.init_node('dynamic_urdf_generator')

        # Almacenar los valores iniciales de los parámetros
        self.upperarm_length = rospy.get_param('/human_body/right_arm/dynamic_params/upperarm_length', 0.28)
        self.forearm_length = rospy.get_param('/human_body/right_arm/dynamic_params/forearm_length', 0.25)
        self.neck_shoulder_length = rospy.get_param('/human_body/right_arm/dynamic_params/neck_shoulder_length', 0.175)

        # Resolver la ruta del archivo Xacro
        package_path = roslib.packages.get_pkg_dir('human_articular_space')
        self.xacro_path = os.path.join(package_path, 'urdf', 'human_right_arm.xacro')

        # Suscribirse al tópico de notificación de cambios
        rospy.Subscriber("/parameter_update", Bool, self.parameter_update_callback)

        # Generar el URDF inicial
        self.generate_urdf()

    def generate_urdf(self):
        """Generar el URDF y publicarlo en el servidor de parámetros."""
        try:
            xacro_command = [
                'rosrun', 'xacro', 'xacro',
                self.xacro_path,
                f'upperarm_length:={self.upperarm_length}',
                f'forearm_length:={self.forearm_length}',
                f'neck_shoulder_length:={self.neck_shoulder_length}'
            ]
            urdf_output = subprocess.check_output(xacro_command).decode('utf-8')

            # Publicar el URDF en el servidor de parámetros
            rospy.set_param('/robot_description', urdf_output)
            rospy.loginfo("URDF regenerado y almacenado en /robot_description")

        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Error al ejecutar xacro: {e.output}")
        except Exception as e:
            rospy.logerr(f"Error inesperado: {str(e)}")

    def parameter_update_callback(self, msg):
        rospy.loginfo("Notificación de cambio de parámetros recibida.")

        new_upperarm_length = rospy.get_param('/human_body/right_arm/dynamic_params/upperarm_length')
        new_forearm_length = rospy.get_param('/human_body/right_arm/dynamic_params/forearm_length')
        new_neck_shoulder_length = rospy.get_param('/human_body/right_arm/dynamic_params/neck_shoulder_length')

        if (new_upperarm_length != self.upperarm_length or
            new_forearm_length != self.forearm_length or
            new_neck_shoulder_length != self.neck_shoulder_length):
            
            self.upperarm_length = new_upperarm_length
            self.forearm_length = new_forearm_length
            self.neck_shoulder_length = new_neck_shoulder_length

            # Regenerar el URDF
            self.generate_urdf()
        else:
            rospy.loginfo("Los parámetros no han cambiado. No se regenera el URDF.")


if __name__ == "__main__":
    try:
        generator = DynamicURDFGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
