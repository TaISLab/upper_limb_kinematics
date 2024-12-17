#!/usr/bin/env python

import rospy
from skeleton_3d.msg import Skeleton3D
from geometry_msgs.msg import Point
import time

def publish_skeleton():
    # Inicializar el nodo
    rospy.init_node('skeleton_publisher', anonymous=True)
    pub = rospy.Publisher('/skeleton_3D', Skeleton3D, queue_size=10)

    # Crear el mensaje Skeleton3D
    msg = Skeleton3D()
    # # Caso1.2.
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-47.5, 0, 0),     # 7 LElbow
    #     Point(47.5, 0, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(72.5, 0, 0),      # 10 RWrist
    #     Point(-22.5, 0, -50), # 11 LHip
    #     Point(22.5, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso1.3. 
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-47.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 0, 25),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 0, 50),      # 10 RWrist
    #     Point(-22.5, 0, -50), # 11 LHip
    #     Point(22.5, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso1.5. 
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-47.5, 0, 0),     # 7 LElbow
    #     Point(22.5+17.67766953, 0, 17.67766953),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 0, 0),      # 10 RWrist
    #     Point(-22.5, 0, -50), # 11 LHip
    #     Point(22.5, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso2.1.
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 25, 0),     # 7 LElbow
    #     Point(22.5, 25, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 50, 0),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso2.2.
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 17.67766953, 17.67766953),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 35.355, 35.355),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso2.3.
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 0, 25),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 0, 0),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso2.4.
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 17.67766953, -17.67766953),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 35.355, 35.355),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso3.1. verificar q3
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 25, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(-2.5, 25, 0),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso3.2. verificar q3
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(22.5, 25, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(22.5, 25, 25),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # # Caso3.3. verificar q3
    # # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(45, 22.5, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(45, 22.5, 25),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # Caso 4. q1=0 q2=90 q3=0 q4=90 NO FUNCIONA
    # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(-22.5, 0, 0),     # 5 LShoulder
    #     Point(22.5, 0, 0),       # 6 RShoulder
    #     Point(-22.5, 0, 0),     # 7 LElbow
    #     Point(45, 0, 0),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(45, 22.5, 0),      # 10 RWrist
    #     Point(-15, 0, -50), # 11 LHip
    #     Point(15, 0, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # Caso 5. q1=0 q2=0 q3=0 q4=90
    # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 22.5, 0),     # 5 LShoulder
    #     Point(0, -22.5, 0),       # 6 RShoulder
    #     Point(0, 0.1, 0),     # 7 LElbow
    #     Point(0, -22.5, -25),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(0, 2.5, -25),      # 10 RWrist
    #     Point(0, 22.5, 50), # 11 LHip
    #     Point(0, 22.5, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # Caso 6. q1=0 q2=0 q3=-90 q4=90 FUNCIONA. REQUIERE q1=q2=0 manual
    # Asignar valores a keypoints
    # msg.keypoints = [
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 22.5, 0),     # 5 LShoulder
    #     Point(0, -22.5, 0),       # 6 RShoulder
    #     Point(0, 0.1, 0),     # 7 LElbow
    #     Point(0, -22.5, -25),      # 8 RElbow
    #     Point(0, 0, 0), 
    #     Point(-25, -22.5, -25),      # 10 RWrist
    #     Point(0, 22.5, 50), # 11 LHip
    #     Point(0, 22.5, -50),  # 12 RHip
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0),
    #     Point(0, 0, 0)
    # ]

    # Caso 7. q1=0 q2=90 q3=0 q4=90 REQUIERE q1=0 manual
    # Asignar valores a keypoints
    msg.keypoints = [
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 25, 0),     # 5 LShoulder
        Point(0, -25, 0),       # 6 RShoulder
        Point(0, 0.1, 0),     # 7 LElbow
        Point(0, -50, 0),      # 8 RElbow
        Point(0, 0, 0), 
        Point(25, -50, 0),     # 10 RWrist
        Point(0, 25, -50), # 11 LHip
        Point(0, -25, -50),  # 12 RHip
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0),
        Point(0, 0, 0)
    ]

    


    
    while not rospy.is_shutdown():
        # Asignar el timestamp
        current_time = rospy.get_rostime()
        msg.header.stamp = current_time

        # Publicar el mensaje
        pub.publish(msg)

        # Imprimir en consola para verificar
        rospy.loginfo(f"Published Skeleton3D message at {current_time}")

        # Esperar un poco antes de publicar nuevamente
        rospy.sleep(0.1)  # Publica a 10Hz

if __name__ == '__main__':
    try:
        publish_skeleton()
    except rospy.ROSInterruptException:
        pass
