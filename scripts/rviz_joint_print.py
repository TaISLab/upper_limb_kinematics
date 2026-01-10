#!/usr/bin/env python3

# Nodo para visualizar en RViz los valores de las articulaciones del brazo derecho.
# Es necesario tener instalado el paquete jsk_rviz_plugins


import rospy, math
from sensor_msgs.msg import JointState
from jsk_rviz_plugins.msg import OverlayText
from std_msgs.msg import ColorRGBA

JOINTS = ['right_arm_q1', 'right_arm_q2', 'right_arm_q3', 'right_arm_q4', 'right_arm_q5']

def cb(msg):
    # rospy.loginfo(f"Received joint state")
    name2pos = dict(zip(msg.name, msg.position))
    lines = []
    for j in JOINTS:
        if j in name2pos:
            deg = name2pos[j] * 180.0 / math.pi
            joint_name = j.replace('right_arm_', '')  # Suprimir el prefijo
            lines.append(f"{joint_name:16s}: {deg:7.2f}°")
    text = OverlayText()
    text.width = 140; text.height = 150
    text.left = 10; text.top = 10
    text.text_size = 18; text.line_width = 2
    text.bg_color = ColorRGBA(0,0,0,0.5)
    text.fg_color = ColorRGBA(1,1,1,1)
    text.text = "\n".join(lines) if lines else "No joints yet…"
    pub.publish(text)

if __name__ == "__main__":
    rospy.init_node("joint_overlay")
    pub = rospy.Publisher("/right_arm_description_overlay", OverlayText, queue_size=1, latch=True)
    rospy.Subscriber("/right_arm/joint_states", JointState, cb, queue_size=10)
    rospy.loginfo("Joint overlay node started")
    rospy.spin()
