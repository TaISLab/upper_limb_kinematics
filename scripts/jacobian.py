#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cálculo de la Jacobiana 6x7 del brazo derecho humano con PyKDL

- Asume que el URDF está cargado en /robot_description.
- Lee un JointState con 7 articulaciones y publica un Float64MultiArray 6x7.
- Cada fila i ∈ [0..5] corresponde a:
    0: ∂v_x/∂q̇
    1: ∂v_y/∂q̇
    2: ∂v_z/∂q̇
    3: ∂ω_x/∂q̇
    4: ∂ω_y/∂q̇
    5: ∂ω_z/∂q̇
- Cada columna j ∈ [0..6] corresponde a la articulación j (q̇_j).
"""

import rospy
import numpy as np
import PyKDL
from kdl_parser_py.urdf import treeFromParam
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

# Variables globales (se inicializan en main)
jac_solver = None
n_joints   = 0
debug      = False

def cb_js(msg: JointState):
    global jac_solver, n_joints, debug

    # 1) Asegurarnos de que el JointState tiene el mismo número de articulaciones
    if len(msg.position) != n_joints:
        rospy.logwarn_throttle(
            5,
            "Se esperaban %d articulaciones, pero /joint_states trae %d. Ignorando este mensaje.",
            n_joints, len(msg.position)
        )
        return

    # 2) Construir JntArray para posiciones y velocidades
    q = PyKDL.JntArray(n_joints)
    q_dot = PyKDL.JntArray(n_joints)
    for i in range(n_joints):
        q[i]     = msg.position[i]
        q_dot[i] = msg.velocity[i] if len(msg.velocity) == n_joints else 0.0

    # 3) Calcular la Jacobiana original (6×n_joints) con PyKDL
    jac = PyKDL.Jacobian(n_joints)
    jac_solver.JntToJac(q, jac)

    # 4) Volcarla a un numpy array de tamaño (6 × n_joints)
    J_canonica = np.zeros((6, n_joints))
    for i in range(6):         # filas = componentes espaciales (v_x,v_y,v_z, ω_x,ω_y,ω_z)
        for j in range(n_joints):  # columnas = articulaciones 0..6
            J_canonica[i, j] = jac[i, j]

    # 5) Publicar la matriz 6×7 como Float64MultiArray
    arr = Float64MultiArray()
    arr.layout.dim = [
        MultiArrayDimension(label="rows", size=6,         stride=n_joints * 6),
        MultiArrayDimension(label="cols", size=n_joints,  stride=n_joints)
    ]
    arr.data = J_canonica.flatten().tolist()
    pub.publish(arr)

    # 6) [Opcional] Si quieres ver el twist (v,ω), multiplica J_canonica · q_dot
    if debug:
        twist = np.zeros(6)
        q_dot = np.array([10, 0, 0, 0, 0, 0, 0])  # Ejemplo de velocidades articulares

        for i in range(6):
            for j in range(n_joints):
                twist[i] += J_canonica[i, j] * q_dot[j]

        v     = twist[0:3]
        omega = twist[3:6]
        rospy.loginfo("Jacobiana 6×%d:\n%s", n_joints, J_canonica)
        rospy.loginfo("Twist (v,ω) = [%s | %s]", np.array2string(v, precision=4), np.array2string(omega, precision=4))


if __name__ == "__main__":
    rospy.init_node("jacobian_node_6x7")

    # 1) Leer parámetros del Launch
    urdf_param   = rospy.get_param('~urdf_param',   'robot_description')
    initial_link = rospy.get_param('~initial_link')
    final_link   = rospy.get_param('~final_link')
    topic_js     = rospy.get_param('~joint_states_topic')
    debug        = rospy.get_param('~debug', False)

    # 2) Cargar URDF y construir KDL Tree
    ok, tree = treeFromParam(urdf_param)
    if not ok:
        rospy.logerr("Error al cargar el URDF desde '%s'", urdf_param)
        exit(1)

    # 3) Obtener la cadena base → efector y número de articulaciones
    chain = tree.getChain(initial_link, final_link)
    n_joints = chain.getNrOfJoints()
    if n_joints == 0:
        rospy.logerr("La KDL chain '%s'→'%s' no tiene articulaciones (n_joints=0).", initial_link, final_link)
        exit(1)
    rospy.loginfo("Número de articulaciones en la cadena: %d", n_joints)

    # 4) Crear solver de Jacobiana
    jac_solver = PyKDL.ChainJntToJacSolver(chain)

    # 5) Inicializar Publisher y Subscriber
    pub = rospy.Publisher('jacobian_6x7', Float64MultiArray, queue_size=1)
    rospy.Subscriber(topic_js, JointState, cb_js)

    rospy.loginfo("Nodo 'jacobian_node_6x7' iniciado. Esperando /joint_states…")
    rospy.spin()
