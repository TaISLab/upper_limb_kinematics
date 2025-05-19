#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cálculo del jacobiano del brazo derecho del humano

Para tal fin se utiliza la librería PyKDL.
Se utiliza el tipo Float64MultiArray que no tiene un tamaño establecido. 
Esto permite que la jacobiana tenga un tamaño de filas variable.
Componentes del tipo Float64MultiArray:
- rows: Filas. Tamaño N
- cols: Columnas. Tamaño 6
- data: Contiene todos los datos de la matriz concatenados

Es necesario reconstruir la matriz J en la subscripción.

##### Código para el nodo de subscripción #####

from std_msgs.msg import Float64MultiArray

def cb_jac(msg: Float64MultiArray):
    # Obtener dimensiones
    rows = msg.layout.dim[0].size   # 6
    cols = msg.layout.dim[1].size   # n

    # Reconstrucción de la matriz
    J = np.array(msg.data).reshape(rows, cols)

    # Acceder a un elemento (i,j)
    i, j = 2, 3
    valor = J[i, j]
    rospy.loginfo("J[%d,%d] = %f", i, j, valor)

#################################################
"""

import rospy
import numpy as np
import PyKDL
from kdl_parser_py.urdf import treeFromParam
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension


def cb_js(msg):
    n = len(msg.position)
    q = PyKDL.JntArray(n)
    for i, qi in enumerate(msg.position):
        q[i] = qi

    jac = PyKDL.Jacobian(n)
    jac_solver.JntToJac(q, jac)

    # Convertir a numpy 6×n
    J = np.zeros((6, n))
    for i in range(6):
        for j in range(n):
            J[i, j] = jac[i, j]

    # Publicar como Float64MultiArray. Este tipo funciona como:
    # Se empaquetan todas las componentes de manera sucesiva y cómo se conoce el shape de la matriz
    # tras la subscripción se reconstruye la matriz
    # ¿Por qué así? Porque tenemos una matriz de tamaño 6xN. Tenemos 6 componentes (Jlx, Jly, Jlz, Jwx, Jwy, Jwz). Tenemos N articulaciones.
    arr = Float64MultiArray()
    arr.layout.dim = [
        MultiArrayDimension(label="rows", size=6,  stride=6*n),
        MultiArrayDimension(label="cols", size=n,  stride=n)
    ]
    arr.data = J.flatten().tolist()
    pub.publish(arr)

    # Debug: imprimir en consola si corresponde
    if debug:
        rospy.loginfo("Jacobiana %dx%d:\n%s", 6, n, J)

if __name__=="__main__":
    # inicializa el nodo
    rospy.init_node("jacobian_node")

    # Leer parámetros del Launch
    urdf_param        = rospy.get_param('~urdf_param', 'robot_description')
    initial_link         = rospy.get_param('~initial_link')
    final_link          = rospy.get_param('~final_link')
    topic_js          = rospy.get_param('~joint_states_topic')
    debug             = rospy.get_param('~debug', False)


    # Carga el URDF
    ok, tree = treeFromParam(urdf_param)
    if not ok:
        rospy.logerr("Error al cargar el URDF desde '%s' ", urdf_param)
        exit(1)

    # Solver
    chain      = tree.getChain(initial_link, final_link)
    jac_solver = PyKDL.ChainJntToJacSolver(chain)

    # Publisher
    pub = rospy.Publisher('jacobian', Float64MultiArray, queue_size=1)

    # Subscriber
    rospy.Subscriber(topic_js, JointState, cb_js)

    rospy.spin()