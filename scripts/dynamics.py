#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
from geometry_msgs.msg import Point, PoseStamped
from scipy.spatial.transform import Rotation as R
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState

from std_msgs.msg import Bool, Int32, Float64MultiArray, MultiArrayDimension
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromParam

# Import custom message types
from skeleton_3d.msg import Skeleton3D  # Mensaje con info de los KP
from upper_limb_kinematics.msg import RightArm, RightArmState # Mensaje a publicar


FLAG_GRIPPED=False

# TODO: Subscribirse al grasp state y al F_Ext del franka
# TODO: Revisar la dirección de la gravedad respecto al modelo URDF

"""
Para probar el switch entre gripper y skeleton:
rostopic pub /grasp_state std_msgs/Bool "data: True"

"""

class ChainIdSolver_RNE_Py:
    def __init__(self, chain, gravity=kdl.Vector(0,0,-9.81)):
        self.n = chain.getNrOfJoints()

        # Inicializa el modelo dinámico de la cadena
        self.dyn = kdl.ChainDynParam(chain, gravity)


    def CartToJnt(self, q, forearm_grasp=False, jacobian=None, f_ext=None):
        """
        Compute joint torques given joint positions, with the simplification that joint velocities (qd) and accelerations (qdd) are zero.
        Returns the computed tau_out array.

        Parameters:
        - q: Joint positions
        - forearm_grasp: Boolean indicating if forearm is grasped
        - jacobian: Jacobian matrix (optional)
        - f_ext: External force vector (optional)
        """

        rospy.logdebug("CartToJnt called with forearm_grasp=%s", forearm_grasp)
        rospy.logdebug("Joint positions (q): %s", [q[i] for i in range(self.n)])

        # Calcular G(q) correctamente como variable local
        G = kdl.JntArray(self.n)
        self.dyn.JntToGravity(q, G)
        rospy.logdebug("Gravity torques (G): %s", [G[i] for i in range(self.n)])

        tau_out = kdl.JntArray(self.n)
        if forearm_grasp and f_ext is not None and jacobian is not None:
            rospy.logdebug("External force (f_ext): %s", f_ext)
            rospy.logdebug("Jacobian: %s", jacobian)
            # Suponiendo que f_ext es un vector numpy y jacobian es una matriz numpy
            tau_ext = jacobian.T @ f_ext
            rospy.logdebug("External torque contribution (jacobian.T @ f_ext): %s", tau_ext)
            for i in range(self.n):
                tau_out[i] = G[i] + tau_ext[i]
        else:
            for i in range(self.n):
                tau_out[i] = G[i]
        rospy.logdebug("Output torques (tau_out): %s", [tau_out[i] for i in range(self.n)])
        return tau_out

class ROSInterface:

    def __init__(self):
        """
        Initialize the ROS node, set up subscribers, and prepare the 3D skeleton tracker.
        """

        # Inicializar nodo ROS
        rospy.init_node('dynamic_model_human_arm', anonymous=False)

        # 1) Leer parámetros del Launch
        urdf_param   = rospy.get_param('~urdf_param',   'robot_description')
        initial_link = rospy.get_param('~initial_link')
        final_link   = rospy.get_param('~final_link')
        debug        = rospy.get_param('~debug', False)

        # Cargar URDF y construir KDL Tree
        ok, tree = treeFromParam(urdf_param)
        if not ok:
            rospy.logerr("Error al cargar el URDF desde '%s'", urdf_param)
            exit(1)

        # Obtener la cadena base -> efector y número de articulaciones
        chain = tree.getChain(initial_link, final_link)
        n_joints = chain.getNrOfJoints()
        if n_joints == 0:
            rospy.logerr("La KDL chain '%s'->'%s' no tiene articulaciones (n_joints=0).", initial_link, final_link)
            exit(1)
        rospy.loginfo("Número de articulaciones en la cadena: %d", n_joints)

        # Atributos del modelo dinámico
        self.chain = chain
        self.n_joints = n_joints
        self.solver = ChainIdSolver_RNE_Py(self.chain, kdl.Vector(0,0,-9.81))

        self.q = kdl.JntArray(n_joints)
        self.tau_out = kdl.JntArray(n_joints)
        self.G = kdl.JntArray(n_joints)
        self.f_ext = None # Placeholder for external force vector

        # Jacobian solver
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacobian = np.zeros((6, n_joints))

        # Subscribers and Publishers
        rospy.Subscriber('/right_arm/joint_states', JointState, self.q_callback)
        # rospy.Subscriber('/grasp_state', Bool, self.grasp_state_callback)
        
        # TODO: Subscribirse a la fuerza externa cuando esté disponible
        # rospy.Subscriber('/right_arm/external_force', kdl.Wrench, self.external_force_callback)
        
        self.pub_dynamics = rospy.Publisher('/right_arm/joint_states_with_dynamics', JointState, queue_size=1)
        self.pub_jacobian = rospy.Publisher('/right_arm/jacobian_6x7', Float64MultiArray, queue_size=1)

        rospy.loginfo("Dynamic model human node initialized successfully.")

    def q_callback(self, msg):
        """
        Callback function to handle incoming joint state messages.
        Updates the joint positions and computes the dynamics.

        Parameters:
        - msg: JointState message containing the current joint states
        """

        # Validar que llegue la cantidad correcta de posiciones articulares
        if len(msg.position) < self.n_joints:
            rospy.logerr(f"Received joint state with insufficient positions: expected {self.n_joints}, got {len(msg.position)}")
            return

        
        # Almacenar las posiciones articulares
        for i in range(self.n_joints):
            self.q[i] = msg.position[i]

        # Bloquear articulaciones 2 y 5 a valores fijos. Estas son las articulaciones prismaticas ficticias.
        # TODO: Que los valores los lea del launch
        self.q[2] = 0.3  # Valor fijo deseado
        self.q[5] = 0.3  # Valor fijo deseado

        # Calcular la jacobiana y almacenarla en self.jacobian
        jac_kdl = kdl.Jacobian(self.n_joints)
        self.jac_solver.JntToJac(self.q, jac_kdl)
        for i in range(6):
            for j in range(self.n_joints):
                self.jacobian[i, j] = jac_kdl[i, j]

        # Publicar la jacobiana como Float64MultiArray (igual que jacobian.py)
        arr = Float64MultiArray()
        arr.layout.dim = [
            MultiArrayDimension(label="rows", size=6,         stride=self.n_joints * 6),
            MultiArrayDimension(label="cols", size=self.n_joints,  stride=self.n_joints)
        ]
        arr.data = self.jacobian.flatten().tolist()
        self.pub_jacobian.publish(arr)


        # Dinámica inversa (pares según q y F_ext)
        forearm_grasp = False # TODO: Actualizar según el estado real del gripper
        tau_out = self.solver.CartToJnt(
            self.q,
            forearm_grasp=False,
            jacobian=self.jacobian,
            f_ext=self.f_ext
        )
        
        rospy.logdebug("Calculated dynamics (tau_out)")

        # Poner indices 2 y 5 a 0 (articulaciones ficticias)
        tau_out[2] = 0.0
        tau_out[5] = 0.0

        # Usar tau_out calculado
        if tau_out is not None: # OK
            # Publicar los torques calculados en un nuevo JointState
            joint_state_with_dynamics = JointState()
            joint_state_with_dynamics.header = msg.header
            joint_state_with_dynamics.name = msg.name
            joint_state_with_dynamics.position = msg.position
            joint_state_with_dynamics.velocity = msg.velocity
            joint_state_with_dynamics.effort = [tau_out[i] for i in range(self.n_joints)]

            self.pub_dynamics.publish(joint_state_with_dynamics)
        else: 
            rospy.logerr("Error calculating dynamics: tau_out is None")
            # Publica la posición sin esfuerzos si hay error
            joint_state_with_dynamics = JointState()
            joint_state_with_dynamics.header = msg.header
            joint_state_with_dynamics.name = msg.name
            joint_state_with_dynamics.position = msg.position
            joint_state_with_dynamics.velocity = msg.velocity
            joint_state_with_dynamics.effort = [0.0] * self.n_joints
            self.pub_dynamics.publish(joint_state_with_dynamics)


if __name__ == '__main__':
    try:
        # Create an instance of the ROSInterface and start listening for messages
        interface = ROSInterface()

        def shutdown_callback():
            """
            Handles the shutdown of the ROS node, ensuring clean closure of resources.
            """
            rospy.loginfo("Shutting down dynamic_model_human_arm node...")

        # Register a shutdown hook
        rospy.on_shutdown(shutdown_callback)

        rospy.spin()  # Keep the node running

    except rospy.ROSInterruptException:
        pass
