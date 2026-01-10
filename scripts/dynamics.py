#!/usr/bin/env python3

import rospy
import numpy as np
import roslib.packages
from geometry_msgs.msg import Point, PoseStamped, WrenchStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromParam
from tf.transformations import quaternion_matrix
import tf2_ros

class ChainIdSolver_RNE_Py:
    def __init__(self, chain, gravity=kdl.Vector(0,0,-9.81)):
        self.n = chain.getNrOfJoints()

        # Inicializa el modelo dinámico de la cadena
        self.dyn = kdl.ChainDynParam(chain, gravity)

    def CartToJnt(self, q, jacobian=None, f_ext=None):
        """
        Compute joint torques given joint positions, with the simplification that joint velocities (qd) and accelerations (qdd) are zero.
        Returns the computed tau_out array.

        Parameters:
        - q: Joint positions
        - jacobian: Jacobian matrix (optional)
        - f_ext: External force vector humano -> robot (optional)
        """

        rospy.logdebug("Joint positions (q): %s", [q[i] for i in range(self.n)])

        # Calcular G(q) correctamente como variable local
        G = kdl.JntArray(self.n)
        self.dyn.JntToGravity(q, G)
        rospy.logdebug("Gravity torques (G): %s", [G[i] for i in range(self.n)])

        tau_out = kdl.JntArray(self.n)
        if f_ext is not None and jacobian is not None:
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

        # Parámetros del launch
        urdf_param   = rospy.get_param('~urdf_param',   'robot_description')
        initial_link = rospy.get_param('~initial_link')
        final_link   = rospy.get_param('~final_link')
        debug        = rospy.get_param('~debug', False)
        
        ###############################
        #   Cadena cinemática KDL
        ###############################

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

        # Solver del modelo dinámico inverso
        self.chain = chain
        self.n_joints = n_joints
        self.InvDynSolver = ChainIdSolver_RNE_Py(self.chain, kdl.Vector(0,0,-9.81))

        self.q = kdl.JntArray(n_joints)
        self.tau_out = kdl.JntArray(n_joints)
        self.G = kdl.JntArray(n_joints)
        self.f_human = None # Fuerza aplicada por el humano al robot (vector 6x1)

        # Solver de la jacobiana
        self.jac_solver = kdl.ChainJntToJacSolver(self.chain)
        self.jacobian = np.zeros((6, n_joints))

        # Subscribers and Publishers
        rospy.Subscriber('/right_arm/joint_states', JointState, self.q_callback)
        rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.force_cb, queue_size=1)
        rospy.Subscriber('/gripper_4f/grasp_state', Bool, self.grasp_state_callback) # Para saber si el gripper esta activo o no
    
        self.pub_dynamics = rospy.Publisher('/right_arm/joint_states_with_dynamics', JointState, queue_size=1)
        self.pub_jacobian = rospy.Publisher('/right_arm/jacobian_6x7', Float64MultiArray, queue_size=1)
        
        #####################
        #  ROS TF listener
        #####################
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lst = tf2_ros.TransformListener(self.tf_buf)

        # Otros atributos
        self.flag_gripped = False  # Estado inicial del gripper
        self.target_frame = final_link  # Frame del antebrazo del humano

        rospy.loginfo("Dynamic model human node initialized successfully.")

    def grasp_state_callback(self, msg):
        """
        Callback function to handle incoming grasp state messages.
        Updates the flag_gripped variable based on the gripper state.
        Parameters:
        - msg: Bool message indicating the gripper state (True if grasping, False otherwise)
        """

        if msg.data == True:
            if msg.data != self.flag_gripped:
                rospy.loginfo("Gripper is grasping. Using gripper position for wrist keypoint.")
            self.flag_gripped = True
        else:
            if msg.data != self.flag_gripped:
                rospy.loginfo("Gripper is not grasping. Using skeleton keypoint for wrist.")
            self.flag_gripped = False

    def force_cb(self, msg):
        """
        Callback function to handle incoming external force messages.
        Updates the external force vector used in dynamics calculations.

        Parameters:
        - msg: WrenchStamped message containing the external force data

        La frecuencia de actualización de esta fuerza debe ser alta para que la dinámica sea precisa. 
        Si es a 1KHz, no es necesario sincronizar con el callback de valores articulares q porque va a 100Hz
        """

        if self.flag_gripped:

            # Extraer la fuerza externa del mensaje
            fx = msg.wrench.force.x
            fy = msg.wrench.force.y
            fz = msg.wrench.force.z
            tx = msg.wrench.torque.x
            ty = msg.wrench.torque.y
            tz = msg.wrench.torque.z

            f_ext = np.array([fx, fy, fz, tx, ty, tz])


            try:
                tr = self.tf_buf.lookup_transform(
                    self.target_frame,               # Frame destino A
                    'fr3_K',                         # Frame de medida K. Origen
                    rospy.Time(0),                # última TF en el buffer
                    rospy.Duration(0.02) # timeout
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(1.0, f"[force_cb] TF fail {e}")
                return

            t = tr.transform.translation
            q = tr.transform.rotation
            R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]  # ^A R_B
            p = np.array([t.x, t.y, t.z], dtype=float)           # ^A p_AB


            # 3) Transformación del wrench
            F_B = f_ext[0:3]  # Fuerza en el frame B (efector final)
            T_B = f_ext[3:6]  # Torque en el frame B (

            F_A = R.dot(F_B)
            T_A = R.dot(T_B) + np.cross(p, F_A)

            self.f_human = np.hstack((F_A, T_A))
        
        else:
            self.f_human = None
            rospy.logdebug_throttle(5.0, "Force callback: Not gripped, ignoring external force.")

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
        tau_out = self.InvDynSolver.CartToJnt(
            self.q,
            jacobian=self.jacobian,
            f_ext=self.f_human
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
