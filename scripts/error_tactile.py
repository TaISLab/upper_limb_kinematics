#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import message_filters
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Float64

class ErrorComputerNode:
    def __init__(self):
        rospy.init_node('tactile_optitrack_error_node')
        
        # --- Configuración ---
        self.target_frame = "fr3_EE"  # Frame local de la garra/efector final
        self.queue_size = 10
        self.slop = 0.1  # Tolerancia de tiempo en segundos para la sincronización

        # --- TF Buffer ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ================= PUBLISHERS =================
        
        # --- Tactile vs Optitrack ---
        self.pub_err_elbow_tac_vec = rospy.Publisher('/error/tactile/elbow/vector_local', Vector3Stamped, queue_size=10)
        self.pub_err_elbow_tac_euc = rospy.Publisher('/error/tactile/elbow/euclidean', Float64, queue_size=10)
        
        self.pub_err_wrist_tac_vec = rospy.Publisher('/error/tactile/wrist/vector_local', Vector3Stamped, queue_size=10)
        self.pub_err_wrist_tac_euc = rospy.Publisher('/error/tactile/wrist/euclidean', Float64, queue_size=10)

        # --- Dummy vs Optitrack (NUEVO) ---
        self.pub_err_elbow_dum_vec = rospy.Publisher('/error/dummy/elbow/vector_local', Vector3Stamped, queue_size=10)
        self.pub_err_elbow_dum_euc = rospy.Publisher('/error/dummy/elbow/euclidean', Float64, queue_size=10)
        
        self.pub_err_wrist_dum_vec = rospy.Publisher('/error/dummy/wrist/vector_local', Vector3Stamped, queue_size=10)
        self.pub_err_wrist_dum_euc = rospy.Publisher('/error/dummy/wrist/euclidean', Float64, queue_size=10)

        # ================= SUBSCRIBERS =================

        # 1. OPTITRACK (Ground Truth)
        self.sub_opt_elbow = message_filters.Subscriber("/optitrack/elbow", PointStamped)
        self.sub_opt_wrist = message_filters.Subscriber("/optitrack/wrist", PointStamped)

        # 2. TACTILE (Estimation A)
        self.sub_tac_elbow = message_filters.Subscriber("/tactile/elbow", PointStamped)
        self.sub_tac_wrist = message_filters.Subscriber("/tactile/wrist", PointStamped)

        # 3. DUMMY (Estimation B - NUEVO)
        self.sub_dum_elbow = message_filters.Subscriber("/dummy/elbow", PointStamped)
        self.sub_dum_wrist = message_filters.Subscriber("/dummy/wrist", PointStamped)

        # ================= SYNCHRONIZERS =================

        # --- Sync: Optitrack & Tactile (ELBOW) ---
        self.sync_elbow_tac = message_filters.ApproximateTimeSynchronizer(
            [self.sub_opt_elbow, self.sub_tac_elbow], queue_size=self.queue_size, slop=self.slop
        )
        self.sync_elbow_tac.registerCallback(self.callback_elbow_tactile)

        # --- Sync: Optitrack & Tactile (WRIST) ---
        self.sync_wrist_tac = message_filters.ApproximateTimeSynchronizer(
            [self.sub_opt_wrist, self.sub_tac_wrist], queue_size=self.queue_size, slop=self.slop
        )
        self.sync_wrist_tac.registerCallback(self.callback_wrist_tactile)

        # --- Sync: Optitrack & Dummy (ELBOW) - NUEVO ---
        self.sync_elbow_dum = message_filters.ApproximateTimeSynchronizer(
            [self.sub_opt_elbow, self.sub_dum_elbow], queue_size=self.queue_size, slop=self.slop
        )
        self.sync_elbow_dum.registerCallback(self.callback_elbow_dummy)

        # --- Sync: Optitrack & Dummy (WRIST) - NUEVO ---
        self.sync_wrist_dum = message_filters.ApproximateTimeSynchronizer(
            [self.sub_opt_wrist, self.sub_dum_wrist], queue_size=self.queue_size, slop=self.slop
        )
        self.sync_wrist_dum.registerCallback(self.callback_wrist_dummy)

        rospy.loginfo(f"Error Computer Node Started. Projecting errors (Tactile & Dummy) to {self.target_frame}")

    def transform_point(self, point_msg, target_frame):
        """
        Transforma un PointStamped al frame deseado usando TF2.
        Retorna: numpy array [x, y, z] o None si falla.
        """
        try:
            # Intentamos transformación precisa en el tiempo
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point_msg.header.frame_id,
                point_msg.header.stamp,
                rospy.Duration(0.1)
            )
            p_transformed = tf2_geometry_msgs.do_transform_point(point_msg, transform)
            return np.array([p_transformed.point.x, p_transformed.point.y, p_transformed.point.z])

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # Fallback a la última transformación disponible si falla la sincronización exacta
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    point_msg.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(0.1)
                )
                p_transformed = tf2_geometry_msgs.do_transform_point(point_msg, transform)
                return np.array([p_transformed.point.x, p_transformed.point.y, p_transformed.point.z])
            except Exception as e2:
                rospy.logwarn_throttle(2.0, f"TF Error transforming to {target_frame}: {e2}")
                return None

    def compute_and_publish(self, p_gt, p_est, header, pub_vec, pub_euc):
        """
        Calcula error (Estimación - GroundTruth) y publica sin valor absoluto.
        p_gt: Ground Truth (Optitrack)
        p_est: Estimación (Tactile o Dummy)
        """
        # Transformar ambos puntos al frame del End-Effector (fr3_EE)
        v_gt = self.transform_point(p_gt, self.target_frame)
        v_est = self.transform_point(p_est, self.target_frame)

        if v_gt is None or v_est is None:
            return

        # Calcular Error Vectorial: Error = Estimacion - Realidad
        # Mantenemos el signo (NO usamos abs)
        error_vec = v_est - v_gt
        
        # Calcular Error Euclidiano (Magnitud, siempre positivo)
        error_norm = np.linalg.norm(error_vec)

        # 1. Publicar Componentes (Vector3Stamped) - CON SIGNO
        msg_vec = Vector3Stamped()
        msg_vec.header.stamp = header.stamp
        msg_vec.header.frame_id = self.target_frame
        msg_vec.vector.x = error_vec[0]
        msg_vec.vector.y = error_vec[1]
        msg_vec.vector.z = error_vec[2]
        pub_vec.publish(msg_vec)

        # 2. Publicar Euclídeo (Float64)
        pub_euc.publish(Float64(error_norm))

    # --- CALLBACKS TACTILE ---
    def callback_elbow_tactile(self, opt_msg, tac_msg):
        self.compute_and_publish(
            opt_msg, tac_msg, opt_msg.header, 
            self.pub_err_elbow_tac_vec, self.pub_err_elbow_tac_euc
        )

    def callback_wrist_tactile(self, opt_msg, tac_msg):
        self.compute_and_publish(
            opt_msg, tac_msg, opt_msg.header, 
            self.pub_err_wrist_tac_vec, self.pub_err_wrist_tac_euc
        )

    # --- CALLBACKS DUMMY ---
    def callback_elbow_dummy(self, opt_msg, dum_msg):
        self.compute_and_publish(
            opt_msg, dum_msg, opt_msg.header, 
            self.pub_err_elbow_dum_vec, self.pub_err_elbow_dum_euc
        )

    def callback_wrist_dummy(self, opt_msg, dum_msg):
        self.compute_and_publish(
            opt_msg, dum_msg, opt_msg.header, 
            self.pub_err_wrist_dum_vec, self.pub_err_wrist_dum_euc
        )

if __name__ == '__main__':
    try:
        node = ErrorComputerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass