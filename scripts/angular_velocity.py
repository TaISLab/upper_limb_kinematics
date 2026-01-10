#!/usr/bin/env python3

import rospy
import numpy as np
from collections import defaultdict
from sensor_msgs.msg import JointState
from scipy.signal import butter, sosfilt, sosfilt_zi

class AngularVelocityNode:
    def __init__(self):
        rospy.init_node('angular_velocity_node')

        # ---------------- Parámetros ----------------
        # Tasa nominal y cortes (puedes ajustarlos por rosparam)
        self.fs_nom   = rospy.get_param("~fs_hz", 30.0)   # Hz (tasa nominal de /joint_states)
        self.order    = rospy.get_param("~order", 2)       # Orden Butterworth (2–4 típico)
        self.fc_pre   = rospy.get_param("~fc_pre_hz", 8.0) # Pasa-bajo previo (posición)
        self.fc_post  = rospy.get_param("~fc_post_hz", 10.0) # Pasa-bajo posterior (velocidad)
        self.log_debug= rospy.get_param("~debug", False)
        # Si tu tasa real cambia mucho, puedes rediseñar coeficientes al vuelo (simple):
        self.redesign_if_drift = rospy.get_param("~redesign_if_drift", False)
        self.redesign_tol = rospy.get_param("~fs_drift_tol", 0.10)  # 10%

        # -------------- Filtros (forma SOS) --------------
        self.sos_pre  = butter(self.order, self.fc_pre,  btype='low', fs=self.fs_nom, output='sos')
        self.sos_post = butter(self.order, self.fc_post, btype='low', fs=self.fs_nom, output='sos')

        # Estado por articulación para filtros SOS (streaming)
        self.zi_pre = defaultdict(lambda: sosfilt_zi(self.sos_pre))
        self.zi_post= defaultdict(lambda: sosfilt_zi(self.sos_post))

        # Buffers de 3 muestras (k-1, k, k+1) — centrado
        self.names_prev = None
        self.names_curr = None
        self.names_post = None

        self.q_prev_f = None
        self.q_curr_f = None
        self.q_post_f = None

        self.t_prev = None
        self.t_curr = None
        self.t_post = None

        self.header_prev = None
        self.header_curr = None
        self.header_post = None

        # Subs / Pubs
        self.sub = rospy.Subscriber('/right_arm/joint_states', JointState, self.callback, queue_size=100)
        self.pub = rospy.Publisher('/right_arm/angular_velocity', JointState, queue_size=10)

        rospy.loginfo("AngularVelocityNode con Butterworth pre/post: fs=%.1fHz, order=%d, fc_pre=%.1fHz, fc_post=%.1fHz",
                      self.fs_nom, self.order, self.fc_pre, self.fc_post)

    # ---- util: rediseñar si fs deriva mucho ----
    def maybe_redesign_filters(self, fs_est):
        if not self.redesign_if_drift:
            return
        if fs_est <= 0:
            return
        drift = abs(fs_est - self.fs_nom) / self.fs_nom
        if drift > self.redesign_tol:
            self.fs_nom = fs_est
            self.sos_pre  = butter(self.order, self.fc_pre,  btype='low', fs=self.fs_nom, output='sos')
            self.sos_post = butter(self.order, self.fc_post, btype='low', fs=self.fs_nom, output='sos')
            # Al rediseñar, reiniciamos estados (opción simple y segura)
            self.zi_pre.clear()
            self.zi_post.clear()
            if self.log_debug:
                rospy.logwarn("Rediseñados filtros Butterworth por drift de fs. Nueva fs_nom=%.2f Hz", self.fs_nom)

    def _filter_position_sample(self, names, q_raw):
        """
        Filtra una muestra de posición (lista de floats) con Butterworth (SOS).
        Mantiene estado por 'nombre de articulación'.
        Devuelve lista 'q_f' filtrada en el mismo orden de 'names'.
        """
        q_f = []
        for name, x in zip(names, q_raw):
            y, self.zi_pre[name] = sosfilt(self.sos_pre, np.array([x], dtype=float), zi=self.zi_pre[name])
            q_f.append(float(y[0]))
        return q_f

    def _filter_velocity_sample(self, names, dq_raw):
        """
        Filtra una muestra de velocidad (lista) con Butterworth (SOS) post-derivación.
        """
        dq_f = []
        for name, x in zip(names, dq_raw):
            y, self.zi_post[name] = sosfilt(self.sos_post, np.array([x], dtype=float), zi=self.zi_post[name])
            dq_f.append(float(y[0]))
        return dq_f

    def callback(self, msg: JointState):
        # Tiempo del mensaje (segundos)
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        names_in = list(msg.name) if msg.name else [f'joint_{i}' for i in range(len(msg.position))]
        q_in = list(msg.position)

        # Estimación simple de fs para (opcional) rediseño
        if self.t_curr is not None and self.t_prev is not None:
            # periodo centrado (k-1 a k) o (k a k+1); tomamos el último
            dt_est = t - self.t_curr
            if dt_est > 1e-6:
                self.maybe_redesign_filters(1.0/dt_est)

        # ---- 1) Filtrar posición entrante (k+1) ----
        q_post_f = self._filter_position_sample(names_in, q_in)

        # Actualizamos buffers (rotamos k-1 <- k, k <- k+1)
        self.names_prev, self.names_curr, self.names_post = self.names_curr, self.names_post, names_in
        self.q_prev_f,   self.q_curr_f,   self.q_post_f   = self.q_curr_f,   self.q_post_f,   q_post_f
        self.t_prev,     self.t_curr,     self.t_post     = self.t_curr,     self.t_post,     t
        self.header_prev,self.header_curr,self.header_post= self.header_curr,self.header_post,msg.header

        # Necesitamos las 3 muestras (k-1, k, k+1) para derivada centrada en k
        if self.q_prev_f is None or self.q_curr_f is None or self.q_post_f is None:
            return

        # Aseguramos correspondencia de orden de nombres entre las 3 muestras:
        # Asumimos que el orden de joints es estable; si no lo fuera, podrías mapear por nombre con dicts.

        # ---- 2) Diferencia centrada (sobre posiciones YA filtradas) ----
        delta_t = self.t_post - self.t_prev
        if delta_t <= 0:
            rospy.logwarn_throttle(1.0, "Delta t no válido (<=0); no publico velocidades.")
            return

        # dv = (q(k+1) - q(k-1)) / (2*dt)  con dt = t(k+1) - t(k-1)
        dq_raw = [ (qp - qm) / (2.0 * delta_t) for qp, qm in zip(self.q_post_f, self.q_prev_f) ]

        # ---- 3) Filtrado posterior de velocidad ----
        dq_f = self._filter_velocity_sample(self.names_curr, dq_raw)

        # --- Forzar velocidad cero en upperarm_length y forearm_length ---
        if self.names_curr is not None:
            for idx, name in enumerate(self.names_curr):
                if name in ("upperarm_length", "forearm_length"):
                    dq_f[idx] = 0.0

        if self.log_debug:
            rospy.loginfo_throttle(1.0, f"vel (centrada, filtrada) = {np.array(dq_f)}")

        # ---- 4) Publicar: centramos en 'k' (header_curr) ----
        out = JointState()
        out.header = self.header_curr if self.header_curr is not None else msg.header
        out.name = self.names_curr if self.names_curr is not None else names_in

        # Posición a tiempo 'k': usamos la posición FILTRADA del centro (mejor coherencia)
        out.position = self.q_curr_f
        out.velocity = dq_f
        # Esfuerzo: si quieres, puedes copiar el del mensaje 'k', pero aquí no lo conservamos.
        # out.effort = [...]

        self.pub.publish(out)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = AngularVelocityNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
