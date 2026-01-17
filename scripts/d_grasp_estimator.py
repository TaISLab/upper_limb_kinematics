#!/usr/bin/env python3
import rospy
import numpy as np
from skeleton_3d.msg import Skeleton3D
from franka_msgs.msg import FrankaState

class DGraspEstimator:
    def __init__(self):
        rospy.init_node('simple_grasp_estimator')
        rospy.loginfo("DGraspEstimator (Simplified) started.")

        # --- Suscriptores ---
        rospy.Subscriber("/skeleton_3D", Skeleton3D, self.skeleton_callback)
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.franka_callback)

        # --- Parámetros de tiempo (Cargados del YAML) ---
        # Asegúrate de que el namespace '/exp_optitrack_25' coincida con tu .yaml o launch file
        self.bag_start_time = rospy.get_param('/exp_optitrack_25/rosbag_start_time', 0.0)

        t_placement_rel = rospy.get_param('/exp_optitrack_25/phases/t_placement', 0.0)
        t_occlusion_rel = rospy.get_param('/exp_optitrack_25/phases/t_occlusion_start', 0.0)
        t_grasp_rel = rospy.get_param('/exp_optitrack_25/phases/t_grasp', 0.0)

        # --- Cálculo de tiempos ABSOLUTOS (Epoch) ---
        if self.bag_start_time == 0.0:
            rospy.logwarn("!!! CUIDADO: rosbag_start_time es 0.0. Usando tiempos relativos. Si reproduces un bag, esto fallará.")
            self.target_t_placement = t_placement_rel
            self.target_t_occlusion = t_occlusion_rel
            self.target_t_grasp = t_grasp_rel
        else:
            self.target_t_placement = self.bag_start_time + t_placement_rel
            self.target_t_occlusion = self.bag_start_time + t_occlusion_rel
            self.target_t_grasp = self.bag_start_time + t_grasp_rel

            rospy.loginfo(f"--- VENTANA DE MEDICIÓN (Epoch) ---")
            rospy.loginfo(f"Inicio Promedio (Placement): {self.target_t_placement:.2f}")
            rospy.loginfo(f"Fin Promedio (Occlusion):    {self.target_t_occlusion:.2f}")
            rospy.loginfo(f"Captura Robot (Grasp):       {self.target_t_grasp:.2f}")
            rospy.loginfo(f"-----------------------------------")

        # --- Variables de estado ---
        self.wrist_placement_pos = None # Almacena el resultado del promedio
        self.ee_grasp_pos = None        # Almacena la posición del robot
        self.measurement_done = False   # Flag para finalizar
        
        # --- Variables acumuladoras para el promedio ---
        self.wrist_sum = np.array([0.0, 0.0, 0.0])
        self.wrist_count = 0

    def skeleton_callback(self, msg):
        """
        Calcula el promedio de la posición de la muñeca filtrando valores NaN.
        """
        if self.measurement_done or self.wrist_placement_pos is not None:
            return

        current_time = msg.header.stamp.to_sec()

        # CASO A: Estamos DENTRO de la ventana temporal (Acumular datos)
        if current_time >= self.target_t_placement and current_time < self.target_t_occlusion:
            try:
                # Índice 10 = Right Wrist (según tu código anterior)
                rwrist = msg.keypoints[10]
                
                # --- CORRECCIÓN CRÍTICA: FILTRAR NaNs ---
                # Si el tracker pierde el punto momentáneamente, suele enviar NaNs.
                # Si sumamos un NaN, corrompemos todo el promedio.
                if np.isnan(rwrist.x) or np.isnan(rwrist.y) or np.isnan(rwrist.z):
                    return # Saltamos este frame defectuoso

                current_point = np.array([rwrist.x, rwrist.y, rwrist.z])
                
                # Acumular solo si es válido
                self.wrist_sum += current_point
                self.wrist_count += 1
                
            except IndexError:
                pass

        # CASO B: Salimos de la ventana y tenemos datos válidos acumulados
        elif current_time >= self.target_t_occlusion:
            
            if self.wrist_count > 0:
                # Calcular la media
                self.wrist_placement_pos = self.wrist_sum / self.wrist_count
                
                rospy.logwarn(f"\n>>> [1/2] PROMEDIO MUÑECA CALCULADO")
                rospy.loginfo(f"Ventana finalizada en t={current_time:.2f}")
                rospy.loginfo(f"Muestras VÁLIDAS promediadas: {self.wrist_count}")
                rospy.loginfo(f"Posición Media Wrist: {self.wrist_placement_pos}\n")
                
                # Intentar cálculo final por si el robot ya se capturó
                self.calculate_final_distance()
            
            elif self.wrist_placement_pos is None:
                # Si llegamos aquí y count es 0, es que todos los datos eran NaN o no llegaron mensajes
                rospy.logwarn_throttle(2.0, "Saliendo de la ventana sin muestras válidas (Count=0). Esperando datos...")
    def franka_callback(self, msg):
        """
        Captura la posición del robot en el instante t_grasp.
        """
        if self.measurement_done or self.ee_grasp_pos is not None:
            return

        current_time = msg.header.stamp.to_sec()

        # --- CAPTURA END EFFECTOR ---
        if current_time >= self.target_t_grasp:
            # O_T_EE es column-major. Índices 12, 13, 14 son x, y, z de la traslación.
            self.ee_grasp_pos = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
            
            rospy.logwarn(f"\n>>> [2/2] GRIPPER CAPTURADO")
            rospy.loginfo(f"Tiempo: {current_time:.4f} (Target: {self.target_t_grasp:.4f})")
            rospy.loginfo(f"Posición EE: {self.ee_grasp_pos}\n")

            # Intentar cálculo final
            self.calculate_final_distance()

    def calculate_final_distance(self):
        """
        Realiza el cálculo de distancia si ambos datos están listos.
        """
        if self.wrist_placement_pos is not None and self.ee_grasp_pos is not None:
            distance = np.linalg.norm(self.wrist_placement_pos - self.ee_grasp_pos)
            
            rospy.logwarn("************************************************************")
            rospy.logwarn(f" MEDICIÓN COMPLETADA")
            rospy.logwarn(f" Distancia Euclídea: {distance:.5f} metros")
            rospy.logwarn("************************************************************")
            
            self.measurement_done = True
            # Opcional: Cerrar el nodo si solo querías hacer esto una vez
            # rospy.signal_shutdown("Medición terminada")

    def run(self):
        """
        Mantiene el nodo vivo esperando callbacks.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DGraspEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass