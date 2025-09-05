# Angulos articulares del espacio articular del brazo

std_msgs/Header header

# Hombro
float32 q1 # elevacion frontal del brazo
float32 q2 # elevacion lateral del brazo
float32 q3 # rotacion interna/externa del hombro

# Codo
float32 q4 # flexion de codo
float32 q5 # pronacion/supinacion

# Longitudes
float32 upperarm_length
float32 forearm_length

# KP
geometry_msgs/Point right_shoulder
geometry_msgs/Point right_elbow
geometry_msgs/Point right_wrist

geometry_msgs/Point left_shoulder