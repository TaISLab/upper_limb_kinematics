# Angulos articulares del espacio articular del brazo

std_msgs/Header header

# Hombro
float32 q1 # elevacion frontal del brazo
float32 q2 # elevacion lateral del brazo
float32 q3 # rotacion interna/externa del hombro

# Codo
float32 q4 # flexion de codo: 0 -> Brazo extendido, 150 -> Antebrazo cerca del cuerpo
float32 q5 # Positivo - supinación, negativo - pronación, neutro - 0º

# Longitudes
float32 upperarm_length
float32 forearm_length