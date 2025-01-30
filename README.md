# human_articular_space

## Overview

Este paquete calcula la cinemática del brazo humano usando las posiciones cartesianas obtenidas del topic `/skeleton_3D` y ejecuta un modelo URDF sobre el esqueleto de puntos obtenido del sistema de visión.

## Table of Contents
- 0. URDF
- 1. IK_arm_calculator.py
- 2. pose_shoulder_TF_publisher.py
- 3. human_boy_to_joint_states_converter.py
- 4. tf2_listener.py
- 5. Lanzamiento del sistema

## 0. URDF del brazo humano
El archivo `human_right_arm.xacro` contiene un modelo URDF del brazo humano con la longitud visual del upperarm y del forearm parametrizadas. Se especifican desde el archivo de lanzamiento.

El URDF se ha modelado con 4 articulaciones rotativas y 2 prismáticas. Estas son:
- q1: Elevación frontal del hombro, en rad.
- q2: Elevación lateral del hombro, en rad.
- q3: Rotación interna/externa del hombro, en rad.
- q4: Flexión del codo, en rad.
- upperarm_length: Longitud entre el hombro y el codo, medida en m.
- forearm_length: Longitud entre el codo y la muñeca, medida en m.

Adicionalmente, el URDF consta de una quinta articulación de rotación, q5, encargada de modelar la pronación y supinación del antebrazo.

Se han utilizado articulaciones prismáticas para adecuar el modelo al sistema de visión y eliminar el error entre las posiciones cartesianas de las articulaciones. 

## 1. IK_arm_calculator.py

### Description
Este script calcula el modelo cinemático inverso del brazo humano mediante la resolución del problema geométrico.

### Funcionalidad
- Calcula el modelo cinemático inverso y publica las posiciones articulares q1, q2, q3, q4
- Publica las longitudes articulares del upperarm y del forearm, medidas como diferencias entre keypoints.

### key Components
- **Subscribers**: Receives `Skeleton3D` messages del topic `/skeleton_3D`
- **Publisher**: Publishes a `HumanBody` message containing the articular positions and link lengths en el topic `/human_body/description`

## 2. pose_shoulder_TF_publisher.py
Este nodo localiza el URDF del brazo en el kp del hombro derecho.

## 3. human_body_to_joint_states_converter.py
Este script convierte el mensaje `HumanBody` contenido en el topic `/human_body/description` en un mensaje del tipo `sensor_msgs/JointState`, apto para el nodo robot_state_publisher.

### Key components
- **Subscribers**: Se subscribe al topic `/human_body` y recibe un mensaje del tipo `HumanBody`
- **Publisher**: Publica en el topic `/joint_states` un mensaje del tipo `sensor_msgs/JointState`.

## 4. tf2_listener.py

El nodo `robot_state_publisher` calcula la Cinemática Directa del URDF y este nodo extrae las transformaciones y las posiciones cartesianas de las articulaciones

### Key components
- **Subscribers**: Se subscribe al topic `/TF` y obtiene las matrices de transformación entre base_link y las articulaciones.
- **Publisher**:
    - Publica en el topic `/kp_URDF` las posiciones cartesianas de las articulaciones del URDF.
    - Publica en el topic `/kp_vision` las posiciones cartesianas de las articulaciones obtenidas del sistema de visión. Esta información ya está disponible en otros topics, simplemente se publica aqui para un análisis más sencillo en MATLAB.

## 5. Lanzamiento del sistema

Para el uso del sistema se debe ejecutar el siguiente comando:

```bash
roslaunch human_articular_space kinematics_right_arm.launch
```
Flags:
- `use_rviz`. Visualización del URDF sobre el esqueleto articular.
- `use_tf2_listener`. Ejecuta el nodo encargado de publicar la posición cartesiana de las articulaciones.

Ejemplos de uso:

1. Uso básico con rviz
```bash
roslaunch human_articular_space kinematics_right_arm.launch
```
2. Uso básico sin rviz
```bash
roslaunch human_articular_space kinematics_right_arm.launch use_rviz:=false use_tf2_listener:=false
```
3. Uso completo: Con RViz y obteniendo las posiciones cartesianas del URDF.
```bash
roslaunch human_articular_space kinematics_right_arm.launch use_rviz:=true use_tf2_listener:=true
```

