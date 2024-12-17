Guide of types of messages:

Related to human articular space:
- HumanJointAngles: Angles between diferents vectors.
    - Elbow_angles:
        - float32 alpha: angle between vect(Shoulder-Elbow) and vect(Elbow-Wrist)
        - Probably is necessary another component to the internal/external rotation
    - Shoulder_angles:
        - float32 alpha: angle between vect(hip-shoulder) and vect(shoulder-elbow)
        - float32 beta: angle between vect(Shoulder-Other_shoulder) and vect(Shoulder-Elbow)


- HumanAnatomicSpace. Angles between keypoint projected in anatomic planes.
    - Elbow_anatomic:
        - float32 sagittal
        - float32 rotation. Internal/external rotation. I dont know how to calculate it.
    - Shoulder_anatomic:
        - float32 sagittal
        - float32 coronal
        - float32 transverse
        - float32 rotation. Internal/external rotation. I dont know how to calculate it.


