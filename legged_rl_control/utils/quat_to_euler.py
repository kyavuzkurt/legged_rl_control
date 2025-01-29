import numpy as np
def quat_to_euler(q):
    """Convert quaternion [w,x,y,z] to Euler angles (roll, pitch, yaw)"""
    # Implementation of the conversion
    roll = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]))
    yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return roll, pitch, yaw