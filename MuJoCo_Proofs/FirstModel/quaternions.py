import numpy as np
from math import radians

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to a quaternion.

    :param roll: Rotation around the X-axis in degrees.
    :param pitch: Rotation around the Y-axis in degrees.
    :param yaw: Rotation around the Z-axis in degrees.
    :return: A numpy array representing the quaternion [w, x, y, z].
    """
    roll_rad = radians(roll)
    pitch_rad = radians(pitch)
    yaw_rad = radians(yaw)
    
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

# Example usage
roll_deg = 0
pitch_deg = 45
yaw_deg = 0

quaternion = euler_to_quaternion(roll_deg, pitch_deg, yaw_deg)
print("Quaternion:", np.round(quaternion,3))
