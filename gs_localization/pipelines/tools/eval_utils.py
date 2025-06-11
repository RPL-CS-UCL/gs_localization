import numpy as np

def rotation_error(R_est, R_gt):
    """
    Calculate the rotation error between two rotation matrices.
    
    Parameters:
    R_est (np.ndarray): Estimated rotation matrix of shape (3, 3)
    R_gt (np.ndarray): Ground truth rotation matrix of shape (3, 3)
    
    Returns:
    float: Rotation error in degrees
    """
    R_diff = np.dot(R_est.T, R_gt)
    trace = np.trace(R_diff)
    # ensure (trace - 1) / 2 is whithin [-1, 1]
    value = (trace - 1) / 2.0
    value = np.clip(value, -1.0, 1.0)
    angle_rad = np.arccos(value)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def translation_error(T_est, T_gt):
    """
    Calculate the relative translation error.
    
    Parameters:
    T_est (np.ndarray): Estimated translation vector of shape (3,)
    T_gt (np.ndarray): Ground truth translation vector of shape (3,)
    
    Returns:
    float: Relative translation error
    """
    error = np.linalg.norm(T_est - T_gt) 
    return error