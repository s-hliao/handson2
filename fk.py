import numpy as np


def forward_kinematics_RR(theta1, theta2, l1, l2):
    """
    Returns the forward kinematics for an RR robot given the joint angle positions in radians.
    """
    # NOTE: The convention is that links in 2D lie along the x-axis of the starting frame, 
    # and the joint angles are measured counter-clockwise from the x-axis of the previous link.
    
    # To-Do 1: Compute the homogeneous transformation matrices H_1_0, H_2_1, H_3_2, H_4_3, H_5_4, H_6_5
    H_1_0 = np.eye(3)    
    H_2_1 = np.eye(3)
    H_3_2 = np.eye(3)
    H_4_3 = np.eye(3)
    
    # To-Do 2: Compute the homogeneous transformation matrix H_6_0
    H_4_0 = None  # TODO
    
    return {
        'H_1_0': H_1_0,
        'H_2_1': H_2_1,
        'H_3_2': H_3_2,
        'H_4_3': H_4_3,
        'H_6_0': H_4_0
    }


def load_trajectory(path):
    """
    Loads a trajectory recorded by `goto.py --guided` and returns its RR joint angles.

    The recording is a NumPy .npz file. np.load(path) returns a dictionary-like object;
    the entries you need are:
        'theta'  shape (N, 2)  joint angles in radians, one row per sample:
                               column 0 is theta1, column 1 is theta2
        'rate'   a number      samples per second; sample k was taken at k / rate seconds

    Returns two NumPy arrays, theta1 and theta2, each of length N and aligned in time:
    theta1[k] and theta2[k] are the two joint angles at the same instant.
    """
    # To-Do 3: Load the recording at `path`
    data = None  # TODO

    # To-Do 4: Split the (N, 2) array of joint angles into one array per joint
    theta1 = None  # TODO
    theta2 = None  # TODO

    return theta1, theta2