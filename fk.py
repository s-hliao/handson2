from pathlib import Path

import numpy as np


def forward_kinematics_RR(theta1, theta2, l1, l2):
    """
    Returns the forward kinematics for an RR robot given the joint angle positions in radians.
    """
    # NOTE: The convention is that links in 2D lie along the x-axis of the starting frame,
    # and the joint angles are measured counter-clockwise from the x-axis of the previous link.

    # To-Do 1: Compute the homogeneous transformation matrices H_1_0, H_2_1, H_3_2, H_4_3, H_5_4, H_6_5
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    H_1_0 = np.array([[c1, -s1, 0.0],     # rotate by theta1 at the base
                      [s1,  c1, 0.0],
                      [0.0, 0.0, 1.0]])
    H_2_1 = np.array([[1.0, 0.0, l1],     # translate along link 1
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
    H_3_2 = np.array([[c2, -s2, 0.0],     # rotate by theta2 at the elbow
                      [s2,  c2, 0.0],
                      [0.0, 0.0, 1.0]])
    H_4_3 = np.array([[1.0, 0.0, l2],     # translate along link 2
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

    # To-Do 2: Compute the homogeneous transformation matrix H_6_0
    H_4_0 = H_1_0 @ H_2_1 @ H_3_2 @ H_4_3

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
    data = np.load(path)

    # To-Do 4: Split the (N, 2) array of joint angles into one array per joint
    theta1 = data['theta'][:, 0]
    theta2 = data['theta'][:, 1]

    return theta1, theta2


# To-Do 5: List the recordings replay.py should play, by file name, in the order to play them.
# goto.py saves each one in the recordings folder as rr-<date>-<time>.npz, for example
#     TRAJECTORIES = ["rr-20260911-101500.npz", "rr-20260911-102233.npz"]
# (For now: every recording in the folder, oldest first.)
TRAJECTORIES = sorted(path.name for path in (Path(__file__).parent / "recordings").glob("rr-*.npz"))


def run_trajectory(arm, path):
    """
    Plays the recording at `path` on the arm. `arm` gives you two methods:
        arm.set_position(theta1, theta2)       moves the arm to one pose and waits
                                               until it has arrived there
        arm.servo_to_position(theta1, theta2)  streams one sample to the arm and waits
                                               until it is time for the next one
    Streaming has to start from where the arm already is, so move it to the first
    sample before servoing through the rest.
    """
    # To-Do 6: Load the recording's joint angles
    theta1, theta2 = load_trajectory(path)

    # To-Do 7: Move the arm to the first sample
    arm.set_position(theta1[0], theta2[0])

    # To-Do 8: Servo through every sample, in order
    for k in range(len(theta1)):
        arm.servo_to_position(theta1[k], theta2[k])
