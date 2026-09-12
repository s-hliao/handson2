import numpy as np

from robot_info import robot_info


def forward_kinematics_RR(theta1, theta2):
    """
    Returns the forward kinematics for an RR robot given the joint angle positions in radians.

    The link lengths are the arm's own, from `robot_info.py`.
    """
    # NOTE: The convention is that links in 2D lie along the x-axis of the starting frame,
    # and the joint angles are measured counter-clockwise from the x-axis of the previous link.
    l1, l2 = robot_info()['link_lengths']

    # To-Do 1: Compute the homogeneous transformation matrices H_1_0, H_2_1, H_3_2,
    # H_4_3, then compose the two the base can see: H_2_0 (the elbow) and H_4_0 (the
    # end effector).
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

    H_2_0 = H_1_0 @ H_2_1
    H_4_0 = H_2_0 @ H_3_2 @ H_4_3

    return {
        'H_2_0': H_2_0,  # the elbow, in the base frame
        'H_4_0': H_4_0,  # the end effector, in the base frame
    }


def load_trajectory(path):
    """
    Loads a trajectory recorded by `goto.py --guided` and returns its RR joint angles.

    The recording is a comma-separated text file, one row per sample, taken at
    100 Hz — so sample k was taken at k / 100 seconds. Each row is two joint angles
    in radians: column 0 is theta1, column 1 is theta2. np.loadtxt(path, delimiter=',')
    reads the whole thing into one (N, 2) array (the header line is a comment, and
    numpy skips it).

    Returns two NumPy arrays, theta1 and theta2, each of length N and aligned in time:
    theta1[k] and theta2[k] are the two joint angles at the same instant.
    """
    data = np.loadtxt(path, delimiter=',')
    theta1 = data[:, 0]
    theta2 = data[:, 1]
    return theta1, theta2


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
    theta1, theta2 = load_trajectory(path)

    # To-Do 2: Command the arm to follow loaded trajectory.
    arm.set_position(theta1[0], theta2[0])
    for k in range(len(theta1)):
        arm.servo_to_position(theta1[k], theta2[k])
