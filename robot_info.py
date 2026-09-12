import numpy as np

# Joint origins of the xArm7, from UFACTORY's model, in mm.
# Shoulder (joint 1) to elbow (joint 4 axis): 293 out, 52.5 across.
# Elbow to wrist (joint 7 axis): 418.5 back the other way, 77.5 across.
LINK1_MM = np.array([293.0, 52.5])
LINK2_MM = np.array([-418.5, 77.5])

# The joints held at +-90 deg to make the arm planar — 2, 3, 5 and 6, counting
# from 1 — and the angles they are held at. That is the "90 90 -90 90" set: it
# leaves the axes of joints 1, 4 and 7 all parallel to world z.
LOCKED_INDICES = np.array([1, 2, 4, 5])
LOCKED_ANGLES_DEG = np.array([90.0, 90.0, -90.0, 90.0])

# What is left is an exact planar RR turning in a horizontal plane: joint 1 at
# the base, joint 4 at the elbow (joint 7 only spins the tool).
RR_INDICES = np.array([0, 3])

# The whole 7-joint pose the RR is driven from, in degrees. At -70 and +60 the
# wrist sits 478 mm in front of the base, near the robot's centre line.
HOME_DEG = np.array([-70.0, 90.0, 90.0, 60.0, -90.0, 90.0, 0.0])

# Hz. Free drive samples the arm this often, so goto.py writes its recordings
# on this grid — row k is the arm at k / RECORD_RATE — and replay.py streams
# them back at it. Nothing in a recording says what it was, so both ends read
# it from here.
RECORD_RATE = 100.0


def robot_info():
    robot_info = {}
    # length of the links [m], from the joint origins above
    robot_info['link_lengths'] = np.array([
        np.hypot(*LINK1_MM) / 1000.0,  # 0.2977 m
        np.hypot(*LINK2_MM) / 1000.0,  # 0.4256 m
    ])
    return robot_info


def q2rr(q):
    """(theta1, theta2) of the planar RR, from a 7-joint configuration."""
    A1 = np.arctan2(LINK1_MM[1], LINK1_MM[0])  # 10.16 deg
    A2 = np.arctan2(LINK2_MM[1], LINK2_MM[0]) - A1  # 159.35 deg
    return q[0] + A1, A2 - q[3]


def adjust_rr(theta1, theta2):
    """(q1, q4) of the arm, from the planar RR's angles — the inverse of `rr_angles`."""
    A1 = np.arctan2(LINK1_MM[1], LINK1_MM[0])  # 10.16 deg
    A2 = np.arctan2(LINK2_MM[1], LINK2_MM[0]) - A1  # 159.35 deg
    return theta1 - A1, A2 - theta2
