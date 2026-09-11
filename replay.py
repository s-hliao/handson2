"""Replay a recorded RR trajectory on the arm, plotted with the student's FK.

    python replay.py                          # the newest recordings/rr-*.npz
    python replay.py recordings/rr-<stamp>.npz

The real arm when ROBOT_IP is set, the MuJoCo simulation otherwise.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xarm7_lib import DEFAULT_BOX, Robot

from fk import forward_kinematics_RR, load_trajectory

# The recording is in planar RR angles: theta1 = q1 + A1, theta2 = A2 - q4.
A1 = np.arctan2(52.5, 293.0)
A2 = np.arctan2(77.5, -418.5) - A1

path = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
    (Path(__file__).parent / "recordings").glob("rr-*.npz"))
theta1, theta2 = load_trajectory(path)
rec = np.load(path)  # the rest of the recording: the pose, link lengths, rate
home, rate = rec["home"], float(rec["rate"])
l1, l2 = float(rec["l1"]), float(rec["l2"])
print(f"replaying {path}: {len(theta1)} samples over {len(theta1) / rate:.1f}s")


def pose(th1, th2):
    """The 7-joint command for one RR sample; the locked joints stay at home."""
    q = home.copy()
    q[0], q[3] = th1 - A1, A2 - th2
    return q


def arm_xy(q):
    """Base, elbow and end effector of the arm at `q`, by the student's FK."""
    th = (q[0] + A1, A2 - q[3])
    elbow = forward_kinematics_RR(*th, l1, 0.0)["H_6_0"][:2, 2]
    end = forward_kinematics_RR(*th, l1, l2)["H_6_0"][:2, 2]
    return np.array([[0.0, 0.0], elbow, end])


# Seen from in front of the robot: points go in as (y, x), +x pointing down.
plt.ion()
(x_lo, x_hi), (y_lo, y_hi), _ = DEFAULT_BOX
fig, ax = plt.subplots(figsize=(6, 6 * (x_hi - x_lo) / (y_hi - y_lo)))
ax.plot([y_lo, y_hi, y_hi, y_lo, y_lo], [x_lo, x_lo, x_hi, x_hi, x_lo],
        color="tab:red", lw=1, label="safety box")
ax.plot(*rec["xy"].T[::-1], color="tab:orange", lw=2, label="recorded")
(trail,) = ax.plot([], [], color="tab:blue", lw=1, label="replayed")
(links,) = ax.plot([], [], "-o", color="tab:blue", lw=3, label="arm")
ax.set(xlim=(y_lo - 0.05, y_hi + 0.05), ylim=(x_hi + 0.05, x_lo - 0.05),
       aspect="equal", xlabel="y (m)", ylabel="x (m)  — towards you",
       title=path.name)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize="small")
plt.pause(0.1)

robot = Robot()
robot.set_joint_targets(pose(theta1[0], theta2[0]), speed=0.3)

measured, drawn = [], 0.0
start = time.perf_counter()
try:
    for k, (th1, th2) in enumerate(zip(theta1, theta2)):
        robot.servo_joints(pose(th1, th2))
        points = arm_xy(robot.joint_values)
        measured.append(points[2])
        if time.perf_counter() - drawn > 0.05:  # redraw at ~20 Hz
            drawn = time.perf_counter()
            links.set_data(*points.T[::-1])
            trail.set_data(*np.transpose(measured)[::-1])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        time.sleep(max(0.0, start + k / rate - time.perf_counter()))
except KeyboardInterrupt:
    print("interrupted")
finally:
    robot.stop()

error = np.linalg.norm(np.asarray(measured) - rec["xy"][:len(measured)], axis=1)
print(f"xy tracking error: max {error.max() * 1000:.1f} mm, "
      f"rms {np.sqrt(np.mean(error ** 2)) * 1000:.1f} mm")
plt.ioff()
plt.show()
