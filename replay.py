"""Replay the recordings listed in fk.TRAJECTORIES, in order, with the student's run_trajectory.

    python replay.py

The real arm when ROBOT_IP is set, the MuJoCo simulation otherwise. Each
trajectory is drawn with the student's FK: the planned (recorded) path dotted,
and the path the arm actually takes as it replays solid.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xarm7_lib import Robot

from fk import TRAJECTORIES, run_trajectory
from live_plot import LivePlot
from robot_info import rr_angles, rr_joints

RECORDINGS = Path(__file__).parent / "recordings"
START_SPEED = 0.3  # rad/s, for set_position's planned move
# rad in one sample. Hand-guided motion moves a joint a few hundredths of a
# radian per sample (more in an older recording, where the controller's slower
# position reports show up as steps); a bigger jump is a bug in the student's
# code — degrees, theta1/theta2 swapped, or samples skipped.
MAX_STEP = 0.15


class RRArm:
    """The xArm driven as the planar RR arm, drawn live with the student's FK."""

    def __init__(self):
        self.plot = LivePlot(planned=True)
        self.robot = Robot()

    def new_trajectory(self, path, name):
        """Take the locked pose and rate from the recording, and clear the plot."""
        rec = np.load(path)
        self.home, self.rate, self.xy = rec["home"], float(rec["rate"]), rec["xy"]
        self.name, self.last, self.next_tick = name, None, None
        self.plot.reset(f"{name} — waiting for set_position", planned=self.xy)

    def set_position(self, theta1, theta2):
        """Planned move to (theta1, theta2); returns once the arm has settled there."""
        self._redraw(f"{self.name} — moving to start")
        q = self._pose(theta1, theta2)
        if not self.robot.set_joint_targets(q, speed=START_SPEED):
            raise RuntimeError(f"{self.name}: the arm never reached the start pose")
        self.last, self.next_tick = q, time.perf_counter()  # the stream starts from here
        self._redraw(f"{self.name} — replaying")

    def servo_to_position(self, theta1, theta2):
        """Stream one sample, then wait out the rest of its 1/rate tick."""
        if self.last is None:
            raise RuntimeError("call arm.set_position before arm.servo_to_position")
        q = self._pose(theta1, theta2)
        step = np.max(np.abs(q - self.last))
        if step > MAX_STEP:
            raise RuntimeError(
                f"{self.name}: a {step:.3f} rad jump at sample {len(self.measured)} — are "
                "the samples in order, in radians, and theta1 / theta2 the right way round?")
        self.robot.servo_joints(q)
        self.last = q
        # the angles the arm really reached, drawn at the plot's own rate
        self.plot.update(rr_angles(self.robot.joint_values))
        self.next_tick += 1.0 / self.rate
        time.sleep(max(0.0, self.next_tick - time.perf_counter()))

    @property
    def measured(self):
        """The end-effector points the arm has actually been through."""
        return self.plot.visited

    def report(self):
        n = min(len(self.measured), len(self.xy))
        if n == 0:
            print(f"{self.name}: nothing was replayed")
            return
        error = np.linalg.norm(np.asarray(self.measured[:n]) - self.xy[:n], axis=1)
        print(f"{self.name}: {len(self.measured)} of {len(self.xy)} samples replayed, "
              f"xy tracking error max {error.max() * 1000:.1f} mm, "
              f"rms {np.sqrt(np.mean(error ** 2)) * 1000:.1f} mm")
        self._redraw(f"{self.name} — done")

    def _pose(self, theta1, theta2):
        """The 7-joint command for one RR sample; the locked joints stay at home."""
        q = self.home.copy()
        q[0], q[3] = rr_joints(theta1, theta2)
        return q

    def _redraw(self, title=None):
        """Show where the arm is now, whatever the plot's own redraw rate."""
        self.plot.update(rr_angles(self.robot.joint_values), record=False,
                         title=title, force=True)


if not TRAJECTORIES:
    raise SystemExit("fk.TRAJECTORIES is empty — list the recordings to replay there.")
arm = RRArm()
try:
    for i, name in enumerate(TRAJECTORIES, 1):
        arm.new_trajectory(RECORDINGS / name, f"{i}/{len(TRAJECTORIES)}  {name}")
        run_trajectory(arm, RECORDINGS / name)
        arm.report()
except KeyboardInterrupt:
    print("interrupted")
finally:
    arm.robot.stop()

plt.ioff()
plt.show()
