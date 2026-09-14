"""Replay one recording with the student's run_trajectory.

    python replay.py rr-20260911-101500.csv     # from recordings/
    python replay.py some/where/else.csv        # or from anywhere

The real arm when ROBOT_IP is set, the MuJoCo simulation otherwise. The
trajectory is drawn with the student's FK: the planned (recorded) path dotted,
and the path the arm actually takes as it replays solid. The arrow off the end
effector is its velocity, the measured joint speeds through the student's
`jacobian_RR`.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xarm7_lib import Robot

from fk import run_trajectory
from live_plot import LivePlot, arm_points
from robot_info import RECORD_RATE, q2rr, qd2rr, rr2q

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

    def new_trajectory(self, path):
        """Read the angles to follow, and clear the plot.

        The file holds those alone, so the path drawn to follow is them through
        the student's FK.
        """
        theta = np.loadtxt(path, delimiter=",")
        self.xy = np.array([arm_points(th) for th in theta])[:, 2]
        self.name, self.last, self.next_tick = path.name, None, None
        self.plot.reset(f"{self.name} — waiting for set_position", planned=self.xy)

    def set_position(self, theta1, theta2):
        """Planned move to (theta1, theta2); returns once the arm has settled there."""
        self._redraw(f"{self.name} — moving to start")
        q = rr2q(theta1, theta2)
        if not self.robot.set_joint_targets(q, speed=START_SPEED):
            raise RuntimeError(f"{self.name}: the arm never reached the start pose")
        self.last, self.next_tick = q, time.perf_counter()  # the stream starts from here
        self._redraw(f"{self.name} — replaying")

    def servo_to_position(self, theta1, theta2):
        """Stream one sample, then wait out the rest of its 1/rate tick."""
        if self.last is None:
            raise RuntimeError("call arm.set_position before arm.servo_to_position")
        q = rr2q(theta1, theta2)
        step = np.max(np.abs(q - self.last))
        if step > MAX_STEP:
            raise RuntimeError(
                f"{self.name}: a {step:.3f} rad jump at sample {len(self.measured)} — are "
                "the samples in order, in radians, and theta1 / theta2 the right way round?")
        self.robot.servo_joints(q)
        self.last = q
        # the angles the arm really reached, drawn at the plot's own rate
        self.plot.update(q2rr(self.robot.joint_values),
                         qd2rr(self.robot.joint_velocities))
        self.next_tick += 1.0 / RECORD_RATE
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
        print(f"{self.name}: {len(self.measured)} of {len(self.xy)} samples replayed.")
        self._redraw(f"{self.name} — done")

    def _redraw(self, title=None):
        """Show where the arm is now, whatever the plot's own redraw rate."""
        self.plot.update(q2rr(self.robot.joint_values),
                         qd2rr(self.robot.joint_velocities),
                         record=False, title=title, force=True)


def recording_path(name):
    """The file to play. A name with no folder is looked for in recordings/,
    so the file name record.py prints when it saves can be pasted straight in."""
    path = Path(name)
    if not path.exists():
        path = RECORDINGS / name
    if not path.exists():
        raise SystemExit(f"no such recording: {name}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", help="the recording to replay, by path or "
                                          "by name within recordings/")
    path = recording_path(parser.parse_args(argv).recording)

    arm = RRArm()
    try:
        arm.new_trajectory(path)
        run_trajectory(arm, path)
        arm.report()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        arm.robot.stop()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
