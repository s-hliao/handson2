"""Replay the recordings listed in fk.TRAJECTORIES, in order, with the student's run_trajectory.

    python replay.py

The real arm when ROBOT_IP is set, the MuJoCo simulation otherwise. On the
real arm the wrist (joints 5-7) is left however it was found, bent for a pen
say, and if a gripper is on the tool connector the claw is closed until it
can't close any further, then held there. Each trajectory is drawn with the
student's FK: the planned (recorded) path dotted, and the path the arm actually
takes as it replays solid.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from xarm7_lib import DEFAULT_BOX, Robot

from fk import TRAJECTORIES, forward_kinematics_RR, run_trajectory

# The recording is in planar RR angles: theta1 = q1 + A1, theta2 = A2 - q4.
A1 = np.arctan2(52.5, 293.0)
A2 = np.arctan2(77.5, -418.5) - A1
RECORDINGS = Path(__file__).parent / "recordings"
START_SPEED = 0.3  # rad/s, for set_position's planned move
# rad in one sample. Hand-guided motion moves a joint a few hundredths of a
# radian per sample (more in an older recording, where the controller's slower
# position reports show up as steps); a bigger jump is a bug in the student's
# code — degrees, theta1/theta2 swapped, or samples skipped.
MAX_STEP = 0.15
CLAW_SPEED = 1500  # r/min, a slow squeeze (the gripper takes 1000-5000)
CLAW_PRELOAD = 10  # pulses past where it stopped, so it keeps a grip (850 = open)


def lock_claw(sdk):
    """Close the xArm Gripper until it stops on whatever it holds, then hold it there.

    The SDK's own wait is for reaching the target, which a claw closed on a pen
    never does, so this watches the position until it stops changing instead.
    """
    code, pos = sdk.get_gripper_position()
    if code != 0 or pos is None:
        print("no gripper answering on the tool connector; leaving the claw alone")
        return
    sdk.clean_gripper_error()
    sdk.set_gripper_mode(0)  # position mode
    sdk.set_gripper_enable(True)
    sdk.set_gripper_speed(CLAW_SPEED)
    sdk.set_gripper_position(0, wait=False)
    started = still_since = time.monotonic()
    last = pos
    while time.monotonic() - started < 10.0:
        time.sleep(0.1)
        code, pos = sdk.get_gripper_position()
        if code != 0:
            break
        if abs(pos - last) > 1:
            last, still_since = pos, time.monotonic()
        elif time.monotonic() - started > 1.0 and time.monotonic() - still_since > 0.5:
            break  # stalled on the pen, or fully closed
    # Held a little tighter than where it stopped, so what it holds can't slip.
    sdk.set_gripper_position(max(last - CLAW_PRELOAD, 0), wait=False)
    print(f"claw closed to {last:.0f} (0 = shut, 850 = open) and held there")


class RRArm:
    """The xArm driven as the planar RR arm, drawn live with the student's FK."""

    def __init__(self):
        # Seen from in front of the robot: points go in as (y, x), +x pointing down.
        plt.ion()
        (x_lo, x_hi), (y_lo, y_hi), _ = DEFAULT_BOX
        self.fig, self.ax = plt.subplots(figsize=(6, 6 * (x_hi - x_lo) / (y_hi - y_lo)))
        ax = self.ax
        ax.plot([y_lo, y_hi, y_hi, y_lo, y_lo], [x_lo, x_lo, x_hi, x_hi, x_lo],
                color="tab:red", lw=1, label="safety box")
        # The planned path goes on top, so its dots still show where the two overlap.
        (self.recorded,) = ax.plot([], [], ":", color="tab:orange", lw=2.5, zorder=3,
                                   label="planned")
        (self.trail,) = ax.plot([], [], "-", color="tab:blue", lw=1.5, label="actual")
        (self.links,) = ax.plot([], [], "-o", color="tab:blue", lw=3, label="arm")
        ax.set(xlim=(y_lo - 0.05, y_hi + 0.05), ylim=(x_hi + 0.05, x_lo - 0.05),
               aspect="equal", xlabel="y (m)", ylabel="x (m)  — towards you")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")
        self.robot = Robot()
        # Only the real arm has a claw and a wrist worth keeping; the simulation's
        # wrist is wherever its model starts, so there it follows the recording.
        sdk = getattr(self.robot.robot, "arm", None)
        self.wrist = None
        if sdk is not None:
            self.wrist = self.robot.joint_values[4:7].copy()
            print(f"keeping the wrist as found: joints 5-7 at "
                  f"{np.round(np.degrees(self.wrist), 1)} deg")
            lock_claw(sdk)

    def new_trajectory(self, path, name):
        """Take the locked pose and rate from the recording, and clear the plot."""
        rec = np.load(path)
        self.home, self.rate, self.xy = rec["home"], float(rec["rate"]), rec["xy"]
        self.l1, self.l2 = float(rec["l1"]), float(rec["l2"])
        self.name, self.measured, self.last, self.next_tick = name, [], None, None
        self.recorded.set_data(*self.xy.T[::-1])
        self.trail.set_data([], [])
        self._redraw(f"{name} — waiting for set_position")

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
        self.measured.append(self._points(self.robot.joint_values)[2])
        if len(self.measured) % max(1, round(self.rate / 20)) == 0:  # redraw at ~20 Hz
            self._redraw()
        self.next_tick += 1.0 / self.rate
        time.sleep(max(0.0, self.next_tick - time.perf_counter()))

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
        """The 7-joint command for one RR sample: joints 2 and 3 at the recording's
        home, the wrist as found (or at home in simulation)."""
        q = self.home.copy()
        if self.wrist is not None:
            q[4:7] = self.wrist
        q[0], q[3] = theta1 - A1, A2 - theta2
        return q

    def _points(self, q):
        """Base, elbow and end effector of the arm at `q`, by the student's FK."""
        th = (q[0] + A1, A2 - q[3])
        elbow = forward_kinematics_RR(*th, self.l1, 0.0)["H_6_0"][:2, 2]
        end = forward_kinematics_RR(*th, self.l1, self.l2)["H_6_0"][:2, 2]
        return np.array([[0.0, 0.0], elbow, end])

    def _redraw(self, title=None):
        if title:
            self.ax.set_title(title)
        self.links.set_data(*self._points(self.robot.joint_values).T[::-1])
        if self.measured:
            self.trail.set_data(*np.transpose(self.measured)[::-1])
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


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
