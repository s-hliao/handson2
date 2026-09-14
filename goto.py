"""Send the arm to the planar pose, then record hand-guided RR paths.

    python goto.py

Set ROBOT_IP first: free drive is the controller's own teaching mode, so this
script wants the real arm, and refuses the simulation `xarm7_lib.Robot` would
otherwise fall back to.

The move squares joints 2, 3, 5 and 6 up to +90, +90, -90 and +90 degrees and
leaves 1, 4 and 7 wherever it finds them. That is the "90 90 -90 90" locked
set, and it leaves the axes of joints 1, 4 and 7 all parallel to world z.
Joint 7's axis runs straight through the flange, so turning it only spins the
tool: what is left is an exact planar RR turning in a horizontal plane, joint 1
at the base and joint 4 at the elbow.

Because the free joints are left alone, the arm starts from wherever the last
session put it. If that is somewhere it may not legally go — joint4 near 0 is
the folded end of the elbow's travel, and puts the forearm back through link2 —
the arm's own guard refuses the move, and the arm has to be moved clear first.

After the move:

    1. free drive. The controller's joint teaching mode lets the arm be
       pushed by hand; joints 2, 3, 5 and 6 are watched and put back if they
       drift (see `xarm7_lib/free_drive.py`). A live plot, drawn with the
       student's `forward_kinematics_RR` from `fk.py`, shows the RR arm and
       the path it has been through. Everything from here on is recorded.
    2. ctrl-c ends it: the arm stops where it stands, and the path is saved
       to recordings/.

That is one recording; run the script again for the next. The arm is left
holding wherever free drive ended, which is where the next run starts from —
it only squares the locked joints up again. Nothing checks that the workspace
is clear, so only start the script when it already is.

A recording is a plain csv of the RR joint angles and nothing else, two
columns (theta1, theta2) in radians, resampled onto an even 100 Hz grid so
that row k is the arm at k / 100 s:

    # theta1,theta2 in radians, 100 Hz
    -1.0442802157,2.8065783519
    -1.0442798226,2.8065779411
    ...

That is what `replay.py` streams back. Everything else about the arm — the
pose the locked joints are held at, the link lengths, where the FK puts the
end effector — comes from `robot_info.py` and `fk.py` at replay time, not from
the file. So a replay always uses today's `fk.py`, on the angles as recorded.

The pose is printed in degrees; everything else here is radians, like the rest
of the library.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import argparse
import select
import termios
import signal
import math
import sys
import os

from xarm7_lib import RealXArm7, Robot
from xarm7_lib.free_drive import FREE_JOINTS

from live_plot import LivePlot
from robot_info import LOCKED_ANGLES_DEG, LOCKED_INDICES, RECORD_RATE, q2rr


class CtrlC:
    """Ctrl-c as a request to finish, noticed at the next safe point.

    A KeyboardInterrupt goes off in whatever happens to be running. In free
    drive that is mostly matplotlib redrawing inside Tk, whose callback wrapper
    catches it, prints it as an error, and carries on. So inside this context
    SIGINT only sets `requested`, which the free-drive tick and the prompts
    check. A second ctrl-c raises anyway, in case something is stuck.
    """

    def __init__(self):
        self.requested = False

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\n[goto] ctrl-c: finishing (again to force)")

    def __enter__(self):
        self._previous = signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc):
        signal.signal(signal.SIGINT, self._previous)


def connect():
    """Connect to the real arm. Refuses the simulation, which has nothing to push."""
    if not os.environ.get("ROBOT_IP", "").strip():
        raise SystemExit(
            "[goto] ROBOT_IP is not set, so there is no arm to hand-guide.\n"
            "       Set it to the controller's address and run again."
        )
    arm = Robot()
    if not isinstance(arm.robot, RealXArm7):
        raise SystemExit("[goto] Robot() gave the simulation, not the real arm.")
    return arm

def reset(arm):
    """Move the arm to the locked pose, leaving the free joints wherever they are.

    The locked joints are 2, 3, 5 and 6 at +90, +90, -90 and +90 degrees.
    """
    start = arm.joint_values
    locked = start.copy()
    locked[LOCKED_INDICES] = np.radians(LOCKED_ANGLES_DEG)
    if not arm.set_joint_targets(locked):
        raise RuntimeError("Arm never reached the locked pose.")


def save_recording(traj):
    """A free-drive `Trajectory` as the file's (N, 2) angles, one row per tick."""
    theta = np.column_stack(q2rr(traj.q.T))
    changed = np.any(np.diff(theta, axis=0) != 0, axis=1)
    reports = np.flatnonzero(np.concatenate([[True], changed]))
    ticks = np.arange(len(theta))
    return np.column_stack(
        [np.interp(ticks, reports, column[reports]) for column in theta.T]
    )


def main(arm: Robot, out, plot, ctrl_c):
    """One free drive: record until ctrl-c, then save."""
    plot.reset("recording — ctrl-c to stop")
    print("[goto] free drive: push the arm through the path to record.\n"
          "       Ctrl-c stops it there and saves what it has been through.")

    def on_sample(t, q):
        plot.update(q2rr(q))
        return ctrl_c.requested  # True ends the run, and the arm holds where it is

    def pause_until_safe(message):
        """Hold the recovery move until hands are clear. False ends the run.

        The library's own prompt blocks in `input()`, which would sit on a
        ctrl-c until someone pressed enter. Poll instead, so ctrl-c is noticed
        and the plot keeps its window alive while it waits. A closed stdin
        means nobody is at the terminal to say when it is safe, so the run
        ends rather than moving the arm at them.
        """
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)  # only a fresh enter counts
        print(message)
        print("        press enter when your hands are clear: ", end="", flush=True)
        while not ctrl_c.requested:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return sys.stdin.readline() != ""
            plot.idle()
        return False

    # math.inf: the run ends when `on_sample` says so, not on a clock.
    traj = arm.free_drive(FREE_JOINTS, duration=math.inf,
                          on_sample=on_sample,
                          confirm=pause_until_safe)

    theta = save_recording(traj)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, theta, delimiter=",", fmt="%.10f",
               header=f"theta1,theta2 in radians, {RECORD_RATE:.0f} Hz")
    n = len(theta)
    print(f"[goto] {n} samples over {n / RECORD_RATE:.1f}s written to {out}")
    plot.draw(f"saved {out.name} — {n} samples", force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        help="file to write the recording to "
        "(default: recordings/rr-<timestamp>.csv)",
    )
    return parser.parse_args()

def default_recording_path():
    """recordings/rr-<timestamp>.csv, stamped when free drive starts.

    One run records once and takes longer than a second to do it, so two of
    them can never land on the same name.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("recordings") / f"rr-{stamp}.csv"

if __name__ == "__main__":
    args = parse_args()
    np.set_printoptions(precision=3, suppress=True)

    robot = connect()
    with robot.robot:
        reset(robot)
        out = Path(args.out) if args.out else default_recording_path()
        with CtrlC() as ctrl_c:
            main(robot, out, LivePlot(), ctrl_c)
