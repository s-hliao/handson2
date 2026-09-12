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
import termios
import select
import signal
import math
import sys
import os

from xarm7_lib import RealXArm7, Robot
from xarm7_lib.free_drive import FREE_JOINTS

from live_plot import LivePlot
from robot_info import LOCKED_ANGLES_DEG, LOCKED_INDICES, RECORD_RATE, q2rr

RECORDINGS = Path(__file__).resolve().parent / "recordings"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        help="folder to write the recordings to, each as rr-<timestamp>.csv "
        "(default: recordings/)",
    )
    return parser.parse_args(argv)


def default_recording_path(folder):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = Path(folder) / f"rr-{stamp}.csv", 1
    while path.exists():  # two recordings inside the same second
        n += 1
        path = Path(folder) / f"rr-{stamp}-{n}.csv"
    return path


def degrees(q):
    return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]"


def connect():
    """The real arm. Refuses the simulation, which has nothing to push.

    `Robot` picks the backend off ROBOT_IP and silently falls back to MuJoCo,
    so say what is missing before it starts one, and check what came back.
    """
    if not os.environ.get("ROBOT_IP", "").strip():
        raise SystemExit(
            "[goto] ROBOT_IP is not set, so there is no arm to hand-guide.\n"
            "       Set it to the controller's address and run again."
        )
    arm = Robot()
    if not isinstance(arm.robot, RealXArm7):
        raise SystemExit("[goto] Robot() gave the simulation, not the real arm.")
    return arm


def flush_stdin():
    """Throw away anything typed ahead, so only a fresh enter counts."""
    if sys.stdin.isatty():

        termios.tcflush(sys.stdin, termios.TCIFLUSH)


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


def wait_for_enter(message, ctrl_c, plot=None):
    """Hold until someone presses enter. False on ctrl-c, or if nobody is at
    the terminal.

    The guided phases gate on this even under `-y`: it is a beat to get hands
    clear, not a question. Nobody at the terminal means nobody watching the
    arm, so the caller skips instead. An enter pressed before the prompt
    appeared doesn't count — it was meant for something else. Polled rather
    than a blocking `input()`, so that ctrl-c is seen and the plot stays
    responsive while it waits.
    """
    flush_stdin()
    print(message, end="", flush=True)
    while not ctrl_c.requested:
        if select.select([sys.stdin], [], [], 0.05)[0]:
            if sys.stdin.readline() == "":  # stdin closed: nobody there
                print("\n[goto] no terminal to pause at; skipping.")
                return False
            return True
        if plot is not None:
            plot.idle()
    return False


# ----------------------------------------------------------------------
# The guided session
# ----------------------------------------------------------------------


def rr_recording(traj):
    """A free-drive `Trajectory`, as the file's (N, 2) angles."""
    t = traj.t
    theta = np.column_stack(q2rr(traj.q.T))

    # Every time a watched joint had to be put back, the samples either side
    # are seconds apart. The free joints are left where they were, so there
    # is no jump across the seam — just a pause that isn't worth replaying.
    period = 1.0 / traj.rate
    dt = np.minimum(np.diff(t), 2 * period)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    # The controller reports positions less often than the loop samples them,
    # so the raw angles are a staircase: each report held for a few ticks,
    # then a jump. Only the ticks where a new report arrived carry anything,
    # so interpolate between those, or a replay would servo every step.
    fresh = np.concatenate([[True], np.any(np.diff(theta, axis=0) != 0, axis=1)])
    fresh[-1] = True
    t, theta = t[fresh], theta[fresh]

    # The loop's ticks jitter and occasionally overrun, so put the samples on
    # an exact 1/RECORD_RATE grid: row k is then the arm at k / RECORD_RATE,
    # and the two angle columns are all a replay needs — no timestamps to pace
    # by, and no rate to read out of the file.
    step = 1.0 / RECORD_RATE
    grid = np.arange(0.0, t[-1] + step / 2, step)
    return np.column_stack([np.interp(grid, t, column) for column in theta.T])


def guided_session(arm, folder, plot, ctrl_c):
    """One free drive: record until ctrl-c, then save."""
    plot.reset("recording — ctrl-c to stop")
    print("[goto] free drive: push the arm through the path to record.\n"
          "       Ctrl-c stops it there and saves what it has been through.")

    def on_sample(t, q):
        plot.update(q2rr(q))
        return ctrl_c.requested  # True ends the run, and the arm holds where it is

    def hands_off(message):
        # The library's own prompt blocks in input(); this one sees ctrl-c.
        print(message)
        return wait_for_enter("        press enter when your hands are clear: ",
                              ctrl_c, plot)

    # math.inf: the run ends when `on_sample` says so, not on a clock.
    traj = arm.free_drive(FREE_JOINTS, math.inf,
                          on_sample=on_sample,
                          confirm=hands_off)
    print(f"[goto] {traj}")
    if arm.robot.has_error:  # `Robot` doesn't forward these; the real arm has them
        print("[goto] the controller latched an error during free drive; "
              "clearing it.")
        arm.robot.clear_errors()

    if traj.t.size == 0:
        print("[goto] nothing was recorded; nothing saved.")
        plot.draw("not recorded", force=True)
        return

    theta = rr_recording(traj)
    out = default_recording_path(folder)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, theta, delimiter=",", fmt="%.10f",
               header=f"theta1,theta2 in radians, {RECORD_RATE:.0f} Hz")
    n = len(theta)
    print(f"[goto] {n} samples over {n / RECORD_RATE:.1f}s written to {out}")
    print(f"[goto] sampled at {traj.rate:.0f} Hz; the controller's report was "
          f"seen changing at {traj.report_rate:.0f} Hz")
    if traj.interruptions:
        print(f"[goto] {traj.interruptions} interruption(s) were closed up in t")
    plot.draw(f"saved {out.name} — {n} samples", force=True)


# ----------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    np.set_printoptions(precision=3, suppress=True)

    arm = connect()
    # `Robot` isn't a context manager itself; the backend it wraps is, and its
    # exit is what puts the arm back in position control and disconnects.
    with arm.robot:
        start = arm.joint_values
        # Only the locked joints are commanded; 1, 4 and 7 stay where they are,
        # so the arm squares itself up without swinging the RR across the desk.
        goal = start.copy()
        goal[LOCKED_INDICES] = np.radians(LOCKED_ANGLES_DEG)
        print(f"[goto] now at  {degrees(start)} deg")
        print(f"[goto] going to {degrees(goal)} deg")
        print("[goto]   (joints 2, 3, 5 and 6 to +-90; 1, 4 and 7 left alone)")

        if not arm.set_joint_targets(goal):
            print("[goto] not starting free drive: the arm never reached the pose.")
            return 1

        # ---- free drive: one recording --------------------------------
        print("[goto] free drive: joints 1, 4 and 7 can be pushed by hand; 2, 3, "
              "5 and 6 are\n       watched, and the arm stops to put them back "
              "if they drift.\n       Everything is recorded; ctrl-c to stop and "
              "save.")
        folder = Path(args.out) if args.out else RECORDINGS
        with CtrlC() as ctrl_c:
            try:
                guided_session(arm, folder, LivePlot(), ctrl_c)
                # Free drive has already handed the arm back to position
                # control, so it is standing still wherever it was left.
                print("[goto] done. The arm is holding where it is.")
                return 0
            except KeyboardInterrupt:  # a second ctrl-c: something is stuck
                arm.stop()
                print("\n[goto] stopped. The arm is holding where it is.")
                return 130


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # outside the guided session: the arm's own
        print("\n[goto] interrupted.")  # context has stopped it on the way out
        sys.exit(130)
