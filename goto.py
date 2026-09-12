"""Send the arm to the planar pose, then record hand-guided RR paths.

    python goto.py

`xarm7_lib.Robot` picks the arm: the real one when ROBOT_IP is set, the MuJoCo
simulation otherwise — and free drive needs the real one, there being nothing
to push in simulation.

The target holds joints 2, 3, 5 and 6 at +90, +90, -90 and +90 degrees.
That is the "90 90 -90 90" locked set, and it leaves joints 1, 4 and 7 with
their axes all parallel to world z. Joint 7's axis runs straight through the
flange, so turning it only spins the tool: what is left is an exact planar RR
turning in a horizontal plane, joint 1 at the base and joint 4 at the elbow.
At -70 and +60 degrees the wrist sits 478 mm in front of the base, near the
robot's centre line.

Leaving the free joints at zero instead — the bare "90 90 -90 90" — would not
work: joint4 = 0 is the folded end of the elbow's travel, and the forearm ends
up behind the shoulder, through link2. The collision model refuses it.

After the move:

    1. free drive, positioning. The controller's joint teaching mode lets the
       arm be pushed by hand; joints 2, 3, 5 and 6 are watched and put back if
       they drift (see `xarm7_lib/free_drive.py`). A live plot, drawn with the
       student's `forward_kinematics_RR` from `fk.py`, shows the RR arm.
       Press enter to start recording.
    2. free drive, recording. The same session carries on, the plot is
       cleared, and every visited point is drawn. Press enter again to stop.
    3. the recording is saved to recordings/, and the arm drives back home —
       then straight back to 1 for the next recording.

It only finishes on ctrl-c, which works at any point: a recording in progress
is saved, and the arm is left stopped and holding where it is (the next run's
first move takes it home). During a move, ctrl-c stops the arm at once.

The recording is the RR joint angles, resampled onto an even grid at the
free-drive rate (100 Hz) so that sample k is at k / rate, plus the elbow and
end-effector xy the student's FK puts them at:

    t (N,)  theta (N,2)  q (N,2)  xy (N,2)  elbow_xy (N,2)  home (7,)  l1  l2  rate

`theta` is what `replay.py` streams back. The locked joints are replayed at
their nominal +-90 (`home`), not at wherever they drifted to during the
recording, so the replay lands on exactly the planar `xy` that was saved.

The pose is printed in degrees; everything else here is radians, like the rest
of the library.
"""

import argparse
import contextlib
import math
import select
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from live_plot import LivePlot

# joints 1..7, degrees. Joints 2, 3, 5 and 6 are the "90 90 -90 90" that
# makes the arm planar; 1 and 4 are the RR, here reaching out along +x.
DEFAULT_TARGET_DEG = (-70.0, 90.0, 90.0, 60.0, -90.0, 90.0, 0.0)

# The joints that must sit at +-90 for joints 1, 4 and 7 to be vertical.
LOCKED_INDICES = (1, 2, 4, 5)

DEFAULT_SPEED = 0.3  # rad/s, the library's own default
RETURN_SPEED_CAP = 0.2  # rad/s; the return moves with people close to the arm
_SETTLE_GRACE = 15.0  # s added to a move's travel time before the wait gives up

# How far a locked joint may sit from +-90 and still leave the free joints
# vertical enough to free-drive. 3 degrees tilts a 426 mm forearm by 22 mm.
_PLANAR_TOLERANCE = math.radians(3.0)

RECORDINGS = Path(__file__).resolve().parent / "recordings"

# ----------------------------------------------------------------------
# The planar RR, from UFACTORY's xarm7 model (joint origins in mm).
# ----------------------------------------------------------------------
# Shoulder (joint1) to elbow (joint4 axis): 293 out, 52.5 across.
# Elbow to wrist (joint7 axis): 418.5 and 77.5 across the other way.
L1 = math.hypot(293.0, 52.5) / 1000.0  # 0.2977 m
L2 = math.hypot(418.5, 77.5) / 1000.0  # 0.4256 m
# Those sideways offsets put the links at fixed angles to the joint zeros,
# and joint 4's axis points down (-z), so it turns the forearm clockwise:
#     theta1 = q1 + A1,   theta2 = A2 - q4
A1 = math.atan2(52.5, 293.0)  # 10.16 deg
A2 = math.atan2(77.5, -418.5) - A1  # 159.35 deg


def rr_angles(q):
    """(theta1, theta2) of the planar RR, from a 7-joint configuration."""
    return q[0] + A1, A2 - q[3]


def rr_points(theta1, theta2):
    """Base, elbow and end effector in the plane, (3, 2) m, by the student's FK.

    The elbow is the same FK with a zero-length forearm, which works whichever
    way `fk.py` splits the chain into matrices.
    """
    from fk import forward_kinematics_RR

    end = forward_kinematics_RR(theta1, theta2, L1, L2)["H_6_0"]
    elbow = forward_kinematics_RR(theta1, theta2, L1, 0.0)["H_6_0"]
    return np.array([[0.0, 0.0], elbow[:2, 2], end[:2, 2]])


def check_fk(q):
    """Refuse to start unless `fk.py` puts the wrist where the arm really is.

    Checked against the closed form here rather than trusted, so an unfinished
    or wrong FK is found at the terminal, not halfway through a session.
    """
    theta1, theta2 = rr_angles(q)
    expected = np.array([
        L1 * math.cos(theta1) + L2 * math.cos(theta1 + theta2),
        L1 * math.sin(theta1) + L2 * math.sin(theta1 + theta2),
    ])
    try:
        got = rr_points(theta1, theta2)[2]
    except Exception as err:  # the stub returns None, the student's may raise
        raise SystemExit(f"[goto] fk.py isn't finished yet ({type(err).__name__}: "
                         f"{err}); complete forward_kinematics_RR first.")
    miss = float(np.linalg.norm(got - expected))
    if miss > 1e-3:
        raise SystemExit(
            f"[goto] fk.py puts the wrist at {np.round(got, 3)} m, but the arm "
            f"would be at {np.round(expected, 3)} m ({miss * 1000:.0f} mm off).\n"
            "       Fix forward_kinematics_RR before hand-guiding the arm."
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        help="folder to write the recordings to, each as rr-<timestamp>.npz "
        "(default: recordings/)",
    )
    return parser.parse_args(argv)


def default_recording_path(folder):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path, n = Path(folder) / f"rr-{stamp}.npz", 1
    while path.exists():  # two recordings inside the same second
        n += 1
        path = Path(folder) / f"rr-{stamp}-{n}.npz"
    return path


def degrees(q):
    return "[" + ", ".join(f"{math.degrees(v):7.2f}" for v in q) + "]"


def preflight(start, goal):
    """Report on the pose and on the straight line to it. True if both are clear."""
    from xarm7_lib.safety import DEFAULT_BOX, DEFAULT_MARGIN, SafetyGuard

    print("[goto] loading the collision model...")
    guard = SafetyGuard(box=DEFAULT_BOX, margin=DEFAULT_MARGIN)

    pose = guard.check(goal)
    print(f"[goto] target pose: {'allowed' if pose is None else pose}")

    path, reached = guard.check_path(start, goal)
    if path is None:
        print("[goto] path from here: clear the whole way")
    else:
        # `reached` is the fraction of the line that is safe, so this is the
        # furthest the arm could legally get before something touches.
        stopped = start + reached * (goal - start)
        print(f"[goto] path from here: {path}")
        print(f"[goto]   clear for {reached * 100:.0f}% of the way, to {degrees(stopped)}")
    return pose is None and path is None


def connect():
    """The arm this run drives: the real one if ROBOT_IP is set, else the sim.

    Imported here rather than at the top so that importing this module costs
    nothing but the standard library and numpy.
    """
    from xarm7_lib import Robot

    return Robot()


def confirm(question):
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # not a terminal: treat silence as no
        return False


def flush_stdin():
    """Throw away anything typed ahead, so only a fresh enter counts."""
    if sys.stdin.isatty():
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)


class CtrlC:
    """Ctrl-c as a request to finish, noticed at the next safe point.

    A KeyboardInterrupt goes off in whatever happens to be running. In free
    drive that is mostly matplotlib redrawing inside Tk, whose callback wrapper
    catches it, prints it as an error, and carries on. So inside this context
    SIGINT only sets `requested`, which the free-drive tick and the prompts
    check. Moves are the exception: ctrl-c during one should stop the arm at
    once, which the library does on a KeyboardInterrupt, so `moving()` puts
    that back for the move's duration. A second ctrl-c raises anyway, in
    case something is stuck.
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

    @contextlib.contextmanager
    def moving(self):
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, self._handler)


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


def travel_timeout(start, goal, speed):
    span = float(np.max(np.abs(np.asarray(goal) - np.asarray(start))))
    return span / max(speed, 1e-6) + _SETTLE_GRACE


def move_to(arm, goal, speed):
    """Drive to `goal`. Returns (reached, exit code) — the code is None if the
    move was allowed to happen at all, whatever came of it."""
    from xarm7_lib import XArmError
    from xarm7_lib.safety import SafetyError

    try:
        reached = arm.set_joint_targets(
            goal, speed=speed, wait=True,
            timeout=travel_timeout(arm.joint_values, goal, speed),
        )
    except SafetyError as err:
        print(f"[goto] refused by the guard: {err}")
        return False, 2
    except XArmError as err:
        print(f"[goto] the controller refused it: {err}")
        return False, 3
    except KeyboardInterrupt:
        arm.stop()
        print("\n[goto] interrupted; the arm is stopped and holding.")
        return False, 130

    final = arm.joint_values
    residual = math.degrees(float(np.max(np.abs(final - goal))))
    print(f"[goto] {'arrived' if reached else 'did NOT arrive'}")
    print(f"[goto] now at  {degrees(final)} deg  (worst joint off by "
          f"{residual:.2f} deg)")
    return reached, None


# ----------------------------------------------------------------------
# The guided session
# ----------------------------------------------------------------------


def out_of_plane(q):
    """The locked joints that are too far from +-90 to free-drive the rest.

    At +-90 the axes of joints 4 and 7 are vertical, and gravity has no moment
    about a vertical axis — so the free joints carry nothing, wherever they go.
    """
    return [
        index for index in LOCKED_INDICES
        if abs(math.cos(q[index])) > math.sin(_PLANAR_TOLERANCE)
    ]


def enter_pressed():
    """True if enter has been pressed at the terminal, without blocking."""
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    # A closed stdin is always readable, and reads as "" rather than "\n".
    return bool(readable) and sys.stdin.readline() != ""


def rr_recording(traj, t_start, home):
    """The recorded part of a free-drive `Trajectory`, as the RR file's arrays."""
    keep = traj.t >= t_start
    t = traj.t[keep]
    theta = np.column_stack(rr_angles(traj.q[keep].T))

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
    # an exact 1/rate grid: sample k is then at k / rate, and the two angle
    # columns are all a replay needs — no timestamps to pace by.
    grid = np.arange(0.0, t[-1] + period / 2, period)
    theta = np.column_stack([np.interp(grid, t, column) for column in theta.T])
    q = np.column_stack([theta[:, 0] - A1, A2 - theta[:, 1]])

    points = np.array([rr_points(a, b) for a, b in theta])
    return dict(
        t=grid, theta=theta, q=q, xy=points[:, 2], elbow_xy=points[:, 1],
        home=np.asarray(home, dtype=float), l1=L1, l2=L2, rate=traj.rate,
    )


def guided_session(arm, home, folder, plot, ctrl_c):
    """One free drive: position, record, save."""
    from xarm7_lib.free_drive import FREE_JOINTS

    plot.reset("positioning — press enter to start recording")
    flush_stdin()  # an extra enter from before must not start the recording
    print("[goto] free drive: move the arm to where the trajectory should "
          "start,\n       then press enter to record and enter again to stop.")
    state = {"t_start": None}

    def on_sample(t, q):
        if ctrl_c.requested:
            return True
        pressed = enter_pressed()
        if pressed and state["t_start"] is None:
            state["t_start"] = t
            print("[goto] recording — press enter to stop")
            pressed = False
            # A fresh plot for each recording: only this one's points.
            plot.reset("recording — press enter to stop")
        plot.update(rr_angles(q), record=state["t_start"] is not None)
        return pressed

    def hands_off(message):
        # The library's own prompt blocks in input(); this one sees ctrl-c.
        print(message)
        return wait_for_enter("        press enter when your hands are clear: ",
                              ctrl_c, plot)

    # math.inf: the run ends when `on_sample` says so, not on a clock.
    traj = arm.free_drive(FREE_JOINTS, math.inf, on_sample=on_sample,
                          confirm=hands_off)
    print(f"[goto] {traj}")
    if arm.robot.has_error:  # `Robot` doesn't forward these; the real arm has them
        print("[goto] the controller latched an error during free drive; "
              "clearing it.")
        arm.robot.clear_errors()

    if state["t_start"] is None or not np.any(traj.t >= state["t_start"]):
        print("[goto] recording never started; nothing saved.")
        plot.draw("not recorded", force=True)
        return

    recording = rr_recording(traj, state["t_start"], home)
    out = default_recording_path(folder)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **recording)
    t = recording["t"]
    print(f"[goto] {t.size} samples over {t[-1]:.1f}s written to {out}")
    print(f"[goto] sampled at {traj.rate:.0f} Hz; the controller's report was "
          f"seen changing at {traj.report_rate:.0f} Hz")
    if traj.interruptions:
        print(f"[goto] {traj.interruptions} interruption(s) were closed up in t")
    plot.draw(f"saved {out.name} — {t.size} samples", force=True)


def return_home(arm, home, speed, ctrl_c, plot):
    """Drive back to `home`. None once there, or the exit code if not."""
    print(f"[goto] returning to {degrees(home)} deg")
    if not preflight(arm.joint_values, home):
        print("[goto] the way back isn't clear from where the arm was "
              "left.\n       Leaving it as it is — move it clear and "
              "re-run.")
        return 2
    if not wait_for_enter(
        f"[goto] hands clear — press enter to drive back at {speed} rad/s. ",
        ctrl_c, plot,
    ):
        return 0 if ctrl_c.requested else 1
    with ctrl_c.moving():
        reached, code = move_to(arm, home, speed)
    if code is not None:
        return code
    return None if reached else 1


def guided_loop(arm, home, args):
    """Record, return home, and go again, until ctrl-c. Returns the exit code."""
    offenders = out_of_plane(arm.joint_values)
    if offenders:
        names = ", ".join(f"joint{i + 1}" for i in offenders)
        print(
            f"[goto] refusing to free-drive: {names} not within "
            f"{math.degrees(_PLANAR_TOLERANCE):.0f} deg of +-90, so joints 4 "
            "and 7 are not vertical here.\n"
            "       Run without --joints to get the planar pose first."
        )
        return 1

    print("[goto] free drive: joints 1, 4 and 7 can be pushed by hand; 2, 3, 5 "
          "and 6 are\n       watched, and the arm stops to put them back if "
          "they drift.\n       After each recording the arm drives home and "
          "free drive starts again.\n       Ctrl-c to finish.")
    folder = Path(args.out) if args.out else RECORDINGS
    plot = None
    with CtrlC() as ctrl_c:
        try:
            if not wait_for_enter("[goto] hands clear, then press enter to "
                                  "start free drive. ", ctrl_c):
                return 0 if ctrl_c.requested else 1
            while True:
                if plot is None or plot.closed:
                    plot = LivePlot()
                if not ctrl_c.requested:
                    guided_session(arm, home, folder, plot, ctrl_c)
                if ctrl_c.requested:
                    # Free drive has already handed the arm back to position
                    # control, so it is standing still wherever it was left.
                    print("[goto] done. The arm is stopped and holding where it is.")
                    return 0
                code = return_home(arm, home, RETURN_SPEED_CAP, ctrl_c, plot)
                if code is not None:
                    return code
        except KeyboardInterrupt:  # a second ctrl-c, or one during a move
            arm.stop()
            print("\n[goto] stopped. The arm is holding where it is.")
            return 130


# ----------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    goal = np.radians(DEFAULT_TARGET_DEG)
    np.set_printoptions(precision=3, suppress=True)
    check_fk(goal)

    arm = connect()
    # `Robot` isn't a context manager itself; the backend it wraps is, and its
    # exit is what puts the arm back in position control and disconnects.
    with arm.robot:
        start = arm.joint_values
        print(f"[goto] now at  {degrees(start)} deg")
        print(f"[goto] going to {degrees(goal)} deg at {DEFAULT_SPEED} rad/s")

        if not preflight(start, goal):
            print("[goto] refusing to command a pose the collision model rejects.")
            return 2

        if not confirm("Clear the workspace. Move now?"):
            return 1

        reached, code = move_to(arm, goal, DEFAULT_SPEED)
        if code is not None:
            return code
        if not reached:
            print("[goto] not starting free drive: the arm never reached the pose.")
            return 1

        # ---- free drive: record, return home, repeat until ctrl-c -------
        return guided_loop(arm, goal, args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # outside the guided session: the arm's own
        print("\n[goto] interrupted.")  # context has stopped it on the way out
        sys.exit(130)
