"""The live top-down plot of the planar RR arm, drawn with the student's FK.

One view, used twice:

    LivePlot()              `record.py`: the arm as it is hand-guided,
                            and the path recorded.
    LivePlot(planned=True)  `replay.py`: the same, plus the recorded path it
                            is meant to follow.

The arm is drawn from the planar RR angles alone: `robot_info.py` gives the
link lengths, and `fk.py` turns the angles into points. Callers holding a
7-joint configuration convert it with `robot_info.rr_angles` first.

Hand a pair of joint *velocities* in alongside the angles and the end
effector's own velocity is drawn there too, as an arrow off the tip — the
student's `jacobian_RR` applied to those joint velocities. `robot_info.qd2rr`
converts a 7-joint velocity the same way `q2rr` converts a configuration.
"""
import matplotlib.pyplot as plt
import numpy as np
import time

from xarm7_lib.safety import DEFAULT_BOX

from fk import forward_kinematics_RR, jacobian_RR

_REDRAW_PERIOD = 0.05  # s between redraws; record.py's free-drive loop runs at 100 Hz
# m of arrow per m/s of end-effector speed. Hand guiding runs at a few tenths
# of a m/s, so this draws those as an arrow a few cm long in a workspace that is
# most of a metre across — visible, without covering the arm it comes off.
_VELOCITY_SCALE = 0.3


def arm_points(q):
    """Base, elbow and end effector at `q` = (theta1, theta2), (3, 2) m, by the
    student's FK.

    The two frames the base can see: `H_2_0` is the elbow and `H_4_0` the end
    effector, so the arm is those two origins hung off the base.
    """
    H = forward_kinematics_RR(*q)
    return np.array([[0.0, 0.0], H["H_2_0"][:2, 2], H["H_4_0"][:2, 2]])


def eef_velocity(q, qd):
    """End-effector velocity at `q` = (theta1, theta2) with the joints turning at
    `qd` = (theta1_dot, theta2_dot), (2,) m/s, by the student's Jacobian.

    Which is all the Jacobian is for: the matrix at this pose, times the joint
    velocities, is how fast the tip is going and which way.
    """
    return jacobian_RR(*q) @ np.asarray(qd, dtype=float)


class LivePlot:
    """The RR arm from above, drawn with the student's FK, updated live.

    Drawn as seen by someone standing in front of the robot: the base at the
    top, the arm reaching down the screen towards them (+x down), and the
    robot's +y on their right. Every point is plotted as (y, x) on an inverted
    vertical axis, so the ticks still read the robot's own coordinates. The
    view is framed on the safety box's footprint.

    The moving lines are `animated`, so they stay out of the cached background
    and a redraw is a blit of that background plus those few lines: a few
    milliseconds instead of tens. `update` never redraws more often than
    `_REDRAW_PERIOD` either, so it can be called on every sample — which
    `record.py` does, from inside the free-drive watch loop.
    """

    def __init__(self, planned=False):
        plt.ion()
        (x_lo, x_hi), (y_lo, y_hi), _ = DEFAULT_BOX
        self.fig, self.ax = plt.subplots(figsize=(6, 6 * (x_hi - x_lo) / (y_hi - y_lo)))
        ax = self.ax
        ax.plot([y_lo, y_hi, y_hi, y_lo, y_lo], [x_lo, x_lo, x_hi, x_hi, x_lo],
                color="tab:red", lw=1, label="safety box")
        self.planned = None
        if planned:
            # The path to follow goes on top, so its dots still show where the
            # two overlap; the path taken is then blue, to tell them apart.
            (self.planned,) = ax.plot([], [], ":", color="tab:orange", lw=2.5,
                                      zorder=3, animated=True, label="planned")
        (self.trace,) = ax.plot([], [], "-", lw=1.5, animated=True,
                                color="black",
                                label="eef path (student FK)")
        (self.links,) = ax.plot([], [], "-o", color="tab:blue", lw=3, animated=True,
                                label="arm (student FK)")
        # Off the end effector, `_VELOCITY_SCALE` m long per m/s: `angles` and
        # `scale_units` of "xy" put both the arrow's direction and its length in
        # the data's own units, so it turns and stretches with the axes.
        self.velocity = ax.quiver([0.0], [0.0], [0.0], [0.0],
                                  angles="xy", scale_units="xy",
                                  scale=1.0 / _VELOCITY_SCALE, width=0.008,
                                  color="tab:green", zorder=4, animated=True)
        # The arrow stays out of the legend, which would key a quiver with a
        # plain green block; this readout names it instead, in the same green,
        # and only while there is an arrow to name.
        self.speed = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                             color="tab:green", fontsize="small", animated=True)
        self.lines = [line for line in (self.planned, self.trace, self.links,
                                        self.velocity, self.speed)
                      if line is not None]
        margin = 0.05
        ax.set_xlim(y_lo - margin, y_hi + margin)
        ax.set_ylim(x_hi + margin, x_lo - margin)  # inverted: +x points down
        ax.set(aspect="equal", xlabel="y (m)", ylabel="x (m)  — towards you")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")

        self.visited = []  # end-effector points, drawn through `trace`
        self.closed = False
        self._background = None
        self._drawn_at = -np.inf
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        ax.title.set_animated(True)  # so a title change is only a blit
        plt.show(block=False)
        plt.pause(0.1)

    def _on_close(self, _event):
        self.closed = True

    def _on_draw(self, _event):
        # Any full draw (the first, a resize, a new title) re-caches the
        # background the moving lines are blitted onto.
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._blit()

    def _blit(self):
        canvas = self.fig.canvas
        canvas.restore_region(self._background)
        for artist in [self.ax.title] + self.lines:
            self.ax.draw_artist(artist)
        canvas.blit(self.fig.bbox)

    def reset(self, title, planned=None):
        """Start a fresh path, under `title`, with `planned` the one to follow."""
        self.visited = []
        self.trace.set_data([], [])
        self._set_velocity(None, None)
        if planned is not None:
            self.planned.set_data(*np.asarray(planned).T[::-1])
        self.draw(title, force=True)

    def update(self, q, qd=None, record=True, title=None, force=False):
        """Put the arm at `q` = (theta1, theta2), its end effector on the path
        if `record`.

        `qd` = (theta1_dot, theta2_dot) in rad/s, if the caller has it, draws
        the end effector's velocity off the tip. Without it there is no arrow.
        """
        points = arm_points(q)
        self.links.set_data(points[:, 1], points[:, 0])  # (y, x)
        self._set_velocity(points[2], None if qd is None else eef_velocity(q, qd))
        if record:
            self.visited.append(points[2])
        self.draw(title, force)

    def _set_velocity(self, point, v):
        """Hang the velocity arrow `v` off `point`, or clear it if either is None."""
        if point is None or v is None:
            self.velocity.set_UVC([0.0], [0.0])
            self.speed.set_text("")
            return
        self.velocity.set_offsets([[point[1], point[0]]])  # (y, x)
        self.velocity.set_UVC([v[1]], [v[0]])
        self.speed.set_text(f"eef velocity (student J)  {np.hypot(*v):.3f} m/s")

    def draw(self, title=None, force=False):
        """Redraw, at most every `_REDRAW_PERIOD` unless `force`."""
        if title is not None:
            self.ax.set_title(title)
        now = time.perf_counter()
        if self.closed or (not force and now - self._drawn_at < _REDRAW_PERIOD):
            return
        self._drawn_at = now
        if self.visited:
            self.trace.set_data(*np.transpose(self.visited)[::-1])
        if self._background is None:
            self.fig.canvas.draw()  # full draw; `_on_draw` blits on top
        else:
            self._blit()
        self.fig.canvas.flush_events()

    def idle(self):
        """Let the window handle its events while nothing is being drawn."""
        if not self.closed:
            self.fig.canvas.flush_events()
