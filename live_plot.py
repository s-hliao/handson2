"""The live top-down plot of the planar RR arm, drawn with the student's FK.

One view, used twice:

    LivePlot()              `goto.py --guided`: the arm as it is hand-guided,
                            and the path recorded.
    LivePlot(planned=True)  `replay.py`: the same, plus the recorded path it
                            is meant to follow.

The arm is drawn from the planar RR angles alone: `robot_info.py` gives the
link lengths, and `fk.py` turns the angles into points. Callers holding a
7-joint configuration convert it with `robot_info.rr_angles` first.
"""
import matplotlib.pyplot as plt
import numpy as np
import time

from xarm7_lib.safety import DEFAULT_BOX

from fk import forward_kinematics_RR

_REDRAW_PERIOD = 0.05  # s between redraws; goto's free-drive loop samples at 100 Hz


def arm_points(q):
    """Base, elbow and end effector at `q` = (theta1, theta2), (3, 2) m, by the
    student's FK.

    The two frames the base can see: `H_2_0` is the elbow and `H_4_0` the end
    effector, so the arm is those two origins hung off the base.
    """
    H = forward_kinematics_RR(*q)
    return np.array([[0.0, 0.0], H["H_2_0"][:2, 2], H["H_4_0"][:2, 2]])


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
    `goto.py` does, from inside the free-drive watch loop.
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
        self.lines = [line for line in (self.planned, self.trace, self.links)
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
        if planned is not None:
            self.planned.set_data(*np.asarray(planned).T[::-1])
        self.draw(title, force=True)

    def update(self, q, record=True, title=None, force=False):
        """Put the arm at `q` = (theta1, theta2), its end effector on the path
        if `record`."""
        points = arm_points(q)
        self.links.set_data(points[:, 1], points[:, 0])  # (y, x)
        if record:
            self.visited.append(points[2])
        self.draw(title, force)

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
