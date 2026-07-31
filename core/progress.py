"""
Progress reporting for a single-image analysis.

The progress bar used to advance on hard-coded percentages that bore no relation to where the time
actually goes: 5 % covered the whole detection phase (99 % of the elapsed time) and the last 25 %
covered work taking about a tenth of a second. The weights below replace them with measured ones.

Two pieces live here:

* :class:`ProgressReporter` -- maps a *local* 0..1 fraction onto a sub-range of the global bar, and
  can carve further sub-ranges out of itself. Each level of the analysis therefore only needs to
  know its own internal weights, never its position in the whole run.
* :data:`STAGE_SECONDS` -- the measured cost of every stage, per detection method.

**The weights are stored as measured seconds, not as normalised fractions**, so they stay traceable
to the measurement they came from and can be re-pasted after a re-measurement without anyone having
to re-normalise by hand. :func:`stage_bounds` normalises them.

The numbers were obtained by wrapping every stage boundary in a separate process, so no
instrumentation could leak into the application: once at stage level and once inside the two stages
that dominate, both on a 1024x1024 image yielding 5 nuclei and 51 foci.

**These are warm-run weights, by decision.** The first analysis in a process additionally pays
~8 s of numba JIT compilation, which lands almost entirely in ellipse parameter calculation -- 24 %
of a cold run against 0.04 % of a warm one. No static weighting can serve both, and modelling the
cold case was deliberately left out of scope. The visible consequence is that on the first analysis
after launch the bar sits still at the ellipse step for ~5 s.

Stage shares scale with nucleus and foci count, not only with pixel count, so on an image with many
more nuclei the per-ROI stages would deserve more weight than they get here. The bar is an estimate
and is allowed to be wrong; the monotonicity clamp on the GUI side is what keeps a wrong estimate
from looking like a bug.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

# Stage keys, in the order they execute. LOAD covers everything before detection (metadata, hash,
# pixel load, channel split); the last three run in NucDetect.analyze_image after the Detector has
# returned, which is why this table is the single source of truth for both sides.
LOAD = "load"
NUCLEUS = "nucleus"
FOCI_IP = "foci_ip"
FOCI_ML = "foci_ml"
MERGE = "merge"
QUALITY = "quality"
ELLIPSE = "ellipse"
DATABASE = "database"
TABLE = "table"

STAGE_ORDER = (LOAD, NUCLEUS, FOCI_IP, FOCI_ML, MERGE, QUALITY, ELLIPSE, DATABASE, TABLE)

# Measured warm-run seconds per stage and detection method. A stage that does not run under a given
# method is 0.0 and collapses to a point on the bar, which is exactly right.
STAGE_SECONDS: Dict[str, Dict[str, float]] = {
    "image processing": {
        LOAD: 0.025, NUCLEUS: 7.377, FOCI_IP: 4.668, FOCI_ML: 0.0, MERGE: 0.0,
        QUALITY: 0.048, ELLIPSE: 0.005, DATABASE: 0.067, TABLE: 0.021,
    },
    "u-net": {
        LOAD: 0.024, NUCLEUS: 7.583, FOCI_IP: 0.0, FOCI_ML: 21.221, MERGE: 0.0,
        QUALITY: 0.033, ELLIPSE: 0.005, DATABASE: 0.060, TABLE: 0.022,
    },
    "combined": {
        # MERGE covers merge_overlapping_foci (0.379) plus get_match_for_nuclei (0.069)
        LOAD: 0.024, NUCLEUS: 7.550, FOCI_IP: 4.749, FOCI_ML: 20.105, MERGE: 0.448,
        QUALITY: 0.044, ELLIPSE: 0.005, DATABASE: 0.080, TABLE: 0.022,
    },
}

# Sub-stage weights inside nucleus extraction, as measured seconds. Sum matches
# STAGE_SECONDS[*][NUCLEUS]; the mapper normalises them the same way stage_bounds does.
NUCLEUS_SECONDS = {
    "threshold": 0.057,
    "edm": 0.164,
    "itermax": 5.590,
    "centers": 0.779,
    "watershed": 0.193,
    "extract": 0.593,
}

# ...and inside get_iterative_max_map, which alone is 76 % of nucleus extraction. "loop" is the
# whole `while` loop regardless of the configured iteration count -- each iteration costs the same,
# so the reporter divides this span evenly by however many iterations the settings ask for.
ITERMAX_SECONDS = {
    "seed": 0.469,
    "loop": 4.447,
    "threshold_local": 0.133,
    "fill": 0.050,
    "opening": 0.490,
}

# Inside image-processing foci detection, per channel. preprocess is ~0 with the default settings
# (smoothing and background reduction both off) but is not free when they are enabled, so it keeps a
# nominal share rather than none at all.
FOCI_IP_SECONDS = {
    "preprocess": 0.05,
    "blob_log": 3.967,
    "extract": 0.701,
}


def stage_bounds(method: str) -> Dict[str, tuple]:
    """Return ``{stage: (lo, hi)}`` covering 0..1, normalised from the measured seconds.

    :param method: The detection method, as stored in ``analysis_settings["method"]``
    :return: The bar range belonging to each stage, in execution order
    """
    seconds = STAGE_SECONDS.get(method, STAGE_SECONDS["image processing"])
    total = sum(seconds.values()) or 1.0
    bounds = {}
    cursor = 0.0
    for stage in STAGE_ORDER:
        share = seconds.get(stage, 0.0) / total
        bounds[stage] = (cursor, cursor + share)
        cursor += share
    return bounds


def _bounds_from_seconds(seconds: Dict[str, float], order) -> Dict[str, tuple]:
    """Normalise an ordered mapping of measured seconds into 0..1 sub-ranges."""
    total = sum(seconds.values()) or 1.0
    bounds = {}
    cursor = 0.0
    for key in order:
        share = seconds.get(key, 0.0) / total
        bounds[key] = (cursor, cursor + share)
        cursor += share
    return bounds


NUCLEUS_BOUNDS = _bounds_from_seconds(
    NUCLEUS_SECONDS, ("threshold", "edm", "itermax", "centers", "watershed", "extract"))
ITERMAX_BOUNDS = _bounds_from_seconds(
    ITERMAX_SECONDS, ("seed", "loop", "threshold_local", "fill", "opening"))
FOCI_IP_BOUNDS = _bounds_from_seconds(
    FOCI_IP_SECONDS, ("preprocess", "blob_log", "extract"))


class ProgressReporter:
    """Maps a local 0..1 fraction onto a sub-range of the global progress bar.

    A reporter with no callback is a no-op, so code that reports progress runs unchanged when
    nobody is listening -- batch analysis, the verification harnesses and any direct use of
    ``Detector`` outside the GUI all take that path. Callers therefore never need to test whether a
    reporter exists before using it.

    Sub-ranges nest without any level knowing its absolute position::

        root = ProgressReporter(cb)                  # owns 0.00 .. 1.00
        nucleus = root.sub(0.02, 0.63)               # owns 0.02 .. 0.63
        itermax = nucleus.sub(0.03, 0.79)            # owns 0.04 .. 0.49 of the bar
        itermax(0.5, "...")                          # emits ~0.26
    """

    __slots__ = ("_callback", "_lo", "_span")

    def __init__(self, callback: Optional[Callable[[float, str], None]] = None,
                 lo: float = 0.0, hi: float = 1.0):
        """
        :param callback: Called as ``callback(global_fraction, message)``. None makes this a no-op
        :param lo: Start of the range this reporter owns, as a fraction of the whole bar
        :param hi: End of that range
        """
        self._callback = callback
        self._lo = lo
        self._span = hi - lo

    def __call__(self, fraction: float, message: str = "") -> None:
        """Report progress within this reporter's range.

        :param fraction: How far through this reporter's own work, 0..1. Clamped
        :param message: The text to show above the bar
        :return: None
        """
        if self._callback is None:
            return
        fraction = 0.0 if fraction < 0.0 else 1.0 if fraction > 1.0 else fraction
        self._callback(self._lo + self._span * fraction, message)

    def sub(self, lo: float, hi: float) -> "ProgressReporter":
        """Carve a sub-range out of this reporter, addressed in *this* reporter's 0..1 space.

        :param lo: Start of the sub-range, as a fraction of this reporter's range
        :param hi: End of the sub-range
        :return: A reporter owning that slice of the bar
        """
        return ProgressReporter(self._callback,
                                self._lo + self._span * lo,
                                self._lo + self._span * hi)

    def span(self, key: str, bounds: Dict[str, tuple]) -> "ProgressReporter":
        """Convenience for :meth:`sub` using one of the measured bound tables above.

        :param key: The sub-stage name
        :param bounds: One of NUCLEUS_BOUNDS / ITERMAX_BOUNDS / FOCI_IP_BOUNDS
        :return: A reporter owning that sub-stage's slice of the bar
        """
        lo, hi = bounds[key]
        return self.sub(lo, hi)


#: A reporter that discards everything. Used as the default wherever progress is optional.
NO_PROGRESS = ProgressReporter()
