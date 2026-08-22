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
that dominate, both on a 1024x1024 image.

**RE-MEASURED 2026-08-22, and the previous set is superseded.** Replacing the rank filter in
``NucleusMapper.get_iterative_max_map`` with an equivalent dilation cut that method from 5.590 s to
0.527 s, which moved nucleus extraction from the largest stage of an image-processing run to a
third of the size of foci detection. Weights derived from the old measurement would have raced the
bar through nucleus extraction and then stalled it, so re-measuring was part of that change rather
than a follow-up.

**Read the absolute seconds as a ratio, not as a clock.** The whole 2026-08-22 set is roughly twice
as fast as the 2026-07-31 set, and only part of that is the code: ``blob_log`` alone went 3.967 s to
1.914 s without being touched. Wall-clock on this machine is not comparable across process
invocations -- the same code measured 6.9 s and 13.1 s hours apart on 2026-08-21 -- which is
precisely why every stage here is re-measured in ONE session whenever any of them changes. Mixing
figures from two sessions would produce weights that describe no run that ever happened.

The reference image yields 5 nuclei and 58 foci today, against 5 and 51 on 2026-07-31. That drift is
not from the progress work: detection output was deliberately changed twice in between (the
Butterworth fix and the 2026-08-17 focus-association rule).

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
        LOAD: 0.012, NUCLEUS: 0.941, FOCI_IP: 2.164, FOCI_ML: 0.0, MERGE: 0.0,
        QUALITY: 0.013, ELLIPSE: 0.002, DATABASE: 0.034, TABLE: 0.006,
    },
    "u-net": {
        LOAD: 0.011, NUCLEUS: 0.937, FOCI_IP: 0.0, FOCI_ML: 6.760, MERGE: 0.0,
        QUALITY: 0.010, ELLIPSE: 0.002, DATABASE: 0.034, TABLE: 0.006,
    },
    "combined": {
        # MERGE covers merge_overlapping_foci (0.088) plus get_match_for_nuclei (0.017)
        LOAD: 0.013, NUCLEUS: 1.032, FOCI_IP: 2.271, FOCI_ML: 7.334, MERGE: 0.104,
        QUALITY: 0.013, ELLIPSE: 0.002, DATABASE: 0.036, TABLE: 0.007,
    },
}

# Sub-stage weights inside nucleus extraction, as measured seconds. Sum matches
# STAGE_SECONDS[*][NUCLEUS]; the mapper normalises them the same way stage_bounds does.
NUCLEUS_SECONDS = {
    "threshold": 0.026,
    "edm": 0.078,
    "itermax": 0.527,
    "centers": 0.336,
    "watershed": 0.083,
    "extract": 0.250,
}

# ...and inside get_iterative_max_map, which alone is 76 % of nucleus extraction. "loop" is the
# whole `while` loop regardless of the configured iteration count -- each iteration costs the same,
# so the reporter divides this span evenly by however many iterations the settings ask for.
ITERMAX_SECONDS = {
    "seed": 0.021,
    "loop": 0.207,
    "threshold_local": 0.060,
    "fill": 0.025,
    "opening": 0.214,
}

# Inside image-processing foci detection, per channel. preprocess is ~0 with the default settings
# (smoothing and background reduction both off) but is not free when they are enabled, so it keeps a
# nominal share rather than none at all.
FOCI_IP_SECONDS = {
    "preprocess": 0.05,
    "blob_log": 1.914,
    "extract": 0.327,
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
