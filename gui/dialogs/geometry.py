"""
Geometry for the manual editor's direct-manipulation gestures

Deliberately free of Qt, so the arithmetic can be exercised without a QApplication and without a
window: these are the parts of a drag that are easy to get subtly wrong and impossible to eyeball,
because an error of a fraction of a pixel per event only becomes visible after a long gesture.

The editor's items keep their geometry as an **unrotated, axis-aligned** bounding box plus an angle
applied about that box's centre (see ROIItem.update_data). Resizing a rotated item therefore is not
a matter of moving a corner: the mouse delta has to be rotated into the item's own frame, the
unrotated box resized there, and the centre moved so that the corner OPPOSITE the one being dragged
stays where it is on screen. That last step is the whole reason this module exists.

Angles are in degrees, clockwise, matching QGraphicsItem.setRotation and the spb_angle spin box. The
image coordinate system has y increasing downwards, so "n" is the smaller y.
"""
import math
from typing import Dict, Tuple

# role -> (x, y) direction the handle sits in, from the centre of the box. A zero component means
# that dimension is not touched by this role, which is what makes the edge handles single-axis
HANDLE_SIGNS: Dict[str, Tuple[int, int]] = {
    "n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
    "nw": (-1, -1), "ne": (1, -1), "sw": (-1, 1), "se": (1, 1),
}

# An item smaller than this cannot be grabbed again, and rasterises to nothing
MIN_SIZE = 2.0


def to_local(dx: float, dy: float, angle: float) -> Tuple[float, float]:
    """
    Rotate a delta from image coordinates into the unrotated frame of an item

    :param dx: The x component of the delta, in image coordinates
    :param dy: The y component of the delta, in image coordinates
    :param angle: The item's angle in degrees, clockwise
    :return: The delta in the item's own frame
    """
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    return dx * cos + dy * sin, -dx * sin + dy * cos


def to_scene(dx: float, dy: float, angle: float) -> Tuple[float, float]:
    """
    Rotate a delta from the unrotated frame of an item back into image coordinates

    The inverse of to_local, and the reason both are here rather than inline: the anchor calculation
    below needs to go in both directions within one gesture

    :param dx: The x component of the delta, in the item's own frame
    :param dy: The y component of the delta, in the item's own frame
    :param angle: The item's angle in degrees, clockwise
    :return: The delta in image coordinates
    """
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    return dx * cos - dy * sin, dx * sin + dy * cos


def anchor_position(rect: Tuple[float, float, float, float], role: str,
                    angle: float) -> Tuple[float, float]:
    """
    The image position of the point a resize must hold still

    For a corner role that is the opposite corner; for an edge role the midpoint of the opposite
    edge. Exposed rather than kept private because it is what a test can assert on: the anchor is
    the invariant of the whole operation

    :param rect: The item's unrotated box as (x, y, width, height)
    :param role: One of HANDLE_SIGNS
    :param angle: The item's angle in degrees, clockwise
    :return: The anchor position in image coordinates
    """
    x, y, width, height = rect
    sign_x, sign_y = HANDLE_SIGNS[role]
    centre_x, centre_y = x + width / 2, y + height / 2
    local_x, local_y = -sign_x * width / 2, -sign_y * height / 2
    offset_x, offset_y = to_scene(local_x, local_y, angle)
    return centre_x + offset_x, centre_y + offset_y


def resize_about_anchor(rect: Tuple[float, float, float, float], role: str,
                        local_dx: float, local_dy: float, angle: float,
                        lock_aspect: bool = False) -> Tuple[float, float, float, float]:
    """
    Resize an unrotated box by one of its handles, holding the opposite corner or edge still

    :param rect: The box before the gesture, as (x, y, width, height)
    :param role: One of HANDLE_SIGNS -- which handle is being dragged
    :param local_dx: The x component of the mouse delta, already rotated into the item's frame
    :param local_dy: The y component of the mouse delta, already rotated into the item's frame
    :param angle: The item's angle in degrees, clockwise. Only the centre correction needs it; the
    box itself is resized in its own frame, where it is axis-aligned
    :param lock_aspect: Keep the width-to-height ratio. Applies to the CORNER roles only -- an edge
    handle is single-axis by definition, and making Shift change that would take away the one
    gesture that resizes exactly one dimension
    :return: The resized box as (x, y, width, height)
    """
    x, y, width, height = rect
    sign_x, sign_y = HANDLE_SIGNS[role]
    half_width, half_height = width / 2, height / 2

    # The handle sits at (sign_x * half_width, sign_y * half_height) in the item's frame; moving it
    # by the local delta grows that half-dimension by the component pointing away from the centre
    new_half_width = half_width + sign_x * local_dx if sign_x else half_width
    new_half_height = half_height + sign_y * local_dy if sign_y else half_height

    if lock_aspect and sign_x and sign_y and half_width > 0 and half_height > 0:
        # The mean of the two independent scale factors: following only the dominant axis makes the
        # box jump when the mouse crosses the diagonal
        scale = (new_half_width / half_width + new_half_height / half_height) / 2
        new_half_width, new_half_height = half_width * scale, half_height * scale

    new_half_width = max(MIN_SIZE / 2, new_half_width)
    new_half_height = max(MIN_SIZE / 2, new_half_height)

    # Hold the anchor still. In the item's own frame the anchor is at (-sign_x * half, -sign_y *
    # half), so it moves by (-sign_x * (new_half - half), ...) unless the centre is moved to
    # compensate -- rotated back into image coordinates, because that is where the centre lives
    shift_x = sign_x * (new_half_width - half_width)
    shift_y = sign_y * (new_half_height - half_height)
    offset_x, offset_y = to_scene(shift_x, shift_y, angle)
    centre_x = x + half_width + offset_x
    centre_y = y + half_height + offset_y
    return (centre_x - new_half_width, centre_y - new_half_height,
            new_half_width * 2, new_half_height * 2)


def angle_from_vector(centre_x: float, centre_y: float, position_x: float, position_y: float,
                      grab_offset: float = 0.0, snap: float = 0.0) -> float:
    """
    The angle of the line from a centre to a position, as the editor stores angles

    :param centre_x: The x coordinate of the item's centre
    :param centre_y: The y coordinate of the item's centre
    :param position_x: The x coordinate of the cursor
    :param position_y: The y coordinate of the cursor
    :param grab_offset: Subtracted from the raw angle, so that a rotation gesture starts from
    wherever the grip was grabbed rather than snapping the item to the cursor on the first event
    :param snap: When non-zero, the result is rounded to a multiple of this many degrees
    :return: The angle in degrees, clockwise, in [-180, 180]
    """
    angle = math.degrees(math.atan2(position_y - centre_y, position_x - centre_x)) - grab_offset
    if snap:
        angle = round(angle / snap) * snap
    # Normalised so that turning a full circle does not accumulate, and so the value stays inside
    # spb_angle's -360..360 range no matter how long the gesture runs
    return (angle + 180) % 360 - 180
