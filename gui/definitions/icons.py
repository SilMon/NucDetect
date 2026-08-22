import os
from typing import Dict

import qtawesome as qta
from PyQt5 import QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor

import gui.Paths as gpaths
from core.logging_config import get_logger

LOGGER = get_logger(__name__)

#: Returned when an identifier is not known. It is a real, visible icon rather than an empty one so
#: that a typo shows up in the interface instead of leaving a blank the reader takes for intentional
FALLBACK_ICON_IDENT = "QUESTION"

#: Returned when a size identifier is not known
DEFAULT_ICON_SIZE = QSize(75, 75)


class Color:
    ITEM_ANALYSED = QColor("#5A597387")
    ITEM_MODIFIED = QColor("#5A8c9ea3")
    #: Text colour for the image(s) whose results the table is currently showing. A FOREGROUND
    #: colour on purpose: the two above are backgrounds carrying the analysed/modified state, and
    #: "is being shown" is orthogonal to both -- an image can be any combination of the three, so
    #: reusing the background would let one state hide another. Paired with a bold font in
    #: NucDetect._mark_displayed_images
    ITEM_DISPLAYED = QColor("#ffd479")
    #: The unmarked text colour. QStandardItem has no "clear the foreground" call, so the default
    #: has to be stated rather than omitted. White, matching the stylesheet's QWidget colour
    ITEM_DEFAULT_TEXT = QColor("#ffffff")
    BRIGHT_RED = QColor(222, 23, 56)
    LIGHT_BLUE = QColor(47, 167, 212)
    INVISIBLE = QColor(0, 0, 0, 0)


class Icon:
    STANDARD = Color.LIGHT_BLUE
    STANDARD_OFF = Color.LIGHT_BLUE.darker()
    HIGHLIGHT = Color.BRIGHT_RED

    # Filled by _build_icons on the first get_icon call and kept for the lifetime of the process.
    # It is NOT a module-level literal: qtawesome renders each icon through a QPixmap, which
    # requires a live QApplication, and this module is imported before one exists
    _icons: Dict[str, QIcon] = {}

    @staticmethod
    def get_icon(ident: str) -> QIcon:
        """
        Method to get a predefined icon

        An unknown identifier is logged as an error and answered with the fallback icon. It is not
        raised: every identifier in the program is a string literal, so an unknown one is a typo in
        the source rather than bad input, and taking a dialog down over a decoration costs more than
        it reports. It is not answered with an empty icon either -- that was the silent failure this
        replaces, because a blank space in the interface reads as intentional

        :param ident: The identifier of the icon
        :return: The icon registered for the identifier, or the fallback icon
        """
        if not Icon._icons:
            Icon._icons = Icon._build_icons()
        icon = Icon._icons.get(ident)
        if icon is None:
            LOGGER.error("Unknown icon identifier %r -- falling back to %s",
                         ident, FALLBACK_ICON_IDENT)
            return Icon._icons[FALLBACK_ICON_IDENT]
        return icon

    @staticmethod
    def _build_icons() -> Dict[str, QIcon]:
        """
        Method to render every predefined icon once

        Called by get_icon on first use. Building the dictionary renders 33 qtawesome icons and
        loads a PNG, measured at 4.06 ms, so it must happen once per process and not once per
        lookup -- the image list asks for an icon per row

        :return: A dict mapping each identifier to its icon
        """
        return {
            "LOGO": QtGui.QIcon(gpaths.logo_dir + os.sep + "logo.png"),
            "RULER": qta.icon("fa5s.ruler", color=Icon.STANDARD),
            "EYE": qta.icon("fa5.eye", color=Icon.STANDARD),
            "EYE_OFF": qta.icon("fa5.eye-slash", color=Icon.STANDARD_OFF),
            "OBJECT_GROUP": qta.icon("fa5.object-group", color=Icon.STANDARD),
            "EDIT": qta.icon("fa5.edit", color=Icon.STANDARD),
            "EDIT_OFF": qta.icon("fa5.edit", color=Icon.STANDARD_OFF),
            "CLIPBOARD": qta.icon("fa5.clipboard", color=Icon.STANDARD),
            "FOLDER_OPEN": qta.icon("fa5.folder-open", color=Icon.STANDARD),
            "FLASK": qta.icon("fa5s.flask", color=Icon.STANDARD),
            "SAVE": qta.icon("fa5.save", color=Icon.STANDARD),
            "MICROSCOPE": qta.icon("fa5s.microscope", color=Icon.STANDARD),
            "CHART_BAR": qta.icon("fa5.chart-bar", color=Icon.STANDARD),
            "LIST_UL": qta.icon("fa5s.list-ul", color=Icon.STANDARD),
            "COGS": qta.icon("fa.cogs", color=Icon.STANDARD),
            "TOOLS": qta.icon("fa5s.tools", color=Icon.STANDARD),
            "HAT_WIZARD_BLUE": qta.icon("fa5s.hat-wizard", color=Icon.STANDARD),
            "HAT_WIZARD_RED": qta.icon("fa5s.hat-wizard", color=Icon.HIGHLIGHT),
            "TIMES": qta.icon("fa5s.times", color=Icon.STANDARD),
            "TRASH_ALT": qta.icon("fa5s.trash-alt", color=Icon.STANDARD),
            "SYNC": qta.icon("fa5s.sync", color=Icon.STANDARD),
            "PLUS_CIRCLE": qta.icon("fa5s.plus-circle", color=Icon.STANDARD),
            "CIRCLE": qta.icon("fa5.circle", color=Icon.STANDARD),
            "DOT_CIRCLE": qta.icon("fa5.dot-circle", color=Icon.STANDARD),
            "DRAFTING_COMPASS": qta.icon("fa5s.drafting-compass", color=Icon.STANDARD),
            "MOUSE": qta.icon("fa5s.mouse-pointer", color=Icon.STANDARD),
            "CHECK": qta.icon("fa5s.check", color=Icon.HIGHLIGHT),
            "MAGIC": qta.icon("fa5s.magic", color=Icon.HIGHLIGHT),
            "LOCK": qta.icon("fa5s.lock", color=Icon.STANDARD),
            "UNDO": qta.icon("fa5s.undo", color=Icon.STANDARD),
            "IMAGE": qta.icon("fa5s.image", color=Icon.STANDARD),
            "SEARCH": qta.icon("fa5s.search", color=Icon.STANDARD),
            "QUESTION": qta.icon("fa5s.question", color=Icon.STANDARD)
        }

    @staticmethod
    def get_icon_size(ident: str) -> QSize:
        """
        Method to get the display size registered for an identifier

        The default happens to equal the only registered size, so an unknown identifier returns the
        right answer by accident and cannot be noticed from the interface. Logging it is therefore
        the whole of the report -- see get_icon for why this does not raise

        :param ident: The identifier of the size
        :return: The size registered for the identifier, or the default size
        """
        sizes = {
            "LIST_ITEM": DEFAULT_ICON_SIZE
        }
        size = sizes.get(ident)
        if size is None:
            LOGGER.error("Unknown icon size identifier %r -- falling back to %dx%d",
                         ident, DEFAULT_ICON_SIZE.width(), DEFAULT_ICON_SIZE.height())
            size = DEFAULT_ICON_SIZE
        # A copy: QSize is mutable, and handing out the module constant would let any caller resize
        # every list in the program
        return QSize(size)
