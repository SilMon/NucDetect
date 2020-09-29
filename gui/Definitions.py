import qtawesome as qta
from PyQt5 import QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor


class Color:
    BRIGHT_RED = QColor(222, 23, 56)
    LIGHT_BLUE = QColor(47, 167, 212)


class Icon:
    STANDARD = Color.LIGHT_BLUE
    STANDARD_OFF = Color.LIGHT_BLUE.darker()
    HIGHLIGHT = Color.BRIGHT_RED

    @staticmethod
    def get_icon(ident: str) -> QIcon:
        """
        Method to get a predefined icon

        :param ident: The identifier of the icon
        :return: QIcon
        """
        icons = {
            "LOGO": QtGui.QIcon("logo.png"),
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
            "CHECK": qta.icon("fa5s.check", color=Icon.HIGHLIGHT)
        }
        return icons.get(ident, QIcon())

    @staticmethod
    def get_icon_size(ident: str) -> QSize:
        sizes = {
            "LIST_ITEM": QSize(75, 75)
        }
        return sizes.get(ident, QSize(75, 75))
