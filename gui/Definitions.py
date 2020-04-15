import qtawesome as qta
from PyQt5 import QtGui
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor


class Icon:

    @staticmethod
    def get_icon(ident: str) -> QIcon:
        """
        Method to get a predefined icon

        :param ident: The identifier of the icon
        :return: QIcon
        """
        icons = {
            "LOGO": QtGui.QIcon('logo.png'),
            "CLIPBOARD": qta.icon("fa5.clipboard", color=Color.LIGHT_BLUE),
            "FOLDER_OPEN": qta.icon("fa5.folder-open", color=Color.LIGHT_BLUE),
            "FLASK": qta.icon("fa5s.flask", color=Color.LIGHT_BLUE),
            "SAVE": qta.icon("fa5.save", color=Color.LIGHT_BLUE),
            "MICROSCOPE": qta.icon("fa5s.microscope", color=Color.LIGHT_BLUE),
            "CHART_BAR": qta.icon("fa5.chart-bar", color=Color.LIGHT_BLUE),
            "LIST_UL": qta.icon("fa5s.list-ul", color=Color.LIGHT_BLUE),
            "COGS": qta.icon("fa.cogs", color=Color.LIGHT_BLUE),
            "TOOLS": qta.icon("fa5s.tools", color=Color.LIGHT_BLUE),
            "HAT_WIZARD_BLUE": qta.icon("fa5s.hat-wizard", color=Color.LIGHT_BLUE),
            "HAT_WIZARD_RED": qta.icon("fa5s.hat-wizard", color=Color.BRIGHT_RED),
            "TIMES": qta.icon("fa5s.times", color=Color.LIGHT_BLUE),
            "TRASH_ALT": qta.icon("fa5s.trash-alt", color=Color.LIGHT_BLUE),
            "SYNC": qta.icon("fa5s.sync", color=Color.LIGHT_BLUE)
        }
        return icons.get(ident, QIcon())

    @staticmethod
    def get_icon_size(ident: str) -> QSize:
        sizes = {
            "LIST_ITEM": QSize(75, 75)
        }
        return sizes.get(ident, QSize(75, 75))

class Color:
    BRIGHT_RED = QColor(222, 23, 56)
    LIGHT_BLUE = QColor(47, 167, 212)
