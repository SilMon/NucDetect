import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtProperty, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSizePolicy


class SettingsWidget(QWidget):
    type = None
    value = None
    _title = None
    _description = None
    changed = pyqtSignal()

    def __init__(self, _type, value, ui_file, title="", desc="", parent=None):
        super(SettingsWidget, self).__init__(parent)
        self.type = _type
        self.value = value
        self._title = title
        self._description = desc
        """
        self.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        """
        self._initialize_ui(ui_file)

    def _initialize_ui(self, ui_file):
        self.ui = uic.loadUi(os.path.join(os.getcwd(), os.path.join("settings", ui_file)), self)
        self.ui.title.setText(self._title)
        self.ui.description.setText(self._description)


class SettingsShowWidget(SettingsWidget):
    """
    Class to show a custom widgets in the settings.
    """
    def __init__(self, value, parent=None, title="", desc="",):
        """
        Constructor of SettingsShoWidget
        :param value: The QWidget to show
        :param parent:
        """
        super(SettingsShowWidget, self).__init__("ShowWidget", value,
                                                 "menu_show.ui", title, desc, parent)
        self.ui.vl_widget.addWidget(self.value)


class SettingsTextWidget(SettingsWidget):
    """
    Class to show an text input in the settings
    """

    def __init__(self, value, parent=None, title="", desc="",):
        super(SettingsTextWidget, self).__init__("TextWidget", value,
                                                 "menu_text.ui", title, desc, parent)
        self.ui.text.setText(value)
        self.text = self.ui.text
        self.text.editingFinished.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.text.text()
        self.changed.emit()


class SettingsSlider(SettingsWidget):
    """
    Class to show an slider in the settings
    """

    def __init__(self, min_val, max_val, value, parent=None, title="", desc="", step=1):
        super(SettingsSlider, self).__init__("SliderWidget", value,
                                             "menu_slider.ui", title, desc, parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.slider = self.ui.slider
        self.slider.setMinimum(self.min_val)
        self.slider.setMaximum(self.max_val)
        self.slider.setSingleStep(self.step)
        self.slider.sliderReleased.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.ui.slider.getValue()
        self.changed.emit()


class SettingsComboBox(SettingsWidget):
    """
    Class to show a combo box in the settings
    """

    def __init__(self, data, value, parent=None, title="", desc=""):
        super(SettingsComboBox, self).__init__("SliderComboBox", value, "menu_combo.ui",
                                               title, desc, parent)
        self.data = data
        self.combo = self.ui.combo
        for item in self.data:
            self.combo.addItem(item)
        self.combo.setCurrentText(value)
        self.combo.currentIndexChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.combo.currentText()
        self.changed.emit()
