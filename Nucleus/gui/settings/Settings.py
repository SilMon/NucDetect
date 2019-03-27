import os

from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QSizePolicy


class SettingsWidget(QWidget):
    """
    Base class for all settings widgets
    """
    changed = pyqtSignal(str, list)

    def __init__(self, _id, _type, value, ui_file, title="", desc="", parent=None, callback=None):
        super(SettingsWidget, self).__init__(parent)
        self._id = _id
        self.type = _type
        self.value = value
        self.callback = callback
        self._title = title
        self._description = desc
        self.changed.connect(callback)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum
        )
        self._initialize_ui(ui_file)

    def _initialize_ui(self, ui_file):
        self.ui = uic.loadUi(os.path.join(os.getcwd(), os.path.join("settings", ui_file)), self)
        self.ui.title.setText(self._title)
        self.ui.description.setText(self._description)

    def _change_emit(self):
        self.changed.emit(self._id, [self.value])


class SettingsShowWidget(SettingsWidget):
    """
    Class to show a custom widgets in the settings.
    """
    def __init__(self, _id, value, parent=None, title="", desc="", callback=None):
        """
        Constructor of SettingsShoWidget
        :param value: The QWidget to show
        :param parent:
        """
        super(SettingsShowWidget, self).__init__(_id, "ShowWidget", value,
                                                 "menu_show.ui", title, desc, parent, callback)
        self.ui.vl_widget.addWidget(self.value)


class SettingsText(SettingsWidget):
    """
    Class to show an text input in the settings
    """

    def __init__(self, _id, value, parent=None, title="", desc="", callback=None):
        super(SettingsText, self).__init__(_id, "TextWidget", value,
                                           "menu_text.ui", title, desc, parent, callback)
        self.ui.text.setText(value)
        self.text = self.ui.text
        self.text.editingFinished.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.text.text()
        super(SettingsText, self)._change_emit()


class SettingsSlider(SettingsWidget):
    """
    Class to show an slider in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, unit="%", callback=None):
        super(SettingsSlider, self).__init__(_id, "SliderWidget", value,
                                             "menu_slider.ui", title, desc, parent, callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        self.slider = self.ui.slider
        self.ui.val.setText("{} {}".format(value, unit))
        self.slider.setMinimum(self.min_val)
        self.slider.setMaximum(self.max_val)
        self.slider.setSingleStep(self.step)
        self.slider.setValue(self.value)
        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        val = self.slider.value()
        if val % self.step != 0:
            self.value = val - val % self.step
            self.slider.setValue(val - val % self.step)
        else:
            self.value = val
        self.ui.val.setText("{} {}".format(self.value, self.unit))
        super(SettingsSlider, self)._change_emit()


class SettingsDial(SettingsWidget):
    """
    Class to show a dial in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, unit="%", callback=None):
        super(SettingsDial, self).__init__(_id, "DialWidget", value,
                                           "menu_dial.ui", title, desc, parent, callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        self.dial = self.ui.dial
        self.ui.val.setText("{} {}".format(value, unit))
        self.dial.setMinimum(self.min_val)
        self.dial.setMaximum(self.max_val)
        self.dial.setSingleStep(step)
        self.dial.setValue(self.value)
        self.dial.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        val = self.dial.value()
        if val % self.step != 0:
            self.value = val - val % self.step
            self.dial.setValue(val - val % self.step)
        else:
            self.value = val
        self.ui.val.setText("{} {}".format(self.value, self.unit))
        super(SettingsDial, self)._change_emit()


class SettingsSpinner(SettingsWidget):
    """
    Class to show an integer spinner in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, prefix="", suffix="%",
                 callback=None):
        super(SettingsSpinner, self).__init__(_id, "IntegerSpinnerWidget", value, "menu_spin.ui",
                                              title, desc, parent, callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prefix = prefix
        self.suffix = suffix
        self.spin = self.ui.spin
        self.spin.setMinimum(self.min_val)
        self.spin.setMaximum(self.max_val)
        self.spin.setPrefix(self.prefix)
        self.spin.setSuffix(self.suffix)
        self.spin.setSingleStep(step)
        self.spin.setValue(self.value)
        self.spin.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        val = self.spin.value()
        if val % self.step != 0:
            self.value = val - val % self.step
            self.spin.setValue(val - val % self.step)
        else:
            self.value = val
        super(SettingsSpinner, self)._change_emit()


class SettingsDecimalSpinner(SettingsWidget):
    """
    Class to show an integer spinner in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1.0, decimals=2,
                 prefix="", suffix="%", callback=None):
        super(SettingsDecimalSpinner, self).__init__(_id, "DecimalSpinnerWidget", value, "menu_decimal_spin.ui",
                                                     title, desc, parent, callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.spin = self.ui.spin
        self.spin.setMinimum(self.min_val)
        self.spin.setMaximum(self.max_val)
        self.spin.setPrefix(self.prefix)
        self.spin.setSuffix(self.suffix)
        self.spin.setSingleStep(step)
        self.spin.setValue(self.value)
        self.spin.setDecimals(self.decimals)
        self.spin.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        val = self.dial.value()
        if val % self.step != 0:
            self.value = val - val % self.step
            self.spin.setValue(val - val % self.step)
        else:
            self.value = val
        super(SettingsDecimalSpinner, self)._change_emit()


class SettingsComboBox(SettingsWidget):
    """
    Class to show a combo box in the settings
    """

    def __init__(self, _id, data, value, parent=None, title="", desc="", callback=None):
        super(SettingsComboBox, self).__init__(_id, "ComboBoxWidget", value, "menu_combo.ui",
                                               title, desc, parent, callback)
        self.data = data
        self.combo = self.ui.combo
        for item in self.data:
            self.combo.addItem(item)
        self.combo.setCurrentText(value)
        self.combo.currentIndexChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.combo.currentText()
        super(SettingsComboBox, self)._change_emit()


class SettingsCheckBox(SettingsWidget):
    """
    Class to show a checkbox in the settings
    """

    def __init__(self, _id, value, parent=None, title="", desc="", tristate=False, callback=None):
        super(SettingsCheckBox, self).__init__(_id, "CheckBoxWidget", value, "menu_checkbox.ui",
                                               title, desc, parent, callback)
        self.ui.check.setCheckState(value)
        self.ui.check.setTristate(tristate)
        self.ui.check.stateChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.ui.check.checkState()
        super(SettingsCheckBox, self)._change_emit()


