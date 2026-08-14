import os
from typing import Any

import gui.Paths as gpaths
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QSizePolicy

from core.logging_config import get_logger

LOGGER = get_logger(__name__)


def _snap_to_step(value, min_val, max_val, step):
    """
    Snap an integer widget value to the nearest multiple of step, measured from min_val

    Measuring from zero instead lands below the minimum whenever min_val is not itself a multiple
    of the step -- min=5, step=10 snaps to 0.

    :param value: The value reported by the widget
    :param min_val: The minimum of the widget
    :param max_val: The maximum of the widget
    :param step: The step size
    :return: The snapped value, clamped to [min_val, max_val], as int
    """
    if not step:
        return int(value)
    snapped = min_val + round((value - min_val) / step) * step
    return int(min(max(snapped, min_val), max_val))


def _as_check_state(value, tristate):
    """
    Convert a stored setting value into a Qt.CheckState

    QCheckBox.setCheckState takes the enum, not a bool: True is read as the enum value 1, which is
    PartiallyChecked rather than Checked.

    :param value: The stored value, a bool or one of 0/1/2
    :param tristate: Whether the box may hold the partially checked state
    :return: The matching Qt.CheckState
    """
    if isinstance(value, bool) or not tristate:
        return Qt.Checked if value else Qt.Unchecked
    if int(value) not in (Qt.Unchecked, Qt.PartiallyChecked, Qt.Checked):
        LOGGER.warning(f"Unknown check state {value!r}, falling back to a two-state reading")
        return Qt.Checked if value else Qt.Unchecked
    return Qt.CheckState(int(value))


class SettingsWidget(QWidget):
    """
    Base class for all settings widgets
    """
    changed = pyqtSignal(str, list)

    def __init__(self, _id, _type, value, ui_file, title="", desc="", parent=None, *, callback):
        # callback is keyword-only and mandatory: pyqtSignal.connect(None) raises TypeError, so a
        # widget built without one was never constructible in the first place
        super(SettingsWidget, self).__init__(parent)
        self._id = _id
        self.type = _type
        self.value = value
        self._title = title
        self._description = desc
        self.changed.connect(callback)
        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum
        )
        self._initialize_ui(ui_file)

    def _initialize_ui(self, ui_file):
        # Annotated Any deliberately -- see the comment on the same assignment in
        # gui/NucDetectAppQT.py: uic has no stubs, so the inferred type is "Unknown | None"
        self.ui: Any = uic.loadUi(os.path.join(gpaths.settings_path, ui_file), self)
        self.ui.title.setText(self._title)
        self.ui.description.setText(self._description)

    def _change_emit(self):
        self.changed.emit(self._id, [self.value])


class SettingsText(SettingsWidget):
    """
    Class to show a text input in the settings
    """

    def __init__(self, _id, value, parent=None, title="", desc="", *, callback):
        super(SettingsText, self).__init__(_id, "TextWidget", value,
                                           "menu_text.ui", title, desc, parent, callback=callback)
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

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, unit="%", *, callback):
        super(SettingsSlider, self).__init__(_id, "SliderWidget", value,
                                             "menu_slider.ui", title, desc, parent, callback=callback)
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
        self.value = _snap_to_step(self.slider.value(), self.min_val, self.max_val, self.step)
        if self.value != self.slider.value():
            # setValue re-enters this handler, so the correction would emit `changed` a second time
            # and the dialog would record the unsnapped value first
            self.slider.blockSignals(True)
            self.slider.setValue(self.value)
            self.slider.blockSignals(False)
        self.ui.val.setText("{} {}".format(self.value, self.unit))
        super(SettingsSlider, self)._change_emit()


class SettingsDial(SettingsWidget):
    """
    Class to show a dial in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, unit="%", *, callback):
        super(SettingsDial, self).__init__(_id, "DialWidget", value,
                                           "menu_dial.ui", title, desc, parent, callback=callback)
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
        self.value = _snap_to_step(self.dial.value(), self.min_val, self.max_val, self.step)
        if self.value != self.dial.value():
            # see SettingsSlider._on_value_changed
            self.dial.blockSignals(True)
            self.dial.setValue(self.value)
            self.dial.blockSignals(False)
        self.ui.val.setText("{} {}".format(self.value, self.unit))
        super(SettingsDial, self)._change_emit()


class SettingsSpinner(SettingsWidget):
    """
    Class to show an integer spinner in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1, prefix="", suffix="%",
                 *, callback):
        super(SettingsSpinner, self).__init__(_id, "IntegerSpinnerWidget", value, "menu_spin.ui",
                                              title, desc, parent, callback=callback)
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
        self.value = self.spin.value()
        super(SettingsSpinner, self)._change_emit()


class SettingsDecimalSpinner(SettingsWidget):
    """
    Class to show an integer spinner in the settings
    """

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1.0, decimals=2,
                 prefix="", suffix="%", *, callback):
        super(SettingsDecimalSpinner, self).__init__(_id, "DecimalSpinnerWidget", value, "menu_decimal_spin.ui",
                                                     title, desc, parent, callback=callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.spin = self.ui.spin
        # decimals first: QDoubleSpinBox rounds every bound and the value to the decimals in force
        # when they are set, and a later setDecimals does not restore the lost precision
        self.spin.setDecimals(self.decimals)
        self.spin.setMinimum(self.min_val)
        self.spin.setMaximum(self.max_val)
        self.spin.setPrefix(self.prefix)
        self.spin.setSuffix(self.suffix)
        self.spin.setSingleStep(step)
        self.spin.setValue(self.value)
        self.spin.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.spin.value()
        super(SettingsDecimalSpinner, self)._change_emit()


class SettingsComboBox(SettingsWidget):
    """
    Class to show a combo box in the settings
    """

    def __init__(self, _id, data, value, parent=None, title="", desc="", *, callback):
        super(SettingsComboBox, self).__init__(_id, "ComboBoxWidget", value, "menu_combo.ui",
                                               title, desc, parent, callback=callback)
        self.data = data
        self.combo = self.ui.combo
        for item in self.data:
            self.combo.addItem(item)
        # setCurrentText is a silent no-op on a non-editable box when the text is absent, which
        # would leave the box showing entry 0 while self.value still held the unknown value
        if self.combo.findText(value) == -1:
            LOGGER.warning(f"Setting {_id}: stored value '{value}' is not one of {self.data}, "
                           f"falling back to '{self.combo.itemText(0)}'")
            self.combo.setCurrentIndex(0)
            self.value = self.combo.currentText()
        else:
            self.combo.setCurrentText(value)
        self.combo.currentIndexChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.combo.currentText()
        super(SettingsComboBox, self)._change_emit()


class SettingsCheckBox(SettingsWidget):
    """
    Class to show a checkbox in the settings
    """

    def __init__(self, _id, value, parent=None, title="", desc="", tristate=False, *, callback):
        super(SettingsCheckBox, self).__init__(_id, "CheckBoxWidget", value, "menu_checkbox.ui",
                                               title, desc, parent, callback=callback)
        # tristate first, so it constrains the state that follows; and a bare bool passed to
        # setCheckState is read as the enum value 1 -- PartiallyChecked, not Checked
        self.ui.check.setTristate(tristate)
        self.ui.check.setCheckState(_as_check_state(value, tristate))
        self.ui.check.stateChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = self.ui.check.checkState()
        super(SettingsCheckBox, self)._change_emit()


