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
    # (id, value) -- the value itself, NOT wrapped in a list.
    #
    # It was `pyqtSignal(str, list)` until 2026-09-20, and every widget wrapped its scalar in a
    # one-element list that every receiver indexed straight back out: `menupoint_changed` stored
    # the list shape into `self.changed`, and `accept()` did `value[0]` on the way to the database.
    # So the wrapping was not confined to the signal -- it propagated into the dialog's own state,
    # where a reader had to know it was there.
    #
    # RW's ruling, 2026-09-20: *"change the signal"*. No setting carries more than one value, so
    # the list never described anything. `object` rather than a union, because the values really
    # are heterogeneous -- str, int, float and bool all travel this signal.
    changed = pyqtSignal(str, object)

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
        self.changed.emit(self._id, self.value)


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


class _SettingsRangeWidget(SettingsWidget):
    """
    Base for the settings widgets backed by an integer range control

    SettingsSlider and SettingsDial were the same thirty lines twice until 2026-09-20, differing
    only in which .ui file they loaded, the type string they reported and the name of the child
    widget inside that file. Everything else -- the min/max/step/unit state, the setup order, the
    snap-to-step handler and its re-entry guard -- was duplicated character for character, and
    SettingsDial's copy of the guard had already been reduced to a "see SettingsSlider" comment,
    which is the duplication admitting itself.

    A subclass supplies three class attributes and nothing else. Both concrete classes keep their
    original constructor signature, so no call site changes.
    """
    #: The type string reported to the settings dialog
    _TYPE = None
    #: The .ui file to load from gui/settings
    _UI_FILE = None
    #: The name of the range control inside that .ui file
    _CONTROL = None

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1,
                 unit="%", *, callback):
        super(_SettingsRangeWidget, self).__init__(_id, self._TYPE, value, self._UI_FILE,
                                                   title, desc, parent, callback=callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.unit = unit
        self.control = getattr(self.ui, self._CONTROL)
        self.ui.val.setText("{} {}".format(value, unit))
        self.control.setMinimum(self.min_val)
        self.control.setMaximum(self.max_val)
        self.control.setSingleStep(self.step)
        self.control.setValue(self.value)
        self.control.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self):
        self.value = _snap_to_step(self.control.value(), self.min_val, self.max_val, self.step)
        if self.value != self.control.value():
            # setValue re-enters this handler, so the correction would emit `changed` a second time
            # and the dialog would record the unsnapped value first
            self.control.blockSignals(True)
            self.control.setValue(self.value)
            self.control.blockSignals(False)
        self.ui.val.setText("{} {}".format(self.value, self.unit))
        super(_SettingsRangeWidget, self)._change_emit()


class SettingsSlider(_SettingsRangeWidget):
    """
    Class to show an slider in the settings
    """
    _TYPE = "SliderWidget"
    _UI_FILE = "menu_slider.ui"
    _CONTROL = "slider"

    def __init__(self, *args, **kwargs):
        super(SettingsSlider, self).__init__(*args, **kwargs)
        # Kept as an alias of self.control: this attribute was public before the classes were
        # merged, and dropping it would be a silent break for anything reaching in by name
        self.slider = self.control


class SettingsDial(_SettingsRangeWidget):
    """
    Class to show a dial in the settings
    """
    _TYPE = "DialWidget"
    _UI_FILE = "menu_dial.ui"
    _CONTROL = "dial"

    def __init__(self, *args, **kwargs):
        super(SettingsDial, self).__init__(*args, **kwargs)
        # see SettingsSlider -- same reason
        self.dial = self.control


class _SettingsSpinnerWidget(SettingsWidget):
    """
    Base for the settings widgets backed by a spin box

    The same de-duplication as _SettingsRangeWidget above, applied to the other pair on 2026-09-20.
    SettingsSpinner and SettingsDecimalSpinner differed in the .ui file, the type string, and the
    decimal one's extra `decimals` argument; the bounds, prefix, suffix, step and the value handler
    were duplicated.

    A subclass supplies two class attributes, and overrides _configure_control only if it has
    something to set before the bounds -- which the decimal spinner does, and the ordering matters:
    see its override.
    """
    #: The type string reported to the settings dialog
    _TYPE = None
    #: The .ui file to load from gui/settings
    _UI_FILE = None

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1,
                 prefix="", suffix="%", *, callback):
        super(_SettingsSpinnerWidget, self).__init__(_id, self._TYPE, value, self._UI_FILE,
                                                     title, desc, parent, callback=callback)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.prefix = prefix
        self.suffix = suffix
        self.spin = self.ui.spin
        self._configure_control()
        self.spin.setMinimum(self.min_val)
        self.spin.setMaximum(self.max_val)
        self.spin.setPrefix(self.prefix)
        self.spin.setSuffix(self.suffix)
        self.spin.setSingleStep(self.step)
        self.spin.setValue(self.value)
        self.spin.valueChanged.connect(self._on_value_changed)

    def _configure_control(self):
        """
        Hook for whatever a subclass must set BEFORE the bounds and the value

        :return: None
        """

    def _on_value_changed(self):
        self.value = self.spin.value()
        super(_SettingsSpinnerWidget, self)._change_emit()


class SettingsSpinner(_SettingsSpinnerWidget):
    """
    Class to show an integer spinner in the settings
    """
    _TYPE = "IntegerSpinnerWidget"
    _UI_FILE = "menu_spin.ui"


class SettingsDecimalSpinner(_SettingsSpinnerWidget):
    """
    Class to show a decimal spinner in the settings
    """
    _TYPE = "DecimalSpinnerWidget"
    _UI_FILE = "menu_decimal_spin.ui"

    def __init__(self, _id, min_val, max_val, value, parent=None, title="", desc="", step=1.0,
                 decimals=2, prefix="", suffix="%", *, callback):
        # Set before super().__init__, because _configure_control runs inside it and needs this
        self.decimals = decimals
        super(SettingsDecimalSpinner, self).__init__(_id, min_val, max_val, value, parent, title,
                                                     desc, step, prefix, suffix, callback=callback)

    def _configure_control(self):
        # decimals first: QDoubleSpinBox rounds every bound and the value to the decimals in force
        # when they are set, and a later setDecimals does not restore the lost precision. This is
        # the whole reason the base class has this hook rather than one fixed setup order
        self.spin.setDecimals(self.decimals)


class SettingsChannelNames(SettingsWidget):
    """
    Class to show one field per standard channel in the settings

    Replaced a single `SettingsText` on 2026-09-20. The channel names were edited as one
    semicolon-delimited string -- `Red;Green;Blue;Cyan;Magenta` -- with the format carried in the
    description as prose, so the delimiter, the ordering and the count were all the user's
    responsibility and nothing validated any of them. **A missing semicolon silently renamed two
    channels into one.** RW raised it on 2026-08-22 and ruled on the shape on 2026-09-20:

        *"the Settings widget should allow always to set the standard names of all 5 channels"*

    So the field count is FIXED at five and does not follow the image: these are the program's
    standard channel names, which an analysis offers before any image has been looked at.

    **The stored format is unchanged** -- still `a;b;c;d;e` in one settings row -- because the
    database column, `AnalysisSettingsDialog` and every consumer already read it that way. Only the
    editing surface changed.
    """
    #: The standard channels this program names, in the order the analysis dialog lists them
    CHANNELS = ("le_one", "le_two", "le_three", "le_four", "le_five")

    def __init__(self, _id, value, parent=None, title="", desc="", *, callback):
        super(SettingsChannelNames, self).__init__(_id, "ChannelNamesWidget", value,
                                                   "menu_channels.ui", title, desc, parent,
                                                   callback=callback)
        self.fields = [getattr(self.ui, name) for name in self.CHANNELS]
        for field, name in zip(self.fields, self._split(value)):
            field.setText(name)
            # editingFinished, matching SettingsText: it fires on focus loss and on Return, so a
            # half-typed name does not reach the database on every keystroke
            field.editingFinished.connect(self._on_value_changed)
        self.value = self._join()

    @classmethod
    def _split(cls, value):
        """
        Method to read the stored string into exactly one name per standard channel

        Short values are PADDED rather than rejected: the database legitimately holds three names
        where the file holds five, which is the state RW met on 2026-08-22 -- the box went from
        five names to three and there was no way to type the other two back without knowing the
        delimiter. Surplus names are dropped, with a warning, because there is nowhere to show them.

        :param value: The stored `a;b;c` string
        :return: A list of exactly len(CHANNELS) names
        """
        names = str(value).split(";") if value else []
        if len(names) > len(cls.CHANNELS):
            LOGGER.warning(f"{len(names)} channel names stored, but the program has "
                           f"{len(cls.CHANNELS)} standard channels -- the surplus is dropped")
        names = names[:len(cls.CHANNELS)]
        return names + [""] * (len(cls.CHANNELS) - len(names))

    def _join(self):
        """
        Method to render the fields back into the stored format

        Trailing empties are KEPT, not stripped. Dropping them would make "Red;Green;Blue;;" read
        back as three names, and the next time the dialog opened the last two fields would be empty
        for a different reason than the user left them -- the round trip has to be exact.

        :return: The `a;b;c;d;e` string to store
        """
        return ";".join(field.text().strip() for field in self.fields)

    def _on_value_changed(self):
        joined = self._join()
        # editingFinished fires on every focus change, including ones that altered nothing. Emitting
        # regardless would mark the settings dirty for a user who only tabbed through the dialog
        if joined == self.value:
            return
        self.value = joined
        super(SettingsChannelNames, self)._change_emit()


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
        # A BOOL, not the raw Qt.CheckState. The state is an enum whose Checked member is 2, and the
        # settings column this ends up in is typed "bool" -- so emitting the enum stored a 2 that
        # every later read turned back into False. Ticking a box switched its setting OFF, which is
        # how it was reported from real use on 2026-08-22.
        #
        # Two-state is the whole truth here: every check menu point declares "tristate": 0, which
        # Romano ruled on 2026-08-13. Should a tristate box ever be wanted, this is the line that
        # has to grow a third value -- and the settings column would need a type that can hold it
        self.value = self.ui.check.checkState() == Qt.Checked
        super(SettingsCheckBox, self)._change_emit()


