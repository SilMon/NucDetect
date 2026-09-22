"""
Which database the program is currently working with.

Created 2026-09-22 for RW's per-experiment databases: *"It might be useful to allow per experiment
databases instead of relying on one database for everything."*

WHY A MODULE-LEVEL POINTER RATHER THAN A PARAMETER THREADED THROUGH THE CALL SITES:

``DatabaseInteractor.__init__`` builds its own ``Connector`` when it is handed none, and **nine
places in the GUI take that offer** -- `data.py` five times, `GraphicsItems.py`, `selection.py`,
`NucDetectAppQT.py` twice. Every one of them opened ``Paths.database`` independently. Threading a
path through them would mean touching 79 call sites, every dialog constructor, and every future
one; and the first dialog that forgot would silently read a different database from the rest of the
program, which is the worst failure this change can have.

**So the selection is where the default already was -- one place -- and moving it moves every
consumer at once.** A connector given an explicit path still wins, which is what the converter
needs in order to open a database it is not switching to.

THIS MODULE HOLDS PROCESS STATE, AND THAT IS A REAL COST. It is mutable global state, and the two
rules below exist because of it:

* **Only the main window sets it**, on an explicit user action, and it closes and rebuilds its own
  connectors around the change -- an open connection keeps talking to the file it was opened on,
  whatever this module says afterwards.
* **Worker processes never inherit it.** Batch analysis pickles the detector, not this module, and
  a worker does not touch the database at all: results travel back to the parent, which writes
  them. If that ever changes, the active path has to be passed explicitly, because a fresh process
  starts at the default.
"""
import os
from typing import Optional

from gui import Paths

#: None means "the default", resolved on every read rather than captured at import -- Paths
#: resolves HOME at import time and the tests point HOME at a sandbox
_active: Optional[str] = None


def get_active() -> str:
    """
    Return the database the program is currently working with

    :return: An absolute path. The default database when nothing else has been selected
    """
    return _active or Paths.database


def set_active(path: str) -> str:
    """
    Select the database the program is to work with from now on

    **This does not move any existing connection.** A ``Connector`` opened before this call keeps
    talking to the file it was opened on -- SQLite binds the file at connect time, and nothing here
    can reach back into it. The caller must close and rebuild its connectors, which is what
    ``NucDetect.switch_database`` does.

    :param path: The database to use. It need not exist yet; a new one is created and stamped on
        first connect, which is how a per-experiment database comes into being
    :return: The path that is now active, normalised
    :raises ValueError: if the path is empty, or names a directory
    """
    global _active
    if not path:
        raise ValueError("no database path given -- use reset_active() to return to the default")
    resolved = os.path.abspath(path)
    if os.path.isdir(resolved):
        raise ValueError(f"{resolved} is a directory, not a database")
    _active = resolved
    return _active


def reset_active() -> str:
    """
    Return to the default database

    :return: The path that is now active
    """
    global _active
    _active = None
    return get_active()


def is_default() -> bool:
    """
    Report whether the default database is the active one

    :return: True when nothing has been selected, or the selection IS the default
    """
    return os.path.abspath(get_active()) == os.path.abspath(Paths.database)


def describe_active() -> str:
    """
    One human-readable line naming the active database, for a status bar or a log

    :return: The description
    """
    path = get_active()
    name = os.path.splitext(os.path.basename(path))[0]
    if is_default():
        return f"{name} (default database)"
    return f"{name} ({path})"
