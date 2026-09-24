"""
Schema versioning and conversion for NucDetect databases.

Created 2026-09-22 on RW's instruction: *"Create one [a schema version] for this version, inspect
the existing databases for any differences and assign the scheme a second version"*, together with
*"Do not migrate the existing databases. Provide an option in the Settings dialog (main tab) to
load an existing database and convert it to the most recent version if possible."*

WHY THIS IS ITS OWN MODULE rather than more of ``connections.py``: that file is already ~1640 lines
carrying three responsibilities, which is a filed structural concern of its own. Versioning is a
fourth, it is self-contained, and nothing here needs the connector's cursor discipline.

TWO RULES THIS MODULE KEEPS, BOTH OF THEM RW'S:

1. **Opening a database never changes it.** ``detect`` is read-only: it reads ``PRAGMA
   user_version`` and, when that is 0, INFERS the version from the schema. It does not stamp what
   it finds, because stamping an existing file is still writing to it.
2. **Conversion is explicit.** ``upgrade`` is only ever called from the user-facing converter, and
   it reports what it would do before it does it.

WHAT ``CREATE TABLE IF NOT EXISTS`` ALREADY HEALS, AND WHAT IT CANNOT -- measured 2026-09-22 by
comparing six real databases against this build's schema:

* missing TABLES and missing INDICES heal themselves, because ``Connector.create_tables`` runs the
  whole script at every startup. Two of those six were missing all eight indices and would have
  regained them on the next open;
* missing COLUMNS and wrong column TYPES never heal. ``CREATE TABLE IF NOT EXISTS`` does nothing at
  all when the table exists.

**So a converter's job is columns and types, and nothing else.** That boundary is why version 1
below is defined by a missing column rather than by anything else the inspection found.

**Version 3 (2026-09-24) is the exception, and says why where it is defined**: it adds two tables,
which would heal on open anyway, but a version number has to describe the whole shape or a stamped
file cannot be trusted to have it.
"""
import sqlite3
from typing import Callable, Dict, List, NamedTuple, Optional

#: The schema this build writes. Bump it when a NEW version is added to HISTORY below, never on its
#: own -- a version number with no entry describing it cannot be converted to or from.
SCHEMA_VERSION = 3

#: What `PRAGMA user_version` reads on every database written before 2026-09-22. It is not a
#: version: it is SQLite's default, and it means "ask the schema instead".
UNVERSIONED = 0


class SchemaVersion(NamedTuple):
    """One numbered schema shape, and how to recognise and reach it."""
    number: int
    #: One line, for the converter dialog and the log
    description: str
    #: True when a connection's schema IS this version. Used only to resolve UNVERSIONED files
    recognise: Callable[[sqlite3.Connection], bool]
    #: Statements that bring the PREVIOUS version up to this one. Empty for the oldest known
    upgrade_from_previous: List[str]


def _roi_columns(connection: sqlite3.Connection) -> List[str]:
    return [row[1] for row in connection.execute('PRAGMA table_info("roi")').fetchall()]


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return bool(connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def _has_roi_table(connection: sqlite3.Connection) -> bool:
    return _has_table(connection, "roi")


def _has_colocalization_tables(connection: sqlite3.Connection) -> bool:
    return _has_table(connection, "colocalization") and _has_table(connection,
                                                                    "colocalization_pairs")


HISTORY: Dict[int, SchemaVersion] = {
    1: SchemaVersion(
        number=1,
        description="before roi.co_localized existed",
        # Anything with a roi table but no co_localized column. Found in the wild on 2026-09-22 in
        # nucdetect_sicherung.db -- one of six databases, and the ONLY genuine schema difference
        # among them. The other five already matched the current shape.
        recognise=lambda con: _has_roi_table(con) and "co_localized" not in _roi_columns(con),
        upgrade_from_previous=[],
    ),
    2: SchemaVersion(
        number=2,
        description="roi.co_localized present, one co-localization pair per image",
        recognise=lambda con: (_has_roi_table(con) and "co_localized" in _roi_columns(con)
                               and not _has_colocalization_tables(con)),
        # ALTER TABLE ADD COLUMN is the one schema change SQLite makes in place and in O(1): it
        # rewrites the header, not the rows, and existing rows read the column as NULL. That is
        # correct here -- a roi analysed before co-localization existed has no partner, and NULL
        # says so where 0 would claim "co-localizes with hash 0"
        upgrade_from_previous=['ALTER TABLE "roi" ADD COLUMN "co_localized" INTEGER'],
    ),
    3: SchemaVersion(
        number=3,
        description="co-localization stored per channel pair, in its own two tables",
        recognise=lambda con: (_has_roi_table(con) and "co_localized" in _roi_columns(con)
                               and _has_colocalization_tables(con)),
        # THE FIRST VERSION DEFINED BY TABLES RATHER THAN A COLUMN, which the module docstring's
        # "columns and types, and nothing else" has to be read against: these two tables DO heal
        # on open, like any other table. The version exists anyway because a version number has to
        # describe the whole shape -- a file stamped 2 that this build has opened holds both tables
        # and still says 2, and a file converted by path without ever being opened would not get
        # them at all. IF NOT EXISTS makes the step a no-op on the first kind and correct on the
        # second.
        #
        # Duplicated from create_tables.sql, which builds NEW databases. The two must produce the
        # same tables, column by column -- change one and the other changes with it
        upgrade_from_previous=[
            'CREATE TABLE IF NOT EXISTS "colocalization_pairs" ("image" TEXT, "channel_a" TEXT, '
            '"channel_b" TEXT, "max_distance" REAL, '
            'PRIMARY KEY ("image", "channel_a", "channel_b")) WITHOUT ROWID',
            'CREATE TABLE IF NOT EXISTS "colocalization" ("image" TEXT, "focus" INTEGER, '
            '"channel_a" TEXT, "channel_b" TEXT, "partner" INTEGER, '
            'PRIMARY KEY ("image", "focus", "channel_a", "channel_b")) WITHOUT ROWID',
        ],
    ),
}


class Unconvertible(Exception):
    """Raised when a database cannot be brought to SCHEMA_VERSION -- RW's *"if possible"*."""


def detect(connection: sqlite3.Connection) -> Optional[int]:
    """
    Determine which schema version a database holds, without writing to it

    ``PRAGMA user_version`` is authoritative when it is set. It is 0 on every database written
    before this module existed -- measured, all six of them -- so a 0 is resolved by asking the
    schema which shape it has.

    :param connection: An open connection, which is only read from
    :return: The version number, or None when the schema matches no known version
    """
    stamped = connection.execute("PRAGMA user_version").fetchone()[0]
    if stamped != UNVERSIONED:
        return stamped
    # Newest first: version 2's test is "has the column" and version 1's is "has the table but
    # not the column", so they are mutually exclusive -- but ordering the scan newest-first keeps
    # that a property of this loop rather than of every future recognise()
    for number in sorted(HISTORY, reverse=True):
        if HISTORY[number].recognise(connection):
            return number
    return None


def plan(connection: sqlite3.Connection) -> List[str]:
    """
    Describe what converting this database to SCHEMA_VERSION would do, without doing any of it

    This exists so the converter can tell the user what is about to happen. RW's instruction is
    that existing databases are not migrated behind the user's back; showing the steps first is
    what makes the conversion a decision rather than a side effect.

    :param connection: An open connection, which is only read from
    :return: The statements that would run, in order. Empty when already current
    :raises Unconvertible: when the schema matches no known version, or is NEWER than this build
    """
    current = detect(connection)
    if current is None:
        raise Unconvertible(
            "the schema matches no version this build knows -- it was not written by NucDetect, "
            "or it is from a build newer than this one")
    if current > SCHEMA_VERSION:
        raise Unconvertible(
            f"the database is version {current} and this build understands {SCHEMA_VERSION} -- "
            f"converting it would mean removing data. Use a newer NucDetect")
    steps: List[str] = []
    for number in range(current + 1, SCHEMA_VERSION + 1):
        steps.extend(HISTORY[number].upgrade_from_previous)
    return steps


def upgrade(connection: sqlite3.Connection) -> List[str]:
    """
    Convert a database to SCHEMA_VERSION and stamp it

    **Only ever call this from the user-facing converter.** Opening a database must not change it.

    The whole conversion runs in one transaction, so a failure part-way leaves the file as it was
    rather than half-converted -- which matters most for the multi-step upgrades this will grow.
    The stamp is written inside the same transaction, so a file can never be stamped as a version
    it did not reach.

    :param connection: An open, writable connection
    :return: The statements that were executed, in order. Empty when it was already current
    :raises Unconvertible: as ``plan``
    """
    steps = plan(connection)
    # PRAGMA user_version cannot be parameterised, hence the f-string. SCHEMA_VERSION is a module
    # constant and never user input, which is the only reason that is acceptable here
    with connection:
        for statement in steps:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
    return steps


def stamp_new(connection: sqlite3.Connection) -> None:
    """
    Record the current version on a database this build just created

    Separate from ``upgrade`` on purpose: a NEW database is at SCHEMA_VERSION by construction --
    ``create_tables.sql`` just built it -- so there is nothing to convert, and calling the
    converter for it would report an empty plan and look like a no-op that failed.

    **Safe to call only when the database was created by this build.** It writes, so it must not
    be called on an existing file; that is what ``detect`` plus the user-invoked converter are for.

    :param connection: An open, writable connection
    :return: None
    """
    with connection:
        connection.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")


def describe(connection: sqlite3.Connection) -> str:
    """
    One human-readable line about a database's version, for the log and the converter dialog

    :param connection: An open connection, which is only read from
    :return: The description
    """
    try:
        current = detect(connection)
    except sqlite3.DatabaseError as exc:
        return f"unreadable: {exc}"
    if current is None:
        return "unrecognised schema -- not written by NucDetect, or newer than this build"
    if current == SCHEMA_VERSION:
        return f"version {current} (current)"
    if current > SCHEMA_VERSION:
        return f"version {current}, NEWER than this build's {SCHEMA_VERSION}"
    missing = SCHEMA_VERSION - current
    return (f"version {current} ({HISTORY[current].description}) -- {missing} version"
            f"{'s' if missing > 1 else ''} behind, conversion available")
