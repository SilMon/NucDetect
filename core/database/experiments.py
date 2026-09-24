"""
Experiments across databases.

RW's rulings of 2026-09-22 and 2026-09-24, which this module is the whole of:

* *"Nucdetect.db should remain the standard database for analysis that does not create an
  experiment. If the user assigns images to experiments, the data should be stored in the
  experiment database."*
* *"Creating an experiment should create the database."* -- ONE EXPERIMENT, ONE DATABASE, named after
  it by ``Paths.database_for``.
* The experiment dialog lists **every** experiment, in whichever database it lives, not only the
  active database's.
* Experiments that already live in the standard database stay there until the user moves them --
  *"Do not migrate the existing databases"* -- and are reported as LEGACY so the dialog can offer
  the move.
* **An analysis is written to the standard database AND to every experiment database the image
  belongs to** -- *"Re-Analysis should always update all experiment databases the image is part
  of. This is the whole point of re-analyzing an image."* ``analysis_targets`` is that rule.

**Every read here is READ-ONLY** (``file:...?mode=ro``): listing experiments or asking where an
image lives must not create, heal or stamp any file it looks at. Only ``create_experiment_database``
writes, and only to a file that does not exist yet.

WHAT COUNTS AS MEMBERSHIP is what the rest of the program already uses: a ``groups`` row for the
image, or -- for rows written before groups existed -- ``images.experiment`` set. See
``Requester.get_associated_images_for_experiment``.
"""
import os
import sqlite3
from contextlib import closing
from typing import Dict, Iterable, List, NamedTuple, Optional, Set

from core.database.connections import Connector, Inserter, Requester
from gui import Paths


class ExperimentLocation(NamedTuple):
    """Where one experiment lives."""
    name: str
    #: The database holding it, absolute
    path: str
    #: True for an experiment stored in the STANDARD database -- the shape every experiment had
    #: before 2026-09-24, offered a move to its own database rather than moved automatically
    legacy: bool


def standard() -> str:
    """The standard database, absolute -- where an analysis always goes."""
    return os.path.abspath(Paths.database)


def _read_only(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.replace(os.sep, '/')}?mode=ro", uri=True)


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return bool(connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def experiments_in(path: str) -> List[str]:
    """
    Name the experiments a database holds

    :param path: The database
    :return: The names, sorted. Empty for a missing, unreadable or experiment-less file
    """
    if not os.path.isfile(path):
        return []
    try:
        with closing(_read_only(path)) as connection:
            if not _has_table(connection, "experiments"):
                return []
            return sorted(row[0] for row in connection.execute("SELECT name FROM experiments"))
    except sqlite3.DatabaseError:
        return []


def experiment_databases() -> List[str]:
    """
    List the databases in the data folder that hold at least one experiment, the standard one excluded

    :return: Absolute paths, sorted by file name
    """
    found = [os.path.abspath(path) for path in Paths.list_databases()
             if os.path.abspath(path) != standard() and experiments_in(path)]
    return sorted(found, key=lambda p: os.path.basename(p).lower())


def all_experiments() -> List[ExperimentLocation]:
    """
    List every experiment the program can reach, with the database it lives in

    Experiment databases first, then the legacy ones still in the standard database. A name that
    exists in both is listed twice, with its two locations -- that is a real state (an experiment
    COPIED out of the standard database, RW's "copy" option) and hiding either would hide data.

    :return: The experiments, sorted by name within each group
    """
    located = [ExperimentLocation(name, path, False)
               for path in experiment_databases() for name in experiments_in(path)]
    located += [ExperimentLocation(name, standard(), True) for name in experiments_in(standard())]
    return located


def create_experiment_database(name: str, details: str = "", notes: str = "") -> str:
    """
    Create the database of a new experiment, with the experiment in it

    The new file gets the full schema -- stamped with the current version, as every database this
    build creates -- and its own standard settings, like any database opened for the first time.

    :param name: The experiment's name. It names the file too, through ``Paths.database_for``
    :param details: The experiment's details
    :param notes: The experiment's notes
    :return: The path of the new database
    :raises FileExistsError: if a database of that name already exists -- two experiments whose
        names sanitise to the same file name would otherwise share one database without either
        being told
    :raises ValueError: if the name yields no usable file name
    """
    path = os.path.abspath(Paths.database_for(name))
    if os.path.exists(path):
        raise FileExistsError(f"a database named {os.path.basename(path)} already exists")
    connector = Connector(path=path)
    try:
        connector.create_tables()
        connector.create_standard_settings()
        Inserter(connector).add_new_experiment(name, details, notes)
        connector.commit_changes()
    finally:
        connector.close_connection()
    return path


def database_of(name: str) -> Optional[str]:
    """
    Find the experiment database holding an experiment of this name

    Only experiment databases are searched: a legacy experiment in the standard database is not
    "the experiment's database", and writing a new image's membership there is what RW ruled out.

    :param name: The experiment's name
    :return: The path, or None if no experiment database holds it
    """
    expected = os.path.abspath(Paths.database_for(name))
    if name in experiments_in(expected):
        return expected
    for path in experiment_databases():
        if name in experiments_in(path):
            return path
    return None


def membership_index(md5s: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    """
    Map images to the experiment databases they belong to, reading each database once

    Built once per analysis run so that the warning shown before the run and the databases written
    during it come from the SAME answer: what the user was told is exactly what is written.

    :param md5s: Restrict the index to these images; None indexes every image found
    :return: {image md5: [experiment database paths]}, only for images with at least one
    """
    wanted: Optional[Set[str]] = set(md5s) if md5s is not None else None
    index: Dict[str, List[str]] = {}
    for path in experiment_databases():
        try:
            with closing(_read_only(path)) as connection:
                members = {row[0] for row in connection.execute("SELECT image FROM groups")}
                members |= {row[0] for row in connection.execute(
                    "SELECT md5 FROM images WHERE experiment IS NOT NULL")}
        except sqlite3.DatabaseError:
            continue
        for md5 in members if wanted is None else members & wanted:
            index.setdefault(md5, []).append(path)
    return index


def analysis_targets(md5: str, index: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Name every database an analysis of this image is written to

    RW, 2026-09-24: the standard database, plus every experiment database the image belongs to.
    The ACTIVE database is not added on its own: when it is neither, it does not receive the
    result, and the result table -- which reads the active database -- will not show it.

    :param md5: The image
    :param index: A ``membership_index`` built for the run; built for this one image when absent
    :return: The standard database first, then the experiment databases, without repetition
    """
    index = membership_index([md5]) if index is None else index
    targets = [standard()]
    for path in index.get(md5, []):
        if os.path.abspath(path) not in targets:
            targets.append(os.path.abspath(path))
    return targets


# ---------------------------------------------------------------------------------------------
# Reading an experiment wherever it lives -- RW, 2026-09-24: the export, the statistics and the
# main window's experiment view "should read accross databases". They name an experiment; these
# find its database, so none of them has to know that experiments moved out of the active one.
# ---------------------------------------------------------------------------------------------

def experiment_names() -> List[str]:
    """
    Name every experiment in every database, once each

    :return: The names, sorted. A name in two databases -- a COPY out of the standard one -- is
        listed once, and `location_of` decides which copy is read
    """
    return sorted({location.name for location in all_experiments()})


def location_of(name: str) -> Optional[str]:
    """
    Find the database to READ an experiment from

    Its own experiment database when it has one -- after a copy out of the standard database that
    is where it is kept current -- else the standard database for a legacy experiment.

    :param name: The experiment's name
    :return: The path, or None if no database holds it
    """
    own = database_of(name)
    if own is not None:
        return own
    return standard() if name in experiments_in(standard()) else None


def requester_for(name: str) -> Requester:
    """
    Open a Requester on the database an experiment lives in

    **The caller closes it** -- ``requester.connector.close_connection()``. Opened unprotected,
    because the main window builds its tables on worker threads.

    :param name: The experiment's name
    :return: The Requester; on the standard database when no database holds the experiment, which
        then answers with nothing, as the active database used to
    """
    return Requester(Connector(path=location_of(name) or standard(), protected=False))


def experiments_of_image(md5: str) -> List[str]:
    """
    Name the experiments an image belongs to, in any database

    :param md5: The image
    :return: The experiment names, sorted
    """
    names: Set[str] = set()
    for path in experiment_databases() + [standard()]:
        if not os.path.isfile(path):
            continue
        try:
            with closing(_read_only(path)) as connection:
                if not _has_table(connection, "groups"):
                    continue
                names |= {row[0] for row in connection.execute(
                    "SELECT experiment FROM groups WHERE image = ?", (md5,))}
                names |= {row[0] for row in connection.execute(
                    "SELECT experiment FROM images WHERE md5 = ? AND experiment IS NOT NULL",
                    (md5,))}
        except sqlite3.DatabaseError:
            continue
    return sorted(str(name) for name in names)
