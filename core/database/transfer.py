"""
Moving or copying an image's analysis data between databases.

Created 2026-09-22 on RW's instruction:

    *"Nucdetect.db should remain the standard database for analysis that does not create an
    experiment. If the user assigns images to experiments, the data should be stored in the
    experiment database. If some or all of the required analysis data is already in the standard
    database, the user should have two options: Migrate the data to the new database or copy it.
    The standard database should from now on not contain any experiment data."*

So: the standard database is where un-experimented work lives, an experiment's database is where
that experiment's work lives, and this module is the road between them. **Both directions of RW's
choice are one function with one flag** -- a copy and a migrate differ only in whether the source
rows are deleted afterwards, and writing them as two functions would let the two drift.

HOW AN IMAGE'S DATA IS SPREAD OVER THE SCHEMA, because the transfer is only correct if it is
complete. Everything hangs off the image's md5, except the geometry, which hangs off the roi hash:

    images.md5 ─┬─ channels.md5
                ├─ encountered_names.md5
                ├─ groups.image
                ├─ statistics.image
                └─ roi.image ── roi.hash ── points.hash

**`roi.image`, `statistics.image` and `groups.image` are declared INTEGER and hold the md5 TEXT**
(verified against the live database 2026-09-22). SQLite's type affinity stores them intact, which
is why this has never failed -- but it means a join written expecting integers would silently match
nothing, so every comparison here is on the value as stored.

WHAT IS DELIBERATELY NOT TRANSFERRED: `settings`. It is per-database configuration, not analysis
data, and a new database gets its own standard settings when it is created. Carrying the source's
settings across would make an experiment database silently inherit whatever the standard database
was last configured with.

**THE HEAVY LIFTING IS DONE BY SQLITE, NOT BY PYTHON.** The tables involved reach 23.4 million rows
on RW's largest database; reading them into Python and writing them back would be slow and would
hold them all in memory. ``ATTACH DATABASE`` lets a single ``INSERT INTO target.x SELECT ... FROM
main.x`` do the work inside the engine.
"""
import os
import sqlite3
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

#: The tables carrying one image's analysis data, in FOREIGN-KEY-SAFE INSERT ORDER -- parents
#: before children. The reverse of this order is the safe DELETE order, which is why it is a list
#: and not a set. Each entry is (table, the WHERE clause selecting one image's rows).
#:
#: `points` is selected through `roi` because it has no image column of its own: its only link to
#: an image is the roi hash. That subquery is the reason `points` must be deleted BEFORE `roi` on
#: a move -- once the roi rows are gone there is nothing left to identify the points by.
IMAGE_TABLES: List[tuple] = [
    ("images", "md5 IN ({placeholders})"),
    ("encountered_names", "md5 IN ({placeholders})"),
    ("channels", "md5 IN ({placeholders})"),
    ("roi", "image IN ({placeholders})"),
    ("statistics", "image IN ({placeholders})"),
    ("points", "hash IN (SELECT hash FROM {schema}.roi WHERE image IN ({placeholders}))"),
    ("groups", "image IN ({placeholders})"),
]


class TransferReport(NamedTuple):
    """What a transfer did, per table, so the caller can show it rather than assert it."""
    #: Rows written into the target, keyed by table
    copied: Dict[str, int]
    #: Rows removed from the source, keyed by table. Empty for a copy
    removed: Dict[str, int]
    #: Images that already had rows in the target and were overwritten
    already_present: List[str]
    #: The experiment whose membership was carried across, if any
    experiment: Optional[str]

    def summary(self) -> str:
        """One line for a status bar or a log."""
        written = sum(self.copied.values())
        deleted = sum(self.removed.values())
        verb = "Migrated" if deleted else "Copied"
        extra = f", overwriting {len(self.already_present)}" if self.already_present else ""
        return (f"{verb} {written} rows for {self.copied.get('images', 0)} image(s)"
                f"{extra}" + (f", removing {deleted} from the source" if deleted else ""))


def images_in_target(target: str, md5s: Sequence[str]) -> List[str]:
    """
    Report which of the given images already have data in the target database

    Asked BEFORE a transfer so the caller can warn. RW's two options are migrate and copy; neither
    of them says what to do about an image already present, so the decision is surfaced rather than
    made here.

    :param target: Path of the database to look in
    :param md5s: The images to look for
    :return: Those that are already present
    """
    if not md5s or not os.path.isfile(target):
        return []
    con = sqlite3.connect(f"file:{target.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        holders = ",".join("?" * len(md5s))
        return [row[0] for row in
                con.execute(f"SELECT md5 FROM images WHERE md5 IN ({holders})", tuple(md5s))]
    except sqlite3.DatabaseError:
        # A target that cannot be read has nothing in it as far as this question goes; the
        # transfer itself will fail loudly enough
        return []
    finally:
        con.close()


def transfer_images(source: str, target: str, md5s: Iterable[str],
                    experiment: Optional[str] = None, move: bool = False) -> TransferReport:
    """
    Copy -- or migrate -- the analysis data of the given images from one database to another

    **The whole transfer is one transaction.** A failure part-way leaves the source untouched and
    the target unchanged, which matters most for the move: the alternative is data deleted from one
    database and not present in the other.

    **The deletes happen inside the same transaction as the inserts**, so a migrate cannot lose
    rows even if the process dies between the two.

    :param source: Path of the database to take the data from
    :param target: Path of the database to put it in. It must already have the schema -- create it
        with a ``Connector`` first, which also stamps its version
    :param md5s: The images to transfer
    :param experiment: The experiment these images belong to. Its row and their group memberships
        travel with them; None transfers the image data alone
    :param move: True to remove the rows from the source afterwards -- RW's *migrate*. False
        leaves the source intact -- RW's *copy*
    :return: What was done, per table
    :raises FileNotFoundError: if either database is missing
    :raises sqlite3.DatabaseError: if the transfer fails, having changed nothing
    """
    md5s = list(dict.fromkeys(md5s))          # de-duplicated, order preserved
    if not md5s:
        return TransferReport({}, {}, [], experiment)
    for path, label in ((source, "source"), (target, "target")):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} database does not exist: {path}")

    present = images_in_target(target, md5s)
    holders = ",".join("?" * len(md5s))
    params = tuple(md5s)
    copied: Dict[str, int] = {}
    removed: Dict[str, int] = {}

    # isolation_level=None hands the transaction to us. The default mode issues an implicit COMMIT
    # before anything it considers DDL, which would break the all-or-nothing guarantee above
    con = sqlite3.connect(source, isolation_level=None)
    try:
        con.execute("ATTACH DATABASE ? AS target", (target,))
        con.execute("BEGIN IMMEDIATE")
        try:
            if experiment is not None:
                con.execute("INSERT OR REPLACE INTO target.experiments "
                            "SELECT * FROM main.experiments WHERE name = ?", (experiment,))
            for table, where in IMAGE_TABLES:
                clause = where.format(placeholders=holders, schema="main")
                cur = con.execute(
                    f"INSERT OR REPLACE INTO target.{table} "
                    f"SELECT * FROM main.{table} WHERE {clause}", params)
                copied[table] = cur.rowcount if cur.rowcount > 0 else 0
            if move:
                # Reverse order: points before roi, because points are identified THROUGH roi
                for table, where in reversed(IMAGE_TABLES):
                    clause = where.format(placeholders=holders, schema="main")
                    cur = con.execute(f"DELETE FROM main.{table} WHERE {clause}", params)
                    removed[table] = cur.rowcount if cur.rowcount > 0 else 0
                if experiment is not None:
                    # The standard database is to hold no experiment data at all -- RW. The row
                    # goes only when no image anywhere in this database still claims it
                    still_used = con.execute(
                        "SELECT 1 FROM main.groups WHERE experiment = ? LIMIT 1",
                        (experiment,)).fetchone()
                    if not still_used:
                        cur = con.execute("DELETE FROM main.experiments WHERE name = ?",
                                          (experiment,))
                        removed["experiments"] = cur.rowcount if cur.rowcount > 0 else 0
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        try:
            con.execute("DETACH DATABASE target")
        except sqlite3.DatabaseError:
            pass                              # already detached by a failed ATTACH
        con.close()
    # NOT vacuumed. A migrate leaves free pages in the source, and reclaiming them rewrites the
    # whole file -- seconds and a full-size temporary copy on the 749 MB databases this project
    # has. Whoever wants the space back can ask for it explicitly
    return TransferReport(copied, removed, present, experiment)


def experiment_data_in(path: str) -> Dict[str, int]:
    """
    Report how much experiment data a database holds

    RW: *"The standard database should from now on not contain any experiment data."* This is how
    that is checked -- on the standard database it should report zeros.

    :param path: The database to inspect
    :return: Row counts for the experiment-bearing tables
    """
    if not os.path.isfile(path):
        return {"experiments": 0, "groups": 0}
    con = sqlite3.connect(f"file:{path.replace(os.sep, '/')}?mode=ro", uri=True)
    try:
        return {table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("experiments", "groups")}
    except sqlite3.DatabaseError:
        return {"experiments": 0, "groups": 0}
    finally:
        con.close()
