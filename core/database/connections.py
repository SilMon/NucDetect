import os
import sqlite3
import time
from enum import Enum
from typing import Tuple, Dict, List, Optional, Set, Union, Iterable, Any

from core.detector_modules.ImageLoader import ANALYSIS_SCALE_UNIT, ImageLoader
from core.logging_config import get_logger
from core.database import schema_version, selection
from gui import Paths
from core.roi.ROI import ROI

LOGGER = get_logger(__name__)

# Shown in the result table in place of a measurement when a nucleus has no statistics row. Every
# cell of that table is a preformatted string and the sort key falls back to text comparison for
# anything that is not a number, so this sorts as one block rather than breaking the column
NO_STATISTICS = "Not calculated"
# Shown in the Co-Loc. column when co-localization is not a meaningful measurement for the row --
# see the comment in get_table_data_for_image. Deliberately non-numeric, like NO_STATISTICS: a
# placeholder that parses as a number would be indistinguishable from a real result, and the result
# table's sort key already groups non-numeric cells together after the numeric ones
NO_COLOCALIZATION = "n/a"
# Shown in the Edge column of the result table. A nucleus cut off by the image border is measured
# as though it were whole -- half the size of an uncut one, on the reference images -- so RW ruled
# on 2026-09-15 that those nuclei are FLAGGED rather than dropped: they stay in the table, in the
# exports and in the statistics, and the cell says which they are. Words rather than yes/no so the
# column reads without its header, and so it sorts into two blocks
CLIPPED_BY_BORDER = "clipped"
NOT_CLIPPED_BY_BORDER = "whole"


class Specifiers(Enum):
    ALL = "*"
    IS = "IS"
    ISNOT = "IS NOT"
    NULL = "NULL"
    EQUALS = "="
    NOTEQUALS = "!="
    GREATER = ">"
    GREATEREQUALS = ">="
    LESSER = "<"
    LESSEREQUALS = "<="


class Connector:

    def __init__(self, protected: bool = True, path: str = None):
        """
        :param protected: If true, no concurrent access to the database is allowed
        :param path: The database to open. Defaults to ``Paths.database``, the one used when no
            experiment-specific database is chosen. It is a parameter so that per-experiment
            databases -- RW, 2026-09-22 -- need no second connector class
        """
        # selection.get_active(), not Paths.database: the default follows the program's current
        # choice of database, so every Requester()/Inserter() built without a connector -- nine of
        # them across the dialogs -- moves with it. An explicit path still wins, which is what the
        # converter needs to open a database it is not switching to
        self.path = path or selection.get_active()
        # Whether THIS call created the file. sqlite3.connect creates it silently, so the question
        # can only be asked before connecting, and the answer is what decides whether the schema
        # version is stamped: a new database is at the current version by construction, an
        # existing one must not be stamped without the user asking. See schema_version
        self.created = not os.path.isfile(self.path)
        self.connection, self.cursor = self.connect_to_database(protected, self.path)
        # Set the cache size to 50000 pages (2 GB)
        self.cursor.execute("PRAGMA cache_size=50000;")
        # Load needed scripts
        self.commands = self.load_sql_commands()
        # Get information about the tables of the database
        self.table_info = self.get_table_info_from_database()

    @staticmethod
    def connect_to_database(protected: bool = True,
                            path: str = None) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """
        Method to connect to a database

        :param protected: If true, no concurrent access to the database is allowed
        :param path: The database to open, defaulting to ``Paths.database``
        :return: The connection and cursor to the database
        """
        # sqlite3.connect creates the database file but NOT the directory holding it, so a fresh
        # HOME raised "unable to open database file" here for anything using the core without the
        # GUI. Asking Paths for its directories is what makes a headless Connector work
        Paths.ensure_directories()
        connection = sqlite3.connect(path or selection.get_active(), check_same_thread=protected)
        return connection, connection.cursor()

    @staticmethod
    def load_sql_commands() -> Dict[str, str]:
        """
        Method to load the pre-defined sql commands

        :return: Dictionary containing the file names and texts
        """
        commands = {}
        for root, dirs, files in os.walk(Paths.sql_dir):
            for file in files:
                # splitext, not file[:-4] -- that assumed a 3-character extension and silently
                # mis-keyed anything else, so a stray .bak or an editor swap file next to the
                # scripts was loaded as a command under a truncated name. Only .sql is a command.
                name, extension = os.path.splitext(file)
                if extension.lower() != ".sql":
                    continue
                # Explicit encoding: without it these fall back to the locale codepage, so the
                # same script parses differently on a machine that is not cp1252
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    commands[name] = f.read()
        return commands

    def get_table_info_from_database(self) -> Dict[str, Dict]:
        """
        Method to get the table information from the database

        :return: The information as dictionary
        """
        table_info = {x[0]: {} for x in self.cursor.execute(self.commands["get_tables_from_database"]).fetchall()}
        for table in table_info.keys():
            query = self.commands["get_columns_from_table"].replace("<table_name>", table)
            info = [x[1] for x in self.cursor.execute(query).fetchall()]
            table_info[table]["columns"] = info
            # The same names as a set, built ONCE per schema read rather than once per query.
            # check_identifiers runs on every statement this class builds, and the result table
            # issues thousands in one call: building the set per call cost 14 % of that call,
            # measured 2026-09-16 against the previous commit on the same database
            table_info[table]["column_set"] = set(info)
            table_info[table]["column_number"] = len(info)
        return table_info

    def check_for_table(self, table: str) -> None:
        """
        Method that raises an ValueError if the table is not in the database

        :param table: The table to check for
        :return: None
        """
        if table not in self.table_info:
            raise ValueError(f"Table \"{table}\" not in database!")

    def commit_changes(self) -> None:
        """
        Methods to commit made changes to the database

        :return: None
        """
        self.connection.commit()

    def rollback_changes(self) -> None:
        """
        Method to discard all changes made since the last commit

        :return: None
        """
        self.connection.rollback()

    def close_connection(self) -> None:
        """
        Method to close the established connection

        :return: None
        """
        self.connection.close()

    def create_tables(self) -> None:
        """
        Method to create the tables if necessary

        :return: None
        """
        # Asked BEFORE the script runs, and asked of the SCHEMA rather than of the file.
        #
        # `self.created` -- whether this connector created the file -- is the obvious test and it
        # is wrong: sqlite3.connect creates the file, while the schema is built here, so a
        # Connector constructed and dropped without calling this leaves an empty unstamped file
        # that no later open will ever stamp (it exists by then, so `created` is False). Caught by
        # a test that did exactly that by accident.
        #
        # No user tables before the script + tables after = this call built the schema, so the
        # database is at the current version by construction and may be stamped. An existing
        # database is left alone, which is RW's "do not migrate the existing databases".
        fresh = not self.cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        self.cursor.executescript(self.commands["create_tables"])
        if fresh:
            schema_version.stamp_new(self.connection)
        self.table_info = self.get_table_info_from_database()

    def check_identifiers(self, table: str, columns) -> None:
        """
        Method to reject any table or column name that is not in the database's own schema

        **This replaced a blacklist of illegal characters on 2026-09-16, and the two are not the
        same kind of check.** The blacklist existed because every value was interpolated into the
        SQL text, so a value carrying a quote or a semicolon could end one statement and start
        another. Values are BOUND now -- see build_where and build_set -- so no value reaches the
        SQL text at all and the blacklist had nothing left to protect.

        What still has to be interpolated is identifiers: SQLite cannot bind a table or a column
        name. Those are checked against the real schema instead -- an allow-list of what exists
        beats a deny-list of what looked dangerous, and it cannot be defeated by a character
        nobody thought of.

        **The blacklist was also wrong in the other direction**, which is why it is gone rather
        than kept alongside: `.` was illegal, so the ordinary value "1.5" was rejected and no
        decimal could be written through `update`.

        :param table: The table the columns belong to
        :param columns: A column name, an iterable of them, or Specifiers.ALL
        :raises ValueError: If the table is unknown, or a column is not one of its columns
        :return: None
        """
        self.check_for_table(table)
        known = self.table_info[table]["column_set"]
        for column in Connector.iterate_column_names(columns):
            if column not in known:
                raise ValueError(f"Query rejected: {column!r} is not a column of {table!r} "
                                 f"(it has {sorted(known)})")

    @staticmethod
    def iterate_column_names(columns):
        """
        Method to yield the bare column names out of whatever a caller passed as `column`

        Callers pass a name, a tuple of names, `Specifiers.ALL`, or a name behind a keyword --
        "DISTINCT name" is the only such form in the tree. `*` names no column and yields
        nothing; the keyword is stripped so the name behind it is checked.

        :param columns: The column argument as handed to a query builder
        :return: The column names to validate, one at a time
        """
        if isinstance(columns, Specifiers):
            return
        if isinstance(columns, str):
            columns = (columns,)
        for column in columns:
            if isinstance(column, Specifiers) or column == "*":
                continue
            name = column.strip()
            if name.lower().startswith("distinct "):
                name = name[len("distinct "):].strip()
            yield name


    def create_standard_settings(self) -> None:
        """
        Method to create the standard settings if necessary

        :return: None
        """
        self.cursor.executescript(self.commands["create_settings"])
        self.commit_changes()

    def delete_existing_image_data(self, image: str) -> None:
        """
        Method to delete all saved data for the given image

        **NO executescript, since 2026-09-16.** `executescript` issues a COMMIT before it runs
        anything, so clearing an image committed whatever transaction was already open -- and
        `save_rois_to_database` calls this and THEN writes the new results, so the old data was
        committed away before the new data existed. A write that failed in between left the image
        with neither.

        The script it used to run needed several statements only because it built a temporary
        VIEW to find the hashes that no other image shares. A subquery says the same thing, so
        this is three ordinary statements that run inside the caller's transaction and commit
        when the caller does.

        **Why points are deleted by a subquery and not by hash**: hash(roi) is md5(channel + area)
        and carries no image, so two images holding an identical focus in the same channel share
        one hash -- 4388 such hashes in the live database -- and the points table has no image
        column, so they share ONE set of points. Deleting by hash alone removed the other image's
        geometry and left its roi row standing with nothing under it: 57 such rows existed across
        33 images, and the manual editor raised "ROI ... does not contain any points!" on them.

        :param image: md5 hash of the image
        :return: None
        """
        self.cursor.execute(
            "DELETE FROM points WHERE hash IN ("
            "    SELECT hash FROM roi WHERE image = ?"
            "    AND hash NOT IN (SELECT hash FROM roi WHERE image <> ?))",
            (image, image))
        # BY IMAGE, not by hash: statistics has an image column, so the row belonging to another
        # image with the same hash must survive
        self.cursor.execute("DELETE FROM statistics WHERE image = ?", (image,))
        self.cursor.execute("DELETE FROM roi WHERE image = ?", (image,))
        # A re-analysis may compare different pairs than the last one did, so the old pairs go too
        # -- a pair left behind would be offered in the table's pair selector with no rows under it
        self.cursor.execute("DELETE FROM colocalization WHERE image = ?", (image,))
        self.cursor.execute("DELETE FROM colocalization_pairs WHERE image = ?", (image,))

    def count_colocalized_foci(self, image: str, channel_a: str,
                               channel_b: str) -> Dict[int, Tuple[int, int]]:
        """
        Method to count, per nucleus, the foci of one channel pair and how many of them have a
        partner

        A dedicated query rather than get_view_from_table, because it needs a join and an
        aggregate, and that method's parameter check rejects both. The per-nucleus percentage is
        derived here from the per-focus rows instead of being stored, so the two cannot disagree.

        The image is compared against the bound value on BOTH sides rather than joined column to
        column: ``roi.image`` is declared INTEGER and ``colocalization.image`` TEXT, and a
        column-to-column comparison would apply numeric affinity to one of them. Binding the md5
        is what every other query against ``roi.image`` does, so it matches the same rows.

        Foci marked "Removed" are left out, exactly as count_foci_for_nucleus_and_channel leaves
        them out -- the percentage must describe the foci the Foci column counts.

        :param image: The md5 hash of the image
        :param channel_a: The first channel of the pair
        :param channel_b: The second channel of the pair
        :return: {nucleus hash: (foci of the pair in it, of those with a partner)}. A nucleus with
            no focus in either channel is absent: it has nothing to co-localize
        """
        rows = self.cursor.execute(
            "SELECT r.associated, COUNT(*), COUNT(c.partner) "
            "FROM colocalization AS c JOIN roi AS r ON r.hash = c.focus AND r.image = ? "
            "WHERE c.image = ? AND c.channel_a = ? AND c.channel_b = ? "
            "AND r.associated IS NOT NULL AND r.detection_method IS NOT 'Removed' "
            "GROUP BY r.associated",
            (image, image, channel_a, channel_b)).fetchall()
        return {int(nucleus): (total, partnered) for nucleus, total, partnered in rows}


    def count_instances(self, column: str, table: str, where: Tuple = ()) -> int:
        """
        Method to count the instances of column in the given table

        :param column: The column to count
        :param table: The table where the column can be found
        :param where: The condition to count
        :return: The number of found instances
        """
        self.check_identifiers(table, column)
        if not where:
            query = self.commands["count"].replace("<column>", column).replace("<table_name>", table)
            return self.cursor.execute(query).fetchall()[0][0]
        condition, params = self.build_where(table, where)
        query = self.commands["count_where"].replace("<column>", column) \
            .replace("<table_name>", table).replace("<condition>", condition)
        return self.cursor.execute(query, params).fetchall()[0][0]

    def insert_or_replace_into(self, table: str, columns: Union[List, Tuple],
                               values: Union[List, Tuple], many: bool = False) -> None:
        """
        Method to insert a new row into the given table

        :param table: The table to insert to
        :param columns: The columns to add
        :param values: The values to insert. Length has to match the number of columns
        :param many: If true, values will be seen as list of entries to insert
        :return: None
        """
        # An empty batch is not an error -- an image in which nothing was detected produces one --
        # and there is nothing to insert, so return before len(values[0]) below indexes into it
        if not values:
            return
        # Identifiers against the schema. The values are bound by execute/executemany below
        # and never needed the character blacklist that used to be applied to them here
        self.check_identifiers(table, columns)
        # Convert the column list
        columns = self.convert_column_list(columns)
        # Get value string
        vals = self.get_value_string(len(values) if not many else len(values[0]))
        query = self.commands["insert_or_replace_into"].replace("<table_name>", table) \
            .replace("<columns>", columns).replace("<data>", vals)
        if not many:
            self.cursor.execute(query, values)
        else:
            self.cursor.executemany(query, values)

    def update(self, table: str, values: Iterable, where: Tuple = ()) -> None:
        """
        Method to update a row in the database

        :param table: The table where the row can be found
        :param values: The values to insert
        :param where: The condition for the update
        :return: None
        """
        if not where:
            raise ValueError("No condition for update given!")
        self.check_for_table(table)
        set_stm, set_params = self.build_set(table, values)
        condition, where_params = self.build_where(table, where)
        query = self.commands["update"].replace("<table_name>", table) \
            .replace("<set_values>", set_stm).replace("<condition>", condition)
        # SET parameters before WHERE: placeholders bind in the order they appear in the text
        self.cursor.execute(query, set_params + where_params)

    def delete(self, table: str, where: Tuple = ()) -> None:
        """
        Method to delete an entries from the given table

        :param table: The table which contains the entries
        :param where: The condition for deletion
        :return: None
        """
        if not where:
            raise ValueError("An empty condition would delete the whole table!")
        self.check_for_table(table)
        condition, params = self.build_where(table, where)
        query = self.commands["delete"].replace("<table_name>", table).replace("<condition>", condition)
        self.cursor.execute(query, params)

    def reset_database(self) -> None:
        """
        Method to reset the database

        :return: None
        """
        self.cursor.executescript(self.commands["reset_database"])
        self.connection.commit()

    def reset_analysis_data(self) -> None:
        """
        Method to reset the analysis data

        :return: None
        """
        self.cursor.executescript(self.commands["reset_analysis_data"])
        self.connection.commit()

    def get_view_from_table(self, column: Union[str, List, Tuple, Specifiers],
                            table: str, where: Tuple = ()) -> List[Tuple[Union[str, int, float]]]:
        """
        Method to get information from the given table

        :param column: The column(s) to select. * for all columns. If a list, the given columns will be selected
        :param table: The table to select information from
        :param where: The condition that need to be passed. If empty, it will be ignored
        :return: The requested information
        """
        # Convert list of columns
        columns = self.convert_column_list(column)
        # Identifiers against the schema; every value below is bound
        self.check_identifiers(table, column)
        if not where:
            query = self.commands["select_from"].replace("<columns>", columns).replace("<table_name>", table)
            return self.cursor.execute(query).fetchall()
        condition, params = self.build_where(table, where)
        query = self.commands["select_from_where"].replace("<columns>", columns) \
            .replace("<table_name>", table).replace("<condition>", condition)
        return self.cursor.execute(query, params).fetchall()

    @staticmethod
    def get_value_string(number: int) -> str:
        """
        Method to get the question mark string for insert queries

        :param number: Number of parameters
        :return: The created string
        """
        return f"{','.join('?' for _ in range(number))}"

    @staticmethod
    def convert_column_list(columns: Union[str, List]) -> str:
        """
        Method to convert the given lists of columns to a usable string

        :param columns: The columns to convert
        :return: The string to insert into an SQL statement
        """
        if isinstance(columns, Specifiers):
            return columns.value
        return ",".join(columns) if isinstance(columns, tuple) or isinstance(columns, list) else columns

    # convert_set_statement was removed here on 2026-09-16. It rendered an update's values
    # into the SQL text; build_set binds them instead, and nothing else called it.

    def build_where(self, table: str, where) -> Tuple[str, List]:
        """
        Method to turn a condition into WHERE text with placeholders, plus the values to bind

        **Replaced convert_where_statement on 2026-09-16.** That method rendered the whole
        condition -- column, operator AND value -- into the SQL text, which is what made a
        character blacklist necessary and what made nested conditions dangerous: `check_parameters`
        recursed exactly one level, so the values inside a nested tuple were never inspected at
        all. Binding removes both problems rather than deepening the inspection.

        The column is interpolated, because SQLite cannot bind an identifier -- it is checked
        against the schema by check_identifiers instead. The value is always bound. The one
        exception is `Specifiers.NULL`, which is a SQL keyword rather than a value: `IS ?` with
        None bound is never true, so `IS NULL` has to reach the text.

        :param table: The table the condition applies to, for validating its columns
        :param where: A (column, operator, value) triple, or a tuple of such triples
        :return: The WHERE text and the parameters to bind, in order
        """
        if isinstance(where[0], tuple):
            parts, params = [], []
            for condition in where:
                text, values = self.build_where(table, condition)
                parts.append(text)
                params.extend(values)
            return " AND ".join(parts), params
        column, operator, value = where
        self.check_identifiers(table, column)
        operator = operator.value if isinstance(operator, Specifiers) else str(operator)
        if value is Specifiers.NULL:
            return f"{column} {operator} NULL", []
        if isinstance(value, Specifiers):
            return f"{column} {operator} {value.value}", []
        return f"{column} {operator} ?", [value]

    def build_set(self, table: str, values) -> Tuple[str, List]:
        """
        Method to turn an update's values into SET text with placeholders, plus what to bind

        Same change as build_where, and the same reason: the value used to be rendered into the
        text by convert_value, so `update` could not write a string containing a quote and --
        because of the blacklist that protected it -- could not write "1.5" either.

        :param table: The table being updated, for validating its columns
        :param values: A (column, value) pair, or an iterable of them
        :return: The SET text and the parameters to bind, in order
        """
        # WHAT BINDING DOES NATIVELY, moved here when convert_value was deleted on 2026-09-20.
        # Both behaviours cost real debugging to establish and neither needs a branch in this
        # file any more; they are written down so nobody reintroduces the rendering they belong
        # to, having rediscovered the same two traps:
        #
        #   * None binds as SQL NULL. Until 2026-09-14 the renderer had no branch for None at
        #     all: it fell off the end, returned the Python None, and the caller's f-string wrote
        #     the bare word `None`, which SQLite then read as a COLUMN NAME.
        #     `Inserter.set_image_scale(md5, None, None)` raised `no such column: None`, and
        #     every nullable column had the same hole -- x_res, y_res, unit, associated, match
        #     and co_localized are all nullable by design, and writing NULL to any of them had no
        #     working route.
        #   * True binds as 1. The renderer had to test bool BEFORE int, because bool is a
        #     subclass of int and the other order rendered True as the string "True". SQLite
        #     accepts the bare TRUE/FALSE literals only from 3.23 onwards, and only while the
        #     value is interpolated unquoted, so the old order worked by two coincidences at once.
        #
        # WHAT BINDING DOES NOT DO is the WHERE-clause caveat: `<col> = NULL` is never true in
        # SQL, so a None reaching a condition matches nothing rather than raising. Use
        # Specifiers.IS with Specifiers.NULL there -- build_where renders that pair as `IS NULL`.
        pairs = values if isinstance(values[0], tuple) else (tuple(values),)
        self.check_identifiers(table, [pair[0] for pair in pairs])
        parts, params = [], []
        for column, value in pairs:
            # A Specifier is a SQL KEYWORD, not a value -- `SET associated = NULL` is written by
            # reset_nucleus_focus_association exactly that way, and binding it raises
            # "type 'Specifiers' is not supported". Python's None, by contrast, IS bound: sqlite
            # binds it as SQL NULL natively, which is the correct route and the one the
            # 2026-09-14 convert_value fix was reaching for by hand
            if isinstance(value, Specifiers):
                parts.append(f"{column}={value.value}")
            else:
                parts.append(f"{column}=?")
                params.append(value)
        return ",".join(parts), params

    # convert_value was removed here on 2026-09-20, the last of the three renderers.
    # It turned a Python value into SQL text; build_where and build_set bind their values
    # instead, and nothing had called it since. The two behaviours it documented are
    # recorded on build_set, because they are the ones sqlite now provides natively.


class DatabaseInteractor:
    """
    Base class for database interactions
    """

    def __init__(self, connector: Connector = None, protected: bool = True):
        self.connector = connector if connector else Connector(protected)

    def commit(self) -> None:
        """
        Method to commit the made changes

        :return: None
        """
        self.connector.commit_changes()

    def rollback_and_close(self) -> None:
        """
        Method to discard all changes made since the last commit and close the connection

        The counterpart of commit_and_close, for a save the user cancels partway through. Every
        write between them is in one open transaction, so discarding it is a true cancel rather
        than a half-applied save.

        :return: None
        """
        self.connector.rollback_changes()
        self.connector.close_connection()

    def commit_and_close(self) -> None:
        """
        Method to commit all changes and close the connection
        :return: None
        """
        self.connector.commit_changes()
        self.connector.close_connection()


class Requester(DatabaseInteractor):
    """
    Class to request data from the database
    """

    def get_all_settings(self) -> List[Tuple[Union[str, int, float], ...]]:
        """
        Method to load the settings from the database

        The trailing ellipsis in the return type is load-bearing: Tuple[X] means a tuple of exactly
        one element, and each row here carries the three settings columns.

        :return: One row per setting, as (key, value, type) -- the last two are what
                 Connector.convert_to_type needs to rebuild the stored value
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "settings")

    def get_all_experiments(self) -> List[str]:
        """
        Method to get all existing experiments

        :return: The experiment names as lists
        """
        return [x[0] for x in self.connector.get_view_from_table("name", "experiments")]

    def get_info_for_experiment(self, experiment: str) -> List[str]:
        """
        Method to get the details for an experiment

        :param experiment: Name of the experiment
        :return: The details and notes for the given experiment
        """
        rows = self.connector.get_view_from_table(("details", "notes"),
                                                  "experiments",
                                                  ("name", Specifiers.EQUALS, experiment))
        # None for "no such experiment", the contract get_info_for_image adopted on 2026-08-15 and
        # the rest of these accessors took on 2026-08-17. Indexing [0] raised IndexError from three
        # frames down, naming neither the experiment nor the query
        return rows[0] if rows else None

    def get_channels_for_experiment(self, experiment: str, include_main: bool = False) -> List[str]:
        """
        Method to get the channel names associated with the given experiment

        :param experiment: The name of the experiment
        :param include_main: If true, the name of the main channel will be included
        :return: The name of the channels
        """
        # Select the images corresponding to the experiment
        imgs = self.get_associated_images_for_experiment(experiment)
        # An experiment with no images has no channels. imgs[0] raised IndexError instead of saying
        # so, and an experiment can legitimately be empty -- the dialog creates it before any image
        # is assigned
        if not imgs:
            return []
        channels = [x[0] for x in self.connector.get_view_from_table("DISTINCT name", "channels",
                                                                     ("md5", Specifiers.EQUALS, imgs[0]))]
        # Get the main channel
        main = self.get_main_channel(imgs[0])
        # `main in channels`, not a bare remove. get_main_channel answers None rather than raising
        # for an image with no nominated channel row, and CHANNEL ROWS ARE WRITTEN BY THE ANALYSIS
        # -- so an experiment holding an image nobody has analysed yet reached `[].remove(None)` and
        # took the statistics dialog down before it opened. An experiment being set up is exactly
        # where that state lives, and the empty-experiment guard above exists for the same reason
        if not include_main and main in channels:
            channels.remove(main)
        return channels

    def get_main_channel_for_experiment(self, experiment: str) -> str:
        """
        Method to get the main channel of the given experiment

        :param experiment: The experiment
        :return: The name of the main channel
        """
        # Get first associated image
        imgs = self.get_associated_images_for_experiment(experiment)
        # Same empty-experiment case as get_channels_for_experiment above
        if not imgs:
            return None
        return self.get_main_channel(imgs[0])

    def get_associated_images_for_experiment(self, experiment: str) -> List[str]:
        """
        Method to get the associated images for a given experiment

        :param experiment: The name of the experiment
        :return: List of the associated image hashes
        """
        imgs = self.connector.get_view_from_table("image", "groups",
                                                  ("experiment", Specifiers.EQUALS,
                                                   experiment))
        if not imgs:
            imgs = self.connector.get_view_from_table("md5", "images",
                                                      ("experiment", Specifiers.EQUALS, experiment))
        return [x[0] for x in imgs]

    def get_number_of_associated_images_for_experiment(self, experiment: str) -> int:
        """
        Method to get the number of associated images for the given experiment
        :param experiment: The name of the experiment
        :return: The number of associated images
        """
        return self.connector.count_instances("md5", "images", ("experiment", Specifiers.EQUALS, experiment))

    def get_all_images(self) -> List[str]:
        """
        Method to get all saved images

        :return: The images as list of md5 hashes
        """
        return [x[0] for x in self.connector.get_view_from_table("md5", "images")]

    def get_experiment_for_image(self, image: str) -> Union[str, None]:
        """
        Method to get the associated experiment for the given image

        :param image: md5 hash of the image
        :return: The name of the experiment, or None if the image is not in the images table
        """
        info = self.get_info_for_image(image)
        return info[14] if info is not None else None

    def get_info_for_image(self, image: str) -> Union[Tuple[Union[str, int, float, None], ...], None]:
        """
        Method to get all saved information for the given image

        Returns None -- not an empty tuple -- when the image is not in the table. The empty tuple
        that stood here until 2026-08-15 was meant as a "no such image" signal but no caller could
        act on it: every one of them indexes the result, so a miss raised IndexError rather than
        evaluating falsy, and a miss was distinguishable from a row only by the exception. That is
        what the "throws error after analyse all" TODO removed from above this query described --
        reproduced against a sandbox database, `get_info_for_image("unregistered")[8]` raises
        `IndexError: tuple index out of range`.

        The row is the images table in schema order: md5, year, month, day, hour, minute, channels,
        width, height, x_res, y_res, unit, analysed, settings, experiment, modified.

        :param image: The md5 hash of the image
        :return: The image's row, or None if there is no such image
        """
        info = self.connector.get_view_from_table(Specifiers.ALL, "images",
                                                  ("md5", Specifiers.EQUALS, image))
        return info[0] if info else None

    def check_if_image_was_analysed(self, image: str) -> bool:
        """
        Method to check if the given image was analysed

        :param image: The md5 hash of the image
        :return: True if the image was analysed
        """
        rows = self.connector.get_view_from_table("analysed", "images",
                                                  ("md5", Specifiers.EQUALS, image))
        # An unknown hash returns an empty list, and [0][0] raised IndexError instead of answering
        # "no". Found by a test driving _analyze_all over paths that were
        # never registered -- one such path used to take the whole batch run down before the loop
        if not rows:
            return False
        return bool(rows[0][0])


    def check_if_image_is_registered(self, image: str) -> bool:
        """
        Method to check if the given image is already registred in the database

        :param image: The md5 hash of the image
        :return: True if the image was found in the database
        """
        return bool(self.connector.get_view_from_table("file_name",
                                                       "encountered_names",
                                                       ("md5", Specifiers.EQUALS, image)))

    # get_image_x_scale, get_image_y_scale and get_image_scale were removed here. All three were
    # unreachable and none of them could have worked: the y variant queried the x_res column, and
    # both getters indexed an already-unpacked row tuple a second time, raising TypeError on the
    # first call. Image scales are read through gui.Util.get_image_scale, which is a separate
    # module-level function and the only live path.

    def get_groups_for_experiment(self, experiment: str) -> List[str]:
        """
        Method to get all associated groups for the given experiment

        :param experiment: The experiment
        :return: List of all associated groups
        """
        return [x[0] for x in self.connector.get_view_from_table("DISTINCT name", "groups",
                                                                 ("experiment", Specifiers.EQUALS, experiment))]

    def get_associated_group_for_image(self, image: str, experiment: str):
        """
        Method to get the groups this image was associated with

        :param image: The md5 hash of the image
        :param experiment: The experiment this image was associated with
        :return: The group(s)
        """
        group = self.connector.get_view_from_table("name", "groups",
                                                   (("experiment", Specifiers.EQUALS, experiment),
                                                    ("image", Specifiers.EQUALS, image)))
        return group[0][0] if group else "No Group"

    def get_nuclei_hashes_for_image(self, md5: str) -> List[int]:
        """
        Method to get the detected nuclei for each image

        :param md5: The md5 hash of the image
        :return: List of database entries for each nucleus
        """
        return [int(x[0]) for x in self.connector.get_view_from_table("hash", "roi",
                                                                      (("associated", Specifiers.IS,
                                                                        Specifiers.NULL),
                                                                       ("image", Specifiers.EQUALS, md5)))]

    def get_hashes_of_associated_foci(self, nucleus: str, image: str) -> List[str]:
        """
        Method to get the hashes of associated foci for the given nucleus

        The image is REQUIRED and is half of the roi table's primary key. A roi hash is derived
        from the channel name and the AREA, so two images holding a roi with an identical run list
        hash to the same value -- by design, which is what PRIMARY KEY ("hash", "image") is for.
        Without the image this returns the foci of every image whose nucleus hashes the same.

        :param nucleus: md5 hash of the nucleus
        :param image: The md5 hash of the image the nucleus belongs to
        :return: List of all focus hashes
        """
        return [x[0] for x in self.connector.get_view_from_table("hash", "roi",
                                                                 (("associated", Specifiers.EQUALS, nucleus),
                                                                  ("image", Specifiers.EQUALS, image)))]

    def count_foci_for_nucleus_and_channel(self, nucleus: int, channel: str, image: str) -> int:
        """
        Method to count the associated foci for the given nucleus and channel

        The image is REQUIRED -- see get_hashes_of_associated_foci. Measured on the testing
        database before this filter existed: a nucleus of demo.tif reported 558 Green foci where
        the image holds 146, because the same nucleus area exists in four images.

        :param nucleus: The md5 hash of the nucleus
        :param channel:The name of the channel
        :param image: The md5 hash of the image the nucleus belongs to
        :return: The number of associated foci
        """
        return self.connector.count_instances("hash", "roi", (
            ("associated", Specifiers.EQUALS, nucleus), ("channel", Specifiers.EQUALS, channel),
            ("image", Specifiers.EQUALS, image),
            ("detection_method", Specifiers.NOTEQUALS, "Removed")))

    def get_modified_images(self) -> List[str]:
        """
        Method to get all images that were manually modified

        :return: The hashes of all modified images
        """
        return [x[0] for x in self.connector.get_view_from_table("md5", "images",
                                                                 ("modified", Specifiers.EQUALS, "1"))]

    def get_associated_roi(self, image: str) -> List[Tuple]:
        """
        Method to get information about the ROI associated with this image

        :param image: The image to get the ROI for
        :return: The retrieved information
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "roi",
                                                  ("image", Specifiers.EQUALS, image))

    def get_channels(self, image: str) -> List[Tuple]:
        """
        Method to get information about the channels of this image

        :param image: The image to get the information for
        :return: None
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "channels",
                                                  ("md5", Specifiers.EQUALS, image))

    def get_channel_names(self, img: str, include_main: bool = True) -> List[str]:
        """
        Method to get the names of all active channels for the given image

        :param img: The md5 hash of the image
        :param include_main: If true, the name of the main channel will be included
        :return: The channel names
        """
        where = ("md5", Specifiers.EQUALS, img)
        if not include_main:
            where = (where, ("active", Specifiers.EQUALS, 1), ("main", Specifiers.EQUALS, 0))
        else:
            where = (where, ("active", Specifiers.EQUALS, 1))
        return [x[0] for x in self.connector.get_view_from_table("name", "channels", where)]

    def get_colocalization_pairs(self, image: str) -> List[Tuple[str, str]]:
        """
        Method to get the channel pairs the analysis of an image compared

        :param image: The md5 hash of the image
        :return: The pairs, by channel name. In primary-key order -- alphabetical by the first
            channel, then the second -- not in the order they were configured, which is not stored.
            Empty for an image analysed before pairs existed, and for one analysed without any
        """
        return [(a, b) for a, b in self.connector.get_view_from_table(
            ("channel_a", "channel_b"), "colocalization_pairs", ("image", Specifiers.EQUALS, image))]

    def get_colocalization_distance(self, image: str) -> Optional[float]:
        """
        Method to get the distance, in pixels, at which an image's channel pairs were compared

        :param image: The md5 hash of the image
        :return: The distance as applied to this image, or None if it has no co-localization
        """
        rows = self.connector.get_view_from_table("max_distance", "colocalization_pairs",
                                                  ("image", Specifiers.EQUALS, image))
        return float(rows[0][0]) if rows and rows[0][0] is not None else None

    def get_colocalization_by_nucleus(self, image: str,
                                      pair: Tuple[str, str]) -> Dict[int, float]:
        """
        Method to get the share of each nucleus's foci that co-localize, for one channel pair

        The share counts the foci of BOTH channels of the pair: a nucleus with 3 foci in one
        channel, 2 in the other and 2 pairs between them is 4 of 5, 80 %. That is the definition
        roi.match used, except that each focus now counts towards its OWN nucleus -- the old pass
        credited both foci of a pair to the nucleus of the first one.

        :param image: The md5 hash of the image
        :param pair: The pair, by channel name, as get_colocalization_pairs returns it
        :return: {nucleus hash: share between 0 and 1}. A nucleus with no focus in either channel
            of the pair is absent -- there is nothing to co-localize, which is not the same as 0
        """
        counts = self.connector.count_colocalized_foci(image, pair[0], pair[1])
        return {nucleus: partnered / total for nucleus, (total, partnered) in counts.items()}

    def get_image_scale(self, image: str) -> Union[Tuple[float, float], None]:
        """
        Method to get the pixels-per-micrometre scale stored for an image

        `images.x_res` / `y_res` hold the conversion factor the user entered for this image, written
        per image at the end of the analysis. They are NOT the TIFF tags -- a file's declared
        resolution is not trusted, because not every microscope writes a meaningful one, so the
        value here is always one a person supplied.

        **ONLY WHEN `unit` SAYS SO, since 2026-09-24.** Registration used to store the file's raw
        declared resolution in the same two columns -- pixels per `Inch` or `Centimeter` -- and this
        returned it as pixels per micrometre: 68493.72 for `demo.tif`, which declares 2.70 px/um.
        The analysis writes `ANALYSIS_SCALE_UNIT` beside its factor, so the unit is what tells the
        two apart; registration no longer writes either, but rows registered before still hold the
        raw value, and they answer None here like any image nobody has set a factor for. A file's
        declaration is read from the file instead -- `ImageLoader.declared_pixels_per_micron`.

        Both columns are nullable and always have been, so None is a legitimate answer meaning
        "nobody has said what scale this image was acquired at". Callers must show pixels and say so
        rather than substituting a default: a wrong scale silently reports wrong micrometres.

        :param image: The md5 hash of the image
        :return: (x, y) in pixels per micrometre, or None if either is missing
        """
        rows = self.connector.get_view_from_table(("x_res", "y_res", "unit"), "images",
                                                  ("md5", Specifiers.EQUALS, image))
        if not rows:
            return None
        x_res, y_res, unit = rows[0]
        if unit != ANALYSIS_SCALE_UNIT:
            return None
        if x_res is None or y_res is None or x_res <= 0 or y_res <= 0:
            return None
        return float(x_res), float(y_res)

    def get_main_channel(self, image: str) -> str:
        """
        Method to get the main channel of the given image

        :param image: The md5 hash of the image
        :return: The name of the main channel
        """
        rows = self.connector.get_view_from_table("name", "channels",
                                                  (("md5", Specifiers.EQUALS, image),
                                                   ("main", Specifiers.EQUALS, 1)))
        # An image whose channels were never written, or written without a main channel, is a
        # legitimate state -- and this runs in EditorView.__init__, so [0][0] took the editor down
        # on open rather than reporting which image had no main channel
        return rows[0][0] if rows else None

    def get_roi_info(self, roi: int, image: str) -> Tuple:
        """
        Method to get general information about the roi

        The image is REQUIRED -- see get_hashes_of_associated_foci. Without it this returned
        rows[0] of a multi-image result, i.e. ANOTHER image's roi row, in whichever order SQLite
        happened to scan. The geometry columns are safe either way, because they derive from the
        area the hash is made of, but the per-image columns are not: measured on the testing
        database, of 4506 roi hashes shared between images, 2836 disagree on `associated`, 2353 on
        `co_localized` and 847 on `detection_method`.

        :param roi: The md5 hash of the roi
        :param image: The md5 hash of the image the roi belongs to
        :return: The retrieved information
        """
        rows = self.connector.get_view_from_table(Specifiers.ALL, "roi",
                                                  (("hash", Specifiers.EQUALS, roi),
                                                   ("image", Specifiers.EQUALS, image)))
        return rows[0] if rows else None

    def get_statistics_for_roi(self, roi: int, image: str) -> Tuple:
        """
        Method to get the statistics for the given roi

        The image is REQUIRED -- see get_hashes_of_associated_foci, and the statistics table
        carries the same composite key. The AREA cannot differ between the images sharing a hash,
        since the area is what the hash is derived from, but the INTENSITIES can: they are read
        out of that image's own pixels. Measured on the testing database, of 2270 statistics
        hashes spanning more than one image, 0 disagree on area and 575 disagree on intensity.

        :param roi: The roi hash to get the statistics for
        :param image: The md5 hash of the image the roi belongs to
        :return: The statistics
        """
        stats = self.connector.get_view_from_table(Specifiers.ALL, "statistics",
                                                  (("hash", Specifiers.EQUALS, roi),
                                                   ("image", Specifiers.EQUALS, image)))
        # None, not (): an empty tuple is falsy AND indexable-with-IndexError, so it read as a
        # row that happens to be empty. Every accessor in this class now answers None for "no such
        # row" -- see get_info_for_image for where the convention was first written down
        return stats[0] if stats else None

    def get_points_for_roi(self, roi: ROI) -> List[Tuple]:
        """
        Method to get the points of a roi

        This one takes NO image, deliberately, and it is the exception among the hash-keyed
        queries. The points table has no image column at all -- its key is ("hash", "row",
        "column_") -- so a run list is shared by every image whose roi hashes to the same value,
        by construction. That is not a defect of this query: the hash IS the area, so the single
        stored run list is the right answer for all of them. Verified on the testing database --
        a hash present in five images has exactly five points rows, all five distinct.

        The sharing IS a defect elsewhere: delete_existing_image_data removes these rows by hash
        alone, so re-analysing one image destroys the geometry of every other image sharing it.

        :param roi: The roi hash to get the points for
        :return: The saved points
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "points",
                                                  ("hash", Specifiers.EQUALS, roi))

    def get_nuclei_clipped_by_border(self, image: str,
                                     nuclei: Iterable[int] = None) -> Set[int]:
        """
        Method to get the hashes of the nuclei that are cut off by the edge of the image

        **Derived, not stored.** The flag is computed from the geometry already in the database --
        the run list in `points` against `images.width`/`height` -- so it needs no column, no
        migration, and it answers for images analysed long before the flag existed. `roi`'s own
        `center_x`/`center_y`/`width`/`height` cannot be used: the centre is the CENTROID, not the
        centre of the bounding box, so the box position is not recoverable from them.

        Rows are bounded by `height` and columns by `width`, verified against the testing database
        on the five non-square images it holds -- 1384 x 1032 rows reaching row 1031 and column
        1384. Runs are (row, first_col, length) and half open, so `first_col + length` is one past
        the last pixel and the right edge is `>= width` while the bottom edge is `>= height - 1`.

        One query per nucleus rather than one aggregate query for the image: `MIN`/`MAX` cannot go
        through Connector.get_view_from_table, whose parameter check rejects parentheses. Measured
        on the testing database, the loop costs 3 ms for a 15-nucleus image against the several
        queries per nucleus this table already runs.

        :param image: The md5 hash of the image
        :param nuclei: Optional; the nucleus hashes to check. Queried if not given
        :return: The hashes of the nuclei touching any of the four image edges
        """
        info = self.get_info_for_image(image)
        if info is None:
            LOGGER.warning("No row for image %s -- no nucleus can be checked against its border",
                           image)
            return set()
        width, height = info[7], info[8]
        if nuclei is None:
            nuclei = self.get_nuclei_hashes_for_image(image)
        clipped = set()
        for nucleus in nuclei:
            points = self.connector.get_view_from_table(("row", "column_", "width"), "points",
                                                        ("hash", Specifiers.EQUALS, nucleus))
            # A nucleus whose points are gone cannot be judged, and saying "whole" would be an
            # answer rather than an absence. It is left out and the missing geometry is already
            # reported by the row builder
            if not points:
                continue
            if (min(p[0] for p in points) <= 0
                    or min(p[1] for p in points) <= 0
                    or max(p[0] for p in points) >= height - 1
                    or max(p[1] + p[2] for p in points) >= width):
                clipped.add(nucleus)
        return clipped

    def get_table_data_for_image(self, image: str, name: str = None,
                                 pair: Optional[Tuple[str, str]] = None) -> List[List]:
        """
        Method to create a result table for the given image

        :param image: The md5 hash of the image
        :param name: Optional: The file name of the image
        :param pair: Optional: the channel pair whose co-localization fills the Co-Loc. cell. None
            means this image's first pair -- see ``colocalization_cells`` for what an image
            analysed before pairs existed shows instead
        :return: The created table
        """
        # Once per image, not once per nucleus: the scale is a property of the image
        scale = self.get_image_scale(image)
        if scale is None:
            LOGGER.warning("Image %s has no conversion factor -- areas and axes are reported in "
                           "PIXELS, not micrometres. Set the factor for this image to convert "
                           "them", image)
        # Get all nuclei associated with this image
        nucs = self.get_nuclei_hashes_for_image(image)
        # Once per image, like the scale: the border test needs the image dimensions, which are one
        # row, and the set is then a lookup per nucleus rather than a query per nucleus per channel
        clipped = self.get_nuclei_clipped_by_border(image, nucs)
        # Hoisted out of the nucleus loop: it does not depend on the nucleus, so it was one query
        # per nucleus for one answer. Fetching it here is also what makes the check below possible
        # exactly once per image rather than once per row.
        channels = sorted(self.get_channel_names(image, False))
        # A nucleus row is only emitted once per channel, so with no channel the assembled row is
        # discarded and this method returns [] -- however many nuclei the image has. Reaching this
        # with nuclei present means measured results are being dropped, so it is logged rather than
        # left to surface as an empty table with no explanation.
        if not channels and nucs:
            LOGGER.error("No active non-main channel for image %s -- its result table will be "
                         "empty despite %d nuclei", image, len(nucs))
        # Once per image, like the scale: one aggregate query for every nucleus. None means the
        # image predates per-pair storage and the cell falls back to roi.match below
        coloc = self.colocalization_cells(image, pair)
        rows = []
        for nuc in nucs:
            # Get the name of the image
            name = name if name else "Name not available"
            # Get the general ROI information
            general = self.get_roi_info(nuc, image)
            # None means the hash came back from get_nuclei_hashes_for_image but its roi row is
            # gone -- there is no row to render, so the nucleus is skipped loudly rather than
            # raising three frames further down on general[10]
            if general is None:
                LOGGER.warning("No roi row for nucleus %s of image %s -- skipped in the result "
                               "table", nuc, image)
                continue
            # Get nucleus statistics
            stats = self.get_statistics_for_roi(nuc, image)
            if coloc is not None:
                # Per pair, from the colocalization table. A nucleus absent from it has no focus
                # in either channel of the pair -- nothing to co-localize, which is not 0 %
                share = coloc.get(nuc)
                match = NO_COLOCALIZATION if share is None else f"{share * 100:.2f}"
            # LEGACY: an image analysed before 2026-09-24 has no pairs and carries ONE value per
            # nucleus in roi.match, computed on its first two foci channels. RW ruled it is shown
            # as it is rather than recomputed. roi.match is -1 when the image has a single
            # channel, where co-localization is not a meaningful concept, and None when it was never
            # computed -- which is also what this build writes -- and both render as
            # NO_COLOCALIZATION. The test is explicit rather than a truthiness check because a
            # match of exactly 0 is a real measurement -- "these foci co-localize with nothing" --
            # and used to be reported as 100 % by the old
            # `general[10] * 100 if general[10] else 100`, which caught 0 along with the sentinels
            elif general[10] is None or general[10] == -1:
                match = NO_COLOCALIZATION
            else:
                match = f"{general[10] * 100:.2f}"
            # Create row for this nucleus. get_statistics_for_roi returns None for a nucleus
            # with no statistics row (an empty tuple until 2026-08-17), so every stats[] below
            # would raise and take the whole result table with it. The row is kept and the affected cells say so instead:
            # the nucleus exists and its foci counts are still countable.
            #
            # The MISSING-ROW case is not the only one. A row can exist with NULL ellipse columns:
            # calculate_ellipse_parameters runs only for roi marked main, and a focus that lies
            # outside every nucleus is stored with `associated = NULL` -- which is how a nucleus is
            # spelled, so get_nuclei_hashes_for_image hands it back here as one. `stats` is then
            # truthy and float(None) raised, taking the result table down. Reported from real use on
            # 2026-08-22, with the quality check switched off; with it on, delete_unassociated_foci
            # removes exactly those foci, which is why it had never surfaced.
            #
            # Widening this guard stops the crash, and that is ALL it does. The real defect is that
            # `associated IS NULL` means both "this is a nucleus" and "this focus belongs to
            # nothing", and repairing that touches stored data.
            if stats:
                # Center Y and Center X go through the same :.2f as every other numeric column.
                # get_center returns round(...), which under @njit yields a float for some ROI, and
                # SQLite's INTEGER affinity then stores the losslessly-representable ones as integer
                # and the rest as real -- so bare str() put "433" next to "435.9399961797561" in one
                # column. Measured before the fix: 151 of 4418 nuclei across 39 images stored a real
                # centre
                def _measure(value: Optional[float], factor: float = 1.0) -> str:
                    """One cell: the number, or NO_STATISTICS when the column is NULL"""
                    return NO_STATISTICS if value is None else f"{float(value) * factor:.2f}"

                # LENGTHS AND AREAS ARE CONVERTED FOR DISPLAY; COORDINATES ARE NOT.
                # RW, 2026-09-14: *"Convert every length or area. Centers and the like should still
                # be displayed as pixels, because they are literal coordinates in the image."* So
                # the ellipse area becomes um^2 and the two axes become um, while the centre stays
                # where it is -- a pixel position in the image the user is looking at.
                #
                # The stored values are untouched: every area in the database is still a pixel
                # count, which is ruling 1 of the same day. Only the cells change.
                #
                # scale is None when nobody has said what this image was acquired at. The values are
                # then shown in PIXELS rather than converted with a guessed factor, and the caller
                # is told -- see the warning below.
                # ELLIPTICITY IS DERIVED FROM THE AXES, not from the stored `ellipticity`
                # column. That column holds `shape_match` -- the fitted ELLIPSE AREA divided by the
                # measured area -- which is a goodness-of-fit ratio, not an elongation: a circle
                # and a long thin ellipse both score about 1, because both are described well by an
                # ellipse. It is unbounded above, so multiplying by 100 and calling it a percentage
                # produced values over 100 %: measured on the real database, 511 of 1712 rows did,
                # up to 138 %.
                #
                # 1 - minor/major is the usual meaning of ellipticity: 0 for a circle, approaching
                # 1 for a line, and bounded. The two are barely related -- Pearson r = 0.36 over
                # those same rows -- so this is a different quantity rather than a rescaling of the
                # old one, and stored results will not agree with re-displayed ones.
                #
                # The shape_match column is untouched and still stored; it simply stopped being
                # displayed under a name that does not describe it. It was NOT given a column of
                # its own: both table headers are at 13 columns and the main one already keeps its
                # labels short because Qt was eliding them and clipping the sort arrows.
                area_factor = scale[0] * scale[1] if scale else 1.0      # px^2 per um^2
                length_factor = scale[0] if scale else 1.0                # px per um
                major, minor = stats[12], stats[13]
                ellipticity = (None if major is None or minor is None or float(major) <= 0
                               else 1 - float(minor) / float(major))
                # stats[2] is the MEASURED area -- the pixel count of the roi. stats[15] is
                # `ellipse_area`, pi * r_major * r_minor of the fitted ellipse, which is what this
                # cell held until 2026-09-14 under a header reading "Area". The two differ by
                # exactly the fit ratio that used to be displayed as Ellipticity[%]: measured
                # 0.95 to 1.38 on the real database, so up to 38 % apart. RW: display the actual
                # area. The ellipse area stays in the statistics table and is simply not shown.
                measurements = [_measure(stats[11]), _measure(stats[10]),
                                _measure(stats[2], 1 / area_factor),
                                _measure(ellipticity, 100), _measure(stats[14]),
                                _measure(stats[12], 1 / length_factor),
                                _measure(stats[13], 1 / length_factor)]
            else:
                measurements = [NO_STATISTICS] * 7
            # The Edge cell is a property of the NUCLEUS, so it sits with the other nucleus-level
            # cells, before the per-channel pair appended below -- the result table merges its
            # nucleus-level columns across the channel rows and picks them out by header name
            edge = CLIPPED_BY_BORDER if nuc in clipped else NOT_CLIPPED_BY_BORDER
            row = [name, str(image), str(nuc)] + measurements + [match, edge]
            # Count the foci
            for channel in channels:
                rows.append(row + [channel,
                                   str(self.count_foci_for_nucleus_and_channel(nuc, channel, image))])
        return rows

    def colocalization_cells(self, image: str,
                             pair: Optional[Tuple[str, str]] = None) -> Optional[Dict[int, float]]:
        """
        Method to decide which co-localization an image's Co-Loc. cells show

        Three cases, and the first is the one that needs saying:

        * **no pair requested, and the image has no pairs**: None. The image was analysed before
          pairs existed, and its cells show the single stored ``roi.match`` value -- RW ruled on
          2026-09-24 that those are shown as they are, not recomputed. It is also what an image
          analysed with fewer than two foci channels gets, and there ``roi.match`` is NULL;
        * **no pair requested, and the image has pairs**: its first pair;
        * **a pair requested**: that pair, or an empty mapping -- every cell "n/a" -- when this
          image did not compare it. Deliberately so for an image from before pairs existed as
          well: its single value was computed on whichever two channels came first, and showing it
          under a pair the user picked BY NAME would be claiming a correspondence nothing records.

        :param image: The md5 hash of the image
        :param pair: The pair asked for, or None for each image's own first pair
        :return: {nucleus hash: share}, or None to fall back to roi.match
        """
        pairs = self.get_colocalization_pairs(image)
        if pair is None:
            return self.get_colocalization_by_nucleus(image, pairs[0]) if pairs else None
        pair = (pair[0], pair[1])
        return self.get_colocalization_by_nucleus(image, pair) if pair in pairs else {}

    def get_table_data_for_experiment(self, experiment: str,
                                      pair: Optional[Tuple[str, str]] = None):
        """
        Method to create a result table for the given experiment

        :param experiment: Name of the experiment
        :param pair: Optional: the channel pair for the Co-Loc. column, as for
            ``get_table_data_for_image``
        :return: The created table
        """
        # Get all images associated with the experiment
        imgs = self.get_associated_images_for_experiment(experiment)
        rows = []
        # Iterate over all images
        for ind, img in enumerate(imgs):
            start = time.time()
            img_name = self.get_image_filename(img)
            img_data = self.get_table_data_for_image(img, name=img_name, pair=pair)
            # Check if the image was assigned to a group
            group = self.get_associated_group_for_image(img, experiment)
            for row in img_data:
                row.insert(2, group)
                rows.append(row)
            LOGGER.debug("%04d:%04d\tGot data for: %s in %.2f secs",
                         ind + 1, len(imgs), img, time.time() - start)
        return rows

    def get_image_filename(self, md5: str) -> str:
        """
        Method to get the file name of the given image

        The one accessor that does NOT answer None for a missing row, and deliberately so: its
        result is a display label, every caller substitutes it straight into text, and "" is the
        empty label. Returning None here would push a None-check into each of those callers to
        produce the same string. Recorded because the rest of this class was brought onto the
        None contract on 2026-08-17 and this is the exception to it

        :param md5: The md5 hash of the image
        :return: The associated file name, or "" if the image has no recorded name
        """
        data = self.connector.get_view_from_table("file_name",
                                                  "encountered_names",
                                                  ("md5", Specifiers.EQUALS, md5))
        return data[0][0] if data else ""


class Inserter(DatabaseInteractor):
    """
    Class to modify the database
    """

    def add_new_image(self, md5: str, year: int, month: int, day: int, hour: int, minute: int,
                      channels: int, width: int, height: int, xres: Optional[float] = None,
                      yres: Optional[float] = None, res_unit: Optional[str] = None) -> None:
        """
        Method to add a new image to the database

        The date parts and the resolutions are numbers, not strings. ImageLoader.get_image_data
        produces year..minute as int (from datetime.timetuple()) and x_res/y_res as float (from
        _rational_to_scale), and create_tables.sql declares the five date columns INTEGER. The
        annotations said str until 2026-08-15, which nothing caught because SQLite accepts either.

        **xres/yres are Optional since 2026-08-21.** An image that declares no usable resolution
        yields None, which is stored as SQL NULL rather than as an in-band numeric sentinel -- the
        x_res/y_res columns are nullable and have always been. Readers must treat NULL as "unknown"
        and not as a scale.

        **Registration passes none of the three since 2026-09-24.** The columns hold the factor an
        analysis used, with `unit` naming it, and a file's own declaration -- in pixels per inch
        or centimetre -- in the same columns is what `get_image_scale` read as pixels per
        micrometre. The parameters stay for callers that set a known factor directly.

        :param md5: The md5 hash of the image
        :param year: The year the image was created
        :param month: The month the image was created
        :param day: The day the image was created
        :param hour: The hour the image was created
        :param minute: The minute the image was created
        :param channels: Number of image channels
        :param width: The width of the image
        :param height: The height of the image
        :param xres: The x resolution of the image, or None if it declares none
        :param yres: The y resolution of the image, or None if it declares none
        :param res_unit: The resolution unit of the image
        :return: None
        """
        self.connector.insert_or_replace_into("images",
                                              ("md5", "year", "month", "day", "hour", "minute",
                                               "channels", "width", "height", "x_res", "y_res",
                                               "unit", "analysed", "settings", "experiment", "modified"),
                                              (md5, year, month, day, hour, minute, channels, width, height,
                                               xres, yres, res_unit, 0, -1, None, 0))

    def add_new_experiment(self, name: str, details: str = "", notes: str = "") -> None:
        """
        Method to add a new experiment

        :param name: Name of the experiment
        :param details: Details about the experiment
        :param notes: Additional notes
        :return: None
        """
        self.connector.insert_or_replace_into("experiments", ("name", "details", "notes"), (name, details, notes))

    def add_image_to_experiment(self, image: str, name: str, details: str,
                                notes: str, group: str, create_new_exp: bool = True) -> None:
        """
        Method to add the given image to the experiment

        :param image: The image to associate with the experiment
        :param name: The name of the experiment
        :param details: Details of the experiment
        :param notes: Notes associated with the experiment
        :param group: The group to add the image to
        :param create_new_exp: Should the experiment be added to the database?
        :return: None
        """
        if create_new_exp:
            self.add_new_experiment(name, details, notes)
        # Add image to standard group
        self.add_image_to_experiment_group(image, name, group)
        # Update experiment column in images table
        self.associate_image_with_experiment(name, image)

    def add_image_to_experiment_group(self, image: str, experiment: str, group: str) -> None:
        """
        Method to add the given image to the given experiment group

        :param image: The hash of the image to add to the group
        :param experiment: Name of the experiment the group is associated with
        :param group: The name of the group
        :return: None
        """
        self.connector.insert_or_replace_into("groups", ("image", "experiment", "name"),
                                              (image, experiment, group))

    def delete_existing_image_data(self, image: str) -> None:
        """
        Method to delete all saved data for the given image

        :param image: The md5 hash of the image
        :return: None
        """
        self.connector.delete_existing_image_data(image)

    def associate_image_with_experiment(self, experiment: str, image: str) -> None:
        """
        Method to add an image to the given experiment

        :param experiment: The experiment to add the image to
        :param image: The md5 hash of the image
        :return: None
        """
        self.connector.update("images", ("experiment", experiment), ("md5", Specifiers.EQUALS, image))

    def set_image_scale(self, image: str, x_scale: float, y_scale: float) -> None:
        """
        Method to set the scale of the given image
        :param image: The md5 hash of the image
        :param x_scale: The x-axis scale of the image
        :param y_scale: The y-axis scale of the image
        :return: None
        """
        self.connector.update("images", (("x_res", x_scale), ("y_res", y_scale)),
                              ("md5", Specifiers.EQUALS, image))

    def set_image_scale_unit(self, image: str, unit: str) -> None:
        """
        Method to set the scale unit for the given image

        :param image: The md5 hash of the image
        :param unit: The unit in dots/x
        :return: None
        """
        self.connector.update("images", ("unit", unit), ("md5", Specifiers.EQUALS, image))

    def mark_image_as_modified(self, image: str) -> None:
        """
        Method to mark the given image as modified

        :param image: md5 hash of the image
        :return: None
        """
        self.connector.update("images", ("modified", True), ("md5", Specifiers.EQUALS, image))

    def remove_channels_for_image(self, image: str) -> None:
        """
        Method to remove every channel row of the given image

        add_channel is an INSERT OR REPLACE keyed on (md5, index), and delete_existing_image_data
        clears roi, points and statistics but never channels -- so registering an image with FEWER
        channels than a previous run left the surplus rows behind for good. The editor builds its
        channel list from this table and indexes the loaded array with it, so a stale row offered a
        channel the image does not have and raised IndexError on selection. Measured on the live
        database: one image declared 4 channels and carried 5 rows, the fifth named "Channel 5".

        :param image: The md5 hash of the image
        :return: None
        """
        self.connector.delete("channels", ("md5", Specifiers.EQUALS, image))

    def add_channel(self, image: str, index: int, name: str, active: bool, main: bool) -> None:
        """
        Method to add a new image channel to the database

        :param image: The image the channel is associated with
        :param index: The index of the channel
        :param name: The name of the channel
        :param active: Is the channel active or ignored?
        :param main: Is this channel the main channel of the image?
        :return: None
        """
        self.connector.insert_or_replace_into("channels", ("md5", "index_", "name", "active", "main"),
                                              (image, index, name, active, main))

    def save_roi_to_database(self,
                             roi_data: List,
                             line_data: List,
                             stat_data: List) -> None:
        """

        :param roi_data: The general ROI data
        :param line_data: The ROI line data
        :param stat_data: The ROI statistics data
        :return: None
        """
        self.save_general_roi_data(roi_data)
        self.save_roi_line_data(line_data)
        self.save_roi_statistics(stat_data)

    def save_roi_data_for_image(self, image: str,
                                roi_data: List,
                                line_data: List,
                                stat_data: List) -> None:
        """
        Method to save the roi of the given image to the database

        :param image: The md5 hash of the image
        :param roi_data: The general ROI data
        :param line_data: The ROI line data
        :param stat_data: The ROI statistics data
        :return: None
        """
        self.save_roi_to_database(roi_data, line_data, stat_data)
        self.connector.update("images", ("analysed", True), ("md5", Specifiers.EQUALS, image))

    def save_colocalization(self, image: str, pairs: Iterable[Tuple[str, str]],
                            max_distance: float, rows: Iterable[Tuple]) -> None:
        """
        Method to replace an image's co-localization with a new result

        REPLACES rather than adds, for both tables: the analysis and the editor's recomputation
        both hand over the complete result for the image, and a pair or a focus from the previous
        one left standing would be counted alongside it.

        :param image: The md5 hash of the image
        :param pairs: The channel pairs that were compared, by name. Recorded even when a pair
            produced no rows, so "compared, nothing found" stays distinct from "not compared"
        :param max_distance: The distance, in pixels for this image, the pairs were compared at
        :param rows: (focus hash, channel_a, channel_b, partner hash or None), as
            MapComparator.colocalize returns them
        :return: None
        """
        self.connector.delete("colocalization", ("image", Specifiers.EQUALS, image))
        self.connector.delete("colocalization_pairs", ("image", Specifiers.EQUALS, image))
        pair_rows = [(image, a, b, float(max_distance)) for a, b in pairs]
        if pair_rows:
            self.connector.insert_or_replace_into(
                "colocalization_pairs", ("image", "channel_a", "channel_b", "max_distance"),
                pair_rows, True)
        focus_rows = [(image, int(focus), a, b, None if partner is None else int(partner))
                      for focus, a, b, partner in rows]
        if focus_rows:
            self.connector.insert_or_replace_into(
                "colocalization", ("image", "focus", "channel_a", "channel_b", "partner"),
                focus_rows, True)

    def set_image_analysed(self, image: str, analysed: bool = True) -> None:
        """
        Method to mark an image as analysed, independently of whether anything was found

        **This exists because "analysed" means "an analysis ran", not "an analysis found
        something", and the code used to conflate the two.** The flag was set only as a side effect
        of `save_roi_data_for_image`, which the caller skips when there is no ROI data -- so an
        image whose nuclei the detector could not find was never marked, and the manual editor,
        which is gated on the flag, could not be opened to add them by hand. That is precisely the
        image a user most needs the editor for.

        :param image: The md5 hash of the image
        :param analysed: The value to set
        :return: None
        """
        self.connector.update("images", ("analysed", analysed),
                              ("md5", Specifiers.EQUALS, image))

    def save_general_roi_data(self, roi_data: List) -> None:
        """
        Method to save the given general ROI data to the database

        :param roi_data: The general ROI data to save
        :return: None
        """
        # Nothing detected in this image -- the isinstance check below indexes element 0
        if not roi_data:
            return
        self.connector.insert_or_replace_into("roi", ("hash", "image", "auto", "channel",
                                                      "center_x", "center_y", "width", "height",
                                                      "associated", "detection_method", "match", "co_localized"),
                                              roi_data, isinstance(roi_data[0], tuple))

    def save_roi_line_data(self, line_data: List) -> None:
        """
        Method to save the line data

        :param line_data: The line data to save
        :return: None
        """
        # Nothing detected in this image -- the check below indexes two levels in
        if not line_data or not line_data[0]:
            return
        # Check if many
        many = isinstance(line_data[0][0], tuple)
        if many:
            for ld in line_data:
                self.connector.insert_or_replace_into("points", ("hash", "row", "column_", "width"),
                                                      ld, True)
        else:
            self.connector.insert_or_replace_into("points", ("hash", "row", "column_", "width"),
                                                  line_data, True)

    def save_roi_statistics(self, stat_data: List) -> None:
        """
        Method to save the given statistics data

        :param stat_data: The data to save
        :return: None
        """
        # Nothing detected in this image -- the isinstance check below indexes element 0
        if not stat_data:
            return
        self.connector.insert_or_replace_into("statistics", ("hash", "image", "area", "intensity_average",
                                                             "intensity_median", "intensity_maximum",
                                                             "intensity_minimum", "intensity_std", "eccentricity",
                                                             "roundness", "ellipse_center_x", "ellipse_center_y",
                                                             "ellipse_major", "ellipse_minor", "ellipse_angle",
                                                             "ellipse_area", "orientation_vector_x",
                                                             "orientation_vector_y", "ellipticity"),
                                              stat_data, isinstance(stat_data[0], tuple))

    def remove_image_from_group(self, image: str, experiment: str) -> None:
        """
        Method to remove the given image from the groups of the given experiment

        :param image: The image in question
        :param experiment: The experiment the group belongs to
        :return: None
        """
        self.connector.delete("groups", (("image", Specifiers.EQUALS, image),
                                         ("experiment", Specifiers.EQUALS, experiment)))

    def remove_image_from_experiment(self, image: str) -> None:
        """
        Method to remove the given image from its experiment

        :param image: The md5 hash of the image
        :return: None
        """
        self.connector.update("images", ("experiment", Specifiers.NULL), ("md5", Specifiers.EQUALS, image))

    def remove_all_images_from_experiment(self, experiment: str) -> None:
        """
        Method to remove all images from a given experiment

        :param experiment: Name of the experiment
        :return: None
        """
        self.connector.update("images", ("experiment", Specifiers.NULL), ("experiment", Specifiers.EQUALS, experiment))

    def remove_group_associations_for_experiment(self, experiment: str) -> None:
        """
        Method to remove every group association of the given experiment

        The counterpart add_image_to_experiment_group had none, so nothing in the project could
        take a row OUT of the groups table -- and that table is what
        get_associated_images_for_experiment reads experiment membership from, falling back to
        images.experiment only when it is empty. An image removed from an experiment or from a
        group was therefore re-inserted by the next save and came back on the next load.

        :param experiment: Name of the experiment whose group rows should be removed
        :return: None
        """
        self.connector.delete("groups", ("experiment", Specifiers.EQUALS, experiment))

    def rename_experiment(self, old_name: str, new_name: str) -> None:
        """
        Method to rename an experiment, carrying its associations with it

        All three tables are updated together because the schema declares NO foreign keys: nothing
        cascades, so a rename that touched only `experiments` would strand every group and image
        under a name that no longer exists. Renaming was previously done by writing a row under the
        new name and leaving the old one, which is why an experiment appeared twice.

        :param old_name: The name the experiment currently has
        :param new_name: The name it should have
        :return: None
        """
        self.connector.update("experiments", ("name", new_name),
                              ("name", Specifiers.EQUALS, old_name))
        self.connector.update("groups", ("experiment", new_name),
                              ("experiment", Specifiers.EQUALS, old_name))
        self.connector.update("images", ("experiment", new_name),
                              ("experiment", Specifiers.EQUALS, old_name))

    def update_setting(self, key: str, value: Union[str, int, float]) -> None:
        """
        Method to update the given setting in the database

        :param key: The key of the setting
        :param value: The value to save
        :return: None
        """
        self.connector.update("settings", ("value", value), ("key_", Specifiers.EQUALS, key))

    def update_image_experiment_association(self, image: str, experiment: str) -> None:
        """
        Method to change the associated experiment of an image

        :param image: The md5 hash of the image
        :param experiment: The name of the experiment
        :return: None
        """
        self.connector.update("images", ("experiment", experiment), ("md5", Specifiers.EQUALS, image))

    def associate_focus_with_nucleus(self, nucleus: int, focus: int) -> None:
        """
        Method to associate the given focus-nucleus pair

        :param nucleus: Hash of the nucleus
        :param focus: Hash of the focus
        :return: None
        """
        self.connector.update("roi", ("associated", nucleus), ("hash", Specifiers.EQUALS, focus))

    def reset_nucleus_focus_association(self, nucleus: int) -> None:
        """
        Function to disassociate all foci from the given nucleus

        :param nucleus: Hash of the nucleus
        :return: None
        """
        self.connector.update("roi", ("associated", Specifiers.NULL),
                              ("associated", Specifiers.EQUALS, nucleus))

    def reset_nuclei_foci_associations(self, nuclei: Tuple[int]) -> None:
        """
        Function to disassociate all foci from the given nuclei

        :param nuclei: List of nucleus hashes
        :return: None
        """
        for nucleus in nuclei:
            self.reset_nucleus_focus_association(nucleus)

    def delete_roi_from_database(self, ident: int, image: str) -> None:
        """
        Method to remove the given roi of the given image from the database

        The image is REQUIRED, and that is the whole point: hash(roi) is md5(channel name + area)
        and carries no image, so an identical small focus in the same channel of two different
        images gets the same hash. Deleting by hash alone removed the other image's roi outright.

        :param ident: md5 hash of the roi
        :param image: The md5 hash of the image the roi belongs to
        :return: None
        """
        self.delete_roi_data(ident, image)
        self.delete_roi_points(ident, image)
        self.delete_roi_statistics(ident, image)

    def delete_roi_data(self, ident: int, image: str) -> None:
        """
        Method to remove the given roi of the given image from the roi table

        :param ident: The md5 hash of the roi
        :param image: The md5 hash of the image the roi belongs to
        :return: None
        """
        self.connector.delete("roi", (("hash", Specifiers.EQUALS, ident),
                                      ("image", Specifiers.EQUALS, image)))

    def delete_roi_points(self, ident: int, image: str) -> None:
        """
        Method to delete the saved area data of the given roi

        **Only when no other image's roi carries the same hash.** The points table keys on
        (hash, row, column_) and has no image column, so identically-hashed roi on two images share
        one set of rows; deleting them for one image left the other with a roi row and no area, and
        the manual editor raised on it. The rows stay until the last holder of the hash goes.

        :param ident: The md5 hash of the roi
        :param image: The md5 hash of the image the roi belongs to
        :return: None
        """
        shared = self.connector.count_instances(
            "hash", "roi", (("hash", Specifiers.EQUALS, ident),
                            ("image", Specifiers.NOTEQUALS, image)))
        if shared:
            LOGGER.debug("Keeping the points of roi %s: %d other image(s) share its hash",
                         ident, shared)
            return
        self.connector.delete("points", ("hash", Specifiers.EQUALS, ident))

    def delete_roi_statistics(self, ident: int, image: str) -> None:
        """
        Method to delete the saved roi statistics

        :param ident: The md5 hash of the roi
        :param image: The md5 hash of the image the roi belongs to
        :return: None
        """
        self.connector.delete("statistics", (("hash", Specifiers.EQUALS, ident),
                                             ("image", Specifiers.EQUALS, image)))

    def reset_database(self) -> None:
        """
        Method to reset the database

        :return: None
        """
        self.connector.reset_database()

    def reset_analysis_data(self) -> None:
        """
        Method to reset the analysis data

        :return: None
        """
        self.connector.reset_analysis_data()

    def register_image_filename(self, path: str) -> None:
        """
        Method to add the file name to the database

        :param path: The path leading to the file
        :return: None
        """
        # Get the md5 hash of the image
        md5 = ImageLoader.calculate_image_id(path)
        # Add the file to the database
        filename = os.path.splitext(os.path.basename(path))[0]
        self.connector.insert_or_replace_into("encountered_names", ("md5", "file_name"), (md5, filename))

    def register_image_filenames(self, paths: Union[List[str], Tuple[str]]) -> None:
        """
        Method to register the given files in the database

        :param paths: Paths leading to the images
        :return: None
        """
        # Calculate all needed values
        vals = []
        for path in paths:
            md5 = ImageLoader.calculate_image_id(path)
            # Without the extension, matching register_image_filename -- the stored file_name feeds
            # the result tables and the CSV export, so the two registration paths must agree
            filename = os.path.splitext(os.path.basename(path))[0]
            vals.append((md5, filename))
        self.connector.insert_or_replace_into("encountered_names", ("md5", "file_name"), vals, True)
