import os
import sqlite3
import time
from enum import Enum
from typing import Tuple, Dict, List, Union, Iterable, Any

from detector_modules.ImageLoader import ImageLoader
from gui import Paths
from roi.ROI import ROI


class Specifiers(Enum):
    ALL = "*"
    IS = "IS"
    NULL = "NULL"
    EQUALS = "="
    GREATER = ">"
    GREATEREQUALS = ">="
    LESSER = "<"
    LESSEREQUALS = "<="


class Connector:

    def __init__(self, protected: bool = True):
        self.connection, self.cursor = self.connect_to_database(protected)
        # Set the cache size to 50000 pages (2 GB)
        self.cursor.execute("PRAGMA cache_size=50000;")
        # Load needed scripts
        self.commands = self.load_sql_commands()
        # Get information about the tables of the database
        self.table_info = self.get_table_info_from_database()

    @staticmethod
    def connect_to_database(protected: bool = True) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        """
        Method to connect to the general database

        :param protected: If true, no concurrent access to the database is allowed
        :return: The connection and cursor to the database
        """
        connection = sqlite3.connect(Paths.database, check_same_thread=protected)
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
                # Read file and append to command list
                with open(os.path.join(root, file), "r") as f:
                    commands[file[:-4]] = f.read()
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
        self.cursor.executescript(self.commands["create_tables"])
        self.table_info = self.get_table_info_from_database()

    @staticmethod
    def check_parameters(*args) -> None:
        """
        Method to check multiple parameters for illegal characters

        :param args: All passed parameters
        :return: None
        """
        for param in args:
            if isinstance(param, list) or isinstance(param, tuple):
                for p in param:
                    Connector.check_parameter(p)
            else:
                Connector.check_parameter(param)

    @staticmethod
    def check_parameter(param: Any) -> None:
        """
        Throws Exception if the parameter contains illegal characters

        :param param: The parameter to check
        :return:None
        """
        if isinstance(param, Specifiers) or not isinstance(param, str):
            return
        for c in ";,():\'\"\\/<>!$§%&[]{}´`|~#*=": # TODO Add point back
            if c in param:
                error = f"Query rejected: Parameter contains illegal character \"{c}\""
                raise ValueError(error)

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

        :param image: md5 hash of the image
        :return: None
        """
        # Check parameter
        self.check_parameter(image)
        query = self.commands["delete_existing_image_data"].replace("<img_hash>",
                                                                    self.convert_value(image))
        self.cursor.executescript(query)

    def count_instances(self, column: str, table: str, where: Tuple = ()) -> int:
        """
        Method to count the instances of column in the given table

        :param column: The column to count
        :param table: The table where the column can be found
        :param where: The condition to count
        :return: The number of found instances
        """
        self.check_for_table(table)
        self.check_parameters(column, table, where)
        if not where:
            query = self.commands["count"].replace("<column>", column).replace("<table_name>", table)
            return self.cursor.execute(query).fetchall()[0][0]
        else:
            where = self.convert_where_statement(where)
            query = self.commands["count_where"].replace("<column>", column) \
                .replace("<table_name>", table).replace("<condition>", where)
            return self.cursor.execute(query).fetchall()[0][0]

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
        # Check parameters for illegal characters
        self.check_parameters(table, columns, values)
        # Check if the requested table exists
        self.check_for_table(table)
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
        self.check_parameters(table, values, where)
        set_stm = self.convert_set_statement(values)
        where = self.convert_where_statement(where)
        query = self.commands["update"].replace("<table_name>", table) \
            .replace("<set_values>", set_stm).replace("<condition>", where)
        self.cursor.execute(query)

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
        self.check_parameters(table, where)
        where = self.convert_where_statement(where)
        query = self.commands["delete"].replace("<table_name>", table).replace("<condition>", where)
        self.cursor.execute(query)

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
        # Check for table
        self.check_for_table(table)
        # Check table name for illegal characters
        self.check_parameters(column, table, where)
        if not where:
            query = self.commands["select_from"].replace("<columns>", columns).replace("<table_name>", table)
            return self.cursor.execute(query).fetchall()
        else:
            where = self.convert_where_statement(where)
            query = self.commands["select_from_where"].replace("<columns>", columns) \
                .replace("<table_name>", table).replace("<condition>", where)
            return self.cursor.execute(query).fetchall()

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

    @staticmethod
    def convert_set_statement(values: Iterable[Tuple[str, str]]) -> str:
        """
        Method to convert the given iterable to a usable SET statment

        :param values: The values to convert
        :return: The usable SET statement
        """
        if isinstance(values[0], tuple):
            return ",".join([f"{x[0]}={Connector.convert_value(x[1])}" for x in values])
        else:
            return f"{values[0]}={Connector.convert_value(values[1])}"

    @staticmethod
    def convert_where_statement(where: Union[str, Tuple[Union[Tuple[str, Union[str, Specifiers], str]]]]) -> str:
        """
        Method to convert the given conditions to a usable where statement

        :param where: List of where statements to convert
        :return: The usable where statement
        """
        if not isinstance(where[0], tuple):
            cond1 = Connector.convert_value(where[0])
            sign = Connector.convert_value(where[1])
            cond2 = Connector.convert_value(where[2])
            return cond1 + sign + cond2
        else:
            return " AND ".join([Connector.convert_where_statement(x) for x in where])

    @staticmethod
    def convert_value(value: Union[float, int, str, Specifiers], quote: bool = True) -> str:
        """
        Method to convert the given value to a SQLite compatible string

        :param value: The value to convert
        :param quote: If true, strings will be quoted
        :return: The converted value
        """
        if isinstance(value, (float, int)):
            return f"{value}"
        elif isinstance(value, str):
            if quote:
                return f"\"{value}\""
            else:
                return value
        elif isinstance(value, Specifiers):
            if value is Specifiers.IS or value is Specifiers.NULL:
                return f" {value.value} "
            return f"{value.value}"
        elif isinstance(value, bool):
            return f"{int(value)}"


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

    def get_all_settings(self) -> List[Tuple[Union[str, int, float]]]:
        """
        Method to load the settings from the database

        :return: The saved settings
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
        details, notes = self.connector.get_view_from_table(("details", "notes"),
                                                            "experiments",
                                                            ("name", Specifiers.EQUALS, experiment))[0]
        return details, notes

    def get_channels_for_experiment(self, experiment: str, include_main: bool = False) -> List[str]:
        """
        Method to get the channel names associated with the given experiment

        :param experiment: The name of the experiment
        :param include_main: If true, the name of the main channel will be included
        :return: The name of the channels
        """
        # Select the images corresponding to the experiment
        imgs = self.get_associated_images_for_experiment(experiment)
        channels = [x[0] for x in self.connector.get_view_from_table("DISTINCT name", "channels",
                                                                     ("md5", Specifiers.EQUALS, imgs[0]))]
        # Get the main channel
        main = self.get_main_channel(imgs[0])
        if not include_main:
            channels.remove(main)
        return channels

    def get_main_channel_for_experiment(self, experiment: str) -> str:
        """
        Method to get the main channel of the given experiment

        :param experiment: The experiment
        :return: The name of the main channel
        """
        # Get first associated image
        img = self.get_associated_images_for_experiment(experiment)[0]
        return self.get_main_channel(img)

    def get_associated_images_for_experiment(self, experiment: str) -> List[str]:
        """
        Method to get the associated images for a given experiment

        :param experiment: The name of the experiment
        :return: List of the associated image hashes
        """
        return [x[0] for x in self.connector.get_view_from_table("md5", "images", ("experiment", Specifiers.EQUALS,
                                                                                   experiment))]

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

    def get_experiment_for_image(self, image: str) -> str:
        """
        Method to get the associated experiment for the given image

        :param image: md5 hash of the image
        :return: The name of the experiment
        """
        return self.get_info_for_image(image)[14]

    def get_info_for_image(self, image: str) -> Tuple[Union[str, int, float, None]]:
        """
        Method to get all saved information for the given image

        :param image: The md5 hash of the image
        :return: The information as list of strings
        """
        info = self.connector.get_view_from_table(Specifiers.ALL, "images", ("md5", Specifiers.EQUALS, image))
        return info[0] if info else ()

    def check_if_image_was_analysed(self, image: str) -> bool:
        """
        Method to check if the given image was analysed

        :param image: The md5 hash of the image
        :return: True if the image was analysed
        """
        return bool(self.connector.get_view_from_table("analysed", "images", ("md5", Specifiers.EQUALS, image))[0][0])

    def check_if_image_is_registered(self, image: str) -> bool:
        """
        Method to check if the given image is already registred in the database

        :param image: The md5 hash of the image
        :return: True if the image was found in the database
        """
        return bool(self.connector.get_view_from_table("file_name",
                                                       "encountered_names",
                                                       ("md5", Specifiers.EQUALS, image)))

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
                                                                      (("associated", Specifiers.IS, Specifiers.NULL),
                                                                       ("image", Specifiers.EQUALS, md5)))]

    def get_hashes_of_associated_foci(self, nucleus: str) -> List[str]:
        """
        Method to get the hashes of associated foci for the given nucleus

        :param nucleus: md5 hash of the nucleus
        :return: List of all focus hashes
        """
        return [x[0] for x in self.connector.get_view_from_table("hash", "roi",
                                                                 ("associated", Specifiers.EQUALS, nucleus))]

    def count_foci_for_nucleus_and_channel(self, nucleus: int, channel: str) -> int:
        """
        Method to count the associated foci for the given nucleus and channel

        :param nucleus: The md5 hash of the nucleus
        :param channel:The name of the channel
        :return: The number of associated foci
        """
        return self.connector.count_instances("hash", "roi", (("associated", Specifiers.EQUALS, nucleus),
                                                              ("channel", Specifiers.EQUALS, channel)))

    def get_modified_images(self) -> List[str]:
        """
        Method to get all images that were manually modified

        :return: The hashes of all modified images
        """
        return [x[0] for x in self.connector.get_view_from_table("md5", "images", ("modified", Specifiers.EQUALS, "1"))]

    def get_associated_roi(self, image: str) -> List[Tuple]:
        """
        Method to get information about the ROI associated with this image

        :param image: The image to get the ROI for
        :return: The retrieved information
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "roi", ("image", Specifiers.EQUALS, image))

    def get_channels(self, image: str) -> List[Tuple]:
        """
        Method to get information about the channels of this image

        :param image: The image to get the information for
        :return: None
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "channels", ("md5", Specifiers.EQUALS, image))

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

    def get_main_channel(self, image: str) -> str:
        """
        Method to get the main channel of the given image

        :param image: The md5 hash of the image
        :return: The name of the main channel
        """
        return self.connector.get_view_from_table("name", "channels", (("md5", Specifiers.EQUALS, image),
                                                                       ("main", Specifiers.EQUALS, 1)))[0][0]

    def get_roi_info(self, roi: int) -> Tuple:
        """
        Method to get general information about the roi

        :param roi: The md5 hash of the roi
        :return: The retrieved information
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "roi", ("hash", Specifiers.EQUALS, roi))[0]

    def get_statistics_for_roi(self, roi: int) -> Tuple:
        """
        Method to get the statistics for the given roi

        :param roi: The roi hash to get the statistics for
        :return: The statistics
        """
        stats = self.connector.get_view_from_table(Specifiers.ALL, "statistics", ("hash", Specifiers.EQUALS, roi))
        return stats[0] if stats else ()

    def get_points_for_roi(self, roi: ROI) -> List[Tuple]:
        """
        Method to get the points of a roi

        :param roi: The roi hash to get the points for
        :return: The saved points
        """
        return self.connector.get_view_from_table(Specifiers.ALL, "points", ("hash", Specifiers.EQUALS, roi))

    def get_table_data_for_image(self, image: str, name: str = None) -> List[List]:
        """
        Method to create a result table for the given image

        :param image: The md5 hash of the image
        :param name: Optional: The file name of the image
        :return: The created table
        """
        # Get all nuclei associated with this image
        nucs = self.get_nuclei_hashes_for_image(image)
        rows = []
        for nuc in nucs:
            # Get the name of the image
            name = name if name else "Name not available"
            # Get the general ROI information
            general = self.get_roi_info(nuc)
            # Get nucleus statistics
            stats = self.get_statistics_for_roi(nuc)
            # Calculate overall match for this nucleus
            match = general[10] * 100 if general[10] else 100
            # Create row for this nucleus
            row = [name, str(image), str(nuc), str(stats[11]), str(stats[10]), f"{stats[15]:.2f}",
                   f"{float(stats[18]) * 100:.2f}", f"{float(stats[14]):.2f}",
                   f"{float(stats[12]):.2f}", f"{float(stats[13]):.2f}", f"{match:.2f}"]
            # Count the foci
            for channel in sorted(self.get_channel_names(image, False)):
                row.append(str(self.count_foci_for_nucleus_and_channel(nuc, channel)))
            rows.append(row)
        return rows

    def get_table_data_for_experiment(self, experiment: str):
        """
        Method to create a result table for the given experiment

        :param experiment: Name of the experiment
        :return: The created table
        """
        # Get all images associated with the experiment
        imgs = self.get_associated_images_for_experiment(experiment)
        rows = []
        # Iterate over all images
        for ind, img in enumerate(imgs):
            start = time.time()
            img_name = self.get_image_filename(img)
            img_data = self.get_table_data_for_image(img, name=img_name)
            # Check if the image was assigned to a group
            group = self.get_associated_group_for_image(img, experiment)
            for row in img_data:
                row.insert(2, group)
                rows.append(row)
            print(f"{ind + 1:04d}:{len(imgs):04d}\tGot data for: {img} in {time.time() - start:.2f} secs")
        return rows

    def get_image_filename(self, md5: str) -> str:
        """
        Method to get the file name of the given image

        :param md5: The md5 hash of the image
        :return: The associated file name
        """
        return self.connector.get_view_from_table("file_name",
                                                  "encountered_names",
                                                  ("md5", Specifiers.EQUALS, md5))[0][0]


class Inserter(DatabaseInteractor):
    """
    Class to modify the database
    """

    def add_new_image(self, md5: str, year: str, month: str, day: str, hour: str, minute: str,
                      channels: int, width: int, height: int, xres: str, yres: str, res_unit: str) -> None:
        """
        Method to add a new image to the database
        :param md5: The md5 hash of the image
        :param year: The year the image was created
        :param month: The month the image was created
        :param day: The day the image was created
        :param hour: The hour the image was created
        :param minute: The minute the image was created
        :param channels: Number of image channels
        :param width: The width of the image
        :param height: The height of the imge
        :param xres: The x resolution of the image
        :param yres: The y resolution of the image
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
        self.connector.insert_or_replace_into("groups", ("image", "experiment", "name"), (image, experiment, group))

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

    def mark_image_as_modified(self, image: str) -> None:
        """
        Method to mark the given image as modified

        :param image: md5 hash of the image
        :return: None
        """
        self.connector.update("images", ("modified", True), ("md5", Specifiers.EQUALS, image))

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

    def save_general_roi_data(self, roi_data: List) -> None:
        """
        Method to save the given general ROI data to the database

        :param roi_data: The general ROI data to save
        :return: None
        """
        self.connector.insert_or_replace_into("roi", ("hash", "image", "auto", "channel",
                                                      "center_x", "center_y", "width", "height",
                                                      "associated", "detection_method", "match"),
                                              roi_data, isinstance(roi_data[0], tuple))

    def save_roi_line_data(self, line_data: List) -> None:
        """
        Method to save the line data

        :param line_data: The line data to save
        :return: None
        """
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
        self.connector.update("roi", ("associated", Specifiers.NULL), ("associated", Specifiers.EQUALS, nucleus))

    def reset_nuclei_foci_associations(self, nuclei: Tuple[int]) -> None:
        """
        Function to disassociate all foci from the given nuclei

        :param nuclei: List of nucleus hashes
        :return: None
        """
        for nucleus in nuclei:
            self.reset_nucleus_focus_association(nucleus)

    def delete_roi_from_database(self, ident: int) -> None:
        """
        Method to remove the given roi from the database

        :param ident: md5 hash of the roi
        :return: None
        """
        self.delete_roi_data(ident)
        self.delete_roi_points(ident)
        self.delete_roi_statistics(ident)

    def delete_roi_data(self, ident: int) -> None:
        """
        Method to remove the given roi from the roi table

        :param ident: The md5 hash of the roi
        :return: None
        """
        self.connector.delete("roi", ("hash", Specifiers.EQUALS, ident))

    def delete_roi_points(self, ident: int) -> None:
        """
        Method to delete the saved area data of the given roi

        :param ident: The md5 hash of the roi
        :return: None
        """
        self.connector.delete("points", ("hash", Specifiers.EQUALS, ident))

    def delete_roi_statistics(self, ident: int) -> None:
        """
        Method to delete the saved roi statistics

        :param ident: The md5 hash of the roi
        :return: None
        """
        self.connector.delete("statistics", ("hash", Specifiers.EQUALS, ident))

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
            filename = os.path.basename(path)
            vals.append((md5, filename))
        self.connector.insert_or_replace_into("encountered_names", ("md5", "file_name"), vals, True)
