import configparser
import os

class IniParser(configparser):
    __slots__ = (
        "config_path"
    )

    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self._load_ini_file(config_path)

    def _load_ini_file(self, path):
        """
        Method to load settings.ini at the given path

        :param path: Path leading to the folder of settings.ini
        :return: None
        """
        pass

    def _check_if_ini_file_exists(self, path: str) -> None:
        """
        Method to check if the standard
        :param path:
        :return:
        """
        pass
