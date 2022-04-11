import platform
import os
from gui import Paths


def create_shortcut():
    # Check os
    os_name = platform.system()
    if os_name == "Windows":
        _create_shortcut_windows()
    elif os_name == "Linux":
        _create_shortcut_linux()
    elif os_name == "Darwin":
        _create_shortcut_apple()
    else:
        raise ValueError("System not supported!")


def _create_shortcut_windows():
    # Get user desktop path
    desk_path = os.path.join(os.path.join(os.environ['USERPROFILE'][3:]), 'Desktop', "NucDetect.bat")
    # Create a shortcut
    with open(desk_path, "w") as sc:
        sc.write(f"cd {Paths.nuc_detect_dir}{os.sep}gui\npython -m NucDetectAppQt")


def _create_shortcut_linux():
    pass


def _create_shortcut_apple():
    pass


if __name__ == "__main__":
    create_shortcut()
