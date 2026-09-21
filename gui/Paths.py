import os
import sys
from typing import List

def get_main_folder_path() -> str:
    """
    Function to get the main folder of the project

    :return: The path to the folder
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.join(os.path.dirname(sys.executable), "_internal")
    else:
        # during development, use project root (one level above gui/)
        exe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # DO NOT os.chdir(exe_dir) HERE. This function used to, and because it runs at import time the
    # whole program's working directory depended on when this module was first imported. Every path
    # in this module is absolute already, so nothing here ever needed it; what needed it was the
    # relative url()s in main.css, which resolved against the cwd and therefore only worked in a
    # development run. Those now go through Util.load_stylesheet's @@CSS_DIR@@ placeholder, so the
    # side effect has no remaining dependent -- verified by grep before removal: no relative open(),
    # os.getcwd or Path().resolve() anywhere in core/, gui/ or fcn/.
    #
    # Restoring it would also silently re-hide the class of defect it caused: any future path
    # resolved against the cwd would appear to work, but only for whoever imports this module first.
    return exe_dir

gen = get_main_folder_path()
gui = os.path.join(gen, "gui")

nuc_detect_dir = os.path.join(os.path.expanduser("~"), "NucDetect")
script_dir = os.path.join(gui, "definitions", "ui")
logo_dir = os.path.join(gui, "definitions", "images")
model_dir = os.path.join(gen, "fcn", "model")
css_dir = os.path.join(gui, "definitions", "css")
sql_dir = os.path.join(gen, "core", "database", "scripts")
log_path = os.path.join(nuc_detect_dir, "logs", "nucdetect.log")
# PROGRAM RESOURCES, NOT USER STATE -- it stays inside the package, and the dead alternative that
# stood on the next line until 2026-09-20 must not be restored.
#
# That line read `os.path.join(nuc_detect_dir, "settings")`, and a finding filed 2026-07-26 asked
# for it to be uncommented because "user settings are stored inside the installation directory".
# **Both halves of that are now false, and acting on it would break the settings dialog outright:**
#
#   * this directory holds `Widgets.py`, `__init__.py`, EIGHT `.ui` templates and `settings.json`,
#     every one of them version-controlled and shipped with the program. `Widgets.py` loads the
#     templates from here (`uic.loadUi(os.path.join(gpaths.settings_path, ui_file))`), so pointing
#     this at the user's home would look for `menu_slider.ui` in a directory that has never
#     contained one;
#   * nothing writes here at runtime -- no `json.dump`, no `open(..., "w")` anywhere under `gui/`
#     or `core/`. `settings.json` describes the WIDGETS; the values an analysis runs with live in
#     the database (`load_settings` -> `Requester.get_all_settings`). That has been true since
#     2026-08-22, when `save_menu_settings` was deleted and the JSON stopped being a second,
#     competing store of the settings.
#
# So there are no user settings in the installation directory to move. The user state that does
# exist -- database, images, logs, thumbnails, results -- is already under `nuc_detect_dir` above.
settings_path = os.path.join(gui, "settings")
about_txt_path = os.path.join(gui, "definitions", "about.txt")
# Inside the gui package on purpose. It used to sit at the project root and be resolved from
# NucDetectAppQT's __file__, which works from a checkout but not from an installed copy: only files
# inside a package are installed, so the seed image was missing wherever it was most needed.
demo_image = os.path.join(gui, "definitions", "demo.tif")
ui_main = os.path.join(script_dir, "nucdetect.ui")
ui_result_image_dialog = os.path.join(script_dir, "result_image_dialog.ui")
ui_exp_dial = os.path.join(script_dir, "experiment_dialog.ui")
ui_exp_dial_group_dial = os.path.join(script_dir, "group_dialog.ui")
ui_img_sel_dial = os.path.join(script_dir, "image_selection_dialog.ui")
ui_stat_dial = os.path.join(script_dir, "statistics_dialog.ui")
ui_settings_dial = os.path.join(script_dir, "settings_dialog.ui")
ui_editor_dial = os.path.join(script_dir, "result_editor_dialog.ui")
ui_experiment_selection_dial = os.path.join(script_dir, "experiment_selection_dialog.ui")
ui_analysis_settings_dial = os.path.join(script_dir, "analysis_settings_dialog.ui")
ui_save_dial = os.path.join(script_dir, "data_export_dialog.ui")
ui_stat_plot_settings_dial = os.path.join(script_dir, "statistics_settings.ui")

database = os.path.join(nuc_detect_dir, "nucdetect.db")
result_path = os.path.join(nuc_detect_dir, "results")
images_path = os.path.join(nuc_detect_dir, "images")
log_dir_path = os.path.join(nuc_detect_dir, "logs")
thumb_path = os.path.join(nuc_detect_dir, "thumbnails")


def ensure_directories() -> List[str]:
    """
    Function to create the working directories this module declares, if they do not exist yet

    Deliberately a function and NOT run at import time. This module declares the directories, so
    creating them belongs here rather than only in the GUI -- without it, anything using the core
    without the GUI fails on a missing folder, and constructing a ``Connector`` against a fresh
    HOME raises ``sqlite3.OperationalError: unable to open database file``. But doing it on import
    would make merely importing this module create folders in the user's home, which is the same
    class of hidden side effect as the import-time ``os.chdir`` this module used to perform -- and a
    more consequential one, since it writes to the filesystem. That ``chdir`` has since been removed
    (see ``get_main_folder_path``); this is the reasoning it left behind. Callers ask for it
    explicitly.

    :return: The directories that were created by this call, in creation order. Empty if they all
             existed already. Callers that seed a newly created directory -- see
             NucDetect.create_required_dirs, which copies the demo image into a new images folder --
             must key off this rather than re-testing the directory afterwards, since by then it
             exists either way
    """
    created = []
    for directory in (nuc_detect_dir, thumb_path, result_path, images_path, log_dir_path):
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
            created.append(directory)
    return created
