import os
import sys
from typing import List, Optional

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

# Databases live in their own directory, not loose in the NucDetect folder beside images,
# logs and results -- RW, 2026-09-22: *"After the change, databases should not exist in the main
# 'NucDetect' folder, but at 'NucDetect/data'."* It is also what makes per-experiment databases
# tractable: one directory to list, rather than picking .db files out of the user's working folder
# alongside their own backups.
data_dir = os.path.join(nuc_detect_dir, "data")
#: The database used when no experiment-specific one is chosen. Its NAME is unchanged, so an
#: existing file keeps working once it is in the new directory -- see `relocate_legacy_database`.
database = os.path.join(data_dir, "nucdetect.db")
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
    for directory in (nuc_detect_dir, data_dir, thumb_path, result_path, images_path,
                      log_dir_path):
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
            created.append(directory)
    return created


def database_for(experiment: str) -> str:
    """
    Build the path of the database belonging to one experiment

    Per-experiment databases were RW's instruction of 2026-09-22. This function is the whole of
    the naming rule, so that the rule lives in one place rather than at every call site.

    **The name is sanitised, not trusted.** An experiment name is free text a user types, and it
    reaches the filesystem here: a name containing a separator would otherwise write outside
    ``data_dir``, and one containing a character Windows reserves would fail at ``open`` with an
    error naming neither the experiment nor the reason.

    :param experiment: The experiment's name
    :return: The absolute path of its database
    :raises ValueError: if the name has no usable characters at all
    """
    # Windows reserves < > : " / \ | ? * and the control characters; POSIX reserves / only.
    # Replacing rather than rejecting, because the alternative is telling a user their experiment
    # cannot be stored because of a colon they cannot see the significance of
    reserved = '<>:"/\\|?*'
    safe = "".join("_" if c in reserved or ord(c) < 32 else c for c in experiment).strip()
    # A trailing dot or space is legal in a name and illegal in a Windows filename
    safe = safe.rstrip(". ")
    # Tested on what SURVIVED, not on the sanitised string: a name made entirely of reserved
    # characters sanitises to a run of underscores, which is non-empty and says nothing about
    # which experiment it holds. "///" producing "___.db" was caught by the harness, not by
    # reading this
    survivors = "".join(c for c in experiment if c not in reserved and ord(c) >= 32).strip(". ")
    if not safe or not survivors:
        raise ValueError(f"experiment name {experiment!r} yields no usable file name")
    return os.path.join(data_dir, f"{safe}.db")


def list_databases() -> List[str]:
    """
    List the databases in the data directory, newest first

    :return: Absolute paths. Empty when the directory does not exist yet
    """
    if not os.path.isdir(data_dir):
        return []
    found = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".db")]
    return sorted(found, key=os.path.getmtime, reverse=True)


def relocate_legacy_database() -> Optional[str]:
    """
    Move a database left in the NucDetect folder by an older build into the data directory

    **A rename, not a conversion.** The file's contents are untouched; only where it sits changes,
    and both locations are under the same directory, so ``os.replace`` is atomic and instant even
    for the 749 MB files this project actually has. RW's *"do not migrate the existing databases"*
    is about their SCHEMA -- see ``core.database.schema_version`` -- and does not apply here.

    **It moves only the application's own database.** The NucDetect folder is also where users keep
    their own backups and exported copies; those are not this program's to move, and sweeping every
    `.db` file would take them too.

    **It never overwrites.** If a database already exists in the data directory, the legacy file is
    left exactly where it is and the caller is told nothing moved -- two databases with a claim to
    the same name is a question for the user, not for a startup path.

    :return: The path it was moved to, or None if there was nothing to move
    """
    legacy = os.path.join(nuc_detect_dir, "nucdetect.db")
    if not os.path.isfile(legacy) or os.path.exists(database):
        return None
    os.makedirs(data_dir, exist_ok=True)
    os.replace(legacy, database)
    return database
