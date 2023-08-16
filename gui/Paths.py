import os


def get_main_folder_path() -> str:
    """
    Function to get the main folder of the project

    :return: The path to the folder
    """
    # Get the current file path
    fp = __file__
    # Split the path and find NucDetect folder
    while os.path.split(fp)[1] != "NucDetect":
        fp = os.path.split(fp)[0]
    return fp


gen = get_main_folder_path()
gui = os.path.join(gen, "gui")

nuc_detect_dir = os.path.join(os.path.expanduser("~"), "NucDetect")
script_dir = os.path.join(gui, "definitions", "ui")
logo_dir = os.path.join(gui, "definitions", "images")
model_dir = os.path.join(gen, "fcn", "model")
css_dir = os.path.join(gui, "definitions", "css")
sql_dir = os.path.join(gen, "core", "database", "scripts")
log_path = os.path.join(gen, "logs")

ui_main = os.path.join(script_dir, "nucdetect.ui")
ui_result_image_dialog = os.path.join(script_dir, "result_image_dialog.ui")
ui_exp_dial = os.path.join(script_dir, "experiment_dialog.ui")
ui_exp_dial_group_dial = os.path.join(script_dir, "group_dialog.ui")
ui_img_sel_dial = os.path.join(script_dir, "image_selection_dialog.ui")
ui_stat_dial = os.path.join(script_dir, "statistics_dialog.ui")
ui_settings_dial = os.path.join(script_dir, "settings_dialog.ui")
ui_editor_dial = os.path.join(script_dir, "result_editor_dialog.ui")
ui_editor_auto_dial = os.path.join(script_dir, "auto_edit_dialog.ui")
ui_experiment_selection_dial = os.path.join(script_dir, "experiment_selection_dialog.ui")
ui_analysis_settings_dial = os.path.join(script_dir, "analysis_settings_dialog.ui")
ui_save_dial = os.path.join(script_dir, "data_export_dialog.ui")

database = os.path.join(nuc_detect_dir, "nucdetect.db")
result_path = os.path.join(nuc_detect_dir, "results")
images_path = os.path.join(nuc_detect_dir, "images")
thumb_path = os.path.join(nuc_detect_dir, "thumbnails")
