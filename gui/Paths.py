import copy
import os
import sys

gen = copy.copy(sys.path[1])
gui = copy.copy(sys.path[0])
nuc_detect_dir = os.path.join(os.path.expanduser("~"), "NucDetect")
script_dir = os.path.join(gui, "definitions", "ui")
img_dir = os.path.join(gui, "definitions", "images")
css_dir = os.path.join(gui, "definitions", "css")
sql_dir = os.path.join(gen, "core", "database", "scripts")

ui_main = os.path.join(script_dir, "nucdetect.ui")
ui_result_image_dialog = os.path.join(script_dir, "result_image_dialog.ui")
ui_exp_dial = os.path.join(script_dir, "experiment_dialog.ui")
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
