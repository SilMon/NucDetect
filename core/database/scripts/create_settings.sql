BEGIN TRANSACTION;
-- Analysis settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("logging", 1, "bool");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("exp_std_name", "Default", "str");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("names", "Red;Green;Blue", "str");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("main_channel", 2, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("logging", 1, "bool");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("ml_analysis", 0, "bool");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("quality_check", 1, "bool");
-- Pre-Processing Settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("filter_radius", 3, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("gaussian_sigma", 1.5, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("denoising_weight", 0.15, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("sigma_color", 0.1, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("sigma_spatial", 15, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("bckg_subtr_order", 2, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("bckg_subtr_diameter", 3, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("bckg_subtr_feature_min", 1.3668, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("bckg_subtr_feature_max", 9.5676, "float");
-- Image Processing Settings
/*
min_sigma and max_sigma are in MICROMETRES since 2026-09-14 and are multiplied by
dots_per_micron before they reach the blob detector. They were pixel values; 1.5 and 3.5 px
at the 6.412 px/um default are the 0.2339 and 0.5459 um seeded here, so a fresh install detects
exactly what it detected before on a 40x image.
*/
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_sigma", 0.2339, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_sigma", 0.5459, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("num_sigma", 10, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("acc_thresh", 0.02, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("iterations", 10, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("mask_size", 7, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("percent_hmax", 0.05, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("local_threshold_multiplier", 8, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("maximum_size_multiplier", 2, "int");
-- Machine Learning Settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("fcn_certainty_nuclei", 0.95, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("fcn_certainty_foci", 0.25, "float");
-- Matching Settings
/*
Both in MICROMETRES since 2026-09-24, and converted with the image's own conversion factor like
min_sigma above. They were hard-coded PIXEL defaults in MapComparator -- 9 for co-localization, 5
for the combined method's merge -- so whether two foci co-localized depended on the objective the
image was taken with. The seeds are those pixel values at the 6.412 px/um default, so a 40x image
is compared exactly as before.
*/
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("colocalization_distance", 1.4036, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("merge_distance", 0.7798, "float");
-- Quality check settings
/*
The four size bounds below are in SQUARE MICROMETRES, which is what the settings dialog has always
said. Until 2026-09-14 they were compared against an area in PIXELS, so the unit had no effect.

The two NUCLEUS seeds were already written as um^2 and are unchanged -- 115 um^2 is a nucleus about
12.1 um across, which is right, while as a pixel count it would be 1.9 um, which is not. The two
FOCUS seeds were pixel counts wearing the same label and ARE converted: 8 px^2 -> 0.195 um^2 (a
focus 0.5 um across) and 70 px^2 -> 1.703 um^2 (1.5 um across). As um^2 the old numbers would have
described foci 3.2 and 9.4 um across, which no focus is.

So this file mixed two units between its nucleus and focus rows. Check the unit of any bound added
here against a physical size before trusting the number.
*/
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_main_area", 115, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_main_area", 4650, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_area", 0.195, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_foc_area", 1.703, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_int", 0.055, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_cont", 0.005, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("overlap", 0.5, "float");
-- General settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("size_factor", 1, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("num_threads", 8, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("track_mouse", 1, "bool");
COMMIT;