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
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_sigma", 1.5, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_sigma", 3.5, "float");
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
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_foc_overlap", 0.5, "float");
-- Quality check settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_main_area", 750, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_area", 12, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_main_area", 30000, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("max_foc_area", 70, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_int", 0.055, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("min_foc_cont", 0.005, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("overlap", 0.5, "float");
-- General settings
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("size_factor", 1, "float");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("num_threads", 8, "int");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("show_ellipsis", 1, "bool");
INSERT OR IGNORE INTO settings (key_, value, type_) VALUES ("track_mouse", 1, "bool");
COMMIT;