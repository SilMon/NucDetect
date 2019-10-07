BEGIN TRANSACTION;
INSERT OR IGNORE INTO settings (key_, value) VALUES ("logging", 1);
INSERT OR IGNORE INTO settings (key_, value) VALUES ("res_path", "./results");
INSERT OR IGNORE INTO settings (key_, value) VALUES ("names", "Blue;Red;Green");
INSERT OR IGNORE INTO settings (key_, value) VALUES ("main_channel", 2);
COMMIT;