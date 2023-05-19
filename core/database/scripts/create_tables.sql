BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "encountered_names"
(
    "md5"       TEXT,
    "file_name" TEXT,
    PRIMARY KEY ("md5")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "channels"
(
    "md5"    TEXT,
    "index_" INTEGER,
    "name"   INTEGER,
    "active" INTEGER,
    "main"   INTEGER,
    PRIMARY KEY ("md5", "index_")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "experiments"
(
    "name"    TEXT,
    "details" TEXT,
    "notes"   TEXT,
    PRIMARY KEY ("name")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "groups"
(
    "image"      INTEGER,
    "experiment" INTEGER,
    "name"       TEXT,
    PRIMARY KEY ("image", "experiment")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "images"
(
    "md5"        TEXT,
    "year"       INTEGER,
    "month"      INTEGER,
    "day"        INTEGER,
    "hour"       INTEGER,
    "minute"     INTEGER,
    "channels"   INTEGER NOT NULL,
    "width"      INTEGER NOT NULL,
    "height"     INTEGER NOT NULL,
    "x_res"      INTEGER,
    "y_res"      INTEGER,
    "unit"       INTEGER,
    "analysed"   INTEGER NOT NULL,
    "settings"   TEXT,
    "experiment" TEXT,
    "modified"   INTEGER NOT NULL,
    PRIMARY KEY ("md5")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "points"
(
    "hash"    INTEGER,
    "row"     INTEGER,
    "column_" INTEGER,
    "width"   INTEGER,
    PRIMARY KEY ("hash", "row", "column_")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "roi"
(
    "hash"             INTEGER,
    "image"            INTEGER,
    "auto"             INTEGER,
    "channel"          TEXT,
    "center_x"         INTEGER,
    "center_y"         INTEGER,
    "width"            INTEGER,
    "height"           INTEGER,
    "associated"       INTEGER,
    "detection_method" TEXT,
    "match"            INTEGER,
    PRIMARY KEY ("hash", "image")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "settings"
(
    "key_"  TEXT,
    "value" TEXT,
    "type_" TEXT,
    PRIMARY KEY ("key_")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "statistics"
(
    "hash"                 INTEGER,
    "image"                INTEGER,
    "area"                 INTEGER,
    "intensity_average"    INTEGER,
    "intensity_median"     INTEGER,
    "intensity_maximum"    INTEGER,
    "intensity_minimum"    INTEGER,
    "intensity_std"        INTEGER,
    "eccentricity"         INTEGER,
    "roundness"            INTEGER,
    "ellipse_center_x"     INTEGER,
    "ellipse_center_y"     INTEGER,
    "ellipse_major"        INTEGER,
    "ellipse_minor"        INTEGER,
    "ellipse_angle"        INTEGER,
    "ellipse_area"         INTEGER,
    "orientation_vector_x" INTEGER,
    "orientation_vector_y" INTEGER,
    "ellipticity"          INTEGER,
    PRIMARY KEY ("hash", "image")
) WITHOUT ROWID;
COMMIT;
