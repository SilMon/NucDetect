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
    "co_localized"     INTEGER,
    PRIMARY KEY ("hash", "image")
) WITHOUT ROWID;
/*
Co-localization, per CHANNEL PAIR -- schema version 3, 2026-09-24. RW: "The co-localization should
be stored for each defined channel pair. This requires to move it out of the main table."

roi.match and roi.co_localized above are the version-2 form: ONE percentage per nucleus and ONE
partner per focus, which cannot express more than one pair. They stay, because removing a column
needs a table rebuild and older builds still read them, but this build writes NULL to both and
reads them only for an image analysed before these tables existed.

colocalization_pairs records what an analysis was CONFIGURED to compare, including a pair that
found no foci at all -- that pair's answer is "nothing to compare", which an absent row cannot
distinguish from "never computed". max_distance is in PIXELS as applied to this image, so a
recomputation after an edit reproduces the analysis exactly.

colocalization holds one row per focus per pair it takes part in; partner is NULL for a focus with
none. The per-nucleus percentage is derived from these rows, never stored, so it cannot disagree
with them. image is TEXT, as the md5 it holds is -- unlike roi.image, which is declared INTEGER.
*/
CREATE TABLE IF NOT EXISTS "colocalization_pairs"
(
    "image"        TEXT,
    "channel_a"    TEXT,
    "channel_b"    TEXT,
    "max_distance" REAL,
    PRIMARY KEY ("image", "channel_a", "channel_b")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "colocalization"
(
    "image"     TEXT,
    "focus"     INTEGER,
    "channel_a" TEXT,
    "channel_b" TEXT,
    "partner"   INTEGER,
    PRIMARY KEY ("image", "focus", "channel_a", "channel_b")
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
-- Create search-relevant indices to speed up the search
CREATE INDEX IF NOT EXISTS associated_idx ON roi(associated);
CREATE INDEX IF NOT EXISTS channel_idx ON roi(channel);
CREATE INDEX IF NOT EXISTS detection_idx ON roi(detection_method);
CREATE INDEX  IF NOT EXISTS image_idx ON roi(image);
CREATE INDEX IF NOT EXISTS images_md5_idx ON images(md5);
CREATE INDEX IF NOT EXISTS points_hash_idx ON points(hash);
CREATE INDEX IF NOT EXISTS roi_hash_idx ON roi(hash);
CREATE INDEX IF NOT EXISTS statistics_hash_idx ON statistics(hash);
COMMIT;
