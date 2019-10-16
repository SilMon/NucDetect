BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "statistics" (
	"hash"	INTEGER,
	"image"	INTEGER,
	"area"	INTEGER,
	"intensity_average"	INTEGER,
	"intensity_median"	INTEGER,
	"intensity_maximum"	INTEGER,
	"intensity_minimum"	INTEGER,
	"intensity_std"	INTEGER,
	"ellipse_center"	INTEGER,
	"ellipse_major_axis_p0"	INTEGER,
	"ellipse_major_axis_p1"	INTEGER,
	"ellipse_major_axis_slope"	INTEGER,
	"ellipse_major_axis_length"	INTEGER,
	"ellipse_major_axis_angle"	INTEGER,
	"ellipse_minor_axis_p0"	INTEGER,
	"ellipse_minor_axis_p1"	INTEGER,
	"ellipse_minor_axis_length"	INTEGER,
	"ellipticity"	INTEGER,
	PRIMARY KEY("hash","image")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "groups" (
	"image"	INTEGER,
	"experiment"	INTEGER,
	"name"	INTEGER,
	PRIMARY KEY("image","experiment")
);
CREATE TABLE IF NOT EXISTS "roi" (
	"hash"	INTEGER,
	"image"	INTEGER,
	"auto"	INTEGER,
	"channel"	TEXT,
	"center"	TEXT,
	"width"	INTEGER,
	"height"	INTEGER,
	"associated"	INTEGER,
	PRIMARY KEY("hash","image")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "images" (
	"md5"	INTEGER,
	"datetime"	TEXT,
	"channels"	INTEGER NOT NULL,
	"width"	INTEGER NOT NULL,
	"height"	INTEGER NOT NULL,
	"x_res"	INTEGER,
	"y_res"	INTEGER,
	"unit"	INTEGER,
	"analysed"	INTEGER NOT NULL,
	"settings"	TEXT,
	PRIMARY KEY("md5")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "channels" (
	"md5"	INTEGER,
	"index"	INTEGER,
	"name"	INTEGER,
	PRIMARY KEY("md5","index")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "categories" (
	"image"	INTEGER,
	"category"	TEXT,
	PRIMARY KEY("image","category")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "experiments" (
	"image"	INTEGER,
	"name"	INTEGER,
	"details"	INTEGER,
	"notes"	INTEGER,
	PRIMARY KEY("image","name")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "settings" (
	"key_"	TEXT,
	"value"	TEXT,
	PRIMARY KEY("key_")
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS "points" (
	"hash"	INTEGER,
	"x"	INTEGER,
	"y"	INTEGER,
	"intensity"	INTEGER,
	PRIMARY KEY("hash","x","y")
) WITHOUT ROWID;
COMMIT;
