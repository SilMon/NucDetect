CREATE TABLE "categories" (
	"image"	TEXT,
	"category"	TEXT,
	PRIMARY KEY("image","category")
) WITHOUT ROWID;

CREATE TABLE "channels" (
	"md5"	TEXT,
	"index"	INTEGER,
	"name"	INTEGER
);

CREATE TABLE "images" (
	"md5"	TEXT,
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

CREATE TABLE "points" (
	"hash"	INTEGER,
	"x"	INTEGER,
	"y"	INTEGER,
	"intensity"	INTEGER,
	PRIMARY KEY("hash","x","y")
) WITHOUT ROWID;

CREATE TABLE "roi" (
	"hash"	INTEGER,
	"image"	TEXT,
	"auto"	INTEGER,
	"channel"	TEXT,
	"center"	TEXT,
	"width"	INTEGER,
	"height"	INTEGER,
	"associated"	INTEGER,
	PRIMARY KEY("hash")
) WITHOUT ROWID;

CREATE TABLE "roi_categories" (
	"md5"	TEXT,
	"category"	TEXT,
	PRIMARY KEY("md5")
) WITHOUT ROWID;

CREATE TABLE "settings" (
	"key_"	TEXT,
	"value"	TEXT,
	PRIMARY KEY("key_")
) WITHOUT ROWID;