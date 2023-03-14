/*Delete the needed view if it already exists*/
DROP VIEW IF EXISTS roi_hashes;
/*Get all associated ROI for the given image*/
CREATE VIEW roi_hashes AS SELECT hash FROM roi WHERE image=<img_hash>;
/*Delete all saved rle lines*/
DELETE FROM points WHERE hash in roi_hashes;
/*Delete all saved statistics*/
DELETE  FROM statistics WHERE hash in roi_hashes;
/*Delete all saved ROI*/
DELETE FROM roi WHERE image=<img_hash>;
/*Delete the needed view if it already exists*/
DROP VIEW IF EXISTS roi_hashes;
