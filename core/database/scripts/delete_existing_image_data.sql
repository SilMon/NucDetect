/*Delete the needed view if it already exists*/
DROP VIEW IF EXISTS roi_hashes;
/*
Hashes of this image's roi that NO OTHER image also holds.

hash(roi) is md5(channel name + area) and carries no image, so two images containing an identical
small focus in the same channel get the SAME hash -- 4388 hashes in the live database are shared by
more than one image. The roi and statistics tables key on (hash, image) and can hold both; the
points table keys on (hash, row, column_) and has NO image column, so the two roi share ONE set of
points.

Deleting points by hash alone therefore removed the area of another image's roi and left its roi row
standing with nothing under it -- 57 such rows exist, and the manual editor raised
"ROI ... does not contain any points!" on 33 images because of them.
*/
CREATE VIEW roi_hashes AS
    SELECT hash FROM roi
    WHERE image=<img_hash>
      AND hash NOT IN (SELECT hash FROM roi WHERE image<><img_hash>);
/*Delete all saved rle lines, except those another image's roi also depends on*/
DELETE FROM points WHERE hash in roi_hashes;
/*
Delete all saved statistics. BY IMAGE, not by hash: this table has an image column, so the row
belonging to another image with the same hash must survive.
*/
DELETE FROM statistics WHERE image=<img_hash>;
/*Delete all saved ROI*/
DELETE FROM roi WHERE image=<img_hash>;
/*Delete the needed view if it already exists*/
DROP VIEW IF EXISTS roi_hashes;
