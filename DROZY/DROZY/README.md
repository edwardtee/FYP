DROZY: The ULg Multimodality Drowsiness Database
================================================


Description
-----------
The "ULg Multimodality Drowsiness Database", also called DROZY, is a database containing various types of drowsiness-related data (signal, images, etc.) and intended to help researchers to carry out experiments, and to develop and evaluate systems (i.e. algorithms), in the area of drowsiness monitoring. 

The (multimodality) data were collected by the Laboratory for Signal and Image Exploitation (INTELSIG), which is part of the Department of Electrical Engineering and Computer Science of the University of Liège (ULg), Liège, Belgium.


\*\*UPDATE\*\*
------
As of July 2018, the DROZY database is available for download on the Open Repository and Bibliography (ORBi) of the ULg, instead of on a dedicated server.  For storage reasons, we excluded from DROZY the raw near-infrared intensity and range images. However, the encoded videos of the intensity images are still included. Furthermore, given that some frames are missing, we added the interpIndices folder (see below).

Contact
-------
Please visit our website http://www.drozy.ulg.ac.be to get the most updated contact information.


Missing tests
-------------
Because of backup incidents, some of the data are lost (tests 9-1, 10-2, 12-2, 12-3, 13-3). Test 7-1 did not happen.


Content overview
----------------

- `kinect-intrinsics.yaml` (file) : intrinsics parameters of the Microsoft Kinect v2 sensor
- `KSS.txt` (file) : "Karolinska Sleepiness Scale" scores
- `annotations-manual` (folder) : manual subject-level 68-landmarks annotations
- `annotations-auto` (folder) : automatic frame-level 68-landmarks annotations
- ~~ `images_nir16` (folder) : near-infrared intensity images~~
- ~~ `images_depth16` (folder) : range images (depth maps)~~
- `videos_i8` (folder) : encoded videos
- `timestamps` (folder) : timestamps
- `psg` (folder) : polysomnographic (PSG) signals
- `pvt-rt` (folder) : reaction times of the Psychomotor Vigilance Tests (PVTs)
- **!NEW! ** `interpIndices` (folder) : interpolation indices for missing frames

More formatting details follow below. Check our website, read our paper(s), or contact us to get other informations.


Intrinsics parameters
---------------------
The `kinect-intrinsics.yaml` file contains the intrinsics matrix (`intrinsics` node) and the distorsion parameters (`k` and `p` nodes). This file can be parsed easily with the cv.FileStorage class of OpenCV.


KSS
---
The file `KSS.txt` has values ranging from 1 to 9 (i.e. KSS score range) inside 14 lines (1 per subject) and 3 columns (1 per test). Test 7-1 has been arbitrarily given a KSS score of 0, since there is no recorded value. 


Facial shape annotations
------------------------
* 2D-shape format = [x1, y1, x2, y2, ... x68, y68] coordinates (length=136) in pixels in the image axes; top-left corner is (x,y) = (0,0); x is column, y is row
* 3D-shape format = [X1, Y1, Z1, X2, Y2, Z2, ... X68, Y68, Z68] coordinates (length=204) in millimeters in the camera axes; center of the camera/IR sensor is (X,Y,Z) = (0,0,0); X grows to the right, Y grows downward, Z grows forward

### Manual (subject-level)
The folder `annotations-manual` contains the folders `SUBJECT`. The folders `SUBJECT` contains the files `annot-timestamps.txt`, `annot-s2.txt`, and `annot-s3.txt`and the folders `depth`, and `nir`.

- `annot-timestamps.txt` (file) : 1 line = 1 manual annotation = 1 parsed filename = [TEST, year, month, day, hours, minutes, seconds, milliseconds]
- `annot-s2.txt` (file) : 1 line = 1 manual annotation = 1 2D-shape vector
- `annot-s3.txt` (file) : 1 line = 1 manual annotation = 1 3D-shape vector
- `depth` (folder) : contains all the annotated range images, in .png format, 16 bits
- `nir` (folder) : contains all the annotated intensity images, in .png format, 16 bits

### Automatic (frame-level)
The folder `annotations-auto` contains the files `SUBJECT-TEST-s2.txt` and `SUBJECT-TEST-s3.txt`, containing respectively the 2D and 3D automatic annotations of the 68 face landmarks for all frames.

- `SUBJECT-TEST-s2.txt` : 1 line = 1 frame = 1 2D-shape vector
- `SUBJECT-TEST-s3.txt` : 1 line = 1 frame = 1 3D-shape vector


Face images
-----------
The intensity and range images have both a resolution of 512x424 and 1 channel. Data are stored as 16-bit unsigned integers in .png image format. Intensity and range images are perfectly aligned since they are derived from the same sensor. The framerate is at 30 fps. Beware, some tests (tests 2 and 3 of subjects 1->8) are at 15 fps because of a recording bug occurring in darkness.

### ~~Near-infrared intensity images~~
~~All images of a same test are grouped in a tarball (named `SUBJECT-TEST.tar`) in the `images_nir16` folder. ~~

### ~~Range images~~
~~All images of a same test are grouped in a tarball (named `SUBJECT-TEST.tar`) in the `images_depth16` folder. Each value represents the distance in millimeters.~~

### Encoded videos
The folder `videos_i8` contains the encoded (intensity) videos `SUBJECT-TEST.mp4` (1 video per test). Data are stored as 8-bit unsigned integers.


Timestamps
----------
The folder `timestamps` contains the files `SUBJECT-TEST.txt` (1 file per test). The file `SUBJECT-TEST.txt` contains, for each frame (1 line per frame), the parsed timestamps and the elapsed time since the beginning of the test (in milliseconds). Each line has the following format : [year, month, day, hours, minutes, seconds, milliseconds, elapsedtime]. 


PSG
---
The folder `psg` contains the files `SUBJECT-TEST.edf` (EDF = European Data Format). EDF parsers can be found online. Data are recorded at a framerate of 512 Hz. Here's a list of all channels :

- `Fz` : midline frontal electrode (EEG)
- `Cz` : midline central electrode (EEG)
- `C3` : left hemisphere central electrode (EEG)
- `C4` : right hemisphere central electrode (EEG)
- `Pz` : midline parietal electrode (EEG)
- `Cam-Sync` : square signal that toggle its value whenever an image is recorded
- `PVT` : square signal that switch to +3V when a visual stimulus appears, and switch to -3V when the subject reacts with a mouse click
- `EOG-V` : vertical EOG
- `EOG-H` : horizontal EOG
- `EMG` : EMG
- `ECG` : ECG

EEG channels are all referenced to A1 in the International 10-20 system. For unknown reasons, tests 1-1 and 2-1 have a 12th recorded channel named `Oz` (midline occipital electrode).


PVT-RT
------
The folder `pvt-rt` contains the files `SUBJECT-TEST.csv`. The 1st line (header) of the file `SUBJECT-TEST.csv` is the timestamp of the beginnning of the test (yyyy-mm-dd_hh.mm.ss.mmm). The remaining lines (1 line per stimulus) contain 2 timestamps (separated by ';') : 1) the time of occurence of the stimulus, and 2) the time when the subject reacted (mouse click). The difference between these 2 timestamps is the reaction time (RT).


Interpolation indices (!NEW!)
--------------------
The folder `interpIndices` contains the files `SUBJECT-TEST.txt`. Each file contains 18,000 lines, corresponding to the 18,000 frames a PVT would have if there was no missing frames (30 FPS during 10 minutes). Each line has 3 columns, so as to be able to interpolate any desired metric at the associated, PVT-synchronized time. The first and second columns contain indices of frames (referenced in the encoded video) to use for the interpolation. The third column contains an interpolation coefficient that weights the video frame indexed by the first column. The complement of this interpolation coefficient weights the video frame indexed by the second column. Examples follow.

- If the PVT-frame at line *k* already exists (e.g. line *k* = '35 0 1.000000'), you can use the video frame indexed by the first column (e.g. frame 35 in the encoded video corresponds to frame *k* in the PVT).
- If the PVT-frame at line *k* does not exist (e.g. line *k* = '201 202 0.666667'), you need to interpolate between the video frame indexed by the first column (e.g. 201) with the coefficient at the third column (e.g. 0.666667), and the video frame index by the second columne (e.g. 202) with the complement of the coefficient at the third column (e.g. 1-0.666667=0.333333).

