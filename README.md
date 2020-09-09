# Analyze Bias In Movies and Videos

## Overview
This python script conducts frame-by-frame analysis on video files in order to determine over and underrepresented groups.
Currently, it conducts analysis by gender only, but ethnicity and age statistics are next steps.

## How it Works
* The script uses opencv to split an mp4 file into individual frames, which are stored temporarily on the hard drive.
* The library _face_recognition_ is used to detect and extract faces in each frame. Face cutouts are also stored temporarily.
* Facebook's _Deepface_ library is used to conduct gender analysis on all of the face frames.
* Frame-by-frame data is aggregated and reduced to determine the overall gender representation.

## Requirements
A complete list of requirements can be found in [requirements.txt]("requirements.txt")

## Next Steps
1. Implement ethnicity-representation analysis
2. Aggregate more robust statistics, such as screentime per gender, number of frames where each gender was more prevalent, etc.
