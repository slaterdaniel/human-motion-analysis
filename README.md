# Human Motion Analysis: Tracking Athletic Performance, Injury Prevention, and Movement Disabilities

## Overview:
This project aims to create an inexpensive and accessible system for analyzing human movement from video. Using pose estimation and machine learning, the system extracts biomechanical features from activities such as walking and running to evaluate performance, detect movement patterns, and identify potential injury risks.

The long-term goal is to support applications in athletic performance analysis, injury prevention, and movement disorder assessment.

## Pipeline:
| Step | Goal                  | Tool                                  | Use                                                                                                                                                                       |
|------|-----------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Pose Estimation       | MMPose, Yolo26, Mediapipe             | Extract pose landmarks, joint angles, velocities using Google's MediaPipe library with custom normalization to enable standard comparison across datasets                 |
| 2    | Phase Identification  | Keras 1D Convolutional Neural Network | Predict gait phases using a 1D Convolutional Neural Network that analyzes temporal patterns in user landmark sequences.                                                   |
| 3    | Form Scoring          | Median Absolute Deviation (MAD)       | Compute deviation from reference motion patterns using Median Absolute Deviation (MAD) and calculate each feature's similarity scores over time using MAD-based Z-scores. |
| 4    | Output                | OpenCV, Matplotlib, Plotly            | Generate visualizations using OpenCV and Matplotlib including pose skeleton overlays comparing user movement to reference motion and plots of feature Z-scores over time. |

## Example:

<table border="0">
  <tr>
    <td width="50%">
      <p align="center"><b>Input & Skeleton Overlay</b></p>
      <video src="https://github.com/user-attachments/assets/624b0c33-01cf-457a-b329-eabef0ab66b9" width="50%" controls></video>
    </td>
    <td width="50%">
      <p align="center"><b>Biomechanical Dashboard (Annotated) </b></p>
      <video src="https://github.com/user-attachments/assets/7e5ee360-6ffe-48e4-9887-f4fd103ee66f" width="50%" controls></video>
    </td>
  </tr>
</table>

***NOTICE:*** 

In dashboard video (pictured right), the large spike in *Total Form Deviation* at ~11sec aligns with form error of athletes left arm being raised 
This validates the ability of the pipeline to identify abnormalities in running form.

## Outputted Metrics:
| Metric                     | Example Output |
|----------------------------|----------------|
| Dashboard Video            |                |
| Phase Overlay Videos       |                |
| Ground Contact Stats       |                |
| Stride Frequency Stats     |                |
| Feature Deviation Tracking |                |
| Phase Stats                |                |
| Phase Breakdown Isolation  |                |
| Phase Deviation Tracking   |                |


