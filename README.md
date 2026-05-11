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

- Dashboard Video
- Phase Overlay Videos
- Feature Deviation Tracking
- Total Form Deviation Tracking
- Phase Deviation Tracking
- Phase Breakdown Tracking
- Phase Breakdown Isolation
- Ground Contact Statistics
    - Ground Contact Times
    - Left / Right Imbalance
    - Strike Point Statistics
      
### Example Outputs:  

<table>
  <thead>
    <tr>
      <th align="left">Metric</th>
      <th align="left">Example Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Dashboard Video</td>
      <td><video src="https://github.com/user-attachments/assets/8310d80e-8310-4bb8-abc9-473e36e34515" width="10" controls></video>
</td>
    </tr>
    <tr>
      <td>Phase Overlay Videos</td>
      <td><img width="782.5" height="480" alt="overlay_example" src="https://github.com/user-attachments/assets/16837a4c-4709-4609-9e8e-590b804c681d" />
</td>
    </tr>
    <tr>
      <td>Feature Deviation Tracking</td>
      <td><img height="480" alt="LEFT ELBOW X" src="https://github.com/user-attachments/assets/1683339d-88a4-4612-8506-41c48f910201" />
</td>
    </tr>
    <tr>
      <td>Total Form Deviation</td>
      <td><img height="480" alt="Total_Z-Score" src="https://github.com/user-attachments/assets/7e6fa8f9-4644-4843-bf23-64a24a53d04f" />
</td>
    </tr>
    <tr>
      <td>Phase Deviation Tracking</td>
      <td><img height="480" alt="Phase_Z-Scores" src="https://github.com/user-attachments/assets/983d928c-be88-454c-b962-ad4ca380882e" />
</td>
    </tr>
    <tr>
      <td>Phase Breakdown Isolation</td>
      <td><img height="360" alt="phase_breakdown_comparison" src="https://github.com/user-attachments/assets/802505a5-e65c-4931-a068-4d560f22709c" />
</td>
    </tr>
    <tr>
      <td>Ground Contact Stats</td>
    <td><p>

**----------------STRIDE FREQUENCY----------------**

202.068 Steps per Minute

**--------------GROUND STRIKE POINTS--------------**

**Average Right Ground Strike Point:**

0.6188108389691567

<br>

**Average Left Ground Strike Point:**

0.5970236599713672

<br>

**Average Strike Point Imbalance: (negative = left | positive = right)**

0.02178717899778948

<br>

**--------------GROUND CONTACT TIMES--------------**

**Average Ground Contact Time:**

Frames: 6

Seconds: 0.195

<br>

**Average Right Ground Contact Time:**

Frames: 6

Seconds: 0.193

<br>

**Average Left Ground Contact Time:**

Frames: 6

Seconds: 0.197

</p></td>
    </tr>
  </tbody>
</table>


