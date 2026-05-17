# Human Motion Analysis: Tracking Athletic Performance, Injury Prevention, and Movement Disabilities

#### *Preview:*
<div>
    <img width="400" alt="Screenshot 2026-05-16 at 4 23 29 PM" src="https://github.com/user-attachments/assets/527a0038-3499-46ac-95f6-ebe92b4ba4eb" />
    <img width="400" alt="bottom-dash-preview" src="https://github.com/user-attachments/assets/2769c20b-6331-4075-893a-4bb130f594e6" />
</div>

# Contents:
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Example Input --> Output](#example)
- [Outputted Metrics](#outputted-metrics)
    - [Dashboard Video](#dashboard)	
    - [Phase Overlay Videos](#phase-overlay-videos)
    - [Feature Deviation Tracking](#feature-deviation-tracking)
    - [Total Form Deviation](#total-form-deviation)	
    - [Phase Breakdown Isolation](#phase-breakdown-isolation)	
    - [Phase Deviation Tracking](#phase-deviation-tracking)
    - [Ground Contact Stats](#ground-contact-stats)	


# Overview:

#### Current Capabilities:
<ins>**This project aims to create an inexpensive and accessible system for analyzing human movement from video.**</ins>

Using pose estimation and machine learning, the system extracts biomechanical features from videos of users sprinting to evaluate performance, detect movement patterns, and provide feedback suggestions.

---

#### Future Implementations:
The long-term goal is to support applications in: 
- Other athletic actions *(baseball swing, barbell squat, basketball jumpshot, etc.)*
- Injury prevention feedback
- Rehabilitation
- Motor Planning
- Movement Disorder Assessment

---

#### Project Motivations:
As a track athlete, I wanted to create a tool that people can use to evalutate sprint performance without access to expensive lab equipment.

I got the idea to expand the capabilites of the tool to help enhance motor planning because of my younger brother with special needs.

---

# Pipeline:
| Step | Goal                  | Tool                                  | Use                                                                                                                                                                       |
|------|-----------------------|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Pose Estimation       | MMPose, Yolo26, Mediapipe             | Extract pose landmarks, joint angles, velocities using the user's choice of an engine for pose estimation between: Google's MediaPipe, Ultralytics YOLO26, or OpenMMLab's MMPose libraries, with custom normalization to enable standard comparison across datasets.                 |
| 2    | Phase Identification  | Keras 1D Convolutional Neural Network | Predict gait phases using a 1D Convolutional Neural Network that analyzes temporal patterns in user landmark sequences. <br> <br> ****Phases are separated into:**** <br> <br> - Left/Right Ground Contact (LGC/RGC) <br> <br> - Left/Right Propulsion (LP/RP) <br> <br> - Left/Right Flight (LF/RF)                                                  |
| 3    | Form Scoring          | Median Absolute Deviation (MAD)       | Compute deviation from reference motion patterns using Median Absolute Deviation (MAD) and calculate each feature's similarity scores over time using MAD-based Z-scores. |
| 4    | Output                | OpenCV, Matplotlib, Plotly            | Generate visualizations: <br> <br> ****OpenCV:**** <br> - Dashboard Video containing Form Deviation over time, color coded skeleton, and form correction suggestions <br> - Phase Overlays of Correct Form <br> <br> ****Matplotlib:**** <br> - Individual Z-scores over time <br> - Total Form Deviation Scoring over time <br> - Phase Z-scores over time <br> <br> ****Plotly:**** <br> - Individual Phase Breakdown Isolation |

# Example:

*Model Credit: [Zoe Johnson - Sprinter, MKA Class of 2028](https://www.athletic.net/athlete/23306526/track-and-field/all)*

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

# Outputted Metrics:

****Full Video Analysis:****
- Dashboard Video
- Feature Deviation Tracking
- Total Form Deviation Tracking


****Individual Phase Analysis:****
- Phase Overlay Videos
- Phase Deviation Tracking
- Phase Breakdown Tracking
- Phase Breakdown Isolation


****Ground Contact Statistics:****
- Ground Contact Times
- Left / Right Imbalance
- Strike Point Statistics
<br>
      
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
      <td id="dashboard">Dashboard Video</td>
      <td><video src="https://github.com/user-attachments/assets/8310d80e-8310-4bb8-abc9-473e36e34515" width="10" controls></video>
</td>
    </tr>
    <tr>
      <td id="phase-overlay-videos">Phase Overlay Videos</td>
      <td><img width="782.5" height="480" alt="overlay_example" src="https://github.com/user-attachments/assets/16837a4c-4709-4609-9e8e-590b804c681d" />
</td>
    </tr>
    <tr>
      <td id="feature-deviation-tracking">Feature Deviation Tracking</td>
      <td><img height="480" alt="LEFT ELBOW X" src="https://github.com/user-attachments/assets/1683339d-88a4-4612-8506-41c48f910201" />
</td>
    </tr>
    <tr>
      <td id="total-form-deviation">Total Form Deviation</td>
      <td><img height="480" alt="Total_Z-Score" src="https://github.com/user-attachments/assets/7e6fa8f9-4644-4843-bf23-64a24a53d04f" />
</td>
    </tr>
    <tr>
      <td id="phase-deviation-tracking">Phase Deviation Tracking</td>
      <td><img height="480" alt="Phase_Z-Scores" src="https://github.com/user-attachments/assets/983d928c-be88-454c-b962-ad4ca380882e" />
</td>
    </tr>
    <tr>
      <td id="phase-breakdown-isolation">Phase Breakdown Isolation</td>
      <td><img height="360" alt="phase_breakdown_comparison" src="https://github.com/user-attachments/assets/802505a5-e65c-4931-a068-4d560f22709c" />
</td>
    </tr>
    <tr>
      <td id="ground-contact-stats">Ground Contact Stats</td>
    <td><p>

**----------------STRIDE FREQUENCY----------------**

202.068 Steps per Minute

**--------------GROUND STRIKE POINTS--------------**

**Average Right Ground Strike Point:**

0.6188

<br>

**Average Left Ground Strike Point:**

0.5970

<br>

**Average Strike Point Imbalance: (negative = left | positive = right)**

0.0217

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


