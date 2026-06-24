# 3D Motion-Quality Analysis

A computer-vision pipeline that **quantifies the quality of human 3D motion from
cameras** and compares it against a reference, using pose estimation, multi-view
geometry, sensor fusion and dynamic time warping.

> Independent M.Sc. project. **No data, videos or subject information are included
> in this repository** — only the source code.

## Pipeline

```
 video(s) ──► MediaPipe pose ──► camera calibration / undistortion
                                          │
                                          ▼
                         multi-view 3D reconstruction
                    (epipolar geometry · triangulation · solvePnP)
                                          │
                                          ▼
          fuse with a wearable motion sensor  ── cross-correlation
          (time-synchronization + resampling)      alignment
                                          │
                                          ▼
                 multi-dimensional Dynamic Time Warping (mDTW)
                                          │
                                          ▼
        Random Forest / Decision Tree / Logistic Regression
              (motion-quality classification, cross-validated)
```

## What it does

1. **Pose estimation** — extracts 3D body landmarks with **MediaPipe**.
2. **Camera calibration & undistortion** — OpenCV checkerboard calibration,
   distortion removal.
3. **Multi-view 3D reconstruction** — fundamental/essential matrix (RANSAC),
   `recoverPose`, triangulation and `solvePnP` to lift 2D keypoints into 3D.
4. **Sensor fusion** — aligns the camera-derived motion with a wearable motion
   sensor by **cross-correlation** time-synchronization and resampling.
5. **Feature extraction** — joint angles, trajectory metrics and
   **multi-dimensional DTW** distances between sequences.
6. **Classification** — Random Forest / Decision Tree / Logistic Regression with
   grid-search cross-validation and confusion-matrix evaluation.

## Modules (overview)

| File | Role |
|---|---|
| `estimation_mediapipe_pose.py`, `plot_function_mediapipe_pose.py` | pose estimation + visualization |
| `calibration.py` | camera calibration / undistortion |
| `projection.py`, `transformation.py` | multi-view geometry, 2D↔3D projection |
| `functions_general.py`, `Class_MP.py` | shared helpers / data structures |
| `DTW_analysis.py`, `DTW_analysis_3d.py` | (multi-dimensional) dynamic time warping |
| `main_mediapipe_loc_lab_angle_estimation.py`, `main_02.py` | pipeline entry points |
| `Abschlussarbeit/` (`M.py`, `rf.py`, `main.py`) | feature aggregation + classification |

## Tech stack

Python · OpenCV · MediaPipe · NumPy · SciPy · pandas · scikit-learn · Matplotlib

## Data & privacy

This repository contains **no recordings, sensor data, or any participant
information**. The scripts expect you to point them at your own local data
directory. Paths are configurable at the top of the entry-point scripts.

## License

See [`LICENSE`](./LICENSE) — shared for evaluation purposes only.
