# Setup — assembling this repo (code only, no data)

This folder already has README, LICENSE, .gitignore and requirements.
You add the Python source — **never** the data, videos, frames or IDE history.

## 1. Copy ONLY the source code

From `D:\studium\studium\Masterarbeit\projectcode\` copy into this repo folder:

- **All `*.py` files in the top level** (e.g. `calibration.py`, `projection.py`,
  `transformation.py`, `Class_MP.py`, `functions_general.py`,
  `estimation_mediapipe_pose.py`, `plot_function_mediapipe_pose.py`,
  `DTW_analysis.py`, `DTW_analysis_3d.py`,
  `main_mediapipe_loc_lab_angle_estimation.py`, `main_02.py`, `main.py`,
  `M.py`, `rf.py`, `posee.py`, `test.py`, `randommmm.py`).
- The **`Abschlussarbeit\`** subfolder (its `*.py` only).

**Do NOT copy:**
- `.history\`  ← hundreds of editor auto-save versions (noise)
- `.idea\`     ← IDE config
- Any `*.csv`, `*.xlsx`, `*.mat`  ← sensor / mocap / mediapipe data (privacy)
- Any `*.mp4`, `*.avi`, `*.jpg`, `*.png`, `*.zip`  ← videos / frames / archives

> The `.gitignore` in this folder blocks all of the above as a safety net, so even
> if you copy extra files by accident, git will not stage data/videos/history.

## 2. Scrub privacy from the code (important)

Some scripts contain hard-coded absolute paths and participant names in
comments / example calls (e.g. video filenames with a person's name). Remove
them before committing.

Find them (PowerShell, run inside the repo folder):
```powershell
Select-String -Path *.py,Abschlussarbeit\*.py -Pattern "F:\\","C:\\","\.mp4","\.csv","sebastian" -CaseSensitive:$false
```
For each hit: delete the participant name and replace absolute paths with a
relative placeholder (e.g. `data/...`) or a config variable. Most are in
commented-out example blocks at the bottom of `main.py` — safe to delete.

## 3. Sanity check before pushing
```powershell
# from the repo folder — list exactly what git will commit:
git init
git add -A
git status            # confirm: only *.py + README/LICENSE/.gitignore/requirements
                      # NO .csv / .mp4 / .png / .history here
```
If you see any data/video/image file in `git status`, stop and remove it.

## 4. Commit & push
```bash
git commit -m "3D motion-quality analysis (computer vision + DTW)"
git branch -M main
git remote add origin https://github.com/Xinxcc/motion-quality-3d.git   # create the empty repo first
git push -u origin main
```

## Notes
- This is **your own M.Sc. thesis code**, so publishing it is fine; the only hard
  rule is **no participant data/videos**. Public or private is your choice —
  private + read-access-on-request is the safest if you'd rather not have it copied.
- If a script imports a DTW library, add it to `requirements.txt`.
