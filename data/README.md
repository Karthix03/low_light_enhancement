# Data Directory

This folder contains input test datasets, sample night-time images, and prerecorded videos.

## Contents
* `sample_night.jpg`: Synthetic low-light sample image for testing enhancement and detection without a physical camera.
* Add your own `.mp4` or `.avi` night road driving footage here (e.g. `data/night_traffic.mp4`) to test offline video inference.

## How to use custom video files
1. Place your video file inside `data/`.
2. Update `config.yaml`:
   ```yaml
   camera:
     source_type: "video"
     video_path: "data/night_traffic.mp4"
   ```
3. Run `python main.py` or launch the Streamlit app `streamlit run app.py`.
