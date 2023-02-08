# OccupancySurface

## Install Libraries

For <strong>Anaconda</strong> environments:<br>

conda create -n OccS_env python=3.6<br>
conda activate OccS_env<br>
conda install -c conda-forge tensorflow-gpu=1.14<br>
conda install -c conda-forge opencv<br>
conda install -c conda-forge shapely<br>
conda install -c conda-forge matplotlib<br>
conda install -c conda-forge xlsxwriter<br>

## Parser Instructions

* Model: -m, --model, required=True  &rarr;  Path to object detection model (inference graph)<br>
* Labels: -l, --labels, required=True  &rarr;  Path to labels file (labelmap)<br>
* Input: -i, --input, required=True  &rarr;  Path to input image file<br>
* Output: -o, --output, default="results/output.jpg"  &rarr;  Path to optional output image file<br>
* Threshold: -t, --threshold, default=0.8  &rarr;  Minimum probability to filter detections<br>
* Calibration: -c, --calibration, action="store_true"  &rarr;  Option for un-distort input image<br>
* Resize: -r, --resize,  &rarr;  Resize input image in format "1,1" (comma separated)<br>
* Camera Height: -H, --camera_height  &rarr;  z-coordinate for camera positioning<br>
* People Height: -p, --people_height  &rarr;  z-coordinate for people height<br>
* Angle: -a, --angle  &rarr;  positioning angle in degrees<br>

### Step 0


