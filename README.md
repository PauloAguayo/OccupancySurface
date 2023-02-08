# OccupancySurface

## Install Libraries

For Anaconda environments:<br>

conda create -n OccS_env python=3.6<br>
conda activate OccS_env<br>
conda install -c conda-forge tensorflow-gpu=1.14<br>
conda install -c conda-forge opencv<br>
conda install -c conda-forge shapely<br>
conda install -c conda-forge matplotlib<br>
conda install -c conda-forge xlsxwriter<br>

## Parser Instructions

* Model: -m, --model, required=True ==> path to object detection model (inference graph)<br>
"-l", "--labels", required=True, help="path to labels file")
"-i", "--input", default=0, type=str, help="path to optional input image file", required=True)
"-o", "--output", type=str, default="results/output.jpg", help="path and name to optional output image file")
"-t", "--threshold", type=float, default=0.8, help="minimum probability to filter weak detection")
"-c", "--calibration", action="store_true", help="option for un-distort input image")
"-r", "--resize", type=str, default="1,1", help="resize input image")
"-H", "--camera_height", type=float, default=2.5, help="z-coordinate for camera positioning")
"-p", "--people_height", type=float, default=1.7, help="z-coordinate for people high")
"-a", "--angle", type=float, default=14, help="positioning angle in degrees")

### Step 0


