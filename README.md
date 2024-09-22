# <img src="assets/n-pose-logo.png" alt="drawing" width="100"/>  Bachelor Thesis - n-pose

This is the repository for my bachelor thesis. The goal of this thesis is
to recognize surgical tools in action. I proposed a new method (n-pose)
which tries to estimate an objects posed given a polygon describing the
object's segmentation and a list of keypoints.

For segmentation and keypoint detection, I used [YOLO](https://docs.ultralytics.com/).

## Requirements

- Python `>3.11`


- Alphashape `pip install alphashape`
- Matplotlib `pip install matplotlib`
- Numpy `pip install numpy`
- OpenCV `pip install opencv-python`
- Pandas `pip install pandas`
- Progressbar `pip install progressbar2`
- Pytorch `pip install torch`
- SKLearn `pip install scikit-learn`
- Scipy `pip install scipy`
- Shapely `pip install shapely`
- TQDM `pip install tqdm`
- Ultralytics `pip install ultralytics`

> 💡This is also listed in the `requirements.txt` file, thus it can be installed using `pip install -r requirements.txt`.

## Usage

For training and evaluation, the scripts in the `scripts/` folder can be used.

> For training of the YOLO models, all datasets that have Unity's instance segmentation, keypoint and bounding box
> annotation can be used.
>
> However, for the n-pose model, only the `solo_159` and `solo_161` datasets can be used, as they have
> the `KIARATransform` annotation that has the position and orientation data for pose estimation.

**Important:** To use the code in the `src` directory, the `src` directory must be added to the `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/n-pose/src
```

### Train YOLO

For training of the YOLO model's the scripts in `scripts/YOLO` can be used.
`train_yolo_seg.py` converts the data for segmentation and trains the `YOLOv8x-seg` model. `train_yolo_pose.py` converts
the data for keypoint detection and trains the `YOLOv8x-pose` model. `train_yolo.py` trains both models.

First, adapt the path constants and device in the scripts to match your setup. Afterwards just run the script:

```bash
python3.12 scripts/YOLO/train_yolo.py
```

### Train n-pose

> Use the `solo_159` for the evaluation dataset and the `solo_161` for the training dataset.
>
> For training use the csv files generated in step 1.

For training of the n-pose model, the scripts in `scripts/n-pose` can be used:

1. Convert the data for n-pose using `convert_data.py`. In this script you can adapt the PATHS to match your setup. Then
   run

```bash
python3.12 scripts/n-pose/convert_data.py
```

2. Train the n-pose model using `train_n_pose.py`. In this script you can adapt the PATHS to match your setup. Then run

```bash
python3.12 scripts/n-pose/train_n_pose.py
```

> The training of the n-pose model generates logs in the directory specified in the script.
> It is structured as follows:
>
> - `training_meta.csv`: Contains each combination that was trained along with its metrics.
> - `training.csv`: Contains the logs for each epoch of all trainings. These logs contain the id of the model in the
    meta fil.
> - `models/`: Contains the trained models to be used later. The models are saved as `model_{id}.pth`.

### Training on the a SLURM cluster.

First create a Python Virtual Environment in the `.venv/` folder and install the requirements. The virtual environments
are
automatically activated in the scripts.

For training on a SLURM cluster the `scripts/` directory also contains scripts to do this. For training on the cluster
run:

```bash
source scripts/batch_job.sh <PATH_TO_PROJECT_ROOT> <PATH_TO_CONFIG>
```

The configs are located in the `configs/` directory. Use the config for the training you want to run. The config files
are named according to the training script they are used for.