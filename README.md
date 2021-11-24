# 3DG-STFM: 3D Geometric Guided Student-Teacher Feature Matching

## Requirements
Enviroment setup:
```shell
conda env create -f environment.yaml
conda activate stfm
pip install torch einops yacs kornia
```
## Get Started
We provide training and test code for both indoor and outdoor datasets.
The code scripts include training scripts for mono-modal baseline, multi-modal teacher model, and student-teacher learning model.
The inference code is also provided.
We also provide a simple demo code for performance visualization.
### Training
Both indoor and outdoor training code are released.
To reproduce the result, the multi-modal teacher model need to be trained first and used for student-teacher learning.
```shell
conda activate stfm

bash ./scripts/train/indoor_ds_rgbd.sh ##teacher model training
## After teacher model training
bash ./scripts/train/indoor_ds_rgbd_t_s.sh ##student-teacher learning
```
### Test
```shell
conda activate stfm

bash ./scripts/test/indoor_ds.sh ##teacher model training
## if you want to see the multi-modal model's performance 
bash ./scripts/test/indoor_ds_rgbd.sh 
```
### Demo
We also provide demo script for visualization.
We provide the [download link](https://drive.google.com/file/d/1S5uscMp-3el4AUALQJznG1lyN-_O4Ks7/view?usp=sharing)
The link include both indoor and outdoor model weights.
Please put the ckpt files in the folder weights.
```shell
conda activate stfm
# python script
python demo.py ##this sample code use indoor student model for correspondence prediction
```

