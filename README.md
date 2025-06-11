## LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images
Accepted by ICRA2025

# 1. Installation
## Clone this repository.
```

git clone https://github.com/RPL-CS-UCL/gs_localization.git --recursive
```

## Install dependencies.
1. create an environment
```
conda create -n gsloc python=3.9
conda activate gsloc
```

2. install pytorch and other dependencies.

for cuda11.6
```
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt 
```
for cuda11.8
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

3. install submodules
```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

## LoGS Pipeline
Here is how LoGS re-localizes the scenes we experimented in our paper, e.g. 7-scenes (dslam ground truth and full training images). You need to run four python files one by one.

1. ```
python gs_localization/process/train_test_split_7scenes_full_dslam.py
``` 
This processes the 7-scenes dataset.

2. ```
python gs_localization/sfm/7scenes_sfm_full_dslam.py
``` 
This gives the SfM point could of training images and rough initial pose of testing images through PnP-RANSAC.

3. ```
python gs_localization/gs/7scenes_gs_full_dslam.py" 
```
This trains the 3DGS map of training images.

4. ```
python gs_localization/pipelines/7scenes_localize_full_dslam.py
``` 
This refines the poses with gradient decent and we are done.

### Acknowledgement

```bibtex
@inproceedings{cheng2025logs,
  title = {LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images},
  author = {Cheng, Yuzhou and Jiao, Jianhao and Wang, Yue and Kanoulas, Dimitrios},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  pages = {},
  year = {2025},
  organization = {IEEE},
  dimensions = {true},
}
```

```bibtex
This work was supported by the UKRI FLF [MR/V025333/1] (RoboHike).
For the purpose of Open Access, the author has applied a CC BY public copyright license to any Author Accepted Manuscript version arising from this submission.
Prof.Dimitrios Kanoulas is also with Archimedes/Athena RC, Greece.
