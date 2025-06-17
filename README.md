## LoGS: Visual Localization via Gaussian Splatting with Fewer Training Images  
Accepted by ICRA 2025.  
🔗 [Project Website](https://yuzhoucheng66.github.io/logs.github.io/)


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
# 2. Dataset Directory Structure

The following structure shows how the dataset folder is organized.  
Please ensure your local `datasets/` directory follows this layout:

```
gs-localization/
└── datasets/
    ├── 7scenes/
    │   ├── chess/
    │   ├── fire/
    │   ├── heads/
    │   ├── office/
    │   ├── pumpkin/
    │   ├── redkitchen/
    │   ├── stairs/
    │   ├── depth/
    │   ├── 7scenes_densevald_retrieval_top_10/
    │   ├── 7scenes_sfm_triangulated/
    │   └── train_fewshot_all
    ├── 7scenes_additional/
    ├── 360_v2/
    ├── cambridge/
    ├── cambridge_additional/
    └── nerf_llff_data/
```

# 3. LoGS Pipeline
Here is how LoGS re-localizes the scenes we experimented in our paper, e.g. 7-scenes (dslam ground truth and full training images). You need to run four python files one by one.

1. pre-process the 7-scenes dataset.
```
python gs_localization/process/train_test_split_7scenes_full_dslam.py
``` 

2. obtain a SfM point could of training images and rough initial poses of testing images through PnP-RANSAC.
```
python gs_localization/sfm/7scenes_sfm_full_dslam.py
``` 

3. train a 3DGS map of training images.
```
python gs_localization/gs/7scenes_gs_full_dslam.py
```

4. refines poses with gradient decent and we are done.
```
python gs_localization/pipelines/7scenes_localize_full_dslam.py
``` 

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
