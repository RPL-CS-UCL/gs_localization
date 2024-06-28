# gs_localization

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
