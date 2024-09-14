## Environment Setup
step 1、Prepare conda environment
```bash
conda create --name DHD python=3.8.5
conda activate DHD
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3
pip install mmdet==2.25.1
pip install mmsegmentation==0.25.0

pip install lyft_dataset_sdk
pip install networkx==2.2
pip install numba==0.53.0
pip install numpy==1.23.4
pip install nuscenes-devkit
pip install plyfile
pip install scikit-image
pip install tensorboard
pip install trimesh==2.35.39
pip install setuptools==59.5.0
pip install yapf==0.40.1

cd Path_to_DHD
git clone https://github.com/yanzq95/DHD.git

cd Path_to_DHD/DHD
git clone https://github.com/open-mmlab/mmdetection3d.git

cd Path_to_DHD/DHD/mmdetection3d
git checkout v1.0.0rc4
pip install -v -e . 

cd Path_to_DHD/DHD/projects
pip install -v -e . 
```

step 2、Prepare data 

```python
└── data	
  └── nuscenes
      ├── v1.0-trainval 
      ├── sweeps  
      ├── samples
      └── gts
```

Download [nuScene dataset](https://www.nuscenes.org) 
Download 'gts' from [CVPR2023-3D-Occupancy-Prediction](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction)



