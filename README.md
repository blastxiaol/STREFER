# Environment
Python 3.8, pytorch 1.11.0 and cuda 11.4 are used for this project.
```
conda create -n strefer python=3.8
conda activate strefer
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
```

# Download Data
Dataset will be released later.

# Install packages
Install some addition packages:
```
pip install -r requirement.txt
```
Install Pointnet++:
```
cd external_tools/pointnet2
python setup.py install
cd ../..
```

# Install Detector
```
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.24.0
pip install mmsegmentation==0.26.0
cd detection3d
pip install -v -e .
cd ..
```

# Training
## Detection
We use pretrained CenterPoint from STCrowd
```
cd detection3d
python tools/test.py configs/_b/center_point_vel.py configs/_b/multi_frame_dim4.pth --eval mAP
``` 
## Grounding
```
python tools/train.py --use_view --frame_num 2
``` 

# Test
```
python tools/test.py --use_view --frame_num 2
``` 