# Environment
Python 3.8, pytorch 1.11.0 and cuda 11.4 are used for this project.
```
conda create -n strefer python=3.8
conda activate strefer
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch
```

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
```
