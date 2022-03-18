# Mar 12-13

## RL_Snake
If we only allow the vertical joint to move, we can have a great result! --- see M2_run_ppo.py in RL_Snake_v2


## MPC
### Prepare the conda env following [instruction](https://github.com/homangab/gradcem#requirements)
```Shell
conda create -n MPC python=3.9 -y
```

#### Install [DeepMind Control Suite] & [Mujoco]

1. Download MuJoCo 2.1.1 from [the Releases page on the MuJoCo GitHub repository](https://github.com/deepmind/mujoco/releases). On macOS, either place ```MuJoCo.app``` into ```/Applications```, or place ```MuJoCo.Framework``` into ```~/.mujoco```.
Then, run ```pip3 install -U 'mujoco-py<2.2,>=2.1'``` to install the mujoco-py for using it in python env. 

2. Install the ```dm_control``` Python package by running ```pip3 install dm_control```. -- Works for me.

#### Install gym
```Shell
pip3 install gym
```

#### Install OpenCV
```Shell
pip3 install opencv-python
```

#### Install Pytorch
```Shell
pip3 install torch torchvision
```

#### Other requirements
```Shell
pip3 install matplotlib
```
### Start to run the MPC experiment

