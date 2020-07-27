# Adversarial Scene Flow

Self-Supervised Scene Flow estimation via Adversarial Metric Learning. 

## Get started
To use this repository first clone this repository in your system. 

```git clone https://github.com/VictorZuanazzi/AdversarialSceneFlow.git```

## Downloading data:

Install `pytorch_geometric` to use the ShapeNet pointclouds. Follow the steps from https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html .

Download FlyingThings3D from: https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing (thanks to https://github.com/xingyul/flownet3d)

Download KITTI withiout ground from: https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi 

Download KITTI with grodun from: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php

Download and isntall Lyft from: https://self-driving.lyft.com/level5/

### Models for flow extraction:

Install Kaolin, follow the instructions from: https://github.com/NVIDIAGameWorks/kaolin

Clone and install Pytorch's implementation of FlowNet3D:

```
cd AdversarialSceneFlow
git clone https://github.com/hyangwinter/flownet3d_pytorch
cd flownet3d_pytorch/lib
python setup.py install
cd ../../
```

Clone and install PointPWC-net:

```
git clone https://github.com/DylanWusee/PointPWC.git
cd PointPWC/pointnet2
python setup.py install
cd ../../
```

## Training

```
python main.py --exp_name aml --train_type triplet --flow_extractor flownet3d --dataset lyft --dataset_eval kitting
```

## Evaluate

```
python main.py --exp_name aml_test --train_type evaluate --flow_extractor flownet3d --dataset kitting --load_model <path/to/model>
```
