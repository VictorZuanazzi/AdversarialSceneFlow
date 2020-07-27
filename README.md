# Adversarial Scene Flow

Self-Supervised Scene Flow estimation via Adversarial Metric Learning. 

## Get started
To use this repository first clone this repository in your system. 

```
conda create --name aml python=3.6
conda activate aml
git clone https://github.com/VictorZuanazzi/AdversarialSceneFlow.git
cd AdversarialSceneFlow
```

## Downloading data:

Install [pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to use the ShapeNet pointclouds. 

Download [FlyingThings3D](https://drive.google.com/file/d/1CMaxdt-Tg1Wct8v8eGNwuT7qRSIyJPY-/view?usp=sharing), [KITTI without ground](https://drive.google.com/open?id=1XBsF35wKY0rmaL7x7grD_evvKCAccbKi) and [KITTI with ground](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)

Download and install [Lyft Level 5](https://pypi.org/project/lyft-dataset-sdk/)

```
pip install lyft-dataset-sdk
```

### Models for flow extraction:

We encorage you reading the documentations before installing the external libraries.


Clone and install [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) :

(From experience, Kaolin does not integrate greatly with different GPUs, execute the following commands using the GPU model you will use in your experiments).

```
git clone https://github.com/NVIDIAGameWorks/kaolin.git
cd kaolin
python setup.py install
cd ..
```

Clone and install Pytorch's implementation of [FlowNet3D](https://github.com/hyangwinter/flownet3d_pytorch) (requires GPU to run):

```
cd AdversarialSceneFlow
git clone https://github.com/hyangwinter/flownet3d_pytorch
cd flownet3d_pytorch/lib
python setup.py install
cd ../../
```

Clone and install [PointPWC-net](https://github.com/DylanWusee/PointPWC.git):

```
git clone https://github.com/DylanWusee/PointPWC.git
cd PointPWC/pointnet2
python setup.py install
cd ../../
```

## Training

```
python main.py --exp_name aml --train_type triplet --flow_extractor flownet3d --dataset shapenet
```

## Evaluate

```
python main.py --exp_name aml_test --train_type evaluate --flow_extractor flownet3d --dataset shapenet --load_model <path/to/model/best_model.pt>
```
