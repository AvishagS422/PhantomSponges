# Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors

This is a PyTorch implementation of [Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors](https://arxiv.org/pdf/2111.10759.pdf--change!!!!!) by Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai.

<p align="center">
<img src="https://github.com/AvishagS422/PhantomSponges/blob/main/images/intro.png" width=70% height=70% />
</p>

![projection pipeline](https://github.com/AvishagS422/PhantomSponges/blob/main/images/pipleline.png?raw=true)

## Datasets

### Berkeley DeepDrive (BDD)
The dataset can be found [here](https://doc.bdd100k.com/download.html#k-images).

## Installation
####Installing packages
Install the required packages in [req.txt](https://github.com/AvishagS422/PhantomSponges/tree/master/req.txt).

####Installing weights files
To attack YOLOv4, install the weights file and locate it in ["local_yolos/yolov4/weights"](https://github.com/AvishagS422/PhantomSponges/blob/master/local_yolos/yolov4/weights) folder.
The Weights file (yolov4.weights) for YOLOv4 can be found [here](https://github.com/WongKinYiu/PyTorch_YOLOv4).

## Usage

### Configuration
Set the configuration's values in the first cell in the [run_attack.ipynb](https://github.com/AvishagS422/PhantomSponges/blob/master/run_attack.ipynb) notebook.

### Train

Follow the instruction in the [run_attack.ipynb](https://github.com/AvishagS422/PhantomSponges/blob/master/run_attack.ipynb) notebook.

## Citation
```
@article{shapira2022denial,
  title={Denial-of-Service Attack on Object Detection Model Using Universal Adversarial Perturbation},
  author={Shapira, Avishag and Zolfi, Alon and Demetrio, Luca and Biggio, Battista and Shabtai, Asaf},
  journal={arXiv preprint arXiv:2205.13618},
  year={2022}
}
```
