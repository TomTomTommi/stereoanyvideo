# [ICCV2025 Highlight] Stereo Any Video: Temporally Consistent Stereo Matching

[`Paper`](https://arxiv.org/abs/2503.05549) [[`Project`](https://tomtomtommi.github.io/StereoAnyVideo/)]

![Demo](./assets/stereoanyvideo.gif)


## Installation

Installation with cuda 12.2

<details>
  <summary>Setup the root for all source files</summary>
  <pre><code>
    git clone https://github.com/tomtomtommi/stereoanyvideo
    cd stereoanyvideo
    export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
  </code></pre>
</details>

<details>
  <summary>Create a conda env</summary>
  <pre><code>
    conda create -n sav python=3.10
    conda activate sav
  </code></pre>
</details>

<details>
  <summary>Install requirements</summary>
  <pre><code>
    conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install pip==24.0
    pip install pytorch_lightning==1.6.0
    pip install iopath
    conda install -c bottler nvidiacub
    pip install scikit-image matplotlib imageio plotly opencv-python
    conda install -c fvcore -c conda-forge fvcore
    pip install black usort flake8 flake8-bugbear flake8-comprehensions
    conda install pytorch3d -c pytorch3d
    pip install -r requirements.txt
    pip install timm
  </code></pre>
</details>

<details>
  <summary>Download VDA checkpoints</summary>
  <pre><code>
    cd models/Video-Depth-Anything
    sh get_weights.sh
  </code></pre>
</details>

## Inference a stereo video

```
sh demo.sh
```
Before running, download the checkpoints on [google drive](https://drive.google.com/drive/folders/1c7L065dcBWhCYYjWYo2edGOG605PnpXv?usp=sharing) . 
Copy the checkpoints to `./checkpoints/`

In default, left and right camera videos are supposed to be structured like this:
```none
./demo_video/
        ├── left
            ├── left000000.png
            ├── left000001.png
            ├── left000002.png
            ...
        ├── right
            ├── right000000.png
            ├── right000001.png
            ├── right000002.png
            ...
```

A simple way to run the demo is using SouthKensingtonSV.

To test on your own data, modify `--path ./demo_video/`. More arguments can be found and modified in ` demo.py`

## Dataset

Download the following datasets and put in `./data/datasets/`:
 - [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
 - [Sintel](http://sintel.is.tue.mpg.de/stereo)
 - [Dynamic_Replica](https://dynamic-stereo.github.io/)
 - [KITTI Depth](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
 - [Infinigen SV](https://tomtomtommi.github.io/BiDAVideo/)
 - [Virtual KITTI2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/)
 - [SouthKensington SV](https://tomtomtommi.github.io/BiDAVideo/)


## Evaluation
```
sh evaluate_stereoanyvideo.sh
```

## Training
```
sh train_stereoanyvideo.sh
```

## Citation 
If you use our method in your research, please consider citing:
```
@article{jing2025stereo,
  title={Stereo Any Video: Temporally Consistent Stereo Matching},
  author={Jing, Junpeng and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
  journal={arXiv preprint arXiv:2503.05549},
  year={2025}
}
```
