# CurriculumMARL: Implementation of Curriculum MARL algorithms

This is the code for [**Towards Skilled Population Curriculum for Multi-Agent Reinforcement Learning**](https://arxiv.org/pdf/2302.03429.pdf).
We also provide well-organized implementations of curriculum MARL algorithms based on Ray 2.0:

| Task Generator        | Paper                                                                                        |
|:----------------------|:---------------------------------------------------------------------------------------------|
| SPC                   | Towards Skilled Population Curriculum for Multi-Agent Reinforcement Learning                 |
| VACL                  | Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems |
| ALP-GMM               | Absolute Learning Progress and Gaussian Mixture Models for Automatic Curriculum Learning     |
| Uniform               | /                                                                                            |
| Non-curriculum (IPPO) | /                                                                                            |

## How to Start

### Docker

Docker is recommended.

To install:
```bash
docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t spc -f Dockerfile .
```
To run experiments with detached mode:
```bash
./run_experiments.sh
```
If you want to run in interactive mode, change the `-d` parameter to `-it`.

Configurations can be found in the `configs/` directory.

### Conda

```bash
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

conda create -n spc python=3.8
conda activate spc
conda install -c anaconda libffi
python -m pip install ray[rllib]==2.0.1
python -m pip install -r requirements.txt
python -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Training Examples: 
```bash
python train.py -f configs/football/bandit/corner.yaml
python train.py -f configs/football/bandit/corner-hrl-comm.yaml
python train.py -f configs/football/bandit/corner-no-context.yaml
python train.py -f configs/football/vacl/corner.yaml
python train.py -f configs/football/alp_gmm/corner.yaml
python train.py -f configs/football/uniform/corner.yaml
python train.py -f configs/football/ppo/corner.yaml

python train.py -f configs/starcraft/alp_gmm/8m.yaml
python train.py -f configs/starcraft/uniform/8m.yaml
python train.py -f configs/starcraft/ppo/8m.yaml
```

## Cite Our Paper

Please cite our paper if you've found this repository useful:

```
@misc{
    wang2023towards,
    title={Towards Skilled Population Curriculum for {MARL}},
    author={Rundong Wang and Longtao Zheng and Wei Qiu and Bowei He and Bo An and Zinovi Rabinovich and Yujing Hu and Yingfeng Chen and Tangjie Lv and Changjie Fan},
    year={2023},
    url={https://openreview.net/forum?id=GbsvQSaJV-6}
}
```