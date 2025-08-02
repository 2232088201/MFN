<div align="center">    
 
# Syntax-Image Fusion Framework for Handwritten Mathematical Expression Recognition

</div>

## Project structure
```bash
├── config/         # config for MFN hyperparameter
├── data/
│   └── googleHME    # googleHME Dataset
│   └── HME100k      # HME100k Dataset which needs to be downloaded according to the instructions below.
├── mfn               # model definition folder
├── lightning_logs      # training logs
│   └── version_0      # ckpt for CROHME dataset
│       ├── checkpoints
│       │   └── epoch=125-step=47375-val_ExpRate=0.6101.ckpt
│       ├── config.yaml
│       └── hparams.yaml
│   └── version_1      # ckpt for HME100k dataset
│       ├── checkpoints
│       │   └── epoch=55-step=175503-val_ExpRate=0.6924.ckpt
│       ├── config.yaml
│       └── hparams.yaml
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Install dependencies   
```bash
cd MFN
# install project   
conda create -y -n MFN python=3.9
conda activate MFN
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
 ```
## Dataset Preparation
The HME100K dataset has to be preprocessed. And the processed data will be released in the future. 


For single gpu user, you may change the `config.yaml` file to
```yaml
gpus: 1
```

## Training on HME100k Dataset
It may take about **48** hours on **4** NVIDIA 2080Ti gpus using ddp on HME100k dataset.
```bash
# train MFN model using 4 gpus and ddp on hme100k dataset
python train.py --config /config/hme100k.yaml
```

## Evaluation
Trained MFN weight checkpoints for CROHME and HME100K Datasets have been saved in `lightning_logs/version_0` and `lightning_logs/version_1`, respectively.

```bash

# For HME100K Dataset
bash eval/eval_hme100k.sh 1
```




## Reference
- [ICAL](https://github.com/qingzhenduyu/ICAL) | [arXiv](https://arxiv.org/abs/2405.09032)
- [CoMER](https://github.com/Green-Wood/CoMER) | [arXiv](https://arxiv.org/abs/2207.04410)
- [BTTR](https://github.com/Green-Wood/BTTR) | [arXiv](https://arxiv.org/abs/2105.02412)
- [TreeDecoder](https://github.com/JianshuZhang/TreeDecoder)
- [CAN](https://github.com/LBH1024/CAN) | [arXiv](https://arxiv.org/abs/2207.11463)


