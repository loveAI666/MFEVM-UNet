# MFEVM-UNet
This is the official code repository for "MFEVM-UNet: Multi-scale Feature Fusion and Enhancement Vision Mamba
UNet for medical image segmentation". {[Biomedical Signal Processing and Control](https://www.sciencedirect.com/science/article/abs/pii/S174680942501095X)}

## Abstract
Accurate segmentation of tissue and lesion areas is crucial for precise lesion localization, assisting in surgical planning and providing quantitative lesion assessment. Yet, traditional Convolutional Neural Networks (CNNs) face limitations in local receptive fields and Transformer models exhibit high computational complexity when processing complex medical images. In recent years, State Space Models (SSMs) have exhibited impressive capabilities in the field of vision, in particular Mamba and its variants. They are istinguished in long-distance modeling while preserving linear computational efficiency in the context. However, simple skip connections still have limitations in capturing non-local multi-scale features, and their capacity for feature extraction might not reach the desired level of effectiveness. In response to these challenges, we propose a new model
named Multi-scale Feature Fusion and Enhancement Vision Mamba UNet (MFEVM-UNet). Specifically, we introduce the Spatial Feature Gating Convolution (SFGC) module to improve spatial feature extraction and representation. Additionally, we employ the Mixed-Scale Interaction Module (MSIM) to obtain more enriched multi-scale feature representations. Finally, a balance between local and global features is achieved through a Local-2D-Selective-Scan (Local-SS2D), integrating a local scanning strategy with 2D-Selective-Scan (SS2D). We
performed experiments on three public datasets: ISIC2017, ISIC2018, and CVC-ClinicDB, demonstrating the outstanding performance of our proposed MFEVM-UNet model in 2D medical segmentation tasks.

## 0. Main Environments
(Same as the VM-UNet experimental environment）
```bash
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```
The .whl files of causal_conv1d and mamba_ssm could be found here. {[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)}

## 1. Prepare the dataset

### ISIC datasets
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/19zO90ruVREEqrKHAqOQAtw?pwd=puna)}. 

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

### CVC-ClinicDB

- The CVC-ClinicDB datasets, divided into a 7:3 ratio,, you could follow [](https://pan.baidu.com/s/1FFku4zxyFfEZOF5Zx4AMTQ?pwd=rjgd)}.

- After downloading the datasets, you are supposed to put them into './data/CVC_ClinicDB/', and the file format reference is as follows.

- './data/CVC_ClinicDB/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/13cTgCUhMTvuWQ8HxNVaB8g?pwd=iq3q). After that, the pre-trained weights should be stored in './pretrained_weights/'.



## 3. Train the MFEVM-UNet
```bash
python train.py  # Train and test MFEVM-UNet on the ISIC17 or ISIC18 dataset.

```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 5. Acknowledgments

- We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba) and [VM-UNet]((https://github.com/JCruan519/VM-UNet)) and [LocalMamba](https://github.com/hunto/LocalMamba) for their open-source codes.
