# Flower Classification Experiment
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IcrSSyGDPdUoGuDQ97zpeM-DMxam5wOw?usp=sharing)

## Overview
This repository contains the code for a flower classification project conducted as part of the SYSUSZ ECE371 assignment, with experiments organized into two folders: `Ex1` and `Ex2`. The `Ex1` folder implements a finetuning work of ResNet50 using the OpenMMLab framework. The `Ex2` folder contains the main experiment, focusing on classifying a dataset of flower images into five categories, exploring the impact of network depth and model architecture on classification performance. Two experiments were performed in `Ex2`: the first assesses ResNet models with varying depths (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152), and the second compares six architectures (ResNet101, DenseNet121, EfficientNet-B0, ViT-B-16, Swin-T, ConvNeXt-Tiny). Key findings include ResNet101 achieving a validation accuracy of 0.920, with deeper models showing overfitting, while ConvNeXt-Tiny excelled with a validation accuracy of 0.9474.

## Dataset
The Flower Dataset used in this project consists of images across five flower categories, split into 80% training and 20% validation sets. Preprocessing includes random cropping to 224x224, augmentations (flips, rotations, color jitter), and normalization (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]). The dataset is expected to be located in the directory `EX2/flower_dataset/`.

## Requirements
To run the code, ensure you have the following installed:
- Python 3.8+
- PyTorch 
- torchvision
- matplotlib
Install dependencies using:
```bash
pip install torch torchvision matplotlib
```

## Results
  ![figure1](Ex2/train_acc.png "train_acc")
  
  ![figure2](Ex2/val_acc.png "val_acc")
  
![figure3](Ex2/train_loss_modalcomparision.png "train_loss_modalcomparision")

  ![figure4](Ex2/val_loss_modalcomparision.png "val_loss_modalcomparision")
  ## Insights
  - Moderate network depth (e.g., ResNet101) balances accuracy and generalization for flower classification.
  - Hybrid architectures like ConvNeXt-Tiny offer superior performance and efficiency, making them ideal for this task.
  - Large models like ViT-B-16 may overfit on smaller datasets, requiring careful tuning.
  ## Acknowledgments
  This project was completed as part of the SYSUSZ ECE371 course assignment. Thanks to the course instructors for guidance and Kaggle & Colab for providing computational resources.
