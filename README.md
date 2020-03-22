## RSNA Intracranial Hemorrhage Detection 2019

Our team ranked 152nd place (TOP 12%) in the 2-stage competition [RSNA Intracranial Hemorrhage Detection 2019 on Kaggle platform](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/leaderboard). This repository consists of code and configs that I used to train our models. The solution is powered by awesome [Albumentations](https://github.com/albu/albumentations), [PyTorch](https://pytorch.org), [Cadene models](https://github.com/Cadene/pretrained-models.pytorch), [EfficientNet](https://www.kaggle.com/chanhu/efficientnet) and [FastAi](https://docs.fast.ai/) libraries.

## In this repository you can find:
* `models` - folder with scripts for preprocessing data, learning models **(EfficientNet B0/B2/B5, ResNext101_32x4d)** and inference with D4 TTA
* `creating-a-metadata-dataframe-fastai.ipynb` - script by [Jeremy Howard](https://www.kaggle.com/jhoward)
* `radam.py` - script with RAdam optimizer in PyTorch
* `blending.ipynb` - script with simple blending all our models

## Solution description

### Data
We tried many neural network architectures and preprocessing strategies in this competition, but we achieved our two best single results by two models: cnn and rcnn with hard augmentations. It was an interesting competition, primarily because I never competed in 2-stage competitions before. According to the rules, in the Stage 1 submission we uploaded all our models. And then these models will be used to generate the final submissions for scoring in Stage 2. Public leaderboard in Stage 1 is calculated with approximately ~20% of the test data.

For preprocessing we used other window settings, cleaning of data from low-informative images, fixing metadata (because some images were not in the same format as the others), and so on.

### Models
From the beginning, efficientnet and resnext outperformed other models. Using fp16 allowed to use bigger batch size - speeded up training and inference. Other models (like ResNet, SeResNext or DenseNet) worked worse for us.

Models used in the final submission:
1. EfficientNet-B5 (best single model): 512x512 (pretrained on train data, finetuned on pseudolabeled test data)
2. EfficientNet-B2 (five models): 512x512 (cv 5 folds, each fold grouped by `PatientID` which is important for a stable cross validation due to overlapping patients)
3. EfficientNet-B0 (two models): 512x512 and 224x224 (other preprocessing and augmentations)
4. ResNext101_32x4d: 512x512 (with hard augmentations)

Based on `StudyInstanceUID` and sorting on `ImagePositionPatient` it is possible to reconstruct 3D volumes for each study. However since each study contained a variable number of axial slices (between 20-60) this makes it difficult to create a architecture that implements 3D convolutions. Instead, many winners in this competition created triplets of images from the 3D volumes to represent the RGB channels of an image, i.e. the green channel being the target image and the red & blue channels being the adjacent images. ***The main mistake of our solution is that we have not used this information in any way. After the end of the competition it became the main lesson for us.***

### Augmentations
From [Albumentations](https://github.com/albu/albumentations) library:
HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, Resize, CenterCrop, GridDistortion, RandomGamma, etc...

### Training
All models were trained using FastAi and Pytorch libraries. Adam/RAdam with OneCycle gave us the best results. We tried to use pseudolabeling, it has slightly improved the results of our heaviest model (EfficentNet B5).

### Hardware
We used 1x* *Tesla v40* and GCP on Google Cloud Platform.

## Team:
- Mishunyayev Nikita: [Kaggle](https://www.kaggle.com/mnikita), [GitHub](https://github.com/Mishunyayev-Nikita)
- Zack Pashkin: [Kaggle](https://www.kaggle.com/tienen), [GitHub](https://github.com/ZackPashkin)

