# Brain Tumor Classification using Deep Learning

This project implements and evaluates different deep learning methods for classifying brain tumor images. I compare a custom CNN, a hyperparameter-tuned CNN, and a VGG16-based transfer learning model to address challenges like small datasets, class imbalance, and varying image modalities.

## Overview

- Developed models to classify MRI images into tumor and non-tumor categories.
- Addressed class imbalance using class weighting.
- Applied data augmentation to increase dataset diversity.
- Achieved best results using transfer learning with VGG16.

## Results Summary

| Model                     | Accuracy | Tumor Precision | Non-Tumor Recall | F1-Score (Tumor) |
|--------------------------|----------|------------------|------------------|------------------|
| Baseline CNN             | 67.8%    | 0.70             | 0.09             | 0.82             |
| Hyperparameter-Tuned CNN | 73.4%    | 0.78             | 0.45             | 0.84             |
| **VGG16 Transfer Learning** | **82.4%** | **0.90**         | **0.82**         | **0.86**         |

## Methodology

### Dataset

- Source: MRI brain tumor dataset with two classes: `yes` (tumor) and `no` (non-tumor).
- Challenges:
  - Only ~350 images in total.
  - Highly imbalanced: ~250 tumor vs ~90 non-tumor.
  - Mixed modalities (T1/T2 MRI, CT).

### Preprocessing

- Resized all images to `224x224`.
- Normalized pixel values to `[0, 1]`.
- Applied augmentations: rotation, flipping, zoom.
- Used class weighting to balance training on skewed data.

### Models

#### Baseline CNN

- 3 convolutional blocks + 4 dense layers.
- Trained for 10 epochs with batch size 20.
- No class weights.

#### Hyperparameter-Tuned CNN

- Random Search + Grid Search optimization:
  - Learning rate: 0.001
  - Dropout rate: 0.4
  - Epochs: 20
  - Batch size: 20
- Improved non-tumor recall and model stability.

#### VGG16 Transfer Learning

- Used pre-trained VGG16 base (ImageNet).
- Added custom FC layers with dropout.
- Fine-tuned top convolutional layers with a reduced learning rate (0.00001).
- Best performance across all metrics.

## Learning Curves

- **Baseline CNN**: Fluctuating validation metrics, signs of overfitting.
- **Tuned CNN**: More stable, improved non-tumor detection.
- **VGG16**: High validation accuracy, minor signs of overfitting due to augmentation.

## Insights

- Transfer learning significantly boosts performance with small datasets.
- Class imbalance heavily affects custom models.
- Image augmentation improves generalization but may introduce training instability.
- Validation accuracy > training accuracy suggests future scope for optimization.

## References

- [Pete Warden - Dataset Size Guidelines](https://petewarden.com/2017/12/14/how-many-images-do-you-need-to-train-a-neural-network/)
- [ScienceDirect - Brain Hematoma Misclassification](https://www.sciencedirect.com/science/article/abs/pii/S1878875021013589)
- [Teach Me Anatomy - Imaging Modalities](https://teachmeanatomy.info/the-basics/imaging/)
- [Medium - Image Preprocessing](https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c)
- [TensorFlow - ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Medium - Fine-Tuning Models](https://medium.com/@amanatulla1606/fine-tuning-the-model-what-why-and-how-e7fa52bc8ddf)


---

Feel free to clone, explore, and build on this project!
