# Dual Stream Semi-Supervised Learning for Melanoma Classification

This repository contains a TensorFlow implementation of a dual stream Convolutional Neural Network (CNN) designed for semi-supervised learning, targeting the classification of Melanoma from skin images.

## Architecture

- **Dual stream CNN**: Two parallel CNN streams, each with three convolution and max pooling blocks.
- **Dropout**: Applied with a rate of 20% for regularization.
- **Concatenation**: The flattened outputs of both streams are concatenated.
- **Classification**: The concatenated features are passed through a softmax layer for final classification.
- **Input size**: 64x64 RGB images.

## Dataset

The model is designed to work with the [ISIC 2020 Melanoma Classification Challenge dataset](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data).

## Usage

1. Clone this repository.
2. Prepare your dataset (resize images to 64x64).
3. Adjust the data loading pipeline as needed.
4. Run `dual_stream_cnn.py` to view the model summary or integrate it into your training pipeline.

## License

This project is released under the MIT License.