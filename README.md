# AI Codebase

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/27618c4c51a3408091f5dc4f8a4fbf06)](https://app.codacy.com/gh/nityansuman/ai-codebase?utm_source=github.com&utm_medium=referral&utm_content=nityansuman/ai-codebase&utm_campaign=Badge_Grade_Settings)
![GitHub LICENSE](https://img.shields.io/github/license/nityansuman/ai-codebase)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/nityansuman/ai-codebase)
![GitHub repo size](https://img.shields.io/github/repo-size/nityansuman/ai-codebase)
![GitHub language count](https://img.shields.io/github/languages/count/nityansuman/ai-codebase)
![GitHub last commit](https://img.shields.io/github/last-commit/nityansuman/ai-codebase)

Artificial intelligence, sometimes called machine intelligence, is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.

In the twenty-first century, AI techniques have experienced a resurgence following concurrent advances in computer power, large amounts of data, and theoretical understanding; and AI techniques have become an essential part of the technology industry, helping to solve many challenging problems in computer science, software engineering and operations research.

### Welcome to my GitHub repository

Here you will find some of my machine Learning based solution notebooks for Natural Language Processing & Understanding, Time Series, Computer Vision and everything in between.

---

## Codebase

### Natural Language

Text classification (a.k.a. text categorization or text tagging) is the task of assigning a set of predefined categories to an open-ended text. An embedding is a mapping of a discrete — categorical — variable to a vector of continuous numbers.

In the context of neural networks, embeddings are low-dimensional, learned continuous vector representations of discrete variables.

- [Word embeddings - skip gram model](natural-language/notebooks/skip-gram.ipynb)
- [Training from scratch or using pre-trained embeddings](natural-language/notebooks/embeddings.ipynb)
- [IMDb moview review classification using convolutions](natural-language/notebooks/text-classification-cnn.ipynb)

`Model Architecture` implementations:

- [Simple bi-lstm classifier implementation](natural-language/lstm_classifier.py)
- [Convolutional text classifier implementation](natural-language/conv_classifier.py)

Custom `Layer` implementations:

- [GRU encoder implementation](natural-language/encoder.py)
- [Feed forward network layer implementation from transformer architecture](natural-language/feed_forward_network.py)

<!-- ### Time Series

A time series is a series of data points indexed (or listed or graphed) in time order.

- [Sales Forecasting](time-series/)

### Structured Data

Predictive analytics is the branch of the advanced analytics which is used to make predictions about unknown events using tabulated data points.

- [Customer Churn Prediction](structured-data/)
- [Customer Lifetime Value Prediction](structured-data/) -->

### Computer Vision

Image classification (a.k.a. image categorization) is the task of assigning a set of predefined categories/labels to a groups of pixels or an image.

- [Multi-class flower classification with data augmentation](computer-vision/image-classification-with-data-agumentation.ipynb)

![model-acc](computer-vision/images/flower-classification-acc.png)
![model-loss](computer-vision/images/flower-classification-loss.png)

`Model Architecture` implementations:

- [Simple convolutional classifier implementation](computer-vision/conv_net.py)
- [LeNet model architecture implementation](computer-vision/le_net.py)
- [AlexNet model architecture implementation](computer-vision/alex_net.py)

### TensorFlow 2 API

TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. Tensorflow is a symbolic math library based on dataflow and differentiable programming.

- [Sequential API](tensorflow2-api/sequential-model-api.ipynb)
- [Functional API](tensorflow2-api/functional-model-api.ipynb)
- [Train and Evaluate Model](tensorflow2-api/train-and-evaluate-model.ipynb)
- [Model Run Customizations](tensorflow2-api/model-run-customizations.ipynb)
- [Writing Custom Layers and Models](tensorflow2-api/writing-new-layers-and-models-via-subclassing.ipynb)

[![Forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
