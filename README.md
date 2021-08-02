# Awesome AI: 2021

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/27618c4c51a3408091f5dc4f8a4fbf06)](https://app.codacy.com/gh/nityansuman/awesome-ai-2021?utm_source=github.com&utm_medium=referral&utm_content=nityansuman/awesome-ai-2021&utm_campaign=Badge_Grade_Settings)
![GitHub LICENSE](https://img.shields.io/github/license/nityansuman/awesome-ai-2021)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/nityansuman/awesome-ai-2021)
![GitHub repo size](https://img.shields.io/github/repo-size/nityansuman/awesome-ai-2021)
![GitHub language count](https://img.shields.io/github/languages/count/nityansuman/awesome-ai-2021)
![GitHub last commit](https://img.shields.io/github/last-commit/nityansuman/awesome-ai-2021)

## Welcome

Here you will find some of my AI based solution notebooks for Anaytics, Natural Language Processing, Time Series, Computer Vision and everything in between.

---

## Analytics

- [Churn Prediction Through Usage Analysis](analytics/)
- [Credit Card Fraudulent Transaction Detection](analytics/)

## Natural Language

### Embeddings

In the context, embeddings are low-dimensional, learned continuous vector representations of discrete variables.
An embedding is a mapping of a discrete — categorical — variable to a vector of continuous numbers.
Word embedding is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning

- [Word2Vec Embeddings - Skip Gram Model Using Negative Sampling](natural-language/skip-gram-word2vec.ipynb)
- [Training Embeddings From Scratch Or Use Pre-trained Embeddings](natural-language/embeddings-playground.ipynb)

### Text Classification

Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups.

- [IMDb Moview Review Classification](natural-language/imdb-movie-review-classification.ipynb) - Implemented a stacked bi-directional LSTM architecture which learns embeddings from scratch and classifies movie reviews into `positive` and `negative` sentiments with an accuracy of over 85% on out of sample test dataset.

![model-acc](natural-language/images/movie-review-classification-acc.png)
![model-loss](natural-language/images/movie-review-classification-loss.png)

Custom `Layer` implementations:

- [GRU Encoder Implementation](natural-language/encoder.py)
- [Feed Forward Network Layer Implementation From Transformer Architecture](natural-language/feed_forward_network.py)

<!-- ### Time Series

A time series is a series of data points indexed (or listed or graphed) in time order.

- [Sales Forecasting](time-series/)

### Structured Data

Predictive analytics is the branch of the advanced analytics which is used to make predictions about unknown events using tabulated data points.

- [Customer Churn Prediction](structured-data/)
- [Customer Lifetime Value Prediction](structured-data/) -->

## Computer Vision

### Image Classification

Image classification (a.k.a. image categorization) is the task of assigning a set of predefined categories/labels to a groups of pixels or an image.

- [Multi-class Flower Classification With Data Augmentation](computer-vision/multi-class-image-classification-with-data-agumentation.ipynb)

![model-acc](computer-vision/images/flower-classification-acc.png)
![model-loss](computer-vision/images/flower-classification-loss.png)

`Model Architecture` implementations:

- [LeNet Model Architecture Implementation](computer-vision/le_net.py)
- [AlexNet Model Architecture Implementation](computer-vision/alex_net.py)

## TensorFlow 2 API

TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. Tensorflow is a symbolic math library based on dataflow and differentiable programming.

- [Sequential API](tensorflow2-api/sequential-model-api.ipynb)
- [Functional API](tensorflow2-api/functional-model-api.ipynb)
- [Train and Evaluate Model](tensorflow2-api/train-and-evaluate-model.ipynb)
- [Model Run Customizations](tensorflow2-api/model-run-customizations.ipynb)
- [Writing Custom Layers and Models](tensorflow2-api/writing-new-layers-and-models-via-subclassing.ipynb)

[![Forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
