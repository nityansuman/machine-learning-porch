# Awesome AI

![GitHub LICENSE](https://img.shields.io/github/license/nityansuman/awesome-ai)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/nityansuman/awesome-ai)
![GitHub repo size](https://img.shields.io/github/repo-size/nityansuman/awesome-ai)
![GitHub language count](https://img.shields.io/github/languages/count/nityansuman/awesome-ai)
![GitHub last commit](https://img.shields.io/github/last-commit/nityansuman/awesome-ai)

Here you will find some of my AI based solution notebooks for Anaytics, Natural Language Processing, Time Series, Computer Vision and everything in between.

## Natural Language Processing

#### Embeddings

In the context, embeddings are low-dimensional, learned continuous vector representations of discrete variables.
An embedding is a mapping of a discrete — categorical — variable to a vector of continuous numbers.
Word embedding is a term used for the representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning

- [Word2Vec Embeddings - Skip Gram Model Using Negative Sampling](natural-language/) - The Skip-gram model architecture usually tries to achieve the reverse of what the CBOW model does. It tries to predict the source context words (surrounding words) given a target word (the center word).
- [Training Embeddings From Scratch Or Use Pre-trained Embeddings](natural-language/) - Pretrained Word Embeddings are the embeddings learned in one task that are used for solving another similar task. These embeddings are trained on large datasets, saved, and then used for solving other tasks. That's why pretrained word embeddings are a form of Transfer Learning.

#### Text Classification

Text classification also known as text tagging or text categorization is the process of categorizing text into organized groups.

- [IMDb Moview Review Sentiment Classification](imdb-movie-review-classification.ipynb) - Implemented a stacked bi-directional LSTM architecture which learns embeddings from scratch and classifies movie reviews into positive and negative sentiments with an accuracy of over 85% on out of sample test dataset.

#### Recommendation Engine

Recommender System is a system that seeks to predict or filter preferences according to the user's choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general.

- [Movie Recommendation Engine: Retrieval Stage](natural-language/recommendation-engine-retrieval.ipynb) - Implemented a simple movie retrieval engine on the infamous `movielens` dataset. The retrieval stage is responsible for selecting an initial set of hundreds of candidates from all possible candidates. The main objective of this model is to efficiently weed out all candidates that the user is not interested in. Because the retrieval model may be dealing with millions of candidates, it has to be computationally efficient.
- [Movie Recommendation Engine: Ranking Stage](natural-language/recommendation-engine-retrieval.ipynb) - Implemented a descent movie ranking engine on the infamous `movielens` dataset. The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best possible handful of recommendations. Its task is to narrow down the set of items the user may be interested in to a shortlist of likely candidates.

## Analytics

#### Customer Churn Prediction

Churn quantifies the number of customers who have left the brand by cancelling their subscription or stop paying for services. This is bad news for any business as it costs five times as much to attract a new customer as it does to keep an existing one.

A high customer churn rate will hit any company’s finances hard. By leveraging advanced artificial intelligence techniques like machine learning (ML), one can anticipate potential churners.

- [Customer Churn Prediction Through Usage Analysis](churn-prediction-through-usage-analysis.ipynb) - The bagging based ensemble model (Random Forest) was able to detect 77% (recall on test set) of the churners with an accuracy of above 70% on an imbalanced dataset using random over-sampling, which is a good performance considering our objective is to detect churners.

#### Credit Card Fraudulent Transaction Detection

Fraud detection is a set of processes and analyses that allow businesses to identify and prevent unauthorized financial activity. This can include fraudulent credit card transactions, identify theft, cyber hacking, insurance scams, and more. Companies can incorporate fraud detection into their websites, company policies, employee training, and enhanced security features. The most effective companies combat fraud by using a multifaceted approach that integrates several of these techniques.

- [Credit Card Fraudulent Transaction Detection](credit-fraud-detection.ipynb) - The bagging based ensemble model (Random Forest) was able to identify 100% (recall on test) of the fraud transactions with only 9 instances of FPR (false positive rate) on a very highly imbalanced dataset (where only 0.172% transactions where fraudulent) using SMOTE and weighted objective function.

## Computer Vision

#### Image Classification

Image classification (a.k.a. image categorization) is the task of assigning a set of predefined categories/labels to a groups of pixels or an image.

- [Multi-class Flower Classification With Data Augmentation](multi-class-image-classification-with-data-augmentation.ipynb)

## TensorFlow 2 API

TensorFlow is a free and open-source software library for machine learning. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. Tensorflow is a symbolic math library based on dataflow and differentiable programming.

- [Sequential API](tensorflow2-api/sequential-model-api.ipynb)
- [Functional API](tensorflow2-api/functional-model-api.ipynb)
- [Custom Loss](tensorflow2-api/custom-loss-function.ipynb)
- [Custom Metric](tensorflow2-api/custom-metric-function.ipynb)
- [Callbacks](tensorflow2-api/all-about-callbacks.ipynb)
- [Writing Custom Layers and Models](tensorflow2-api/writing-new-layers-and-models-via-subclassing.ipynb)

[![Forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![Forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
