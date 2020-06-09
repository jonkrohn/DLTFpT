# Deep Learning with TensorFlow, Keras, and PyTorch

This repository is home to the code that accompanies [Jon Krohn's](www.jonkrohn.com) *Deep Learning with TensorFlow, Keras, and PyTorch* series of video tutorials. 

There are three sets of video tutorials in the series: 

1. The eponymous [Deep Learning with TensorFlow, Keras, and PyTorch](https://learning.oreilly.com/videos/deep-learning-with/9780136617617) (released in Feb 2020)
2. [Deep Learning for Natural Language Processing, 2nd Ed.](https://learning.oreilly.com/videos/deep-learning-for/9780136620013) (Feb 2020)
3. [Machine Vision, GANs, and Deep Reinforcement Learning](https://learning.oreilly.com/videos/machine-vision-gans/9780136620181) (Mar 2020)

The above order is the recommended sequence in which to undertake these tutorials. That said, the first in the series provides a strong foundation for either of the other two. 

Taken all together, the series -- over 18 total hours of instruction and hands-on demos -- parallels the entirety of the content in the book [Deep Learning Illustrated](https://www.deeplearningillustrated.com/). This means that the videos introduce **all of deep learning**: 

* **What deep neural networks are** and how they work, both mathematically and using the most popular code libraries
* **Machine vision**, primarily with convolutional neural networks
* **Natural language processing**, including with recurrent neural networks
* **Artistic creativity** with generative adversarial networks (GANs)
* **Complex, sequential decision-making** with deep reinforcement learning

These video tutorials also includes some extra content that is not available in the book, such as: 

* Detailed interactive examples involving training and testing deep learning models in PyTorch
* How to generate novel sequences of natural language in the style of your training data
* High-level discussion of transformer-based natural-language-processing models like BERT, ELMo, and GPT-2 
* Detailed interactive examples of training advanced machine vision models (image segmentation, object detection)
* All hands-on code demos involving TensorFlow or Keras have been updated to TensorFlow 2

## Installation

Installation instructions for running the code in this repository can be found in the [installation directory](https://github.com/jonkrohn/DLTFpT/tree/master/installation).

## Notebooks

There are dozens of meticulously crafted Jupyter notebooks of code associated with these videos. All of them can be found in [this directory](https://github.com/jonkrohn/DLTFpT/tree/master/notebooks). 

Below is a breakdown of the lessons covered across the videos, including their duration and associated notebooks.

#### Deep Learning with TensorFlow, Keras, and PyTorch 

* Seven hours and 13 minutes total runtime
* Lesson 1: Introduction to Deep Learning and Artificial Intelligence (1 hour, 47 min)
	* [Shallow Net in TensorFlow](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/shallow_net_in_tensorflow.ipynb)
* Lesson 2: How Deep Learning Works (2 hours, 16 min) -- free YouTube video [here](https://youtu.be/wBgW3ZtlPT8)
	* [Sigmoid Function](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/sigmoid_function.ipynb)
	* [Softmax Demo](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/softmax_demo.ipynb)
	* [Quadratic Cost](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/quadratic_cost.ipynb)
	* [Cross-Entropy Cost](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/cross_entropy_cost.ipynb)
	* [Intermediate Net in TensorFlow](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/intermediate_net_in_tensorflow.ipynb)
* Lesson 3: High-Performance Deep Learning Networks (1 hour, 16 min)
	* [Weight Initialization](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/weight_initialization.ipynb)
	* [Measuring Speed of Learning](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/measuring_speed_of_learning.ipynb)
	* [Deep Net in TensorFlow](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/deep_net_in_tensorflow.ipynb)
	* [Regression in TensorFlow](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/regression_in_tensorflow.ipynb)
	* [Regression with TensorBoard](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/regression_in_tensorflow_with_tensorboard.ipynb)
* Lesson 4: Convolutional Neural Networks (47 min)
	* [LeNet in TensorFlow](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/lenet_in_tensorflow.ipynb)
* Lesson 5: Moving Forward with Your Own Deep Learning Projects (1 hour, 4 min)
	* [Shallow Net in PyTorch](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/shallow_net_in_pytorch.ipynb)
	* [Deep Net in PyTorch](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/deep_net_in_pytorch.ipynb)
	* [LeNet in TensorFlow for Fashion MNIST](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/lenet_in_tensorflow_for_fashion_MNIST.ipynb)

#### Deep Learning for Natural Language Processing

* Five hours total runtime
* Lesson 1: The Power and Elegance of Deep Learning for NLP (46 min)
* Lesson 2: Word Vectors (1 hour, 7 min) -- free YouTube video [here](https://youtu.be/rqyw06k91pA)
	* [Creating Word Vectors with word2vec](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/creating_word_vectors_with_word2vec.ipynb)
* Lesson 3: Modeling Natural Language Data (1 hour, 43 min)
	* [Natural Language Preprocessing](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/natural_language_preprocessing.ipynb)
	* [Document Classification with a Dense Neural Net](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/dense_sentiment_classifier.ipynb)
	* [Classification with a Convolutional Neural Net](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/convolutional_sentiment_classifier.ipynb)
* Lesson 4: Recurrent Neural Networks (25 min)
	* [RNN](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/rnn_sentiment_classifier.ipynb)
	* [LSTM](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/lstm_sentiment_classifier.ipynb)
	* [GRU](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/gru_sentiment_classifier.ipynb)
* Lesson 5: Advanced Models (54 min)
	* [Bidirectional LSTM](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/bidirectional_lstm_sentiment_classifier.ipynb)
	* [Stacked Bi-LSTM](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/stacked_bi_lstm_sentiment_classifier.ipynb)
	* [Convolutional-LSTM Stack](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/conv_lstm_stack_sentiment_classifier.ipynb)
	* [Sequence Generation](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/sequence_generation.ipynb)
	* [Keras Functional API](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/multi_convnet_sentiment_classifier.ipynb)

#### Machine Vision, GANs, and Deep Reinforcement Learning

* Six hours and six minutes total runtime
* Lesson 1: Orientation (35 min)
* Lesson 2: Convolutional Neural Networks for Machine Vision (2 hours, 2 min) -- free YouTube video [here](https://youtu.be/c4e7nleyoZM)
	* [LeNet](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/lenet_in_tensorflow.ipynb)
	* [AlexNet](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/alexnet_in_tensorflow.ipynb)
	* [VGGNet](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/vggnet_in_tensorflow.ipynb)
	* [Object Detection](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/object_detection.ipynb)
	* [Transfer Learning](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/transfer_learning_in_tensorflow.ipynb)
* Lesson 3: Generative Adversarial Networks for Creativity (1 hour, 22 min)
	* [Cartoon-Drawing GAN](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/generative_adversarial_network.ipynb)
* Lesson 4: Deep Reinforcement Learning (38 min)
* Lesson 5: Deep Q-Learning and Beyond (1 hour, 25 min)
	* [Cartpole Game-Playing DQN](https://github.com/jonkrohn/DLTFpT/blob/master/notebooks/cartpole_dqn.ipynb)
	
You've reached the bottom of this page! As a reward, here's a myopic trilobite created by Aglae Bassens, a co-author of the book [Deep Learning Illustrated](https://deeplearningillustrated.com):  

![](https://github.com/illustrated-series/deep-learning-illustrated/blob/master/img/bespectacled_trilobite.jpeg)
