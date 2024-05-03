# Kaggle Titanic Machine Learning Competition

## About
Kaggle has a [competition](https://www.kaggle.com/competitions/titanic) where you build a ML model to predict whether Titanic passengers survive. I wanted to experiment on the efficiency of different models.

### Multilayer Perceptron 
With a simple 2 layer MLP model, I achieved a wide range of accuracy ranging from 38% to 62%. The large range is due to the shuffling of training data and different models being trained on different difficulties of training/validation set.

It was also difficult to obtain a model with a validation loss is higher than its training loss. These model's would also have a very jagged graph.

### Support Vector Machine
Using a svm with linear kernel, and little data preprocessing, it was easy to achieve an accuracy of 77%

