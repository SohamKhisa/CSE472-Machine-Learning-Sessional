# Machine Learning Sessional: CSE472

## Offline-1: Linear Algebra
This one is the first assignment of this course. We had to implement some linear algebra techniques in python.

## Offline-2: Logistic Regression
In this assignment we had to implement Logistic regression in python for bank note authentication.
The dataset is given <a href="https://archive.ics.uci.edu/dataset/267/banknote+authentication">here</a>.
We also had to implement Logistic regression with bagging alongside the base logistic regression. We had to compare
these two methods.

## Offline-3: Expectation Maximization (EM) algorithm and Gaussian Mixture Model (GMM)
In this assignment, we had to learn unsupervised learning and implement Gaussian Mixture Model.
We were given n data points, each of which has m attributes. The samples
were generated from a mixture of a k number of unknown Gaussian distributions.
often referred to as Gaussian Mixture Model (GMM). Our task was to estimate the parameters of
k unknown Gaussian distributions. We used the EM (Expectation-Maximization)
algorithm for this task. We also had to plot a graph on how the convergence takes place for
each iteration in the EM algorithm.

## Offline-4: Convolution Neural Network from Scratch
We had to implement the Convolution Neural Network from scratch (Or, using very limited libraries)
to detect Bangla handwritten digits. Please check the problem specification to get an idea about what set of libraries were allowed to
use for which tasks. We implemented LeNet for this assignment.

## Capstone Project: Retinal Disease Clasification

<b>Problem Definition</b>

We wish to work on a project to create a system for diagnosing retinal
illnesses based on machine learning. The project's objective is to use
retinal images to recognize and classify various retinal illnesses, including
age-related macular degeneration and diabetic retinopathy. A label
identifying the presence or absence of a certain illness will be produced by
the system after receiving an input retinal picture. The goal is to enhance
the precision and accuracy of illness detection and diagnosis, which will
eventually improve patient outcomes.

<b>Dataset</b>

The dataset we used to train our model is from <a href="https://www.kaggle.com/code/vexxingbanana/retinal-disease-classification/data">Kaggle</a>. The training dataset consists of 1920 sample images, whereas the test set and evaluation set both contain 640 sample images. The three datasets are each accompanied by a CSV file.

<b>Architecture</b>

At first we used Restricted Boltzmann Machine but faced some difficulties to implement it. Worked rigorously to fix it. But it was not fruitful.
Thus, we finally switched to CNN-AlexNet and implemented in a very short time.
