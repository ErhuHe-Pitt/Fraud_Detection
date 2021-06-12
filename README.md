====================================================


OVERVIEW

For this project, I aim to explore current machine learning models to recognize the fraud transaction.

The dataset I used is “Credit Card Fraud Detection” from Kaggle.com (https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3)

Eight models I implemented are respectively Logistic Regression Model, Naive Bayes Model, Decision Tree Model, Random Forest Model, Isolation Forest Model, Elliptic Envelope Model, One-class SVM Model and Local Outlier Factor Model.

Note that the code can successfully execute, but sometimes it takes a longer time with certain models (roughly 2 mins).

=====================================================


PROJECT LANGUAGE IMPLEMENTATION

I implemented the machine learning models with Python 3.6.8

The packages I used for the term project are as below:
•pandas. https://pandas.pydata.org/
•numpy. https://numpy.org/
•seaborn. https://seaborn.pydata.org/
•matplotlib. https://matplotlib.org/
•scikit-learn. https://scikit-learn.org/stable/


======================================================


EXPLANATION OF THE CODE

Procedures:
1. Take a glimpse into the dataset, check no-missing values
2. Plot the distribution of classes. It shows the data is hugely imbalanced.
3. Plot the distribution of features "Time" and "Amount", rescale both of these features. (Other features have been obtained by running PCA.)
4. Show the correlation among features and show the correlation of features with column "Class".
5. With imbalanced dataset, implement four supervised classifiers (Logistic Regression Model, Gaussian Naive Bayes Model, Decision Tree Model and Random Forest Model).
6. Use upsampling to generate a balanced dataset, and then implement four supervised classifiers
7. Use downsampling to generate a balanced dataset, and then implement four supervised classifiers
8. With imbalanced dataset, implement outlier detection models (Isolation Forest Model, Elliptic Envelope Model, One-class SVM Model and Local Outlier Factor Model).
