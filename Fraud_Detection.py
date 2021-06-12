# -*- coding: utf-8 -*-
#@author: Erhu He

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings("ignore")

#Read the dataset
data = pd.read_csv('creditcard.csv')


#Take a glimpse into the dataset, check no-missing values
print("\n======Take a glimpse into the dataset:=======")
print(data.head(10))
print(data.columns) # dataset column
print("")
print(data.info())  # dataset information
print("")
# show information of features without PCA transformation
print(data[['Time', 'Amount', 'Class']].describe(include='all'))
print("\n")

#Plot the distribution of classes, the dataset is imbalanced
print("\n======Plot the distribution of classes:=======")

def plot_class_distribution(data):
    sns.countplot('Class', data=data)
    plt.ylabel('Transactions')
    plt.xticks([0, 1], ['Genuine', 'Fraud'])
    plt.show()

genuine = len(data[data['Class'] == 0])
fraud = len(data) - genuine 
total_transactions = len(data)
print('Labels distribution:\n{}'.format(data['Class'].value_counts()))
print('{:.3%} of transactions are genuine and {:.3%} of them are fraud.'.format(genuine / total_transactions, fraud / total_transactions))
plot_class_distribution(data)

#Plot the distribution of columns "Time" and "Amount"
#28 features have been prepocessed by PCA
print("\n======Plot the distribution of Feature Time and Amount:=======")
def plot_time_amount(time_name, amount_name, data):    
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    sns.distplot(data[time_name], ax=ax[0])
    sns.distplot(data[amount_name], ax=ax[1], color='green')
    ax[0].set_xlabel('Feature: Time' )
    ax[1].set_xlabel('Feature: Amount')
    plt.show()
    
plot_time_amount('Time', 'Amount', data)

#Data Processing (rescaling/normalization)
print("\n======Rescaling/Normalization for Feature Time and Amount:=======")
rb_scaler = RobustScaler() # RobustScaler is good in rescaling with outliers
data['normalized_time'] = rb_scaler.fit_transform(data['Time'].values.reshape(-1, 1))
data['normalized_amount'] = rb_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data[[ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
       'normalized_time', 'normalized_amount', 'Class']]
plot_time_amount('normalized_time', 'normalized_amount', data)
data.info()

#The heat map of correlation of features
print("\n======The heat map of correlation of features:=======")
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.8,square = True)
plt.show()

#The correlation of features with column Class
print("\n======The correlation of features with column Class:=======")
def correlation_with_class(data, col_fig = 3, skip_features={'Class'}):
    len_skip = len(skip_features)
    n_col = len(data.columns) - len_skip # -1 for ignoring the feature 'Class'
    row_fig = (n_col + col_fig - 1) // col_fig
    figs, axes = plt.subplots(row_fig, col_fig, figsize=(col_fig * 8, row_fig * 3.2))
    figs.subplots_adjust(hspace=0.3, wspace=0.2)
    axes = axes.flat
    idx = 0
    for col in data.columns: # do not plot feature 'Class'
        if col not in skip_features:
            sns.boxplot(x='Class', y=col, data=data, ax=axes[idx])
            axes[idx].set_title('Feature {}'.format(col))
            axes[idx].set_ylabel('')
            idx += 1
    plt.show()
correlation_with_class(data)

#-------------------------------------------------------------------------
print("\n")
#Training on normalized imbalanced data
print("\n======Training on normalized imbalanced data:=======")
# function of ROC plot 
def plot_ROC(y_true, y_score):   
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc

# results after classifiction
def classification(X_train, y_train, X_test, y_test, classifier):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    y_score = classifier.predict_proba(X_test)
    y_score = y_score[:,1]
    #print(y_score)
    #print("Confusin Matix:")
    #plot_confusion_matrix(y_test,y_pred, classes=['Genuine', 'Fraud'],title='Confusion matrix' )
    plot_ROC(y_test, y_score)
    ax = sns.heatmap(cm, cmap='Greens', annot=True, square=True, xticklabels=['Genuine', 'Fraud'], yticklabels=['genuine', 'Fraud'])
    ax.set_xlabel('Predicted Transaction')
    ax.set_ylabel('Truth Transaction')
    print(classification_report(y_test, y_pred))
    correct = np.sum(np.equal(y_pred, y_test))
    print('{} corrects out of {} test samples'.format(correct, len(y_test)))
    print('Accuracy on validation data: {:.2%}'.format(correct / len(y_test)))
    plt.show()
    
#Split the imbalanced dataset with simple holdout method
X_imblc = data.drop(columns='Class')
y_imblc = data['Class']
X_imblc_train, X_imblc_test, y_imblc_train, y_imblc_test = train_test_split(X_imblc, y_imblc, test_size=1/3, random_state=42)

print("\n======Logistic Regression with imbalanced data:=======")
#start = time.time()
LR_classfication = LogisticRegression()
classification(X_imblc_train, y_imblc_train, X_imblc_test, y_imblc_test, LR_classfication)
#end = time.time()
#print("Training time: ", end-start)


print("\n======Naive Bayes (Guassian Naive Bayes) with imbalanced data:=======")
GNB_classfication = GaussianNB()
classification(X_imblc_train, y_imblc_train, X_imblc_test, y_imblc_test, GNB_classfication)

print("\n======Decision Tree with imbalanced data:=======")
DT_classfication = DecisionTreeClassifier(max_depth=3)
classification(X_imblc_train, y_imblc_train, X_imblc_test, y_imblc_test, DT_classfication)

print("\n======Random Forest with imbalanced data:=======")
RF_classfication = RandomForestClassifier()
classification(X_imblc_train, y_imblc_train, X_imblc_test, y_imblc_test, RF_classfication)


#-------------------------------------------------------------------------
print("\n\n")
#Training on normalized balanced data
print("\n======Training on normalized balanced data:=======")

print("\n======Resampling - Upsampling:=======")
# upsampling function
def upsampling_balance(data):
    genuine = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    genuine = genuine.sample(frac=1)
    fraudOverSample = fraud
    while len(genuine) - len(fraudOverSample) > 0:
        fraudOverSample = fraudOverSample.append(fraud, ignore_index=True)
    balanced_data = genuine.append(fraudOverSample, ignore_index=True)
    balanced_data = balanced_data.sample(frac=1)
    return balanced_data

balanced_data_up = upsampling_balance(data)
plot_class_distribution(balanced_data_up)

#Split the balanced dataset
X_blc_up = balanced_data_up.drop(columns='Class')
y_blc_up = balanced_data_up['Class']
X_blc_up_train, X_blc_up_test, y_blc_up_train, y_blc_up_test = train_test_split(X_blc_up, y_blc_up, test_size=1/3, random_state=42)

print("\n======Logistic Regression with balanced data:=======")
LR_classfication = LogisticRegression()
classification(X_blc_up_train, y_blc_up_train, X_blc_up_test, y_blc_up_test, LR_classfication)

print("\n======Naive Bayes (Guassian Naive Bayes) with balanced data:=======")
GNB_classfication = GaussianNB()
classification(X_blc_up_train, y_blc_up_train, X_blc_up_test, y_blc_up_test, GNB_classfication)

print("\n======Decision Tree with balanced data:=======")
DT_classfication = DecisionTreeClassifier(max_depth=3)
classification(X_blc_up_train, y_blc_up_train, X_blc_up_test, y_blc_up_test, DT_classfication)

print("\n======Random Forest with balanced data:=======")
RF_classfication = RandomForestClassifier()
classification(X_blc_up_train, y_blc_up_train, X_blc_up_test, y_blc_up_test, RF_classfication)

# Downsampling
print("\n\n\n======Resampling - Downsampling:=======")
# downsampling function
def downsampling_balance(data):
    genuine = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    genuine = genuine.sample(frac=1)
    genuine = genuine[: len(fraud)]
    balanced_data = genuine.append(fraud, ignore_index=True)
    balanced_data = balanced_data.sample(frac=1)
    return balanced_data

balanced_data_down = downsampling_balance(data)
plot_class_distribution(balanced_data_down)

#Split the balanced dataset
X_blc_down = balanced_data_down.drop(columns='Class')
y_blc_down = balanced_data_down['Class']
X_blc_down_train, X_blc_down_test, y_blc_down_train, y_blc_down_test = train_test_split(X_blc_down, y_blc_down, test_size=1/3, random_state=42)

print("\n======Logistic Regression with balanced data:=======")
LR_classfication = LogisticRegression()
classification(X_blc_down_train, y_blc_down_train, X_blc_down_test, y_blc_down_test, LR_classfication)

print("\n======Naive Bayes (Guassian Naive Bayes) with balanced data:=======")
GNB_classfication = GaussianNB()
classification(X_blc_down_train, y_blc_down_train, X_blc_down_test, y_blc_down_test, GNB_classfication)

print("\n======Decision Tree with balanced data:=======")
DT_classfication = DecisionTreeClassifier(max_depth=3)
classification(X_blc_down_train, y_blc_down_train, X_blc_down_test, y_blc_down_test, DT_classfication)

print("\n======Random Forest with balanced data:=======")
RF_classfication = RandomForestClassifier()
classification(X_blc_down_train, y_blc_down_train, X_blc_down_test, y_blc_down_test, RF_classfication)

#-------------------------------------------------------------------------
print("\n\n\n")
#Training on normalized imbalanced data with outlier detection methods
print("======Training on normalized imbalanced data with outlier detection methods:=======")

def plot_confusion_matrix(y_true, y_pred, classes,
                          title,
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


#Read the original dataset
data1 = pd.read_csv('creditcard.csv')
X = data1.drop(columns='Class')
y = data1['Class']

# Calculate outliers fraction
outliers_fraction = data["Class"].value_counts()[1]/data["Class"].value_counts()[0]

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)



#LOF model
lof = LocalOutlierFactor(novelty = True, contamination=outliers_fraction, n_neighbors = 5)
print("=======LocalOutlierFactor Parameter=======")
print(lof.get_params())

lof.fit(X_train)
pred = -(lof.predict(X_test) - 1)/2
pred = pred.astype(int)
yscore = - lof.decision_function(X_test)
print("=======LocalOutlierFactor Performance=======")
print(classification_report(y_test, pred, target_names=['Genuine', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred, classes=['Genuine', 'Fraud'],title='LocalOutlierFactor' )
print("ROC curve:")
score = plot_ROC(y_test, yscore)
print("The AUC is:" + str(score))
print("\n")


#split data into train and test
train, test = train_test_split(data, test_size=1/3, random_state=42)
train_normal = train[train['Class']==0]
X_train = train_normal.drop(columns='Class')
y_train = train_normal['Class']
X_test = test.drop(columns='Class')
y_test = test['Class']

#One class SVM model
svm = svm.OneClassSVM(gamma = 'auto', nu = outliers_fraction)
print("=======One-Class SVM Parameter:=======")
print(svm.get_params())

svm.fit(X_train)
pred = -(svm.predict(X_test) - 1)/2
pred = pred.astype(int)
yscore = - svm.decision_function(X_test)
print("=======One-Class SVM Performance:=======")
print(classification_report(y_test, pred, target_names=['Genuine', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred, classes=['Genuine', 'Fraud'],title='Confusion matrix of One-Class SVM' )
print("ROC curve:")
score = plot_ROC(y_test, yscore)
print("The AUC is:" + str(score))
print("\n")


#Read the normalized dataset
X = data.drop(columns='Class')
y = data['Class']


#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)


#Isolation forest model
i_f = IsolationForest(max_features=30, n_estimators=100, contamination=outliers_fraction)
print("=======Isolation Forest Parameters:======")
print(i_f.get_params())


i_f.fit(X_train)
pred = -(i_f.predict(X_test) - 1)/2
pred = pred.astype(int)
yscore = - i_f.decision_function(X_test)
print("=======Isolation Forest Performance:=======")
print(classification_report(y_test, pred, target_names=['Genuine', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred, classes=['Genuine', 'Fraud'],title='Confusion matrix of Isolation Forest' )
print("ROC curve:")
score = plot_ROC(y_test, yscore)
print("The AUC is:" + str(score))
print("\n")


#Elliptic Envelipe model
ee = EllipticEnvelope(support_fraction = 1, contamination=outliers_fraction)
print("=======Elliptic Envelope Parameter:=======")
print(ee.get_params())

ee.fit(X_train)
pred = -(ee.predict(X_test) - 1)/2
pred = pred.astype(int)
yscore = - ee.decision_function(X_test)
print("=======Elliptic Envelope Performance:=======")
print(classification_report(y_test, pred, target_names=['Genuine', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred, classes=['Genuine', 'Fraud'],title='Confusion matrix of Elliptic Envelope' )
print("ROC curve:")
score = plot_ROC(y_test, yscore)
print("The AUC is:" + str(score))
print("\n")

