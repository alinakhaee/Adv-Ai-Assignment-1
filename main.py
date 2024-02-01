import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'Quadratic': QuadraticDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB(),
    'Linear SVM': SVC(kernel='linear'),
    'Poly SVM': SVC(kernel='poly'),
    'RBF SVM': SVC(kernel='rbf')
}

datasets = ['circles0.3', 'moons1', 'spiral1', 'twogaussians33', 'twogaussians42', 'halfkernel']

def plot_dataset(X, y, title):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', marker='o', label='Class 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', marker='s', label='Class 1')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.legend()
    # plt.savefig('.\\graphs\\' + title + '.png', bbox_inches='tight')
    plt.show()

def load_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    coordinates = data[:, :-1]
    labels = data[:, -1]
    return coordinates, labels

def calculate_performance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    ppv = precision_score(y_true, y_pred)
    npv = tn / (tn + fn)
    accuracy = accuracy_score(y_true, y_pred)
    return ppv, npv, specificity, sensitivity, accuracy



for dataset in datasets:
    file_path = f'..\\SampleDatasets\\{dataset}.csv'
    coordinates, labels = load_dataset(file_path)

    print(f"Dataset: {dataset}")
    for clf_name, clf in classifiers.items():
        y_pred = cross_val_predict(clf, coordinates, labels, cv=10)
        ppv, npv, specificity, sensitivity, accuracy = calculate_performance(labels, y_pred)

        print(f"\nClassifier: {clf_name}")
        print(f"PPV: {ppv:.4f}")
        print(f"NPV: {npv:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        plot_dataset(coordinates, y_pred, f'{dataset} - Samples with {clf_name}')