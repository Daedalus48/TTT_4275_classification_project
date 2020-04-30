#ttt_4275_classification_project_classification_of_pronounced_vowels_part_01

import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


def get_data():
    data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
    ident = data[:, 0]
    data = data[:, 7:16].astype(np.int)
    return data, ident

def split_training_test(ident, data):
    train_idx, test_idx = [], []

    for vowel in vowels:
        idx = find_indeces(vowel, ident)
        train_idx.extend(idx[:70])
        test_idx.extend(idx[70:])

    return data[train_idx], ident[train_idx], data[test_idx], ident[test_idx]


def find_indeces(find, data):
    return np.flatnonzero(np.core.defchararray.find(data,find)!=-1)


def gen_mean_covariance(x):
    sample_mean = x.mean(axis=0)
    temp = class_samples - sample_mean
    covariance_matrix = np.dot(temp.T, temp)/(len(x)-1)    
    return sample_mean, covariance_matrix

def train_and_predict_single_Gaussian_model(x, y, x_test, y_test, diag=False):

    # empty arrays
    class_probabilities = np.empty((12, x.shape[0]))
    test_probabilities = np.empty((12, x_test.shape[0]))

    for i, vowel in enumerate(vowels):

        # Filter classes to only be one vowel
        idx = find_indeces(vowel, y)
        class_samples = x[idx]

        # Sample mean
        sample_mean = class_samples.mean(axis=0)

        # Covariance matrix
        subt = class_samples - sample_mean
        cov1 = np.dot(subt.T, subt)/(len(class_samples)-1)

        # Make matrix diagonal if diag is True
        if diag:
            cov1 = np.diag(np.diag(cov1))

        # Create a multivariate normal distribution
        rv = multivariate_normal(mean=sample_mean, cov=cov1)

        # Classify all samples with the given distribution
        class_probabilities[i] = rv.pdf(x)
        test_probabilities[i] = rv.pdf(x_test)

    # The prediction is the class that had the highest probability
    predicted_train = np.argmax(class_probabilities, axis=0)
    predicted_test = np.argmax(test_probabilities, axis=0)

    # Get the true values for the training and test sets
    true_train = np.asarray([i for i in range(12) for _ in range(70)])
    true_test = np.asarray([i for i in range(12) for _ in range(69)])

    # Return confusion matrixes and correctness
    return (confusion_matrix(predicted_train, true_train, 12),
            np.sum(predicted_train==true_train)/len(predicted_train),
            confusion_matrix(predicted_test, true_test, 12),
            np.sum(predicted_test==true_test)/len(predicted_test))


def confusion_matrix(predicted, real, counter):
    conf = np.zeros((counter, counter))
    for i in range(len(predicted)):
        conf[real[i]][predicted[i]] += 1
    return conf.T


def print_confusion(conf):
    conf = conf.astype(int)
    print('class '+'  '.join(vowels) )
    for i, row in enumerate(conf):
        rw = vowels[i]
        for j, elem in enumerate(row):
            rw += '   '
            rw += str(elem)
        print(rw)
    print()


def main():
    # Get data
    data, identifiers = get_data()

    # Split into training and test data
    x_train, y_train, x_test, y_test = split_training_test(identifiers, data)

    # Train and predic the classes using full covariance
    a = train_and_predict_single_Gaussian_model(x_train, y_train, x_test, y_test, diag=False)
    print_confusion(a[0])   # Training confusion matrix
    print_confusion(a[2])   # Testing confusion matrix
    print('===')
    print(100*(1-a[1]))     # Training error rate
    print(100*(1-a[3]))     # Testing error rate

    # Train and predic the classes using diagonal covariance
    a = train_and_predict_single_Gaussian_model(x_train, y_train, x_test, y_test, diag=True)
    print_confusion(a[0])   # Training confusion matrix
    print_confusion(a[2])   # Testing confusion matrix
    print('===')
    print(100*(1-a[1]))     # Training error rate
    print(100*(1-a[3]))     # Testing error rate


if __name__ == '__main__':
    main()
