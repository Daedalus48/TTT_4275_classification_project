#ttt_4275_classification_project_classification_of_pronounced_vowels_part_01

import numpy as np
import scipy
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


def get_data():
    """
    Returns data from vowdata_nohead.dat in a numpy array
    """
    data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
    identifiers = data[:, 0]
    data = data[:, 7:16].astype(np.int)
    return data, identifiers


def find_indeces(find, data):
    """
    find - string to search for
    data - iris data array
    returns the indexes of samples where "find" is a substring of the identifier.
    if find is "m" then all male sample indexes will be returned.
    if find is "ae" then all "ae" sample indexes will be returned. 
    """
    return np.flatnonzero(np.core.defchararray.find(data,find)!=-1)


def split_training_test(identifiers, data):
    """
    Splits the data set into a training set and a test set.
    """
    train_idx, test_idx = [], []

    for vowel in vowels:
        idx = find_indeces(vowel, identifiers)
        train_idx.extend(idx[:70])
        test_idx.extend(idx[70:])

    return data[train_idx], identifiers[train_idx], data[test_idx], identifiers[test_idx]


def train_and_predict_GMM_classifier(x, y, x_test, y_test, n_components):

    # Initialize empty arrays
    training_preds = np.zeros((12, x.shape[0]))
    testing_preds = np.zeros((12, x_test.shape[0]))

    for i, vowel in enumerate(vowels):
        # Filter classes
        idx = find_indeces(vowel, y)
        class_samples = x[idx]

        # Greate a GMM
        gmm = GMM(n_components=n_components, covariance_type='diag', reg_covar=1e-4, random_state=0)

        # Train the GMM on the training data
        gmm.fit(class_samples, y[idx])

        # Find the total predicted probability over all the components of the mixture model
        for j in range(n_components):
            N = multivariate_normal(mean=gmm.means_[j], cov=gmm.covariances_[j], allow_singular=True)
            training_preds[i] += gmm.weights_[j] * N.pdf(x)
            testing_preds[i] += gmm.weights_[j] * N.pdf(x_test)
        # This is equivalent to: testing_preds[i] = gmm.score_samples(x_test)

    # The prediction is the class that had the highest probability
    predicted_train = np.argmax(training_preds, axis=0)
    predicted_test = np.argmax(testing_preds, axis=0)

    # Get the true values for the training and test sets
    true_train = np.asarray([i for i in range(12) for _ in range(70)])
    true_test = np.asarray([i for i in range(12) for _ in range(69)])

    # Return confusion matrixes and correctness
    return (confusion_matrix(predicted_train, true_train, 12),
            np.sum(predicted_train==true_train)/len(predicted_train),
            confusion_matrix(predicted_test, true_test, 12),
            np.sum(predicted_test==true_test)/len(predicted_test))


def confusion_matrix(pred, true, num_classes):
    """
    Creates a confusion matrix.
    pred - predicted classes.
    true - true classes.
    num_casses - the number of classes predicted.
    """
    conf = np.zeros((num_classes, num_classes))
    for i in range(len(pred)):
        conf[true[i]][pred[i]] += 1
    return conf.T


def print_confusion(conf):
    """
    Prints a latex formatted confusion matrix
    """
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(vowels) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = vowels[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 11:
            rw += '\\hline'
        print(rw)
    print()


def main():
    data, identifiers = get_data()
    x_train, y_train, x_test, y_test = split_training_test(identifiers, data)

    # The last argument in the function is the number of classes, currently manually changed between runs
    a = train_and_predict_GMM_classifier(x_train, y_train, x_test, y_test, 3)
    print_confusion(a[0])   # Training confusion matrix
    print_confusion(a[2])   # Testing confusion matrix
    print('===')
    print(100*(1-a[1]))     # Training error rate
    print(100*(1-a[3]))     # Testing error rate


if __name__ == '__main__':
    main()
