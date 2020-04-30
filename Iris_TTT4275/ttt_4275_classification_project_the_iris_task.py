import numpy as np
import matplotlib.pyplot as plt

# One hot encode data
classmapping = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}


def mse(A, B):
    """ 
    Calculates the mean square error between two numbers, lists
    or matrices.
    """
    return ((A-B)**2).mean(axis=1)


def mse_gradient(t, g, x):
    """ Calculates the MSE gradient. """
    mse_grad = g-t
    g_grad = g * (1-g) # Element wise
    W_grad = x.T

    return np.dot(W_grad, mse_grad*g_grad)


def sigmoid(z):
    """ Simoid function. """
    return 1 / (1 + np.exp(-z))


def predict(W, x):
    """ 
    W - weight matrix,
    x - samples.
    Returns the predicted classes of a set of samples.
    """
    x = sigmoid(np.dot(x, W.T))
    if x.ndim == 1:
        return np.argmax(x)
    else:
        return np.argmax(x, axis=1)


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


def get_data():
    """
    Retrieves data from the iris.data file.
    The samples get shuffled to be on the form
    class 1, class 2, class 3, class 1, ...
    """
    x = []
    t = []
    with open('iris.data', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if line == '':
                continue
            splt = line.split(',')
            properties = [float(i) for i in splt[:-1]]
            classification = splt[-1]
            x.append(np.array(properties))
            t.append(classmapping.get(classification,''))

    proper_x = []
    proper_t = []
    for i in range(50):
        for j in range(3):
            proper_x.append(x[50*j+i])
            proper_t.append(t[50*j+i])

    x = np.array(proper_x)
    t = np.array(proper_t)

    x = np.true_divide(x, np.amax(x)/2) - 1
    # An extra column of ones is inserted to act as the bias
    x = np.insert(x, x.shape[1], 1, axis=1)
    return x, t


def training_loop(x, t, a, iterations):
    """
    x - training data,
    t - true values/classes,
    a - learning rate,'
    iterations - the number of iterations to run.
    Iterates through the given training data and returns a weight matrix W
    as well as the average MSE as a function of iterations.
    """
    W = np.random.normal(0, 1, (3, x.shape[1]))

    mse_vals = []
    for i in range(iterations):
        classifications = np.dot(x, W.T)
        g = sigmoid(classifications)
        W = W - a * mse_gradient(t, g, x).T
        mse_vals.append(mse(t, g).mean())

    return W, mse_vals


def print_confusion(conf):
    """
    Creates a latex table that is pre-formatted for a given confusion matrix.
    """
    classes = ['1', '2', '3']


    print("""\\begin{table}[H]
\\caption{}
\\centering
\\begin{tabular}{|c|lll|}""")
    conf = conf.astype(int)
    print('\\hline\nclass & '+' & '.join(classes) + '\\\\' + '\\hline')
    for i, row in enumerate(conf):
        rw = classes[i]
        for j, elem in enumerate(row):
            rw += ' & '
            if elem == 0:
                rw += '-'
            else:
                rw += str(elem)
        rw += '\\\\'
        if i == 2:
            rw += '\\hline'
        print(rw)
    print("""\\end{tabular}
\\end{table}""")
    print()


def part_1_linear_classifier():
    """
    A function that runs the code for part 1 of the iris task.
    It trains on the first 90 samples at first then it trains on the
    last 90 saples.
    """
    iterations = 100

    # Get data and run training
    x, t = get_data()

    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)
    steps = list(range(iterations))

    # Check results
    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print training results
    print_confusion(confusion_matrix(preds[:90], true[:90], 3))
    print('Correctness:', np.sum(preds[:90]==true[:90])/len(preds[:90]))
    print('\n\n\n\n\n\n\n')

    # Print test results
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))

    # Create and print confusion matrixes
    round1 = confusion_matrix(preds[:90], true[:90], 3)
    round1 = np.concatenate((round1, confusion_matrix(preds[90:], true[90:], 3)), axis=1)
    print_confusion(round1)


    # Retrain with new training and test data
    W, mse_vals2 = training_loop(x[60:], t[60:], 0.2, iterations)
    steps = list(range(iterations))

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print training results
    print('Inverse order')
    print_confusion(confusion_matrix(preds[60:], true[60:], 3))
    print('Correctness:', np.sum(preds[60:]==true[60:])/len(preds[60:]))

    # Print test results
    print_confusion(confusion_matrix(preds[:60], true[:60], 3))
    print('Correctness:', np.sum(preds[:60]==true[:60])/len(preds[:60]))

    # Create and print confusion matrixes
    round2 = confusion_matrix(preds[60:], true[60:], 3)
    round2 = np.concatenate((round2, confusion_matrix(preds[:60], true[:60], 3)), axis=1)
    print_confusion(round2)

    # Plot the average mse development as a function of iterations
    plt.plot(steps, mse_vals1, steps, mse_vals2)
    plt.ylim(0, 0.3)
    plt.show()


def part_2_feature_removal():
    """
    A function that runs the code for part 2 of the iris task.
    Features are removed in the order determined by the histogram.
    """
    # Get data and run training with all 4 features
    iterations = 2000
    x, t = get_data()

    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print correctness and confusion matrixes
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))

    # Create a matrix to show confusion matrices side by side
    combined_confusion = confusion_matrix(preds[90:], true[90:], 3)


    # Remove feature 2 and retrain
    x = np.delete(x, 1, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print correctness and confusion matrixes
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))
    # Append the new confusion matrix
    combined_confusion = np.concatenate((combined_confusion, confusion_matrix(preds[90:], true[90:], 3)), axis=1)


    # Remove feature 1 and retrain
    x = np.delete(x, 0, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print correctness and confusion matrixes
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))
    # Append the new confusion matrix
    combined_confusion = np.concatenate((combined_confusion, confusion_matrix(preds[90:], true[90:], 3)), axis=1)


    # Remove feature 4 and run training again
    x = np.delete(x, 1, axis=1)
    W, mse_vals1 = training_loop(x[:90], t[:90], 0.2, iterations)

    preds = predict(W, x)
    true = np.argmax(t, axis=1)

    # Print correctness and confusion matrixes
    print_confusion(confusion_matrix(preds[90:], true[90:], 3))
    print('Correctness:', np.sum(preds[90:]==true[90:])/len(preds[90:]))
    # Append the new confusion matrix
    combined_confusion = np.concatenate((combined_confusion, confusion_matrix(preds[90:], true[90:], 3)), axis=1)

    # Print the combined confusion matrix
    print(combined_confusion)
    print_confusion(combined_confusion)


if __name__ == '__main__':
    part_1_linear_classifier()
    part_2_feature_removal()
        
