# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

ROOT = '../data_hw2'  # root directory of this code


def main():
    # Relevant files
    datafile = os.path.expanduser(os.path.join(ROOT, "trainingdata.txt"))
    labelfile = os.path.expanduser(os.path.join(ROOT, "traininglabels.txt"))
    attributesfile = os.path.expanduser(os.path.join(ROOT, "attributes.txt"))
    testdatafile = os.path.expanduser(os.path.join(ROOT,"testingdata.txt"))
    testlabelfile = os.path.expanduser(os.path.join(ROOT, "testinglabels.txt"))
    
    # Load data from relevant files
    train_values = np.loadtxt(datafile, delimiter = ",") # Training data (floats separated by commas)
    train_labels = np.loadtxt(labelfile, dtype = int) # Training labels (either a 0 or 1)
    attributes = np.loadtxt(attributesfile, dtype = str) # Names of attributes
    test_values = np.loadtxt(testdatafile, delimiter = ",") # Testing data (floats separated by commas)
    test_labels = np.loadtxt(testlabelfile, dtype = int) # Testing labels (either a 0 or 1)

    pdb.set_trace()
    # Train a decision tree via information gain on the training data
    tree = DecisionTreeClassifier(criterion = 'entropy')
    tree.fit(train_values, train_labels)

    # Test the decision tree
    print('\nPrediction for the testing data:')
    print(tree.predict(test_values))

    # Show the confusion matrix for test data
    print('\nConfusion Matrix:')
    print(confusion_matrix(test_labels, tree.predict(test_values)))

    # Compare training and test accuracy
    print('\nAccuracies:')
    print(f'Training Accuracy: {round(tree.score(train_values, train_labels) * 100, 2)}%')
    print(f'Testing Accuracy: {round(tree.score(test_values, test_labels)* 100, 2)}%')

    # Visualize the tree using matplotlib and plot_tree
    plot_tree(tree, feature_names = attributes, class_names = ['Healthy', 'Parkinson\'s'], filled = True, rounded = True, fontsize = 7)
    plt.show()

    # pdb.set_trace()


if __name__ == '__main__':
    main()
