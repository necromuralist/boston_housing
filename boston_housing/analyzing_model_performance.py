
# python standard library
import os
from distutils.util import strtobool
import warnings

# third party
import matplotlib.pyplot as plot
import numpy
import seaborn
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor

# this code
from boston_housing.evaluating_model_performance import shuffle_split_data
from boston_housing.evaluating_model_performance import performance_metric
from boston_housing.common import print_image_directive

seaborn.set_style('whitegrid')
DEBUG = strtobool(os.environ.get('DEBUG', 'off'))
city_data = load_boston()
housing_features = city_data.data
housing_prices = city_data.target
X_train, y_train, X_test, y_test = shuffle_split_data(housing_features,
                                                      housing_prices)
feature_length = len(housing_features)
train_length = round(.7 * feature_length)
test_length = round(.3 * feature_length)
assert len(X_train) == train_length, "Expected: {0} Actual: {1}".format(.7 * feature_length, len(X_train))
assert len(X_test) == test_length, "Expected: {0} Actual: {1}".format(int(.3 * feature_length), len(X_test))
assert len(y_train) == train_length
assert len(y_test) == test_length

def learning_curves(X_train, y_train, X_test, y_test):
    """ 
    Calculates performance of several models with varying training data sizes
    Then plots learning and testing error rates for each model 
    """
    if DEBUG:
        print( "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . .")
    # Create the figure window
    fig = plot.figure(figsize=(10,8))

    # We will vary the training set size so that we have 50 different sizes
    sizes = numpy.round(numpy.linspace(1, len(X_train), 50))
    train_err = numpy.zeros(len(sizes))
    test_err = numpy.zeros(len(sizes))

    # Create four different models based on max_depth
    for k, depth in enumerate([1,3,6,10]):
        
        for i, s in enumerate(sizes):
            
            # Setup a decision tree regressor so that it learns a tree with max_depth = depth
            regressor = DecisionTreeRegressor(max_depth = depth)
            
            # Fit the learner to the training data
            regressor.fit(X_train[:s], y_train[:s])

            # Find the performance on the training set
            train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            
            # Find the performance on the testing set
            test_err[i] = performance_metric(y_test, regressor.predict(X_test))

        # Subplot the learning curve graph
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, test_err, lw = 2, label = 'Testing Error')
        ax.plot(sizes, train_err, lw = 2, label = 'Training Error')
        ax.legend()
        ax.set_title('max_depth = %s'%(depth))
        ax.set_xlabel('Number of Data Points in Training Set')
        ax.set_ylabel('Total Error')
        ax.set_xlim([0, len(X_train)])
    
    # Visual aesthetics
    fig.suptitle('Decision Tree Regressor Learning Performances', fontsize=18, y=1.03)
    fig.tight_layout()
    filename = 'learning_curves'
    print_image_directive(filename, fig)

def model_complexity(X_train, y_train, X_test, y_test):
    """ 
    Calculates the performance of the model as model complexity increases.
    Then plots the learning and testing errors rates
    """
    if DEBUG:
        print( "Creating a model complexity graph. . . ")


    # We will vary the max_depth of a decision tree model from 1 to 14
    max_depth = numpy.arange(1, 14)
    train_err = numpy.zeros(len(max_depth))
    test_err = numpy.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth = d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    figure = plot.figure(figsize=(7, 5))
    axe = figure.gca()
    axe.set_title('Decision Tree Regressor Complexity Performance')
    axe.plot(max_depth, test_err, lw=2, label = 'Testing Error')
    axe.plot(max_depth, train_err, lw=2, label = 'Training Error')
    axe.legend()
    axe.set_xlabel('Maximum Depth')
    axe.set_ylabel('Total Error')
    filename = 'model_complexity'
    print_image_directive(filename, figure)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    learning_curves(X_train, y_train, X_test, y_test)

model_complexity(X_train, y_train, X_test, y_test)