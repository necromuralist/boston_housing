
# python standard library
import os
from distutils.util import strtobool

# third party
import numpy
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

city_data = load_boston()
housing_prices = city_data.target
housing_features = city_data.data
DEBUG = strtobool(os.environ.get('DEBUG', 'off'))
IN_PWEAVE = __name__ in ('builtin', '__bultin__')

def shuffle_split_data(X, y, test_size=.3, random_state=0):
    """ 
    Shuffles and splits data into training and testing subsets

    :param:
     - `X`: feature array
     - `y`: target array
     - `test_size`: fraction of data to use for testing
     - `random_state`: seed for the random number generator
    :return: x-train, y-train, x-test, y-test
    """
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=test_size,
                                                         random_state=random_state)
    return X_train, y_train, X_test, y_test

def performance_metric(y_true, y_predict):
    """
    Calculates total error between true and predicted values

    :param:
     - `y_true`: array of target values
     - `y_predict`: array of values the model predicted
    :return: mean_squared_error for the prediction
    """
    return mean_squared_error(y_true, y_predict)

expected = 32.167
tolerance = 0.01
actual = performance_metric(numpy.arange(12), numpy.ones(12))
assert abs(expected - actual) < tolerance

def fit_model(X, y, k=10, n_jobs=1):
    """ 
    Tunes a decision tree regressor model using GridSearchCV

    :param:
     - `X`:  the input data
     - `y`:  target labels y
     - `k`: number of cross-validation folds
     - `n_jobs`: number of parallel jobs to run
    :return: the optimal model
    """

    # Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # Set up the parameters we wish to tune
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    # Make an appropriate scoring function
    scoring_function = make_scorer(mean_squared_error, greater_is_better=False)

    # Make the GridSearchCV object
    reg = GridSearchCV(regressor, param_grid=parameters,
                       scoring=scoring_function, cv=k,
                       n_jobs=n_jobs)

    # Fit the learner to the data to obtain the optimal model with tuned parameters
    reg.fit(X, y)

    # Return the optimal model
    return reg


# Test fit_model on entire dataset
reg = fit_model(housing_features, housing_prices)
if DEBUG:
    print( "Successfully fit a model!")