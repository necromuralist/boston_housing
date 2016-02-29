
Machine Learning Engineer Nanodegree
====================================

Model Evaluation & Validation
-----------------------------

Project 1: Predicting Boston Housing Prices
-------------------------------------------

Welcome to the first project of the Machine Learning Engineer
Nanodegree! In this notebook, some template code has already been
written. You will need to implement additional functionality to
successfully answer all of the questions for this project. Unless it is
requested, do not modify any of the code that has already been included.
In this template code, there are four sections which you must complete
to successfully produce a prediction with your model. Each section where
you will write code is preceded by a **STEP X** header with comments
describing what must be done. Please read the instructions carefully!

In addition to implementing code, there will be questions that you must
answer that relate to the project and your implementation. Each section
where you will answer a question is preceded by a **QUESTION X** header.
Be sure that you have carefully read each question and provide thorough
answers in the text boxes that begin with "**Answer:**\ ". Your project
submission will be evaluated based on your answers to each of the
questions.

A description of the dataset can be found
`here <https://archive.ics.uci.edu/ml/datasets/Housing>`__, which is
provided by the **UCI Machine Learning Repository**.

Getting Started
===============

To familiarize yourself with an iPython Notebook, **try double clicking
on this cell**. You will notice that the text changes so that all the
formatting is removed. This allows you to make edits to the block of
text you see here. This block of text (and mostly anything that's not
code) is written using
`Markdown <http://daringfireball.net/projects/markdown/syntax>`__, which
is a way to format text using headers, links, italics, and many other
options! Whether you're editing a Markdown text block or a code block
(like the one below), you can use the keyboard shortcut **Shift +
Enter** or **Shift + Return** to execute the code or text block. In this
case, it will show the formatted text.

Let's start by setting up some code we will need to get the rest of the
project up and running. Use the keyboard shortcut mentioned above on the
following code block to execute it. Alternatively, depending on your
iPython Notebook program, you can press the **Play** button in the
hotbar. You'll know the code block executes successfully if the message
*"Boston Housing dataset loaded successfully!"* is printed.

.. code:: python

    # Importing a few necessary libraries
    # python standard library
    import warnings
    
    # third party
    import numpy as np
    import numpy
    import matplotlib.pyplot as pl
    from matplotlib import pylab
    import pandas
    import seaborn
    from sklearn import datasets
    from sklearn.tree import DecisionTreeRegressor
    
    # Make matplotlib show our plots inline (nicely formatted in the notebook)
    %matplotlib inline
    
    # Create our client's feature set for which we will be predicting a selling price
    CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
    
    # Load the Boston Housing dataset into the city_data variable
    city_data = datasets.load_boston()
    
    # Initialize the housing prices and housing features
    housing_prices = city_data.target
    housing_features = city_data.data
    
    print "Boston Housing dataset loaded successfully!"


.. parsed-literal::

    Boston Housing dataset loaded successfully!


.. code:: python

    pylab.rcParams['figure.figsize'] = (10, 8)

.. code:: python

    columns = 'crime big_lots industrial charles_river nox rooms old distance highway_access tax_rate pupil_teacher_ratio blacks lower_status'.split()
    assert len(columns) == housing_features.shape[1], len(columns)
    assert len(CLIENT_FEATURES[0]) == len(columns)

.. code:: python

    client_data = pandas.DataFrame(CLIENT_FEATURES, columns=columns)

.. code:: python

    housing_data = pandas.DataFrame(housing_features, columns=columns)
    housing_data['median_value'] = housing_prices

.. code:: python

    housing_data.describe()




.. parsed-literal::

                crime    big_lots  industrial  charles_river         nox  \
    count  506.000000  506.000000  506.000000     506.000000  506.000000   
    mean     3.593761   11.363636   11.136779       0.069170    0.554695   
    std      8.596783   23.322453    6.860353       0.253994    0.115878   
    min      0.006320    0.000000    0.460000       0.000000    0.385000   
    25%      0.082045    0.000000    5.190000       0.000000    0.449000   
    50%      0.256510    0.000000    9.690000       0.000000    0.538000   
    75%      3.647423   12.500000   18.100000       0.000000    0.624000   
    max     88.976200  100.000000   27.740000       1.000000    0.871000   
    
                rooms         old    distance  highway_access    tax_rate  \
    count  506.000000  506.000000  506.000000      506.000000  506.000000   
    mean     6.284634   68.574901    3.795043        9.549407  408.237154   
    std      0.702617   28.148861    2.105710        8.707259  168.537116   
    min      3.561000    2.900000    1.129600        1.000000  187.000000   
    25%      5.885500   45.025000    2.100175        4.000000  279.000000   
    50%      6.208500   77.500000    3.207450        5.000000  330.000000   
    75%      6.623500   94.075000    5.188425       24.000000  666.000000   
    max      8.780000  100.000000   12.126500       24.000000  711.000000   
    
           pupil_teacher_ratio      blacks  lower_status  median_value  
    count           506.000000  506.000000    506.000000    506.000000  
    mean             18.455534  356.674032     12.653063     22.532806  
    std               2.164946   91.294864      7.141062      9.197104  
    min              12.600000    0.320000      1.730000      5.000000  
    25%              17.400000  375.377500      6.950000     17.025000  
    50%              19.050000  391.440000     11.360000     21.200000  
    75%              20.200000  396.225000     16.955000     25.000000  
    max              22.000000  396.900000     37.970000     50.000000  



Statistical Analysis and Data Exploration
=========================================

In this first section of the project, you will quickly investigate a few
basic statistics about the dataset you are working with. In addition,
you'll look at the client's feature set in ``CLIENT_FEATURES`` and see
how this particular sample relates to the features of the dataset.
Familiarizing yourself with the data through an explorative process is a
fundamental practice to help you better understand your results.

Step 1
------

In the code block below, use the imported ``numpy`` library to calculate
the requested statistics. You will need to replace each ``None`` you
find with the appropriate ``numpy`` coding for the proper statistic to
be printed. Be sure to execute the code block each time to test if your
implementation is working successfully. The print statements will show
the statistics you calculate!

.. code:: python

    # Number of houses in the dataset
    total_houses = housing_features.shape[0]
    
    # Number of features in the dataset
    total_features = housing_features.shape[1]
    
    # Minimum housing value in the dataset
    minimum_price = housing_prices.min()
    
    # Maximum housing value in the dataset
    maximum_price = housing_prices.max()
    
    # Mean house value of the dataset
    mean_price = housing_prices.mean()
    
    # Median house value of the dataset
    median_price = numpy.median(housing_prices)
    
    # Standard deviation of housing values of the dataset
    std_dev = numpy.std(housing_prices)
    
    # Show the calculated statistics
    print "Boston Housing dataset statistics (in $1000's):\n"
    print "Total number of houses:", total_houses
    print "Total number of features:", total_features
    print "Minimum house price:", minimum_price
    print "Maximum house price:", maximum_price
    print "Mean house price: {0:.3f}".format(mean_price)
    print "Median house price:", median_price
    print "Standard deviation of house price: {0:.3f}".format(std_dev)


.. parsed-literal::

    Boston Housing dataset statistics (in $1000's):
    
    Total number of houses: 506
    Total number of features: 13
    Minimum house price: 5.0
    Maximum house price: 50.0
    Mean house price: 22.533
    Median house price: 21.2
    Standard deviation of house price: 9.188


.. code:: python

    axe = seaborn.distplot(housing_data.median_value)
    title = axe.set_title('Median Housing Prices')



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e9023b250>


Question 1
----------

As a reminder, you can view a description of the Boston Housing dataset
`here <https://archive.ics.uci.edu/ml/datasets/Housing>`__, where you
can find the different features under **Attribute Information**. The
``MEDV`` attribute relates to the values stored in our
``housing_prices`` variable, so we do not consider that a feature of the
data.

*Of the features available for each data point, choose three that you
feel are significant and give a brief description for each of what they
measure.*

Remember, you can **double click the text box below** to add your
answer!

.. code:: python

    seaborn.set_style('whitegrid')
    for column in housing_data.columns:
        grid = seaborn.lmplot(column, 'median_value', data=housing_data, size=8)
        axe = grid.fig.gca()
        title = axe.set_title('{0} vs price'.format(column))




.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e671a37d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e6721e4d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e67304c90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e673cdf10>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e6742c110>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e674e6a10>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e67472190>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e6760a850>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e676ec610>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e6766f0d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e679f6c90>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e6797b9d0>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e679d7b10>



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e901de610>


.. parsed-literal::

    /home/charon/.virtualenvs/machinelearning/local/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      if self._edgecolors == str('face'):


CRIM - the per-capita crime rate. INDUS - the proportion of non-retail
business acres per town. LSTAT - the percentage of the population that
is of lower status.

Question 2
----------

| *Using your client's feature set ``CLIENT_FEATURES``, which values
  correspond with the features you've chosen above?*
| **Hint: ** Run the code block below to see the client's data.

.. code:: python

    print CLIENT_FEATURES


.. parsed-literal::

    [[11.95, 0.0, 18.1, 0, 0.659, 5.609, 90.0, 1.385, 24, 680.0, 20.2, 332.09, 12.13]]


.. code:: python

    print(client_data.crime)
    print(client_data.industrial)
    print(client_data.lower_status)


.. parsed-literal::

    0    11.95
    Name: crime, dtype: float64
    0    18.1
    Name: industrial, dtype: float64
    0    12.13
    Name: lower_status, dtype: float64


CRIM : 11.95, INDUS: 18.1, LSTAT: 12.13

Evaluating Model Performance
============================

In this second section of the project, you will begin to develop the
tools necessary for a model to make a prediction. Being able to
accurately evaluate each model's performance through the use of these
tools helps to greatly reinforce the confidence in your predictions.

Step 2
------

In the code block below, you will need to implement code so that the
``shuffle_split_data`` function does the following: - Randomly shuffle
the input data ``X`` and target labels (housing values) ``y``. - Split
the data into training and testing subsets, holding 30% of the data for
testing.

| If you use any functions not already acessible from the imported
  libraries above, remember to include your import statement below as
  well!
| Ensure that you have executed the code block once you are done. You'll
  know if the ``shuffle_split_data`` function is working if the
  statement *"Successfully shuffled and split the data!"* is printed.

.. code:: python

    # Put any import statements you need for this code block here
    from sklearn import cross_validation
    def shuffle_split_data(X, y):
        """ Shuffles and splits data into 70% training and 30% testing subsets,
            then returns the training and testing subsets. """
    
        # Shuffle and split the data
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                             y,
                                                                             test_size=.3,
                                                                             random_state=0)
    
        # Return the training and testing data subsets
        return X_train, y_train, X_test, y_test
    
    
    # Test shuffle_split_data
    X_train, y_train, X_test, y_test = shuffle_split_data(housing_features, housing_prices)
    feature_length = len(housing_features)
    train_length = round(.7 * feature_length)
    test_length = round(.3 * feature_length)
    assert len(X_train) == train_length, "Expected: {0} Actual: {1}".format(.7 * feature_length, len(X_train))
    assert len(X_test) == test_length, "Expected: {0} Actual: {1}".format(int(.3 * feature_length), len(X_test))
    assert len(y_train) == train_length
    assert len(y_test) == test_length
    print "Successfully shuffled and split the data!"



.. parsed-literal::

    Successfully shuffled and split the data!


Question 4
----------

*Why do we split the data into training and testing subsets for our
model?*

So that we can assess the model using a different data-set than what it
was trained on, thus reducing the likelihood of overfitting the model to
the training data and increasing the likelihood that it will generalize
to other data.

Step 3
------

In the code block below, you will need to implement code so that the
``performance_metric`` function does the following: - Perform a total
error calculation between the true values of the ``y`` labels ``y_true``
and the predicted values of the ``y`` labels ``y_predict``.

You will need to first choose an appropriate performance metric for this
problem. See `the sklearn metrics
documentation <http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics>`__
to view a list of available metric functions. **Hint: ** Look at the
question below to see a list of the metrics that were covered in the
supporting course for this project.

| Once you have determined which metric you will use, remember to
  include the necessary import statement as well!
| Ensure that you have executed the code block once you are done. You'll
  know if the ``performance_metric`` function is working if the
  statement *"Successfully performed a metric calculation!"* is printed.

.. code:: python

    # Put any import statements you need for this code block here
    from sklearn.metrics import mean_squared_error
    
    def performance_metric(y_true, y_predict):
        """ Calculates and returns the total error between true and predicted values
            based on a performance metric chosen by the student. """
        
        error = mean_squared_error(y_true, y_predict)
        return error
    
    
    # Test performance_metric
    try:
        total_error = performance_metric(y_train, y_train)
        print "Successfully performed a metric calculation!"
    except:
        print "Something went wrong with performing a metric calculation."


.. parsed-literal::

    Successfully performed a metric calculation!


Question 4
----------

*Which performance metric below did you find was most appropriate for
predicting housing prices and analyzing the total error. Why?* -
*Accuracy* - *Precision* - *Recall* - *F1 Score* - *Mean Squared Error
(MSE)* - *Mean Absolute Error (MAE)*

Mean Squared Error was the most appropriate performance metric for
predicting housing prices because we are predicting a numeric value
(this is a regression problem) and while Mean Absolute Error could also
be used, the MSE emphasizes larger errors more (due to the squaring) and
so is preferable.

Step 4 (Final Step)
-------------------

In the code block below, you will need to implement code so that the
``fit_model`` function does the following: - Create a scoring function
using the same performance metric as in **Step 2**. See the `sklearn
``make_scorer``
documentation <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html>`__.
- Build a GridSearchCV object using ``regressor``, ``parameters``, and
``scoring_function``. See the `sklearn documentation on
GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html>`__.

When building the scoring function and GridSearchCV object, *be sure
that you read the parameters documentation thoroughly.* It is not always
the case that a default parameter for a function is the appropriate
setting for the problem you are working on.

| Since you are using ``sklearn`` functions, remember to include the
  necessary import statements below as well!
| Ensure that you have executed the code block once you are done. You'll
  know if the ``fit_model`` function is working if the statement
  *"Successfully fit a model to the data!"* is printed.

.. code:: python

    # Put any import statements you need for this code block
    from sklearn.metrics import make_scorer
    from sklearn.grid_search import GridSearchCV
    
    def fit_model(X, y):
        """ Tunes a decision tree regressor model using GridSearchCV on the input data X 
            and target labels y and returns this optimal model. """
    
        # Create a decision tree regressor object
        regressor = DecisionTreeRegressor()
    
        # Set up the parameters we wish to tune
        parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    
        # Make an appropriate scoring function
        scoring_function = make_scorer(mean_squared_error)
    
        # Make the GridSearchCV object
        reg = GridSearchCV(regressor, param_grid=parameters, scoring=scoring_function)
    
        # Fit the learner to the data to obtain the optimal model with tuned parameters
        reg.fit(X, y)
    
        # Return the optimal model
        return reg
    
    
    # Test fit_model on entire dataset
    reg = fit_model(housing_features, housing_prices)
    print "Successfully fit a model!"



.. parsed-literal::

    Successfully fit a model!


Question 5
----------

*What is the grid search algorithm and when is it applicable?*

The GridSearchCV algorithm exhaustively works through the parameters it
is given to tune the model. Because it is exhaustive it is appropriate
when the parameters are relatively limited and the model-creation is not
computationally intensive, otherwise its run-time might be infeasible.

Question 6
----------

*What is cross-validation, and how is it performed on a model? Why would
cross-validation be helpful when using grid search?*

Cross-validation is a method of testing a model by partitioning the data
into subsets, with each subset taking a turn as the test set while the
data not being used as a test-set is used as the training set. This
allows the model to be tested against all the data-points, rather than
having some data reserved exclusively as training data and the remainder
exclusively as testing data.

Checkpoint!
===========

You have now successfully completed your last code implementation
section. Pat yourself on the back! All of your functions written above
will be executed in the remaining sections below, and questions will be
asked about various results for you to analyze. To prepare the
**Analysis** and **Prediction** sections, you will need to intialize the
two functions below. Remember, there's no need to implement any more
code, so sit back and execute the code blocks! Some code comments are
provided if you find yourself interested in the functionality.

.. code:: python

    def learning_curves(X_train, y_train, X_test, y_test):
        """ Calculates the performance of several models with varying sizes of training data.
            The learning and testing error rates for each model are then plotted. """
        
        print "Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . ."
        
        # Create the figure window
        fig = pl.figure(figsize=(10,8))
    
        # We will vary the training set size so that we have 50 different sizes
        sizes = np.round(np.linspace(1, len(X_train), 50))
        train_err = np.zeros(len(sizes))
        test_err = np.zeros(len(sizes))
    
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
        fig.show()

.. code:: python

    def model_complexity(X_train, y_train, X_test, y_test):
        """ Calculates the performance of the model as model complexity increases.
            The learning and testing errors rates are then plotted. """
        
        print "Creating a model complexity graph. . . "
    
        # We will vary the max_depth of a decision tree model from 1 to 14
        max_depth = np.arange(1, 14)
        train_err = np.zeros(len(max_depth))
        test_err = np.zeros(len(max_depth))
    
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
        pl.figure(figsize=(7, 5))
        pl.title('Decision Tree Regressor Complexity Performance')
        pl.plot(max_depth, test_err, lw=2, label = 'Testing Error')
        pl.plot(max_depth, train_err, lw=2, label = 'Training Error')
        pl.legend()
        pl.xlabel('Maximum Depth')
        pl.ylabel('Total Error')
        pl.show()

Analyzing Model Performance
===========================

In this third section of the project, you'll take a look at several
models' learning and testing error rates on various subsets of training
data. Additionally, you'll investigate one particular algorithm with an
increasing ``max_depth`` parameter on the full training set to observe
how model complexity affects learning and testing errors. Graphing your
model's performance based on varying criteria can be beneficial in the
analysis process, such as visualizing behavior that may not have been
apparent from the results alone.

.. code:: python

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        learning_curves(X_train, y_train, X_test, y_test)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e67accd50>


.. parsed-literal::

    Creating learning curve graphs for max_depths of 1, 3, 6, and 10. . .


Question 7
----------

*Choose one of the learning curve graphs that are created above. What is
the max depth for the chosen model? As the size of the training set
increases, what happens to the training error? What happens to the
testing error?*

Looking at the model with max-depth of 3, as the size of the training
set increases, the training error gradually increases. The testing error
initially decreases, the seems to more or less stabilize.

Question 8
----------

*Look at the learning curve graphs for the model with a max depth of 1
and a max depth of 10. When the model is using the full training set,
does it suffer from high bias or high variance when the max depth is 1?
What about when the max depth is 10?*

The training and testing plots for the model with max-depth 1 move
toward convergence with an error near 50, indicating a high bias (the
model is too simple, and the additional data isn't improving the
generalization of the model). For the model with max-depth 1, the curves
haven't converged, and the training error remains near 0, indicating
that it suffers from high variance, and should be improved with more
data.

.. code:: python

    model_complexity(X_train, y_train, X_test, y_test)



.. parsed-literal::

    <matplotlib.figure.Figure at 0x7f1e67adaf90>


.. parsed-literal::

    Creating a model complexity graph. . . 


Question 9
----------

*From the model complexity graph above, describe the training and
testing errors as the max depth increases. Based on your interpretation
of the graph, which max depth results in a model that best generalizes
the dataset? Why?*

As max-depth increases the training error improves, while the testing
error decreases up until a depth of 6 and then begins a slight increase
as the depth is increased. Based on this I would say that the max-depth
of 6 created the model that best generalized the dataset, as it
minimized the testing error.

Model Prediction
================

In this final section of the project, you will make a prediction on the
client's feature set using an optimized model from ``fit_model``. *To
answer the following questions, it is recommended that you run the code
blocks several times and use the median or mean value of the results.*

Question 10
-----------

| *Using grid search on the entire dataset, what is the optimal
  ``max_depth`` parameter for your model? How does this result compare
  to your intial intuition?*
| **Hint: ** Run the code block below to see the max depth produced by
  your optimized model.

.. code:: python

    print "Final model optimal parameters:", reg.best_params_


.. parsed-literal::

    Final model optimal parameters: {'max_depth': 1}


The optimal max-depth parameter was 1, which did not match my intuitive
interpretation, as I thought that the more complex models did better,
and specifically that a max-depth of 6 was optimal for this data set.

Question 11
-----------

*With your parameter-tuned model, what is the best selling price for
your client's home? How does this selling price compare to the basic
statistics you calculated on the dataset?*

**Hint: ** Run the code block below to have your parameter-tuned model
make a prediction on the client's home.

.. code:: python

    sale_price = reg.predict(CLIENT_FEATURES)
    print "Predicted value of client's home: {0:.3f}".format(sale_price[0])


.. parsed-literal::

    Predicted value of client's home: 19.934


The predicted value of the client's home is $19,934.

Question 12 (Final Question):
-----------------------------

*In a few sentences, discuss whether you would use this model or not to
predict the selling price of future clients' homes in the Greater Boston
area.*

I don't think that I would use this model. The optimal model with a
max-depth of 1 showed a high-bias and this likely underfits the data.
