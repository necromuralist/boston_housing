Evaluating Model Performance
============================








.. currentmodule:: boston_housing.evaluating_model_performance

.. .. autosummary::
..    :toctree: api
.. 
..    shuffle_split_data
.. 
.. .. currentmodule:: sklearn.cross_validation
.. .. autosummary::
..    :toctree: api
.. 
..    train_test_split
   
Question 3
----------

*Why do we split the data into training and testing subsets for our model?*

We split the data into training and testing subsets so that we can assess the model using a different data-set than what it
was trained on, thus reducing the likelihood of overfitting the model to the training data and increasing the likelihood that it will generalize to other data.




.. .. currentmodule:: boston_housing.evaluating_model_performance
.. .. autosummary::
..    :toctree: api
.. 
..    performance_metric
.. 
.. .. currentmodule:: sklearn.metrics
.. .. autosummary::
..    :toctree: api
.. 
..    mean_squared_error
   
Question 4
----------

*Which performance metric below did you find was most appropriate for predicting housing prices and analyzing the total error. Why?* - *Accuracy* - *Precision* - *Recall* - *F1 Score* - *Mean Squared Error (MSE)* - *Mean Absolute Error (MAE)*

I chose *Mean Squared Error* as the most appropriate performance metric for predicting housing prices because we are predicting a numeric value (a regression problem) and while Mean Absolute Error could also be used, the MSE emphasizes larger errors more and so I felt it would be preferable.

Step 4 (Final Step)
-------------------




.. .. currentmodule:: boston_housing.evaluating_model_performance
.. .. autosummary::
..    :toctree: api
.. 
..    fit_model
.. 
.. .. currentmodule:: sklearn.tree
.. .. autosummary::
..    :toctree: api
.. 
..    DecisionTreeRegressor
   
Question 5
----------

*What is the grid search algorithm and when is it applicable?*

The `GridSearchCV <http://scikit-learn.org/stable/modules/grid_search.html>`_ algorithm exhaustively works through the parameters it is given to find the parameters that create the best model using cross-validation. Because it is exhaustive it is appropriate when the model-creation is not excessively computationally intensive, otherwise its run-time might be infeasible.

.. .. currentmodule:: sklearn.grid_search
.. 
.. .. autosummary::
..    :toctree: api
.. 
..    GridSearchCV

Question 6
----------


*What is cross-validation, and how is it performed on a model? Why would cross-validation be helpful when using grid search?*

Cross-validation is a method of testing a model by partitioning the data into subsets, with each subset taking a turn as the test set while the data not being used as a test-set is used as the training set. This allows the model to be tested against all the data-points, rather than having some data reserved exclusively as training data and the remainder exclusively as testing data.

Because grid-search attempts to find the optimal parameters for a model, it's advantageous to use the same training and testing data in each case (case meaning a particular permutation of the parameters) so that the comparisons are equitable. One could simply perform an initial train-validation-test split and use this throughout the grid search, but this then risks the possibility that there was something in the initial split that will bias the outcome. By using all the partitions of the data as both test and training data, as cross-validation does, the chance of a bias in the splitting is reduced and at the same time all the parameter permutations are given the same data to be tested against.

.. '
