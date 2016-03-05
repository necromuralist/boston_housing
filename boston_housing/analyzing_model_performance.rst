Analyzing Model Performance
===========================










The two methods used here for analyzing how the model is performing with the data are `Learning Curves` and a `Model Complexity` plot.

Learning Curves
---------------

The `Learning Curves` show how a model's performance changes as it is given more data. In this case four `max_depth` sizes were chosen for comparison.

.. '


.. image:: figures/learning_curves.*
   :align: center
   :scale: 95%



Looking at the model with max-depth of 3, as the size of the training set increases, the training error gradually increases. The testing error initially decreases, then seems to more or less stabilize.

The training and testing plots for the model with max-depth 1 move toward convergence with an error near 50, indicating a high bias (the model is too simple, and the additional data isn't improving the generalization of the model). 

For the model with max-depth 10, the curves haven't converged, and the training error remains near 0, indicating that it suffers from high variance, and should be improved with more data.

Model Complexity
----------------

The `Model Complexity` plot allows us to see how the model's performance changes as the max-depth is increased.

.. '


.. image:: figures/model_complexity.*
   :align: center
   :scale: 95%



As max-depth increases the training error improves, while the testing error decreases up until a depth of 5 and then begins a slight increase as the depth is increased. Based on this I would say that the max-depth of 5 created the model that best generalized the data set, as it minimized the testing error, while the models with greater max-depth parameters likely overfitted the training data.

