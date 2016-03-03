Model Prediction
================





Question 10
-----------

*Using grid search on the entire data set, what is the optimal  ``max_depth`` parameter for your model? How does this result compare to your initial intuition?*

To find the 'best' model I ran the `fit_model` function 1,000 times and took the `best_params_` (max-depth) and `best_score_` (negative MSE) for each trial.




.. image:: figures/best_parameters.png.*
   :align: center
   :scale: 95%



.. csv-table: Best Score
   :header: Description, Value

   Best Score, -30.55
   max-depth parameter with best score,5



.. csv-table:: Parameter Counts
   :header: Max-Depth, Count

   4,295
   5,231
   7,161
   6,141
   8,102
   9,70



.. csv-table:: Median Scores
   :header: Max-Depth, Median Score

   4,-34.58
   5,-32.48
   6,-32.54
   7,-33.25
   8,-32.98
   9,-33.07
   10,-33.31



.. csv-table:: Max Scores
   :header: Max-Depth, Max Score

   4,-34.35
   5,-30.55
   6,-30.59
   7,-31.62
   8,-30.56
   9,-30.77
   10,-31.62



.. note:: Since the `GridSearchCV` normally tries to maximize the output of the scoring-function, but the goal in this case was to minimize it, the scores are negations of the MSE, thus the higher the score, the lower the MSE.

While a max-depth of 4 was the most common best-parameter, the max-depth of 5 was the median max-depth, had the highest median score, and had the highest overall score, so I will say that the optimal `max_depth` parameter is 5. This is in line with what I had guessed, based on the Complexity Performance plot.

Question 11
-----------

*With your parameter-tuned model, what is the best selling price for your client's home? How does this selling price compare to the basic statistics you calculated on the dataset?*

.. '

.. csv-table:: Predicted Price
   :delim: ;

   Predicted value of client's home; $20,967.76
   Median for all suburbs - predicted; $232.24



My three chosen features (`lower_status`, `nitric_oxide`, and `rooms`) seemed to indicate that the client's house might be a lower-valued house, and the predicted value was about $232 less than the median median-value, so our model predicts that the client has a below-median-value house.

.. '

Question 12
-----------

*In a few sentences, discuss whether you would use this model or not to predict the selling price of future clients' homes in the Greater Boston area.*

.. '

I think that this model seems reasonable for the given data (Boston Suburbs in 1970), but I think that I might be hesitant to predict the value for a specific house using it, given that we are using aggregate-values for entire suburbs, not values for individual houses. I would also think that separating out the upper-class houses would give a better model for certain clients, given the right-skew of the data. Also, the median MSE for the best model was ~32 so taking the square root of this gives an 'average' error of about $5,700, which seems fairly high, given the low median-values for the houses. I think that the model gives a useful ball-park-figure estimate, but I think I'd have to qualify the certainty of prediction for future clients, noting also the age of the data and not extrapolating much beyond 1970.
