Model Prediction
================





To find the 'best' model I ran the `fit_model` function 1,000 times and took the `best_params_` (max-depth) and `best_score_` (negative MSE) for each trial.




.. image:: figures/best_parameters.png.*
   :align: center
   :scale: 95%



.. csv-table: Best Score
   :header: Description, Value

   Best Score, -30.46
   max-depth parameter with best score,5



.. csv-table:: Parameter Counts
   :header: Max-Depth, Count

   4,315
   5,190
   7,166
   6,136
   8,111
   9,82



.. csv-table:: Median Scores
   :header: Max-Depth, Median Score

   4,-34.44
   5,-32.54
   6,-32.55
   7,-32.83
   8,-32.80
   9,-32.94
   10,-33.54



.. csv-table:: Max Scores
   :header: Max-Depth, Max Score

   4,-34.35
   5,-30.46
   6,-30.67
   7,-30.88
   8,-30.93
   9,-30.79
   10,-31.32



.. note:: Since the `GridSearchCV` normally tries to maximize the output of the scoring-function, but the goal in this case was to minimize it, the scores are negations of the MSE, thus the higher the score, the lower the MSE.

While a max-depth of 4 was the most common best-parameter, the max-depth of 5 was the median max-depth, had the highest median score, and had the highest overall score, so I will say that the optimal `max_depth` parameter is 5. This is in line with what I had guessed, based on the Complexity Performance plot.

Predicting the Client's Price
-----------------------------

Using the model that had the lowest MSE (30.46) out of the 1,000 generated, I then made a prediction for the price of the client's house.

.. csv-table:: Predicted Price
   :delim: ;

   Predicted value of client's home; $20,967.76
   Difference between median and predicted; $232.24



My three chosen features (`lower_status`, `nitric_oxide`, and `rooms`) seemed to indicate that the client's house might be a lower-valued house, and the predicted value was about $232 less than the median median-value, so it appears that our model predicts that the client has a below-median-value house.

.. '

Confidence Interval
~~~~~~~~~~~~~~~~~~~

Although this isn't an inferential analysis, I'll calculate the 95% Confidence Interval for the median-value so that I'll have a range to compare the prediction to. Since the data isn't symmetric I'll use a bootstrapped confidence interval (bias-corrected and accelerated (BCA))of the median instead of one based on the standard error.

.. '


95% CI [20.40, 21.75]



Our prediction for the client's house falls within a 95% confidence interval for the median, so although I predicted that it would be below the median, there's insufficient evidence to conclude that it differs from the median house price.

Assessing the Model
-------------------

I think that this model seems reasonable for the given data (Boston Suburbs in 1970), but I think that I might be hesitant to predict the value for a specific house using it, given that we are using aggregate-values for entire suburbs, not values for individual houses. I would also think that separating out the upper-class houses would give a better model for certain clients, given the right-skew of the data. Also, the median MSE for the best model was ~32 so taking the square root of this gives an 'average' error of about $5,700, which seems fairly high, given the low median-values for the houses. I think that the model gives a useful ball-park-figure estimate, but I think I'd have to qualify the certainty of prediction for future clients, noting also the age of the data and not extrapolating much beyond 1970.
