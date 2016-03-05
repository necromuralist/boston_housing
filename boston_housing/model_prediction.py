
# python standard library
import os
import pickle

# third-party
import matplotlib.pylab as plot
import numpy
import pandas
import seaborn
from sklearn.datasets import load_boston

# this code
from boston_housing.common import load_housing_data, CLIENT_FEATURES
from boston_housing.evaluating_model_performance import fit_model
from boston_housing.common import ValueCountsPrinter, print_image_directive

housing_frame = pandas.read_hdf('data/housing_data.h5', 'table')
housing_data = load_housing_data()
housing_features = housing_data.features
housing_prices = housing_data.prices
seaborn.set_style('whitegrid')
seaborn.set_palette('husl')

# there appears to be a bug that will cause parallel jobs to break
# unless you set the environment variable JOBLIB_START_METHOD to 'forkserver'
# also may not work on some versions of python 2
# -1 means use parallel sub-processes
parallel_jobs = -1

# this will determine the running time overall for this code
repetitions = 1000
model_file = 'pickles/models.pkl'
if not os.path.isfile(model_file):
    models = [fit_model(housing_features, housing_prices, n_jobs=parallel_jobs) for model in range(repetitions)]
    with open(model_file, 'wb') as pickler:
        pickle.dump(models, pickler)
else:
    with open(model_file, 'rb') as unpickler:
        models = pickle.load(unpickler)
params_scores = [(model.best_params_, model.best_score_) for model in models]
parameters = numpy.array([param_score[0]['max_depth'] for param_score in params_scores])
scores = numpy.array([param_score[1] for param_score in params_scores])

best_models = pandas.DataFrame.from_dict({'parameter':parameters, 'score': scores})
x_labels = sorted(best_models.parameter.unique())
figure = plot.figure()
axe = figure.gca()
grid = seaborn.boxplot('parameter', 'score', data = best_models,
                       order=x_labels, ax=axe)
title = axe.set_title("Best Parameters vs Best Scores")
filename = 'best_parameters.png'
print_image_directive(filename, figure)

best_index = numpy.where(scores==numpy.max(scores))
print("   Best Score, {0:.2f}".format(scores[best_index][0]))
print("   max-depth parameter with best score,{0}".format(parameters[best_index][0]))

bin_range = best_models.parameter.max() - best_models.parameter.min()
bins = pandas.cut(best_models.parameter,
                  bin_range)
counts = bins.value_counts()
for bounds in counts.index:
    parameter = bounds.split(',')[0].lstrip('()')
    print('   {0},{1}'.format(int(round(float(parameter))),
                              counts.loc[bounds][0]))

parameter_group = pandas.groupby(best_models, 'parameter')
medians = parameter_group.score.median()
for max_depth in medians.index:
    print('   {0},{1:.2f}'.format(max_depth, medians.loc[max_depth]))

maxes = parameter_group.score.max()
for max_depth in maxes.index:
    print('   {0},{1:.2f}'.format(max_depth, maxes.loc[max_depth]))

best_model = models[best_index[0][0]]
sale_price = best_model.predict(CLIENT_FEATURES)
predicted = sale_price[0] * 1000
actual_median = housing_frame.median_value.median() * 1000
print("   Predicted value of client's home; ${0:,.2f}".format(predicted))
print("   Difference between median and predicted; ${0:,.2f}".format(actual_median - predicted))

filename = 'pickles/confidence_interval.pkl'
if not os.path.isfile(filename):
    alpha = 0.05
    resamples = 10**5
    low, high = ci(housing_data.median_value.values, numpy.median, alpha,
                   resamples, method='bca')
    confidence_interval = {'low': low, 'high': high}
    with open(filename, 'wb') as pickler:
        pickle.dump(confidence_interval, pickler)
else:
    with open(filename, 'rb') as unpickler:
        confidence_interval = pickle.load(unpickler)
print("95% CI [{0:.2f}, {1:.2f}]".format(confidence_interval['low'],
                                         confidence_interval['high']))