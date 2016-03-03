
# third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
import numpy
import pandas
import seaborn
import statsmodels.api as statsmodels

# this code
from boston_housing.common import load_housing_data, CLIENT_FEATURES
from boston_housing.common import print_image_directive

seaborn.set_style('whitegrid')
seaborn.color_palette('hls')

housing_features, housing_prices, feature_names = load_housing_data()
housing_data = pandas.DataFrame(housing_features, columns=feature_names)
housing_data['median_value'] = housing_prices

new_columns =  ('crime_rate',
                'large_lots',
                'industrial',
                'charles_river',
                'nitric_oxide',
                'rooms',
                'old_houses',
                'distances',
                'highway_access',
                'property_taxes',
                'pupil_teacher_ratio',
                'proportion_blacks',
                'lower_status')
old_names = ('CRIM',
             'ZN',
             'INDUS',
             'CHAS',
             'NOX',
             'RM',
             'AGE',
             'DIS',
             'RAD',
             'TAX',
             'PTRATIO',
             'B',
             'LSTAT')
re_map_names = dict(zip(new_columns, old_names))

for new_key, old_key in re_map_names.iteritems():
    housing_data[new_key] = housing_data[old_key]
client_features = pandas.DataFrame(CLIENT_FEATURES, columns=new_columns)

housing_data.to_hdf('data/housing_data.h5', 'table')
client_features.to_hdf('data/client_features.h5', 'table')

for index, old_name in enumerate(old_names):
    print("   {0},{1}".format(old_name, new_columns[index]))

description = housing_data.describe()

print("   {0},{1}".format(len(housing_prices),
                          len(feature_names)))

for item in description.index:
    formatter = "   {0},{1:.0f}" if item == 'count' else '   {0},{1:.2f}'
    print(formatter.format(item, description.median_value.loc[item]))

filename = 'median_value_distribution'
figure = plot.figure()
axe = figure.gca()
grid = seaborn.distplot(housing_data.median_value, ax=axe)
axe.axvline(housing_data.median_value.mean(), label='mean')
axe.axvline(housing_data.median_value.median(), label='median',
            color=seaborn.xkcd_rgb['medium green'])
axe.legend()
title = axe.set_title("Boston Housing Median Values")
print_image_directive(filename, figure, scale='95%')

filename = 'median_value_boxplots'
figure = plot.figure()
axe = figure.gca()
grid = seaborn.boxplot(housing_data.median_value, ax=axe)
title = axe.set_title("Boston Housing Median Values")
print_image_directive(filename, figure, scale='95%')

def qqline_s(ax, x, y, dist, fmt='r-', **plot_kwargs):
    """
    plot qq-line (taken from statsmodels.graphics.qqplot)

    :param:
     - `ax`: matplotlib axes
     - `x`: theoretical_quantiles
     - `y`: sample_quantiles
     - `dist`: scipy.stats distribution
     - `fmt`: format string for line
     - `plot_kwargs`: matplotlib 2Dline keyword arguments
    """
    m, b = y.std(), y.mean()
    reference_line = m * x + b
    ax.plot(x, reference_line, fmt, **plot_kwargs)
    return

filename = 'median_value_qqplot'
figure = plot.figure()
axe = figure.gca()
color_map = plot.get_cmap('Blues')
prob_plot = statsmodels.ProbPlot(housing_data.median_value)
prob_plot.qqplot(ax=axe, color='b', alpha=.25)

qqline_s(ax=axe, dist=prob_plot.dist,
         x=prob_plot.theoretical_quantiles, y=prob_plot.sample_quantiles,
         fmt='-', color=seaborn.xkcd_rgb['medium green'])
         #color=(.33, .66, .27))

title = axe.set_title("Boston Housing Median Values (QQ-Plot)")
print_image_directive(filename, figure)

filename = 'median_value_cdf'
figure = plot.figure()
axe = figure.gca()
grid = plot.plot(sorted(housing_data.median_value), numpy.linspace(0, 1, housing_data.median_value.count()))
title = axe.set_title("Boston Housing Median Values (CDF)")
axe.set_xlabel("Median Home Value in $1,000's")
print_image_directive(filename, figure)

percentile_90 = housing_data.quantile(.90).median_value

def summary_table(variables, title='Variables Summaries',
                  number_format="{0:.2f}", data=housing_data):
    """
    Print a csv-table with variable summaries
    :param:
     - `variables`: collection of variables to summarize
     - `title`: Title for the table
     - `number_format`: format string to set decimals
     - `data`: source data to summarize
    """
    statistics = ('min', '25%', '50%', '75%', 'max', 'mean', 'std')
    print(".. csv-table:: {0}".format(title))
    print("   :header: Variable, Min, Q1, Median, Q3, Max, Mean, Std\n")
    for variable in variables:
        description = data[variable].describe()
        stats = ','.join([number_format.format(description.loc[stat])
                          for stat in statistics])
        print("   {0},{1}".format(variable, stats))                                                   
    return

features = re_map_names.keys()
rows = (len(features) // 3)
slice_start = 0

for row in range(1, rows + 1):
    filename = 'housing_data_regression_plots_{0}.png'.format(row)
    grid = seaborn.PairGrid(housing_data, x_vars=features[slice_start:row * 3], y_vars=['median_value'])
    grid.map(seaborn.regplot)
    print_image_directive(filename, grid)
    slice_start = row * 3

if rows % 3:
    print()
    filename = 'figures/housing_data_regression_plots_{0}.png'.format(row + 1)
    grid = seaborn.PairGrid(housing_data, x_vars=features[slice_start:slice_start + rows % 3], y_vars=['median_value'])
    grid.map(seaborn.regplot, ci=95)
    grid.savefig(filename)
    print('.. image:: {0}'.format(filename))

chosen_variables = ('lower_status', 'nitric_oxide', 'rooms')
for variable in chosen_variables:
    print("    {0},{1:.2f}".format(variable, client_features[variable][0]))

summary_table(chosen_variables)