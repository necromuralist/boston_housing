
# third-party
import pandas
from sklearn.datasets import load_boston

# this code
from boston_housing.common import load_housing_data, CLIENT_FEATURES
from boston_housing.evaluating_model_performance import fit_model

housing_frame = pandas.read_hdf('data/housing_data.h5', 'table')
housing_data = load_housing_data()
model = fit_model(housing_data.features, housing_data.prices)

print( "Final model optimal parameters:", model.best_params_)

sale_price = model.predict(CLIENT_FEATURES)
predicted = sale_price[0] * 1000
actual_median = housing_frame.median_value.median() * 1000
print ("Predicted value of client's home: ${0:,.2f}".format(predicted))
print("Median Value - predicted: ${0:,.2f}".format(actual_median - predicted))