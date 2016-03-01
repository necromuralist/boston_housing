
# python standard library
from collections import namedtuple

# third party
from sklearn import datasets
from tabulate import tabulate

HousingData = namedtuple("HousingData", 'features prices names'.split())
def load_housing_data():
    """
    Convenience function to get the Boston housing data
    :return: housing_features, housing_prices
    """
    city_data = datasets.load_boston()
    return HousingData(features=city_data.data, prices=city_data.target,
                       names=city_data.feature_names)

CLIENT_FEATURES = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24,
                    680.0, 20.20, 332.09, 12.13]]

class PrinterConstants(object):
    __slots__ = ()
    as_is = '{0}'
    two_digits = '{0:.2f}'
    count = 'Count'
    proportion = 'Proportion'
# end PrinterConstants

class ValueCountsPrinter(object):
    """
    A class to print a value-counts table
    """
    def __init__(self, value_counts,
                 label,
                 format_string=PrinterConstants.as_is,
                 count_or_proportion=PrinterConstants.count):
        """
        :param:
         - `value_counts`: pandas value_counts Series
         - `label`: header-label for the data
         - `format_string`: format string for the count/proportion column
         - `count_or_proportion`: Header for the count/proportion column
        """
        self.value_counts = value_counts
        self.label = label
        self.format_string = format_string
        self.count_or_proportion = count_or_proportion
        self._first_width = None
        self._second_width = None
        self._row_format_string = None
        self._header_string = None
        self._top_separator = None
        self._bottom_separator = None
        self._sum_row = None
        return

    @property
    def first_width(self):
        """
        Width of first column's longest label
        """
        if self._first_width is None:
            self._first_width = len(self.label)
            self._first_width = max(self._first_width,
                                    max(len(str(i))
                                        for i in self.value_counts.index))
        return self._first_width

    @property
    def second_width(self):
        """
        Width of the second column header
        """
        if self._second_width is None:
            self._second_width = len(self.count_or_proportion)
        return self._second_width

    @property
    def row_format_string(self):
        """
        Format-string for the rows
        """
        if self._row_format_string is None:
            self._row_format_string = "{{0:<{0}}} {{1:>{1}}}".format(self.first_width,
                                                                     self.second_width)
        return self._row_format_string

    @property
    def header_string(self):
        """
        First line of the output
        """
        if self._header_string is None:
            self._header_string = self.row_format_string.format(self.label,
                                                                self.count_or_proportion)
        return self._header_string

    @property
    def top_separator(self):
        """
        Separator between header and counts
        """
        if self._top_separator is None:
            self._top_separator = '=' *  (self.first_width + self.second_width + 1)
        return self._top_separator

    @property
    def bottom_separator(self):
        """
        Separator between counts and total
        """
        if self._bottom_separator is None:
            self._bottom_separator = '-' * len(self.top_separator)
        return self._bottom_separator

    @property
    def sum_row(self):
        """
        Final row with sum of count column
        """
        if self._sum_row is None:
            format_string = '{{0}} {{1:>{0}}}'.format(self.second_width)
            sum_value = self.format_string.format(self.value_counts.values.sum())
            self._sum_row = format_string.format(' ' * self.first_width,
                                                 sum_value)
        return self._sum_row

    def __str__(self):
        content = '\n'.join((self.row_format_string.format(value,
                                                           self.format_string.format(self.value_counts.values[index]))
                             for index,value in enumerate(self.value_counts.index)))
        return "{0}\n{1}\n{2}\n{3}\n{4}".format(self.header_string,
                                                self.top_separator,
                                                content,
                                                self.bottom_separator,
                                                self.sum_row)

    def __call__(self):
        """
        Convenience method to print the string
        """
        print(str(self))
# end ValueCountsPrinter

class ValueProportionsPrinter(ValueCountsPrinter):
    """
    Printer for proportion tables
    """
    def __init__(self, value_counts, label,
                 format_string=PrinterConstants.two_digits,
                 count_or_proportion=PrinterConstants.proportion):
        super(ValueProportionsPrinter, self).__init__(value_counts=value_counts,
                                                      label=label,
                                                      format_string=format_string,
                                                      count_or_proportion=count_or_proportion)
        return
# end ValueProportionsPrinter

def print_value_counts(value_counts, header, format_string='{0}'):
    """
    prints the value counts
    :param:
     - `value_counts`: pandas value_counts returned object
     - `header`: list of header names (exactly two)
     - `format_string`: format string for values
    """
    first_width = len(header[0])
    if value_counts.index.dtype == 'object':
        first_width = max(first_width, max(len(i) for i in value_counts.index))
    second_width = len(header[1])
    format_string = "{{0:<{0}}} {{1:>{1}}}".format(first_width, second_width)
    
    header_string = format_string.format(*header)

    top_separator = '=' * (first_width + len(header[1]) + 1)
    separator = '-' * len(top_separator)
    print(header_string)
    print(top_separator)
    for index, value in enumerate(value_counts.index):
        print(format_string.format(value,
                                   format_string.format(value_counts
                                                        .values[index])))
    print(separator)
    print('{0} {1:>{2}}'.format(' ' * first_width,
                                format_string
                                .format(value_counts.values.sum()),
                                second_width))
    return

def print_properties(data_type, values, construction, missing='None', table_format='orgtbl'):
    """
    Prints out the table of properties
    """
    print(tabulate([['Data Type', data_type],
                    ['Values', values],
                    ['Missing Values', missing],
                    ['Construction', "Created from '{0}'".format(construction)]],
                   headers='Property Description'.split(),
                   tablefmt=table_format))