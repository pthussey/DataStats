# DataStats
Description
----------

This package serves as a tool box of useful functions and classes for exploratory statistical data analysis in Python. Most of the functions and classes use non-parametric computational methods, such as resampling and permutation, rather than analytical methods.

The datastats package has four modules.
- singlevar: Single variable analysis functions
- multivar: Multiple variable analysis functions
- hypotest: Hypothesis test classes and functions
- plotting: Exploratory data analysis plotting functions

The hypotest module is particularly unique as it has classes for performing various types of hypothesis testing using only non-parametric computational methods. The classes can not only produce p-values but can also compute the statistical power of the test being used and plot a cdf of the sampling distribution.

I started creating the functions and classes in this package while studying Alan Downey's ThinkStats book, so much of functionality in it is inspired by, and based on, code from this book. I chose to build my own functions and classes rather than just use those supplied in the book because going through this process has proven to be a great way to learn the material more deeply. Also it means I don't need to be dependent on using the code the book supplies, and I can continue building new functionality as I need it.

See the provided usage demonstration Jupyter notebooks for examples of how to use the functions and classes defined in each module. Note that the datastats package along with numpy, pandas, scipy.stats, statsmodels, matplotlib, and seaborn are required to run these demonstrations.

Installation
------------

datastats can be installed using the following command:
pip install git+https://github.com/pthussey/DataStats

Version History
---------------

1.0:  
  - Initial version

1.1:  
  - Moved two functions (PercentileRow and PercentileRows) from singlevar to multivar module
  - Added two functions (ListsToDataFrame and DataFrameToLists) to singlevar module
  - Added an ExpectedFromObserved function to hypotest module
  - Made changes ChiSquareContribution and AnovaPostHoc functions in hypotest module to include index and column labels
  - Made some minor changes PlotCdf method in hypotest module classes
  - Removed PairwiseTukeyHsd function since it's not really needed
  - Added NormProbPlot, AllNumVarDensityPlot, AllCatVarCountPlot, and TwoCatProportionPlot functions to plotting module
  - Changed function name from AnnotateLine to AnnotateLineYs in plotting module and made some fixes to the function
  - Added usage demonstration Jupyter notebookes for each module.
