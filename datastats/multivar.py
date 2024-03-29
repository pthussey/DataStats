"""This module contains functions for working with multivariate data. 
These include functions for working with lines of best fit and regression analysis. 
There are also a couple additional functions used in hazard survival analysis.
"""

import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm

from collections import Counter

from .singlevar import DiscreteRv


def FitLine(xs, inter, slope):
    """Generates x and y values to plot a model fitline.

    Arguments:
        xs {array-like} -- sequence of xs (for std normal distribution can just choose values that cover the range of the data (ie.[-5,5]))
        inter {float} -- intercept of the line (for std normal distribution this is the mean of the data)
        slope {float} -- slope of the line (for std normal distribution this is the standard deviation of the data)

    Returns:
        A tuple of numpy arrays (sorted xs, fit ys)
    """
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    
    return fit_xs, fit_ys


def Residuals(xs, ys, inter, slope):
    """Generates a sequence of residuals for a fitline.

    Args:
        xs (array-like): sequence of xs
        ys (array-like): sequence of ys
        inter (float): intercept of the line
        slope (float): slope of the line

    Returns:
        array: a sequence of residuals
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res


def ResidualPercentilePlotData(x, y, n_bins=10):
    """For two variables of interest, generates two sequences of length equal to n_bins: 
    The first is a sequence of mean values for each bin. 
    The second is a sequence of residual value rvs (scipy.stats discrete_rv) for each bin.
    These rvs can then be used to plot cdf at different percentiles: 
    (ie. x_means vs rv.ppf(p) at different values for p, normally .25, .50, .75)
    This plot helps to test fit with a linear model.

    Args:
        x (array-like): x data 
        y (array-like): y data
        n_bins (int, optional): Number of bins to be used. Defaults to 10.

    Returns:
        x_means (array): a sequence of means for each bin 
        res_rvs (array): a sequence of residual value rvs for each bin
    """
    # Calculate the intercept and slope of data
    linreg_result = stats.linregress(x, y)
    inter = linreg_result.intercept
    slope = linreg_result.slope
    
    # Calculate the residuals for each data point
    res = Residuals(x, y,inter, slope)
    
    # Bin the x data
    bins = np.linspace(min(x), max(x), num=n_bins)
    x_bins = pd.cut(x, bins)
    
    # Build a DataFrame to hold everything
    res_df = pd.DataFrame({'x':x, 'y':y, 'res':res, 'x_bins':x_bins})
    
    # Group the data by bins
    res_df_grouped = res_df.groupby('x_bins')
    
    # Get the mean of x for each bin
    x_means = res_df_grouped.x.mean().values
    
    # Build an rv of residual data for each bin
    res_rvs = np.array([DiscreteRv(data.res) for _,data in res_df_grouped])
    
    return x_means, res_rvs


def ResampleInterSlope(x, y, iters=1000):
    """Uses sampling with replacement to generate intercept and slope sampling distributions for two variables of interest.
    Also generates a sequence of fys to be used when adding a CI to a regression plot.
    Put the fit_ys_list into datastats.singlevar.PercentileRows to get the low/high lines for plotting the CI.
    Can also make rvs of the inter/slope distributions to plot cdf, compute p-value of hypothesized values (eg. rv.cdf at 0), 
    and calculate sample distribution mean, std deviation (std error), and confidence interval (rv.interval).
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        x (array-like): x data 
        y (array-like): y data
        iters (int, optional): Number of resampling iterations. Defaults to 1000.

    Returns:
        inters (array): intercept sampling distribution 
        slopes (array): slope sampling distribution
        fit_ys_list (list) : fys to be used for regression plot CI 
    """
    
    # Make a DataFrame to hold the two sequences
    df = pd.DataFrame({'x':x, 'y':y})
    
    # Initialize intercept and slope lists
    inters = []
    slopes = []
    
    # Resample the DataFrame and build lists of intercepts and slopes which are the sampling distributions
    for _ in range(iters):
        sample = df.sample(n=len(df), replace=True)
        x_sample = sample.x
        y_sample = sample.y
        regress_result = stats.linregress(x_sample, y_sample)
        inters.append(regress_result.intercept)
        slopes.append(regress_result.slope)
    
    fit_ys_list = []
    for inter, slope in zip(inters, slopes):
        _, fys = FitLine(x, inter, slope)
        fit_ys_list.append(fys)

    return np.array(inters), np.array(slopes), fit_ys_list


def PercentileRow(array, p):
    """Selects the row from a sorted array that maps to percentile p.

    p: float 0--100

    returns: NumPy array (one row)
    """
    rows, _ = array.shape
    index = int(rows * p / 100)
    return array[index,]


def PercentileRows(ys_seq, percents = [2.5, 97.5]):
    """Given a collection of lines, selects percentiles along vertical axis. 
    This can be used after building a list of sequences using resampling, 
    and then the returned rows can be plotted (fill between) to produce a CI.

    For example, if ys_seq contains simulation results such as lists of y values for fit lines, 
    and percents contains ([2.5, 97.5]), the result would be a 95% CI for the simulation results.

    ys_seq: sequence of lines (y values)
    percents: list of percentiles (0-100) to select, defaults to [2.5, 97.5] for a 95% CI

    returns: list of NumPy arrays, one for each percentile
    """
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))

    for i, ys in enumerate(ys_seq):
        array[i,] = ys

    array = np.sort(array, axis=0)

    percentile_rows = [PercentileRow(array, p) for p in percents]
    return percentile_rows


def CorrelationRandCI(x, y, alpha=0.05, method='pearson'):
    ''' Calculate a correlation coefficient and a correlation confidence interval (CI) for two variables. 
    Uses a parametric approach to calculate the CI, which assumes bivariate normality. 
    A non-parametric CI can be obtained from the rv attribute of the correlation hypothesis test classes.
    
    Args:
        x, y {array-like} -- Input data sets
        alpha {float} -- Significance level (default: 0.05)
        method {string} -- Select 'pearson' or 'spearman' method (default: 'pearson')
   
    Returns:
        r {float} -- The correlation coefficient
        p {float} -- The corresponding p value
        lo, hi {float} -- The lower and upper bounds of the confidence interval
    '''

    if method == 'pearson':
        r, p = stats.pearsonr(x,y)
    elif method == 'spearman':
        r, p = stats.spearmanr(x,y)
    else:
        raise Exception('Must enter either pearson or spearman as a string for method argument')

    r_z = np.arctanh(r)
    stderr = 1 / np.sqrt(len(x) - 3)
    z = stats.norm.ppf(1 - alpha / 2)
    low_z, high_z = r_z - (z * stderr), r_z + (z * stderr)
    low, high = np.tanh((low_z, high_z))
    return r, p, low, high


def VariableMiningOLS(df, response_var, method='pinv'):
    """Performs single variable ordinary least squares regression 
    on each variable in a dataset, with respect to a designated response variable. 
    Potentially relevant variables can then be identified 
    from the returned list of r-squared values for each regression. 
    Variables in the dataset must first be cleaned to ensure no missing values, 
    and if categorical variables will be included among the explanatory variables 
    then those must first be converted to dummy indicator variables.

    Args:
        df (DataFrame): DataFrame that holds all the variables
        response_var (string): Column name of response (dependent) variable
        method (string): Solver method to use, see the statsmodels OLS fit method parameters for options

    Returns:
        variables (list): A sorted list of tuples each containing r-squared value and variable name
    """
    # Initiate the variables results list
    variables = []
    
    for name in df.columns:
        try:
            # Exclude the response variable from the results list
            if name == response_var:
                continue
            
            # Convert variable to a float
            try:
                df = df.astype({name:float})
            
            except ValueError as error:
                print(f'Exception raised: {error}\n' +
                      'Variable will be excluded from mining results.\n' +
                      'If variable is categorical, convert to dummies to include it.')
                continue
                
            # Exclude variables that have extremely low variance
            if df[name].var() < 1e-7:
                continue

            x = df[name]
            y = df[response_var]
            
            x = sm.add_constant(x)
            
            model = sm.OLS(y, x).fit(method=method)
        
        except (TypeError, ValueError):
            print('Exception encountered when running regression on \"{}\" variable.'.format(name))
            continue

        variables.append((model.rsquared, name))

    return sorted(variables, reverse=True)


def VariableMiningLogit(df, response_var, method='newton', maxiter=35):
    """Performs single variable logistic regression on each variable in a dataset, 
    with respect to a designated response variable. 
    Potentially relevant variables can then be identified 
    from the returned list of r-squared values for each regression. 
    Variables in the dataset must first be cleaned to ensure no missing values, 
    and if categorical variables will be included among the explanatory variables 
    then those must first be converted to dummy indicator variables.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y. Must be a binary categorical variable expressed in integer values (ie. 1 for True).
        method (string): Solver method to use. See the statsmodels Logit fit method parameters for options.
        maxiter (int): The maximum number of iterations to perform in the fit method.

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
   # Initiate the variables results list
    variables = []
    
    for name in df.columns:
        try:
            # Exclude the response variable from the results list
            if name == response_var:
                continue
            
            # Convert variable to a float
            try:
                df = df.astype({name:float})
            
            except ValueError as error:
                print(f'Exception raised: {error}\n' +
                      'Variable will be excluded from mining results.\n' +
                      'If variable is categorical, convert to dummies to include it.')
                continue
                
            # Exclude variables that have extremely low variance
            if df[name].var() < 1e-7:
                continue

            x = df[name]
            y = df[response_var]
            
            x = sm.add_constant(x)
            
            model = sm.Logit(y, x).fit(method=method, maxiter=maxiter)
        
        except (TypeError, ValueError):
            print('Exception encountered when running regression on \"{}\" variable.'.format(name))
            continue
            
        # Adds a 'convergence warning' message to mining results
        # to indicate which variables produced this warning
        if not model.mle_retvals['converged']:
            variables.append((model.prsquared, name, 'Convergence warning'))
        else: 
            variables.append((model.prsquared, name))

    return sorted(variables, reverse=True)


def VariableMiningPoisson(df, response_var, method='newton', maxiter=35):
    """Performs single variable Poisson regression on each variable in a dataset, 
    with respect to a designated response variable. 
    Potentially relevant variables can then be identified 
    from the returned list of r-squared values for each regression. 
    Variables in the dataset must first be cleaned to ensure no missing values, 
    and if categorical variables will be included among the explanatory variables 
    then those must first be converted to dummy indicator variables.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y. Must be a 'count' variable.
        method (string): Solver method to use. See the statsmodels Poisson fit method parameters for options.
        maxiter (int): The maximum number of iterations to perform in the fit method.

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
   # Initiate the variables results list
    variables = []
    
    for name in df.columns:       
        try:
            # Exclude the response variable from the results list
            if name == response_var:
                continue
                
            # Convert variable to a float
            try:
                df = df.astype({name:float})
            
            except ValueError as error:
                print(f'Exception raised: {error}\n' +
                      'Variable will be excluded from mining results.\n' +
                      'If variable is categorical, convert to dummies to include it.')
                continue
                
            # Exclude variables that have extremely low variance
            if df[name].var() < 1e-7:
                continue

            x = df[name]
            y = df[response_var]
            
            x = sm.add_constant(x)
            
            model = sm.Poisson(y, x).fit(method=method, maxiter=maxiter)
        
        except (TypeError, ValueError):
            print('Exception encountered when running regression on \"{}\" variable.'.format(name))
            continue
            
        # Adds a 'convergence warning' message to mining results
        # to indicate which variables produced this warning
        if not model.mle_retvals['converged']:
            variables.append((model.prsquared, name, 'Convergence warning'))
        else: 
            variables.append((model.prsquared, name))
            
    return sorted(variables, reverse=True)


def VariableMiningMNLogit(df, response_var, method='newton', maxiter=35):
    """Performs single variable multinomial logistic regression on each variable in a dataset, 
    with respect to a designated response variable. 
    Potentially relevant variables can then be identified 
    from the returned list of r-squared values for each regression. 
    Variables in the dataset must first be cleaned to ensure no missing values, 
    and if categorical variables will be included among the explanatory variables 
    then those must first be converted to dummy indicator variables.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y. Must be categorical variable that can take on at least two values.
        method (string): Solver method to use. See the statsmodels MNLogit fit method parameters for options.
        maxiter (int): The maximum number of iterations to perform in the fit method.

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
   # Initiate the variables results list
    variables = []
    
    for name in df.columns:
        try:
            # Exclude the response variable from the results list
            if name == response_var:
                continue
         
            # Convert variable to a float
            try:
                df = df.astype({name:float})
            
            except ValueError as error:
                print(f'Exception raised: {error}\n' +
                      'Variable will be excluded from mining results.\n' +
                      'If variable is categorical, convert to dummies to include it.')
                continue
                
            # Exclude variables that have extremely low variance
            if df[name].var() < 1e-7:
                continue

            x = df[name]
            y = df[response_var]
            
            x = sm.add_constant(x)
            
            model = sm.MNLogit(y, x).fit(method=method, maxiter=maxiter)
        
        except (TypeError, ValueError):
            print('Exception encountered when running regression on \"{}\" variable.'.format(name))
            continue
            
        # Adds a 'convergence warning' message to mining results
        # to indicate which variables produced this warning
        if not model.mle_retvals['converged']:
            variables.append((model.prsquared, name, 'Convergence warning'))
        else: 
            variables.append((model.prsquared, name))

    return sorted(variables, reverse=True)


def SummarizeOLSRegressionResults(results):
    """Takes a statsmodels OLS regression results object 
    and prints the most important parts of results summary. 
    The printed results include the intercept (const) and its p-value, 
    the coefficients and p-values for each variable, and the R2 value for the model. 
    Additionally, std(ys) and std(res) are also included. 
    std(ys) is the root mean squared error (RMSE) of the response variable values 
    without using any explanatory variables. 
    std(res) is the RMSE of the residuals from the model that uses the explanatory variables. 
    A comparison of std(ys) and std(res) shows the reduction in RMSE, 
    which is helpful for inferring the predictive power of the model.

    Args:
        results : A statsmodels OLS regression results object (model.fit()).
    """

    if type(results._results).__name__ == 'OLSResults':
        pass
 
    else:
        raise Exception('Unsupported regression results object used. ' +
                        'Only OLS regression results are supported.' +
                        'For all other regression result types use the .summary() method.')

    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print('%s   coef : %0.3g   p-value: (%.3g)' % (name, param, pvalue))
        
    print('R^2 %.4g' % results.rsquared)
    print('Std(ys) %.4g' % results.model.endog.std())
    print('Std(res) %.4g' % results.resid.std())


def HazardValues(rv):
    """Takes a scipy.stats discrete rv and generates the x and y sequences 
    to plot a hazard function for the data.

    Args:
        rv (scipy.stats discrete_rv): An rv representing the data

    Returns:
        xs (array): The xk values of the data
        ys (array): The hazard function values
    """
    hazards=[]
    
    for k in rv.xk[:-1]:
        hazard = (rv.sf(k) - rv.sf(k+1)) / rv.sf(k)
        hazards.append(hazard)
    
    return rv.xk[:-1], np.array(hazards)


def EstimateHazardValues(duration, event_observed, verbose=False):
    """Estimates the hazard function by Kaplan-Meier. 
    The returned series can be used to plot the hazard function (s.index versus s.values)

    Args:
        duration (array-like): list of time durations until event (or up until time of measure if ongoing)
        event_observed (array-like): list indicating whether event was observed or not (1,0)
        verbose (bool, optional): Whether to display intermediate results. Defaults to False.

    Returns:
        lams (pandas Series): A Series with index of durations and corresponding hazard values
    """
    if np.sum(np.isnan(duration)):
        raise ValueError("duration contains NaNs")
    if np.sum(np.isnan(event_observed)):
        raise ValueError("event_observed contains NaNs")

    # Create a DataFrame to hold both the duration and event data
    df = pd.DataFrame({'duration' : duration, 'event' : event_observed})
        
    hist_complete = Counter(df[df.event == 1].duration)
    hist_ongoing = Counter(df[df.event == 0].duration)

    ts = list(hist_complete | hist_ongoing)
    ts.sort()

    at_risk = len(df)

    lams = pd.Series(index=ts)
    for t in ts:
        ended = hist_complete[t]
        censored = hist_ongoing[t]

        lams[t] = ended / at_risk
        if verbose:
            print(t, at_risk, ended, censored, lams[t])
        at_risk -= ended + censored

    return lams


def main():
    pass

if __name__ == '__main__':
    main()
