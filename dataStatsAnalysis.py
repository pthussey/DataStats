import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

from collections import defaultdict, Counter


def DiscreteRv(a):
    """Creates a scipy.stats discrete_rv.

    Arguments:
        a {array-like} -- a single data set, will be flattened if it is not already 1-D

    Returns:
        An instance of scipy.stats discrete_rv representing the input data
    """
    val,cnt = np.unique(a, return_counts=True)
    
    return stats.rv_discrete(values=(val,cnt/sum(cnt)))


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


def NormalCdfValues(mean, std, n=1001):
    """Generates sequences of xs and ps to plot a normal distribution cdf model.

    Arguments:
        mean {float} -- mean
        std {float} -- standard deviation

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {1001})

    Returns:
        xs {array} -- The values to use for x in a cdf model plot
        ps {array} -- The values to use for y in a cdf model plot
       
    """
    xmin = mean - 4 * std
    xmax = mean + 4 * std
    xs = np.linspace(xmin, xmax, n)
    ps = stats.norm.cdf(xs, loc=mean, scale=std)

    return xs, ps


def NormalPdfValues(mean, std, n=1001):
    """Generates sequences of xs and ps to plot a normal distribution pdf model.

    Arguments:
        mean {float} -- mean
        std {float} -- standard deviation

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {1001})

    Returns:
        xs {array} -- The values to use for x in a pdf model plot
        ps {array} -- The values to use for y in a pdf model plot
    """
    xmin = mean - 4 * std
    xmax = mean + 4 * std
    xs = np.linspace(xmin, xmax, n)
    ps = stats.norm.pdf(xs, loc=mean, scale=std)
    return xs, ps


def NormalProbabilityValues(a):
    """Creates x and y values to be used in a normal probability plot.

    Arguments:
        a {array-like} -- A single input data set

    Returns:
        sorted_norm {list} -- Sorted random data from the standard normal distribution,
        that is the same length as the input data. To be used as x values.

        sorted_data: {list} -- Sorted input data to be used as y values.
    """
    sorted_norm = sorted(np.random.normal(0,1,len(a)))
    sorted_data = sorted(a)
    
    return sorted_norm, sorted_data


def ParetoCdfValues(xmin, xmax, b, n=50):
    """Generates sequences of xs and ps to plot a Pareto cdf model.

    Arguments:
        xmin {float} -- The minimum possible value for x
        xmax {float} -- The maximum value for x
        b {float} -- The shape parameter, also referred to as alpha

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {50})

    Returns:
        xs {array} -- The values to use for x in a cdf model plot
        ps {array} -- The values to use for y in a cdf model plot
    """
    xs = np.linspace(xmin, xmax, n)
    ps = stats.pareto.cdf(xs, scale=xmin, b=b)
    return xs, ps


def ParetoPdfValues(xmin, xmax, b, n=50):
    """Generates sequences of xs and ps to plot a Pareto pdf model.

    Arguments:
        xmin {float} -- The minimum possible value for x
        xmax {float} -- The maximum value for x
        b {float} -- The shape parameter, also referred to as alpha

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {50})

    Returns:
        xs {array} -- The values to use for x in a pdf model plot
        ps {array} -- The values to use for y in a pdf model plot
    """
    xs = np.linspace(xmin, xmax, n)
    ps = stats.pareto.pdf(xs, scale=xmin, b=b)
    return xs, ps


def ExponentialCdfValues(xmin, xmax, lam, n=50):
    """Generates sequences of xs and ps to plot an exponential cdf model.

    Arguments:
        xmin {float} -- The minimum possible value for x
        xmax {float} -- The maximum value for x
        lam {float} -- The shape parameter, lambda

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {50})

    Returns:
        xs {array} -- The values to use for x in a cdf model plot
        ps {array} -- The values to use for y in a cdf model plot
    """
    xs = np.linspace(xmin, xmax, n)
    ps = stats.expon.cdf(xs, scale=1/lam)
    return xs, ps


def ExponentialPdfValues(xmin, xmax, lam, n=50):
    """Generates sequences of xs and ps to plot an exponential pdf model.

    Arguments:
        xmin {float} -- The minimum possible value for x
        xmax {float} -- The maximum value for x
        lam {float} -- The shape parameter, lambda

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {50})

    Returns:
        xs {array} -- The values to use for x in a pdf model plot
        ps {array} -- The values to use for y in a pdf model plot
    """
    xs = np.linspace(xmin, xmax, n)
    ps = stats.expon.pdf(xs, scale=1/lam)
    return xs, ps


def CohenEffectSize(group1, group2):
    """Computes Cohen's effect size for two groups.

    Arguments:
        group1 -- Series or DataFrame
        group2 -- Series or DataFrame

    Returns:
        A float if the arguments are Series; Series if the arguments are DataFrames
    """
    diff = group1.mean() - group2.mean()

    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d


def MultipleBarValues(*args):
    """Generates a common set of x values and heights for multiple bars to be used in a matplotlib bar plot.

    Arguments:
        Multiple data sets in any array-like format

    Returns:
        bar_xvalues {array} -- Values to be used for x in the bar plot.
        A range of values from minimum to maximum taken from all values in all the data sets.

        results_list {a list of Series} -- The Series in the list are the value counts of the original data sets,
        reindexed to match up with the bar_xvalues. bar_xvalues that do not exist in the data set are NaN.
    """
    # Initiate data set list
    series_list = []
    
    # Change data sets to Series format if needed and append to list
    for x in args:
        if isinstance(x, pd.Series) == False:
            x = pd.Series(x)
        series_list.append(x)
    
    # Find the minimum and maximum values from all values in all data sets
    series_total = pd.concat(series_list)
    total_min = series_total.min()
    total_max = series_total.max()
    
    # Create the x values
    bar_xvalues = np.arange(total_min,total_max+1)
    
    # Initiate the results list and append the value count reindexed Series
    results_list = []
    for x in series_list:
        x_reidx = x.value_counts().reindex(bar_xvalues)
        results_list.append(x_reidx)
    
    return bar_xvalues, results_list


def RvPmfDiffs(rv1,rv2):
    """Computes a shared range of values and the percentage point differences 
    between the pmfs of two scipy.stats discrete_rvs.

    Arguments:
        rv1, rv2 -- two discrete_rvs

    Returns:
        shared_xk {list} -- A list of values from the minimum to maximum 
        among all values from both input rvs

        diffs {list} -- A list of the percentage point differences at each value in shared_xk for the rvs
    """
    # Compute the the shared_xk values
    total = np.concatenate((rv1.xk, rv2.xk))
    xk_min = total.min()
    xk_max = total.max()
    shared_xk = list(range(xk_min, xk_max+1))
        
    # Compute the percentage point differences at each value in shared_xk
    diffs = []
    
    for x in shared_xk:
        p1 = rv1.pmf(x)
        p2 = rv2.pmf(x)
        diff = (p1-p2)*100
        diffs.append(diff)
    
    return shared_xk, diffs


def BiasRv(rv):
    """Computes a biased version of a scipy.stats discrete_rv.
    Replicates the situation in which a survey is asking respondents
    to report the size of a group they belong to.
    """
    new_probs = []
    for x in rv.xk:
        prob = rv.pmf(x)*x
        new_probs.append(prob)
    new_pk = np.array(new_probs)/sum(new_probs)
    return stats.rv_discrete(values=(rv.xk, new_pk))


def UnbiasRv(rv):
    """Computes an unbiased version of a scipy.stats discrete_rv.
    To be used in situations where a survey is asking respondents
    to report the size of a group they are a part of.
    """
    new_probs = []
    for x in rv.xk:
        prob = rv.pmf(x)*1/x
        new_probs.append(prob)
    new_pk = np.array(new_probs)/sum(new_probs)
    return stats.rv_discrete(values=(rv.xk, new_pk))


def PercentileRank(values, x):
    """Computes percentile rank for a certain value (x) from among a set of values.
    """
    count = 0
    values = sorted(values)
    for value in values:
        if value <= x:
            count += 1

    percentile_rank = 100.0 * count / len(values)
    return percentile_rank


def EvalCdf(values, x):
    """Computes the CDF for a certain value (x) from among a set of values.
    This is the same as percentile rank except return value is between 0 and 1.
    """
    count = 0.0
    values = sorted(values)
    for value in values:
        if value <= x:
            count += 1

    prob = count / len(values)
    return prob


def KdeValues(sample, n=101):
    """Generates sequences of x and y values for a kernel density estimation (kde) plot.

    Arguments:
        sample {array-like} -- A single input data set

    Keyword Arguments:
        n {int} -- The number of x values to use (default: {101})

    Returns:
        xs {array} -- The values to use for x in a kde plot
        ys {array} -- The values to use for y in a kde plot
    """
    xs = np.linspace(min(sample), max(sample), n)
    sorted_sample = sorted(sample)
    kde = stats.gaussian_kde(sorted_sample)
    ys = kde.evaluate(xs)
    return xs,ys


def Jitter(values, jitter=0.5):
    """Adds jitter to a scatter plot to remove 'column' effects of rounding for better visualization.

    Args:
        values (array-like): The sequence of values to which jitter will be added.
        jitter (float): The max amount of jitter to add. (default: 0.5)

    Returns:
        numpy array: The array of values with jitter added.
    """
    n = len(values)
    return np.random.normal(0, jitter, n) + values


def CorrelationRandCI(x, y, alpha=0.05, method='pearson'):
    ''' Calculate a correlation coefficient and a correlation confidence interval (CI) for two variables. 
    Uses a parametric approach to calculate the CI. 
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


def SampleRows(df, nrows, replace=False):
    """Generates a random sample of rows from a dataframe. 
    Use replace = True and nrows = len(df) to do a resampling of the dataframe. 
    This is used to estimate sampling error and build CIs by computation.

    Arguments:
        df {dataframe} -- The input dataframe
        nrows {integer} -- The number of rows to sample

    Keyword Arguments:
        replace {bool} -- [Select whether or not to use replacement in sampling] (default: {False})

    Returns:
        sample {dataframe} -- The sample dataframe
    """
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample


def PercentileRow(array, p):
    """Selects the row from a sorted array that maps to percentile p.

    p: float 0--100

    returns: NumPy array (one row)
    """
    rows, _ = array.shape
    index = int(rows * p / 100)
    return array[index,]


def PercentileRows(ys_seq, percents):
    """Given a collection of lines, selects percentiles along vertical axis. 
    This can be used after building a list of sequences using resampling, 
    and then the returned rows can be plotted (fill between) to produce a CI.

    For example, if ys_seq contains simulation results like ys as a
    function of time, and percents contains ([5, 95]), the result would
    be a 90% CI for each vertical slice of the simulation results.

    ys_seq: sequence of lines (y values)
    percents: list of percentiles (0-100) to select

    returns: list of NumPy arrays, one for each percentile
    """
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))

    for i, ys in enumerate(ys_seq):
        array[i,] = ys

    array = np.sort(array, axis=0)

    rows = [PercentileRow(array, p) for p in percents]
    return rows


def ResampleInterSlope(x, y, iters=1000):
    """Uses sampling with replacement to generate intercept and slope sampling distributions for two variables of interest.
    Also generates a sequence of fys to be used when adding a CI to a regression plot.
    Put the fys_seq into PercentileRows to get the low/high lines for plotting.
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
        fys_seq (list) : fys to be used for regression plot CI 
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
    
    fys_seq = []
    for inter, slope in zip(inters, slopes):
        _, fys = FitLine(x, inter, slope)
        fys_seq.append(fys)

    return np.array(inters), np.array(slopes), fys_seq


def VariableMiningOLS(df, y):
    """Searches variables using ordinary least squares regression to find ones that predict the target dependent variable 'y'.

    Args:
        df (DataFrame): DataFrame that holds all the variables
        y (string): Column name of dependent variable y

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
    
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = '{} ~ '.format(y) + name
            
            # The following seems to be required in some environments
            # formula = formula.encode('ascii')

            model = smf.ols(formula, data=df)
            if model.nobs < len(df)/2:
                continue

            results = model.fit()
        except (ValueError, TypeError):
            continue

        variables.append((results.rsquared, name))

    return variables


def VariableMiningLogit(df, y):
    """Searches variables using logistic regression to find ones that predict the target dependent variable 'y'.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y. Must use integer values (ie. 1 for True).

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = '{} ~ '.format(y) + name
            model = smf.logit(formula, data=df)
            nobs = len(model.endog)
            if nobs < len(df)/2:
                continue

            results = model.fit()
        except:
            continue

        variables.append((results.prsquared, name))

    return variables


def VariableMiningPoisson(df, y):
    """Searches variables using Poisson regression to find ones that predict the target dependent variable 'y'.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y.

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = '{} ~ '.format(y) + name
            model = smf.poisson(formula, data=df)
            nobs = len(model.endog)
            if nobs < len(df)/2:
                continue

            results = model.fit()
        except:
            continue

        variables.append((results.prsquared, name))

    return variables


def VariableMiningMnlogit(df, y):
    """Searches variables using multinomial logistic regression to find ones that predict the target dependent variable 'y'.

    Args:
        df (DataFrame): DataFrame that holds all the variables.
        y (string): Column name of dependent variable y.

    Returns:
        variables (list): A list of tuples each containing r-squared value and variable name
    """
    variables = []
    for name in df.columns:
        try:
            if df[name].var() < 1e-7:
                continue

            formula = '{} ~ '.format(y) + name
            model = smf.mnlogit(formula, data=df)
            nobs = len(model.endog)
            if nobs < len(df)/2:
                continue

            results = model.fit()
        except:
            continue

        variables.append((results.prsquared, name))

    return variables


def SummarizeRegressionResults(results):
    """Takes a statsmodels linear regression results object (model.fit()) and 
    prints the most important parts of linear regression results.
    Printed independent variable results are coefficent value and pvalue.

    Args:
        results (statsmodels.regression.linear_model.RegressionResultsWrapper): 
        statsmodels regression results object
    """
    for name, param in results.params.items():
        pvalue = results.pvalues[name]
        print('%s   %0.3g   (%.3g)' % (name, param, pvalue))

    try:
        print('R^2 %.4g' % results.rsquared)
        ys = results.model.endog
        print('Std(ys) %.4g' % ys.std())
        print('Std(res) %.4g' % results.resid.std())
    except AttributeError:
        print('R^2 %.4g' % results.prsquared)


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


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class HypothesisTest():
    """Hypothesis test superclass. 
    """
    def __init__(self, data, tail='right', iters=1000):
        """Initializes the hypothesis test.

        data: data in whatever form is relevant
        """
        self.data = data
        self.tail = tail
        self.iters = iters
        self.PrepareData(data)
        self.TestStat()
        self.rv = self.ComputeRv() # pylint: disable=assignment-from-no-return

    # Provide the functionality to convert the data into the format needed 
    # for use in ComputeRv and Power functions. 
    # Ex. Convert to array, split data into component groups, etc. 
    # The self data variables must be created in the function, not returned. 
    # See child classes for examples
    def PrepareData(self, data):
        UnimplementedMethodException()
        
    # This function only needs to be written in the case of a null hypothesis based test. 
    # The self.test_stat needs to be created in the function, not returned. 
    # In the case of an alternative hypothesis based test 
    # test_stat will be provided via a class parameter. 
    # See child classes for examples
    def TestStat(self):
        pass
    
    # Provide the functionality that computes the sampling distribution and rv for the data. 
    # See child classes for examples
    def ComputeRv(self):
        UnimplementedMethodException()
        
    # Provide the functionality that computes the power by running multiple iterations of the hypothesis test.
    # The code in the for loop must first create new data for the run, 
    # which simulates taking another sample from the population, and then run the hypothesis test.
    # See child classes for examples
    def Power(self):
        UnimplementedMethodException()
    
    def PValue(self):
        """Computes the distribution of the test statistic and p-value.

        returns: float p-value
        """
        if self.tail == 'left':
            pvalue = self.rv.cdf(self.test_stat) # pylint: disable=no-member
        elif self.tail == 'right':
            pvalue = 1 - self.rv.cdf(self.test_stat) # pylint: disable=no-member
        else:
            raise Exception('The value of \'tail\' can only be either \'left\' or \'right\'')

        return pvalue

    def MinMaxTestStat(self):
        """Returns the smallest and largest test statistics in the sampling distribution.
        """
        return min(self.rv.xk), max(self.rv.xk)

    def PlotCdf(self):
        """Draws a Cdf with a vertical line at the test stat.
        """      
        plt.plot(self.rv.xk, self.rv.cdf(self.rv.xk), color='C0', lw=2) # pylint: disable=no-member
        
        plt.axvline(self.test_stat, color='C1', lw=1.3) # pylint: disable=no-member


class HTMean(HypothesisTest):
    """A one-sample mean hypothesis test. 
    A test_stat to represent the null hypothesis must be provided. 
    This test can only produce a onesided pvalue.
    """  
    # For tests that resample to build a distribution for the alternative hypothesis 
    # a test_stat parameter must be included in __init__ to represent the null hypothesis
    # This test_stat is made an attribute of the class below
    def __init__(self, data, test_stat, tail='right', iters=1000):
        self.test_stat = test_stat
        HypothesisTest.__init__(self, data, tail, iters)

    def PrepareData(self, data):
        self.data = np.array(self.data)
        
    def ComputeRv(self):       
        # Build the sampling distribution
        mean_estimates = [np.random.choice(self.data, size=len(self.data), replace=True).
                          mean() for _ in range(self.iters)]

        return DiscreteRv(mean_estimates)

    def Power(self, alpha=0.05, num_runs=1000):        
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run_data that simulates taking another sample from the population
            run_data = np.random.choice(self.data, size=len(self.data), replace=True)
        
            # Run the hypothesis test with run_data
            test = HTMean(run_data, test_stat=self.test_stat, tail=self.tail, iters=100)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTDiffMeansH0(HypothesisTest):
    """A difference of means hypothesis test. 
    Uses permutation to build a sampling distrubition that represents the null hypothesis.
    """
    def __init__(self, data, onesided=False, tail='right', iters=1000):
        self.onesided = onesided
        HypothesisTest.__init__(self, data, tail, iters)

    def PrepareData(self, data):
        a, b = data
        self.a = np.array(a)
        self.b = np.array(b)
        
    # For tests that resample to build a distribution for the null hypothesis 
    # the calculation of the test stat must be provided. 
    # The calculation must create self.test_stat, not return the value.
    # In the case of a twosided test the absolute value of the difference in means is used
    def TestStat(self):      
        if self.onesided == False:
            self.test_stat = abs(self.a.mean() - self.b.mean())
        elif self.onesided == True:
            self.test_stat = self.a.mean() - self.b.mean()
        else:
            raise TypeError('\'onesided\' parameter only accepts Boolean True or False')
        
    def ComputeRv(self):
        # Create the pooled data
        pooled_data = np.hstack((self.a, self.b))
        # Compute the size of a, size of b is just the rest
        a_size = len(self.a)
        
        # Build the sampling distribution
        diff_mean_results = []
        
        if self.onesided == False:
            for _ in range(self.iters):
                np.random.shuffle(pooled_data)
                group1 = pooled_data[:a_size]
                group2 = pooled_data[a_size:]
                result = abs(group1.mean() - group2.mean())
                
                diff_mean_results.append(result)
                    
        elif self.onesided == True:
            for _ in range(self.iters):
                np.random.shuffle(pooled_data)
                group1 = pooled_data[:a_size]
                group2 = pooled_data[a_size:]
                result = group1.mean() - group2.mean()
                
                diff_mean_results.append(result)
                
        else:
            raise TypeError('\'onesided\' parameter only accepts Boolean True or False')
            
        return DiscreteRv(diff_mean_results)
    
    def Power(self, alpha=0.05, num_runs=1000):      
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run_data that simulates taking another sample from the population
            sample1 = np.random.choice(self.a, size=len(self.a), replace=True)
            sample2 = np.random.choice(self.b, size=len(self.b), replace=True)
            run_data = sample1, sample2
        
            # Run the hypothesis test with run_data
            test = HTDiffMeansH0(run_data, onesided=self.onesided, tail=self.tail, iters=100)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTDiffMeansHa(HypothesisTest):
    """A difference of means hypothesis test. 
    Uses resampling to build a sampling distrubition that represents the alternative hypothesis.
    """
    # For tests that resample to build a distribution for the alternative hypothesis 
    # a test_stat parameter must be included in __init__ to represent the null hypothesis
    # This test_stat is made an attribute of the class below
    def __init__(self, data, test_stat, tail='right', iters=1000):
        self.test_stat = test_stat
        HypothesisTest.__init__(self, data, tail, iters)

    def PrepareData(self, data):
        a, b = data
        self.a = np.array(a)
        self.b = np.array(b)

    def ComputeRv(self):      
        # Build the sampling distribution
        diff_mean_results = []
        
        for _ in range(self.iters):
            group1 = np.random.choice(self.a, size=len(self.a), replace=True)
            group2 = np.random.choice(self.b, size=len(self.b), replace=True)
            result = group1.mean() - group2.mean()
                
            diff_mean_results.append(result)
           
        return DiscreteRv(diff_mean_results)
    
    def Power(self, alpha=0.05, num_runs=1000):      
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run data
            sample1 = np.random.choice(self.a, size=len(self.a), replace=True)
            sample2 = np.random.choice(self.b, size=len(self.b), replace=True)
            run_data = sample1, sample2
        
            # Run the hypothesis test with run_data
            test = HTDiffMeansHa(run_data, test_stat=self.test_stat, tail=self.tail, iters=100)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTCorrelationH0(HypothesisTest):
    """A correlation hypothesis test. 
    Uses permutation to build a sampling distrubition that represents the null hypothesis.
    """
    def __init__(self, data, onesided=False, tail='right', iters=1000, method='pearson'):
        self.onesided = onesided
        self.method = method
        HypothesisTest.__init__(self, data, tail, iters)

    def PrepareData(self, data):
        x, y = data
        self.x = np.array(x)
        self.y = np.array(y)
        
    # For tests that resample to build a distribution for the null hypothesis 
    # the calculation of the test stat must be provided.
    # In the case of a twosided test the absolute value of the difference in means is used
    def TestStat(self):
        if (self.onesided == False) & (self.method == 'pearson'):
            self.test_stat = abs(stats.pearsonr(self.x , self.y)[0])
        elif (self.onesided == False) & (self.method == 'spearman'):
            self.test_stat = abs(stats.spearmanr(self.x , self.y)[0])
        elif (self.onesided == True) & (self.method == 'pearson'):
            self.test_stat = stats.pearsonr(self.x , self.y)[0]
        elif (self.onesided == True) & (self.method == 'spearman'):
            self.test_stat = stats.spearmanr(self.x , self.y)[0]
        else:
            raise ValueError('\'onesided\' parameter only accepts Boolean True or False, ' +
                             'and \'method\' only accepts \'pearson\' or \'spearman\'')
        
    def ComputeRv(self):
        # Build the sampling distribution
        corrs = []
        
        if (self.onesided == False) & (self.method == 'pearson'):
            for _ in range(self.iters):
                x_perm = np.random.permutation(self.x)
                r = abs(stats.pearsonr(x_perm , self.y)[0])
                corrs.append(r)
                
        elif (self.onesided == False) & (self.method == 'spearman'):
            for _ in range(self.iters):
                x_perm = np.random.permutation(self.x)
                r = abs(stats.spearmanr(self.x , self.y)[0])
                corrs.append(r)
        
        elif (self.onesided == True) & (self.method == 'pearson'):
            for _ in range(self.iters):
                x_perm = np.random.permutation(self.x)
                r = stats.pearsonr(x_perm , self.y)[0]
                corrs.append(r)
                
        elif (self.onesided == True) & (self.method == 'spearman'):
            for _ in range(self.iters):
                x_perm = np.random.permutation(self.x)
                r = stats.spearmanr(self.x , self.y)[0]
                corrs.append(r)
    
        else:
            raise ValueError('\'onesided\' parameter only accepts Boolean True or False, ' +
                             'and \'method\' only accepts \'pearson\' or \'spearman\'')   
        
        return DiscreteRv(corrs)
        
    def Power(self, alpha=0.05, num_runs=1000):      
        # Build the df to use to make the run data
        df = pd.DataFrame({'x':self.x, 'y': self.y})
        
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run data
            run_df = df.sample(n=len(df), replace=True)
            run_x = run_df.x.values
            run_y = run_df.y.values
            run_data = run_x, run_y
        
            # Run the hypothesis test with run_data
            test = HTCorrelationH0(run_data, onesided=self.onesided, tail=self.tail, 
                                   iters=100, method=self.method)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTCorrelationHa(HypothesisTest):
    """A correlation hypothesis test. 
    Uses permutation to build a sampling distrubition that represents the null hypothesis.
    """
    def __init__(self, data, test_stat, tail='right', iters=1000, method='pearson'):
        self.test_stat = test_stat
        self.method = method
        HypothesisTest.__init__(self, data, tail, iters)

    def PrepareData(self, data):
        x, y = data
        self.x = np.array(x)
        self.y = np.array(y)
        self.df = pd.DataFrame({'x':self.x, 'y': self.y})
              
    def ComputeRv(self):
        # Build the sampling distribution
        corrs = []
                
        if self.method == 'pearson':
            for _ in range(self.iters):
                sample = self.df.sample(n=len(self.df), replace=True)
                r = stats.pearsonr(sample.x, sample.y)[0]
                corrs.append(r)
                
        elif self.method == 'spearman':
            for _ in range(self.iters):
                sample = self.df.sample(n=len(self.df), replace=True)
                r = stats.spearmanr(sample.x, sample.y)[0]
                corrs.append(r)
    
        else:
            raise Exception('Must enter either \'pearson\' or \'spearman\' ' +
                            'as a string for the \'method\' argument')   
        
        return DiscreteRv(corrs)
    
    def Power(self, alpha=0.05, num_runs=1000):    
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run data
            run_df = self.df.sample(n=len(self.df), replace=True)
            run_x = run_df.x.values
            run_y = run_df.y.values
            run_data = run_x, run_y
        
            # Run the hypothesis test with run_data
            test = HTCorrelationHa(run_data, test_stat=self.test_stat, tail=self.tail, 
                                   iters=100, method=self.method)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTChiSquare(HypothesisTest):
    """A chi square hypothesis test. 
    Uses resampling of the expected sequence to simulate the null hypothesis 
    and build the null hypothesis sampling distribution. 
    Takes data in the form of two sequences: data = observed, expected.
    """
    def PrepareData(self, data):
        self.observed, self.expected = data
        self.observed = np.array(self.observed)
        self.expected = np.array(self.expected)
        
        # Check that sum of values are equal
        if np.isclose(sum(self.observed), sum(self.expected)) == False:
            raise ValueError('The sum of the values for observed and expected must be equal.')
        
    def TestStat(self):
        self.test_stat = sum((self.observed - self.expected)**2 / self.expected)
        
    def ComputeRv(self):
        # Calculate the variables needed for resampling        
        n = sum(self.expected)
        values = list(range(len(self.expected)))
        p_exp = self.expected/sum(self.expected)
        
        # Build the sampling distribution
        chis = []
        
        for _ in range(self.iters):          
            hist = Counter({x:0 for x in values}) # Initialize a Counter with zero values
            hist.update(np.random.choice(values, size=n, replace=True, p=p_exp))
            sorted_hist = sorted(hist.items())
            model_observed = np.array([x[1] for x in sorted_hist])
            chi = sum((model_observed - self.expected)**2 / self.expected)
                
            chis.append(chi)
            
        return DiscreteRv(chis)
    
    def Power(self, alpha=0.05, num_runs=1000):    
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create a new run_observed by resampling the observed sequence 
            # Then create the new run_data using run_observed and the original expected sequence
            n = sum(self.observed)
            values_obs = list(range(len(self.observed)))
            p_obs = self.observed/sum(self.observed)
        
            hist = Counter({x:0 for x in values_obs})
            hist.update(np.random.choice(values_obs, size=n, replace=True, p=p_obs))
            sorted_hist = sorted(hist.items())
            run_observed = np.array([x[1] for x in sorted_hist])
            run_data = run_observed, self.expected

            # Run the hypothesis test with run_data
            test = HTChiSquare(run_data, tail=self.tail, iters=100)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


class HTChiSquareContingency(HypothesisTest):
    """A chi square contingency table hypothesis test. 
    Uses resampling of the expected sequence to simulate the null hypothesis 
    and build the null hypothesis sampling distribution. 
    Takes data in the form of a single observed contingency table (array-like).
    """
    def PrepareData(self, data):
        self.observed = data
        self.observed = np.array(self.observed)
        
    # In the case of this class, TestStat also computes self.expected 
    # for later use in the ComputeRv function
    def TestStat(self):
        self.test_stat,_,_,self.expected = stats.chi2_contingency(self.observed)
        
    def ComputeRv(self):
        # Calculate the variables needed for resampling        
        expected_shape = self.expected.shape
        expected_ps = self.expected / np.sum(self.expected)
        values = np.array(list(range(len(self.expected.ravel())))) # Flatten the array
        n= int(np.sum(self.expected))
        
        # Build the sampling distribution
        chis = []
        
        for _ in range(self.iters):          
            # Initiate an empty histogram to hold resampled values
            hist = Counter({x:0 for x in values})
            
            hist.update(np.random.choice(values, size=n, replace=True, p=expected_ps.ravel()))
            sorted_hist = sorted(hist.items())
            resampled_expected = np.array([x[1] for x in sorted_hist])
            
            # Put the array back into its original shape
            resampled_expected_reshaped = resampled_expected.reshape(expected_shape)

            chi = stats.chi2_contingency(resampled_expected_reshaped)[0]                
            chis.append(chi)
            
        return DiscreteRv(chis)
    
    def Power(self, alpha=0.05, num_runs=1000):    
        pvalue_count = 0
        
        for _ in range(num_runs):
            # Create run_data by resampling the observed data    
            observed_shape = self.observed.shape
            observed_ps = self.observed / np.sum(self.observed)
            values = np.array(list(range(len(self.observed.ravel())))) # Flatten the array
            n= int(np.sum(self.observed))
        
            # Initiate an empty histogram to hold resampled values
            hist = Counter({x:0 for x in values})
            
            hist.update(np.random.choice(values, size=n, replace=True, p=observed_ps.ravel()))
            sorted_hist = sorted(hist.items())
            resampled_observed = np.array([x[1] for x in sorted_hist])
            run_data = resampled_observed.reshape(observed_shape) # Put back into original shape

            # Run the hypothesis test with run_data
            test = HTChiSquareContingency(run_data, tail=self.tail, iters=100)
            pvalue = test.PValue()
            
            if pvalue < alpha:
                pvalue_count += 1
            
        return pvalue_count / num_runs


def ChiSquareContribution(obs, exp):
    """Calculates the Chi square contribution for each element in a pair of observed and expected arrays. 
    If using scipy stats.chi2_contingency, can use the expected frequency array returned by that function. 

    Args:
        obs (array-like): The observed frequency array
        exp (array-like): The expected frequency array

    Returns:
        array: Chi square contribution array
    """
    obs_array = np.array(obs)
    exp_array = np.array(exp)
    
    return (obs_array - exp_array)**2/exp_array


def DollarThousandsFormat(value):
    """Formats a value into dollars with a thousands separator. Absolute value applied.

    Args:
        value (int or float): Value to be formatted

    Returns:
        string: formatted value
    """
    return '${:,.0f}'.format(abs(value))


def TrimData(data, limits=None, prop=None):
    """Trims data by either limits on values or a proportion.

    Args:
        data (array-like): A sequence of data
        limits (list or tuple, optional): Must be entered as a list or tuple of len 2. Defaults to None.
        prop (float, optional): The proportion to be trimmed.
        The entered prop will be trimmed from each end.
        Ex. entering 0.05 trims that amount of each end, trimming 10% in total.
        Must enter as a proportion less than 0.5. Defaults to None.

    Returns:
        array: The trimmed array
    """
    data_array = sorted(np.array(data)) # Convert to array and sort values
    
    if (limits == None) & (prop == None):
        raise Exception('Must use either the limits or prop parameter')
    elif (limits != None) & (prop != None):
        raise Exception('Can only use limits or prop parameter, not both')
    else:
        if (limits != None) & (prop == None):
            if type(limits) not in [list, tuple]:
                raise TypeError('limits must be either a list or a tuple of values')      
            else:
                data_trimmed = [x for x in data_array if ((x>limits[0]) & (x<limits[1]))]
        else:
            if prop >= 0.5:
                raise ValueError('prop must be a proportion less than 0.5')
            else:
                trim = int(prop*len(data_array))
                data_trimmed = data_array[trim:-trim]
    return data_trimmed


def main():
    pass

if __name__ == '__main__':
    main()