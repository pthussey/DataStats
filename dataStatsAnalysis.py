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
    A non-parametric CI can be calculated from the results of the ResampleCorrelation_Ha function.
    
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


def ResampleMean(data, weights=None, iters=1000):
    """Uses sampling with replacement to generate a sampling distribution of mean for a variable.
    Can then make an rv of the sampling distribution to calculate the sampling distribution mean, 
    std deviation (std error), and confidence interval (rv.interval). 
    Using the rv, can also plot the cdf and compute the one-sided p-value of a hypothesized mean (eg. rv.cdf at 0). 
    A two-sided p-value can be obtained by doubling the one-sided p-value if appropriate for the particular case.
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        data (array-like): Data for the variable of interest
        weights (array-like, optional): Can include weights for the data. Used as DataFrame.sample parameter. Defaults to None.
        iters (int, optional): The number of resampling iterations. Defaults to 1000.

    Returns:
        mean_estimates (array): A mean estimates sampling distribution
    """
    # Resample with replacement, calculating the mean of the data and building a list of mean estimates
    if weights is None:   # In case of no weights, use a Series
        s = pd.Series(data)
        mean_estimates = [s.sample(n=len(s), replace=True).mean() for _ in range(iters)]
    
    else:    # In case of weights use a DataFrame
        df = pd.DataFrame({'data':data,'wgt':weights})
        mean_estimates = [df.sample(n=len(df), replace=True, weights=df.wgt).data.mean() for _ in range(iters)]
    
    return np.array(mean_estimates)


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


def ResampleDiffMeansH0(a, b, iters=1000, onesided=False):
    """Generates a difference in means sampling distribution for the null hypothesis that two groups are the same via permutation (randomized shuffling) of pooled data. 
    Can then make an rv of this distribution to plot the cdf and compute the p-value of of the actual difference (eg. rv.cdf at the actual difference). 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        a (array-like): Input data set 1
        b (array-like): Input data set 2
        iters (int): The number of simulations to run (Defaults to 1000)
        onesided (bool): If set to True a onesided test, that does not use absolute value of difference, is run (Defaults to False) 

    Returns:
        test_diff: Original actual difference in means value
        diff_mean_results (array): Sampling distribution for the null hypothesis obtained from resampling
    """
    a = np.array(a)
    b = np.array(b)
    
    # Combine the two data sets
    a_size = len(a)
    pooled_data = np.hstack((a, b))

    diff_mean_results = []
    
    if onesided == False:
        test_diff = abs(a.mean() - b.mean()) # The test stat if twosided

        for _ in range(iters):
            np.random.shuffle(pooled_data)
            group1 = pooled_data[:a_size]
            group2 = pooled_data[a_size:]
            result = abs(group1.mean() - group2.mean())
            diff_mean_results.append(result)
    
    elif onesided == True:
        test_diff = a.mean() - b.mean() # The test stat if onesided

        for _ in range(iters):
            np.random.shuffle(pooled_data)
            group1 = pooled_data[:a_size]
            group2 = pooled_data[a_size:]
            result = group1.mean() - group2.mean()
            diff_mean_results.append(result)
     
    else:
        raise TypeError('\'onesided\' parameter only accepts Boolean True or False')
    
    return test_diff, np.array(diff_mean_results)


def ResampleDiffMeansHa(a, b, iters=1000):
    """Generates a difference in means sampling distribution for the alternative hypothesis that two groups differ via resampling of each group. 
    In this case the resampling is done on each sample separately. 
    (ie. assuming the alternative hypothesis that the samples are different) 
    Can then make an rv of this distribution to calculate sampling distribution mean, std deviation (std error), and confidence interval (rv.interval). 
    Can also get a one-sided p-value for case of no difference null hypothesis using rv.cdf(0). 
    For two-sided p-value, can double the one-sided if sampling distribution is symmetrical or use the H0 version of this function. 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        a (array-like): Input data set 1
        b (array-like): Input data set 2
        iters (int, optional): The number of simulations to run (Defaults to 1000)
        
    Returns:
        test_diff: Original actual difference in means value
        diff_mean_results (array): Sampling distribution for the alternative hypothesis obtained from resampling
    """
    a=pd.Series(a)
    b=pd.Series(b)
    
    diff_mean_results = []
    
    test_diff = a.mean() - b.mean()
    
    for _ in range(iters):
        a_resample = a.sample(n=len(a), replace=True)
        b_resample = b.sample(n=len(b), replace=True)
        resample_diff = a_resample.mean() - b_resample.mean()
        diff_mean_results.append(resample_diff)
        
    return test_diff, np.array(diff_mean_results)


def ResampleCorrelationH0(x, y, iters=1000, onesided=False, method='pearson'):
    """Generates a correlation sampling distribution for the null hypothesis of no correlation between the variables via permutation of one of the variables. 
    Can then make an rv of this distribution to plot cdf, compute p-value for the actual correlation value (eg. rv.cdf at actual correlation(test_r)). 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        x (array-like): Input variable 1
        y (array-like): Input variable 2
        iters (int): The number of simulations to run (Defaults to 1000)
        onesided (bool): If set to True a onesided test, that does not use absolute value of difference, is run (Defaults to False)
        method (string): Select 'pearson' or 'spearman' method (default: 'pearson')

    Returns:
        test_r: Original actual correlation value
        corrs (array): Sampling distribution for the null hypothesis of no correlation obtained from resampling
    """
    xs, ys = np.array(x), np.array(y)
    
    if method == 'pearson':

        corrs=[]    
        if onesided == False:
            test_r = abs(stats.pearsonr(xs, ys)[0])

            for _ in range(iters):
                xs = np.random.permutation(xs)
                corr = abs(stats.pearsonr(xs, ys)[0])
                corrs.append(corr)

        elif onesided == True:
            test_r = stats.pearsonr(xs, ys)[0]

            for _ in range(iters):
                xs = np.random.permutation(xs)
                corr = stats.pearsonr(xs, ys)[0]
                corrs.append(corr)

        else:
            raise TypeError('\'onesided\' parameter only accepts Boolean True or False')
    
    elif method == 'spearman':

        corrs=[]    
        if onesided == False:
            test_r = abs(stats.spearmanr(xs, ys)[0])

            for _ in range(iters):
                xs = np.random.permutation(xs)
                corr = abs(stats.spearmanr(xs, ys)[0])
                corrs.append(corr)

        elif onesided == True:
            test_r = stats.spearmanr(xs, ys)[0]

            for _ in range(iters):
                xs = np.random.permutation(xs)
                corr = stats.spearmanr(xs, ys)[0]
                corrs.append(corr)

        else:
            raise TypeError('\'onesided\' parameter only accepts Boolean True or False')        
    
    else:
        raise Exception('Must enter either pearson or spearman as a string for method argument')       
    
    return test_r, np.array(corrs)


def ResampleCorrelationHa(x, y, iters=1000, method='pearson'):
    """Generates a correlation sampling distribution for the alternative hypothesis of correlation existing between the variables. 
    This is done by resampling x, y pairs and calculating correlation on new samples. 
    Can then make an rv of this distribution to calculate sampling distribution mean, std deviation (std error), and confidence interval (rv.interval). 
    Can also get a one-sided p-value for case of no difference null hypothesis using rv.cdf(0). 
    For two-sided p-value, can double the one-sided if sampling distribution is symmetrical or use the H0 version of this function. 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        x (array-like): Input variable 1
        y (array-like): Input variable 2
        iters (int): The number of simulations to run (Defaults to 1000)
        method (string): Select 'pearson' or 'spearman' method (default: 'pearson')
        
    Returns:
        actual_r: Original actual correlation value
        corrs (array): Sampling distribution for the alternative hypothesis of no correlation obtained from resampling
    """
    if method == 'pearson':  
        # Calculate actual correlation
        actual_r = stats.pearsonr(x, y)[0]

        # Create a dataframe to hold the x and y values as pairs
        df = pd.DataFrame({'x':x, 'y': y})

        corrs=[]    
        for _ in range(iters):
            sample = df.sample(n=len(df), replace=True)
            r = stats.pearsonr(sample.x, sample.y)[0]
            corrs.append(r)
    
    elif method == 'spearman':
        # Calculate actual correlation
        actual_r = stats.spearmanr(x, y)[0]

        # Create a dataframe to hold the x and y values as pairs
        df = pd.DataFrame({'x':x, 'y': y})

        corrs=[]    
        for _ in range(iters):
            sample = df.sample(n=len(df), replace=True)
            r = stats.spearmanr(sample.x, sample.y)[0]
            corrs.append(r)
    
    else:
        raise Exception('Must enter either pearson or spearman as a string for method argument')
      
    return actual_r, np.array(corrs)


def ResampleChiSquare(observed, expected, iters=1000):
    """Generates a chisquared statistic sampling distribution by randomly choosing values 
    according to the expected probablities to simulate the null hypothesis. 
    The sequences must be the same length, be integer counts of a categorical variable 
    and the sum of the sequence values must be the same. 
    If the sum of the sequence values is different, first normalize the expected values 
    and then create a new expected values sequence by multiplying by the total number of observed values. 
    adjust_expected = expected/sum(expected)*sum(observed) 
    Can then make an rv of this distribution to plot cdf and  
    compute a p-value for the actual chi-squared statistic (eg. rv.cdf at actual statistic (test_chi)). 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        observed (array-like): observed values sequence
        expected (array-like): expected values sequence
        iters (int, optional): Number of iterations to run when building distribution. Defaults to 1000.

    Returns:
        test_chi: Original actual chi squared value
        chis (array): Sampling distribution for the null hypothesis obtained from resampling
    """
    observed, expected = np.array(observed), np.array(expected)
    
    # Check that sum of values are equal
    if np.isclose(sum(observed), sum(expected)) == False:
        raise ValueError('The sum of the values for observed and expected must be equal.')
    
    # Calculate the chi square test statistic
    test_chi = sum((observed - expected)**2 / expected)
        
    # Calculate the variables needed for resampling
    n = sum(expected)
    values = list(range(len(expected)))
    p_exp = expected/sum(expected)
    
    # Build the chi square sampling distribution for the null hypothesis
    chis=[]
    for _ in range(iters):
        # Build a model_observed sequence generated by resampling using expected probabilities
        hist = Counter({x:0 for x in values})
        hist.update(np.random.choice(values, size=n, replace=True, p=p_exp))
        sorted_hist = sorted(hist.items())
        model_observed = np.array([x[1] for x in sorted_hist])

        # Compute chi square statistic and append
        chi = sum((model_observed - expected)**2 / expected)
        chis.append(chi)
    
    return test_chi, np.array(chis)


def ResampleChiSquareContingency(observed, iters=1000):
    """Generates a chisquared statistic sampling distribution 
    from a contingency table. 
    Can then make an rv of this distribution to plot cdf and  
    compute a p-value for the actual chi-squared statistic (eg. rv.cdf at actual statistic (test_chi)). 
    Can also use the 'min' and 'max' built-ins to find what the most extreme values are from the simluations.

    Args:
        observed (array-like): observed contingency table
        iters (int, optional): Number of iterations to run when building distribution. Defaults to 1000.

    Returns:
        test_chi: Original actual chi squared value
        chis (array): Sampling distribution for the null hypothesis obtained from resampling
    """
    # Put the data into array form
    observed = np.asarray(observed, dtype=np.float64)
    
    # Calculate the test chi square statistic and the expected array
    test_chi,_,_,expected = stats.chi2_contingency(observed)
    
    # Calculate variables to be used in resampling
    expected = np.asarray(expected, dtype=np.float64)
    expected_shape = expected.shape
    expected_ps = expected / np.sum(expected)
    values = np.array(list(range(len(expected.ravel())))) # Flatten the array and then reshape it later
    n= int(np.sum(expected))
      
    # Compute resampled expected values and compute chi square 
    # to build a sampling distribution that represents the null hypothesis
    chis=[]
    for _ in range(iters):
        hist = Counter({x:0 for x in values}) # Initiate an empty histogram to hold resampled values
        hist.update(np.random.choice(values, size=n, replace=True, p=expected_ps.ravel()))
        sorted_hist = sorted(hist.items())
        resampled_expected = np.array([x[1] for x in sorted_hist])
        resampled_expected_reshaped = resampled_expected.reshape(expected_shape) # Put back into original shape

        chi = stats.chi2_contingency(resampled_expected_reshaped)[0]
        chis.append(chi)

    return test_chi, np.array(chis)


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


def SummarizeEstimates(estimates, conf_int=0.95):
    """Computes the mean, standard deviation (std error), and a confidence interval for a sampling distribution (estimates).

    Args:
        estimates (array-like): A sequence of estimates for a statistic obtained from resampling (sampling distribution)
        conf_int (float): Probability for the confidence interval. Must be between 0 and 1. Defaults to 0.95.

    Returns:
        mean: mean value of the estimates
        std: standard deviation of the estimates (std error)
        confidence interval: interval about the median of the distribution
    """
    rv = DiscreteRv(estimates)
    return np.mean(estimates), np.std(estimates), rv.interval(conf_int)


def PValueFromEstimates(estimates, test_statistic, tail='right'):
    """Generates a pvalue from a sampling distribution (sequence of estimates) for a given test statistic.

    Args:
        estimates (array-like): The sampling distribution sequence
        test_statistic (float): The test statistic to be used to generate the pvalue
        tail (str, optional): Determines which tail to use for pvalue. Accepts 'left' or 'right' only. Defaults to 'right'.

    Returns:
        pvalue: Pvalue for test statistic
    """
    rv = DiscreteRv(estimates)
    
    if tail == 'left':
        pvalue = rv.cdf(test_statistic)
    elif tail == 'right':
        pvalue = 1 - rv.cdf(test_statistic)
    else:
        raise Exception('The value of \'tail\' can only be either \'left\' or \'right\'')
    
    return pvalue


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


class HypothesisTest(object):
    """Represents a hypothesis test. 
    The actual test statistic for the data is available through a .actual attribute. 
    After PValue is run the scipy stats random variable for the sampling distribution is available through a .rv attribute. 
    The cdf of the distribution along with a line representing the test statistic value can be plotted using PlotCdf(). 
    The largest test statistic seen in the simulations is given by MaxTestStat()."""

    def __init__(self, data):
        """Initializes the hypothesis test.

        data: data in whatever form is relevant
        """
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data) # pylint: disable=assignment-from-no-return
        self.test_stats = None
        self.rv = None

    def PValue(self, iters=1000):
        """Computes the distribution of the test statistic and p-value.

        iters: number of iterations

        returns: float p-value
        """
        self.test_stats = [self.TestStatistic(self.RunModel())
                           for _ in range(iters)]
        self.rv = DiscreteRv(self.test_stats)

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def MaxTestStat(self):
        """Returns the largest test statistic seen during simulations.
        """
        return max(self.test_stats)

    def PlotCdf(self, label=None):
        """Draws a Cdf with vertical lines at the observed test stat.
        """      
        def VertLine(x):
            """Draws a vertical line at x."""
            plt.plot([x, x], [0, 1], color='0.8')

        VertLine(self.actual)
        plt.plot(self.rv.xk, self.rv.cdf(self.rv.xk)) # pylint: disable=no-member

    def TestStatistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant        
        """
        raise UnimplementedMethodException()

    def MakeModel(self):
        """Build a model of the null hypothesis.
        """
        pass

    def RunModel(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        raise UnimplementedMethodException()


class HTDiffMeansPermute(HypothesisTest):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = abs(group1.mean() - group2.mean())
        return test_stat

    def MakeModel(self):
        group1, group2 = self.data
        self.n, self.m = len(group1), len(group2)
        self.pool = np.hstack((group1, group2))

    def RunModel(self):
        np.random.shuffle(self.pool)
        data = self.pool[:self.n], self.pool[self.n:]
        return data


class HTDiffMeansPermuteOneSided(HTDiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.mean() - group2.mean()
        return test_stat


class HTDiffMeansRandom(HTDiffMeansPermute):
    '''Tests a difference in means using resampling.'''

    def RunModel(self):
        group1 = np.random.choice(self.pool, self.n, replace=True)
        group2 = np.random.choice(self.pool, self.m, replace=True)
        return group1, group2


class HTDiffStdPermute(HTDiffMeansPermute):

    def TestStatistic(self, data):
        group1, group2 = data
        test_stat = group1.std() - group2.std()
        return test_stat


class HTCorrelationPermute(HypothesisTest):

    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(stats.pearsonr(xs, ys)[0])
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys


class HTChiSquare(HypothesisTest):
    '''Represents a hypothesis test for two sequences, observed and expected. 
    Pass the sequences as arrays. 
    The sequences must be the same length, be integer counts of a categorical variable 
    and have the sum of the sequence values must be the same. 
    If the sum of the sequence values is different, first normalize the expected values 
    and then create a new expected values sequence by multiplying by the total number of observed values. 
    adjust_expected = expected/sum(expected)*sum(observed)'''
    
    def TestStatistic(self, data):
        observed, expected = data
        test_stat = sum((observed - expected)**2 / expected)
        return test_stat

    def RunModel(self):
        observed, expected = self.data
        n = sum(observed)
        values = list(range(len(expected)))
        p_exp = expected/sum(expected)
        hist = Counter({x:0 for x in values}) # Initialize a Counter with zero values
        hist.update(np.random.choice(values, size=n, replace=True, p=p_exp))
        sorted_hist = sorted(hist.items())
        model_observed = np.array([x[1] for x in sorted_hist])
        return model_observed, expected


class PowerTest():
    """Power test superclass. 
    All child classes must provide PrepareData and ComputeTestStatandRv methods.
    """
    
    def __init__(self, data, alpha=0.05, alternative='two-sided', num_runs=1000):
        self.data = data
        self.alpha = alpha
        self.alternative = alternative
        self.num_runs = num_runs
        self.PrepareData()
    
    # Provide functionality to convert the data into format needed for use in BuildRv
    # Ex. Convert to array, split data into component groups, etc.
    # See child classes for examples
    def PrepareData(self):
        UnimplementedMethodException()
    
    # Provide functionality that creates the run data and then computes the run's test stat and rv
    # This involves doing one resample to simulate pulling an additional sample from the population,
    # then calculating the test_stat, building a sampling distribution, and computing the rv
    # See child classes for examples
    def ComputeTestStatandRv(self):
        UnimplementedMethodException()
    
    # Computes the pvalue of test stat from an rv,
    # and adds to pvalue_count if less than significance level
    def _RunPvalueCount(self):
        test_stat, rv = self.ComputeTestStatandRv() # pylint: disable=assignment-from-no-return
        
        p_value_right = 1 - rv.cdf(test_stat)
        p_value_left = rv.cdf(test_stat)
        
        # Two-sided test
        if self.alternative == 'two-sided':
            if (p_value_right < self.alpha/2) or (p_value_left < self.alpha/2):
                self.pvalue_count+= 1
        
        # One-sided test using the right side of the distribution
        elif self.alternative == 'right': 
            if p_value_right < self.alpha:
                self.pvalue_count += 1
        
        # One-sided test using the left side of the distribution
        elif self.alternative == 'left': 
            if p_value_left < self.alpha:
                self.pvalue_count += 1
        
        else:
            raise ValueError("alternative has to be 'two-sided', 'right', or 'left")
    
    # Method for computing power 
    def Power(self):
        self.pvalue_count = 0
        for _ in range(self.num_runs):
            self._RunPvalueCount()
            
        return self.pvalue_count / self.num_runs


class PTMean(PowerTest):
    """Calculates the power of a one-sample mean hypothesis test. 
    A test_stat (eg. zero for no effect) must be provided.
    """
    def __init__(self, data, test_stat, alpha=0.05, alternative='two-sided', num_runs=1000):
        PowerTest.__init__(self, data, alpha, alternative, num_runs)
        # Alternative hypothesis power tests require a test_stat be provided for null hypothesis (eg. zero for no effect)
        self.test_stat = test_stat 
    
    def PrepareData(self):
        self.data = np.array(self.data)
    
    def ComputeTestStatandRv(self):
        run_data = np.random.choice(self.data, size=len(self.data), replace=True)
        mean_estimates = [np.random.choice(run_data, size=len(run_data), replace=True).mean() for _ in range(100)]
        
        test_stat = self.test_stat
        rv = DiscreteRv(mean_estimates)
        
        return test_stat, rv


class PTDiffMeansH0(PowerTest):
    """Calculates the power of a difference of means hypothesis test 
    using permutation of pooled data to simulate the null hypothesis 
    and build the null hypothesis sampling distribution.
    """
    def PrepareData(self):
        self.a, self.b = self.data
        self.a = np.array(self.a)
        self.b = np.array(self.b)
        self.pooled_data = np.hstack((self.a, self.b))
        self.a_size = len(self.a)
    
    def ComputeTestStatandRv(self):
        # Create run data by resampling the two groups
        sample1 = np.random.choice(self.a, size=len(self.a), replace=True)
        sample2 = np.random.choice(self.b, size=len(self.b), replace=True)
        
        # Calculate test_stat for the run data
        test_stat = sample1.mean() - sample2.mean()
        
        diff_mean_results = []
        
        # Build a sampling distribution for the run
        for _ in range(100):
            np.random.shuffle(self.pooled_data)
            group1 = self.pooled_data[:self.a_size]
            group2 = self.pooled_data[self.a_size:]
            result = group1.mean() - group2.mean()
            diff_mean_results.append(result)
        
        rv = DiscreteRv(diff_mean_results)
        
        return test_stat, rv


class PTDiffMeansHa(PowerTest):
    """Calculates the power of a difference of means hypothesis test 
    using resampling of groups to simulate the alternative hypothesis 
    and build the alternative hypothesis sampling distribution. 
    A test_stat (eg. zero for no effect) must be provided.
    """
    def __init__(self, data, test_stat, alpha=0.05, alternative='two-sided', num_runs=1000):
        PowerTest.__init__(self, data, alpha, alternative, num_runs)
        # Alternative hypothesis power tests require a test_stat be provided for null hypothesis (eg. zero for no effect)
        self.test_stat = test_stat 
    
    def PrepareData(self):
        self.a, self.b = self.data
        self.a = np.array(self.a)
        self.b = np.array(self.b)
    
    def ComputeTestStatandRv(self):
        # Create run data
        sample1 = np.random.choice(self.a, size=len(self.a), replace=True)
        sample2 = np.random.choice(self.b, size=len(self.b), replace=True)
        
        diff_mean_results = []
        
        # Build a sampling distribution for the run
        for _ in range(100):
            group1 = np.random.choice(sample1, size=len(sample1), replace=True)
            group2 = np.random.choice(sample2, size=len(sample2), replace=True)
            result = group1.mean() - group2.mean()
            diff_mean_results.append(result)
        
        test_stat = self.test_stat
        rv = DiscreteRv(diff_mean_results)
        
        return test_stat, rv


class PTCorrelationH0(PowerTest):
    """Calculates the power of a correlation hypothesis test 
    using permutation to simulate the null hypothesis of no correlation 
    and build the null hypothesis sampling distribution.
    """
    def __init__(self, data, alpha=0.05, alternative='two-sided', num_runs=1000, method='pearson'):
        PowerTest.__init__(self, data, alpha, alternative, num_runs)
        self.method = method
    
    def PrepareData(self):
        self.x, self.y = self.data
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.df = pd.DataFrame({'x':self.x, 'y': self.y})
    
    def ComputeTestStatandRv(self):
        # Create run data
        run_data = self.df.sample(n=len(self.df), replace=True)
        run_x = run_data.x.values
        run_y = run_data.y.values
        
        corrs=[]
        
        # Compute test_stat and build rv for the run
        if self.method == 'pearson':
            test_stat = stats.pearsonr(run_x , run_y)[0]
            
            for _ in range(100):
                x_perm = np.random.permutation(run_x)
                r = stats.pearsonr(x_perm , run_y)[0]
                corrs.append(r)
    
        elif self.method == 'spearman':
            test_stat = stats.spearmanr(run_x , run_y)[0]
            
            for _ in range(100):
                x_perm = np.random.permutation(run_x)
                r = stats.spearmanr(x_perm , run_y)[0]
                corrs.append(r)
    
        else:
            raise Exception('Must enter either pearson or spearman as a string for method argument')   
        
        rv = DiscreteRv(corrs)
        
        return test_stat, rv


class PTCorrelationHa(PowerTest):
    """Calculates the power of a correlation hypothesis test 
    using resampling of the paired data to simulate the alternative hypothesis 
    and build the alternative hypothesis sampling distribution. 
    A test_stat (eg. zero for no effect) must be provided.
    """
    def __init__(self, data, test_stat, alpha=0.05, alternative='two-sided', num_runs=1000, method='pearson'):
        PowerTest.__init__(self, data, alpha, alternative, num_runs)
        self.method = method
        # Alternative hypothesis power tests require a test_stat be provided for null hypothesis (eg. zero for no effect)
        self.test_stat = test_stat 
    
    def PrepareData(self):
        self.x, self.y = self.data
        self.df = pd.DataFrame({'x':self.x, 'y': self.y})
    
    def ComputeTestStatandRv(self):
        # Create run data
        run_data = self.df.sample(n=len(self.df), replace=True)
        
        corrs=[]
        
        # Build rv
        if self.method == 'pearson':          
            for _ in range(100):
                sample = run_data.sample(n=len(run_data), replace=True)
                r = stats.pearsonr(sample.x, sample.y)[0]
                corrs.append(r)
    
        elif self.method == 'spearman':            
            for _ in range(100):
                sample = run_data.sample(n=len(run_data), replace=True)
                r = stats.spearmanr(sample.x, sample.y)[0]
                corrs.append(r)
    
        else:
            raise Exception('Must enter either pearson or spearman as a string for method argument')
               
        test_stat = self.test_stat
        rv = DiscreteRv(corrs)
        
        return test_stat, rv


class PTChiSquare(PowerTest):
    """Calculates the power of a chi square hypothesis test 
    using resampling of the expected sequence to simulate the null hypothesis 
    and build the null hypothesis sampling distribution. 
    Takes data in the form of two sequences: data = observed, expected
    """    
    def PrepareData(self):
        self.observed, self.expected = self.data
        self.observed = np.array(self.observed)
        self.expected = np.array(self.expected)
    
    def ComputeTestStatandRv(self):
        # Create run data (run_observed) by resampling the observed sequence (assuming the alternative hypothesis)
        n = sum(self.observed)
        values_obs = list(range(len(self.observed)))
        p_obs = self.observed/sum(self.observed)
        
        hist = Counter({x:0 for x in values_obs})
        hist.update(np.random.choice(values_obs, size=n, replace=True, p=p_obs))
        sorted_hist = sorted(hist.items())
        run_observed = np.array([x[1] for x in sorted_hist])
        
        # Calculate chi square test_stat for the run data
        test_stat = sum((run_observed - self.expected)**2 / self.expected)
        
        chis = []
        
        # Build a chi square sampling distribution for the run using the expected sequence (null hypothesis)
        for _ in range(100):
            n = sum(self.expected)
            values = list(range(len(self.expected)))
            p_exp = self.expected/sum(self.expected)
            
            hist = Counter({x:0 for x in values}) # Initialize a Counter with zero values
            hist.update(np.random.choice(values, size=n, replace=True, p=p_exp))
            sorted_hist = sorted(hist.items())
            model_observed = np.array([x[1] for x in sorted_hist])
            chi = sum((model_observed - self.expected)**2 / self.expected)
            chis.append(chi)
        
        rv = DiscreteRv(chis)
        
        return test_stat, rv


class PTChiSquareContingency(PowerTest):
    """Calculates the power of a chi square contingency table hypothesis test 
    using resampling of the expected sequence to simulate the null hypothesis 
    and build the null hypothesis sampling distribution. 
    Takes data in the form of a single observed contingency table (array-like)
    """    
    def PrepareData(self):
        self.observed = self.data
        self.observed = np.array(self.observed)
    
    def ComputeTestStatandRv(self):
        # Create run data (resampled_observed_reshaped) by resampling the observed data (assuming the alternative hypothesis)    
        observed_shape = self.observed.shape
        observed_ps = self.observed / np.sum(self.observed)
        values = np.array(list(range(len(self.observed.ravel())))) # Flatten the array and then reshape it later
        n= int(np.sum(self.observed))
        
        hist = Counter({x:0 for x in values}) # Initiate an empty histogram to hold resampled values
        hist.update(np.random.choice(values, size=n, replace=True, p=observed_ps.ravel()))
        sorted_hist = sorted(hist.items())
        resampled_observed = np.array([x[1] for x in sorted_hist])
        resampled_observed_reshaped = resampled_observed.reshape(observed_shape) # Put back into original shape
        
        # Calculate chi square test_stat and expected contingency table from the run data
        test_stat,_,_,expected = stats.chi2_contingency(resampled_observed_reshaped)
        
        chis = []
        
        # Build a chi square sampling distribution for the run using the expected sequence (null hypothesis)
        for _ in range(100):
            expected_shape = expected.shape
            expected_ps = expected / np.sum(expected)
            values = np.array(list(range(len(expected.ravel())))) # Flatten the array and then reshape it later
            n= int(np.sum(expected))
            
            hist = Counter({x:0 for x in values}) # Initiate an empty histogram to hold resampled values
            hist.update(np.random.choice(values, size=n, replace=True, p=expected_ps.ravel()))
            sorted_hist = sorted(hist.items())
            resampled_expected = np.array([x[1] for x in sorted_hist])
            resampled_expected_reshaped = resampled_expected.reshape(expected_shape) # Put back into original shape

            chi = stats.chi2_contingency(resampled_expected_reshaped)[0]
            chis.append(chi)
        
        rv = DiscreteRv(chis)
        
        return test_stat, rv


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