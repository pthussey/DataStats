import numpy as np
import pandas as pd
import scipy.stats as stats


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
    to report the size of a group the belong to.
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
    to report the size of a group the are a part of.
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


def SampleRows(df, nrows, replace=False):
    """Generates a random sample of rows from a dataframe.

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
    ''' Calculate a correlation coefficient and its confidence interval for two data sets.
    
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


def main():
    pass

if __name__ == '__main__':
    main()