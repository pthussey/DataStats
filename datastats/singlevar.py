"""This module contains functions for working with univariate data. 
These include functions that produce values to plot models of various types of distributions 
and a number of other functions for working with univariate data.
"""

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


def NormalProbabilityValues(a):
    """Creates x and y values to be used in a normal probability plot. 
    This kind of plot is used to determine whether or not a data set is normally distributed.

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


def CohenEffectSize(group1, group2):
    """Computes Cohen's effect size for two groups within a single variable or across multiple variables.

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


def RvPmfDiffs(group1, group2):
    """Computes a shared range of values and the percentage point differences 
    between the data for two different groups of a single discrete numerical variable. 
    The resulting differences can then be plotted to show how the distributions of the two groups compare. 
    For example, the data could be lists of ages for males and females, 
    and the results of this function can be used in a plot to show the difference between them.

    Arguments:
        rv1, rv2 -- two discrete_rvs

    Returns:
        shared_xk {list} -- A list of values from the minimum to maximum 
        among all values from both input rvs

        diffs {list} -- A list of the percentage point differences at each value in shared_xk for the rvs (group1 - group2)
    """
    # Create rvs for each group
    val1, cnt1 = np.unique(group1, return_counts=True)
    group1_rv = stats.rv_discrete(values=(val1,cnt1/sum(cnt1)))

    val2, cnt2 = np.unique(group2, return_counts=True)
    group2_rv = stats.rv_discrete(values=(val2,cnt2/sum(cnt2)))

    # Compute the the shared_xk values
    total = np.concatenate((group1_rv.xk, group2_rv.xk))
    xk_min = total.min()
    xk_max = total.max()
    shared_xk = list(range(xk_min, xk_max+1))
        
    # Compute the percentage point differences at each value in shared_xk
    diffs = []
    
    for x in shared_xk:
        p1 = group1_rv.pmf(x)
        p2 = group2_rv.pmf(x)
        diff = (p1-p2)*100
        diffs.append(diff)
    
    return shared_xk, diffs


def ListsToDataFrame(data, group_labels, group_col_name=None, values_col_name=None):
    """Takes data in the form of a list of lists, 
    along with a list of data group labels of the same length, 
    and produces a long-form DataFrame with two columns: 
    one for the group labels and one for the corresponding values.

    Args:
        data (list): A list of lists holding the data
        group_labels (list): A list of the group labels for each list provided in 'data
        group_col_name (string, optional): 
        A name to give to the group column in the DataFrame. Defaults to None. 
        If not provided uses the pandas melt default 'variable'.
        values_col_name (_type_, optional): 
        A name to give to the values column in the DataFrame. Defaults to None. 
        If not provided uses the pandas melt default 'value'.

    Returns:
        data_df: The resulting DataFrame
    """
    
    if len(data) != len(group_labels):
        raise Exception ("The number of group_labels must equal the number of data groups.")
    
    # Build the DataFrame from the provided data and group labels sequences
    data_df = (pd.DataFrame(data=data, index=group_labels).T
                                                          .melt()
                                                          .dropna())
    
    # Rename the group column if provided, otherwise use pandas melt default 'variable'
    if group_col_name is not None:
        data_df.rename(columns={'variable':group_col_name}, inplace=True)
    
    # Rename the values column if provided, otherwise use pandas melt default 'value'
    if values_col_name is not None:
        data_df.rename(columns={'value':values_col_name}, inplace=True)
    
    return data_df


def DataFrameToLists(data, group_col_name, values_col_name):
    """Does the opposite of the 'ListsToDataFrame' function. 
    Takes a DataFrame, and names of the columns containing the groups and values, 
    and produces two lists: one containing lists of the data for each group, 
    and the other containing the names of each group.

    Args:
        data (pandas DataFrame): The DataFrame that holds the data
        group_col_name (string): The name of the column that holds the group labels
        values_col_name (string): The name of the column that holds the corresponding values

    Returns:
        group_labels_list: A list of the group names
        group_values_list: A list of list containing the values for each group
    """
    
    # Make a dictionary that holds the 'values_col_name' data grouped by 'group_col_name' variable
    groups = dict(list(data.groupby(group_col_name)[values_col_name]))
    
    # Instantiate two lists to hold the group labels and the lists of values for each group
    group_labels_list = []
    group_values_list = []
    
    # Build the lists of labels and corresponding values
    for label,data in groups.items():
        group_labels_list.append(label)
        group_values_list.append(data.values)
    
    return group_labels_list, group_values_list


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
