import numpy as np
import pandas as pd

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

from collections import Counter

from datastats.singlevar import DiscreteRv


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


def AnovaPostHoc(data, alpha=0.05):
    """Performs ANOVA post-hoc analysis using a difference of means hypothesis test 
    on each possible pairing of supplied data sequences.
    This analysis is used to determine which pairs 
    have statistically significant differences in their means
    
    Args
    ----
    data (array-like):
        A list or tuple of data sequences (group_1, group_2... group_n)
    alpha (float)
        The family wise error rate (FWER)
        Must be between 0 and 1. Defaults to 0.05.
    
    Returns
    -------
    results:
        A tuple of tuples containing the results for each pairing of sequences. 
        The results can be printed in an easy-to-read format using a for loop. 
        The results display three pieces of information:
        1) The pairing
        2) The pvalue for the pairing
        3) Whether the pvalue is significant or not (Y or N)
           * The significance level is determined by comparison of the pvalue 
             with the experiment-wise significance level. 
             The Bonferroni correction method is used to compute this significance level.
    corrected_alpha:
        The experiment-wise significance level
    """
    num_comparisons = int(factorial(len(data))/(2*factorial(len(data)-2)))
    corrected_alpha = alpha/num_comparisons
    enum_data = enumerate(data)
    
    results=[]
    for pair in combinations(enum_data, 2):
        test = HTDiffMeansH0((pair[0][1], pair[1][1]))
        pvalue = test.PValue()
        significant = 'Y' if pvalue < corrected_alpha else 'N'
        results.append(((pair[0][0], pair[1][0]), '{:.3f}'.format(pvalue), significant))
    
    return results, corrected_alpha


def PairwiseTukeyHsd(data, alpha=0.05):
    """Uses pairwise Tukey HSD to perform ANOVA post-hoc analysis 
    to determine which paired comparisons have differences that are significant. 
    This function uses statsmodels pairwise_tukeyhsd.
    Accepts data in the form of a list of data sequences (eg.(group_1, group_2... group_n)).
    The returned object can be printed to show the results.
    
    Args
    ----
    data (array-like):
        A list or tuple of data sequences (group_1, group_2... group_n)
    alpha (float)
        The family wise error rate (FWER)
    
    Returns
    -------
    A TukeyHSDResults object 
    """
    x = []
    label = []
    
    for grp_num in range(len(data)):
        x.extend(data[grp_num])
        label.extend([grp_num]*len(data[grp_num]))


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
