"""This module contains functions that are useful in exploratory data analysis plotting.
"""

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from math import ceil

from .multivar import PercentileRows
from .singlevar import NormalProbabilityValues
from .multivar import ResidualPercentilePlotData
from .multivar import FitLine
from .multivar import ResampleInterSlope


# Sets the default rc parameters for plotting
def SetParams(font='DejaVu Sans Mono', basesize=12, basecolor='0.4', style='seaborn-whitegrid'):
    """Sets some matplotlib rcParams and a style.

    Args:
        font (str, optional): Choose the font. Defaults to 'DejaVu Sans Mono'. Use 'Malgun Gothic' for Korean.
        basesize (int, optional): Sets a base font size. Defaults to 12.
        basecolor (str, optional): Sets a base color. Defaults to '0.4'.
        style (str, optional): Sets the style. Defaults to seaborn-whitegrid. Pass None for no style.
    """
    if style == None:
        pass
    else:
        plt.style.use(style)
    
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = basesize
    plt.rcParams["xtick.labelsize"] = basesize
    plt.rcParams["ytick.labelsize"] = basesize
    plt.rcParams["legend.fontsize"] = basesize
    plt.rcParams["axes.titlesize"] = basesize+2
    plt.rcParams["axes.titleweight"] = 'bold'
    plt.rcParams["axes.labelsize"] = basesize+1
    plt.rcParams["text.color"] = basecolor
    plt.rcParams["axes.labelcolor"] = basecolor
    plt.rcParams["axes.edgecolor"] = basecolor
    plt.rcParams["xtick.color"] = basecolor
    plt.rcParams["ytick.color"] = basecolor
    plt.rcParams["ytick.left"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["axes.labelpad"] = basesize
    plt.rcParams['axes.unicode_minus'] = False if font == 'Malgun Gothic' else True


def LinRegPlot(x, y, ci=95, x_label='x', y_label='y', plot_title='Regression Plot'):
    """Plots a Seaborn-like regression plot that shows a scatter plot of the data, 
    along with the best fit line and a confidence interval (CI). 
    The difference with Seaborn is this function uses non-parametric methods to produce the CI lines.

    Args:
        x (array-like): The x variable.
        y (array-like): The y variable
        ci (float): The confidence interval. Must be between 0 and 100. Defaults to 95.
        x_label (str): The label to use on the x-axis. Defaults to 'x'.
        y_label (str): The label to use on the y-axis. Defaults to 'y'.
        plot_title (str): The title for the plot. Defaults to 'Regression Plot'.
    
    Returns:
        fig: The figure for the plot
        ax: The axis for the plot
    """
    # Perform the regression
    regress_results = stats.linregress(x, y)
    
    # Get the best fit line
    fit_xs, fit_ys = FitLine(x, regress_results.intercept, regress_results.slope)
    
    # Build the intercept and slope sampling distributions, as well as the fit_ys_list
    inters, slopes, fit_ys_list = ResampleInterSlope(x, y)
    
    # Get the lines from fit_ys_list to use for the CI
    percentile_rows = PercentileRows(fit_ys_list, [(100-ci)/2,(ci+100)/2])
    
    # Set up the plot figure
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)

    # Our regression plot
    ax.scatter(x, y, color='C0', s=16, alpha=0.8)
    ax.plot(fit_xs, fit_ys, color='C1')
    ax.fill_between(fit_xs, percentile_rows[0], percentile_rows[1], color='C1', alpha=0.2)
    
    # Labels and titles
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    
    return fig, ax


def LinRegPlusResidualPlot(x, y, ci=95, x_label='x', y_label='y', res_plot_bins=10):
    """Plots a Seaborn-like regression plot (datastats.plotting.LineRegPlot) 
    side-by side with a residual percentile plot which is used to help visualize 
    how well a best fit line fits the data.

    Args:
        x (array-like): The x variable.
        y (array-like): The y variable
        ci (float): The confidence interval. Must be between 0 and 100. Defaults to 95.
        x_label (str): The label to use on the x-axis. Defaults to 'x'.
        y_label (str): The label to use on the y-axis. Defaults to 'y'.
        res_plot_bins (int): Number of bins for residual plot. Defaults to 10.
    
    Returns:
        fig: The figure for the plot
        axes: The axes for the plot
    """
    # Perform the regression
    regress_results = stats.linregress(x, y)
    
    # Get the best fit line
    fit_xs, fit_ys = FitLine(x, regress_results.intercept, regress_results.slope)
    
    # Build the intercept and slope sampling distributions, as well as the fit_ys_list
    inters, slopes, fit_ys_list = ResampleInterSlope(x, y)
    
    # Get the lines from fit_ys_list to use for the CI
    percentile_rows = PercentileRows(fit_ys_list, [(100-ci)/2,(ci+100)/2])
    
    # Set up the plot figure
    fig,axes = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(16,6)

    # Our regression plot
    axes[0].scatter(x, y, color='C0', s=16, alpha=0.8)
    axes[0].plot(fit_xs, fit_ys, color='C1')
    axes[0].fill_between(fit_xs, percentile_rows[0], percentile_rows[1], color='C1', alpha=0.2)
    
    # Labels and titles for regression plot
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title('Regression Plot')

    # get x and y limits to set aspect ratio
    x_left, x_right = axes[0].get_xlim()
    y_low, y_high = axes[0].get_ylim()
    #set aspect ratio to 3:4
    axes[0].set_aspect(abs((x_right-x_left)/(y_low-y_high))*0.75)
    
    # Build x_means and res_rvs (residual random variables) for the Residual Percentile Plot
    x_means, res_rvs = ResidualPercentilePlotData(x, y, res_plot_bins)
    
    # Residual percentile plot
    percentiles = [.25,.50,.75]

    for p in percentiles:
        residual_values_at_p = [rv.ppf(p) for rv in res_rvs]
        axes[1].plot(x_means, residual_values_at_p, label= p)
    
    # Labels and titles for residual plot
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].set_title('Residual Percentile Plot')
    axes[1].legend(title='Percentiles')
    
    # Resize the residual plot's y-range to the same scale as that of the regression plot
    axes[1].set_ylim(-max(y)/2,max(y)/2)

    # get x and y limits for aspect ratio
    x_left, x_right = axes[1].get_xlim()
    y_low, y_high = axes[1].get_ylim()
    # set aspect ratio to 3:4
    axes[1].set_aspect(abs((x_right-x_left)/(y_low-y_high))*0.75)
    
    return fig, axes


def CdfPlot(data, ci=95, central_tendency_measure = 'mean', test_stat=None, x_label = 'x', legend=True):
    """Plots a CDF for supplied data. 
    Especially useful for looking at sampling distribution data. 
    Includes parameters to add lines for a ci, measures of central tendency, and a test statistic.

    Args:
        data (array-like): Data to be plotted. Must be one-dimensional.
        ci (float): The confidence interval. Must be between 0 and 100. Defaults to 95.
        central_tendency_measure: Choose 'mean', 'median', or 'both'. Defaults to 'mean'.
        test_stat (float, optional): Test stat to plot. Defaults to None.
        x_label (string): The label to use on the x-axis. Defaults to 'x'.
        legend (boolean): Choose to include a legend or not. Defaults to True.
    
    Returns:
        fig: The figure for the plot
        ax: The axis for the plot
    """
        # Convert to an array
    data = np.asarray(data)

    # Remove any nans
    data = data[~np.isnan(data)]
    
    # Compute an rv for the data
    val,cnt = np.unique(data, return_counts=True)
    rv = stats.rv_discrete(values=(val,cnt/sum(cnt)))

    # Set up figure (single plot)    
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)

    # Plot the cdf
    ax.plot(rv.xk, rv.cdf(rv.xk), lw=2)

    # Add lines for ci
    ax.axvline(rv.interval(ci/100)[0], color='C4', lw=1.3) # CI lower, purple line
    ax.axvline(rv.interval(ci/100)[1], color='C4', lw=1.3, label='CI') # CI upper, purple line
    
    # Add lines for central tendency measures
    if central_tendency_measure == 'mean':
        ax.axvline(rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line
    
    elif central_tendency_measure == 'median':
        ax.axvline(rv.median(), color='C2', lw=1.3, label='Median') # median, green line
    
    elif central_tendency_measure == 'both':
        ax.axvline(rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line
        ax.axvline(rv.median(), color='C2', lw=1.3, label='Median') # median, green line
    else:
        raise Exception('Must enter \'mean\', \'median\', or \'both\' for kind parameter.')
    
    # Add line for test stat
    if test_stat is not None:
        ax.axvline(test_stat, color='C3', lw=1.3, label='Test Stat') # test statistic, red line
    
     # Labels and titles
    ax.set_xlabel(x_label)
    ax.set_ylabel('Cumulative Density')
    ax.set_title('CDF Plot')

    if legend:
        ax.legend()

    return fig, ax


def KdePlot(data, bw_adjust=None, clip=None, ci=95, 
            central_tendency_measure = 'mean', test_stat=None, x_label='x', legend=True):
    """Plots a KDE for supplied data using Seaborn kdeplot. 
    Especially useful for looking at sampling distribution data.  
    Includes parameters to add lines for a ci, measures of central tendency, and a test statistic.

    Args:
        data (array-like): Data to be plotted. Must be one-dimensional.
        bw_adjust (float, optional): Adjusts kde bandwidth. Defaults to None.
        clip (tuple, optional): Clips data at values in the tuple. Defaults to None.
        ci (float): The confidence interval. Must be between 0 and 100. Defaults to 95.
        central_tendency_measure: Choose 'mean', 'median', or 'both'. Defaults to 'mean'.
        test_stat (float, optional): Test stat to plot. Defaults to None.
        x_label (string): The label to use on the x-axis. Defaults to 'x'.
        legend (boolean): Choose to include a legend or not. Defaults to True.
    
    Returns:
        fig: The figure for the plot
        ax: The axis for the plot
    """
    # Convert to an array
    data = np.asarray(data)

    # Remove any nans
    data = data[~np.isnan(data)]
    
    # Compute an rv for the data, used for ci
    val,cnt = np.unique(data, return_counts=True)
    rv = stats.rv_discrete(values=(val,cnt/sum(cnt)))

    # Set up figure (single plot)
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)

    # Change settings if needed and plot kde
    if (bw_adjust != None) and (clip != None):
        sns.kdeplot(data, lw=2, bw_adjust=bw_adjust, clip=clip)
    
    elif (bw_adjust == None) and (clip != None):
        sns.kdeplot(data, lw=2, clip=clip)
    
    elif (bw_adjust != None) and (clip == None):
        sns.kdeplot(data, lw=2, bw_adjust=bw_adjust)
    
    else:
        sns.kdeplot(data, lw=2)
    
    # Add lines for ci
    ax.axvline(rv.interval(ci/100)[0], color='C4', lw=1.3) # CI lower, purple line
    ax.axvline(rv.interval(ci/100)[1], color='C4', lw=1.3, label='CI') # CI upper, purple line
    
    # Add lines for central tendency measures
    if central_tendency_measure == 'mean':
        ax.axvline(rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line
    
    elif central_tendency_measure == 'median':
        ax.axvline(rv.median(), color='C2', lw=1.3, label='Median') # median, green line
    
    elif central_tendency_measure == 'both':
        ax.axvline(rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line
        ax.axvline(rv.median(), color='C2', lw=1.3, label='Median') # median, green line
    else:
        raise Exception('Must enter \'mean\', \'median\', or \'both\' for kind parameter.')
    
    # Add line for test stat
    if test_stat is not None:
        ax.axvline(test_stat, color='C3', lw=1.3, label='Test Stat') # test_stat, red line

    # Labels and titles
    ax.set_xlabel(x_label)
    ax.set_title('KDE Plot')

    if legend:
        ax.legend()
    
    return fig, ax


def NormProbPlot(data, y_label='Values'):
    """Generates a normal probability plot for supplied data. 
     Especially useful for looking at sampling distribution data. 

    Args:
        data (array-like)): Data to be plotted. Must be one-dimensional.
        y_label (str): The label to use on the y-axis. Defaults to 'Values'.
    
    Returns:
        fig: The figure for the plot
        ax: The axis for the plot
    """
    # Convert to an array
    data = np.asarray(data)
       
    # Get the normal probability plot values
    xs, ys = NormalProbabilityValues(data)
    
    # Get the values to use to draw a fit line
    fit_xs, fit_ys = FitLine([min(xs),max(xs)], data.mean(), data.std())
    
    # Set up figure (single plot)
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)
    
    ax.plot(fit_xs, fit_ys, color='0.8')
    ax.plot(xs, ys)
    
    # Labels and titles
    ax.set_xlabel("Standard Deviations from the Mean")
    ax.set_ylabel(y_label)
    ax.set_title('Normal Probability Plot')
    
    return fig, ax


def AllNumVarDensityPlot(variables, data, kind='cdf', central_tendency_measure='mean'):
    """Generates a density plot for each numerical variable in a DataFrame.

    Args:
        variables (list): A list of the numerical variables in data (the supplied DataFrame)
        data (pandas DataFrame): The DataFrame that contains the numerical variables.
        kind (str, optional): The kind of density plot. Must be either 'cdf' or 'kde'. Defaults to 'cdf'.
        central_tendency_measure (str, optional): Can choose between 'mean', 'median', or 'both'. Defaults to 'mean'.

    Returns:
        fig: The figure for the plot
        axes: The axes for the plot
    """
    # Set up number of rows and height for the figure
    numvars = len(variables)
    nrows = ceil(numvars / 2)
    fig_size_y = 3 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=False)
    fig.set_size_inches(8, fig_size_y)     
    
    # Plot each variable
    for i, var in enumerate(variables):
        row = i//2
        pos = i % 2
        if numvars <= 2:
            ax=axes[pos]
        else:
            ax=axes[row][pos]
        
        # Compute rv for each variable (used in cdf and in central tendency measures)
        val,cnt = np.unique(data[var].dropna(), return_counts=True)
        var_rv = stats.rv_discrete(values=(val,cnt/sum(cnt)))
        
        # Plot either cdf or kde depending on the kind parameter
        if kind.lower() == 'cdf':
            ax.plot(var_rv.xk, var_rv.cdf(var_rv.xk))
            ax.set_ylabel('Cumulative Density')
        elif kind.lower() == 'kde':
            sns.kdeplot(data[var], lw=2, ax=ax)
            ax.set_xlabel('')
        else:
            raise Exception('Must enter either \'cdf\' or \'kde\' for kind parameter.')
        
        # Add lines for central tendency measures
        if central_tendency_measure == 'mean':
            ax.axvline(var_rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line

        elif central_tendency_measure == 'median':
            ax.axvline(var_rv.median(), color='C2', lw=1.3, label='Median') # median, green line

        elif central_tendency_measure == 'both':
            ax.axvline(var_rv.mean(), color='C1', lw=1.3, label='Mean') # mean, orange line
            ax.axvline(var_rv.median(), color='C2', lw=1.3, label='Median') # median, green line
        else:
            raise Exception('Must enter \'mean\', \'median\', or \'both\' for kind parameter.')
        
        ax.set_title(var)
        
    plt.tight_layout()
    
    return fig,axes


def AllCatVarCountPlot(variables, data):
    """Generates a count plot for each categorical variable in a DataFrame.

    Args:
        variables (list): A list of the categorical variables in data (the supplied DataFrame)
        data (pandas DataFrame): The DataFrame that contains the categorical variables.

    Returns:
        fig: The figure for the plot
        axes: The axes for the plot
    """
    # Set up number of rows and height for the figure
    numvars = len(variables)
    nrows = ceil(numvars / 2)
    fig_size_y = 3 * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=False)
    fig.set_size_inches(8, fig_size_y)     
    
    # Plot each variable
    for i, var in enumerate(variables):
        row = i//2
        pos = i % 2
        if numvars <= 2:
            ax=axes[pos]
        else:
            ax=axes[row][pos]
        
        sns.countplot(x=var, data=data, ax=ax)
        ax.set_title(var)
        ax.set_xlabel('')
        
    plt.tight_layout()
    
    return fig,axes


def TwoCatProportionPlot(x, hue, data):
    """Computes and plots the proportions (percentages) of one categorical variable within another. 
    An example would be the proportion of males and females within different population groups.

    Args:
        x (str): The x variable for the bar plot.
        hue (str): The variable that will split the bars into different colors within each x variable group.
        data (pandas DataFrame): The DataFrame that contains the x and hue variables.

    Returns:
        fig: The figure for the plot
        ax: The axis for the plot
    """
    # Set up figure and axis
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)
    
    # Build DataFrame for data to be plotted
    plot_df = (data.groupby(x)[hue]
               .value_counts(normalize=True)
               .mul(100)
               .rename('percentage')
               .reset_index()
               .rename(columns={'level_1':hue}))
    
    # Plot the data
    sns.barplot(x=x, y='percentage', hue=hue, data=plot_df, ax=ax)
    
    # Set y limits
    ax.set_ylim(0,100)
    
    return fig,ax


def Despine(ax, spines='topright', remove_y_ticks=False, remove_y_label=False, remove_x_ticks=False, remove_x_label=False):
    """Removes the spines surrounding a plot.

    Args:
        ax: The designated axis. If using seaborn save the plot to an axis variable. ex) g = sns.lineplot()
        spines (str, optional): Can choose 'all', 'toprightleft' or 'toprightbottom'. Defaults to 'topright'.
        remove_y_ticks: Remove values on the y-axis.
        remove_y_label: Remove the label on the y-axis.
        remove_x_ticks: Remove values on the x-axis.
        remove_x_label: Remove the label on the x-axis.
    """
    if spines == 'all':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0.0)
        if remove_y_ticks:
            ax.yaxis.set_ticks([])
        if remove_y_label:
            ax.set_ylabel('')
        if remove_x_ticks:
            ax.xaxis.set_ticks([])
        if remove_x_label:
            ax.set_xlabel('')
    
    elif spines == 'toprightleft':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0.0)
        if remove_y_ticks:
            ax.yaxis.set_ticks([])
        if remove_y_label:
            ax.set_ylabel('')
    
    elif spines == 'toprightbottom':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', length=0.0)
        if remove_x_ticks:
            ax.xaxis.set_ticks([])
        if remove_x_label:
            ax.set_xlabel('')
    
    else:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


def RemoveGridSpinesAxes(ax):
    """Removes grid, spines, and axes to leave just the plot itself.

    Args:
        ax: The designated axis. If using seaborn save the plot to an axis variable. ex) g = sns.lineplot()
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0.0)
    ax.grid(b=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def AnnotateBars(rects, color='0.4', orient='v', offset=3, weight='normal', fontsize=12, decimal_digits=0, percent=False):
    """Adds a text label to each bar in a bar plot. 
    If using seaborn, must save the plot to a variable to gain access to the axis.  
    Pass the patches as ax.patches for the rects parameter. 
    It's also possible to select specific bars only using selection or slicing (eg. ax.patches[4:]).

    Args:
        rects (matplotib patches): The bars to label. Pass as ax.patches.
        color (str, optional): Label color. Defaults to '0.4'.
        orient (str, optional): Orientation of the bars to be labelled. Defaults to 'v'.
        offset (int, optional): Offset from the top / right edge of the bars. Defaults to 3.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 0.
        percent (bool, optional): Choose to add percent sign. Defaults to False.
    """
    if orient == 'h':
        for rect in rects:
            width = rect.get_width()
            if percent == True:
                form='{:.'+str(decimal_digits)+'f'+'}%'
            else:
                form='{:.'+str(decimal_digits)+'f'+'}'
            plt.annotate(form.format(width),
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(offset, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=fontsize,
                    weight=weight, color=color)
    else:
        for rect in rects:
            height = rect.get_height()
            if percent == True:
                form='{:.'+str(decimal_digits)+'f'+'}%'
            else:
                form='{:.'+str(decimal_digits)+'f'+'}'
            plt.annotate(form.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=fontsize,
                        weight=weight, color=color)


def AnnotatePointAbove(coord, color='0.4', weight='normal', fontsize=12, ha='center', offset=10, decimal_digits=1):
    """Adds a label above a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """  
    form = '{:.' + str(decimal_digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(0, offset),
                 textcoords="offset points",
                 ha=ha, va='bottom',
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointBelow(coord, color='0.4', weight='normal', fontsize=12, ha='center', offset=10, decimal_digits=1):
    """Adds a label below a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(decimal_digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(0, -offset),
                 textcoords="offset points",
                 ha=ha, va='top',
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointLeft(coord, color='0.4', weight='normal', fontsize=12, va='center', offset=10, decimal_digits=1):
    """Adds a label to the left of a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(decimal_digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(-offset, 0),
                 textcoords="offset points",
                 ha='right', va=va,
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointRight(coord, color='0.4', weight='normal', fontsize=12, va='center', offset=10, decimal_digits=1):
    """Adds a label to the right of a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(decimal_digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(offset, 0),
                 textcoords="offset points",
                 ha='left', va=va,
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotateLineYs(xs, ys, color='0.4', weight='normal', fontsize=12, offset=10, decimal_digits=1):
    """Adds y coordinate annotations to all the points on a line. 
    Automatically adjusts annotation placement based on angles of incoming and outgoing lines of the points.

    Args:
        xs (array-like): Sequence of xs
        ys (array-like): Sequence of ys
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        decimal_digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """
    d = dict(zip(range(len(xs)), list(zip(xs,ys)))) # Dict of coordinates
    
    for k, coord in d.items():
        if k == 0: # Annotate first point in the line
            if d[k+1][1] > d[k][1]:
                AnnotatePointBelow(coord, color=color,
                                   weight=weight,fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
            else:
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
                
        elif k == len(xs)-1: # Annotate final point in the line
            if d[k-1][1] > d[k][1]:
                AnnotatePointBelow(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
            else:
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
                
        else: # Annotate all other points
            if (d[k-1][1] <= d[k][1]) & (d[k+1][1] <= d[k][1]): # Lines form a peak at the point
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
                
            elif (d[k-1][1] > d[k][1]) & (d[k+1][1] > d[k][1]): # Lines form a trough at the point
                AnnotatePointBelow(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, decimal_digits=decimal_digits)
                
            else:
                # Vectors to use to determine where the annotation needs to go in edge cases
                v0 = np.array(d[k-1]) - np.array(d[k]) # Incoming line vector
                v1 = np.array(d[k+1]) - np.array(d[k]) # Outgoing line vector
                v2 = np.array((d[k-1][0], d[k][1])) - np.array(d[k]) # Incoming horizontal vector
                v3 = np.array((d[k+1][0], d[k][1])) - np.array(d[k]) # Outgoing horizontal vector
                
                # Angles to use to determine where the annotation needs to go in edge cases
                int_angle = np.math.atan2(np.linalg.det([v0,v1]), np.dot(v0,v1)) # The signed interior angle between v0 and v1
                in_hangle = np.math.atan2(np.linalg.det([v0,v2]), np.dot(v0,v2)) # The signed angle between v0 and v2
                out_hangle = np.math.atan2(np.linalg.det([v1,v3]), np.dot(v1,v3)) # The signed angle between v1 and v3
                
                if (d[k-1][1] <= d[k][1]) & (int_angle >= 0): # Incoming line slopes up and interior angle is positive
                    if abs(np.degrees(in_hangle)) < 27: # Angle between incoming line and horizontal is small
                        AnnotatePointAbove(coord, color=color, ha='right',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                    else:
                        AnnotatePointLeft(coord, color=color, va='bottom',
                                          weight=weight, fontsize=fontsize,
                                          offset=offset, decimal_digits=decimal_digits)
                
                elif (d[k-1][1] <= d[k][1]) & (int_angle < 0): # Incoming line slopes up and interior angle is negative
                    if abs(np.degrees(out_hangle)) < 27: # Angle between outgoing line and horizontal is small
                        AnnotatePointBelow(coord, color=color, ha='left',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                    else:
                        AnnotatePointRight(coord, color=color, va='top',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                
                elif (d[k-1][1] >= d[k][1]) & (int_angle >= 0): # Incoming line slopes down and interior angle is positive
                    if abs(np.degrees(out_hangle)) < 27: # Angle between outgoing line and horizontal is small
                        AnnotatePointAbove(coord, color=color, ha='left',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                    else:
                         AnnotatePointRight(coord, color=color, va='bottom',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                
                else:
                    if abs(np.degrees(in_hangle)) < 27: # Angle between outgoing line and horizontal is small
                        AnnotatePointBelow(coord, color=color, ha='right',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)
                    else:
                        AnnotatePointLeft(coord, color=color, va='top',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, decimal_digits=decimal_digits)


def DollarThousandsFormat(value):
    """Formats a value into dollars with a thousands separator. Absolute value applied. 
    Used to add dollar signs to values on plots.

    Args:
        value (int or float): Value to be formatted

    Returns:
        string: formatted value
    """
    return '${:,.0f}'.format(abs(value))


def main():
    pass

if __name__ == '__main__':
    main()