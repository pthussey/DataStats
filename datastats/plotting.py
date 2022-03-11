import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Sets the default rc parameters for plotting
def SetParams(font='Malgun Gothic', basesize=12, basecolor='0.4', style='seaborn-whitegrid'):
    """Sets some matplotlib rcParams and a style.

    Args:
        font (str, optional): Choose the font. Defaults to 'Malgun Gothic'.
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


def Despine(ax, spines='topright'):
    """Removes the spines surrounding a plot.

    Args:
        ax: The designated axis. If using seaborn save the plot to an axis variable. ex) g = sns.lineplot()
        spines (str, optional): Can choose 'all' or 'toprightleft'. Defaults to 'topright'.
    """
    if spines == 'all':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0.0)
    elif spines == 'toprightleft':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0.0)
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


def AnnotateBars(rects, color='0.4', orient='v', offset=3, weight='normal', fontsize=12, digits=0, percent=False):
    """Adds a text label to each bar in a bar plot. 
    If using seaborn, must save plot to a variable 
    and pass the patches (ex. g.patches) for the rects parameter.

    Args:
        rects (matplotib patches): The bars to label.
        color (str, optional): Label color. Defaults to '0.4'.
        orient (str, optional): Orientation of the bars to be labelled. Defaults to 'v'.
        offset (int, optional): Offset from the top / right edge of the bars. Defaults to 3.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        digits (int, optional): Number of digits after the decimal point. Defaults to 0.
        percent (bool, optional): Choose to add percent sign. Defaults to False.
    """
    if orient == 'h':
        for rect in rects:
            width = rect.get_width()
            if percent == True:
                form='{:.'+str(digits)+'f'+'}%'
            else:
                form='{:.'+str(digits)+'f'+'}'
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
                form='{:.'+str(digits)+'f'+'}%'
            else:
                form='{:.'+str(digits)+'f'+'}'
            plt.annotate(form.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=fontsize,
                        weight=weight, color=color)


def AnnotatePointAbove(coord, color='0.4', weight='normal', fontsize=12, ha='center', offset=10, digits=1):
    """Adds a label above a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """  
    form = '{:.' + str(digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(0, offset),
                 textcoords="offset points",
                 ha=ha, va='bottom',
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointBelow(coord, color='0.4', weight='normal', fontsize=12, ha='center', offset=10, digits=1):
    """Adds a label below a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(0, -offset),
                 textcoords="offset points",
                 ha=ha, va='top',
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointLeft(coord, color='0.4', weight='normal', fontsize=12, va='center', offset=10, digits=1):
    """Adds a label to the left of a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(-offset, 0),
                 textcoords="offset points",
                 ha='right', va=va,
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotatePointRight(coord, color='0.4', weight='normal', fontsize=12, va='center', offset=10, digits=1):
    """Adds a label to the right of a point (x,y), for the y-value of the point. Must pass the coordinate as a tuple.

    Args:
        coord (tuple): An x,y coordinate.
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        ha (str, optional): Horizontal alignment of the label. Defaults to 'center'.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """ 
    form = '{:.' + str(digits) + 'f}'
    plt.annotate(form.format(coord[1]),
                 xy=(coord), xytext=(offset, 0),
                 textcoords="offset points",
                 ha='left', va=va,
                 fontsize=fontsize,
                 weight=weight, color=color)


def AnnotateLine(xs, ys, color='0.4', weight='normal', fontsize=12, offset=10, digits=1):
    """Adds labels to all the points on a line.

    Args:
        xs (array-like): Sequence of xs
        ys ([type]): Sequence of ys
        color (str, optional): Label color. Defaults to '0.4'.
        weight (str, optional): Font weight. Defaults to 'normal'.
        fontsize (int, optional): Font size. Defaults to 12.
        offset (int, optional): Offset from the coordinate. Defaults to 10.
        digits (int, optional): Number of digits after the decimal point. Defaults to 1.
    """
    d = dict(zip(range(len(xs)), list(zip(xs,ys)))) # Dict of coordinates
    
    for k, coord in d.items():
        if k == 0: # Annotate first point in the line
            if d[k+1][1] > d[k][1]:
                AnnotatePointBelow(coord, color=color,
                                   weight=weight,fontsize=fontsize,
                                   offset=offset, digits=digits)
            else:
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, digits=digits)
                
        elif k == len(xs)-1: # Annotate final point in the line
            if d[k-1][1] > d[k][1]:
                AnnotatePointBelow(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, digits=digits)
            else:
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, digits=digits)
                
        else: # Annotate all other points
            if (d[k-1][1] <= d[k][1]) & (d[k+1][1] <= d[k][1]): # Lines form a peak at the point
                AnnotatePointAbove(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, digits=digits)
                
            elif (d[k-1][1] > d[k][1]) & (d[k+1][1] > d[k][1]): # Lines form a trough at the point
                AnnotatePointBelow(coord, color=color,
                                   weight=weight, fontsize=fontsize,
                                   offset=offset, digits=digits)
                
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
                    if abs(np.degrees(in_hangle)) < 26: # Angle between incoming line and horizontal is small
                        print(1, k, np.degrees(in_hangle))
                        AnnotatePointAbove(coord, color=color, ha='right',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                    else:
                        AnnotatePointLeft(coord, color=color,
                                          weight=weight, fontsize=fontsize,
                                          offset=offset, digits=digits)
                
                elif (d[k-1][1] <= d[k][1]) & (int_angle < 0): # Incoming line slopes up and interior angle is negative
                    if abs(np.degrees(out_hangle)) < 26: # Angle between outgoing line and horizontal is small
                        print(2, k, np.degrees(out_hangle))
                        AnnotatePointBelow(coord, color=color, ha='left',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                    else:
                        AnnotatePointRight(coord, color=color,
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                
                elif (d[k-1][1] >= d[k][1]) & (int_angle >= 0): # Incoming line slopes down and interior angle is positive
                    if abs(np.degrees(out_hangle)) < 26: # Angle between outgoing line and horizontal is small
                        print(3, k, np.degrees(out_hangle))
                        AnnotatePointAbove(coord, color=color, ha='left',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                    else:
                        AnnotatePointRight(coord, color=color,
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                
                else:
                    if abs(np.degrees(out_hangle)) < 26: # Angle between outgoing line and horizontal is small
                        print(3, k, np.degrees(out_hangle))
                        AnnotatePointBelow(coord, color=color, ha='left',
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)
                    else:
                        AnnotatePointLeft(coord, color=color,
                                           weight=weight, fontsize=fontsize,
                                           offset=offset, digits=digits)


def CdfPlot(data, test_stat=None, mean=False, median=False, CI=False, conf_int=0.95):
    """Plots a cdf for supplied data.

    Args:
        data (array-like): Data to be plotted. Needs to be a one-dimensional sequence.
        test_stat (float, optional): Test stat to plot. Defaults to None.
        mean (bool, optional): If True plots a line at the mean. Defaults to False.
        median (bool, optional): If True plots a line at the median. Defaults to False.
        CI (bool, optional): If True plots lines for the confidence interval. Defaults to False.
        conf_int (float, optional): Sets the range for the confidence interval. Defaults to 0.95.
    """
    # Compute an rv for the data
    val,cnt = np.unique(data, return_counts=True)
    rv = stats.rv_discrete(values=(val,cnt/sum(cnt)))

    # Set up figure (single plot)
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)

    # Plot cdf of the rv computed above
    # Can include an orange line for the mean, a green line for the median, 
    # purple lines for the CI, and red lines for the test stat
    ax.plot(rv.xk, rv.cdf(rv.xk), lw=2.0, label='cdf') # pylint: disable=no-member
        
    if test_stat != None:
        ax.axvline(test_stat, color='C3', lw=1.3, label='test stat') # test_stat vertical, red line
        ax.axhline(rv.cdf(test_stat), color='C3', lw=1.3) # test_stat horizontal, red line
    
    if mean:
        ax.axvline(rv.mean(), color='C1', lw=1.3, label='mean') # mean, orange line
    
    if median:
        ax.axvline(rv.median(), color='C2', lw=1.3, label='median') # median, green line
    
    if CI:
        ax.axvline(rv.interval(conf_int)[0], color='C4', lw=1.3, label='CI') # CI lower, purple line
        ax.axvline(rv.interval(conf_int)[1], color='C4', lw=1.3) # CI upper, purple line
    
    ax.legend(bbox_to_anchor=(1.05, 1))


def KdePlot(data, test_stat=None, bw_adjust=None, clip=None, mean=False, median=False):
    """Plots a kde for supplied data.

    Args:
        data (array-like): Data to be plotted. Needs to be a one-dimensional sequence.
        test_stat (float, optional): Test stat to plot. Defaults to None.
        bw_adjust (float, optional): Adjusts the bandwidth for the kde. Defaults to None.
        clip (tuple, optional): Clips the data at the values given in the tuple. Defaults to None.
        mean (bool, optional): If True plots a line at the mean. Defaults to False.
        median (bool, optional): If True plots a line at the median. Defaults to False.
    """
    # Convert to an array
    data = np.asarray(data)

    # Set up figure (single plot)
    fig,ax = plt.subplots()
    fig.set_size_inches(8,6)

    # Plot kde of the data
    # Can include an orange line for the mean, 
    # a green line for the median, 
    # and a red line for the test stat
    if (bw_adjust != None) and (clip != None):
        sns.kdeplot(data, lw=2, bw_adjust=bw_adjust, clip=clip)
    
    elif (bw_adjust == None) and (clip != None):
        sns.kdeplot(data, lw=2, clip=clip)
    
    elif (bw_adjust != None) and (clip == None):
        sns.kdeplot(data, lw=2, bw_adjust=bw_adjust)
    
    else:
        sns.kdeplot(data, lw=2)
        
    if test_stat != None:
        ax.axvline(test_stat, color='C3', lw=1.3, label='test stat') # test_stat vertical, red line
    
    if mean:
        ax.axvline(np.mean(data), color='C1', lw=1.3, label='mean') # mean, orange line
    
    if median:
        ax.axvline(np.median(data), color='C2', lw=1.3, label='median') # median, green line
    
    ax.legend(bbox_to_anchor=(1.25, 1))


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