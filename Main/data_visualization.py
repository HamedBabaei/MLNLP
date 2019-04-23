"""
This is an easy and simple python library too draw plots without any difficulty!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def bar(x , y , xlabel = 'X Label', ylabel = 'Y Label', title = "Bar Plot"):
    '''
    Drawing a BAR plot
    Example of usage:
    >>> import data_visualization as dv
    >>> dv.bar(x = ('a' , 'b' , 'c', 'd' , 'f') , y = [1, 2, 3 , 4 , 5])
    or
    >>> dv.bar(x = ('a' , 'b' , 'c', 'd' , 'f') , y = [1, 2, 3 , 4 , 5] , xlabel = "chars" , ylabel = "Char Counts")
    '''
    plt.style.use('ggplot')
    y_pos = np.arange(len(x))
    plt.bar(y_pos, y, align='center')
    plt.xticks(y_pos, x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def barh(x , y , xlabel = 'X Label', ylabel = 'Y Label', title = "Barh Plot"):
    '''
    Drawing a Barh plot
    Example of usage:
    >>> import data_visualization as dv
    >>> dv.barh(x = ('a' , 'b' , 'c', 'd' , 'f') , y = [1, 2, 3 , 4 , 5])
    or
    >>> dv.bar(x = ('a' , 'b' , 'c', 'd' , 'f') , y = [1, 2, 3 , 4 , 5] , xlabel = "Char Counts", ylabel = "chars")
    '''
    plt.style.use('ggplot')
    y_pos = np.arange(len(x))
    plt.barh(y_pos, y, align='center')
    plt.yticks(y_pos, x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatter(x , y, colors, xlabel = 'X Label', ylabel = 'Y Label', title = "Scatter Plot"):
    '''
    Drawing a Scatter plot
    Example of usage:
    >>> x = np.random.normal(size = 100)
    >>> y = np.random.normal(size = 100)
    >>> c = np.random.normal(size = 100)
    >>> scatter(x , y,c)
    '''
    plt.style.use('ggplot')
    plt.scatter(x, y , c = colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def hist( data , legend_loc = 'best', labels = {}, xlabel = 'X Label', ylabel = 'Y Label', title = "Hist Plot"):
    '''
    Drawing a Hist plot
    Example of usage:
    >>> x = np.random.normal(size = 100)
    >>> y = np.random.normal(size = 100)
    >>> c = np.random.normal(size = 100)
    >>> d = np.random.normal(size = 100)
    >>> hist(data = (x , y,c , d), labels={'x' , 'y' , 'c', 'd'})
    >>> hist(data = (x , y, c , d)
    '''
    plt.style.use('ggplot')
    plt.hist(data , label = labels)
    plt.legend(loc = legend_loc , framealpha = 0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def distplot(x , hist = True , xlabel = 'X Label', ylabel = 'Y Label', title = "Univariate Distribution Plot"):
    '''
    Univariate distributions
    hist = True will draw the distribution with hist plot
    hist = False will draw the distribution without hist plot
    Usage example:
    >>> x = np.random.normal(size = 100)
    >>> y = np.random.normal(size = 100)
    >>> distplot(x)
    >>> distplot(y , False)    
    '''
    sns.set(style="darkgrid")
    sns.distplot(x , hist=hist)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.show()

def kdeplot(x , x2, shade = True , xlabel = 'X Label', ylabel = 'Y Label', title = "Kernel Density Estimation Plot"):
    '''
    Kernel Density Estimation Plot
    Usage example:
    >>> x = np.random.normal(size = 100)
    >>> y = np.random.normal(size = 100)
    >>> kdeplot(x)
    '''
    sns.kdeplot(x , x2 , shade = True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def jointplot(x , y , data , kind = 'hex' , shade = False, xlabel = 'X Label', 
                ylabel = 'Y Label', title = ""):
    '''
    Joint plot
    Usage example:
    >>> x = np.random.normal(size = 100)
    >>> y = np.random.normal(size = 100)
    >>> z = np.random.normal(size = 100)
    >>> w = np.random.normal(size = 100)
    >>> data={"x_data":x , "y_data":y , "z_data":z , "w_data":w}
    >>> jointplot("x_data" , "y_data" , data , kind = "kde" , shade = False)
    '''
    sns.jointplot(x , y , data = data , kind= kind , shade = shade )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def pieplot(x , y , colors):
    '''
    Pie Plot
    Usage Example:
    >>> x = ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake']
    >>> y = [38.4, 40.6, 20.7, 10.3]
    >>> colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    >>> pieplot(x , y , colors )
    '''
    patches, _ = plt.pie(y, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, x, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
