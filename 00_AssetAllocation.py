import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import utils as u

from scipy import stats
from copy import copy


def normalize(data=None):
    x = data.copy()

    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]

    return x


def normalized_random(N):

    numbers = [np.random.random() for i in range(0, N)]
    norm = sum(numbers)
    numbers = [num / norm for num in numbers]

    return numbers


def make_portfolio(data=None, w=None):
    
    data_w = pd.DataFrame()

    for i, col in enumerate(data.columns[1:]):
        data_w[col] = data[col] * w[i-1]

    return data_w


def interactive_plot(data=None, title=None):

    fig = px.line(title = title)
    
    for i in data.columns[1:]:
        fig.add_trace(go.Scatter(x = data['Date'], y = data[i], mode="lines", name = i))

    fig.show()


def montecarlo(data=None, n_runs=1):
    """ 
        Assign random asset allocations n times 
        - data should be normalized to the starting price of each stock
        - n_runs is the number of montecarlo trials
    """

    portfolios = pd.DataFrame()
    portfolios['Date'] = data['Date']
    data_cols = data.columns[1:]

    for i_mc in range(0, n_runs_mc):
        w = normalized_random(n_cols)
        portfolio = make_portfolio(data=data, w=w)       

        key_p = 'P_' + str(i_mc)        
        
        portfolios[key_p] = portfolio[data_cols].apply(lambda x: sum(x), axis = 1)

    return portfolios


if __name__ == "__main__":

    """ Main program """
    
    data_file = 'data/stock.csv'
    stock = pd.read_csv(data_file)
    #print(stock.head())

    # Plot raw data
    #interactive_plot(data = stock, title = 'Prices')

    # Normalize data and plot it
    stock_norm = normalize(data=stock)
    #interactive_plot(data = stock_norm, title = 'Normalized Prices')
    
    n_cols = len(stock_norm.columns) -1
    n_runs_mc = 100

    portfolios = montecarlo(data=stock_norm, n_runs=n_runs_mc)
    print(portfolios.head())

    interactive_plot(data=portfolios, title='MC')




