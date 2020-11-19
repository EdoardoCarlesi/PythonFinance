import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import utils as u

from scipy import stats
from copy import copy


def normalize(data=None):
    """
        Normalize all data to 1.0 at their starting values
    """

    x = data.copy()

    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]

    return x


def normalized_random(N):
    """
        Returns a list of N random numbers, ensuring the total is normalized to one
    """

    numbers = [np.random.random() for i in range(0, N)]
    norm = sum(numbers)
    numbers = [num / norm for num in numbers]

    return numbers


def make_portfolio(data=None, w=None):
    """
        This function takes a normalized dataset as an input and returns a new dataframe 
        whose columns are the different assets multiplied by the weights
    """

    data_w = pd.DataFrame()

    for i, col in enumerate(data.columns[1:]):
        data_w[col] = data[col] * w[i-1]

    return data_w


def daily_returns(data=None, col=None):
    """
        Compute the daily returns of a single stock / portfolio
    """

    r = []

    # Initialize to zero on the first day of counting
    r.append(0.0)

    for i in range(1, len(data)):
        r.append((data[col][i] - data[col][i-1])/data[col][i-1])

    return np.array(r)


def cumulative_returns(data=None, col=None):
    """
        Compute the final total return of a single stock / portfolio
    """

    return (data[col][-1:] - data[col][0])/data[col][0]


def interactive_plot(data=None, title=None, mode="lines"):
    """
        This is an interactive plot generator for the browser, using plotly
    """

    if mode == "lines":
        fig = px.line(title = title)
    
        for col in data.columns[1:]:
            fig.add_trace(go.Scatter(x = data['Date'], y = data[col], mode=mode, name = col))

    # In histogram mode we assume data is already 1-dimensional
    elif mode == "histogram":
        fig = px.histogram(data, title = title)

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


def sharpe_ratio(Rp=None, Rf=None, sigma=None):
    """
        The Sharpe ratio is defined as 
        SR = (R_p - R_f) / sigma_p
        R_p = return of the portfolio
        R_f = risk free return
        sigma_p = volatility (std dev) of my portfolio
    """

    return (Rp - Rf) / sigma


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
    n_runs_mc = 3

    portfolios = montecarlo(data=stock_norm, n_runs=n_runs_mc)
    #print(portfolios.head())

    #interactive_plot(data=portfolios, title='MC')

    for i, col in enumerate(portfolios.columns[1:]):
        return_col = 'R_' + str(i)
        portfolio_col = 'P_' + str(i)
        portfolios[return_col] = daily_returns(data=portfolios, col=col)
        
        # Plot just one series of return
        df_tmp = portfolios[['Date', return_col]]
        #interactive_plot(data = df_tmp, title = 'Daily returns')

        #interactive_plot(data=portfolios[new_col], mode="histogram", title="Returns")
        rf = 0.00
        rp = portfolios[return_col].mean()
        sigma = portfolios[return_col].std()
        cumret = cumulative_returns(portfolios, col=portfolio_col).values[0]
        sr = sharpe_ratio(Rf=rf, Rp=rp, sigma=sigma)
        sr *= np.sqrt(252) # Adjust to the whole year

        print(f'Portfolio cumulative returns: {cumret}')
        print(f'Portfolio standard deviation: {sigma}')
        print(f'Portfolio avg. daily return : {rp}')
        print(f'Sharpe ratio: {sr}')
    




