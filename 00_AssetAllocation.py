import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import pandas as pd
import numpy as np

from scipy import stats
from copy import copy

import functions as f


if __name__ == "__main__":

    """ Main program """
    
    data_file = 'data/stock.csv'
    stock = pd.read_csv(data_file)
    #print(stock.head())

    # Plot raw data
    #interactive_plot(data = stock, title = 'Prices')

    # Normalize data and plot it
    stock_norm = f.normalize(data=stock)
    #interactive_plot(data = stock_norm, title = 'Normalized Prices')
    
    n_cols = len(stock_norm.columns) -1
    n_runs_mc = 3

    portfolios = f.montecarlo(data=stock_norm, n_runs=n_runs_mc)
    #print(portfolios.head())

    #interactive_plot(data=portfolios, title='MC')

    for i, col in enumerate(portfolios.columns[1:]):
        return_col = 'R_' + str(i)
        portfolio_col = 'P_' + str(i)
        portfolios[return_col] = f.daily_returns(data=portfolios, col=col)
        
        # Plot just one series of return
        df_tmp = portfolios[['Date', return_col]]
        #interactive_plot(data = df_tmp, title = 'Daily returns')

        #interactive_plot(data=portfolios[new_col], mode="histogram", title="Returns")
        rf = 0.00
        rp = portfolios[return_col].mean()
        sigma = portfolios[return_col].std()
        cumret = f.cumulative_returns(portfolios, col=portfolio_col).values[0]
        sr = f.sharpe_ratio(Rf=rf, Rp=rp, sigma=sigma)
        sr *= np.sqrt(252) # Adjust to the whole year

        print(f'Portfolio cumulative returns: {cumret}')
        print(f'Portfolio standard deviation: {sigma}')
        print(f'Portfolio avg. daily return : {rp}')
        print(f'Sharpe ratio: {sr}')
    




