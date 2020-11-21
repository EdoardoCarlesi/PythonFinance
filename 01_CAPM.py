import pandas as pd
import numpy as np
import functions as f

from scipy import stats
from copy import copy


if __name__ == "__main__":

    """ Main program """
    
    data_file = 'data/stock.csv'
    stock = pd.read_csv(data_file)
    stock_norm = f.normalize(data=stock)
    returns = pd.DataFrame()
    
    n_cols = len(stock_norm.columns) -1

    for col in stock_norm.columns[1:]:
        returns[col] = f.daily_returns(data=stock_norm, col=col)

    col_sp500 = returns.columns[-1]

    for col in returns.columns[1:-1]:
        b, a = f.beta(data=returns, col_stock=col, col_market=col_sp500)   
        
        print(f'Stock = {col} has beta {b} and alpha {a}')

    means = returns.mean()

    '''
    print(returns.head())
    print(means)
    print(stock_norm.head())

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
    '''




