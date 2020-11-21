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
 
    """
    means = returns.mean()
    print('Mean stock returns: ')
    print(means)
    """
   
    n_cols = len(stock_norm.columns) -1

    for col in stock_norm.columns[1:]:
        returns[col] = f.daily_returns(data=stock_norm, col=col)

    # (annualized) risk free rate is set to zero
    rf = 0.00

    n_days = 252
    col_sp500 = returns.columns[-1]
    rm = returns[col_sp500].mean() * n_days
    print(f'SP500 annual return: {rm}')

    # Save the betas in this dictionary
    stock_beta = dict()
    stock_return = dict()

    for col in returns.columns[:-1]:
        b, a = f.beta(data=returns, col_stock=col, col_market=col_sp500)   
        #print(f'Stock = {col} has beta {b} and alpha {a}')
        
        # We use the CAPM to get the expected rate of return on a given stock (annualized)
        rs = f.capm(rf=rf, beta=b, rm=rm) * 100.0
        print(f'Stock = {col} should have an annualized rate of return = {rs}%')
        
        stock_beta[col] = b
        stock_return[col] = rs

    weights = 1.0 / 8.0 * np.ones(8)
    #portfolios = f.generate_portfolios(data=returns, w=weights, n_runs=1)

    er_portfolio = sum(list(stock_return.values()) * weights)
    print(f'Expected returns on the portfolio: {er_portfolio}')

    weights_new = np.zeros(8)

    # Test portfolio with apple and amazon only
    weights_new[0] = 0.5
    weights_new[4] = 0.5
    
    er_portfolio = sum(list(stock_return.values()) * weights_new)
    print(f'Expected returns on the portfolio: {er_portfolio}')


