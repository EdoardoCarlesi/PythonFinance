import pandas as pd
import numpy as np
import functions as f

from scipy import stats
from copy import copy



if __name__ == "__main__":

    """ Main program """
    
    stock_file = 'data/stock.csv'
    volume_file = 'data/stock_volume.csv'
    stock = pd.read_csv(stock_file)
    volume = pd.read_csv(volume_file)
    stock_norm = f.normalize(data=stock)
 
    '''
    print(volume.head())
    print(stock.isnull().sum())
    print(volume.isnull().sum())
    print(stock.info())
    print(volume['AAPL'].mean())
    print(volume[volume.columns[-1]].max())
    print(stock.describe())
    print(volume.describe())
    '''

    one_stock = f.individual_stock(price=stock, volume=volume, col='AAPL')
    one_stock = f.trading_window(data=one_stock)
    
    # Do the train - test splitting with normalize data
    X_train, y_train, X_test, y_test = f.prepare_data(data=one_stock, split_fac=0.65)
    split = len(y_train)

    print(f'Train test split, train = {split}')

    # Show some plots
    #f.show_plot(X_train, 'Training data')
    #f.show_plot(X_test, 'Testing data')

    # Do some linear regression




