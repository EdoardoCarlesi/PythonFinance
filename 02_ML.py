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
    #returns = pd.DataFrame()
 
    #print(volume.head())
    #print(stock.isnull().sum())
    #print(volume.isnull().sum())
    #print(stock.info())
     
    print(volume['AAPL'].mean())
    print(volume[volume.columns[-1]].max())

    print(stock.describe())
    print(volume.describe())
