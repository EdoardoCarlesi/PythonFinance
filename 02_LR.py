import pandas as pd
import numpy as np
import functions as f

from scipy import stats
from copy import copy

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


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

    # Do a linear regression with Ridge to avoid overfitting
    regression_model = Ridge(alpha=1.2, fit_intercept=False)
    #regression_model = LinearRegression()

    regression_model.fit(X_train, y_train)

    #Ridge(alpha=0.8, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='auto', tol=0.0001)

    score = regression_model.score(X_test, y_test)
    print('Ridge Regression Score: ', score)

    predictions = regression_model.predict(X_test)

    data = pd.DataFrame()
    data['Date'] = stock['Date'].iloc[split:-1]
    print(len(data['Date']), len(y_test))
    data['True'] = y_test
    data['Pred'] = predictions

    f.interactive_plot(data=data, title='Ridge model')




