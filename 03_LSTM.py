import pandas as pd
import numpy as np
import functions as f

from scipy import stats
from copy import copy

from tensorflow import keras
#import keras



if __name__ == "__main__":

    """ MAIN PROGRAM """
    
    # Read and normalize the data
    stock_file = 'data/stock.csv'
    volume_file = 'data/stock_volume.csv'
    stock = pd.read_csv(stock_file)
    volume = pd.read_csv(volume_file)
    stock_norm = f.normalize(data=stock)
 
    one_stock = f.individual_stock(price=stock, volume=volume, col='AAPL')
    one_stock = f.trading_window(data=one_stock)
    
    # Do the train - test splitting with normalized data
    X_train, y_train, X_test, y_test = f.prepare_data(data=one_stock, split_fac=0.7, LSTM=True)
    split = len(y_train)
    X_train = np.asarray(X_train)

    print(f'Train test split, train = {split}, Shape of X_Train/Test: {X_train.shape}')

    # Reshape the arrays
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f'Reshaping X_Train/Test: {X_train.shape}')

    # Deisgn the network
    inputs = keras.layers.Input(shape = (X_train.shape[1], X_train.shape[2]))

    n_units = 150
    x = keras.layers.LSTM(n_units,return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(n_units,return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(n_units)(x)

    """ 
        WARNING!!!!!
        
        The correct last layer before the output reads:
           x = keras.layers.LSTM(n_units)(x)

        if using: 
            x = keras.layers.LSTM(n_units,return_sequences=True)(x)
            x = keras.layers.Dropout(0.3)(x)

        the network is unable to learn!
    """

    output = keras.layers.Dense(1, activation='linear')(x)

    model = keras.models.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    
    n_epochs = 10
    n_batch = 32
    split_size = 0.2
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, validation_split=split_size)

    predictions = model.predict(X_test)
    
    pred = []
    for p in predictions:
        pred.append(p[0])

    data = pd.DataFrame()
    data['Date'] = stock['Date'].iloc[split:-2]
    print(len(data['Date']), len(y_test))
    data['True'] = y_test
    data['Pred'] = np.array(pred)
    f.interactive_plot(data=data, title='LSTM')





