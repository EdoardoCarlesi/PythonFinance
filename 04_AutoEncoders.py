import pandas as pd
import numpy as np
import functions as f
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize

from scipy import stats
from copy import copy

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.optimizers import SGD


if __name__ == "__main__":
    """ MAIN PROGRAM """
    
    creditcard_df = pd.read_csv('data/Marketing_data.csv')
    creditcard_df.drop("CUST_ID", axis=1, inplace=True)
    creditcard_df.dropna(inplace=True) #("CUST_ID", axis=1, inplace=True)
    scaler = StandardScaler()
    creditcard_df_scaled = scaler.fit_transform(creditcard_df)

    input_df = Input(shape = (17, ))
    
    x = Dense(7, activation='relu')(input_df)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)
    
    encoded = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)

    x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)

    decoded = Dense(17, kernel_initializer='glorot_uniform')(x)

    # Autoencoder
    autoencoder = Model(input_df, decoded)

    # Encoder: this is the encoded, the compressed version
    encoder = Model(input_df, encoded)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    autoencoder.fit(creditcard_df_scaled, creditcard_df_scaled, batch_size=128, epochs=5, verbose=1)

    print(autoencoder.summary())

    pred = encoder.predict(creditcard_df_scaled)

    print(pred)

    





