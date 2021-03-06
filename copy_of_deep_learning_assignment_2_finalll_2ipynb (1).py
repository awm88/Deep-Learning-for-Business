# -*- coding: utf-8 -*-
"""Copy of deep learning assignment 2 finalll 2ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_N8XSrgg4LuSVuOHsYZgNWk10u6UcLU-
"""

pip install pep8

#importing libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras import layers
import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import regularizers
import random #for random seed

#loading data and defining variables
def load_data(assign2_data):
  data = pd.read_csv('assign2_data.csv', index_col=0)
  X = data.values[:, 2:].astype('float64')
  years = data['year']
  X_train = X[years < 2020.]
  X_valid = X[years == 2020.]
  tmp = data.index[data['year'] == 2020.]
  tickers = np.array([ticker.rstrip('_2020') for ticker in tmp])
  random.seed(0)
  return X_train, X_valid, tickers

X_train, X_valid, tickers=load_data('assign2_data.csv')

X_train

X_train.shape[1]

#Initial Parameters: size of encoded
encoding_dim = 5 
input_features = X_train.shape[1]
reg_param = 1e-4
weight_reg = keras.regularizers.l2(reg_param) 
batch_size = 30

input_shape = X_train.shape[1:]
print(input_shape)

encoding_dim = 5

inputs=keras.Input(shape=input_shape,name='encoder_input')
input = tf.keras.layers.Input(shape=input_shape,name='encoder_input')

def model_autoencoder(kt):
  # "encoded" is the encoded representation of the input
  encoded = tf.keras.layers.Dense(encoding_dim, activation='relu',kernel_regularizer = weight_reg)(input)
  decoded = tf.keras.layers.Dense(250, activation='sigmoid',kernel_regularizer = weight_reg)(encoded)
  # This model maps an input to its reconstruction
  autoencoder = tf.keras.models.Model(input, decoded)
  encoder = tf.keras.models.Model(input, encoded)
  # create a placeholder for an encoded input
  encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))
  random.seed(0)
  return autoencoder


encoded = tf.keras.layers.Dense(encoding_dim, activation='relu',kernel_regularizer = weight_reg)(input)
decoded = tf.keras.layers.Dense(250, activation='sigmoid',kernel_regularizer = weight_reg)(encoded)
autoencoder = tf.keras.models.Model(input, decoded)
encoder = tf.keras.models.Model(input, encoded)
# create a placeholder for an encoded input
encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))

# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# create the decoder model
decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mse')
outputs = Activation('sigmoid', name='decoder_output')(input)

#test example of autoencoder with epoch and batch size
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_valid, X_valid))

#encoder summary
encoder = Model(input, encoded, name='encoder')
encoder.summary()

encoded_input

outputs

input

#test autoencoder with more epochs
autoencoder.fit(X_train, X_train, epochs=100, batch_size=256, shuffle=True, validation_data=(X_valid, X_valid))

#testing autoencoder with more batch size 
autoencoder.fit(X_train, X_train, epochs=50, batch_size=400, shuffle=True, validation_data=(X_valid, X_valid))

#decoder summary
decoder.summary()

#autoencoder summary
autoencoder.summary()

data = pd.read_csv('assign2_data.csv', index_col=0)
all_ticks= np.array(data.index.str.split("_").tolist(),dtype=str)[:,0]
print(all_ticks.shape)

predict_valid = autoencoder.predict(X_valid)
model_df = pd.DataFrame(data = X_valid, index = tickers)

mse = np.square(X_valid - predict_valid).sum(axis=1)
#finding the two-norm difference
model_df["mse"] = mse

model_df = model_df.sort_values(by=["mse"])

#model creation
def simple_encoder(input_features, encoding_dim, weight_reg):
    input_data = keras.Input(shape=(input_features,))
    # "encoded" is the encoded representation of the input
    encoded = keras.layers.Dense(encoding_dim, activation='relu',
                                kernel_regularizer = weight_reg )(input_data)
    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Dense(input_features, activation='sigmoid',
                                kernel_regularizer = weight_reg )(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_data, decoded)

    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def optimize_encoder(X_train, X_valid, tickers, 
                     reg_params = [1e-2, 1e-3, 1e-4],
                     batch_sizes = [10, 25, 40],
                     epochs_ls = [10, 40, 70, 100, 150],
                     encoding_dim = 5):
    input_features = X_train.shape[1]
    best_mse = 5000
    model_params = []
    best_params = []
    for reg_param in reg_params:
        weight_reg = keras.regularizers.l2(reg_param) 
        for batch_size in batch_sizes:
            for epoch in epochs_ls:
                autoencoder = simple_encoder(input_features, encoding_dim, weight_reg)
                print()
                print("#"*50)
                autoencoder.fit(X_train, X_train,
                                validation_data=(X_valid, X_valid),
                                epochs = epoch,
                                batch_size = batch_size)

                predict_valid = autoencoder.predict(X_valid) 
                mse_per_stock = np.square(X_valid - predict_valid).mean(axis=1)
                overall_mse = mse_per_stock.mean()

                param = {"reg_param":reg_param,
                                     "batch_size":batch_size,
                                     "epoch":epoch,
                                     "overall_mse":overall_mse}
                print(param)
                model_params.append(param) 

                if overall_mse < best_mse:
                    best_params = [param, autoencoder]
                    best_mse = overall_mse
                    
    autoencoder = best_params[1]
    predict_valid = autoencoder.predict(X_valid)
    mse_per_stock = np.square(X_valid - predict_valid).mean(axis=1)

    output_data = pd.DataFrame(data = predict_valid, index = tickers)
    output_data["mse"] = mse_per_stock
    output_data = output_data.sort_values(by=["mse"])

    return model_params, best_params, output_data

model_params, best_params, output_data = optimize_encoder (X_train, X_valid, tickers, 
                                                         reg_params = [1e-2, 1e-3, 1e-4],
                                                         batch_sizes = [ 10, 25, 40],
                                                         epochs_ls = [10,40,70,100, 150],
                                                         encoding_dim = 5)

model_params, best_params, output_data = optimize_encoder (X_train, X_valid, tickers, 
                                                         reg_params = [1e-4,1e-5,1e-6],
                                                         batch_sizes = [30,60,90],
                                                         epochs_ls = [150],
                                                         encoding_dim = 5)

tickers

predict_valid = autoencoder.predict(X_valid)

X_valid

predict_valid

mse = np.square(X_valid - predict_valid).mean(axis=1)
#finding the two-norm difference
model_df["mse"] = mse

model_df = model_df.sort_values(by=["mse"])

mse

model_df

model_df = pd.DataFrame(data = predict_valid, index = tickers)

mse = np.square(X_valid - predict_valid).mean(axis=1)

model_df["mse"] = mse

model_df = model_df.sort_values(by=["mse"])

model_df

most_communal = model_df.head(10).index

most_communal #10 most communal stocks with the smallest distance

least_communal = model_df.tail(20).index

least_communal #20 of the least communal stocks with the greatest distance