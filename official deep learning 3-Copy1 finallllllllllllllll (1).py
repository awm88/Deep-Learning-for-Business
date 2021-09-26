#!/usr/bin/env python
# coding: utf-8

# In[475]:


import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[476]:


def modify_target(target, threshold=-0.05):
    """
    threshold has to be negative
    """
    cols = target.columns[1:]
    # Get the returns
    returns = target[cols].pct_change(axis=1)
    # Remove drawdowns less than threshold
    returns[returns < threshold] = np.abs(threshold)
    # Reconstruct
    target_mod = target.copy()
    target_mod[cols] = returns
    target_mod[cols[0]] = target[cols[0]]

    for i, col in enumerate(cols[1:]):
        target_mod[col] = target_mod[cols[i]] * (target_mod[col] + 1)

    return target_mod


# In[477]:


def load_modify_normalize(assign3_data, assign3_benchmark):
    """
    Loads, modifies (the IBB index only) and normalizes the data.

    Arguments
    ---------
        data_fname      - file name (including path) for the data file
                          (e.g. assign3_data.csv)
        target_fname    - file name (including path) for the file containing
                          the IBB index (e.g. assign3_benchmark.csv)
    Returns
    -------
        X_train         - shape: (n_stocks * 4, n_times), training data,
                          normalized stock prices.
        X_valid         - shape: (n_stocks, n_times), validation data,
                          normalized stock prices.
        Y_train         - shape: (n_times * 4, 1), training data,
                          normalized IBB index.
        Y_valid         - shape: (n_times, 1), validation data,
                          normalized IBB index.
        Y_train_mod     - shape: (n_times * 4, 1), training data,
                          modified and normalized IBB index.
        Y_valid_mod     - shape: (n_times * 4, 1), validation data,
                          modified and normalized IBB index.
        tickers         - List of the ticker symbols
    """
    data = pd.read_csv('assign3_data.csv', index_col=0)
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data.values[:, 1:].T)
    data.values[:, 1:] = min_max_scaler.transform(data.values[:, 1:].T).T
    X_train = data[data['year'] < 2020].values[:, 1:]
    X_valid = data[data['year'] >= 2020].values[:, 1:]
    tmp = data.index[data['year'] == 2020.]
    # List of the ticker symbols
    tickers = np.array([ticker.rstrip('_2020') for ticker in tmp])

    # Benchmark, i.e. target, IBB
    target = pd.read_csv('assign3_benchmark.csv', index_col=0)
    # Modify the target to remove drawdowns
    target_mod = modify_target(target, threshold=-0.05)
    # Rescale the traget
    min_max_scaler.fit(target.values[:, 1:].T)
    target.values[:, 1:] = min_max_scaler.transform(target.values[:, 1:].T).T
    target_mod.values[:, 1:] = min_max_scaler.transform(target_mod.values[:, 1:].T).T
    # Split into train and valid
    Y_train = target[target['year'] < 2020].values[:, 1:]
    Y_valid = target[target['year'] >= 2020].values[:, 1:]
    Y_train_mod = target_mod[target_mod['year'] < 2020].values[:, 1:]
    Y_valid_mod = target_mod[target_mod['year'] >= 2020].values[:, 1:]
    # Reshape the Y_train_mod_n to (n_times*4, 1)
    # & Y_valid_mod_n to (n_times, 1)
    Y_train_mod = Y_train_mod.reshape(-1, 1)
    Y_valid_mod = Y_valid_mod.reshape(-1, 1)

    return X_train, X_valid, Y_train, Y_valid, Y_train_mod, Y_valid_mod, tickers


# In[478]:


X_train, X_valid, Y_train, Y_valid, Y_train_mod, Y_valid_mod, tickers = load_modify_normalize('assign3_data.csv', 'assign3_benchmark')


# In[479]:


X_train


# In[480]:


X_train.shape


# In[481]:


encoding_sz = 5
epochs = 150
batch_sz = 32
n_most_communal = 5
n_least_communal = 20
lambdas = np.arange(0.05, 0.21, 0.01)


# In[482]:


def build_autoencoder(lmbd, n_times):
    """
    """
    inputs = keras.Input(shape=(n_times,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_sz, activation='relu',
                           kernel_regularizer=regularizers.l2(lmbd))(inputs)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(n_times, activation='sigmoid',
                           kernel_regularizer=regularizers.l2(lmbd))(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


# In[483]:


def train_autoencoder(lambdas, X_train, X_valid):
    """
    """
    n_times = X_valid.shape[1]
    val_losses = []
    best_epochs = []
    for lmbd in lambdas:

        autoencoder = build_autoencoder(lmbd, n_times)

        history = autoencoder.fit(X_train, X_train, epochs=epochs,
                                  batch_size=batch_sz, shuffle=True,
                                  validation_data=(X_valid, X_valid),
                                  verbose=0)

        min_val_loss = np.min(history.history['val_loss'])
        best_epochs.append(np.argmin(history.history['val_loss'])+1)

        if (min_val_loss < val_losses).all():
            best_history = history.history
            best_lambda = lmbd
            best_epoch = best_epochs[-1]

        val_losses.append(min_val_loss)

        return val_losses, best_history, best_lambda, best_epoch


# In[484]:


val_losses, best_history, best_lambda, best_epoch = train_autoencoder(lambdas, X_train, X_valid)


# In[485]:


val_losses


# In[486]:


def select_portfolio(X_train, X_valid, best_lambda, best_epoch,
                     tickers, n_most_communal=5, n_least_communal=20):
    """
    """
    n_times = X_valid.shape[1]
    autoencoder = build_autoencoder(best_lambda, n_times)
    autoencoder.fit(X_train, X_train, epochs=best_epoch,
                    batch_size=batch_sz, shuffle=True, verbose=0)

    losses = np.zeros(X_valid.shape[0])
    for i, x in enumerate(X_valid):
        x = x.reshape((1, -1))
        losses[i] = autoencoder.evaluate(x, x, verbose=0)

    ids = np.argsort(losses)
    most_communal_ids = ids[:n_most_communal]
    least_communal_ids = ids[-n_least_communal:]
    communal_tickers = {'most': tickers[most_communal_ids],
                        'least': tickers[least_communal_ids]}
    portfolio_ids = np.r_[most_communal_ids, least_communal_ids]

    # Transpose the data since now a sample will be the
    # portfolio at a single time point.
    X_valid_port = X_valid[portfolio_ids].T
    # In X_train the stocks are repreated 4 times
    X_train_port = []
    for i in range(1, 5):
        X_train_port.append(X_train[portfolio_ids * i].T)
    X_train_port = np.concatenate(X_train_port)

    return X_train_port, X_valid_port, portfolio_ids, communal_tickers


# In[487]:


X_train_port, X_valid_port, portfolio_ids, communal_tickers = select_portfolio(X_train, X_valid, best_lambda, best_epoch,
                                                                               tickers, n_most_communal=5, n_least_communal=20)


# In[488]:


portfolio_ids


# In[489]:


communal_tickers


# In[490]:


X_train_port.shape


# In[491]:


X_train.shape


# In[492]:


X_valid.shape


# In[493]:


Y_valid.shape


# In[494]:


model = keras.Sequential([
    keras.layers.Dense(64, activation=tensorflow.nn.relu, input_shape=[25]),
    keras.layers.Dense(64, activation=tensorflow.nn.relu),
    keras.layers.Dense(1)
  ])

optimizer = tensorflow.keras.optimizers.RMSprop(0.001)


# In[495]:


optimizer = tensorflow.keras.optimizers.RMSprop(0.001)

model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])


# In[496]:


Y_train_mod


# In[497]:


X_train_port


# In[498]:


Y_train_mod.shape


# In[499]:


Y_train.shape


# In[500]:


X_valid.shape


# In[501]:


X_valid_port.shape


# In[502]:


Y_valid_mod.shape


# In[503]:


X_train_port.shape


# In[504]:


model.fit(X_train_port, Y_train_mod, epochs=500)


# In[505]:


X_valid.shape


# In[506]:


X_valid_port.shape


# In[507]:


trainScore = model.evaluate(X_train_port, Y_train_mod, verbose=0)
print(trainScore)


# In[508]:


model.summary()


# In[509]:


model.weights


# In[510]:


predtest = model.predict(X_valid_port)


# In[511]:


Y_valid = np.transpose(Y_valid)


# In[512]:


x_ax = range(len(X_valid_port))
plt.plot(x_ax, Y_valid_mod[:, 0], label="modified IBB index", color="m")
plt.plot(x_ax, predtest[:, 0], label="predicted performance")
plt.plot(x_ax, Y_valid[:, 0], label="unmodified IBB index")
plt.legend()
plt.show()
