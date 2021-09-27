# Deep-Learning-for-Business
The first assignment is to code and train a neural network in Python using only NumPy. This is in the file 'mmai 5500 assignment deep learning final.py'

The second assignment is where I implemented the deep portfolio method to find the 10 most communal stocks and the 20 least communal stocks. This is in the file 'copy_of_deep_learning_assignment_2_finalll_2ipynb (1).py'.
I trained an autoencoder to encode the timeseries of price data. The network architecture was implemented using Keras.
The autoencoder has a single layer of encoder weights, an encoded dimension of 5 (i.e. 5 hidden units) and a single layer of
decoder weights. Both weight layers are regularized with an L2 penalty. The encoder layer has ReLu activation
and the decoder layer sigmoid activation. The loss is mean squared error ( mse ). The Adam optimizer is recommended.
I used X_valid to find the best value for the L2 regularization parameter (lambda).
Once the network is trained and the best value for lambda was found, I selected the 10 most communal and the 20 least
communal stocks. The degree of communality was measured with the reconstruction loss.

The third assignment is where I did the following:
1. Reproduce the IBB: plot a comparision of the portfolio (from assignment 2) and the IBB on the validation data.
2. Create a modified IBB where all returns less than -5% are replaced by 5%.
3. Select a new portfolio based on the modified IBB.
4. Reproduce the modified IBB: plot a comparison of the new portfolio and the modified IBB on the validation data.
This is in the file 'official deep learning 3-Copy1 finallllllllllllllll (1).py'
