# Lindo Khoza
# Toy Machine Learning Models
# A simple linear model using Tensorflow 2
# 08 October 2019

import numpy as np
import  tensorflow as tf
import matplotlib.pyplot as plt


def simulate_data(n, m):
    """ simulate random linear data

    :param n: number of predictors
    :param m: number of rows
    :return: an artificial data of a linearly dependent model
    """
    c = np.random.normal(size = n)
    X = np.random.uniform(size = (m, n))
    b = np.random.normal(size = m)
    y = X.dot(c) + b
    print('underlying process:', c)
    return X, y


def main():
    X, y = simulate_data(20, 1000)
    out_layer = tf.keras.layers.Dense(units=1,
                               input_shape=[X.shape[1]])
    model = tf.keras.Sequential([out_layer])
    learning_rate = 0.1
    model.compile(loss = 'mse',
                  optimizer = tf.keras.optimizers.Adam(learning_rate))
    history = model.fit(X, y, epochs= 200, verbose= False)
    plt.plot(history.history['loss'])
    plt.show()
    print('trained layer:', out_layer.weights)


if __name__ == "__main__":
    main()