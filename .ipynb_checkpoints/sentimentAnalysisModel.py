# Sentiment Analysis in South African vernac: Sesotho
# Lindo Khoza
# This is one of the components of the Marito Project

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    data = pd.read_csv('/home/lindo/Data/NLP/Marito/st_imbd')
    X = data['st']
    y = data['sentiment']
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    X_tokens = tokenizer.texts_to_sequences(X)
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X_tokens, y, test_size=0.33, random_state=42)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=20, padding='pre', truncating='pre')
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(10000, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')
    ])
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_split=0.20, epochs=30)
    print(history.history['loss'])
    model.save('st_sent2.h5')


if __name__ == "__main__":
    main()
