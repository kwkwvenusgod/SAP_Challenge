from keras import Sequential
from keras.layers import Embedding, LSTM, Dense


class NovelRNN:
    def __init__(self, raw_feature_dim=None, n_classes=None, batches=32, epochs=30):
        self._model = None
        self._batches = batches
        self._epochs = epochs

        model = Sequential()
        model.add(Embedding(raw_feature_dim, 128))
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2,
                       activation='tanh', return_sequences=True))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, activation='tanh'))
        model.add(Dense(n_classes, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                      metrics=['accuracy'])
        self._model = model

    def fit(self,xtrain, ytrain):
        self._model.fit(xtrain,ytrain, batch_size=self._batches, epochs=self._epochs)

    def predict(self,xtest):
        return self._model.predict(xtest)

    def evaluate(self, x, y):
        return self._model.evaluate(x,y)

