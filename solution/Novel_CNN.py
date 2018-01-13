from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class NovelCnn:
    def __init__(self,input_size, n_classes,raw_feature_dim, batch_size=16, epochs=45):
        self._model = None
        self._batch_size = batch_size
        self._epochs = epochs

        NB_FILTER = [64, 128]
        NB_GRAM = [4, 3, 3]
        FULLY_CONNECTED_UNIT = 256
        DROPOUT = [0.7, 0.7]

        model = Sequential()
        model.add(Conv2D(
            NB_FILTER[0], (raw_feature_dim/2, NB_GRAM[0]),
            input_shape=input_size, border_mode='valid', activation='relu'))
        model.add(Conv2D(
            NB_FILTER[0], (2, NB_GRAM[0]),
            input_shape=input_size, border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Conv2D(
            NB_FILTER[0], (1, NB_GRAM[1]),
            border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Conv2D(
            NB_FILTER[0], (1, NB_GRAM[2]),
            border_mode='valid', activation='relu'))
        model.add(Conv2D(
            NB_FILTER[1], (1, NB_GRAM[2]),
            border_mode='valid', activation='relu'))
        model.add(Conv2D(
            NB_FILTER[1], (1, NB_GRAM[2]),
            border_mode='valid', activation='relu'))
        model.add(Conv2D(
            NB_FILTER[1], (1, NB_GRAM[2]),
            border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Flatten())
        model.add(Dropout(DROPOUT[0]))
        model.add(Dense(
            FULLY_CONNECTED_UNIT, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(DROPOUT[1]))
        model.add(Dense(
            FULLY_CONNECTED_UNIT, activation='relu', W_constraint=maxnorm(3)))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(
            loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])

        self._model = model

    def fit(self, xtrain, ytrain):
        self._model.fit(xtrain,ytrain,self._batch_size,self._epochs,verbose=1)

    def predict(self, xvalidate):
        ytest = self._model.predict_on_batch(xvalidate)
        return ytest

    def evaluation(self, xtest, ytest):
        return self._model.evaluate(xtest,ytest)



