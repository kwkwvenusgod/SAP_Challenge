from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class NovelCnn:
    def __init__(self,input_size=None, n_classes=None,raw_feature_dim=None, batch_size=16, epochs=30):
        self._model = None
        self._batch_size = batch_size
        self._epochs = epochs

        NB_FILTER = [64, 128]
        NB_GRAM = [4, 3, 3]
        FULLY_CONNECTED_UNIT = 256
        DROPOUT = [0.3, 0.3]

        model = Sequential()

        model.add(Conv2D(
            NB_FILTER[0], (raw_feature_dim, NB_GRAM[0]),
            input_shape=input_size, border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1,3)))

        model.add(Conv2D(
            NB_FILTER[0], (1, NB_GRAM[0]),
            input_shape=input_size, border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))

        model.add(Conv2D(
            NB_FILTER[0], (1, NB_GRAM[1]),
            input_shape=input_size, border_mode='valid', activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))

        # model.add(Conv2D(
        #     NB_FILTER[0], (1, NB_GRAM[2]),
        #     border_mode='valid', activation='relu'))
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
        ytest = self._model.predict(xvalidate)
        return ytest

    def evaluation(self, xtest, ytest):
        return self._model.evaluate(xtest,ytest)

    def save_ncnn_model(self, savepath):
        save_model(self._model, savepath)

    def load_ncnn_model(self, modelpath):
        self._model = load_model(modelpath, )



