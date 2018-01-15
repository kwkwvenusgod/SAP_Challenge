from keras.models import Model
from keras.models import Sequential
from keras.models import save_model
from keras.models import load_model
from keras.layers import Dense, Merge
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


class NovelCnn:
    def __init__(self,input_size=None, n_classes=None,raw_feature_dim=None, batch_size=8, epochs=15):
        self._model = None
        self._batch_size = batch_size
        self._epochs = epochs

        NB_FILTER = [64, 128]
        Parallel_N_GRAM = [3, 4, 5]
        NB_GRAM = [4, 3, 3]
        FULLY_CONNECTED_UNIT = 256
        DROPOUT = [0.5, 0.5]

        conv_blocks = []
        inp = Input(shape=input_size)
        for n_g in Parallel_N_GRAM:
            conv = Conv2D(
            NB_FILTER[0], (raw_feature_dim, n_g),
            input_shape=input_size, border_mode='valid', activation='relu')(inp)
            conv = MaxPooling2D(pool_size=(1,3))(conv)
            conv_blocks.append(conv)

        if len(Parallel_N_GRAM)>0:
            out_conv = Merge(mode='concat')(conv_blocks)
        else:
            out_conv = conv_blocks[0]

        conv_parallel = Model(inputs=inp, outputs=out_conv)

        model = Sequential()
        model.add(conv_parallel)


        # model.add(Conv2D(
        #     NB_FILTER[0], (raw_feature_dim, NB_GRAM[0]),
        #     input_shape=input_size, border_mode='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(1,3)))


        # model.add(Conv2D(
        #     NB_FILTER[0], (1, NB_GRAM[0]),
        #     input_shape=input_size, border_mode='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(1, 3)))

        # model.add(Conv2D(
        #     NB_FILTER[0], (1, NB_GRAM[1]),
        #     input_shape=input_size, border_mode='valid', activation='relu'))
        # model.add(MaxPooling2D(pool_size=(1, 3)))

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



