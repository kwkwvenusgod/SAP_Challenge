import simplejson as js
import Build_Char_One_Hot_Dic
import PrepareData
import pickle
import numpy as np
import matplotlib.pyplot as plt
from Novel_CNN import NovelCnn as NC
from pathlib import Path
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


def read_raw_data(_file):
    content = _file.readlines()
    content = [x.rstrip('\n') for x in content]
    raw_data_list = [list(x) for x in content]
    return raw_data_list


def read_label(_file):
    labels = _file.readlines()
    labels = [x.rstrip('\n') for x in labels]
    n_classes = len(set(labels))
    youtput = np_utils.to_categorical(labels, n_classes)
    return youtput


def data_set_split(data_size, partition):
    train_size = int((1-partition) * data_size)
    sequence = range(data_size)
    sequence = np.random.permutation(sequence)
    train_sequence = sequence[0:train_size]
    test_sequence = sequence[train_size:-1]
    return train_sequence,test_sequence


if __name__ == "__main__":

    char_file_path = 'char.json'
    with open(char_file_path,'r') as char_file:
        char_list = js.load(char_file)

    char_dic = Build_Char_One_Hot_Dic.one_hot_encoding(char_list)
    raw_feature_dim = len(char_list)

    train_file_path = str(Path().resolve().parent) + '/Offline-Challenge/xtrain_obfuscated.txt'
    with open(train_file_path,'r') as raw_data_file:
        raw_data = read_raw_data(raw_data_file)

    x, text_length = PrepareData.prepare_data(raw_data, char_dic, raw_feature_dim)
    raw_data_size = (raw_feature_dim, text_length, 1)

    label_file_path = str(Path().resolve().parent) + '/Offline-Challenge/ytrain.txt'
    with open(label_file_path, 'r') as label_file:
        label_data = read_label(label_file)
    y = label_data

    train_seq, test_seq = data_set_split(x.shape[0], 0.1)
    n_classes = y.shape[1]
    nc = NC(input_size=raw_data_size,n_classes=n_classes,raw_feature_dim=raw_feature_dim)
    nc.fit(x[train_seq], y[train_seq])
    eval_result = nc.evaluation(x[test_seq],y[test_seq])
    print(eval_result)

    model_name_path = 'myNovelCNN.pickle'
    nc.save_ncnn_model(model_name_path)

    ytrain_pred = nc.predict(x[train_seq])
    train_confusion = confusion_matrix(y[train_seq], ytrain_pred)
    print(train_confusion)

    ytest_pred = nc.predict(x[test_seq])
    test_confusion = confusion_matrix(y[test_seq],ytest_pred)
    print(test_confusion)

    # plot confusion matrix
    label_dict_path = "label_dict.json"
    with open(label_dict_path,'r') as label_dict_file:
        label_dict = js.load(label_dict_file)
    categories = label_dict.keys()

    figure = plt.figure()
    plt.clf()

    train_figure = figure.add_subplot(1,2,1)
    train_figure.set_yticks(categories)
    train_figure.imshow(train_confusion, cmap=plt.cm.jet,
              interpolation='nearest')

    train_figure = figure.add_subplot(1,2,2)
    train_figure.set_yticks(categories)
    train_figure.imshow(train_confusion, cmap=plt.cm.jet,
                        interpolation='nearest')
    plt.savefig("res.eps", format="eps")















