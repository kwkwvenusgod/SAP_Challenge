import simplejson as js
import Build_Char_One_Hot_Dic
import PrepareData
import numpy as np
from Novel_CNN import NovelCnn as NC
from pathlib import Path
from keras.utils import np_utils


def read_raw_data(_file):
    content = _file.readlines();
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





