import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import simplejson as js
import Build_Char_One_Hot_Dic
import PrepareData
import sys
import numpy as np
from Novel_CNN import NovelCnn as NC
from pathlib import Path
from keras.utils import np_utils


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


def data_set_k_fold_separation(data_size, k_fold):
    sequence = range(data_size)
    sequence = np.random.permutation(sequence)
    k_sequence = []
    ratio = np.true_divide(1, k_fold)
    for k in range(k_fold):
        if k == k_fold:
            start = int(k * ratio * data_size)
            tmp = sequence[start:]
        else:
            start = int(k * ratio * data_size)
            end = int((k + 1) * ratio * data_size)
            tmp = sequence[start:end]
        k_sequence.append(tmp)
    return k_sequence

# should be unittest

def main():

    if len(sys.argv) == 1:
        n_gram_list = [1]
    else:
        n_gram_list = sys.argv[1:len(sys.argv)]
        n_gram_list = map(int, n_gram_list)

    x = PrepareData.feat_extraction(n_gram_list, x_one_hot)
    x_validate = PrepareData.feat_extraction(n_gram_list, x_one_hot)
    n_feat = x.shape[1]
    raw_data_size = (n_feat, text_length, 1)

    # train_seq, test_seq = data_set_split(x.shape[0], 0.2)
    n_classes = y.shape[1]
    k = 5
    k_fold_sequence = data_set_k_fold_separation(x.shape[0],k)

    output_train = open('train_acc.txt', 'wb')
    output_test = open('train_acc.txt', 'wb')
    y_validate = []
    for i in range(k):
        test_seq = k_fold_sequence[i]
        train_seq = []
        for j in range(k):
            if i != j:
                train_seq.extend(k_fold_sequence[j])

        nc = NC(input_size=raw_data_size, n_classes=n_classes, raw_feature_dim=n_feat)
        xtrain = x[train_seq]
        ytrain = y[train_seq]
        nc.fit([xtrain,xtrain], ytrain)

        eval_train_result = nc.evaluation(xtrain, ytrain)
        print(eval_train_result)
        print>>output_train,[k,eval_train_result]
        eval_test_result = nc.evaluation(x[test_seq], y[test_seq])
        print(eval_test_result)
        print>>output_test, [k, eval_test_result]

        y_validate_k = nc.predict(x_validate)
        y_validate_k = y_validate_k.argmax(axis=1)

    print>>output_train,{'average', np.mean(eval_train_result, axis=0)}
    print>>output_test, {'average', np.mean(eval_test_result, axis=0)}
    y_validate_file_path = 'ytest.txt'
    np.savetxt(fname=y_validate_file_path, X=np.asarray(y_validate_k), fmt='%1.2f')


if __name__ == "__main__":

    char_file_path = 'char.json'
    with open(char_file_path,'r') as char_file:
        char_list = js.load(char_file)

    char_dic = Build_Char_One_Hot_Dic.one_hot_encoding(char_list)
    one_hot_feature_dim = len(char_list)

    train_file_path = str(Path().resolve().parent) + '/Offline-Challenge/test/xtrain_obfuscated.txt'
    with open(train_file_path,'r') as raw_data_file:
        raw_data = read_raw_data(raw_data_file)

    validate_file_path = str(Path().resolve().parent) + '/Offline-Challenge/xtest_obfuscated.txt'
    with open(validate_file_path, 'r') as validate_raw_file:
        validate_raw = read_raw_data(validate_raw_file)

    x_one_hot, text_length = PrepareData.prepare_data(raw_data, char_dic, one_hot_feature_dim)
    x_validate_one_hot, text_length = PrepareData.prepare_data(validate_raw, char_dic, one_hot_feature_dim, text_length=text_length)

    label_file_path = str(Path().resolve().parent) + '/Offline-Challenge/test/ytrain.txt'
    with open(label_file_path, 'r') as label_file:
        label_data = read_label(label_file)
    y = label_data

    main()













