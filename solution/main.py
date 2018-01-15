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
    for k in range(k_fold-1):
        if k == k_fold-1:
            start = int(k * ratio * data_size)
            tmp = sequence[start:]
        else:
            start = int(k * ratio * data_size)
            end = int((k + 1) * ratio * data_size)
            tmp = sequence[start:end]
        k_sequence.append(tmp)
    return k_sequence


def main():

    if len(sys.argv) == 1:
        n_gram_list = [1]
    else:
        n_gram_list = sys.argv[1:len(sys.argv)]
        n_gram_list = map(int, n_gram_list)

    x = PrepareData.feat_extraction(n_gram_list, x_one_hot)
    n_feat = x.shape[1]
    raw_data_size = (n_feat, text_length, 1)

    # train_seq, test_seq = data_set_split(x.shape[0], 0.2)
    n_classes = y.shape[1]
    k = 5
    k_fold_sequence = data_set_k_fold_separation(x.shape[0],k)

    output_train = open('train_acc.txt', 'wb')
    output_test = open('train_acc.txt', 'wb')

    train_acc = []
    test_acc = []
    for i in range(k):
        test_seq = k_fold_sequence[i]
        train_seq = []
        for j in range(k):
            if i != j:
                train_seq.extend(k_fold_sequence[j])

        nc = NC(input_size=raw_data_size, n_classes=n_classes, raw_feature_dim=n_feat)
        xtrain = x[train_seq]
        ytrain = y[train_seq]
        nc.fit(xtrain, ytrain)

        eval_train_result = nc.evaluation(xtrain, ytrain)
        print(eval_train_result)
        print>>output_train,[k,eval_train_result]
        train_acc.append(eval_train_result)
        eval_test_result = nc.evaluation(x[test_seq], y[test_seq])
        print(eval_test_result)
        print>>output_test, [k, eval_test_result]
        test_acc.append(eval_test_result)




        # model_name_path = 'myNovelCNN.pickle'
        # print("saving model...")
        # nc.save_ncnn_model(model_name_path)

        # ytrain_pred = nc.predict(xtrain)
        # res_train = np.concatenate((ytrain_pred, ytrain), axis=1)
        # np.savetxt('rest_train.txt', res_train, fmt='%1.2f')
        #
        # ytest_pred = nc.predict(x[test_seq])
        # res_test = np.concatenate((ytest_pred, y[test_seq]), axis=1)
        # np.savetxt('rest_test.txt', res_test, fmt='%1.2f')
    print>>output_train,['average', np.mean(np.asarray(train_acc), axis=0)]
    print>>output_test, ['average', np.mean(np.asarray(test_acc), axis=0)]

if __name__ == "__main__":

    char_file_path = 'char.json'
    with open(char_file_path,'r') as char_file:
        char_list = js.load(char_file)

    char_dic = Build_Char_One_Hot_Dic.one_hot_encoding(char_list)
    one_hot_feature_dim = len(char_list)

    train_file_path = str(Path().resolve().parent) + '/Offline-Challenge/xtrain_obfuscated.txt'
    with open(train_file_path,'r') as raw_data_file:
        raw_data = read_raw_data(raw_data_file)

    x_one_hot, text_length = PrepareData.prepare_data(raw_data, char_dic, one_hot_feature_dim)

    label_file_path = str(Path().resolve().parent) + '/Offline-Challenge/ytrain.txt'
    with open(label_file_path, 'r') as label_file:
        label_data = read_label(label_file)
    y = label_data

    main()


    # train_confusion = confusion_matrix(np.nonzero(ytrain)[1], np.nonzero(ytrain_pred)[1])
    # print(train_confusion)

    # test_confusion = confusion_matrix(np.nonzero(ytest_pred)[1],np.nonzero(ytest_pred)[1])
    # print(test_confusion)

    # # plot confusion matrix
    # label_dict_path = "label_dict.json"
    # with open(label_dict_path,'r') as label_dict_file:
    #     label_dict = js.load(label_dict_file)
    # categories = label_dict.keys()
    #
    # figure = plt.figure()
    # plt.clf()
    #
    # train_figure = figure.add_subplot(1,2,1)
    # train_figure.set_yticks(categories)
    # train_figure.imshow(train_confusion, cmap=plt.cm.jet,
    #           interpolation='nearest')
    #
    # train_figure = figure.add_subplot(1,2,2)
    # train_figure.set_yticks(categories)
    # train_figure.imshow(train_confusion, cmap=plt.cm.jet,
    #                     interpolation='nearest')
    # plt.savefig("res.eps", format="eps")















