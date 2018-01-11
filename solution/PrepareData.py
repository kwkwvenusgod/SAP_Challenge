import numpy as np


def prepare_data(raw_data, one_hot_dict, feat_len):
    train_raw = []
    raw_text_length = []
    for data in raw_data:
        raw_text_length.append(len(data))
        train_raw.append([one_hot_dict.get(d) for d in data])

    text_length = max(raw_text_length)

    X = np.zeros(len(raw_data), 1, text_length, feat_len)

    for i in range(len(train_raw)):
        tmp = np.zeros(text_length, feat_len)
        tmp[0:len(train_raw[i])-1] = np.asarray(train_raw)
        X[i,1] = tmp

    return X
