import numpy as np


def prepare_data(raw_data, one_hot_dict, feat_len):
    train_raw = []
    raw_text_length = []
    for data in raw_data:
        raw_text_length.append(len(data))
        train_raw.append([one_hot_dict.get(d) for d in data])

    text_length = max(raw_text_length)

    x = np.zeros((len(raw_data), feat_len, text_length, 1))

    for i in range(len(train_raw)):
        tmp = np.zeros([feat_len,text_length])
        tmp[:, 0:len(train_raw[i])] = np.asarray(train_raw[i]).transpose()
        x[i, :, :, 0] = tmp

    return x, text_length
