import numpy as np

# build char dictionary


def one_hot_encoding(char_list):
    res = {}
    feat_len = len(char_list)
    for i in range(feat_len):
        tmp_hot_vec = np.zeros(feat_len)
        tmp_hot_vec[i] = 1
        char_tmp = char_list[i]
        res.update({char_tmp:tmp_hot_vec})
    return res

