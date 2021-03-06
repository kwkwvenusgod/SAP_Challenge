import numpy as np


def prepare_data(raw_data, one_hot_dict, feat_len, text_length=None):
    train_raw = []
    raw_text_length = []
    for data in raw_data:
        raw_text_length.append(len(data))
        train_raw.append([one_hot_dict.get(d) for d in data])

    if text_length is None:
        text_length = max(raw_text_length)

    x = np.zeros((len(raw_data), feat_len, text_length, 1))

    for i in range(len(train_raw)):
        tmp = np.zeros([feat_len,text_length])
        if text_length>len(train_raw[i]):
            tmp[:, 0:len(train_raw[i])] = np.asarray(train_raw[i]).transpose()
        else:
            tmp[:, 0:len(train_raw[i])] = np.asarray(train_raw[i][0:text_length]).transpose()
        x[i, :, :, 0] = tmp

    return x, text_length


def n_gram_feature_extraction(n_gram, x):
    n_sample = x.shape[0]
    n_feat_basis = x.shape[1]
    n_feat = n_feat_basis * n_gram
    n_text = x.shape[2]
    n_last = x.shape[3]

    res = np.zeros((n_sample,n_feat,n_text,n_last))

    for i in range(x.shape[0]):
        article = x[i]
        tmp = np.zeros((n_feat, n_text,n_last))
        for j in range(x.shape[2]):
            if j+n_gram <= x.shape[2]:
                feat = article[:, j:j + n_gram].reshape(n_feat, n_last)
                tmp[:, j] = feat
            else:
                rest = x.shape[2] - j
                feat = np.zeros((n_feat,n_last))
                feat [0:rest*n_feat_basis] = article[:, j:j+rest].reshape(rest*n_feat_basis, n_last)
                tmp[:, j] = feat
        res[i] = tmp
    return res


def feat_extraction(n_gram_list, x):
    n_sample = x.shape[0]
    n_feat_basis = x.shape[1]
    n_feat = n_feat_basis * sum(n_gram_list)
    n_text = x.shape[2]
    n = x.shape[3]

    res = np.zeros((n_sample,n_feat,n_text,n))
    
    start_dim = 0
    for i in range(len(n_gram_list)):
        ngram_feat = n_gram_feature_extraction(n_gram_list[i], x)
        n_gramtmp = ngram_feat.shape[1]
        for j in range(n_sample):
            res[j, start_dim:start_dim + n_gramtmp] = ngram_feat[j]
        start_dim = start_dim + n_gramtmp

    return res





