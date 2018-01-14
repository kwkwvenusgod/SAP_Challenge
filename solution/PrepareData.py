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


def n_gram_feature_extraction(n_gram, x):
    res = np.zeros(shape=x.shape)

    for i in range(x.shape[0]):
        article = x[i]
        tmp = np.zeros(shape=article.shape)
        for j in range(x.shape[1]):
            feat = np.sum(article[:, j:j+n_gram],axis=1)
            tmp[:,j] = feat
        res[i] = tmp
    return res


def feat_extraction(n_gram_list, x):
    n_sample = x.shape[0]
    n_feat_basis = x.shape[1]
    n_feat = n_feat_basis * len(n_gram_list)
    n_text = x.shape[2]
    n = x.shape[3]

    res = np.zeros((n_sample,n_feat,n_text,n))

    for i in range(len(n_gram_list)):
        n_gram_feat = n_gram_feature_extraction(n_gram_list[i], x)
        for j in range(n_sample):
            res[j, i * n_feat_basis:(i + 1) * n_feat_basis] = n_gram_feat[j]

    return res





