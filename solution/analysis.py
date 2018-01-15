import numpy as np


rest_train_file_path = 'rest_train.txt'
rest_train = np.loadtxt(rest_train_file_path)
ytrain_pred = rest_train[:,0:12]
ytrain_true = rest_train[:,12:-1]

ytrain_pred = ytrain_pred.argmax(axis=1)
ytrain_true = ytrain_true.argmax(axis=1)





